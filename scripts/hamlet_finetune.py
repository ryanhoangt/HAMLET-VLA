"""HAMLET finetuning script.

Loads a pretrained GR00T-N1.5, attaches moment tokens and the memory module,
then finetunes with history-aware batches.  Backbone LLM + visual encoder stay
frozen; moment_tokens, MemoryModule, and action head are trained.

Training loop per step:
  - T-1 historical backbone passes under torch.no_grad() → moment_features
  - 1 current-frame backbone pass with full gradients → moment_features + loss
  - Stack into moment_history (B, T, n_m, d), run MemoryModule, compute action loss

Usage:
    python scripts/hamlet_finetune.py \\
        --dataset_path /path/to/dataset \\
        --data_config so100 \\
        --base_model_path nvidia/GR00T-N1.5-3B \\
        --tcl_checkpoint /path/to/tcl_latest.pt \\
        --output_dir /tmp/hamlet_run
"""

import math
import os
import subprocess
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import List, Literal, Optional

import torch
import torch.nn as nn
import tyro

from gr00t.data.dataset import LeRobotHistoryDataset, LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import load_data_config
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.model.transforms import (
    DEFAULT_EAGLE_PATH,
    EMBODIMENT_TAG_MAPPING,
    build_eagle_processor,
    collate,
)


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------

def hamlet_collate_fn(batch: list[dict], eagle_processor) -> dict:
    """Collate {"frames": [dict_t0, ..., dict_{T-1}]} across a batch.

    Returns {"frames": [collated_t0, ..., collated_{T-1}]} where each
    collated_t is a standard GR00T batch dict with eagle_* tensors.
    """
    T = len(batch[0]["frames"])
    frames_collated = []
    for t in range(T):
        frame_samples = [item["frames"][t] for item in batch]
        frames_collated.append(collate(frame_samples, eagle_processor))
    return {"frames": frames_collated}


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: GR00T_N1_5,
    optimizer: torch.optim.Optimizer,
    scheduler,
    output_dir: str,
    step: int,
) -> None:
    assert model.hamlet_memory is not None, "attach_hamlet() must be called before saving"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ckpt = {
        "step": step,
        # moment_tokens — compatible with load_moment_tokens() from TCL checkpoints
        "moment_tokens": model.backbone.moment_tokens.data.cpu(),
        "hamlet_memory": model.hamlet_memory.state_dict(),
        "action_head": model.action_head.state_dict(),
        "backbone_eagle_linear": model.backbone.eagle_linear.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    path = os.path.join(output_dir, f"hamlet_step{step:06d}.pt")
    torch.save(ckpt, path)
    torch.save(ckpt, os.path.join(output_dir, "hamlet_latest.pt"))
    print(f"  Saved checkpoint → {path}")


def load_checkpoint(
    model: GR00T_N1_5,
    optimizer: torch.optim.Optimizer,
    scheduler,
    ckpt_path: str,
) -> int:
    assert model.hamlet_memory is not None, "attach_hamlet() must be called before loading"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    dtype = model.backbone.moment_tokens.dtype
    model.backbone.moment_tokens.data.copy_(ckpt["moment_tokens"].to(dtype=dtype))
    model.hamlet_memory.load_state_dict(ckpt["hamlet_memory"])
    model.action_head.load_state_dict(ckpt["action_head"])
    if "backbone_eagle_linear" in ckpt:
        model.backbone.eagle_linear.load_state_dict(ckpt["backbone_eagle_linear"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    start_step = ckpt.get("step", 0)
    print(f"Resumed from step {start_step}  ({ckpt_path})")
    return start_step


# ---------------------------------------------------------------------------
# LR schedule: linear warmup → cosine decay
# ---------------------------------------------------------------------------

def _lr_lambda(step: int, warmup_steps: int, max_steps: int) -> float:
    if step < warmup_steps:
        return float(step) / max(float(warmup_steps), 1.0)
    progress = float(step - warmup_steps) / max(float(max_steps - warmup_steps), 1.0)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class HAMLETArgs:
    # --- Data ---
    dataset_path: List[str]
    """One or more LeRobot-format dataset directories.  Multiple paths are concatenated."""

    data_config: str = "so100"
    """Data config name (e.g. 'so100', 'fourier_gr1_arms_only').  Must match the dataset."""

    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "new_embodiment"
    """Embodiment tag to use for training."""

    video_backend: Literal["torchcodec", "decord", "torchvision_av"] = "torchcodec"
    """Video backend for data loading."""

    # --- History ---
    history_len: int = 4
    """T — frames per sample including the current one (paper default: 4)."""

    history_stride: int = 1
    """Frame gap between history entries.  At inference use the action chunk size k."""

    # --- Model ---
    base_model_path: str = "nvidia/GR00T-N1.5-3B"
    """HuggingFace hub ID or local path for the pretrained GR00T checkpoint."""

    tcl_checkpoint: Optional[str] = None
    """Path to a TCL moment_tokens checkpoint (tcl_latest.pt).
    If None, moment_tokens are randomly initialised."""

    resume_from: Optional[str] = None
    """Path to a hamlet_*.pt checkpoint to resume HAMLET training."""

    # --- HAMLET architecture ---
    num_moment_tokens: int = 4
    """n_m — moment tokens injected into the VLM sequence (paper default: 4)."""

    n_heads: int = 8
    """Attention heads in the memory transformer."""

    n_layers: int = 2
    """Memory transformer layers (paper default: 2)."""

    max_history: int = 4
    """Maximum history length T for the memory module (paper default: 4)."""

    memory_dropout: float = 0.1
    """Dropout in the memory transformer."""

    # --- Backbone finetuning flags (same semantics as gr00t_finetune.py) ---
    tune_llm: bool = False
    tune_visual: bool = False
    tune_projector: bool = True
    tune_diffusion_model: bool = True

    # --- Training ---
    output_dir: str = "/tmp/hamlet"
    """Directory to write checkpoints."""

    batch_size: int = 32
    """Per-GPU batch size.  Each sample triggers T backbone passes (paper: 32)."""

    max_steps: int = 60000
    num_gpus: int = 1
    save_steps: int = 5000
    log_every: int = 10

    learning_rate: float = 1e-5
    """LR for MemoryModule and action head (paper: 1e-5)."""

    weight_decay: float = 1e-5
    warmup_steps: int = 3000
    """Linear warmup steps (paper follows gr00t_finetune default warmup_ratio=0.05; 0.05 * 60k = 3k)."""
    gradient_accumulation_steps: int = 1
    num_workers: int = 8
    prefetch_factor: int = 2

    # --- Logging ---
    wandb_project: Optional[str] = None
    """W&B project name.  None = disabled."""

    wandb_run_name: Optional[str] = None
    """W&B run name.  Auto-generated if None."""

    wandb_run_id: Optional[str] = None
    """W&B run ID to resume (e.g. '36u2z7y3' from the run URL).
    Use together with --resume_from to continue a specific W&B run."""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: HAMLETArgs) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # 1. Dataset
    # ------------------------------------------------------------------
    data_cfg = load_data_config(args.data_config)
    modality_configs = data_cfg.modality_config()
    transforms = data_cfg.transform()
    embodiment_tag = EmbodimentTag(args.embodiment_tag)

    history_datasets = []
    for path in args.dataset_path:
        base = LeRobotSingleDataset(
            dataset_path=path,
            modality_configs=modality_configs,
            transforms=transforms,
            embodiment_tag=embodiment_tag,
            video_backend=args.video_backend,
        )
        history_datasets.append(
            LeRobotHistoryDataset(
                base_dataset=base,
                history_len=args.history_len,
                stride=args.history_stride,
            )
        )

    history_dataset = (
        history_datasets[0]
        if len(history_datasets) == 1
        else torch.utils.data.ConcatDataset(history_datasets)
    )
    print(
        f"Dataset: {len(history_dataset)} samples across {len(args.dataset_path)} path(s)  "
        f"(history_len={args.history_len}, stride={args.history_stride})"
    )

    eagle_processor = build_eagle_processor(DEFAULT_EAGLE_PATH)
    loader = torch.utils.data.DataLoader(
        history_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=partial(hamlet_collate_fn, eagle_processor=eagle_processor),
        drop_last=True,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=False,
    )

    # ------------------------------------------------------------------
    # 2. Model
    # ------------------------------------------------------------------
    print(f"Loading model from {args.base_model_path} ...")
    model = GR00T_N1_5.from_pretrained(
        args.base_model_path,
        tune_llm=args.tune_llm,
        tune_visual=args.tune_visual,
        tune_projector=args.tune_projector,
        tune_diffusion_model=args.tune_diffusion_model,
    )
    model.compute_dtype = "bfloat16"
    model.config.compute_dtype = "bfloat16"

    # Validate action head dimensions match data config
    data_action_horizon = len(data_cfg.action_indices)
    from gr00t.model.transforms import GR00TTransform
    last_transform = transforms.transforms[-1]
    assert isinstance(last_transform, GR00TTransform), "Last transform must be GR00TTransform"
    data_max_action_dim = last_transform.max_action_dim

    if data_action_horizon != model.action_head.config.action_horizon or \
       data_max_action_dim != model.action_head.config.action_dim:
        raise ValueError(
            f"Action head mismatch: model expects "
            f"horizon={model.action_head.config.action_horizon}, "
            f"dim={model.action_head.config.action_dim} but data has "
            f"horizon={data_action_horizon}, dim={data_max_action_dim}. "
            "Run gr00t_finetune.py first to adapt the action head, then resume "
            "from that checkpoint with --base_model_path pointing to the output dir."
        )

    # Attach HAMLET components
    model.attach_hamlet(
        num_moment_tokens=args.num_moment_tokens,
        # d_model auto-detected from backbone eagle_linear output dimension
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_history=args.max_history,
        dropout=args.memory_dropout,
    )

    # Load TCL-pretrained moment tokens if available
    if args.tcl_checkpoint:
        model.load_moment_tokens(args.tcl_checkpoint)

    model = model.to(device)
    model.train()
    assert model.hamlet_memory is not None  # set by attach_hamlet() above

    # ------------------------------------------------------------------
    # 3. Optimizer  (two param groups for separate LRs)
    # ------------------------------------------------------------------
    # Paper: "VLM and moment tokens are kept frozen" during fine-tuning.
    # Only MemoryModule + action head are trained.
    model.backbone.moment_tokens.requires_grad_(False)
    for p in model.backbone.eagle_linear.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        list(model.hamlet_memory.parameters()) + list(model.action_head.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.95, 0.999),
        eps=1e-8,
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=partial(
            _lr_lambda,
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps,
        ),
    )

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Trainable parameters: {n_trainable:,}  "
        f"(memory={sum(p.numel() for p in model.hamlet_memory.parameters()):,}  "
        f"action_head={sum(p.numel() for p in model.action_head.parameters()):,})"
    )

    # Collect all trainable params for grad clipping
    trainable_params = list(model.hamlet_memory.parameters()) + list(model.action_head.parameters())

    # ------------------------------------------------------------------
    # 4. Resume if requested
    # ------------------------------------------------------------------
    start_step = 0
    if args.resume_from:
        start_step = load_checkpoint(model, optimizer, scheduler, args.resume_from)

    # ------------------------------------------------------------------
    # 5. W&B
    # ------------------------------------------------------------------
    use_wandb = args.wandb_project is not None
    if use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            id=args.wandb_run_id,
            config=vars(args),
            resume="must" if args.wandb_run_id else "allow",
        )
        print(f"W&B logging enabled → project: {args.wandb_project}"
              + (f"  run_id: {args.wandb_run_id}" if args.wandb_run_id else ""))

    # ------------------------------------------------------------------
    # 6. Training loop
    # ------------------------------------------------------------------
    def _cycle(dataloader):
        """Infinite iterator over a DataLoader — avoids re-creating iter() on
        persistent-worker loaders, which can deadlock in some PyTorch versions."""
        while True:
            for batch in dataloader:
                yield batch

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    data_iter = _cycle(loader)
    running_loss = 0.0
    accum_step = 0

    for step in range(start_step + 1, args.max_steps + 1):
        batch = next(data_iter)

        frames = batch["frames"]  # list of T collated dicts
        T = len(frames)

        # torch.autocast ensures consistent bfloat16 computation across backbone,
        # MemoryModule, and action head — same as HuggingFace Trainer's bf16=True.
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            # --- Historical backbone passes: no gradient needed ---
            moment_features_list = []
            with torch.no_grad():
                for t in range(T - 1):
                    bb_in, _ = model.prepare_input(frames[t])
                    bb_out = model.backbone(bb_in)
                    moment_features_list.append(bb_out["moment_features"])

            # --- Current frame: full gradient graph ---
            bb_in_curr, act_in_curr = model.prepare_input(frames[-1])
            bb_out_curr = model.backbone(bb_in_curr)
            moment_features_list.append(bb_out_curr["moment_features"])

            # moment_history: (B, T, n_m, d_model)
            moment_history = torch.stack(moment_features_list, dim=1)

            # MemoryModule: produces m̃'_t (B, n_m, d) → concat to backbone_features
            bb_out_curr = model._apply_hamlet_memory(bb_out_curr, moment_history)

            action_out = model.action_head(bb_out_curr, act_in_curr)
            loss = action_out["loss"] / args.gradient_accumulation_steps

        # backward outside autocast — safe for bfloat16 (no GradScaler needed)
        loss.backward()

        running_loss += loss.item() * args.gradient_accumulation_steps
        accum_step += 1

        if accum_step == args.gradient_accumulation_steps:
            nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            accum_step = 0

        # --- Logging ---
        if step % args.log_every == 0:
            avg_loss = running_loss / args.log_every
            lrs = scheduler.get_last_lr()
            lr_hamlet = lrs[0]
            lr_action = lrs[1] if len(lrs) > 1 else lr_hamlet
            mt_grad = model.backbone.moment_tokens.grad
            mt_grad_norm = mt_grad.norm().item() if mt_grad is not None else 0.0
            print(
                f"[step {step:>6}/{args.max_steps}]  loss={avg_loss:.4f}  "
                f"lr={lr_hamlet:.2e}  mt_grad_norm={mt_grad_norm:.4f}"
            )
            if use_wandb:
                import wandb
                wandb.log(
                    {
                        "train/loss": avg_loss,
                        "train/lr_hamlet": lr_hamlet,
                        "train/lr_action": lr_action,
                        "train/moment_token_grad_norm": mt_grad_norm,
                        "train/moment_token_norm": model.backbone.moment_tokens.data.norm().item(),
                    },
                    step=step,
                )
            running_loss = 0.0

        # --- Checkpointing ---
        if step % args.save_steps == 0:
            save_checkpoint(model, optimizer, scheduler, args.output_dir, step)

    # Final save
    save_checkpoint(model, optimizer, scheduler, args.output_dir, args.max_steps)
    if use_wandb:
        import wandb
        wandb.finish()
    print("HAMLET finetuning complete.")
    print(f"Latest checkpoint: {args.output_dir}/hamlet_latest.pt")


if __name__ == "__main__":
    config = tyro.cli(HAMLETArgs)

    print("\n" + "=" * 50)
    print("HAMLET FINETUNING CONFIGURATION:")
    print("=" * 50)
    for key, value in vars(config).items():
        print(f"{key}: {value}")
    print("=" * 50 + "\n")

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    assert config.num_gpus <= available_gpus, (
        f"Requested {config.num_gpus} GPUs but only {available_gpus} available"
    )
    assert config.num_gpus > 0

    if config.num_gpus == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        main(config)
    else:
        if os.environ.get("IS_TORCHRUN", "0") == "1":
            main(config)
        else:
            script_path = Path(__file__).absolute()
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]
            cmd = [
                "torchrun",
                "--standalone",
                f"--nproc_per_node={config.num_gpus}",
                "--nnodes=1",
                str(script_path),
                *sys.argv[1:],
            ]
            env = os.environ.copy()
            env["IS_TORCHRUN"] = "1"
            sys.exit(subprocess.run(cmd, env=env).returncode)
