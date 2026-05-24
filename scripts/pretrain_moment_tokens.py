"""TCL pretraining for HAMLET moment tokens.

Trains only moment_tokens (+ a throwaway projection head) with a time-contrastive
loss while the entire GR00T backbone is frozen.  After convergence, saves the
moment_tokens weights for use in subsequent full HAMLET training.

Usage:
    python scripts/pretrain_moment_tokens.py \
        --dataset_path /path/to/dataset \
        --data_config so100 \
        --base_model_path nvidia/GR00T-N1.5-3B \
        --output_dir /tmp/tcl_hamlet
"""

import os
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import tyro

from gr00t.data.dataset import LeRobotSingleDataset, ModalityConfig, TCLTripletDataset
from gr00t.experiment.data_config import load_data_config
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.model.hamlet.tcl import MomentProjectionHead, tcl_loss
from gr00t.model.transforms import DEFAULT_EAGLE_PATH, build_eagle_processor, collate


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------

def tcl_collate_fn(batch: list[dict], eagle_processor) -> dict:
    """Collate a list of {anchor, positive, negative} dicts into batched tensors.

    Applies the eagle processor to each view group independently so that the
    per-sample eagle_content dicts are merged into padded input_ids / pixel_values.
    """
    result = {}
    for view in ("anchor", "positive", "negative"):
        view_samples = [item[view] for item in batch]
        result[view] = collate(view_samples, eagle_processor)
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def move_to_device(d: dict, device: torch.device) -> dict:
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in d.items()
    }


def save_checkpoint(
    model: GR00T_N1_5,
    proj_head: MomentProjectionHead,
    output_dir: str,
    step: int,
) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ckpt = {
        "step": step,
        # Only the moment_tokens are needed downstream; save proj_head too for resuming
        "moment_tokens": model.backbone.moment_tokens.data.cpu(),
        "proj_head": proj_head.state_dict(),
    }
    path = os.path.join(output_dir, f"tcl_step{step:06d}.pt")
    torch.save(ckpt, path)
    # Also overwrite a "latest" pointer for easy resuming
    torch.save(ckpt, os.path.join(output_dir, "tcl_latest.pt"))
    print(f"  Saved checkpoint → {path}")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TCLArgs:
    # --- Data ---
    dataset_path: str
    """Path to the LeRobot-format dataset directory."""

    data_config: str = "so100"
    """Data config name (e.g. 'so100', 'fourier_gr1_arms_only').  Must match the dataset."""

    embodiment_tag: str = ""
    """Embodiment tag string. Inferred from dataset metadata if left empty."""

    min_temporal_distance: int = 5
    """Minimum frame gap between anchor and hard-negative within the same trajectory."""

    # --- Model ---
    base_model_path: str = "nvidia/GR00T-N1.5-3B"
    """HuggingFace hub ID or local path for the pretrained GR00T checkpoint."""

    num_moment_tokens: int = 4
    """Number of learnable moment tokens n_m to inject into the VLM sequence."""

    proj_dim: int = 128
    """Dimension of the contrastive projection head output."""

    # --- Training ---
    output_dir: str = "/tmp/tcl_hamlet"
    """Directory to write checkpoints."""

    batch_size: int = 64
    """Per-GPU batch size.  Each item produces 3 backbone passes (anchor/pos/neg) (paper: 64)."""

    num_steps: int = 30000
    """Total gradient update steps (paper: up to 30k)."""

    lr: float = 1e-5
    """Learning rate for moment_tokens + projection head (paper: 1e-5)."""

    weight_decay: float = 1e-4

    temperature: float = 0.07
    """InfoNCE temperature τ."""

    log_every: int = 50
    save_every: int = 1000
    num_workers: int = 4

    # --- Logging ---
    wandb_project: Optional[str] = None
    """W&B project name. Set to enable W&B logging (e.g. 'hamlet-tcl'). None = disabled."""

    wandb_run_name: Optional[str] = None
    """W&B run name. Auto-generated if None."""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: TCLArgs) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    # ------------------------------------------------------------------
    # 1. Dataset
    # ------------------------------------------------------------------
    data_cfg = load_data_config(args.data_config)
    modality_config: dict[str, ModalityConfig] = data_cfg.modality_config()
    transform = data_cfg.transform()

    base_dataset = LeRobotSingleDataset(
        dataset_path=args.dataset_path,
        modality_configs=modality_config,
        embodiment_tag=args.embodiment_tag if args.embodiment_tag else data_cfg.embodiment_tag
            if hasattr(data_cfg, "embodiment_tag") else "new_embodiment",
        transforms=transform,
    )
    # Normalization stats are loaded from dataset metadata automatically above.

    tcl_dataset = TCLTripletDataset(
        base_dataset=base_dataset,
        min_temporal_distance=args.min_temporal_distance,
    )
    print(f"Dataset: {len(tcl_dataset)} triplet samples")

    eagle_processor = build_eagle_processor(DEFAULT_EAGLE_PATH)
    loader = torch.utils.data.DataLoader(
        tcl_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=partial(tcl_collate_fn, eagle_processor=eagle_processor),
        drop_last=True,
        pin_memory=True,
    )

    # ------------------------------------------------------------------
    # 2. Model — load pretrained, then attach moment_tokens
    # ------------------------------------------------------------------
    print(f"Loading model from {args.base_model_path} ...")
    model = GR00T_N1_5.from_pretrained(
        args.base_model_path,
        tune_llm=False,
        tune_visual=False,
        tune_projector=False,
        tune_diffusion_model=False,
        torch_dtype=dtype,
    )

    # Attach moment tokens directly to the backbone (avoids config schema changes for now)
    model.backbone.num_moment_tokens = args.num_moment_tokens
    model.backbone.moment_tokens = nn.Parameter(
        (0.02 * torch.randn(args.num_moment_tokens, 2048)).to(dtype=dtype)
    )

    # Freeze everything, then unfreeze only moment_tokens
    for p in model.parameters():
        p.requires_grad = False
    model.backbone.moment_tokens.requires_grad = True

    model = model.to(device)
    model.train()
    # Keep frozen modules in eval mode (disables dropout inside frozen LLM)
    model.backbone.set_frozen_modules_to_eval_mode()

    # ------------------------------------------------------------------
    # 3. Projection head (discarded after TCL; not used in HAMLET proper)
    # ------------------------------------------------------------------
    # eagle_linear is Linear(2048, project_to_dim) or Identity; fall back to 2048
    proj_in_dim = getattr(model.backbone.eagle_linear, "out_features", 2048)
    proj_head = MomentProjectionHead(
        d_model=proj_in_dim,
        proj_dim=args.proj_dim,
    ).to(device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # 4. Optimizer  (only moment_tokens + proj_head params)
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        [model.backbone.moment_tokens] + list(proj_head.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_trainable += sum(p.numel() for p in proj_head.parameters())
    mt_numel = model.backbone.moment_tokens.numel()
    print(f"Trainable parameters: {n_trainable:,}  "
          f"(moment_tokens={mt_numel:,}  proj_head={sum(p.numel() for p in proj_head.parameters()):,})")

    # ------------------------------------------------------------------
    # W&B
    # ------------------------------------------------------------------
    use_wandb = args.wandb_project is not None
    if use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )
        print(f"W&B logging enabled → project: {args.wandb_project}")

    # ------------------------------------------------------------------
    # 5. Training loop
    # ------------------------------------------------------------------
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    loader_iter = iter(loader)
    running_loss = 0.0
    running_sim_pos = 0.0
    running_sim_neg = 0.0

    for step in range(1, args.num_steps + 1):
        # Replenish iterator when exhausted
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        anchor_in  = move_to_device(batch["anchor"],   device)
        pos_in     = move_to_device(batch["positive"], device)
        neg_in     = move_to_device(batch["negative"], device)

        # --- Forward ---
        # IMPORTANT: do NOT wrap in torch.no_grad() / inference_mode().
        # Gradients must flow backward through frozen LLM layers to reach moment_tokens.
        # requires_grad=False on LLM params means their weights won't be updated, but
        # the computation graph is still needed for moment_tokens.grad.

        _, m_a, _ = model.backbone.forward_eagle_with_moments(
            model.backbone.prepare_input(anchor_in)
        )
        _, m_p, _ = model.backbone.forward_eagle_with_moments(
            model.backbone.prepare_input(pos_in)
        )
        _, m_n, _ = model.backbone.forward_eagle_with_moments(
            model.backbone.prepare_input(neg_in)
        )

        z_a = proj_head(m_a)
        z_p = proj_head(m_p)
        z_n = proj_head(m_n)

        loss = tcl_loss(z_a, z_p, z_n, temperature=args.temperature)

        # --- Backward ---
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        with torch.no_grad():
            running_sim_pos += (z_a * z_p).sum(dim=-1).mean().item()
            running_sim_neg += (z_a * z_n).sum(dim=-1).mean().item()

        if step % args.log_every == 0:
            avg_loss = running_loss / args.log_every
            avg_sim_pos = running_sim_pos / args.log_every
            avg_sim_neg = running_sim_neg / args.log_every
            mt_grad_norm = model.backbone.moment_tokens.grad.norm().item() \
                if model.backbone.moment_tokens.grad is not None else 0.0
            print(
                f"[step {step:>6}/{args.num_steps}]  loss={avg_loss:.4f}  "
                f"sim_pos={avg_sim_pos:.3f}  sim_neg={avg_sim_neg:.3f}  "
                f"mt_grad_norm={mt_grad_norm:.4f}"
            )
            if use_wandb:
                import wandb
                wandb.log({
                    "loss": avg_loss,
                    "sim_pos": avg_sim_pos,
                    "sim_neg": avg_sim_neg,
                    "sim_margin": avg_sim_pos - avg_sim_neg,
                    "moment_token_grad_norm": mt_grad_norm,
                    "moment_token_norm": model.backbone.moment_tokens.data.norm().item(),
                }, step=step)
            running_loss = 0.0
            running_sim_pos = 0.0
            running_sim_neg = 0.0

        if step % args.save_every == 0:
            save_checkpoint(model, proj_head, args.output_dir, step)

    # Final save
    save_checkpoint(model, proj_head, args.output_dir, args.num_steps)
    if use_wandb:
        import wandb
        wandb.finish()
    print("TCL pretraining complete.")
    print(f"Load moment_tokens for HAMLET training with:")
    print(f"  ckpt = torch.load('{args.output_dir}/tcl_latest.pt')")
    print(f"  model.backbone.moment_tokens.data.copy_(ckpt['moment_tokens'].to(device))")


if __name__ == "__main__":
    tyro.cli(main)
