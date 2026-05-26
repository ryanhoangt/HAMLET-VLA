"""LIBERO evaluation for HAMLET checkpoints.

Uses HAMLETPolicy (local inference, no ZMQ server) with a rolling
moment_history buffer that resets between episodes.

Usage:
    cd /path/to/HAMLET-VLA
    PYTHONPATH=/path/to/LIBERO:$PYTHONPATH \
    python examples/Libero/eval/run_hamlet_libero_eval.py \
        --base_model_path expdata/libero_joint_finetuning_60k_baseline/checkpoint-60000 \
        --hamlet_checkpoint expdata/hamlet_libero_spatial/hamlet_latest.pt \
        --data_config "examples.Libero.custom_data_config:LiberoDataConfig" \
        --task_suite_name libero_spatial \
        --headless --num_trials_per_task 20
"""

import os
import pprint
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import tqdm
import tyro
from libero.libero import benchmark

from examples.Libero.eval.utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    normalize_gripper_action,
    quat2axisangle,
    save_rollout_video,
)
from gr00t.experiment.data_config import load_data_config
from gr00t.model.policy import HAMLETPolicy

log_dir = "/tmp/logs"
os.makedirs(log_dir, exist_ok=True)


@dataclass
class GenerateConfig:
    # fmt: off
    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 5
    headless: bool = False

    # Model paths
    base_model_path: str = "expdata/libero_joint_finetuning_60k_baseline/checkpoint-60000"
    """GR00T checkpoint used as base (also provides experiment_cfg/metadata.json)."""

    hamlet_checkpoint: str = "expdata/hamlet_libero_spatial/hamlet_latest.pt"
    """HAMLET .pt checkpoint from hamlet_finetune.py."""

    data_config: str = "examples.Libero.custom_data_config:LiberoDataConfig"
    embodiment_tag: str = "new_embodiment"
    denoising_steps: Optional[int] = 8

    # HAMLET architecture — must match the training config
    num_moment_tokens: int = 4
    n_heads: int = 8
    n_layers: int = 2
    max_history: int = 4
    # fmt: on


class HAMLETLiberoPolicy:
    """Adapts HAMLETPolicy for LIBERO environments."""

    ACTION_KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]

    def __init__(self, cfg: GenerateConfig):
        data_cfg = load_data_config(cfg.data_config)
        modality_config = data_cfg.modality_config()
        transform = data_cfg.transform()

        self._policy = HAMLETPolicy(
            base_model_path=cfg.base_model_path,
            hamlet_checkpoint_path=cfg.hamlet_checkpoint,
            embodiment_tag=cfg.embodiment_tag,
            modality_config=modality_config,
            modality_transform=transform,
            denoising_steps=cfg.denoising_steps,
            num_moment_tokens=cfg.num_moment_tokens,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            max_history=cfg.max_history,
        )
        self.headless = cfg.headless

    def reset(self) -> None:
        """Clear moment history buffer. Call at the start of each episode."""
        self._policy.reset()

    def get_action(self, obs, lang: str) -> np.ndarray:
        obs_dict = self._process_observation(obs, lang)
        action_chunk = self._policy.get_action(obs_dict)
        return self._convert_to_libero_action(action_chunk, idx=0)

    def _process_observation(self, obs, lang: str) -> dict:
        xyz = obs["robot0_eef_pos"]
        rpy = quat2axisangle(obs["robot0_eef_quat"])
        gripper = obs["robot0_gripper_qpos"]
        img, wrist_img = get_libero_image(obs)
        return {
            "video.image": np.expand_dims(img, axis=0),
            "video.wrist_image": np.expand_dims(wrist_img, axis=0),
            "state.x": np.array([[xyz[0]]]),
            "state.y": np.array([[xyz[1]]]),
            "state.z": np.array([[xyz[2]]]),
            "state.roll": np.array([[rpy[0]]]),
            "state.pitch": np.array([[rpy[1]]]),
            "state.yaw": np.array([[rpy[2]]]),
            "state.gripper": np.expand_dims(gripper, axis=0),
            "annotation.human.action.task_description": [lang],
        }

    def _convert_to_libero_action(self, action_chunk: dict, idx: int = 0) -> np.ndarray:
        action_components = [
            np.atleast_1d(action_chunk[f"action.{key}"][idx])[0] for key in self.ACTION_KEYS
        ]
        action_array = np.array(action_components, dtype=np.float32)
        action_array = normalize_gripper_action(action_array, binarize=True)
        assert len(action_array) == 7, f"Expected 7-dim action, got {len(action_array)}"
        return action_array


def eval_libero(cfg: GenerateConfig) -> None:
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")

    log_file = open(f"{log_dir}/hamlet_libero_eval_{cfg.task_suite_name}.log", "w")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")
    log_file.write(f"HAMLET checkpoint: {cfg.hamlet_checkpoint}\n")

    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, resolution=256)

        # Build a fresh policy per task (reuses the same model, just resets buffer)
        policy = HAMLETLiberoPolicy(cfg)

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            # Reset env and moment history buffer
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])
            policy.reset()

            t = 0
            top_view = []
            wrist_view = []
            if cfg.task_suite_name == "libero_spatial":
                max_steps = 220
            elif cfg.task_suite_name == "libero_object":
                max_steps = 280
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 600
            elif cfg.task_suite_name == "libero_10":
                max_steps = 1000
            elif cfg.task_suite_name == "libero_90":
                max_steps = 400
            else:
                max_steps = 500

            print(f"Starting episode {task_episodes + 1}...")
            log_file.write(f"Starting episode {task_episodes + 1}...\n")
            done = False
            while t < max_steps + cfg.num_steps_wait:
                try:
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action())
                        t += 1
                        continue

                    img, wrist_img = get_libero_image(obs)
                    top_view.append(img)
                    wrist_view.append(wrist_img)

                    action = policy.get_action(obs, task.language)
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    break

            task_episodes += 1
            total_episodes += 1

            save_rollout_video(
                top_view,
                wrist_view,
                total_episodes,
                success=done,
                task_description=task_description,
                log_file=log_file,
            )

            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(
                f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n"
            )
            log_file.flush()

        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(
            f"Current task success rate: {float(task_successes) / float(task_episodes)}\n"
        )
        log_file.write(
            f"Current total success rate: {float(total_successes) / float(total_episodes)}\n"
        )
        log_file.flush()

    log_file.close()


if __name__ == "__main__":
    cfg = tyro.cli(GenerateConfig)
    eval_libero(cfg)
