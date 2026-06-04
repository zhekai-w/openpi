"""
OpenPI policy eval on a local LeRobot dataset.

Steps through each episode, calls policy.infer() every action_horizon frames,
computes MSE against ground-truth action chunks, and reports timing.

Usage:
    conda run -n openpi python pkgs/openpi/scripts/eval_policy.py \
        --config pi0_ur5_lora \
        --checkpoint-dir /path/to/checkpoint \
        --dataset-path /path/to/local_dataset \
        --trajs 5 \
        --plot

Column-name overrides (for datasets that don't match the default UR5 key names):
    --cam1-key observation.images.cam1   # base camera  -> base_rgb
    --cam2-key observation.images.cam2   # wrist camera -> wrist_rgb
    --state-key observation.state        # joint+gripper state
    --action-key action                  # ground-truth actions
"""

import time
from dataclasses import dataclass

import numpy as np
import tyro
import torch
import lerobot.common.datasets.lerobot_dataset as lr_dataset

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


@dataclass
class ArgsConfig:
    """OpenPI policy evaluation on a LeRobot dataset."""

    config: str = "pi0_ur5_lora"
    """OpenPI training config name (e.g. pi0_ur5, pi0_ur5_lora, pi0_fast_ur5_lora)."""

    checkpoint_dir: str = ""
    """Path to the OpenPI checkpoint directory (local or GCS)."""

    dataset_path: str = "."
    """Path to the local LeRobot dataset directory."""

    trajs: int = 1
    """Number of trajectories (episodes) to evaluate."""

    start_traj: int = 0
    """Episode index (0-based among loaded episodes) to start from."""

    steps: int | None = None
    """Max steps per trajectory. None = full episode length."""

    action_horizon: int | None = None
    """Action chunk size. None = auto-detect from config model."""

    # Column name mapping — override for datasets with different key names
    cam1_key: str = "observation.images.cam1"
    """LeRobot key for base camera → passed as base_rgb to policy."""

    cam2_key: str = "observation.images.cam2"
    """LeRobot key for wrist camera → passed as wrist_rgb to policy."""

    state_key: str = "observation.state"
    """LeRobot key for robot state → concatenated to 7-dim [arm(6), gripper(1)]."""

    action_key: str = "action"
    """LeRobot key for ground-truth actions (used for MSE)."""

    default_prompt: str | None = None
    """Fallback language prompt if dataset has no task annotation."""

    plot: bool = False
    """Plot predicted vs ground-truth actions per trajectory."""

    save_plot_path: str | None = None
    """If set, save the plot to this path instead of showing it."""


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.numpy()
    return np.asarray(x)


def _extract_obs(item: dict, cam1_key: str, cam2_key: str, state_key: str,
                 default_prompt: str | None) -> dict:
    """Build the observation dict expected by policy.infer() (UR5Inputs format)."""
    # Images: LeRobot video decoder returns (T, C, H, W) with delta_timestamps,
    # or (C, H, W) for a single frame. We want the first (current) frame as (H,W,3) uint8.
    def _get_image(key):
        img = _to_numpy(item[key])
        if img.ndim == 4:   # (T, C, H, W) — take current frame
            img = img[0]
        # now (C, H, W) float or (H, W, C) — UR5Inputs._parse_image handles both
        return img

    # State: (7,) for UR5 [shoulder_pan, shoulder_lift, elbow, w1, w2, w3, gripper]
    state = _to_numpy(item[state_key]).astype(np.float32)
    if state.ndim == 2:     # (T, D) with delta_timestamps — take current
        state = state[0]

    prompt = item.get("task") or default_prompt or ""

    return {
        "base_rgb": _get_image(cam1_key),
        "wrist_rgb": _get_image(cam2_key),
        "state": state,
        "prompt": prompt,
    }


def _get_gt_actions(item: dict, action_key: str) -> np.ndarray:
    """Extract ground-truth action chunk: (action_horizon, action_dim)."""
    actions = _to_numpy(item[action_key])
    if actions.ndim == 1:
        actions = actions[np.newaxis, :]  # single step → (1, D)
    return actions.astype(np.float64)


def _plot_trajectory(pred_chunks: list, gt_chunks: list, traj_id: int,
                     save_path: str | None):
    import matplotlib.pyplot as plt

    pred = np.concatenate(pred_chunks, axis=0)  # (T, D)
    gt = np.concatenate(gt_chunks, axis=0)      # (T, D)
    n_dims = pred.shape[1]
    steps = np.arange(len(pred))

    fig, axes = plt.subplots(n_dims, 1, figsize=(12, 2 * n_dims), sharex=True)
    if n_dims == 1:
        axes = [axes]
    labels = [f"joint_{i}" for i in range(6)] + ["gripper"]
    for i, ax in enumerate(axes):
        label = labels[i] if i < len(labels) else f"dim_{i}"
        ax.plot(steps, gt[:, i], label="ground truth", color="tab:blue", linewidth=1)
        ax.plot(steps, pred[:, i], label="predicted", color="tab:orange", linestyle="--", linewidth=1)
        ax.set_ylabel(label, fontsize=8)
        ax.legend(fontsize=7, loc="upper right")
    axes[-1].set_xlabel("step")
    fig.suptitle(f"Trajectory {traj_id}: predicted vs ground truth actions")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def main(args: ArgsConfig):
    assert args.checkpoint_dir, "Provide --checkpoint-dir"

    # Resolve action_horizon from config if not given
    train_config = _config.get_config(args.config)
    action_horizon = args.action_horizon or train_config.model.action_horizon
    print(f"Config: {args.config}, action_horizon: {action_horizon}")

    # Load policy
    print(f"Loading policy from {args.checkpoint_dir} ...")
    policy = _policy_config.create_trained_policy(
        train_config,
        args.checkpoint_dir,
        default_prompt=args.default_prompt,
    )
    print("Policy loaded.")

    # Load LeRobot dataset — use root= to load from local path
    fps_dataset = lr_dataset.LeRobotDatasetMetadata(
        repo_id="local", root=args.dataset_path
    ).fps
    delta_ts = {args.action_key: [t / fps_dataset for t in range(action_horizon)]}

    dataset = lr_dataset.LeRobotDataset(
        repo_id="local",
        root=args.dataset_path,
        delta_timestamps=delta_ts,
        download_videos=False,
    )
    print(f"Dataset loaded: {dataset.num_episodes} episodes, {dataset.num_frames} frames, {fps_dataset} fps")

    # Print dataset keys
    sample = dataset[0]
    print("Dataset keys:", list(sample.keys()))
    for k, v in sample.items():
        if isinstance(v, (torch.Tensor, np.ndarray)):
            arr = _to_numpy(v)
            print(f"  {k}: shape={arr.shape} dtype={arr.dtype}")
        else:
            print(f"  {k}: {v!r}")

    all_mse = []
    all_inference_times = []
    total_start = time.perf_counter()

    ep_from = dataset.episode_data_index["from"]
    ep_to = dataset.episode_data_index["to"]

    for ep_i in range(args.start_traj, args.start_traj + args.trajs):
        if ep_i >= dataset.num_episodes:
            print(f"Episode {ep_i} out of range ({dataset.num_episodes} episodes total), stopping.")
            break

        ep_start = ep_from[ep_i].item()
        ep_end = ep_to[ep_i].item()
        ep_len = ep_end - ep_start
        steps = min(args.steps, ep_len) if args.steps is not None else ep_len
        print(f"\n--- Episode {ep_i}: frames [{ep_start}, {ep_end}), evaluating {steps} steps ---")

        pred_chunks = []
        gt_chunks = []
        traj_inference_times = []

        for offset in range(0, steps, action_horizon):
            flat_idx = ep_start + offset
            item = dataset[flat_idx]

            obs = _extract_obs(item, args.cam1_key, args.cam2_key,
                               args.state_key, args.default_prompt)

            t0 = time.perf_counter()
            output = policy.infer(obs)
            elapsed = time.perf_counter() - t0
            traj_inference_times.append(elapsed)

            pred = np.asarray(output["actions"])   # (action_horizon, 7)
            gt = _get_gt_actions(item, args.action_key)  # (action_horizon, action_dim)

            # Trim to common length (near episode end gt may be shorter)
            H = min(len(pred), len(gt))
            pred = pred[:H]
            gt = gt[:H, :pred.shape[1]]

            pred_chunks.append(pred)
            gt_chunks.append(gt)

            mse_chunk = float(np.mean((pred - gt) ** 2))
            print(f"  offset {offset:4d}: infer={elapsed:.3f}s  chunk_mse={mse_chunk:.6f}")

        traj_mse = float(np.mean((np.concatenate(pred_chunks) - np.concatenate(gt_chunks)) ** 2))
        print(f"Episode {ep_i} MSE: {traj_mse:.6f}")
        all_mse.append(traj_mse)
        all_inference_times.extend(traj_inference_times)

        if args.plot or args.save_plot_path:
            save = (
                args.save_plot_path.format(ep=ep_i)
                if args.save_plot_path and "{ep}" in args.save_plot_path
                else args.save_plot_path
            )
            _plot_trajectory(pred_chunks, gt_chunks, ep_i, save)

    total_elapsed = time.perf_counter() - total_start

    print("\n--- Timing Summary ---")
    print(f"Total inference calls: {len(all_inference_times)}")
    if all_inference_times:
        print(f"Mean inference time:   {np.mean(all_inference_times):.4f}s")
        print(f"Min  inference time:   {np.min(all_inference_times):.4f}s")
        print(f"Max  inference time:   {np.max(all_inference_times):.4f}s")
    print(f"Total wall time:       {total_elapsed:.2f}s")
    print(f"Average MSE across all episodes: {np.mean(all_mse):.6f}")
    print("Done")


if __name__ == "__main__":
    main(tyro.cli(ArgsConfig))
