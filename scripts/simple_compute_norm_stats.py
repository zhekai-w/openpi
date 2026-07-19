"""Compute normalization statistics directly from parquet files, skipping video decoding.

Writes to the same location as compute_norm_stats.py: <config.assets_dirs>/<repo_id>/norm_stats.json.
"""

import glob

from lerobot.common.constants import HF_LEROBOT_HOME
import numpy as np
import pandas as pd
import tqdm
import tyro

import openpi.shared.normalize as normalize
import openpi.training.config as _config


def main(config_name: str, repo_id: str | None = None, action_horizon: int | None = None) -> None:
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    if repo_id is None:
        repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Data config must have a repo_id")
    if action_horizon is None:
        action_horizon = config.model.action_horizon

    dataset_dir = HF_LEROBOT_HOME / repo_id
    parquets = sorted(glob.glob(f"{dataset_dir}/data/**/*.parquet", recursive=True))
    if not parquets:
        raise FileNotFoundError(f"No parquet files found in {dataset_dir}/data/")

    stats = {"state": normalize.RunningStats(), "actions": normalize.RunningStats()}

    for p in tqdm.tqdm(parquets, desc="Computing stats"):
        df = pd.read_parquet(p)
        state = np.stack(df["observation.state"].values).astype(np.float32)  # (T, 7)
        actions = np.stack(df["action"].values).astype(np.float32)           # (T, 7)

        T = len(state)
        if T == 0:
            continue

        # Build sliding windows for every frame t in [0, T), mirroring LeRobotDataset's
        # delta_timestamps indexing: future indices beyond the episode end are clamped to
        # the last valid frame (T - 1), i.e. padded by repeating the last action, rather
        # than dropped. This must match exactly, or the tail of every episode is weighted
        # differently and stats (esp. the absolute gripper dim) drift from compute_norm_stats.py.
        idx = np.minimum(np.arange(T)[:, None] + np.arange(action_horizon)[None, :], T - 1)  # (T, H)
        action_seqs = actions[idx]  # (T, H, 7)
        state_t = state  # (T, 7) — current state for each window

        # Mirror DeltaActions: joints 0-5 become action[t+k] - state[t], gripper (dim 6) stays absolute
        delta_seqs = action_seqs.copy()
        delta_seqs[:, :, :6] -= state_t[:, np.newaxis, :6]

        stats["state"].update(state_t)
        stats["actions"].update(delta_seqs)

    norm_stats = {key: s.get_statistics() for key, s in stats.items()}

    output_path = config.assets_dirs / repo_id
    print(f"Writing stats to: {output_path / 'norm_stats.json'}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
