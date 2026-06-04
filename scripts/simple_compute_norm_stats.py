"""Compute normalization statistics directly from parquet files, skipping video decoding."""

import glob
import pathlib

import numpy as np
import pandas as pd
import tqdm
import tyro

import openpi.shared.normalize as normalize


def main(dataset_dir: str, output_dir: str = ".", action_horizon: int = 50) -> None:
    parquets = sorted(glob.glob(f"{dataset_dir}/data/**/*.parquet", recursive=True))
    if not parquets:
        raise FileNotFoundError(f"No parquet files found in {dataset_dir}/data/")

    stats = {"state": normalize.RunningStats(), "actions": normalize.RunningStats()}

    for p in tqdm.tqdm(parquets, desc="Computing stats"):
        df = pd.read_parquet(p)
        state = np.stack(df["observation.state"].values).astype(np.float32)  # (T, 7)
        actions = np.stack(df["action"].values).astype(np.float32)           # (T, 7)

        T = len(state)
        # Build sliding windows: for each t, action sequence is actions[t:t+H]
        # Truncate at episode end (no padding).
        num_windows = max(0, T - action_horizon + 1)
        if num_windows == 0:
            continue

        # Shape: (num_windows, action_horizon, 7)
        action_seqs = np.stack([actions[t : t + action_horizon] for t in range(num_windows)])
        state_t = state[:num_windows]  # (num_windows, 7) — current state for each window

        # Mirror DeltaActions: joints 0-5 become action[t+k] - state[t], gripper (dim 6) stays absolute
        delta_seqs = action_seqs.copy()
        delta_seqs[:, :, :6] -= state_t[:, np.newaxis, :6]

        stats["state"].update(state_t)
        stats["actions"].update(delta_seqs)

    norm_stats = {key: s.get_statistics() for key, s in stats.items()}

    output_path = pathlib.Path(output_dir)
    print(f"Writing stats to: {output_path / 'norm_stats.json'}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
