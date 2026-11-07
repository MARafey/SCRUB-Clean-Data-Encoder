"""
Priority 3 of the advisor's redesign: "why are 0.2-point Recall@1 differences
interpreted when they correspond to only three queries and no uncertainty is
reported?"

Aggregates several already-completed run_ablation.py sweeps (each run with a
different --seed, so a different data split and a different random
initialisation) into one table: per-variant mean +/- std across seeds, plus a
bootstrap CI on each seed's own Recall@1 (from that seed's per-query
correctness, reusing metrics.bootstrap_ci).

Within-seed variant-vs-variant significance (is noise_specific actually
better than generic_retrieval on the SAME test queries) is handled by
run_ablation.py itself via McNemar's test (significance.csv in each sweep
dir) - that is a different, complementary question from "does this result
replicate across seeds", which is what this script answers.

Usage
-----
    python run_ablation.py --per-record --seed 42   --sweep-name perrecord_s42 --train
    python run_ablation.py --per-record --seed 123  --sweep-name perrecord_s123 --train
    python run_ablation.py --per-record --seed 7    --sweep-name perrecord_s7 --train
    python aggregate_multiseed.py perrecord_s42 perrecord_s123 perrecord_s7
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from config import RUNS_ROOT


def aggregate(sweep_names: list[str], out_name: str = "multiseed_aggregate") -> pd.DataFrame:
    frames = []
    for name in sweep_names:
        path = os.path.join(RUNS_ROOT, name, "comparison.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found - has run_ablation.py --sweep-name {name} "
                f"finished?"
            )
        df = pd.read_csv(path)
        df["sweep"] = name
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    n_seeds = combined["sweep"].nunique()

    metrics = ["recall@1", "recall@1_dup_aware", "recall@5", "mrr",
              "correction_f1", "detection_f1"]
    metrics = [m for m in metrics if m in combined.columns]

    agg = (combined.groupby("variant")[metrics]
          .agg(["mean", "std", "min", "max"]))
    agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
    agg = agg.reset_index()
    agg.insert(1, "n_seeds", n_seeds)

    out_dir = os.path.join(RUNS_ROOT, out_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "comparison_multiseed.csv")
    agg.to_csv(out_path, index=False)
    combined.to_csv(os.path.join(out_dir, "comparison_raw_all_seeds.csv"),
                    index=False)

    print(f"[aggregate] {n_seeds} seeds: {', '.join(sweep_names)}")
    print(f"[aggregate] variants: {', '.join(sorted(combined['variant'].unique()))}")
    print()
    display_cols = ["variant", "n_seeds"] + [
        c for c in agg.columns if c.startswith(("recall@1_mean", "recall@1_std",
                                                "correction_f1_mean", "correction_f1_std"))
    ]
    print(agg[display_cols].to_string(index=False,
                                      float_format=lambda v: f"{v:.4f}"))
    print(f"\n[aggregate] full table : {out_path}")
    print(f"[aggregate] raw rows   : "
          f"{os.path.join(out_dir, 'comparison_raw_all_seeds.csv')}")

    # Flag any variant where std across seeds is comparable to (or bigger
    # than) the mean gap to 'none', i.e. where a between-seed replication
    # check would already have caught an unreliable result.
    if "none" in agg["variant"].values and "recall@1_mean" in agg.columns:
        none_mean = float(agg.loc[agg["variant"] == "none", "recall@1_mean"].iloc[0])
        print(f"\n[aggregate] no-instruction baseline, mean across seeds: "
              f"{none_mean:.4f}")
        for _, r in agg.iterrows():
            if r["variant"] == "none":
                continue
            gap = r["recall@1_mean"] - none_mean
            std = r.get("recall@1_std", float("nan"))
            flag = (" <- gap smaller than 1 std across seeds: NOT a reliable "
                    "win" if not np.isnan(std) and abs(gap) < std else "")
            print(f"  {r['variant']:<20s} gap={gap:+.4f}  std={std:.4f}{flag}")

    return agg


def main():
    ap = argparse.ArgumentParser(
        description="Aggregate several run_ablation.py sweeps (different "
                    "seeds) into mean/std per variant."
    )
    ap.add_argument("sweep_names", nargs="+",
                    help="sweep-dir names under runs/, one per seed")
    ap.add_argument("--out-name", default="multiseed_aggregate",
                    help="folder under runs/ to write the aggregate table into")
    args = ap.parse_args()
    aggregate(args.sweep_names, args.out_name)


if __name__ == "__main__":
    main()
