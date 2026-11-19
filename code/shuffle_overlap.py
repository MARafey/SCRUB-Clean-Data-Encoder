"""How different is the Record-Shuffled control from the true label?

For each test row, compare the corruption-type set of the row itself with the
set of the row whose instruction it receives under the seeded derangement
(`AddressData.noise_shuffle_map`). Reports the share of rows whose shuffled
instruction carries an identical, partially overlapping, or disjoint set.

    run.bat shuffle_overlap.py                      # russia_065, seeds 42,123
    run.bat shuffle_overlap.py dataset=de_048 seed=7
"""
from __future__ import annotations

import sys

import numpy as np

import data as D
from config import ExperimentConfig


def overlap_fractions(dataset: str, seed: int) -> dict[str, float]:
    cfg = ExperimentConfig(dataset=dataset, seed=seed)
    d = D.load_dataset(cfg, verbose=False)
    idx = np.asarray(d.splits["test"].indices)
    identical = partial = disjoint = 0
    for i in idx:
        own = set(d.row_noise_types(int(i)))
        other = set(d.row_noise_types(int(d.noise_shuffle_map[i])))
        if own == other:
            identical += 1
        elif own & other:
            partial += 1
        else:
            disjoint += 1
    n = len(idx)
    return {
        "n_test": n,
        "identical": identical / n,
        "partial_overlap": partial / n,
        "disjoint": disjoint / n,
    }


if __name__ == "__main__":
    args = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    dataset = args.get("dataset", "russia_065")
    seeds = [int(args["seed"])] if "seed" in args else [42, 123]
    for seed in seeds:
        r = overlap_fractions(dataset, seed)
        print(f"{dataset} seed={seed} n_test={r['n_test']}: "
              f"identical {100 * r['identical']:.1f}%  "
              f"partial overlap {100 * r['partial_overlap']:.1f}%  "
              f"disjoint {100 * r['disjoint']:.1f}%")
