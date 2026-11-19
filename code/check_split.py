"""Sanity check for split_mode: leakage of identical clean text across partitions."""
import numpy as np
from config import ExperimentConfig
from data import load_dataset

for mode in ("random", "grouped"):
    for seed in (42, 123):
        d = load_dataset(ExperimentConfig(dataset="russia_065", seed=seed, split_mode=mode), verbose=False)
        sp = {k: d.splits[k] for k in ("train", "val", "test")}
        texts = {k: set(v.clean_texts) for k, v in sp.items()}
        tt = len(texts["train"] & texts["test"])
        tv = len(texts["train"] & texts["val"])
        test_rows_text_in_train = sum(t in texts["train"] for t in sp["test"].clean_texts)
        allrows = np.concatenate([sp[k].indices for k in sp])
        print(f"{mode:8s} seed={seed}: sizes {len(sp['train'])}/{len(sp['val'])}/{len(sp['test'])}  "
              f"all rows used once={len(set(allrows)) == len(allrows) == len(d.canonical_texts)}  "
              f"test rows whose clean text is in train: {test_rows_text_in_train}  "
              f"(distinct train&test texts {tt}, train&val {tv})")
