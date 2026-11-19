"""Paired comparisons from saved per-query outputs (no training).

Why this exists. The split is seeded, so every seed has a DIFFERENT test set.
A model scored on the seed-123 split must be compared with a baseline scored on
the seed-123 split, not the default one. This script
  1. scores TF-IDF on the same seeded split as each model run, choosing its
     correction threshold on the VALIDATION split (as the encoders do);
  2. computes strict row-index and duplicate-aware (text-equality) Recall@1 for
     every run from its test_per_query.csv;
  3. reports paired McNemar (continuity-corrected) and a Wald 95% interval for
     the paired difference in Recall@1 for each comparison.

    uv run python paired_analysis.py --runs ../runs --out ../runs/paired_analysis.json
"""
from __future__ import annotations

import argparse
import json
import os
from math import erfc, sqrt

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

import metrics as M
from config import ExperimentConfig
from data import load_dataset


def mcnemar(a: np.ndarray, b: np.ndarray) -> dict:
    """a, b: boolean per-query correctness on the same queries."""
    n = len(a)
    only_a = int(np.sum(a & ~b))
    only_b = int(np.sum(~a & b))
    disc = only_a + only_b
    chi2 = 0.0 if disc == 0 else (abs(only_a - only_b) - 1) ** 2 / disc
    p = 1.0 if disc == 0 else float(erfc(sqrt(chi2 / 2)))
    d = (only_a - only_b) / n
    se = sqrt(max(disc - (only_a - only_b) ** 2 / n, 0.0)) / n
    return {"n": n, "only_a": only_a, "only_b": only_b, "diff_pts": 100 * d,
            "ci_lo": 100 * (d - 1.96 * se), "ci_hi": 100 * (d + 1.96 * se),
            "chi2": chi2, "p": p}


class SplitCtx:
    """Dataset + seeded split + TF-IDF top-1 for one seed."""

    def __init__(self, seed: int, dataset: str = "russia_065",
                 split_mode: str = "random"):
        self.seed = seed
        cfg = ExperimentConfig(dataset=dataset, seed=seed, split_mode=split_mode)
        self.cfg = cfg
        self.data = load_dataset(cfg, verbose=False)
        self.canon = self.data.canonical_texts
        self.test_rows = np.asarray(self.data.splits["test"].indices)
        self.val_rows = np.asarray(self.data.splits["val"].indices)
        self._tfidf = None

    def dup_correct(self, query_rows, top1_rows) -> np.ndarray:
        return np.array([self.canon[t] == self.canon[q]
                         for q, t in zip(query_rows, top1_rows)])

    def _tfidf_top1(self, split: str):
        d = self.data
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2).fit(
            d.canonical_texts + d.dirty_texts)
        C = normalize(vec.transform(d.canonical_texts)).astype(np.float32)
        sp = d.splits[split]
        Q = normalize(vec.transform(sp.dirty_texts)).astype(np.float32)
        rows = np.asarray(sp.indices)
        top1i = np.zeros(len(rows), dtype=np.int64)
        top1s = np.zeros(len(rows), dtype=np.float32)
        for s in range(0, Q.shape[0], 256):
            sims = (Q[s:s + 256] @ C.T).toarray()
            top1i[s:s + 256] = sims.argmax(axis=1)
            top1s[s:s + 256] = sims.max(axis=1)
        return rows, top1i, top1s

    def tfidf(self) -> dict:
        """TF-IDF per-query correctness on test; F1 threshold chosen on val."""
        if self._tfidf is not None:
            return self._tfidf
        d = self.data
        vrows, vi, vs = self._tfidf_top1("val")
        # TF-IDF cosine scores sit lower than the encoders', so the encoder grid
        # (0.50-0.98) would force it to abstain; give it a grid from 0.0. The
        # threshold is still chosen on VALIDATION only.
        grid = [round(0.02 * i, 2) for i in range(0, 50)]
        sweep = M.sweep_thresholds(d.clean_df, d.dirty_df, vrows, vi, vs, d.fields,
                                   grid)
        thr = float(sweep["best_correction"]["threshold"])
        trows, ti, ts = self._tfidf_top1("test")
        strict = ti == trows
        dup = self.dup_correct(trows, ti)
        cell = M.cell_level_metrics(d.clean_df, d.dirty_df, trows, ti, ts, d.fields,
                                    threshold=thr)
        self._tfidf = {"rows": trows, "strict": strict, "dup": dup,
                       "val_threshold": thr,
                       "correction_f1_valthr": cell["correction"]["f1"]}
        return self._tfidf

    def run_correct(self, per_query_csv: str) -> dict:
        df = pd.read_csv(per_query_csv)
        rows = df["query_row"].to_numpy()
        assert set(rows) == set(self.test_rows), "run was not scored on this split"
        # align to the split's query order
        df = df.set_index("query_row").loc[self.test_rows].reset_index()
        # strict = argmax row-index; dup = top-1 text equals gold text. Both are
        # computed from the top-1 row so every system is scored identically.
        # (The CSV's own `correct` column is rank-based and credits exact score
        # ties, which TF-IDF always gets and neural embeddings only sometimes.)
        top1 = df["top1_row"].to_numpy()
        strict = top1 == df["query_row"].to_numpy()
        dup = self.dup_correct(df["query_row"].to_numpy(), top1)
        legacy = df["correct"].to_numpy().astype(bool)
        return {"rows": self.test_rows, "strict": strict, "dup": dup,
                "legacy": legacy}


def run_correct_npz(ctx: "SplitCtx", npz_path: str) -> dict:
    """For runs that only saved test_embeddings.npz (query_rows, top1_idx)."""
    z = np.load(npz_path, allow_pickle=True)
    rows = z["query_rows"]
    assert set(rows) == set(ctx.test_rows), "run was not scored on this split"
    order = {r: i for i, r in enumerate(rows)}
    idx = np.array([order[r] for r in ctx.test_rows])
    top1 = z["top1_idx"][idx]
    strict = top1 == ctx.test_rows
    return {"rows": ctx.test_rows, "strict": strict,
            "dup": ctx.dup_correct(ctx.test_rows, top1)}


def reported_f1(runs_dir: str, run: str):
    p = os.path.join(runs_dir, run, "results", "test_metrics.json")
    if not os.path.exists(p):
        return None
    m = json.load(open(p))
    return 100 * m["best_correction"]["correction_f1"]


def fmt(r: dict) -> str:
    return (f"{r['diff_pts']:+.2f} [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] "
            f"b/c={r['only_a']}/{r['only_b']} chi2={r['chi2']:.3f} p={r['p']:.3f}")


def compare(out: dict, ctx_name: str, name_a: str, A: dict, name_b: str, B: dict):
    for metric in ("strict", "dup"):
        r = mcnemar(A[metric], B[metric])
        key = f"{ctx_name}|{name_a} vs {name_b}|{metric}"
        out[key] = r
        print(f"  [{metric:6s}] {name_a} ({100 * A[metric].mean():.2f}) vs "
              f"{name_b} ({100 * B[metric].mean():.2f}): {fmt(r)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="../runs")
    ap.add_argument("--out", default="../runs/paired_analysis.json")
    ap.add_argument("--modes", default="random,grouped")
    ap.add_argument("--seeds", default="42,123")
    a = ap.parse_args()
    R = a.runs
    pq = lambda p: os.path.join(R, p, "results", "test_per_query.csv")
    npz = lambda p: os.path.join(R, p, "results", "test_embeddings.npz")
    out: dict = {}

    # (mode, seed) -> {display name: run dir}. Earlier runs all used the random
    # split; their directories predate split_mode, so they are listed explicitly.
    known = {
        ("random", 42): {"pretrained b8": "ablation_russia_065_pretrained/none",
                         "pretrained matched b4": "matched_pretrained_s42",
                         "random-init matched b4": "matched_randominit_s42",
                         "v3 scratch": "perrecord_s42/none",
                         "frozen DistilBERT": "frozen_random_s42"},
        ("random", 123): {"pretrained b8": "ablation_russia_065_pretrained_full7/none",
                          "v3 scratch": "perrecord_s123/none",
                          "frozen DistilBERT": "frozen_random_s123"},
    }
    for seed in (42, 123):
        known[("grouped", seed)] = {"pretrained b8": f"grp_pre_s{seed}",
                                    "v3 scratch": f"grp_v3_s{seed}",
                                    "frozen DistilBERT": f"frozen_grouped_s{seed}"}

    for mode in a.modes.split(","):
        for seed in [int(x) for x in a.seeds.split(",")]:
            cand = known.get((mode, seed), {})
            ctx_split = SplitCtx(seed, split_mode=mode)
            ctx = ctx_split
            tf = ctx.tfidf()
            tag = f"{mode}|seed{seed}"
            print(f"\n=== {mode} split, seed {seed} ({len(ctx.test_rows)} test queries) ===")
            print(f"TF-IDF: argmax-strict={100 * tf['strict'].mean():.2f}  "
                  f"dup-aware={100 * tf['dup'].mean():.2f}  "
                  f"corr F1 (val-selected thr {tf['val_threshold']:.2f})="
                  f"{100 * tf['correction_f1_valthr']:.2f}")
            out[f"{tag}|tfidf"] = {k: (float(v.mean()) if hasattr(v, "mean") else v)
                                   for k, v in tf.items() if k != "rows"}
            runs = {}
            for name, path in cand.items():
                if os.path.exists(pq(path)):
                    runs[name] = ctx.run_correct(pq(path))
                elif os.path.exists(npz(path)):
                    runs[name] = run_correct_npz(ctx, npz(path))
            for name, run in runs.items():
                f1 = reported_f1(R, cand[name])
                leg = f" legacy-rank={100 * run['legacy'].mean():.2f}" if "legacy" in run else ""
                print(f"{name}: argmax-strict={100 * run['strict'].mean():.2f}  "
                      f"dup-aware={100 * run['dup'].mean():.2f}{leg}  "
                      f"corr F1={'n/a' if f1 is None else f'{f1:.2f}'}")
                out[f"{tag}|{name}|summary"] = {
                    "strict": float(run["strict"].mean()),
                    "dup": float(run["dup"].mean()), "f1": f1}
                compare(out, tag, name, run, "TF-IDF", tf)
            for x, y in (("pretrained b8", "frozen DistilBERT"),
                         ("pretrained matched b4", "random-init matched b4"),
                         ("pretrained b8", "v3 scratch")):
                if x in runs and y in runs:
                    compare(out, tag, x, runs[x], y, runs[y])

            if mode == "random":   # record-specific family exists on the random split only
                base = f"perrecord_s{seed}"
                fam = {c: ctx.run_correct(pq(f"{base}/{c}"))
                       for c in ("none", "generic_retrieval", "record_specific",
                                 "record_shuffled", "record_nonsense")
                       if os.path.exists(pq(f"{base}/{c}"))}
                print(f"-- record-specific family, seed {seed}")
                for c in ("generic_retrieval", "record_specific", "record_shuffled",
                          "record_nonsense"):
                    if c in fam:
                        compare(out, tag, c, fam[c], "none", fam["none"])
                for c in ("record_shuffled", "record_nonsense"):
                    if c in fam and "record_specific" in fam:
                        compare(out, tag, c, fam[c], "record_specific",
                                fam["record_specific"])

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
