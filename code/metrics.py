"""
Evaluation metrics, aligned with what the literature reports.

Two families, because the papers in Papers/ report two different things:

RETRIEVAL  (INSTRUCTOR / E5 / Sudowoodo / fuzzylink style)
    Recall@k, MRR, median rank. "Given the dirty record, does its clean
    canonical twin come back at the top?"

CELL-LEVEL DETECTION AND CORRECTION  (Clean Me If You Can, Raha/Baran,
HoloClean, DataWig style)
    Precision / Recall / F1 over individual cells. This is the family that
    produces the ~0.29 correction F1 the AddressTable paper reports for its
    best baseline, so it is the number this project has to be compared on.

    Mapping retrieval onto that protocol: retrieve the top-1 canonical record
    for a dirty record, and read its field values as the proposed repair.
      detection  -> a cell is flagged when the retrieved value differs from the
                    dirty value; ground truth is (clean != dirty).
      correction -> a flagged cell is repaired correctly when the retrieved
                    value equals the clean value.
    An abstention threshold on top-1 similarity lets the system decline to
    propose anything, which is what makes precision and recall trade off.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Similarity search
# --------------------------------------------------------------------------
def search(query_emb: np.ndarray,
           canonical_emb: np.ndarray,
           top_k: int = 10,
           chunk_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Exact cosine top-k search, chunked to bound memory.

    Both matrices are assumed L2-normalised, so a dot product is the cosine.

    Returns:
        scores  [N, k] descending similarities
        indices [N, k] row ids into canonical_emb
    """
    n = query_emb.shape[0]
    k = min(top_k, canonical_emb.shape[0])
    out_scores = np.zeros((n, k), dtype=np.float32)
    out_index = np.zeros((n, k), dtype=np.int64)

    for start in range(0, n, chunk_size):
        block = query_emb[start:start + chunk_size]
        sims = block @ canonical_emb.T                       # [b, N_canonical]
        part = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        part_scores = np.take_along_axis(sims, part, axis=1)
        order = np.argsort(-part_scores, axis=1)
        out_index[start:start + chunk_size] = np.take_along_axis(part, order, axis=1)
        out_scores[start:start + chunk_size] = np.take_along_axis(
            part_scores, order, axis=1
        )
    return out_scores, out_index


def rank_of_gold(query_emb: np.ndarray,
                 canonical_emb: np.ndarray,
                 gold_indices: np.ndarray,
                 chunk_size: int = 512) -> np.ndarray:
    """1-based rank of each gold canonical record. Exact, no top-k cutoff."""
    ranks = np.zeros(query_emb.shape[0], dtype=np.int64)
    for start in range(0, query_emb.shape[0], chunk_size):
        block = query_emb[start:start + chunk_size]
        gold = gold_indices[start:start + chunk_size]
        sims = block @ canonical_emb.T
        gold_sim = sims[np.arange(len(gold)), gold][:, None]
        # Count strictly-better candidates; ties resolved pessimistically.
        ranks[start:start + chunk_size] = (sims > gold_sim).sum(axis=1) + 1
    return ranks


# --------------------------------------------------------------------------
# Retrieval metrics
# --------------------------------------------------------------------------
def retrieval_metrics(ranks: np.ndarray,
                      ks: Sequence[int] = (1, 5, 10)) -> Dict[str, float]:
    if len(ranks) == 0:
        return {f"recall@{k}": 0.0 for k in ks} | {"mrr": 0.0, "median_rank": 0.0}
    out = {f"recall@{k}": float((ranks <= k).mean()) for k in ks}
    out["mrr"] = float((1.0 / ranks).mean())
    out["median_rank"] = float(np.median(ranks))
    out["mean_rank"] = float(ranks.mean())
    return out


def duplicate_aware_retrieval_metrics(indices: np.ndarray,
                                      canonical_texts: Sequence[str],
                                      gold_indices: np.ndarray,
                                      ks: Sequence[int] = (1, 5, 10)
                                      ) -> Dict[str, float]:
    """Recall@k that also credits a text-identical duplicate of gold.

    AddressTable has real duplicate canonical records (see losses.py's module
    docstring): the strict, row-index-based `retrieval_metrics` scores a
    retrieval as wrong whenever the top-k candidate is a byte-identical
    duplicate of the true clean record at a *different* row index - which can
    happen purely from floating-point tie-breaking between two embeddings of
    identical text, not a real retrieval failure. This computes recall using
    text equality against the gold record instead of row-index equality.

    Args:
        indices: [N, k] canonical row ids, as returned by `search(...)`.
        canonical_texts: the full canonical text list (same order as the
            embeddings `indices` was computed against).
        gold_indices: true clean row id for each query.
    """
    if len(gold_indices) == 0:
        return {f"recall@{k}_dup_aware": 0.0 for k in ks}
    gold_texts = [canonical_texts[g] for g in gold_indices]
    out = {}
    for k in ks:
        kk = min(k, indices.shape[1])
        hit = np.array([
            any(canonical_texts[indices[i, j]] == gold_texts[i] for j in range(kk))
            for i in range(len(gold_indices))
        ])
        out[f"recall@{k}_dup_aware"] = float(hit.mean())
    return out


def embedding_diagnostics(query_emb: np.ndarray,
                          canonical_emb: np.ndarray,
                          gold_indices: np.ndarray,
                          top1_scores: np.ndarray,
                          top1_indices: np.ndarray) -> Dict[str, float]:
    """Does the instruction actually move the geometry?

    Reported separately from accuracy because the embedding space can tighten
    (larger margin) even when top-1 accuracy is unchanged, and that is evidence
    the instruction is doing something.
    """
    gold_sim = np.sum(query_emb * canonical_emb[gold_indices], axis=1)
    # Best non-gold candidate among the retrieved top-k.
    wrong = top1_indices != gold_indices
    hardest_neg = np.where(wrong, top1_scores, np.nan)

    return {
        "positive_similarity": float(np.mean(gold_sim)),
        "positive_similarity_std": float(np.std(gold_sim)),
        "hardest_negative_similarity": float(np.nanmean(hardest_neg))
        if np.any(wrong) else 0.0,
        "separation_margin": float(
            np.mean(gold_sim) - np.nanmean(hardest_neg)
        ) if np.any(wrong) else float(np.mean(gold_sim)),
    }


# --------------------------------------------------------------------------
# Cell-level detection and correction (paper-comparable)
# --------------------------------------------------------------------------
def _prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": int(tp), "fp": int(fp), "fn": int(fn)}


def cell_level_metrics(clean_df: pd.DataFrame,
                       dirty_df: pd.DataFrame,
                       query_rows: np.ndarray,
                       retrieved_rows: np.ndarray,
                       top1_scores: np.ndarray,
                       fields: Sequence[str],
                       threshold: float = 0.0) -> Dict[str, Dict[str, float]]:
    """Detection and correction P/R/F1 over cells, at one abstention threshold.

    Args:
        query_rows:     row ids of the dirty records being evaluated
        retrieved_rows: row id of the top-1 canonical record for each query
        top1_scores:    top-1 cosine similarity, used for abstention
        threshold:      propose a repair only when top1 similarity >= threshold
    """
    proposes = top1_scores >= threshold

    det_tp = det_fp = det_fn = 0
    cor_tp = cor_fp = cor_fn = 0

    clean_vals = clean_df[list(fields)].astype(str).values
    dirty_vals = dirty_df[list(fields)].astype(str).values

    for i, q in enumerate(query_rows):
        gold_row = clean_vals[q]
        dirty_row = dirty_vals[q]
        proposed_row = clean_vals[retrieved_rows[i]]
        active = proposes[i]

        for f in range(len(fields)):
            truly_wrong = dirty_row[f] != gold_row[f]
            if not active:
                # Abstained: every real error in this record is a miss.
                if truly_wrong:
                    det_fn += 1
                    cor_fn += 1
                continue

            flagged = proposed_row[f] != dirty_row[f]

            # --- detection ---
            if flagged and truly_wrong:
                det_tp += 1
            elif flagged and not truly_wrong:
                det_fp += 1
            elif not flagged and truly_wrong:
                det_fn += 1

            # --- correction ---
            # Only cells the system chose to change count as proposals.
            if flagged:
                if proposed_row[f] == gold_row[f]:
                    cor_tp += 1
                else:
                    cor_fp += 1
            elif truly_wrong:
                cor_fn += 1

    return {
        "detection": _prf(det_tp, det_fp, det_fn),
        "correction": _prf(cor_tp, cor_fp, cor_fn),
    }


def sweep_thresholds(clean_df: pd.DataFrame,
                     dirty_df: pd.DataFrame,
                     query_rows: np.ndarray,
                     retrieved_rows: np.ndarray,
                     top1_scores: np.ndarray,
                     fields: Sequence[str],
                     thresholds: Sequence[float]) -> Dict[str, object]:
    """Run cell_level_metrics across thresholds and pick the best-F1 operating
    point. The best correction F1 is the headline number to place next to the
    AddressTable paper's ~0.29."""
    rows = []
    for t in thresholds:
        m = cell_level_metrics(clean_df, dirty_df, query_rows, retrieved_rows,
                               top1_scores, fields, threshold=t)
        rows.append({
            "threshold": float(t),
            "detection_precision": m["detection"]["precision"],
            "detection_recall": m["detection"]["recall"],
            "detection_f1": m["detection"]["f1"],
            "correction_precision": m["correction"]["precision"],
            "correction_recall": m["correction"]["recall"],
            "correction_f1": m["correction"]["f1"],
        })

    table = pd.DataFrame(rows)
    best_cor = table.loc[table["correction_f1"].idxmax()].to_dict()
    best_det = table.loc[table["detection_f1"].idxmax()].to_dict()
    return {
        "table": table,
        "best_correction": best_cor,
        "best_detection": best_det,
    }


# --------------------------------------------------------------------------
# Significance testing across variants and seeds (Priority 3 of the
# advisor's redesign - "why are 0.2-point Recall@1 differences interpreted
# when they correspond to only three queries and no uncertainty is
# reported?"). Two complementary tools:
#   mcnemar_test       - paired, same test set, two variants/seeds compared
#                        query-by-query (correct/incorrect). The right test
#                        here because the same 1,500ish test queries are
#                        reused across every variant - these are NOT
#                        independent samples.
#   bootstrap_ci       - a confidence interval on one variant's Recall@1
#                        from resampling test queries with replacement.
# --------------------------------------------------------------------------
def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray) -> Dict[str, float]:
    """Paired McNemar's test on two variants' per-query correct/incorrect
    outcomes over the SAME test queries.

    Args:
        correct_a, correct_b: boolean arrays, same length, same query order
            (e.g. `ranks <= 1` for two different instruction variants run on
            the same test split and the same random seed).

    Returns a dict with the counts that matter (n_only_a right, n_only_b
    right, n_both_right, n_both_wrong), the McNemar statistic (continuity-
    corrected chi-square, 1 df) and its p-value. Only the discordant pairs
    (queries where the two variants disagree) carry information; a and b
    getting the same answer never moves the comparison in either direction.
    """
    correct_a = np.asarray(correct_a, dtype=bool)
    correct_b = np.asarray(correct_b, dtype=bool)
    if correct_a.shape != correct_b.shape:
        raise ValueError(
            f"shape mismatch: {correct_a.shape} vs {correct_b.shape} - "
            f"both variants must be evaluated on the same test queries, "
            f"in the same order, for this test to be valid."
        )
    only_a = int(np.sum(correct_a & ~correct_b))   # a right, b wrong
    only_b = int(np.sum(~correct_a & correct_b))   # b right, a wrong
    both_right = int(np.sum(correct_a & correct_b))
    both_wrong = int(np.sum(~correct_a & ~correct_b))

    n_discordant = only_a + only_b
    if n_discordant == 0:
        chi2, p_value = 0.0, 1.0
    else:
        # Continuity-corrected McNemar statistic (standard choice when
        # n_discordant can be small, as it often will be here).
        chi2 = ((abs(only_a - only_b) - 1) ** 2) / n_discordant
        try:
            from scipy import stats as _stats
            p_value = float(1 - _stats.chi2.cdf(chi2, df=1))
        except ImportError:
            # No scipy available: fall back to a normal approximation of the
            # same statistic (McNemar's chi2 with 1 df is z^2 for
            # z = (only_a - only_b) / sqrt(n_discordant)), using the error
            # function from the standard library only.
            import math
            z = abs(only_a - only_b) / math.sqrt(n_discordant)
            p_value = float(math.erfc(z / math.sqrt(2)))

    return {
        "n_queries": int(len(correct_a)),
        "n_only_a_correct": only_a,
        "n_only_b_correct": only_b,
        "n_both_correct": both_right,
        "n_both_wrong": both_wrong,
        "n_discordant": n_discordant,
        "chi2": float(chi2),
        "p_value": float(p_value),
    }


def bootstrap_ci(correct: np.ndarray,
                 n_boot: int = 2000,
                 ci: float = 0.95,
                 seed: int = 0) -> Dict[str, float]:
    """Bootstrap confidence interval on a Recall@k (or any mean-of-binary-
    outcomes metric) by resampling test queries with replacement.

    Args:
        correct: boolean (or 0/1) array, one entry per test query.
    """
    correct = np.asarray(correct, dtype=float)
    n = len(correct)
    point = float(correct.mean()) if n else 0.0
    if n == 0:
        return {"point": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n_boot": n_boot}
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.integers(0, n, size=n)
        boot_means[i] = correct[sample].mean()
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(boot_means, [alpha, 1.0 - alpha])
    return {
        "point": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "ci_level": ci,
        "n_boot": n_boot,
    }


# --------------------------------------------------------------------------
# Breakdown by noise type
# --------------------------------------------------------------------------
def per_noise_type_recall(noise_labels: pd.DataFrame,
                          query_rows: np.ndarray,
                          ranks: np.ndarray,
                          fields: Sequence[str],
                          noise_types: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """Recall@1 restricted to records containing each noise type.

    A record can carry several noise types, so the groups overlap. This is the
    analysis that says *which* corruption the instruction is actually helping
    the model see past, rather than only that it helps on average.
    """
    out: Dict[str, Dict[str, float]] = {}
    labels = noise_labels[list(fields)].values

    for noise in noise_types:
        if noise == "unchanged":
            continue
        member = np.array([
            noise in set(labels[q]) for q in query_rows
        ], dtype=bool)
        if member.sum() == 0:
            continue
        out[noise] = {
            "n_records": int(member.sum()),
            "recall@1": float((ranks[member] <= 1).mean()),
            "mrr": float((1.0 / ranks[member]).mean()),
        }
    return out


def per_corruption_bucket(noise_labels: pd.DataFrame,
                          query_rows: np.ndarray,
                          ranks: np.ndarray,
                          fields: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """Recall@1 bucketed by how many fields in the record were corrupted."""
    labels = noise_labels[list(fields)].values
    counts = np.array([int((labels[q] != "unchanged").sum()) for q in query_rows])

    buckets = {"0": counts == 0, "1": counts == 1, "2": counts == 2,
               "3-4": (counts >= 3) & (counts <= 4), "5+": counts >= 5}
    out = {}
    for name, mask in buckets.items():
        if mask.sum() == 0:
            continue
        out[name] = {
            "n_records": int(mask.sum()),
            "recall@1": float((ranks[mask] <= 1).mean()),
            "mrr": float((1.0 / ranks[mask]).mean()),
        }
    return out
