"""
Visual results for the embedding space.

Three kinds of output, all landing in the run's TensorBoard directory:

1. Embedding Projector - dirty and clean vectors written as a checkpoint plus a
   metadata TSV, so the PROJECTOR tab gives an interactive PCA/t-SNE/UMAP view.
   Colour by `role` to see whether dirty points sit on top of their clean twins;
   search a pair_id to jump to one specific record and its corrupted version.

2. Static figures logged as TB images - pair-displacement scatter, similarity
   histograms, rank CDF, per-noise-type recall, threshold sweep. These are the
   figures that go in a write-up.

3. A nearest-neighbour dump - concrete retrieved examples, right and wrong,
   written to results/ as text. Useful for the failure analysis section.
"""
from __future__ import annotations

import io
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

_PALETTE = {
    "dirty": "#c2660f",
    "clean": "#1a7568",
    "grid": "#c9d2cb",
    "ink": "#1d2b26",
    "accent": "#8a4bd3",
}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _fig_to_image(fig) -> tf.Tensor:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    return tf.expand_dims(image, 0)


def _style(ax, title: str, xlabel: str = "", ylabel: str = ""):
    ax.set_title(title, fontsize=11, color=_PALETTE["ink"], pad=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, color=_PALETTE["grid"], linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=8)


def _project_2d(vectors: np.ndarray, method: str = "tsne",
                seed: int = 42) -> np.ndarray:
    """2-D projection, t-SNE if sklearn is available, PCA otherwise."""
    if method == "tsne":
        try:
            from sklearn.manifold import TSNE
            perplexity = max(5.0, min(30.0, (len(vectors) - 1) / 3.0))
            return TSNE(n_components=2, init="pca", perplexity=perplexity,
                        random_state=seed).fit_transform(vectors)
        except Exception:
            pass
    centred = vectors - vectors.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    return centred @ vt[:2].T


# --------------------------------------------------------------------------
# 1. TensorBoard Embedding Projector
# --------------------------------------------------------------------------
def write_projector(log_dir: str,
                    dirty_emb: np.ndarray,
                    clean_emb: np.ndarray,
                    dirty_texts: Sequence[str],
                    clean_texts: Sequence[str],
                    noise_types: Sequence[str],
                    correct: Sequence[bool],
                    tag: str = "embeddings") -> str:
    """Write vectors + metadata for the PROJECTOR tab.

    Dirty and clean vectors go into one tensor so the projector shows both
    populations in the same space; `role` in the metadata separates them.
    """
    from tensorboard.plugins import projector

    out_dir = os.path.join(log_dir, "projector")
    os.makedirs(out_dir, exist_ok=True)

    stacked = np.concatenate([dirty_emb, clean_emb], axis=0).astype(np.float32)
    n = len(dirty_emb)

    meta_path = os.path.join(out_dir, "metadata.tsv")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("role\tpair_id\tnoise\tretrieved_correct\ttext\n")
        for i in range(n):
            text = " ".join(str(dirty_texts[i]).split())[:120]
            f.write(f"dirty\t{i}\t{noise_types[i]}\t"
                    f"{'yes' if correct[i] else 'no'}\t{text}\n")
        for i in range(len(clean_emb)):
            text = " ".join(str(clean_texts[i]).split())[:120]
            f.write(f"clean\t{i}\tnone\tna\t{text}\n")

    var = tf.Variable(stacked, name=tag)
    ckpt = tf.train.Checkpoint(embedding=var)
    ckpt.save(os.path.join(out_dir, "embedding.ckpt"))

    cfg = projector.ProjectorConfig()
    emb = cfg.embeddings.add()
    emb.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
    emb.metadata_path = "metadata.tsv"
    projector.visualize_embeddings(out_dir, cfg)
    return out_dir


# --------------------------------------------------------------------------
# 2. Figures
# --------------------------------------------------------------------------
def figure_pair_displacement(dirty_emb: np.ndarray,
                             clean_emb: np.ndarray,
                             correct: np.ndarray,
                             max_pairs: int = 300,
                             seed: int = 42,
                             title_suffix: str = ""):
    """Project dirty and clean into 2-D and draw a line between each pair.

    Short lines mean the instruction pulled the corrupted record onto its clean
    twin; long lines are the failures worth reading.
    """
    n = min(max_pairs, len(dirty_emb))
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(dirty_emb), n, replace=False)

    joint = np.concatenate([dirty_emb[pick], clean_emb[pick]], axis=0)
    proj = _project_2d(joint, "tsne", seed)
    d2, c2 = proj[:n], proj[n:]
    ok = correct[pick].astype(bool)

    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    for i in range(n):
        ax.plot([d2[i, 0], c2[i, 0]], [d2[i, 1], c2[i, 1]],
                color=_PALETTE["clean"] if ok[i] else _PALETTE["accent"],
                alpha=0.30 if ok[i] else 0.65,
                linewidth=0.7 if ok[i] else 1.1, zorder=1)
    ax.scatter(c2[:, 0], c2[:, 1], s=26, c=_PALETTE["clean"],
               label="clean canonical", zorder=3, edgecolors="white", linewidths=0.5)
    ax.scatter(d2[:, 0], d2[:, 1], s=26, c=_PALETTE["dirty"], marker="^",
               label="dirty (instruction-conditioned)", zorder=3,
               edgecolors="white", linewidths=0.5)
    _style(ax, f"Pair displacement in embedding space{title_suffix}",
           "t-SNE 1", "t-SNE 2")
    ax.legend(fontsize=8, frameon=False, loc="best")
    fig.text(0.5, -0.02,
             "purple lines = top-1 retrieval was wrong",
             ha="center", fontsize=8, color=_PALETTE["ink"])
    return _fig_to_image(fig)


def figure_similarity_histogram(positive_sims: np.ndarray,
                                negative_sims: np.ndarray,
                                title_suffix: str = ""):
    """Overlapping histograms of true-pair vs hardest-negative similarity.

    The gap between the two modes is the quantity the instruction is supposed
    to widen; it moves before accuracy does.
    """
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    bins = np.linspace(
        float(min(positive_sims.min(), np.nanmin(negative_sims))) - 0.02,
        float(max(positive_sims.max(), np.nanmax(negative_sims))) + 0.02,
        60,
    )
    ax.hist(positive_sims, bins=bins, color=_PALETTE["clean"], alpha=0.72,
            label="true pair (dirty vs its clean twin)")
    ax.hist(negative_sims[~np.isnan(negative_sims)], bins=bins,
            color=_PALETTE["dirty"], alpha=0.62, label="hardest negative")
    ax.axvline(float(np.mean(positive_sims)), color=_PALETTE["clean"],
               linestyle="--", linewidth=1.2)
    _style(ax, f"Cosine similarity separation{title_suffix}",
           "cosine similarity", "records")
    ax.legend(fontsize=8, frameon=False)
    return _fig_to_image(fig)


def figure_rank_cdf(ranks: np.ndarray, title_suffix: str = ""):
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    srt = np.sort(ranks)
    cdf = np.arange(1, len(srt) + 1) / len(srt)
    ax.step(srt, cdf, color=_PALETTE["clean"], linewidth=1.8)
    ax.set_xscale("log")
    for k in (1, 5, 10):
        hit = float((ranks <= k).mean())
        ax.axvline(k, color=_PALETTE["grid"], linewidth=1.0, linestyle=":")
        ax.annotate(f"R@{k}={hit:.3f}", (k, 0.06), fontsize=8,
                    rotation=90, color=_PALETTE["ink"], ha="right")
    _style(ax, f"Rank of the correct canonical record{title_suffix}",
           "rank of gold (log scale)", "cumulative fraction")
    ax.set_ylim(0, 1.02)
    return _fig_to_image(fig)


def figure_noise_breakdown(per_noise: Dict[str, Dict[str, float]],
                           title_suffix: str = ""):
    if not per_noise:
        return None
    items = sorted(per_noise.items(), key=lambda kv: kv[1]["recall@1"])
    names = [f"{k}  (n={v['n_records']})" for k, v in items]
    vals = [v["recall@1"] for _, v in items]

    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(items) + 1.8))
    bars = ax.barh(names, vals, color=_PALETTE["clean"], height=0.6)
    for bar, v in zip(bars, vals):
        ax.annotate(f"{v:.3f}", (v, bar.get_y() + bar.get_height() / 2),
                    xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8, color=_PALETTE["ink"])
    _style(ax, f"Recall@1 by corruption type{title_suffix}", "recall@1", "")
    ax.set_xlim(0, 1.06)
    return _fig_to_image(fig)


def figure_threshold_sweep(table: pd.DataFrame, title_suffix: str = ""):
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(table["threshold"], table["correction_f1"],
            color=_PALETTE["clean"], linewidth=2.0, label="correction F1")
    ax.plot(table["threshold"], table["correction_precision"],
            color=_PALETTE["clean"], linewidth=1.1, linestyle="--",
            label="correction precision")
    ax.plot(table["threshold"], table["correction_recall"],
            color=_PALETTE["clean"], linewidth=1.1, linestyle=":",
            label="correction recall")
    ax.plot(table["threshold"], table["detection_f1"],
            color=_PALETTE["dirty"], linewidth=1.6, label="detection F1")

    # The number to beat, from the AddressTable paper's best baseline.
    ax.axhline(0.29, color=_PALETTE["accent"], linewidth=1.2, linestyle="-.")
    ax.annotate("published best correction F1 = 0.29",
                (table["threshold"].iloc[0], 0.30), fontsize=8,
                color=_PALETTE["accent"])

    _style(ax, f"Abstention threshold sweep{title_suffix}",
           "top-1 cosine similarity threshold", "score")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, frameon=False, loc="best")
    return _fig_to_image(fig)


# --------------------------------------------------------------------------
# 3. Qualitative dump
# --------------------------------------------------------------------------
def dump_examples(path: str,
                  dirty_texts: Sequence[str],
                  clean_texts: Sequence[str],
                  retrieved_texts: Sequence[str],
                  scores: np.ndarray,
                  ranks: np.ndarray,
                  n_success: int = 15,
                  n_failure: int = 25) -> str:
    """Write concrete hits and misses for failure analysis."""
    order_fail = np.argsort(-ranks)
    order_ok = np.where(ranks == 1)[0]

    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 78 + "\nFAILURES (worst ranks first)\n" + "=" * 78 + "\n\n")
        for i in order_fail[:n_failure]:
            f.write(f"rank={ranks[i]}  top1_sim={scores[i]:.4f}\n")
            f.write(f"  dirty     : {dirty_texts[i]}\n")
            f.write(f"  gold clean: {clean_texts[i]}\n")
            f.write(f"  retrieved : {retrieved_texts[i]}\n\n")

        f.write("=" * 78 + "\nSUCCESSES\n" + "=" * 78 + "\n\n")
        for i in order_ok[:n_success]:
            f.write(f"top1_sim={scores[i]:.4f}\n")
            f.write(f"  dirty     : {dirty_texts[i]}\n")
            f.write(f"  retrieved : {retrieved_texts[i]}\n\n")
    return path


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def log_all_visuals(writer: tf.summary.SummaryWriter,
                    step: int,
                    dirty_emb: np.ndarray,
                    clean_emb: np.ndarray,
                    positive_sims: np.ndarray,
                    negative_sims: np.ndarray,
                    ranks: np.ndarray,
                    per_noise: Dict[str, Dict[str, float]],
                    threshold_table: Optional[pd.DataFrame] = None,
                    max_points: int = 400,
                    seed: int = 42,
                    suffix: str = "") -> None:
    correct = (ranks <= 1)
    with writer.as_default():
        img = figure_pair_displacement(dirty_emb, clean_emb, correct,
                                       max_pairs=max_points, seed=seed,
                                       title_suffix=suffix)
        tf.summary.image("viz/pair_displacement", img, step=step)

        tf.summary.image("viz/similarity_separation",
                         figure_similarity_histogram(positive_sims,
                                                     negative_sims, suffix),
                         step=step)
        tf.summary.image("viz/rank_cdf",
                         figure_rank_cdf(ranks, suffix), step=step)

        nb = figure_noise_breakdown(per_noise, suffix)
        if nb is not None:
            tf.summary.image("viz/recall_by_noise_type", nb, step=step)

        if threshold_table is not None and len(threshold_table):
            tf.summary.image("viz/threshold_sweep",
                             figure_threshold_sweep(threshold_table, suffix),
                             step=step)
        writer.flush()
