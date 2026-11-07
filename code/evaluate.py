"""
Evaluation driver.

Encodes every clean record once to build the canonical index, encodes the
split's dirty records with the query instruction, then reports both metric
families (retrieval and cell-level detection/correction) plus the embedding
diagnostics and all visuals.

The canonical index always spans the whole slice, not just the evaluated split,
so a hit has to beat every other clean record in the dataset. Evaluating
against a small candidate pool would inflate every number here.
"""
from __future__ import annotations

import json
import os
import warnings
from typing import Dict, Optional

import numpy as np
import pandas as pd
import tensorflow as tf

import metrics as M
import visualize as V
from config import ExperimentConfig
from data import AddressData, NOISE_TYPES
from instructions import get_instruction, is_per_record
from model import InstructionEncoder


def _to_scalar_dict(prefix: str, d: Dict) -> Dict[str, float]:
    """Flatten nested metric dicts into TB-loggable scalars."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_to_scalar_dict(f"{prefix}{k}/", v))
        elif isinstance(v, (int, float, np.floating, np.integer)):
            out[f"{prefix}{k}"] = float(v)
    return out


def pick_threshold_on_val(encoder: InstructionEncoder,
                          data: AddressData,
                          cfg: ExperimentConfig,
                          canonical_cache: Optional[Dict[str, np.ndarray]] = None
                          ) -> float:
    """Select the abstention threshold on the VALIDATION split only.

    This threshold is what `evaluate(..., frozen_threshold=...)` then applies,
    unseen, to the test split. Selecting it on test instead (as earlier
    versions of this pipeline did, by always sweeping on whatever split was
    passed to `evaluate`) leaks test-set information into the "best" F1
    number: the threshold itself becomes a parameter fit to the test labels.
    Every correction-F1 number reported for test must come from a threshold
    chosen here, on val, never from a sweep run directly on test.
    """
    val = data.splits["val"]
    q_instr = (
        data.per_record_instructions(val.indices,
                                     shuffled=(cfg.query_instruction == "record_shuffled"))
        if is_per_record(cfg.query_instruction) else get_instruction(cfg.query_instruction)
    )
    d_instr = get_instruction(cfg.doc_instruction)

    cache_key = f"{cfg.doc_instruction}|{cfg.serialization}|{cfg.pool_over_instruction}"
    if canonical_cache is not None and cache_key in canonical_cache:
        canonical_emb = canonical_cache[cache_key]
    else:
        canonical_emb = encoder.encode(
            data.canonical_texts, d_instr, batch_size=cfg.eval_batch_size
        )
        if canonical_cache is not None:
            canonical_cache[cache_key] = canonical_emb

    query_emb = encoder.encode(val.dirty_texts, q_instr, batch_size=cfg.eval_batch_size)
    top_k = max(cfg.recall_at_k) if cfg.recall_at_k else 10
    scores, indices = M.search(query_emb, canonical_emb, top_k=top_k)
    top1_scores, top1_idx = scores[:, 0], indices[:, 0]

    sweep = M.sweep_thresholds(
        data.clean_df, data.dirty_df, val.indices, top1_idx, top1_scores,
        data.fields, cfg.similarity_thresholds,
    )
    return float(sweep["best_correction"]["threshold"])


def evaluate(encoder: InstructionEncoder,
             data: AddressData,
             cfg: ExperimentConfig,
             split: str = "test",
             writer: Optional[tf.summary.SummaryWriter] = None,
             step: int = 0,
             log_visuals: bool = True,
             save_results: bool = True,
             verbose: bool = True,
             canonical_cache: Optional[Dict[str, np.ndarray]] = None,
             frozen_threshold: Optional[float] = None) -> Dict:
    """Evaluate one instruction configuration.

    canonical_cache: optional dict reused across calls that share a frozen
    encoder. The canonical side is encoded with `doc_instruction`, which is
    "none" for every row of a standard ablation, so without this the sweep
    re-encodes the entire clean index once per variant for identical results.
    Only pass a cache when the weights cannot have changed between calls.

    frozen_threshold: when evaluating the test split, this must be the output
    of `pick_threshold_on_val(...)` called on the SAME encoder beforehand, so
    the reported correction/detection F1 use a threshold chosen without ever
    looking at test labels. When None (e.g. evaluating the val split itself,
    to pick that threshold), the threshold is swept on whatever split is
    passed in, which is only valid for a split other than "test".
    """
    if frozen_threshold is None and split == "test":
        warnings.warn(
            "evaluate(split='test') called without frozen_threshold: the "
            "correction/detection F1 threshold will be swept on the TEST "
            "split itself, which leaks test labels into model selection. "
            "Call pick_threshold_on_val(...) first and pass its result here.",
            RuntimeWarning,
        )
    split_obj = data.splits[split]
    query_rows = split_obj.indices
    query_texts = split_obj.dirty_texts

    q_instr = (
        data.per_record_instructions(query_rows,
                                     shuffled=(cfg.query_instruction == "record_shuffled"))
        if is_per_record(cfg.query_instruction) else get_instruction(cfg.query_instruction)
    )
    d_instr = get_instruction(cfg.doc_instruction)

    cache_key = f"{cfg.doc_instruction}|{cfg.serialization}|{cfg.pool_over_instruction}"
    if canonical_cache is not None and cache_key in canonical_cache:
        canonical_emb = canonical_cache[cache_key]
        if verbose:
            print(f"[eval:{split}] canonical index reused from cache "
                  f"({canonical_emb.shape[0]} records)", flush=True)
    else:
        if verbose:
            print(f"[eval:{split}] encoding {len(data.canonical_texts)} canonical "
                  f"records ...", flush=True)
        canonical_emb = encoder.encode(
            data.canonical_texts, d_instr, batch_size=cfg.eval_batch_size
        )
        if canonical_cache is not None:
            canonical_cache[cache_key] = canonical_emb

    if verbose:
        print(f"[eval:{split}] encoding {len(query_texts)} dirty records ...",
              flush=True)
    query_emb = encoder.encode(
        query_texts, q_instr, batch_size=cfg.eval_batch_size
    )

    # --- retrieval -------------------------------------------------------
    gold = query_rows.astype(np.int64)                 # row i of dirty <-> row i of clean
    ranks = M.rank_of_gold(query_emb, canonical_emb, gold)
    top_k = max(cfg.recall_at_k) if cfg.recall_at_k else 10
    scores, indices = M.search(query_emb, canonical_emb, top_k=top_k)
    top1_scores, top1_idx = scores[:, 0], indices[:, 0]

    retrieval = M.retrieval_metrics(ranks, cfg.recall_at_k)
    retrieval.update(M.duplicate_aware_retrieval_metrics(
        indices, data.canonical_texts, gold, cfg.recall_at_k
    ))
    diagnostics = M.embedding_diagnostics(
        query_emb, canonical_emb, gold, top1_scores, top1_idx
    )

    # --- cell-level, paper-comparable -----------------------------------
    if frozen_threshold is not None:
        # The threshold was chosen on val (pick_threshold_on_val); apply it
        # here, unseen, rather than re-selecting it on this split's labels.
        frozen = M.cell_level_metrics(
            data.clean_df, data.dirty_df, query_rows, top1_idx, top1_scores,
            data.fields, threshold=frozen_threshold,
        )
        best_correction = {
            "threshold": float(frozen_threshold),
            "correction_precision": frozen["correction"]["precision"],
            "correction_recall": frozen["correction"]["recall"],
            "correction_f1": frozen["correction"]["f1"],
        }
        best_detection = {
            "threshold": float(frozen_threshold),
            "detection_precision": frozen["detection"]["precision"],
            "detection_recall": frozen["detection"]["recall"],
            "detection_f1": frozen["detection"]["f1"],
        }
        # Still compute the full table for reference/plots, but it must never
        # be used to pick "the" threshold for this split.
        sweep = M.sweep_thresholds(
            data.clean_df, data.dirty_df, query_rows, top1_idx, top1_scores,
            data.fields, cfg.similarity_thresholds,
        )
        sweep["best_correction"] = best_correction
        sweep["best_detection"] = best_detection
    else:
        sweep = M.sweep_thresholds(
            data.clean_df, data.dirty_df, query_rows, top1_idx, top1_scores,
            data.fields, cfg.similarity_thresholds,
        )
    at_zero = M.cell_level_metrics(
        data.clean_df, data.dirty_df, query_rows, top1_idx, top1_scores,
        data.fields, threshold=0.0,
    )

    # --- breakdowns ------------------------------------------------------
    per_noise = M.per_noise_type_recall(
        data.noise_labels, query_rows, ranks, data.fields, NOISE_TYPES
    )
    per_bucket = M.per_corruption_bucket(
        data.noise_labels, query_rows, ranks, data.fields
    )

    results = {
        "split": split,
        "step": int(step),
        "n_queries": int(len(query_rows)),
        "n_canonical": int(len(data.canonical_texts)),
        "query_instruction": cfg.query_instruction,
        "doc_instruction": cfg.doc_instruction,
        "retrieval": retrieval,
        "diagnostics": diagnostics,
        "cell_no_abstention": at_zero,
        "best_correction": sweep["best_correction"],
        "best_detection": sweep["best_detection"],
        "per_noise_type": per_noise,
        "per_corruption_count": per_bucket,
        # Per-query rank/correctness, for pairing across variants in
        # significance tests (metrics.mcnemar_test needs the SAME test
        # queries in the SAME order for two variants - this is what makes
        # that pairing possible without re-reading CSVs). Not written to the
        # JSON metrics file (see save_results below) - only kept in the
        # in-memory dict this function returns.
        "ranks": ranks,
        "correct": ranks <= 1,
    }

    # --- TensorBoard -----------------------------------------------------
    if writer is not None:
        flat = {}
        flat.update(_to_scalar_dict(f"{split}/", retrieval))
        flat.update(_to_scalar_dict(f"{split}/diag/", diagnostics))
        flat.update(_to_scalar_dict(f"{split}/cell_t0/", at_zero))
        flat.update({
            f"{split}/best/correction_f1": sweep["best_correction"]["correction_f1"],
            f"{split}/best/correction_precision": sweep["best_correction"]["correction_precision"],
            f"{split}/best/correction_recall": sweep["best_correction"]["correction_recall"],
            f"{split}/best/correction_threshold": sweep["best_correction"]["threshold"],
            f"{split}/best/detection_f1": sweep["best_detection"]["detection_f1"],
        })
        for noise, vals in per_noise.items():
            flat[f"{split}/noise/{noise}/recall@1"] = vals["recall@1"]
        for bucket, vals in per_bucket.items():
            flat[f"{split}/corrupted_fields/{bucket}/recall@1"] = vals["recall@1"]

        with writer.as_default():
            for k, v in flat.items():
                tf.summary.scalar(k, v, step=step)
            writer.flush()

        if log_visuals:
            pos_sims = np.sum(query_emb * canonical_emb[gold], axis=1)
            neg_sims = np.where(top1_idx != gold, top1_scores, np.nan)
            n_viz = min(cfg.tsne_num_points, len(query_emb))
            V.log_all_visuals(
                writer=writer,
                step=step,
                dirty_emb=query_emb[:n_viz],
                clean_emb=canonical_emb[gold][:n_viz],
                positive_sims=pos_sims,
                negative_sims=neg_sims,
                ranks=ranks,
                per_noise=per_noise,
                threshold_table=sweep["table"],
                max_points=n_viz,
                seed=cfg.seed,
                suffix=f"  ({cfg.query_instruction}, {split})",
            )

        if cfg.log_projector:
            n_proj = min(cfg.projector_num_points, len(query_emb))
            noise_tags = [
                "+".join(data.row_noise_types(query_rows[i])) or "none"
                for i in range(n_proj)
            ]
            V.write_projector(
                log_dir=cfg.tb_dir,
                dirty_emb=query_emb[:n_proj],
                clean_emb=canonical_emb[gold][:n_proj],
                dirty_texts=[query_texts[i] for i in range(n_proj)],
                clean_texts=[data.canonical_texts[gold[i]] for i in range(n_proj)],
                noise_types=noise_tags,
                correct=(ranks[:n_proj] <= 1).tolist(),
            )

    # --- artefacts on disk ----------------------------------------------
    if save_results:
        os.makedirs(cfg.results_dir, exist_ok=True)
        with open(os.path.join(cfg.results_dir, f"{split}_metrics.json"),
                  "w", encoding="utf-8") as f:
            json_safe = {k: v for k, v in results.items()
                        if k not in ("ranks", "correct")}
            json.dump(json_safe, f, indent=2)
        sweep["table"].to_csv(
            os.path.join(cfg.results_dir, f"{split}_threshold_sweep.csv"),
            index=False,
        )
        pd.DataFrame({
            "query_row": query_rows,
            "rank": ranks,
            "top1_score": top1_scores,
            "top1_row": top1_idx,
            "correct": ranks <= 1,
            "noise_types": ["+".join(data.row_noise_types(r)) or "none"
                            for r in query_rows],
        }).to_csv(
            os.path.join(cfg.results_dir, f"{split}_per_query.csv"), index=False
        )
        V.dump_examples(
            path=os.path.join(cfg.results_dir, f"{split}_examples.txt"),
            dirty_texts=query_texts,
            clean_texts=[data.canonical_texts[g] for g in gold],
            retrieved_texts=[data.canonical_texts[i] for i in top1_idx],
            scores=top1_scores,
            ranks=ranks,
        )
        np.savez_compressed(
            os.path.join(cfg.results_dir, f"{split}_embeddings.npz"),
            query_emb=query_emb.astype(np.float16),
            query_rows=query_rows,
            ranks=ranks,
            top1_scores=top1_scores,
            top1_idx=top1_idx,
        )

    if verbose:
        best = sweep["best_correction"]
        print(f"[eval:{split}] R@1={retrieval['recall@1']:.4f}  "
              f"R@5={retrieval.get('recall@5', float('nan')):.4f}  "
              f"MRR={retrieval['mrr']:.4f}")
        print(f"[eval:{split}] correction F1={best['correction_f1']:.4f} "
              f"(P={best['correction_precision']:.4f} "
              f"R={best['correction_recall']:.4f} "
              f"@ tau={best['threshold']:.2f})   "
              f"[published baseline 0.29]")
        print(f"[eval:{split}] detection  F1={sweep['best_detection']['detection_f1']:.4f}")
        print(f"[eval:{split}] pos_sim={diagnostics['positive_similarity']:.4f}  "
              f"margin={diagnostics['separation_margin']:.4f}")

    return results


def main():
    import argparse
    from config import parse_overrides
    from data import load_dataset
    from model import build_encoder
    from train import restore_encoder

    ap = argparse.ArgumentParser(description="Evaluate a Clean Data Encoder run")
    ap.add_argument("--run", default=None,
                    help="run name to load config+checkpoint from")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--best", action="store_true",
                    help="restore the best checkpoint instead of the latest")
    ap.add_argument("overrides", nargs="*", help="key=value config overrides")
    args = ap.parse_args()

    if args.run:
        cfg_path = os.path.join(ExperimentConfig(run_name=args.run).run_dir,
                                "config.json")
        cfg = ExperimentConfig.load(cfg_path)
    else:
        cfg = ExperimentConfig()
    if args.overrides:
        cfg = parse_overrides(args.overrides, cfg)
    cfg.make_dirs()

    data = load_dataset(cfg)
    encoder = build_encoder(cfg, data)
    restore_encoder(encoder, cfg, prefer_best=args.best)

    writer = tf.summary.create_file_writer(os.path.join(cfg.tb_dir, "eval"))
    frozen_threshold = None
    if args.split == "test":
        frozen_threshold = pick_threshold_on_val(encoder, data, cfg)
        print(f"[eval] correction/detection threshold chosen on val: "
              f"{frozen_threshold:.2f}")
    evaluate(encoder, data, cfg, split=args.split, writer=writer, step=0,
             frozen_threshold=frozen_threshold)


if __name__ == "__main__":
    main()
