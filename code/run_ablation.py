"""
The core experiment: does naming the noise in the instruction actually help?

Sweeps instruction variants from control to hypothesis, holding everything else
fixed, and writes one comparison table:

    none               no instruction at all         (plain-embedding baseline)
    generic_retrieval  task instruction, no noise    (does ANY instruction help?)
    generic_noisy      vague messiness, no types     (is naming types necessary?)
    noise_terse        noise types, minimal wording  (paraphrase robustness)
    noise_specific     noise types named
    noise_domain       noise types + domain
    noise_verbose      noise types enumerated

The thesis holds only if the noise_* variants beat BOTH `none` and
`generic_retrieval`. If `generic_noisy` matches them, the model is responding to
"this is messy" rather than to the specific noise description, and the claim has
to be weakened accordingly. That negative result is worth reporting too.

Every variant becomes one TensorBoard HParams row, so the comparison is sortable
in the HPARAMS tab as well as in the CSV.

Usage
-----
    python run_ablation.py                          zero-shot, default slice
    python run_ablation.py --train                  fine-tune each variant
    python run_ablation.py --dataset poi_library    harder 64%-corruption slice
    python run_ablation.py --variants none,noise_domain
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

import metrics as M
from config import RUNS_ROOT, ExperimentConfig, parse_overrides
from data import load_dataset
from evaluate import evaluate, pick_threshold_on_val
from instructions import (ABLATION_QUERY_VARIANTS, PER_RECORD_ABLATION_VARIANTS,
                          describe)
from model import build_encoder
from train import restore_encoder, train


HPARAMS = [
    # Free-form string hparams are declared without a domain so a sweep can
    # introduce new instruction keys or models without editing this list.
    hp.HParam("query_instruction"),
    hp.HParam("doc_instruction"),
    hp.HParam("dataset"),
    hp.HParam("num_layers"),
    hp.HParam("pool_over_instruction", hp.Discrete([True, False])),
    hp.HParam("trained", hp.Discrete([True, False])),
]

TRACKED_METRICS = [
    hp.Metric("test/recall@1", display_name="Recall@1"),
    hp.Metric("test/recall@5", display_name="Recall@5"),
    hp.Metric("test/mrr", display_name="MRR"),
    hp.Metric("test/best/correction_f1", display_name="Correction F1"),
    hp.Metric("test/best/detection_f1", display_name="Detection F1"),
    hp.Metric("test/diag/separation_margin", display_name="Separation margin"),
]


def _hparam_dict(cfg: ExperimentConfig, trained: bool) -> Dict:
    return {
        "query_instruction": cfg.query_instruction,
        "doc_instruction": cfg.doc_instruction,
        "dataset": cfg.dataset,
        "num_layers": cfg.num_layers,
        "pool_over_instruction": bool(cfg.pool_over_instruction),
        "trained": bool(trained),
    }


def run_variant(base: ExperimentConfig,
                variant: str,
                do_train: bool,
                sweep_name: str,
                shared_data=None,
                shared_encoder=None,
                canonical_cache=None):
    """Returns (row: Dict of scalar metrics, correct: np.ndarray[bool] per
    test query) - see the return statement at the end of this function."""
    cfg = dataclasses.replace(
        base,
        query_instruction=variant,
        run_name=f"{sweep_name}/{variant}",
        do_train=do_train,
    )
    cfg.make_dirs()
    cfg.save()

    print("\n" + "=" * 76)
    print(f"VARIANT: {variant}")
    print(f"  {describe(variant)}")
    print("=" * 76)

    data = shared_data if shared_data is not None else load_dataset(cfg, verbose=False)

    if do_train:
        # Training mutates weights, so each trained variant needs its own
        # encoder instance, trained from its own random initialisation.
        encoder = build_encoder(cfg, data)
        train(cfg, data, encoder)
        restore_encoder(encoder, cfg, prefer_best=True)
    else:
        # Zero-shot: one frozen encoder can serve every variant, which is the
        # whole point - only the instruction text changes between rows.
        encoder = shared_encoder if shared_encoder is not None else build_encoder(cfg, data)

    # Select the correction/detection F1 threshold on val, never on test:
    # picking it on test would let the "best" operating point be fit to the
    # exact labels it is then scored against.
    frozen_threshold = pick_threshold_on_val(
        encoder, data, cfg, canonical_cache=canonical_cache
    )

    writer = tf.summary.create_file_writer(os.path.join(cfg.tb_dir, "test"))
    results = evaluate(encoder, data, cfg, split="test", writer=writer,
                       step=0, log_visuals=True, save_results=True,
                       canonical_cache=canonical_cache,
                       frozen_threshold=frozen_threshold)

    with writer.as_default():
        hp.hparams(_hparam_dict(cfg, trained=do_train),
                   trial_id=f"{sweep_name}/{variant}")
        tf.summary.scalar("test/recall@1", results["retrieval"]["recall@1"], step=0)
        tf.summary.scalar("test/recall@5",
                          results["retrieval"].get("recall@5", 0.0), step=0)
        tf.summary.scalar("test/mrr", results["retrieval"]["mrr"], step=0)
        tf.summary.scalar("test/best/correction_f1",
                          results["best_correction"]["correction_f1"], step=0)
        tf.summary.scalar("test/best/detection_f1",
                          results["best_detection"]["detection_f1"], step=0)
        tf.summary.scalar("test/diag/separation_margin",
                          results["diagnostics"]["separation_margin"], step=0)
        writer.flush()

    row = {
        "variant": variant,
        "instruction": describe(variant),
        "trained": do_train,
        "recall@1": results["retrieval"]["recall@1"],
        "recall@1_dup_aware": results["retrieval"].get("recall@1_dup_aware", float("nan")),
        "recall@5": results["retrieval"].get("recall@5", float("nan")),
        "recall@10": results["retrieval"].get("recall@10", float("nan")),
        "mrr": results["retrieval"]["mrr"],
        "median_rank": results["retrieval"]["median_rank"],
        "correction_f1": results["best_correction"]["correction_f1"],
        "correction_precision": results["best_correction"]["correction_precision"],
        "correction_recall": results["best_correction"]["correction_recall"],
        "correction_threshold": results["best_correction"]["threshold"],
        "detection_f1": results["best_detection"]["detection_f1"],
        "positive_similarity": results["diagnostics"]["positive_similarity"],
        "separation_margin": results["diagnostics"]["separation_margin"],
    }
    # Per-query correctness, kept alongside the row (not in the CSV table)
    # for the pairwise significance tests main() runs after every variant in
    # the sweep has been evaluated on the same test split.
    return row, results["correct"]


def main():
    ap = argparse.ArgumentParser(description="Instruction ablation sweep")
    ap.add_argument("--sweep-name", default=None,
                    help="folder under runs/ holding the sweep")
    ap.add_argument("--variants", default=None,
                    help="comma-separated instruction keys (default: the full set)")
    ap.add_argument("--per-record", action="store_true",
                    help="use PER_RECORD_ABLATION_VARIANTS (record_specific/"
                         "record_shuffled/record_nonsense, Priority-1 redesign) "
                         "instead of the fixed-string ABLATION_QUERY_VARIANTS")
    ap.add_argument("--seed", type=int, default=None,
                    help="override cfg.seed for this sweep (for multi-seed runs)")
    ap.add_argument("--train", action="store_true",
                    help="fine-tune each variant instead of zero-shot only")
    ap.add_argument("--dataset", default=None, help="dataset slice key")
    ap.add_argument("overrides", nargs="*", help="key=value config overrides")
    args = ap.parse_args()

    base = ExperimentConfig()
    if args.dataset:
        base.dataset = args.dataset
    if args.seed is not None:
        base.seed = args.seed
    if args.overrides:
        base = parse_overrides(args.overrides, base)

    default_variants = (PER_RECORD_ABLATION_VARIANTS if args.per_record
                        else ABLATION_QUERY_VARIANTS)
    variants = ([v.strip() for v in args.variants.split(",")]
                if args.variants else list(default_variants))
    sweep_name = args.sweep_name or (
        f"ablation_{base.dataset}_{'trained' if args.train else 'zeroshot'}"
        f"{'_perrecord' if args.per_record else ''}_seed{base.seed}"
    )
    sweep_dir = os.path.join(RUNS_ROOT, sweep_name)
    os.makedirs(sweep_dir, exist_ok=True)

    print(f"[sweep] {sweep_name}")
    print(f"[sweep] dataset  : {base.dataset}")
    print(f"[sweep] encoder  : SCRUB, from-scratch, {base.num_layers} layers, "
          f"{base.token_embed_dim}d token embed")
    print(f"[sweep] variants : {', '.join(variants)}")
    print(f"[sweep] mode     : {'fine-tuned' if args.train else 'zero-shot'}")

    # Register the hparam schema once at the sweep root so the HPARAMS tab
    # groups every variant into one comparable table.
    with tf.summary.create_file_writer(sweep_dir).as_default():
        hp.hparams_config(hparams=HPARAMS, metrics=TRACKED_METRICS)

    data = load_dataset(base)
    shared_encoder = None if args.train else build_encoder(base, data)
    # Safe only because a zero-shot sweep never mutates the encoder weights;
    # each trained variant builds its own encoder, so it gets no cache.
    canonical_cache = None if args.train else {}

    rows: List[Dict] = []
    correct_by_variant: Dict[str, "np.ndarray"] = {}
    started = time.time()
    for variant in variants:
        row, correct = run_variant(base, variant, args.train, sweep_name,
                                   shared_data=data,
                                   shared_encoder=shared_encoder,
                                   canonical_cache=canonical_cache)
        rows.append(row)
        correct_by_variant[variant] = correct
        pd.DataFrame(rows).to_csv(
            os.path.join(sweep_dir, "comparison.csv"), index=False
        )

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(sweep_dir, "comparison.csv"), index=False)
    with open(os.path.join(sweep_dir, "comparison.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    # ---- pairwise significance (Priority 3: "0.2-point differences with
    # no uncertainty reported") -------------------------------------------
    # Every variant above shares the SAME test split (one `data` object, one
    # `base.seed`), so McNemar's paired test on per-query correct/incorrect
    # is the right tool - these are not independent samples. The pairs
    # compared are chosen to answer the ablation's own questions: does the
    # best-looking variant actually beat the controls it needs to beat.
    sig_rows = []
    if len(correct_by_variant) > 1:
        # Compare the best-Recall@1 variant against EVERY other variant in
        # the sweep, not a fixed control subset - otherwise, whenever the
        # best variant happens to BE a control (as "none" is here), the
        # hypothesis variant (record_specific) never gets compared against
        # it at all, which is the one comparison the ablation exists to
        # make. (Fixed after an earlier run of this sweep silently omitted
        # exactly that comparison - see the seed-42 per-record sweep.)
        best_variant = table.loc[table["recall@1"].idxmax(), "variant"]
        for other in correct_by_variant:
            if other == best_variant:
                continue
            test = M.mcnemar_test(correct_by_variant[best_variant],
                                  correct_by_variant[other])
            test["variant_a"] = best_variant
            test["variant_b"] = other
            sig_rows.append(test)
        if sig_rows:
            sig_table = pd.DataFrame(sig_rows)
            sig_table.to_csv(os.path.join(sweep_dir, "significance.csv"),
                             index=False)
            print(f"\n[significance] McNemar's test, best variant "
                  f"('{best_variant}') vs. each control, on the shared test "
                  f"split ({len(sig_rows)} comparisons):")
            print(sig_table[["variant_a", "variant_b", "n_discordant",
                             "chi2", "p_value"]].to_string(
                index=False, float_format=lambda v: f"{v:.4f}"))
            print(f"[significance] table written to "
                  f"{os.path.join(sweep_dir, 'significance.csv')}")

    # ---- summary ------------------------------------------------------
    cols = ["variant", "recall@1", "recall@5", "mrr",
            "correction_f1", "detection_f1", "separation_margin"]
    print("\n" + "=" * 96)
    print(f"ABLATION RESULTS  ({base.dataset}, "
          f"{'fine-tuned' if args.train else 'zero-shot'}, "
          f"{time.time() - started:.0f}s)")
    print("=" * 96)
    print(table[cols].to_string(index=False,
                                float_format=lambda v: f"{v:.4f}"))

    if "none" in table["variant"].values:
        base_r1 = float(table.loc[table["variant"] == "none", "recall@1"].iloc[0])
        base_f1 = float(table.loc[table["variant"] == "none", "correction_f1"].iloc[0])
        best = table.loc[table["recall@1"].idxmax()]
        print(f"\nno-instruction baseline : recall@1={base_r1:.4f}  "
              f"correction_f1={base_f1:.4f}")
        print(f"best variant            : {best['variant']}  "
              f"recall@1={best['recall@1']:.4f}  "
              f"correction_f1={best['correction_f1']:.4f}")
        print(f"delta over no-instruction: recall@1 "
              f"{best['recall@1'] - base_r1:+.4f},  correction_f1 "
              f"{best['correction_f1'] - base_f1:+.4f}")
        print(f"published baseline to beat: correction F1 0.29 "
              f"(DataWig / Glock's, AddressTable paper)")

    print(f"\nComparison table: {os.path.join(sweep_dir, 'comparison.csv')}")
    print(f"TensorBoard     : tensorboard --logdir \"{sweep_dir}\"")


if __name__ == "__main__":
    main()
