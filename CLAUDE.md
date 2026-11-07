# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research codebase testing whether instruction-conditioned embeddings can be
steered with an explicit *noise-description* instruction to pull a corrupted
address record's embedding onto its clean canonical twin — evaluated on the
AddressTable dataset from *Clean Me If You Can* (arXiv:2606.31983).

The encoder, **SCRUB** (Self-supervised Contrastive Record Understanding,
from-scratch), is built entirely from scratch: no pretrained backbone, no
borrowed weights, no HF `transformers` dependency. Every parameter — including
the word embeddings — starts random and is learned only from this project's
own training pairs plus the instruction bank text. This is a deliberate
project decision (an earlier version fine-tuned `hkunlp/instructor-base`; it
was replaced so no result here can be attributed to a pretrained model's prior
knowledge). Full writeup, architecture rationale, and results are in
[code/README.md](code/README.md) — read it before making changes; it is the
authoritative source on the hypothesis, metrics, and known caveats (notably
that a TF-IDF baseline still beats SCRUB as of the last recorded run).

Repo root layout:
- `code/` — all source (see below)
- `AddressTable/` — the dataset (from the *Clean Me If You Can* paper), sliced by
  corruption rate/locale under `AddressTable/experiments_datasets/`
- `Papers/` — reference PDFs for related work
- `results/` — exported result tables/artifacts from past runs
- `runs/` — created at runtime: per-run checkpoints, TensorBoard logs, vocab (gitignored)

## Commands

All commands run from `code/`. This project uses [uv](https://docs.astral.sh/uv/);
`run.bat` (Windows) / `run.sh` (Linux/macOS) wrap `uv run`. There is no
pretrained backbone for SCRUB itself, so plain TensorFlow is the only
dependency of substance for the main path — no `transformers`/`tf-keras`/
`torch`/`TF_USE_LEGACY_KERAS` chain to manage. Prefer the launcher scripts
over calling `python`/`uv run python` directly.

```bat
:: environment
uv sync

:: smoke test: does the pipeline run end to end on a tiny slice?
run.bat train.py --run smoke max_records=200 epochs=1 --eval-after

:: single run, then eval best checkpoint on test
run.bat train.py --run myrun query_instruction=noise_domain --eval-after

:: resume: same --run name continues from last checkpoint; --no-resume to restart
run.bat train.py --run myrun epochs=100 --eval-after

:: evaluate only (add --best to load the best checkpoint instead of latest)
run.bat evaluate.py --run myrun --best

:: the core experiment: sweep instruction variants with everything else fixed
run.bat run_ablation.py --dataset poi_library --train
run.bat run_ablation.py --dataset poi_library --train --variants none,noise_terse,noise_specific

:: lexical baseline (TF-IDF) — always compare against this; every result needs this row
run.bat baseline_lexical.py poi_library

:: TensorBoard (train/margin, val/diag/separation_margin, HPARAMS per variant, PROJECTOR)
run.bat -m tensorboard.main --logdir runs

:: pretrained-backbone side experiment (separate encoder, not SCRUB itself — see Architecture)
uv sync --extra pretrained
run.bat train.py --run pretrained_run encoder_type=pretrained batch_size=8 --eval-after
```

Any script accepts `key=value` overrides matching `config.ExperimentConfig` fields,
e.g. `dataset=de_020 batch_size=32 epochs=5 pool_over_instruction=true num_layers=6`.
Architecture toggles worth knowing: `use_rope`/`use_rmsnorm`/`use_swiglu` (modernized
Transformer block, meant to be turned on together), `early_stopping_patience`/
`early_stopping_min_delta` (0 disables early stopping), and `encoder_type=pretrained`
with `pretrained_model_name`/`pretrained_trainable` (routes to `model_pretrained.py`
instead of SCRUB; requires the `pretrained` extra and typically a smaller batch size
to avoid GPU OOM — 8 rather than 32).

Because nothing is pretrained, there is no zero-shot mode worth reporting — every
run needs training before its numbers mean anything.

Lint: `uv run ruff check .` (config in `pyproject.toml`; `select = ["E","F","W","I","UP","B"]`).
There is no automated test suite — validation is via `baseline_lexical.py` comparisons
and the metrics/visualizations a run produces.

GPU requires WSL2 on Windows (TensorFlow has no native-Windows GPU support after
2.10); pass `require_gpu=true` so a silent CPU fallback fails loudly instead of
silently running ~40x slower.

## Architecture

One `ExperimentConfig` dataclass (`config.py`) fully describes a run — dataset
slice, instruction variant, SCRUB's hyperparameters (`num_layers`,
`token_embed_dim`, `char_cnn_filters`, etc.) — and is serialized alongside run
outputs for reproducibility. Data flow:

1. **`data.py`** loads a CSV dataset slice (dirty + clean records, corruption
   tags), serializes each record to text (`keyed` or `natural` style), and
   produces train/val/test splits plus the always-full canonical index.
2. **`instructions.py`** — the instruction bank (`INSTRUCTION_BANK`), the
   experimental variable. `query_instruction` is applied to the dirty side
   only; `doc_instruction` (usually `none`) to the clean side, so the
   canonical index stays instruction-free.
3. **`tokenizer.py`** builds SCRUB's vocabulary from scratch — every word in
   the training split's text plus every word in the full instruction bank
   (kept regardless of frequency, since instruction words are rare in the
   address corpus but essential for a prompt to be readable at all).
   Case/punctuation-preserving (unlike `data.py`'s noise-labelling
   normalisation): that surface signal is what the char-CNN half of the
   encoder is meant to learn to handle. Saved to `runs/<name>/vocab.json` so a
   resumed run or `evaluate.py --run <name>` reloads the exact vocabulary a
   checkpoint was trained with, rather than rebuilding a different one.
4. **`model.py`** — SCRUB itself: a trainable word embedding + char-CNN hybrid
   token representation, feeding N pre-norm Transformer blocks
   (`tf.keras.layers.MultiHeadAttention`, no HF classes), then masked mean
   pooling. Instruction and content tokens are tokenized separately so content
   always keeps a `min_content_tokens` budget, and pooling masks the
   instruction positions out by default (`pool_over_instruction=true` is the
   ablation of that masking). `use_rope`/`use_rmsnorm`/`use_swiglu` swap in a
   modernized pre-norm block (RoPE attention, RMSNorm, SwiGLU FFN) in place of
   the baseline block; `build_encoder` in `model.py` dispatches on
   `encoder_type` to either this from-scratch path or `model_pretrained.py`
   (see below). `model_snn.py` holds an earlier, no-longer-primary encoder
   variant.
   - **`model_pretrained.py`** — a deliberate side experiment, *not* SCRUB: an
     otherwise-identical asymmetric-instruction-masking + contrastive setup,
     but fine-tuned from a small pretrained multilingual backbone (default
     `distilbert-base-multilingual-cased`, via the `transformers`/`tf-keras`
     compat shim under the `pretrained` extra) instead of SCRUB's from-scratch
     tokenizer/embedding/Transformer stack. It exists to separate what a
     pretrained language prior buys versus the from-scratch constraint and the
     noise-instruction effect itself — the results are reported side by side
     with SCRUB's, never merged into one number.
5. **`losses.py`** — bidirectional in-batch InfoNCE contrastive loss.
6. **`train.py`** — training loop with checkpoint/resume (`checkpoints/`
   rolling last 3 steps, `checkpoints_best/` overwritten only on val recall@1
   improvement). Relaunching the same `--run` name with a higher `epochs`
   continues training rather than restarting.
7. **`evaluate.py`** — evaluation driver: retrieval metrics (Recall@k, MRR,
   median rank) plus cell-level detection/correction metrics
   (precision/recall/F1, via a similarity-threshold sweep) computed in
   **`metrics.py`**, which also produces the corruption-type and
   #-corrupted-fields breakdowns.
8. **`visualize.py`** — TensorBoard Embedding Projector export, figures (pair
   displacement, similarity histograms, rank CDF, threshold sweep), qualitative
   dumps of best/worst examples.
9. **`run_ablation.py`** — sweeps instruction variants from `instructions.py`
   with everything else held fixed; the main experiment entry point.
10. **`baseline_lexical.py`** — TF-IDF / char-ngram baseline used as the
    sanity floor every result must beat. Note: this script runs its
    computation at import time (no `__main__` guard) — always invoke it as a
    script, never `import` it.

`DATASET_SLICES` in `config.py` maps short dataset keys (e.g. `full_020`,
`poi_library`, `de_048`, `russia_065`, `runtime_100k`) to paths under
`AddressTable/experiments_datasets/`; the trailing number is roughly the
fraction of corrupted cells, giving a difficulty sweep.

## Key caveats to carry into any change

- **TF-IDF is the sanity floor.** A char n-gram TF-IDF baseline currently
  beats SCRUB on every tested slice by a wide margin (see
  [code/README.md](code/README.md#results)). Any code change to the
  encoder/training loop should be checked against this baseline before its
  numbers are trusted.
- **Train margin turning positive happens well before val margin does.**
  Watch `train/margin` and `val/diag/separation_margin` in TensorBoard
  separately — a from-scratch Transformer needs many more steps than a
  fine-tuned pretrained one before either signal is meaningful.
- **The `none` vs. `noise_*` comparison in `run_ablation.py` is the
  load-bearing result**, not absolute accuracy — same encoder, only the
  instruction text differs.
