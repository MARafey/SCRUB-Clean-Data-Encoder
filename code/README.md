# Clean Data Encoder

Noise-instruction-conditioned embeddings for matching corrupted address records
against a clean canonical database, evaluated on the AddressTable dataset from
*Clean Me If You Can* (arXiv:2606.31983).

**SCRUB** (Self-supervised Contrastive Record Understanding, from-scratch) is
the name of this project's encoder architecture (`model.py` / `tokenizer.py`)
— used from here on to distinguish results produced by it from the TF-IDF
baseline or from any published model referenced for comparison.

**The hypothesis.** Published instruction-conditioned embedding models
(INSTRUCTOR, E5-Mistral) steer their embedding space with a prepended
*task*-descriptive instruction ("Represent the Wikipedia question for
retrieving supporting documents") — never a *noise*-descriptive one. This
codebase tests whether describing the corruption explicitly, applied only to
the dirty side while the canonical side stays instruction-free, pulls a
corrupted record's embedding onto its clean twin.

**The encoder is built from scratch.** There is no pretrained backbone here —
no INSTRUCTOR, no T5, no borrowed weights of any kind. Every parameter,
including the word embeddings, starts random and is learned only from this
project's own contrastive training pairs plus the instruction bank text (see
"What's borrowed vs. what's new" below). That has a real consequence for the
instruction mechanism: a prompt like *"ignore capitalisation and abbreviation
noise"* only works if the model learns, purely from co-occurring with those
noise patterns during training, what those specific words predict — there is
no pretrained language understanding underneath it. Report results with that
distinction stated plainly; it is a narrower claim than "the model understands
the instruction."

Everything runs in TensorFlow, logs to TensorBoard, and checkpoints so a run can
be stopped and resumed.

---

## Dataset

AddressTable, from *Clean Me If You Can: A Large Collection of Real-World
Addresses for Data Cleaning Benchmarking* (Ahmadi et al.).

- Paper: <https://arxiv.org/abs/2606.31983>
- Data and extraction code: <https://github.com/D2IP-TUB/Clean-Me-If-You-Can>

The data (~16 GB) is not committed to this repository. Download it and place the
slices under `AddressTable/experiments_datasets/` at the repo root; the short
keys used by `dataset=...` (`poi_library`, `full_020`, `de_048`, `russia_065`,
`runtime_100k`, ...) are mapped to paths in `config.DATASET_SLICES`.

---

## What's borrowed vs. what's new

Being explicit about this split matters: it's the difference between a result
that says something about the noise-instruction idea and a result that is
really just "how good is someone else's pretrained model."

| | Borrowed | New (this project) |
|---|---|---|
| Encoder architecture | the general *technique* — Transformer self-attention blocks (`tf.keras.layers.MultiHeadAttention`), char-CNN sub-word features, mean pooling: all standard, published building blocks | **SCRUB** (`model.py`), the specific architecture assembled from them: word-embedding + char-CNN hybrid token representation, N pre-norm Transformer blocks, asymmetric instruction-masked pooling |
| Weights | nothing — no checkpoint, no external model, no pretraining corpus | every trained parameter, including the word-embedding table, starts random-init and is learned only from this project's own data |
| Vocabulary | nothing | built from scratch (`tokenizer.py`) from this project's own training-split text plus the instruction bank — no external tokenizer, no subword model trained on outside data |
| **Instruction content** | the general idea of prepending a natural-language instruction (INSTRUCTOR, Su et al. 2022) | the noise-description instruction bank in `instructions.py` (`noise_terse` through `noise_verbose`), grounded in this dataset's actual corruption taxonomy, and — because the encoder is untrained — a model that has to *learn what those specific words mean* from the training signal, rather than already knowing English |
| Application pattern | — | **asymmetric**: instruction applied to the dirty/query side only, canonical/document side stays instruction-free |
| Noise characterisation | none | the cell-level noise taxonomy and classifier in `data.py` (`classify_cell`, `profile_rows`) that drives the per-noise-type and per-corruption-count breakdowns in `metrics.py` |
| Training objective | in-batch contrastive InfoNCE is a standard technique | bidirectional in-batch InfoNCE (`losses.py`) applied specifically to noisy/clean address pairs with the instruction attached |
| Evaluation protocol | retrieval metrics (Recall@k, MRR) are standard | retrieval metrics *and* the cell-level detection/correction protocol mapped onto retrieval (`metrics.py`), so results are comparable to both the embedding literature and the data-cleaning literature |
| Sanity baseline | TF-IDF itself is not new | run on identical splits (`baseline_lexical.py`), required before any result from the new encoder is treated as meaningful |

**What a from-scratch encoder does and doesn't buy.** A model trained only on
this project's own 3k-10k-record slices, with no pretraining, will not have
general language understanding — that is real and worth stating rather than
downplaying. What it does buy is independence: no result here can be
attributed to a pretrained model's prior knowledge, so any effect the noise
instruction has is demonstrably coming from this project's own training
signal, not from something INSTRUCTOR or E5 already knew.

---

## Setup

This project uses [uv](https://docs.astral.sh/uv/). Install it once:

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"     # Windows
```
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh                # Linux / macOS
```

Then nothing else — `run.bat` / `run.sh` call `uv run`, which creates the
virtualenv from `pyproject.toml` + `uv.lock` on first use and keeps it in sync
after that:

```
run.bat run_ablation.py --dataset poi_library
```

To materialise the environment ahead of time, or to work in it directly:

```bash
uv sync                        # create .venv with everything
uv run python train.py --run x # run anything inside it
uv add <package>               # add a dependency (updates pyproject + lock)
```

`requirements.txt` is exported from the lock as a pip fallback; `pyproject.toml`
and `uv.lock` are the source of truth.

There is no pretrained backbone to load, so there is no `transformers` /
`tf-keras` / `TF_USE_LEGACY_KERAS` / `torch` / `sentencepiece` chain to manage
— plain TensorFlow is the only dependency of substance (`scikit-learn` and
`matplotlib` are only for `baseline_lexical.py` and the TensorBoard figures).

### GPU

```bash
uv sync --extra gpu
run.sh run_ablation.py --dataset runtime_100k require_gpu=true mixed_precision=true
```

`require_gpu=true` turns "no GPU visible" into a hard error. Without it a run
that silently falls back to CPU looks identical in the logs and just takes ~40x
longer. `mixed_precision=true` is a large speedup on GPU and a slowdown on CPU,
so set it only with a GPU. Every run prints what it is using:

```
[device] GPU 0: NVIDIA GeForce RTX 4070, compute capability 8.9
[device] using 1 GPU(s), memory_growth=True
```

**On Windows this needs WSL2.** TensorFlow dropped native Windows GPU support
after 2.10, so any `tensorflow >= 2.11` on native Windows is CPU-only no matter
what is installed — the `gpu` extra will install and still not use the card.
`wsl --install`, then clone the project inside the WSL filesystem (not
`/mnt/c/...`, which is slow) and `uv sync --extra gpu` there. On native Linux
it works directly; check `nvidia-smi` first.

With a GPU, the sweeps below drop from minutes to seconds, which is what makes
the 100k-record index in "Making the experiment informative" practical.

---

## Quick start

Windows uses `run.bat`, Linux/macOS `./run.sh` — identical arguments.

```bat
:: 1. Smoke test: does the pipeline run end to end on a tiny slice?
run.bat train.py --run smoke max_records=200 epochs=1 --eval-after

:: 2. Fine-tune each instruction variant and compare (the core experiment)
run.bat run_ablation.py --dataset full_020 --train

:: 3. Single run, then evaluate the best checkpoint on test
run.bat train.py --run myrun query_instruction=noise_domain --eval-after

:: 4. Resume: rerun the same --run name, it continues from the last checkpoint
run.bat train.py --run myrun epochs=6 --eval-after

:: 5. Watch everything
run.bat -m tensorboard.main --logdir runs
```

Because nothing is pretrained, there is no zero-shot mode worth reporting —
every run needs `--train` (or `train.py`'s default `do_train=true`) before its
numbers mean anything; an untrained random-init encoder is not a baseline, it's
noise.

---

## The experiment

`run_ablation.py` sweeps instruction variants with everything else held fixed —
this is the "one arm without a prompt, one arm with a prompt describing how the
data is dirty" comparison:

| variant | what it tests |
|---|---|
| `none` | no instruction — the plain-embedding baseline |
| `generic_retrieval` | task instruction, noise never mentioned — does *any* instruction help? |
| `generic_noisy` | "this record is messy", no noise types named — is naming them necessary? |
| `noise_terse` | noise types named, minimal wording — paraphrase robustness |
| `noise_specific` | noise types named |
| `noise_domain` | noise types + domain grounding |
| `noise_verbose` | every measured corruption enumerated with examples |

**The thesis holds only if the `noise_*` variants beat both `none` and
`generic_retrieval`.** If `generic_noisy` matches them, the model is responding
to "this is messy" rather than to the specific description, and the claim has to
be weakened. That negative result is worth reporting.

Edit or add instructions in `instructions.py`; the sweep picks them up, and the
vocabulary in `tokenizer.py` is rebuilt from the full instruction bank the next
time a run needs a fresh vocab.

---

## Metrics

Two families, because the literature reports two different things.

**Retrieval** (INSTRUCTOR / E5 / Sudowoodo / fuzzylink style): Recall@1/5/10,
MRR, median rank. Given the dirty record, does its clean twin come back on top?

**Cell-level detection and correction** (Clean Me If You Can, Raha/Baran,
HoloClean, DataWig): Precision / Recall / F1 over individual cells. Retrieval is
mapped onto this protocol by reading the top-1 canonical record's field values
as the proposed repair — a cell is *flagged* when the retrieved value differs
from the dirty value, and *repaired correctly* when it equals the clean value.
A similarity threshold lets the system abstain, which is what makes precision
and recall trade off; `sweep_thresholds` reports the best-F1 operating point.

**Diagnostics**: mean positive-pair similarity, hardest-negative similarity, and
the margin between them. The margin moves before accuracy does, so it is the
early signal that an instruction is changing the geometry at all.

**Breakdowns**: Recall@1 per corruption type (`emptied`, `abbreviation`,
`code_substitution`, `case`, `whitespace`, `punctuation`, `token_drop`,
`value_error`) and by how many fields were corrupted. This is what turns one F1
number into an argument about *which* noise the instruction helps with.

### A caveat about comparing to the published 0.29

The AddressTable paper's best correction F1 is ~0.29, and `visualize.py` draws
that line on the threshold sweep. **The comparison is not apples-to-apples and
should not be reported as if it were.** The baselines in that paper repair
values without access to a database containing the right answer. This setup
retrieves against a canonical index that *does* contain the gold record, which
is a fundamentally easier task, independent of how good the encoder is.

That is not a bug: matching noisy input against a clean canonical database is
exactly the use case here. But the honest framing is "record matching against a
canonical index", and the 0.29 line is context for the dataset's difficulty, not
a baseline this beats. The defensible comparisons are the ablation rows against
each other (same encoder, only the instruction changes) and against the TF-IDF
baseline run through this same pipeline.

---

## Read this before interpreting any accuracy number

Before the encoder rewrite, a TF-IDF bag-of-words baseline beat every
INSTRUCTOR configuration tested on this dataset, in under a second, with no
neural model at all — see `baseline_lexical.py`. **That baseline requirement
did not go away just because the encoder changed.** Reproduce it fresh for
whatever slice you're about to report:

```bat
run.bat baseline_lexical.py full_020
```

**The task is lexically solvable in several settings, so it cannot by itself
demonstrate the value of semantic embeddings there.** The largest noise class
(`emptied`) deletes fields rather than corrupting them, so surviving fields keep
their exact surface form; abbreviation and case changes preserve character
n-grams; and with eight fields, any one uncorrupted field is close to a primary
key against a 10k index. Every result table needs a TF-IDF row from the current
run of `baseline_lexical.py` next to it — anything that does not beat
bag-of-words is not a result. The cross-lingual `de_*` / `russia_*` slices are
the most promising place to look for a setting where lexical matching actually
fails (transliterated/translated variants destroy token overlap while
preserving meaning), and are still untested.

---

## Results

### SCRUB v1 (4 layers, 128d token embed, single-kernel char-CNN)

**Converged run**, `poi_library`, full data (1,656 records, no `max_records`
cap), 100 epochs (7,200 steps per variant, ~2.5h on CPU), 249 test queries —
`run_ablation.py --dataset poi_library --train --variants
none,noise_terse,noise_specific epochs=100`:

| variant | Recall@1 | Recall@5 | correction F1 | test margin |
|---|---|---|---|---|
| **TF-IDF char 3-5gram** | **0.9799** | 1.0000 | **0.9920** | n/a |
| TF-IDF word unigram | 0.9679 | 0.9799 | 0.9851 | n/a |
| SCRUB v1 `none` | **0.7430** | 0.8916 | 0.8601 | **+0.0053** |
| SCRUB v1 `noise_terse` | 0.7189 | 0.8996 | **0.8707** | -0.0002 |
| SCRUB v1 `noise_specific` | 0.6426 | 0.7992 | 0.8075 | -0.0149 |

Fully converged (margins plateaued, loss flattened before 100 epochs). `none`
beats both instruction variants on Recall@1 and margin, matching every earlier
result in this project (zero-shot INSTRUCTOR included). `noise_terse` had the
best correction F1 despite trailing on Recall@1 — a mixed signal, not a clear
win.

### SCRUB v2 (6 layers, 256d token embed, multi-kernel char-CNN)

Same slice and protocol, larger architecture (see "What's borrowed vs. what's
new"), 100 epochs, ~4.5h on CPU:

| variant | Recall@1 | Recall@5 | correction F1 | test margin |
|---|---|---|---|---|
| **TF-IDF char 3-5gram** | **0.9799** | 1.0000 | **0.9920** | n/a |
| SCRUB v2 `none` | 0.7912 | 0.8996 | 0.8937 | 0.0383 |
| **SCRUB v2 `noise_terse`** | **0.7952** | **0.9197** | **0.9005** | 0.0321 |
| SCRUB v2 `noise_specific` | 0.7912 | 0.8956 | 0.8872 | 0.0218 |

Three things worth separating:

1. **TF-IDF still wins by a wide margin** (0.98 vs. 0.79 Recall@1) even with
   roughly 4x the parameters and full convergence. This is the headline that
   survives both architecture versions: SCRUB losing to bag-of-words is not a
   capacity or training-time artifact.
2. **More capacity closed real ground against TF-IDF.** v2's `none` moved from
   0.7430 to 0.7912 Recall@1 (correction F1 0.8601 -> 0.8937) purely from a
   bigger encoder, same data, same protocol.
3. **`noise_terse` beat `none` for the first time in this project** — on both
   Recall@1 (+0.0040) and correction F1 (+0.0069), not just one metric. The
   margin is small and this is a single run, so treat it as a real but modest
   reversal worth reproducing (different seed) rather than a settled result.
   `noise_specific` did not replicate the win: it tied `none` on Recall@1
   while trailing on correction F1 and margin, so the effect is not uniform
   across instruction variants.

Honest current conclusion: **TF-IDF remains undefeated across both SCRUB
architectures**, but the noise-instruction picture is no longer uniformly
negative the way it was under zero-shot INSTRUCTOR and SCRUB v1 — a bigger,
fully-converged encoder produced the first (modest, single-run) case of an
instruction variant winning outright. The open question flagged before the
encoder rewrite still stands: whether a setting exists where lexical matching
actually fails (cross-lingual `de_*` / `russia_*`), which would be a fairer
test of the mechanism than a slice TF-IDF already nearly solves.

### Making the experiment informative

In priority order:

1. **Test a slice where TF-IDF doesn't already win.** This is now the most
   likely reason to see nothing: `poi_library` may still be too lexically easy
   for any encoder, regardless of size. Try `de_*` / `russia_*` next.
2. **Reproduce the `noise_terse` win on v2** with a different seed (and
   ideally the full 7-variant sweep) before treating it as evidence the
   mechanism works — one run, a small margin.
3. **Scale the index if the small slices turn out to be near-ceiling once
   SCRUB is actually converged.** `runtime_10k` / `runtime_100k` are in the
   dataset for this. Needs a GPU to be practical at 100k+.
4. **Keep instructions terse as a first move; then test whether length
   penalizes SCRUB the way it penalized INSTRUCTOR.** That finding may not
   transfer: INSTRUCTOR's degradation with instruction length was plausibly a
   pretrained-model-specific failure (long unfamiliar text perturbing an
   otherwise-competent representation via self-attention). SCRUB has no prior
   competence to perturb, so the same experiment here is testing something
   different and shouldn't be assumed to replicate.
5. **Track the margin, not just accuracy.** It responds earlier than Recall@1
   and it is what would show a mechanism working before it shows up as hits.
6. **Watch training stability, and beware the LR-schedule reset on resume.**
   A from-scratch Transformer is more sensitive to learning rate / warmup than
   a fine-tuned pretrained one — if loss spikes or margin stays flat near
   zero, check `train/grad_norm` in TensorBoard before concluding the
   instruction doesn't work. Resuming a run with a larger `epochs` value
   reuses the original LR schedule length (`runs/<name>/lr_schedule.json`);
   if that file is deleted, the next resume reopens a fresh, larger LR at the
   current step, which can cause a real (not cosmetic) training regression.

### SCRUB-SNN: spiking-neuron variant (exploratory)

`encoder_type=snn` (`model_snn.py`) swaps the Transformer's self-attention +
FFN blocks for a stack of leaky integrate-and-fire (LIF) spiking neuron
layers, trained with a surrogate-gradient estimator since the spike function
itself is non-differentiable. This changes two things relative to the
Transformer encoder at once (no attention, and a spiking rather than
real-valued nonlinearity), so treat any result as "does this different
architecture family do better or worse," not as an isolated test of spiking
neurons holding everything else equal.

```bat
run.bat run_ablation.py --dataset poi_library --train encoder_type=snn ^
    --variants none,noise_terse,noise_specific epochs=100
```

**SCRUB-SNN v1** (fixed LIF hyperparameters: `snn_decay=0.9`,
`snn_threshold=1.0`), `poi_library`, same protocol as SCRUB v1/v2 above:

| variant | Recall@1 | Recall@5 | correction F1 | test margin |
|---|---|---|---|---|
| **TF-IDF char 3-5gram** | **0.9799** | 1.0000 | **0.9920** | n/a |
| SCRUB-SNN v1 `none` | **0.6345** | 0.8313 | **0.8109** | **+0.0156** |
| SCRUB-SNN v1 `noise_terse` | 0.5904 | 0.8353 | 0.7916 | -0.0197 |
| SCRUB-SNN v1 `noise_specific` | 0.5743 | 0.8153 | 0.7697 | -0.0190 |

Clean, monotonic ordering — `none` beats both instruction variants on every
metric, matching the direction of every result in this project except SCRUB
v2's `noise_terse` exception. More importantly: **SCRUB-SNN v1 loses to both
Transformer versions** (v1: 0.7430, v2: 0.7912 Recall@1 for `none`), not just
to TF-IDF. Diagnosis (see conversation / commit history for the fuller
writeup): the block stack has no attention, so there is no content-based
mixing across token positions — only a leaky, per-channel memory decaying at a
fixed rate (`decay=0.9`, ~35% signal left after 10 tokens) — which is
plausibly too fast for `poi_library`'s longer records and address fields that
depend on each other (e.g. matching a name to a city).

**SCRUB-SNN v2** (learnable per-channel `decay`/`threshold`, initialised from
`decay=0.97` — a direct response to the v1 diagnosis above — via
`snn_learnable_dynamics=true`):

```bat
run.bat run_ablation.py --dataset poi_library --train encoder_type=snn ^
    snn_learnable_dynamics=true snn_decay=0.97 ^
    --variants none,noise_terse,noise_specific epochs=100
```

| variant | Recall@1 | Recall@5 | correction F1 | test margin |
|---|---|---|---|---|
| **TF-IDF char 3-5gram** | **0.9799** | 1.0000 | **0.9920** | n/a |
| SCRUB-SNN v2 `none` | **0.6546** | 0.8554 | 0.8148 | **+0.0097** |
| SCRUB-SNN v2 `noise_terse` | 0.6426 | 0.8273 | **0.8220** | -0.0124 |
| SCRUB-SNN v2 `noise_specific` | 0.5703 | 0.8353 | 0.7741 | -0.0149 |

**The learnable-dynamics fix worked**: v2 beats v1 on every metric for `none`
(0.6546 vs. 0.6345 Recall@1, 0.8148 vs. 0.8109 correction F1), confirming the
diagnosis that the fixed `decay=0.9` was losing context too fast. `noise_terse`
again edges out `none` on correction F1 only (0.8220 vs. 0.8148) while still
trailing on Recall@1 — the same "mixed, not a clean win" pattern seen in SCRUB
v1, and weaker than v2 Transformer's `noise_terse`, which won on *both*
metrics.

### Full comparison across every approach run in this project

`poi_library`, `none` variant (the load-bearing comparison point across
architectures) unless noted:

| approach | Recall@1 | correction F1 | best variant |
|---|---|---|---|
| **TF-IDF char 3-5gram** | **0.9799** | **0.9920** | — (not instruction-conditioned) |
| SCRUB v2 (Transformer, 6L/256d, multi-kernel char-CNN) | 0.7912 | 0.8937 | `noise_terse` wins on Recall@1 *and* correction F1 |
| SCRUB v1 (Transformer, 4L/128d, single-kernel char-CNN) | 0.7430 | 0.8601 | `noise_terse` wins on correction F1 only |
| SCRUB-SNN v2 (spiking LIF, learnable dynamics) | 0.6546 | 0.8148 | `noise_terse` wins on correction F1 only |
| SCRUB-SNN v1 (spiking LIF, fixed dynamics) | 0.6345 | 0.8109 | `none` wins on every metric |

Two independent conclusions survive every architecture tried so far:

1. **TF-IDF is undefeated**, by a wide margin, regardless of encoder family or
   size. This is the headline result of the whole project.
2. **Attention matters more than neuron model.** Both spiking variants trail
   both Transformer variants by a large margin (SCRUB-SNN v2's 0.6546 vs.
   SCRUB v1's 0.7430, despite SNN v2 having *more* parameters than SCRUB v1);
   the fixed-vs-learnable LIF dynamics fix closed real ground within the SNN
   family but nowhere near closed the gap to attention-based SCRUB. The
   diagnosis in the SCRUB-SNN v1 section above (no content-based mixing across
   token positions) looks like the dominant factor, not the spiking mechanism
   itself.

### Cross-lingual test: `russia_065` (in progress)

This directly follows up on the open question flagged above ("whether a
setting exists where lexical matching actually fails: cross-lingual `de_*` /
`russia_*`"). Training was started on `russia_065` (65%-corrupted, Russian/
Cyrillic-locale slice; `none` variant only, checkpoints through
`runs/ablation_russia_065_trained/none`) — a partial run (~800 steps, single
variant, not yet converged, not a full multi-variant sweep) — alongside a
completed `baseline_lexical.py` run on the same slice:

| approach | Recall@1 | Recall@5 | MRR | correction F1 |
|---|---|---|---|---|
| **TF-IDF char 3-5gram** | **0.7827** | 0.8927 | **0.8341** | **0.9080** |
| TF-IDF word unigram | 0.7747 | **0.9267** | 0.8404 | 0.9017 |
| SCRUB `none` (val, step 800, partial run) | 0.6000 | 0.7680 | 0.6732 | 0.7979 |

**Correction, contradicting the "TF-IDF fails cross-lingually" hypothesis
floated earlier:** as actually measured, TF-IDF is still ahead on `russia_065`
too (0.78 vs. 0.60 Recall@1) — it is *not* the slice where lexical matching
breaks down. Two caveats keep this from being a final verdict: the SCRUB
number is a val-set snapshot from an unconverged, single-variant run (not a
test-set `evaluate.py --best` result), and TF-IDF's own margin here (0.78) is
narrower than on `poi_library` (0.98), so the gap is closing even if it hasn't
flipped. The open question from before still stands unresolved — a slice
where TF-IDF actually loses has not yet been found.

**Next step (planned): full multilingual run.** Finish training `russia_065`
to convergence across the full variant sweep (`run_ablation.py --dataset
russia_065 --train`), evaluate with `evaluate.py --best` on the test split for
a like-for-like comparison with the TF-IDF numbers above, then extend to the
full multilingual dataset (not a single locale) to see whether a larger,
more varied non-English set is what it takes to reverse this gap, rather than
one locale slice.

---

## Checkpointing and resume

```
runs/<name>/checkpoints/        rolling, last 3 steps, for resuming
runs/<name>/checkpoints_best/   overwritten only when val recall@1 improves
runs/<name>/vocab.json          the vocabulary this run's checkpoints were trained with
runs/<name>/lr_schedule.json    the LR schedule length this run's checkpoints were trained under
```

Optimiser slots, global step, epoch and best-score-so-far all live inside the
checkpoint, so relaunching the same `--run` name continues from exactly where it
stopped. `--no-resume` starts fresh. `evaluate.py --best` loads the best
checkpoint rather than the latest. The vocabulary is data-dependent (built from
the training split + instruction bank), so it is saved once per run directory
and reloaded rather than rebuilt on every resume/evaluate — reusing a run's
`vocab.json` is what keeps a restored checkpoint's word-embedding ids meaningful.
The LR schedule length is similarly fixed on first training and reused on every
resume, deliberately decoupled from `cfg.epochs` — see the note in "Making the
experiment informative" above.

---

## Visual output

In TensorBoard:

- **PROJECTOR tab** — dirty and clean vectors in one space, coloured by `role`.
  Colour by `noise` to see which corruption types cluster away from their twins;
  search a `pair_id` to jump to one record and its corrupted version.
- **IMAGES tab** — pair-displacement scatter (a line per pair; long lines are
  failures), similarity-separation histograms, rank CDF, Recall@1 by corruption
  type, and the threshold sweep with the 0.29 line drawn on it.
- **SCALARS** — loss, margin, in-batch accuracy, learning rate, grad norm, and
  every eval metric over training.
- **HPARAMS** — one row per instruction variant, sortable by any metric.

On disk, per run under `results/`: `test_metrics.json`, `test_per_query.csv`,
`test_threshold_sweep.csv`, `test_embeddings.npz`, and `test_examples.txt` — the
worst failures and best hits written out with their dirty / gold / retrieved
text, which is the file to read when writing the failure analysis.

---

## Files

| file | role |
|---|---|
| `config.py` | every knob; one `ExperimentConfig` fully describes a run |
| `tokenizer.py` | from-scratch vocabulary + tokenizer, built only from this project's data |
| `instructions.py` | the instruction bank — the experimental variable |
| `data.py` | CSV loading, corruption tagging, serialisation, splits |
| `model.py` | the from-scratch Transformer encoder (`encoder_type=transformer`, default): word+char-CNN token representation, self-attention blocks, instruction-masked mean pooling |
| `model_snn.py` | the spiking-neuron variant (`encoder_type=snn`): same word+char-CNN front end, but LIF spiking blocks instead of self-attention; see its module docstring for why it's a different architecture family, not a controlled ablation |
| `losses.py` | bidirectional in-batch InfoNCE |
| `metrics.py` | retrieval + cell-level detection/correction + breakdowns |
| `visualize.py` | projector export, figures, qualitative dumps |
| `train.py` | training loop, checkpoints, resume |
| `evaluate.py` | evaluation driver |
| `run_ablation.py` | the instruction sweep |
| `baseline_lexical.py` | the TF-IDF sanity baseline every result must beat |

Config overrides are `key=value` arguments on any script:

```bat
run.bat train.py --run x dataset=de_020 num_layers=6 token_embed_dim=256 ^
    batch_size=32 epochs=5 pool_over_instruction=true max_records=2000
```

---

## Implementation notes worth knowing

**Instruction-masked pooling.** The encoder pools over the *content* tokens and
masks the instruction out. Without that, a long instruction dominates the pooled
vector and every record sharing an instruction drifts toward the same point.
`pool_over_instruction=true` disables the masking so the effect can be measured
rather than assumed.

**Token budget.** Instruction and content are tokenised separately so the
content gets a guaranteed budget; truncating the concatenation instead would let
`noise_verbose` crowd out the address entirely.

**Case- and punctuation-preserving tokenisation.** Unlike `data.py`'s
`classify_cell`, which normalises text to *label* noise types, the tokenizer
that feeds the model does not normalise — case, punctuation and spacing inside
a token are exactly the surface signal the char-CNN half of the encoder is
meant to learn to handle. Normalising them away at tokenize time would erase
the thing the noise instruction is supposed to help the model see past.

**Canonical index scope.** The index always spans every clean record in the
slice, not just the evaluated split, so a hit has to beat the whole dataset.

**Dataset slices.** `poi_library` is ~64% corrupted cells; `full_005` through
`full_051` give a 5%–51% difficulty sweep; `de_*` and `russia_*` test whether an
English instruction transfers to German and Russian records. Slice keys are in
`config.DATASET_SLICES`.
