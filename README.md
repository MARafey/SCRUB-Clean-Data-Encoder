# Clean Data Encoder (SCRUB)

Noise-instruction-conditioned embeddings for matching corrupted address records
to a clean canonical database. **SCRUB** is a from-scratch contrastive encoder
(no pretrained backbone) used to test whether an explicit *noise-description*
instruction pulls a corrupted record's embedding onto its clean twin.

Full documentation, results and caveats: **[code/README.md](code/README.md)**.

## Dataset

Experiments use the **AddressTable** benchmark from *Clean Me If You Can: A
Large Collection of Real-World Addresses for Data Cleaning Benchmarking*
(Ahmadi, Bernhard, Abdelmaksoud, Zecchini, Rabl, Abedjan).

- Paper: <https://arxiv.org/abs/2606.31983>
- Data and extraction code: <https://github.com/D2IP-TUB/Clean-Me-If-You-Can>

The dataset (~16 GB) is **not included in this repository**. Download it from
the link above and place it so the slices sit under
`AddressTable/experiments_datasets/` (e.g. `effectiveness/`, `runtime/`) at the
repo root. Dataset keys such as `poi_library`, `full_020`, `de_048`,
`russia_065` and `runtime_100k` map to files there via
`DATASET_SLICES` in `code/config.py`.

## Quick start

Requires [uv](https://docs.astral.sh/uv/). All commands run from `code/`.

```bat
cd code
uv sync
run.bat train.py --run smoke max_records=200 epochs=1 --eval-after
run.bat baseline_lexical.py poi_library
run.bat run_ablation.py --dataset poi_library --train --variants none,noise_terse,noise_specific
```

Use `run.sh` instead of `run.bat` on Linux/macOS. GPU use requires Linux or
WSL2 (`uv sync --extra gpu`).

## Repository layout

| Path | Contents |
|---|---|
| `code/` | All source: encoder (`model.py`), tokenizer, losses, training, evaluation, ablation sweep, TF-IDF baseline |
| `runs/` | Per-run metrics and vocabularies from past experiments (checkpoints and TensorBoard logs are git-ignored) |
| `Papers/` | Index of related-work papers (`README.md`, gap analysis); the PDFs are not redistributed |
| `AddressTable/` | Dataset, downloaded separately (git-ignored) |
