"""
Data loading, noise characterisation and serialisation for AddressTable.

The dataset ships as row-aligned clean.csv / dirty.csv pairs: row i of dirty is
the corrupted version of row i of clean. That alignment is the supervision
signal for contrastive training, and the ground truth for retrieval evaluation.

Beyond loading, this module tags every cell with the *kind* of corruption
applied, which lets evaluation report metrics broken down by noise type. That
breakdown is what turns a single F1 number into an argument about which noise
types instruction-conditioning actually helps with.
"""
from __future__ import annotations

import os
import re
import string
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config import ExperimentConfig

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)

# Noise taxonomy, ordered from most to least specific. The first rule that
# fires wins, so ordering matters.
NOISE_TYPES = [
    "unchanged",
    "emptied",           # clean had a value, dirty is empty
    "added",             # clean was empty, dirty invented a value
    "whitespace",        # identical once whitespace is normalised
    "case",              # identical once lowercased
    "punctuation",       # identical once punctuation is stripped
    "code_substitution", # "Arizona" -> "AZ", "United States" -> "USA"
    "abbreviation",      # "Avenue" -> "Ave", "South Main Street" -> "S. Main St."
    "token_drop",        # dirty tokens are a strict subset of clean tokens
    "value_error",       # genuinely different value
]


# --------------------------------------------------------------------------
# Normalisation helpers
# --------------------------------------------------------------------------
def _ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _lower(s: str) -> str:
    return _ws(s).lower()


def _depunct(s: str) -> str:
    return _ws(s.translate(_PUNCT_TABLE)).lower()


def _tokens(s: str) -> List[str]:
    return [t for t in re.split(r"[\s\-/,.]+", s.lower()) if t]


def classify_cell(clean: str, dirty: str) -> str:
    """Label the corruption applied to a single cell."""
    clean = "" if clean is None else str(clean)
    dirty = "" if dirty is None else str(dirty)

    if clean == dirty:
        return "unchanged"
    if clean.strip() and not dirty.strip():
        return "emptied"
    if not clean.strip() and dirty.strip():
        return "added"
    if _ws(clean) == _ws(dirty):
        return "whitespace"
    if _lower(clean) == _lower(dirty):
        return "case"
    if _depunct(clean) == _depunct(dirty):
        return "punctuation"

    ct, dt = _tokens(clean), _tokens(dirty)

    # Short all-caps stand-in for a longer name: AZ for Arizona, DE for
    # Deutschland, USA for United States.
    compact = dirty.replace(".", "").replace(" ", "")
    if 1 <= len(compact) <= 4 and compact.isupper() and len(clean) > len(dirty):
        return "code_substitution"

    # Every dirty token is a prefix of some clean token, and at least one is a
    # strict shortening: the signature of Street -> St, Avenue -> Ave.
    if ct and dt:
        prefix_hits = 0
        shortened = False
        for d in dt:
            match = next((c for c in ct if c.startswith(d)), None)
            if match is not None:
                prefix_hits += 1
                if len(d) < len(match):
                    shortened = True
        if prefix_hits == len(dt) and shortened:
            return "abbreviation"

        if set(dt) < set(ct):
            return "token_drop"

    return "value_error"


def profile_rows(clean_df: pd.DataFrame,
                 dirty_df: pd.DataFrame,
                 fields: Sequence[str]) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Per-cell noise labels plus a dataset-level summary."""
    labels = pd.DataFrame(index=clean_df.index)
    for col in fields:
        labels[col] = [
            classify_cell(c, d)
            for c, d in zip(clean_df[col].astype(str), dirty_df[col].astype(str))
        ]

    flat = labels.values.ravel()
    total = max(len(flat), 1)
    summary = {t: float((flat == t).sum()) / total for t in NOISE_TYPES}
    summary["corrupted_cell_rate"] = 1.0 - summary.get("unchanged", 0.0)
    summary["corrupted_row_rate"] = float(
        (labels != "unchanged").any(axis=1).mean()
    )
    return labels, summary


# --------------------------------------------------------------------------
# Serialisation
# --------------------------------------------------------------------------
def serialize_record(row: pd.Series,
                     fields: Sequence[str],
                     mode: str = "keyed",
                     drop_empty: bool = True) -> str:
    """Turn one record into the string that gets embedded.

    keyed   -> "name: Mesa Community College | road: West Southern Avenue | ..."
               Explicit schema, in the style Ditto found effective.
    natural -> "Mesa Community College, 1833 West Southern Avenue, 85202 ..."
               Reads like a written address.
    """
    values = {f: _ws(str(row.get(f, "") or "")) for f in fields}

    if mode == "keyed":
        parts = []
        for f in fields:
            v = values[f]
            if not v and drop_empty:
                continue
            parts.append(f"{f}: {v}")
        return " | ".join(parts) if parts else "empty record"

    if mode == "natural":
        street = " ".join(x for x in (values.get("house_number", ""),
                                      values.get("road", "")) if x)
        city = " ".join(x for x in (values.get("postcode", ""),
                                    values.get("locality", "")) if x)
        chunks = [values.get("name", ""), street, city,
                  values.get("region", ""), values.get("country", "")]
        chunks = [c for c in chunks if c] if drop_empty else chunks
        return ", ".join(chunks) if chunks else "empty record"

    raise ValueError(f"Unknown serialization mode '{mode}'")


# --------------------------------------------------------------------------
# Dataset container
# --------------------------------------------------------------------------
@dataclass
class Split:
    name: str
    indices: np.ndarray          # positions into the full arrays
    dirty_texts: List[str]
    clean_texts: List[str]

    def __len__(self) -> int:
        return len(self.indices)


def _derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    """A random permutation of range(n) with no fixed points (perm[i] != i).

    Used to build the record_shuffled control (Priority 1 of the advisor's
    redesign): each row gets another row's ACTUAL noise-type description
    instead of its own, so "shuffled" never silently degenerates to
    "unchanged" for any row.
    """
    if n <= 1:
        return np.arange(n)
    perm = rng.permutation(n)
    fixed = np.where(perm == np.arange(n))[0]
    if len(fixed) == 1:
        j = (fixed[0] + 1) % n
        perm[fixed[0]], perm[j] = perm[j], perm[fixed[0]]
    elif len(fixed) > 1:
        perm[fixed] = np.roll(perm[fixed], 1)
    return perm


@dataclass
class AddressData:
    """Everything an experiment needs from one dataset slice.

    The canonical index (`canonical_texts`) is built from ALL clean records in
    the slice, not just the split being evaluated. A retrieval hit therefore has
    to beat every other clean record in the dataset, not a convenient subset.
    """
    fields: List[str]
    clean_df: pd.DataFrame
    dirty_df: pd.DataFrame
    noise_labels: pd.DataFrame
    noise_summary: Dict[str, float]
    canonical_texts: List[str]           # clean text for every row, in order
    dirty_texts: List[str]               # dirty text for every row, in order
    splits: Dict[str, Split]
    noise_shuffle_map: np.ndarray         # row i -> a different row's noise profile

    @property
    def num_records(self) -> int:
        return len(self.canonical_texts)

    def row_noise_types(self, idx: int) -> List[str]:
        """Noise types present in row `idx`, excluding 'unchanged'."""
        row = self.noise_labels.iloc[idx]
        return sorted({t for t in row.values if t != "unchanged"})

    def per_record_instructions(self, row_indices: Sequence[int],
                                shuffled: bool = False) -> List[str]:
        """Build one noise-description instruction per row, from the noise
        types actually present in that row (or, if shuffled=True, in a
        different, deliberately-wrong row - the record_shuffled control).
        """
        from instructions import record_specific_instruction
        source_rows = (
            [int(self.noise_shuffle_map[i]) for i in row_indices]
            if shuffled else list(row_indices)
        )
        return [
            record_specific_instruction(self.row_noise_types(r)) + " "
            for r in source_rows
        ]


def load_dataset(cfg: ExperimentConfig, verbose: bool = True) -> AddressData:
    clean_path = os.path.join(cfg.data_dir, "clean.csv")
    dirty_path = os.path.join(cfg.data_dir, "dirty.csv")
    for p in (clean_path, dirty_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing {p}. Check DATA_ROOT in config.py points at "
                f"AddressTable/experiments_datasets."
            )

    read_kwargs = dict(dtype=str, keep_default_na=False, na_filter=False)
    clean_df = pd.read_csv(clean_path, **read_kwargs)
    dirty_df = pd.read_csv(dirty_path, **read_kwargs)

    if len(clean_df) != len(dirty_df):
        raise ValueError(
            f"clean/dirty row mismatch: {len(clean_df)} vs {len(dirty_df)}. "
            f"The pipeline assumes row i of dirty corrupts row i of clean."
        )

    missing = [f for f in cfg.fields if f not in clean_df.columns]
    if missing:
        raise KeyError(f"Fields {missing} not in dataset columns "
                       f"{list(clean_df.columns)}")

    if cfg.max_records is not None and cfg.max_records < len(clean_df):
        rng = np.random.default_rng(cfg.seed)
        keep = np.sort(rng.choice(len(clean_df), cfg.max_records, replace=False))
        clean_df = clean_df.iloc[keep].reset_index(drop=True)
        dirty_df = dirty_df.iloc[keep].reset_index(drop=True)

    noise_labels, noise_summary = profile_rows(clean_df, dirty_df, cfg.fields)

    canonical_texts = [
        serialize_record(r, cfg.fields, cfg.serialization, cfg.drop_empty_fields)
        for _, r in clean_df.iterrows()
    ]
    dirty_texts = [
        serialize_record(r, cfg.fields, cfg.serialization, cfg.drop_empty_fields)
        for _, r in dirty_df.iterrows()
    ]

    # Deterministic split over row indices.
    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(len(clean_df))
    n_train = int(cfg.train_frac * len(order))
    n_val = int(cfg.val_frac * len(order))
    bounds = {
        "train": order[:n_train],
        "val": order[n_train:n_train + n_val],
        "test": order[n_train + n_val:],
    }
    splits = {
        name: Split(
            name=name,
            indices=np.sort(idx),
            dirty_texts=[dirty_texts[i] for i in np.sort(idx)],
            clean_texts=[canonical_texts[i] for i in np.sort(idx)],
        )
        for name, idx in bounds.items()
    }

    if verbose:
        print(f"[data] slice          : {cfg.dataset}  ({cfg.data_dir})")
        print(f"[data] records        : {len(clean_df)}")
        print(f"[data] corrupted cells: {noise_summary['corrupted_cell_rate']:.3f}")
        print(f"[data] corrupted rows : {noise_summary['corrupted_row_rate']:.3f}")
        top = sorted(
            ((k, v) for k, v in noise_summary.items()
             if k in NOISE_TYPES and k != "unchanged" and v > 0),
            key=lambda kv: -kv[1],
        )
        print("[data] noise mix      : " +
              ", ".join(f"{k} {v:.3f}" for k, v in top))
        print(f"[data] splits         : " +
              ", ".join(f"{n} {len(s)}" for n, s in splits.items()))
        print(f"[data] example dirty  : {dirty_texts[0][:120]}")
        print(f"[data] example clean  : {canonical_texts[0][:120]}")

    # Deterministic given cfg.seed, independent of the train/val/test split
    # permutation above (a different rng draw), so it doesn't covertly shift
    # which rows land in which split.
    shuffle_rng = np.random.default_rng(cfg.seed + 1)
    noise_shuffle_map = _derangement(len(clean_df), shuffle_rng)

    return AddressData(
        fields=list(cfg.fields),
        clean_df=clean_df,
        dirty_df=dirty_df,
        noise_labels=noise_labels,
        noise_summary=noise_summary,
        canonical_texts=canonical_texts,
        dirty_texts=dirty_texts,
        splits=splits,
        noise_shuffle_map=noise_shuffle_map,
    )


def make_train_arrays(data: AddressData,
                      split: str = "train") -> Tuple[List[str], List[str]]:
    """(dirty, clean) text pairs for contrastive training."""
    s = data.splits[split]
    return s.dirty_texts, s.clean_texts
