"""
The instruction bank.

This is the experimental variable of the whole project. Published
instruction-conditioned embedding models (INSTRUCTOR, E5-Mistral) are trained
on *task*-descriptive instructions ("Represent the Wikipedia question for
retrieving supporting documents"). Neither their training data nor this
project's from-scratch encoder (model.py) has ever seen a *noise*-descriptive
instruction before training starts. The hypothesis under test is that
describing the corruption explicitly, on the dirty side only, pulls the dirty
record's embedding toward its clean canonical twin - and, for the from-scratch
encoder specifically, that the model can learn what these instruction words
mean purely from co-occurring with the corresponding noise pattern in the
contrastive training pairs, with no pretraining to draw on.

The noise descriptions below are grounded in the corruption actually present in
the AddressTable dirty.csv files, measured against clean.csv:

  * fields emptied entirely            (~10% of cells at the 0.20 slice)
  * abbreviation      Avenue -> Ave, Street -> St., North -> N
  * code substitution Arizona -> AZ, United States -> USA, Hessen -> HE
  * case corruption   "2nd Street" -> "2Nd Street", ALL CAPS, Title Casing
  * whitespace        leading/trailing spaces, collapsed or doubled spaces
  * punctuation       apostrophes and hyphens dropped or inserted
  * token dropping    "Bibliotheque de la HETSL" -> "Bibliotheque HETSL"
  * genuine errors    wrong postcode, wrong region value

Add a variant here, reference it by key from ExperimentConfig, and the ablation
runner will pick it up automatically.
"""
from __future__ import annotations

from typing import Dict, Sequence


INSTRUCTION_BANK: Dict[str, str] = {

    # ------------------------------------------------------------------
    # Control: no instruction at all. This is the plain-embedding baseline
    # that everything else has to beat for the thesis to hold.
    # ------------------------------------------------------------------
    "none": "",

    # ------------------------------------------------------------------
    # Control: a task-descriptive instruction in the exact style INSTRUCTOR
    # was trained on. Isolates "does ANY instruction help" from "does a
    # NOISE-DESCRIPTIVE instruction help".
    # ------------------------------------------------------------------
    "generic_retrieval":
        "Represent the address record for retrieving the matching record:",

    "generic_domain":
        "Represent the postal address and point-of-interest record for "
        "retrieving the matching record:",

    # ------------------------------------------------------------------
    # Control: vague gesture at messiness, without naming any noise type.
    # If this matches noise_specific, then naming the noise types adds
    # nothing and the contribution shrinks.
    # ------------------------------------------------------------------
    "generic_noisy":
        "Represent this messy, low-quality address record for retrieving the "
        "matching record:",

    # ------------------------------------------------------------------
    # The hypothesis: name the noise types explicitly.
    # ------------------------------------------------------------------
    "noise_specific":
        "Represent this corrupted address record for retrieving its clean "
        "canonical version. The record may contain irregular spacing, missing "
        "fields, abbreviated street types, inconsistent capitalisation, and "
        "dropped apostrophes or hyphens. Ignore these formatting artifacts and "
        "represent only the underlying real-world location:",

    # ------------------------------------------------------------------
    # Hypothesis + domain grounding, mirroring INSTRUCTOR's optional
    # "domain" slot in its instruction template.
    # ------------------------------------------------------------------
    "noise_domain":
        "Represent this corrupted postal address and point-of-interest record "
        "for retrieving its clean canonical version. Fields may be missing, "
        "street types may be abbreviated (Avenue as Ave, Street as St), "
        "regions and countries may appear as codes (Arizona as AZ, United "
        "States as USA), capitalisation and spacing may be irregular, and "
        "apostrophes or hyphens may be dropped. Ignore these artifacts and "
        "represent only the underlying place:",

    # ------------------------------------------------------------------
    # Maximum specificity: enumerate every measured corruption with an
    # example. Tests whether more detail keeps helping or saturates.
    # ------------------------------------------------------------------
    "noise_verbose":
        "Represent this corrupted address record for retrieving its clean "
        "canonical version. The corruption includes: entire fields left empty; "
        "street types abbreviated (Avenue as Ave, Street as St, Boulevard as "
        "Blvd, North as N); regions and countries replaced by codes (Arizona as "
        "AZ, Hessen as HE, United States as USA or US); irregular "
        "capitalisation (2nd written as 2Nd, names in all capitals, articles "
        "title-cased); leading, trailing or doubled whitespace; apostrophes and "
        "hyphens dropped or inserted; and words dropped from names. Look past "
        "every one of these surface artifacts and represent only the "
        "underlying real-world location the record refers to:",

    # ------------------------------------------------------------------
    # A minimal, imperative phrasing. INSTRUCTOR is reported to be robust to
    # paraphrase; this tests whether that holds for noise instructions too.
    # ------------------------------------------------------------------
    "noise_terse":
        "Ignore spacing, casing, abbreviation and punctuation noise. Represent "
        "the underlying address:",

    # ------------------------------------------------------------------
    # Clean / canonical side instructions.
    # Default is "none" (asymmetric setup). "doc_canonical" makes the setup
    # symmetric-with-different-instructions, mirroring how INSTRUCTOR handles
    # asymmetric retrieval tasks.
    # ------------------------------------------------------------------
    "doc_canonical":
        "Represent the clean canonical address record for retrieval:",

    "doc_domain":
        "Represent the clean canonical postal address and point-of-interest "
        "record for retrieval:",
}


# Variants swept by run_ablation.py by default. Ordered from control to
# hypothesis so the resulting table reads as an argument.
ABLATION_QUERY_VARIANTS = [
    "none",               # no instruction at all
    "generic_retrieval",  # task instruction, no noise mentioned
    "generic_noisy",      # vague messiness, no noise types named
    "noise_terse",        # noise types named, minimal phrasing
    "noise_specific",     # noise types named
    "noise_domain",       # noise types + domain
    "noise_verbose",      # noise types enumerated with examples
]


# ------------------------------------------------------------------
# Per-record instructions (addresses the advisor's Priority 1 critique):
# every variant above is ONE FIXED STRING applied identically to every query
# in a run, enumerating noise types that MIGHT be present anywhere in the
# slice - never the noise actually present in that specific record. That
# means no earlier ablation could distinguish "the model uses the
# instruction's content" from "the model learned a constant extra prefix".
#
# These three variants instead build a different instruction per query, from
# data.row_noise_types(idx) (already computed per-cell by data.py, but never
# previously fed into an instruction):
#   record_specific  - this record's OWN, actually-present noise types.
#                       The genuine test of the hypothesis.
#   record_shuffled  - another, randomly-chosen record's noise types,
#                       deliberately WRONG for this record. If this performs
#                       as well as record_specific, the model is not using
#                       the instruction's content correctly, only reacting to
#                       "an instruction of this general shape is present".
#   record_nonsense  - a token-length-matched instruction with no semantic
#                       content about noise or the task at all (answers "what
#                       result would falsify the claim that semantic noise
#                       information helps, as opposed to merely adding extra
#                       prompt tokens?"). If this matches record_specific, the
#                       effect is prefix-length, not instruction content.
# ------------------------------------------------------------------
NOISE_TYPE_PHRASES: Dict[str, str] = {
    "emptied": "one or more fields have been left empty",
    "added": "a field contains a value not present in the canonical record",
    "whitespace": "spacing is irregular, with leading, trailing, or doubled whitespace",
    "case": "capitalisation is irregular",
    "punctuation": "an apostrophe or hyphen has been dropped or inserted",
    "code_substitution": "a region or country has been replaced by a short code "
                         "(for example Arizona written as AZ)",
    "abbreviation": "a street type has been abbreviated (for example Avenue "
                    "written as Ave)",
    "token_drop": "one or more words have been dropped from a name",
    "value_error": "a field value is simply incorrect",
}

# A filler sentence of comparable length to a 3-4-noise-type record_specific
# instruction, carrying no information about noise, corruption, or even the
# matching task - the length-matched-nonsense control.
RECORD_NONSENSE_FILLER = (
    "Represent this address record. The following sentence is included only "
    "to match instruction length and carries no information about the record: "
    "the quick brown fox jumps over the lazy dog near the river on a quiet "
    "afternoon in early autumn, while clouds drift slowly overhead:"
)
# Fixed string, not per-record - belongs in the regular bank, resolved like
# any other key via get_instruction(). Added here (after the constant it
# reuses) rather than inline above, to keep the filler text next to the
# per-record variants it was designed to compare against.
INSTRUCTION_BANK["record_nonsense"] = RECORD_NONSENSE_FILLER


def record_specific_instruction(noise_types: Sequence[str]) -> str:
    """Build a noise-description instruction from the noise types actually
    present in ONE record (data.row_noise_types(idx)), rather than the fixed,
    generic enumeration every other noise_* variant uses for every row."""
    present = [t for t in noise_types if t in NOISE_TYPE_PHRASES]
    if not present:
        # A row with no corruption (should not occur in a corrupted-rows
        # dataset, but kept for robustness): fall back to a task-only framing
        # rather than claiming noise that is not there.
        return ("Represent this address record for retrieving its clean "
                "canonical version:")
    phrases = "; ".join(NOISE_TYPE_PHRASES[t] for t in present)
    return (
        "Represent this corrupted address record for retrieving its clean "
        f"canonical version. In this record specifically: {phrases}. Ignore "
        "these artifacts and represent only the underlying real-world "
        "location:"
    )


# Variant keys that resolve to a PER-RECORD instruction rather than one fixed
# string from INSTRUCTION_BANK. Callers (train.py, evaluate.py) must check
# `is_per_record(name)` and build a list of instructions, one per row, instead
# of calling get_instruction(name) once for the whole batch. record_nonsense
# is NOT in this set: it is a fixed string (same for every row), resolved
# normally through get_instruction() like any other INSTRUCTION_BANK key.
PER_RECORD_VARIANTS = ("record_specific", "record_shuffled")


def is_per_record(name: str) -> bool:
    return name in PER_RECORD_VARIANTS


# Swept by run_ablation.py's --per-record-ablation path (see run_ablation.py).
# Ordered control -> hypothesis, same convention as ABLATION_QUERY_VARIANTS.
PER_RECORD_ABLATION_VARIANTS = [
    "none",              # no instruction (shared control with the main ablation)
    "generic_retrieval", # task instruction, no noise named (shared control)
    "record_nonsense",   # same instruction LENGTH, no semantic noise content
    "record_shuffled",   # real noise-type content, but for the WRONG record
    "record_specific",   # this record's own, actually-present noise types
]


def get_instruction(name: str) -> str:
    """Resolve an instruction by key. Empty string means 'no instruction'."""
    if name in PER_RECORD_VARIANTS:
        raise ValueError(
            f"'{name}' is a per-record instruction and has no single fixed "
            f"string - build it per row via record_specific_instruction(), "
            f"not get_instruction(). See the per_record_instructions() "
            f"helper in train.py/evaluate.py."
        )
    if name not in INSTRUCTION_BANK:
        raise KeyError(
            f"Unknown instruction '{name}'. "
            f"Available: {sorted(INSTRUCTION_BANK)}"
        )
    text = INSTRUCTION_BANK[name].strip()
    # A single trailing space separates instruction from content when the two
    # are concatenated; the tokenizer handles the rest.
    return (text + " ") if text else ""


def describe(name: str) -> str:
    """Human-readable one-liner for logging."""
    if name in PER_RECORD_VARIANTS:
        return f"{name}: <per-record, built from each row's own noise types>"
    text = INSTRUCTION_BANK.get(name, "")
    if not text:
        return f"{name}: <no instruction>"
    preview = " ".join(text.split())
    if len(preview) > 110:
        preview = preview[:107] + "..."
    return f"{name}: {preview}"
