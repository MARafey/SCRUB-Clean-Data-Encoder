"""
Configuration for the Clean Data Encoder experiments.

Central place for every knob. Each experiment run is fully described by one
ExperimentConfig, which is serialised into the run directory so results are
reproducible and TensorBoard HParams can compare runs.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# Root of the project (the folder that holds AddressTable/, Papers/, code/).
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_ROOT = os.path.join(PROJECT_ROOT, "AddressTable", "experiments_datasets")
RUNS_ROOT = os.path.join(PROJECT_ROOT, "runs")

# Dataset slices available under experiments_datasets/effectiveness/.
# The trailing number is (approximately) the fraction of corrupted cells, so
# these give a natural difficulty sweep.
DATASET_SLICES = {
    "poi_library": "effectiveness/poi_Library",
    "poi_shopping": "effectiveness/poi_ShoppingCenter",
    "full_005": "effectiveness/full/sample_full_0.05",
    "full_010": "effectiveness/full/sample_full_0.10",
    "full_020": "effectiveness/full/sample_full_0.20",
    "full_051": "effectiveness/full/sample_full_0.51",
    "de_005": "effectiveness/de/sample_de_0.05",
    "de_010": "effectiveness/de/sample_de_0.10",
    "de_020": "effectiveness/de/sample_de_0.20",
    "de_048": "effectiveness/de/sample_de_0.48",
    "de_049": "effectiveness/de/sample_de_0.49",
    "russia_005": "effectiveness/russia/sample_russia_0.05",
    "russia_010": "effectiveness/russia/sample_russia_0.10",
    "russia_020": "effectiveness/russia/sample_russia_0.20",
    "russia_065": "effectiveness/russia/sample_russia_0.65",
    "runtime_10k": "runtime/full_10000",
    "runtime_100k": "runtime/full_100000",
}


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
@dataclass
class ExperimentConfig:
    # ---- identity -------------------------------------------------------
    run_name: str = "baseline"
    seed: int = 42

    # ---- data -----------------------------------------------------------
    dataset: str = "full_020"
    # Columns used to build the record text. Order matters for serialisation.
    fields: List[str] = field(default_factory=lambda: [
        "name", "house_number", "road", "postcode",
        "locality", "region", "country", "country_code",
    ])
    # "keyed"   -> "name: X | road: Y | ..."   (Ditto-style, explicit schema)
    # "natural" -> "X, 32 Y, 12345 Z, ..."     (reads like a written address)
    serialization: str = "keyed"
    drop_empty_fields: bool = True
    max_records: Optional[int] = None          # cap dataset size for quick runs
    train_frac: float = 0.70
    val_frac: float = 0.15                     # test_frac is the remainder

    # ---- instructions (THE experimental variable) -----------------------
    # Names resolved against instructions.INSTRUCTION_BANK.
    query_instruction: str = "noise_specific"  # applied to the DIRTY side
    doc_instruction: str = "none"              # applied to the CLEAN side
    # By default the encoder pools only over the content tokens and masks the
    # instruction tokens out. Setting this True pools over instruction tokens
    # too, which is a meaningful ablation of its own.
    pool_over_instruction: bool = False

    # ---- model ------------------------------------------------------------
    # A from-scratch encoder, trained only on this project's own data - no
    # pretrained weights, no external corpus. See tokenizer.py and model.py.
    # v2 defaults (bumped from v1's 128/4/64/[k=3]): a deliberately bigger
    # model, and a multi-kernel char-CNN that mirrors TF-IDF's actual 3-5
    # character n-gram range instead of a single kernel size - a direct shot
    # at the signal TF-IDF has been winning on.
    token_embed_dim: int = 256                 # trainable word-embedding width
    char_embed_dim: int = 32                   # per-character embedding width
    char_cnn_filters: int = 64                 # char-CNN filters PER kernel size
    char_cnn_kernel_sizes: List[int] = field(default_factory=lambda: [3, 4, 5])
    num_layers: int = 6                        # encoder blocks (Transformer or SNN)
    num_heads: int = 4                         # Transformer encoder only
    ffn_dim: int = 512
    dropout: float = 0.1

    # ---- v3 architecture (opt-in) ------------------------------------------
    # RMSNorm + rotary position embeddings (RoPE) + SwiGLU FFN, LLaMA-style.
    # All three are meant to be set together (true/true/true); model.py's v3
    # block hardcodes RMSNorm+SwiGLU whenever any one of them is set, so
    # mixing e.g. use_rope=true with the other two false isn't a supported
    # combination. Default False keeps the v1/v2 architecture (learned
    # position embedding, LayerNorm, GELU FFN) exactly as before.
    use_rope: bool = False
    use_rmsnorm: bool = False
    use_swiglu: bool = False
    # Width the word+char-CNN concat is projected to before the Transformer
    # blocks. None reuses token_embed_dim + char_cnn output width unchanged
    # (the v1/v2 behaviour); set e.g. 512 to decouple block width from the
    # token/char embedding sizes, as v3 does.
    model_dim: Optional[int] = None

    # encoder_type selects the block stack in model.build_encoder():
    #   "transformer" -> InstructionEncoder (model.py), self-attention + FFN
    #   "snn"         -> SCRUBSNNEncoder (model_snn.py), spiking LIF neurons,
    #                    no attention. A distinct architecture family, not a
    #                    controlled ablation of the Transformer (see
    #                    model_snn.py's docstring for why).
    encoder_type: str = "transformer"
    snn_threshold: float = 1.0                 # LIF firing threshold (init if learnable)
    snn_decay: float = 0.9                     # membrane leak per token step (init if learnable)
    snn_surrogate_slope: float = 10.0          # surrogate-gradient sharpness
    # v1 SNN (encoder_type=snn, default False): fixed threshold/decay, hand-set.
    # v2 SNN (snn_learnable_dynamics=true): per-channel threshold/decay become
    # trainable parameters, initialised from the values above so "learnable"
    # starts identical to "fixed" and only diverges where training finds it
    # helps - a response to v1's fixed decay=0.9 plausibly fading context too
    # fast for longer address records.
    snn_learnable_dynamics: bool = False

    # "pretrained" (model_pretrained.py): wraps a small pretrained multilingual
    # HuggingFace TF model instead of a from-scratch one. A deliberate side
    # experiment answering "how much of the gap to TF-IDF is the lack of a
    # pretrained prior" - see model_pretrained.py's module docstring. Results
    # from this path should be reported separately from SCRUB's own numbers,
    # never conflated with them.
    pretrained_model_name: str = "distilbert-base-multilingual-cased"
    pretrained_trainable: bool = True   # fine-tune the backbone vs. freeze it
    # Priority-2 redesign (isolating RQ1's pretraining-vs-architecture
    # confound): when True, model_pretrained.py builds the backbone from
    # pretrained_model_name's CONFIG ONLY (AutoConfig + TFAutoModel.from_config)
    # - same architecture, same hidden size/layers/heads, same tokenizer,
    # same batch size as the pretrained run - but with randomly initialised
    # weights instead of the pretrained checkpoint. This is the matched
    # control the earlier RQ1 comparison lacked: SCRUB-from-scratch vs.
    # pretrained-DistilBERT changed architecture, tokenizer, parameter count
    # AND batch size all at once, so "pretraining explains the gap" was never
    # actually isolated from those other three. This flag holds everything
    # but the weight initialisation fixed.
    pretrained_random_init: bool = False

    vocab_min_freq: int = 2                    # corpus words below this -> [UNK]
    max_chars_per_token: int = 16
    # Sequence budget. Instructions vary from 0 to ~160 tokens across the
    # bank, so this is treated as a floor: the encoder grows it as needed to
    # guarantee every variant `min_content_tokens` for the record itself.
    # Without that, a long instruction crowds out the address and the ablation
    # ends up measuring instruction length instead of instruction content.
    max_seq_length: int = 128
    min_content_tokens: int = 64
    normalize_embeddings: bool = True

    # ---- hardware -------------------------------------------------------
    # require_gpu=true turns "no GPU visible" into a hard error instead of a
    # silent fall back to CPU, which is the failure mode worth catching: a run
    # that quietly takes 40x longer looks identical in the logs otherwise.
    require_gpu: bool = False
    gpu_memory_growth: bool = True             # don't grab all VRAM up front

    # ---- training -------------------------------------------------------
    do_train: bool = True                      # False => zero-shot evaluation
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    temperature: float = 0.05                  # InfoNCE softmax temperature
    bidirectional_loss: bool = True            # query->doc and doc->query
    grad_clip_norm: float = 1.0
    mixed_precision: bool = False

    # ---- evaluation -----------------------------------------------------
    recall_at_k: List[int] = field(default_factory=lambda: [1, 5, 10])
    # Correction is "proposed" only when top-1 cosine similarity >= threshold.
    # Sweeping this gives the precision/recall trade-off; the best-F1 point is
    # what compares against the published 0.29 correction F1.
    similarity_thresholds: List[float] = field(
        default_factory=lambda: [round(0.50 + 0.02 * i, 2) for i in range(25)]
    )
    eval_batch_size: int = 64
    eval_every_steps: int = 200

    # ---- checkpointing --------------------------------------------------
    # The metric that decides which checkpoint is "best".
    best_metric: str = "recall@1"
    max_checkpoints_to_keep: int = 3
    resume: bool = True                        # auto-resume if a ckpt exists

    # ---- early stopping ---------------------------------------------------
    # Stops training once `best_metric` hasn't improved by at least
    # `early_stopping_min_delta` for `early_stopping_patience` consecutive
    # validation evals (each eval_every_steps). 0 disables early stopping.
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0

    # ---- visualisation --------------------------------------------------
    log_projector: bool = True                 # TensorBoard Embedding Projector
    projector_num_points: int = 750            # pairs sampled for projection
    log_tsne: bool = True
    tsne_num_points: int = 400

    # -------------------------------------------------------------------
    @property
    def run_dir(self) -> str:
        return os.path.join(RUNS_ROOT, self.run_name)

    @property
    def ckpt_dir(self) -> str:
        return os.path.join(self.run_dir, "checkpoints")

    @property
    def best_ckpt_dir(self) -> str:
        return os.path.join(self.run_dir, "checkpoints_best")

    @property
    def tb_dir(self) -> str:
        return os.path.join(self.run_dir, "tensorboard")

    @property
    def vocab_path(self) -> str:
        return os.path.join(self.run_dir, "vocab.json")

    @property
    def results_dir(self) -> str:
        return os.path.join(self.run_dir, "results")

    @property
    def data_dir(self) -> str:
        if self.dataset not in DATASET_SLICES:
            raise KeyError(
                f"Unknown dataset '{self.dataset}'. "
                f"Available: {sorted(DATASET_SLICES)}"
            )
        return os.path.join(DATA_ROOT, DATASET_SLICES[self.dataset])

    @property
    def fingerprint(self) -> str:
        """Short hash of the config, useful for spotting accidental changes."""
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()[:10]

    def make_dirs(self) -> None:
        for d in (self.run_dir, self.ckpt_dir, self.best_ckpt_dir,
                  self.tb_dir, self.results_dir):
            os.makedirs(d, exist_ok=True)

    def save(self) -> str:
        self.make_dirs()
        path = os.path.join(self.run_dir, "config.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls(**json.load(f))

    def to_hparams(self) -> dict:
        """Flat, scalar-only view for the TensorBoard HParams plugin."""
        return {
            "dataset": self.dataset,
            "num_layers": self.num_layers,
            "token_embed_dim": self.token_embed_dim,
            "query_instruction": self.query_instruction,
            "doc_instruction": self.doc_instruction,
            "pool_over_instruction": self.pool_over_instruction,
            "serialization": self.serialization,
            "do_train": self.do_train,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "temperature": self.temperature,
            "max_seq_length": self.max_seq_length,
        }


def parse_overrides(pairs: List[str], base: ExperimentConfig) -> ExperimentConfig:
    """Apply `key=value` CLI overrides onto a config, coercing to field types."""
    valid = {f.name: f for f in dataclasses.fields(ExperimentConfig)}
    updates = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Override '{pair}' is not in key=value form")
        key, raw = pair.split("=", 1)
        if key not in valid:
            raise KeyError(f"Unknown config key '{key}'")
        current = getattr(base, key)
        # `from __future__ import annotations` makes field.type a string, which
        # is what lets Optional[int] fields sitting at None still parse as ints.
        annotation = str(valid[key].type)
        raw = raw.strip()

        if raw.lower() == "none" and "Optional" in annotation:
            value = None
        elif isinstance(current, bool) or annotation == "bool":
            value = raw.lower() in ("1", "true", "yes", "y")
        elif isinstance(current, list) or annotation.startswith("List"):
            items = [x.strip() for x in raw.split(",") if x.strip()]
            if "int" in annotation:
                value = [int(x) for x in items]
            elif "float" in annotation:
                value = [float(x) for x in items]
            else:
                value = items
        elif isinstance(current, float) or "float" in annotation:
            value = float(raw)
        elif (isinstance(current, int) and not isinstance(current, bool)) \
                or "int" in annotation:
            value = int(raw)
        else:
            value = raw
        updates[key] = value
    return dataclasses.replace(base, **updates)
