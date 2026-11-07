"""
Pretrained-backbone variant of the encoder: a side experiment, not a
replacement for SCRUB.

Every other encoder in this project (model.py's Transformer, model_snn.py's
spiking variant) is trained entirely from scratch - see model.py's module
docstring for why that is a deliberate project decision. This module exists
specifically to answer a different, narrower question: how much of SCRUB's
gap to the TF-IDF baseline is attributable to having no pretrained language
prior at all, versus limits of the from-scratch architecture/training itself?
It wraps a small pretrained multilingual encoder (HuggingFace `transformers`,
TF backend) and fine-tunes it on the same contrastive pairs and the same
noise-instruction text as SCRUB.

Any result here should be reported as a separate comparison point, not
conflated with "SCRUB's" own numbers - a win here would show pretrained
language knowledge helps this task (already expected from the literature),
not that the noise-instruction mechanism itself works.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Union

import numpy as np
import tensorflow as tf

from config import ExperimentConfig

# transformers' TF model classes need the tf-keras (Keras 2) compatibility
# shim under TensorFlow >= 2.16, which defaults to Keras 3. Must be set before
# `transformers` is imported.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

from transformers import AutoConfig, AutoTokenizer, TFAutoModel  # noqa: E402


class PretrainedInstructionEncoder(tf.keras.Model):
    """Same public surface as model.InstructionEncoder (tokenize, __call__,
    encode, hidden_size, checkpoint-compatible), backed by a pretrained
    HuggingFace TF model instead of a from-scratch one.

    Instruction and content are tokenized together as one string so the
    pretrained subword tokenizer sees natural text, but the instruction's
    token span is tracked and masked out of the pooling step exactly like
    the from-scratch encoder's asymmetric pooling - the clean/canonical
    side stays effectively instruction-free at the representation level.
    """

    def __init__(self, cfg: ExperimentConfig, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        # The tokenizer is rule-based (WordPiece merges fixed by the model's
        # published vocab), not a trained prior in the sense RQ1 cares about,
        # so it is loaded identically in both conditions below - only the
        # BACKBONE's weight initialisation differs.
        self.hf_tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_model_name)
        if cfg.pretrained_random_init:
            # Same architecture (AutoConfig carries hidden size, num layers,
            # num heads, vocab size, every structural hyperparameter) as the
            # pretrained checkpoint, but TFAutoModel.from_config randomly
            # initialises every weight instead of loading the checkpoint.
            # This is the matched control for RQ1: architecture, tokenizer,
            # parameter count and (via cfg.batch_size, set identically by the
            # caller) batch size are now held fixed between the two
            # conditions, so a measured difference can actually be
            # attributed to pretraining rather than to any of those other
            # three variables changing at the same time.
            hf_config = AutoConfig.from_pretrained(cfg.pretrained_model_name)
            self.backbone = TFAutoModel.from_config(hf_config)
            # from_config builds the underlying Keras layers lazily, on
            # first call; from_pretrained eagerly materialises every weight
            # while loading the checkpoint. Without forcing the same eager
            # materialisation here, self.backbone.trainable_variables is
            # empty until the first real training step, and the optimizer
            # crashes on zero (grad, variable) pairs. A one-token dummy
            # forward pass forces the same lazy-build to happen now instead.
            dummy_ids = tf.zeros((1, 2), dtype=tf.int32)
            dummy_mask = tf.ones((1, 2), dtype=tf.int32)
            _ = self.backbone(input_ids=dummy_ids, attention_mask=dummy_mask,
                              training=False)
        else:
            self.backbone = TFAutoModel.from_pretrained(cfg.pretrained_model_name)
        self.backbone.trainable = cfg.pretrained_trainable
        self._hidden_size = int(self.backbone.config.hidden_size)

    # ------------------------------------------------------------------
    # This outer class subclasses Keras 3's tf.keras.Model, but `self.backbone`
    # (from `transformers`, built under TF_USE_LEGACY_KERAS=1) is a Keras 2
    # (tf_keras) Model. Keras 3's automatic sub-layer discovery does not
    # reliably recognise a tf_keras Layer/Model as a trackable sub-layer for a
    # from_config()-built (random-init) backbone: `self.trainable_variables`
    # silently returns an empty list, and the optimizer then crashes on zero
    # (grad, variable) pairs at the first training step. from_pretrained()'s
    # own internal weight-loading path happens to populate variables in a way
    # the outer class's auto-tracking can still see, which is why this was
    # never visible before encoder_type=pretrained gained a
    # pretrained_random_init=True branch. Overriding the trainable_variables/
    # variables/weights *properties* directly is NOT safe here: Keras 3's own
    # Model/Layer machinery uses those exact names internally during
    # __init__ and trainable-flag handling, and shadowing them breaks
    # attribute assignment later in this same __init__ (observed: later
    # self._x = {} assignments silently failing to take effect). Callers that
    # need the real trainable variables for this encoder type must use
    # `encoder.backbone.trainable_variables` directly instead of
    # `encoder.trainable_variables` - see model.py's build_encoder and
    # train.py's step_fn, both of which already branch on cfg.encoder_type
    # for other reasons.

        # Per-text caches: the same record/instruction text repeats every
        # epoch (only batch order changes), so tokenizing it more than once
        # is pure waste. Keyed by raw string -> list[int] subword ids.
        self._content_id_cache: Dict[str, List[int]] = {}
        self._instruction_id_cache: Dict[str, List[int]] = {}

    # ------------------------------------------------------------------
    def _encode_content_ids(self, text: str) -> List[int]:
        cached = self._content_id_cache.get(text)
        if cached is None:
            cached = self.hf_tokenizer(
                text, add_special_tokens=False, truncation=True,
                max_length=self.cfg.max_seq_length,
            )["input_ids"]
            self._content_id_cache[text] = cached
        return cached

    def _encode_instruction_ids(self, instruction: str) -> List[int]:
        cached = self._instruction_id_cache.get(instruction)
        if cached is None:
            cached = (self.hf_tokenizer(instruction, add_special_tokens=False)["input_ids"]
                      if instruction else [])
            self._instruction_id_cache[instruction] = cached
        return cached

    # ------------------------------------------------------------------
    def tokenize(self,
                texts: List[str],
                instruction: Union[str, List[str]] = "") -> Dict[str, tf.Tensor]:
        """See model.InstructionEncoder.tokenize's docstring: `instruction`
        may be one string shared by the whole batch, or a list of one
        instruction per row (record_specific / record_shuffled)."""
        cls_id = self.hf_tokenizer.cls_token_id
        sep_id = self.hf_tokenizer.sep_token_id
        pad_id = self.hf_tokenizer.pad_token_id

        if isinstance(instruction, str):
            instructions = [instruction] * len(texts)
        else:
            instructions = list(instruction)
            if len(instructions) != len(texts):
                raise ValueError(
                    f"per-row instruction list has {len(instructions)} "
                    f"entries but there are {len(texts)} texts"
                )

        inst_id_lists = [self._encode_instruction_ids(ins) for ins in instructions]
        max_inst_len = max((len(i) for i in inst_id_lists), default=0)
        content_budget = max(8, self.cfg.max_seq_length - max_inst_len - 2)

        rows_ids, rows_pool = [], []
        for text, inst_ids in zip(texts, inst_id_lists):
            inst_len = len(inst_ids)
            content_ids = self._encode_content_ids(text)[:content_budget]
            seq = [cls_id] + inst_ids + content_ids + [sep_id]

            pool = np.zeros(len(seq), dtype=np.int32)
            if self.cfg.pool_over_instruction:
                pool[1:-1] = 1                       # everything but [CLS]/[SEP]
            else:
                start = 1 + inst_len
                end = start + len(content_ids)
                pool[start:end] = 1
                if pool.sum() == 0:                  # degenerate: empty content
                    pool[1:-1] = 1

            rows_ids.append(seq)
            rows_pool.append(pool)

        max_len = max((len(r) for r in rows_ids), default=1) or 1
        input_ids = np.full((len(texts), max_len), pad_id, dtype=np.int32)
        attention = np.zeros((len(texts), max_len), dtype=np.int32)
        pool_mask = np.zeros((len(texts), max_len), dtype=np.int32)

        for i, (ids, pool) in enumerate(zip(rows_ids, rows_pool)):
            n = len(ids)
            input_ids[i, :n] = ids
            attention[i, :n] = 1
            pool_mask[i, :n] = pool

        return {
            "input_ids": tf.convert_to_tensor(input_ids),
            "attention_mask": tf.convert_to_tensor(attention),
            "pool_mask": tf.convert_to_tensor(pool_mask),
        }

    # ------------------------------------------------------------------
    def call(self, inputs: Dict[str, tf.Tensor], training: bool = False):
        out = self.backbone(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            training=training,
        )
        hidden = out.last_hidden_state                          # [B,T,H]

        mask = tf.cast(inputs["pool_mask"], hidden.dtype)[..., None]
        summed = tf.reduce_sum(hidden * mask, axis=1)
        counts = tf.maximum(tf.reduce_sum(mask, axis=1), 1e-9)
        pooled = summed / counts

        if self.cfg.normalize_embeddings:
            pooled = tf.math.l2_normalize(pooled, axis=-1)
        return pooled

    # ------------------------------------------------------------------
    def encode(self,
               texts: List[str],
               instruction: Union[str, List[str]] = "",
               batch_size: int = 64,
               training: bool = False,
               progress_every: Optional[int] = None) -> np.ndarray:
        per_row = not isinstance(instruction, str)
        if per_row and len(instruction) != len(texts):
            raise ValueError(
                f"per-row instruction list has {len(instruction)} entries "
                f"but there are {len(texts)} texts"
            )
        vecs = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            inst_chunk = (instruction[start:start + batch_size] if per_row
                         else instruction)
            batch = self.tokenize(chunk, inst_chunk)
            vecs.append(self(batch, training=training).numpy())
            if progress_every and (start // batch_size) % progress_every == 0:
                done = min(start + batch_size, len(texts))
                print(f"    encoded {done}/{len(texts)}", flush=True)
        if not vecs:
            return np.zeros((0, self.hidden_size), dtype=np.float32)
        return np.concatenate(vecs, axis=0).astype(np.float32)

    @property
    def hidden_size(self) -> int:
        return self._hidden_size
