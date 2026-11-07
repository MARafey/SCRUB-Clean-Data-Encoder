"""
SCRUB-SNN: spiking-neuron variant of the SCRUB encoder.

This is a genuinely different architecture family, not a drop-in ablation of
`model.py`'s Transformer blocks: it replaces self-attention + a real-valued
FFN activation with a stack of leaky integrate-and-fire (LIF) neuron layers,
so there is no attention mechanism at all. Two things change at once relative
to `InstructionEncoder` (no attention, and a spiking rather than real-valued
nonlinearity), so a result here should be read as "does this alternative
architecture family do better or worse," not as an isolated test of spiking
neurons specifically holding everything else constant.

Why this exists: TensorFlow has no mature SNN layer, so this is implemented
from scratch - LIF neurons with a surrogate-gradient estimator for the
non-differentiable spike function (SuperSpike-style fast-sigmoid surrogate),
trained end to end with ordinary backprop through the unrolled token sequence
via `tf.scan`. The token representation (word embedding + multi-kernel
char-CNN) is copied unchanged from `model.py` so any accuracy difference is
attributable to the block stack, not the input features.

Membrane dynamics per LIF block, one step per token position:

    mem[t]   = decay * mem[t-1] * (1 - spike[t-1]) + input[t]   (reset-on-spike)
    spike[t] = SurrogateHeaviside(mem[t] - threshold)

Padding positions are zeroed before entering the recurrence so they cannot
inject spurious membrane current.
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np
import tensorflow as tf

from config import ExperimentConfig
from data import AddressData
from instructions import INSTRUCTION_BANK
from tokenizer import Tokenizer, Vocab


def _make_spike_fn(slope: float):
    """Heaviside step forward; fast-sigmoid surrogate gradient backward."""

    @tf.custom_gradient
    def spike_fn(x):
        spikes = tf.cast(x > 0.0, x.dtype)

        def grad(dy):
            surrogate = 1.0 / tf.square(1.0 + slope * tf.abs(x))
            return dy * surrogate

        return spikes, grad

    return spike_fn


class _LIFBlock(tf.keras.layers.Layer):
    """Pre-norm residual block: Dense -> LIF spiking dynamics -> Dense."""

    def __init__(self, d_model: int, ffn_dim: int, decay: float,
                threshold: float, surrogate_slope: float, dropout: float,
                learnable_dynamics: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.in_proj = tf.keras.layers.Dense(ffn_dim)
        self.out_proj = tf.keras.layers.Dense(d_model)
        self.drop = tf.keras.layers.Dropout(dropout)
        self.ffn_dim = ffn_dim
        self.spike_fn = _make_spike_fn(surrogate_slope)
        self.learnable_dynamics = learnable_dynamics

        if learnable_dynamics:
            # Per-channel, trained from an init matching the fixed values, so
            # "learnable" starts identical to "fixed" and only diverges where
            # training finds it helps. decay is squashed through a sigmoid to
            # stay in (0, 1); threshold through a softplus to stay positive.
            decay = min(max(decay, 1e-4), 1 - 1e-4)
            decay_init = float(np.log(decay / (1.0 - decay)))       # logit
            thresh_init = float(np.log(np.expm1(threshold)))         # inv-softplus
            self.decay_raw = self.add_weight(
                name="decay_raw", shape=(ffn_dim,),
                initializer=tf.keras.initializers.Constant(decay_init),
                trainable=True,
            )
            self.threshold_raw = self.add_weight(
                name="threshold_raw", shape=(ffn_dim,),
                initializer=tf.keras.initializers.Constant(thresh_init),
                trainable=True,
            )
        else:
            self.decay = decay
            self.threshold = threshold

    def call(self, x, mask, training: bool = False):
        h = self.norm(x)
        proj = self.in_proj(h)                              # [B,T,ffn_dim]
        proj = proj * mask[..., None]                        # zero padding
        proj_t = tf.transpose(proj, [1, 0, 2])                # [T,B,ffn_dim]

        if self.learnable_dynamics:
            decay = tf.sigmoid(self.decay_raw)                # [ffn_dim], (0,1)
            threshold = tf.math.softplus(self.threshold_raw)  # [ffn_dim], >0
        else:
            decay = self.decay
            threshold = self.threshold

        def step(carry, cur):
            mem, spike_prev = carry
            mem = decay * mem * (1.0 - spike_prev) + cur
            spike = self.spike_fn(mem - threshold)
            return (mem, spike)

        batch = tf.shape(proj)[0]
        init = (tf.zeros([batch, self.ffn_dim], dtype=proj.dtype),
                tf.zeros([batch, self.ffn_dim], dtype=proj.dtype))
        _, spikes_t = tf.scan(step, proj_t, initializer=init)  # [T,B,ffn_dim]

        spikes = tf.transpose(spikes_t, [1, 0, 2])            # [B,T,ffn_dim]
        out = self.out_proj(spikes)
        out = self.drop(out, training=training)
        return x + out


class SCRUBSNNEncoder(tf.keras.Model):
    """Spiking-neuron SCRUB variant. Same public interface as
    `model.InstructionEncoder` (tokenize/call/encode/hidden_size), so
    train.py / evaluate.py / run_ablation.py need no changes to use it -
    selected via `cfg.encoder_type = "snn"`.
    """

    def __init__(self, cfg: ExperimentConfig, vocab: Vocab, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.vocab = vocab
        self.tok = Tokenizer(vocab, max_chars_per_token=cfg.max_chars_per_token)

        self.d_model = cfg.token_embed_dim + cfg.char_cnn_filters * len(
            cfg.char_cnn_kernel_sizes)

        self.word_embed = tf.keras.layers.Embedding(
            len(vocab), cfg.token_embed_dim, name="word_embedding"
        )
        self.char_embed = tf.keras.layers.Embedding(
            vocab.num_chars, cfg.char_embed_dim, name="char_embedding"
        )
        self.char_convs = [
            tf.keras.layers.Conv1D(
                cfg.char_cnn_filters, kernel_size=k, padding="same",
                activation="relu", name=f"char_cnn_k{k}"
            )
            for k in cfg.char_cnn_kernel_sizes
        ]
        self.char_pool = tf.keras.layers.GlobalMaxPooling1D()

        max_positions = cfg.max_seq_length + max(
            (len(self.tok.tokenize_words(t)) for t in INSTRUCTION_BANK.values()),
            default=0,
        ) + 8
        self.pos_embed = tf.keras.layers.Embedding(max_positions, self.d_model,
                                                    name="position_embedding")
        self.input_drop = tf.keras.layers.Dropout(cfg.dropout)

        self.blocks = [
            _LIFBlock(self.d_model, cfg.ffn_dim, cfg.snn_decay,
                     cfg.snn_threshold, cfg.snn_surrogate_slope, cfg.dropout,
                     learnable_dynamics=cfg.snn_learnable_dynamics,
                     name=f"lif_block_{i}")
            for i in range(cfg.num_layers)
        ]
        self.final_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self._instruction_cache: Dict[str, Dict[str, List]] = {}
        self._budget_warned: set = set()

    # ------------------------------------------------------------------
    # Tokenisation - identical scheme to InstructionEncoder (model.py)
    # ------------------------------------------------------------------
    def _instruction_encoded(self, instruction: str) -> Dict[str, List]:
        if instruction not in self._instruction_cache:
            self._instruction_cache[instruction] = (
                self.tok.encode(instruction) if instruction
                else {"word_ids": [], "char_ids": [], "words": []}
            )
        return self._instruction_cache[instruction]

    def _sequence_budget(self, inst_len: int) -> int:
        required = inst_len + self.cfg.min_content_tokens
        budget = max(self.cfg.max_seq_length, required)
        if budget > self.cfg.max_seq_length and budget not in self._budget_warned:
            self._budget_warned.add(budget)
            warnings.warn(
                f"Instruction is {inst_len} tokens, so the sequence window was "
                f"grown from {self.cfg.max_seq_length} to {budget}.",
                RuntimeWarning,
            )
        return budget

    def tokenize(self, texts: List[str], instruction: str = "") -> Dict[str, tf.Tensor]:
        inst = self._instruction_encoded(instruction)
        inst_len = len(inst["word_ids"])
        budget = self._sequence_budget(inst_len)
        content_budget = max(8, budget - inst_len)
        C = self.cfg.max_chars_per_token

        rows_words, rows_chars, rows_pool = [], [], []
        for text in texts:
            enc = self.tok.encode(text)
            word_ids = enc["word_ids"][:content_budget]
            char_ids = enc["char_ids"][:content_budget]

            seq_words = inst["word_ids"] + word_ids
            seq_chars = inst["char_ids"] + char_ids

            pool = np.zeros(len(seq_words), dtype=np.int32)
            if self.cfg.pool_over_instruction:
                pool[:] = 1
            else:
                pool[inst_len:inst_len + len(word_ids)] = 1
                if pool.sum() == 0:
                    pool[:] = 1

            rows_words.append(seq_words)
            rows_chars.append(seq_chars)
            rows_pool.append(pool)

        max_len = max((len(r) for r in rows_words), default=1) or 1
        pad_id = self.vocab.pad_id
        char_pad_row = [self.vocab.char_pad_id] * C

        word_ids_arr = np.full((len(texts), max_len), pad_id, dtype=np.int32)
        char_ids_arr = np.full((len(texts), max_len, C), self.vocab.char_pad_id,
                               dtype=np.int32)
        attention = np.zeros((len(texts), max_len), dtype=np.int32)
        pool_mask = np.zeros((len(texts), max_len), dtype=np.int32)

        for i, (words, chars, pool) in enumerate(
                zip(rows_words, rows_chars, rows_pool)):
            n = len(words)
            word_ids_arr[i, :n] = words
            char_ids_arr[i, :n] = chars if chars else char_pad_row
            attention[i, :n] = 1
            pool_mask[i, :n] = pool

        return {
            "word_ids": tf.convert_to_tensor(word_ids_arr),
            "char_ids": tf.convert_to_tensor(char_ids_arr),
            "attention_mask": tf.convert_to_tensor(attention),
            "pool_mask": tf.convert_to_tensor(pool_mask),
        }

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def call(self, inputs: Dict[str, tf.Tensor], training: bool = False):
        word_ids = inputs["word_ids"]
        char_ids = inputs["char_ids"]
        attention_mask = inputs["attention_mask"]

        B = tf.shape(word_ids)[0]
        T = tf.shape(word_ids)[1]
        C = self.cfg.max_chars_per_token

        word_vec = self.word_embed(word_ids)

        char_in = self.char_embed(char_ids)
        char_in = tf.reshape(char_in, [B * T, C, self.cfg.char_embed_dim])
        char_feats = []
        for conv in self.char_convs:
            feat = conv(char_in)
            feat = self.char_pool(feat)
            char_feats.append(feat)
        char_vec = tf.concat(char_feats, axis=-1)
        char_vec = tf.reshape(
            char_vec,
            [B, T, self.cfg.char_cnn_filters * len(self.cfg.char_cnn_kernel_sizes)]
        )

        x = tf.concat([word_vec, char_vec], axis=-1)
        positions = tf.range(T)[tf.newaxis, :]
        x = x + self.pos_embed(positions)
        x = self.input_drop(x, training=training)

        mask_f = tf.cast(attention_mask, x.dtype)
        for block in self.blocks:
            x = block(x, mask_f, training=training)
        hidden = self.final_norm(x)

        pool = tf.cast(inputs["pool_mask"], hidden.dtype)[..., None]
        summed = tf.reduce_sum(hidden * pool, axis=1)
        counts = tf.maximum(tf.reduce_sum(pool, axis=1), 1e-9)
        pooled = summed / counts

        if self.cfg.normalize_embeddings:
            pooled = tf.math.l2_normalize(pooled, axis=-1)
        return pooled

    def encode(self, texts, instruction: str = "", batch_size: int = 64,
              training: bool = False, progress_every: Optional[int] = None
              ) -> np.ndarray:
        vecs = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            batch = self.tokenize(chunk, instruction)
            vecs.append(self(batch, training=training).numpy())
            if progress_every and (start // batch_size) % progress_every == 0:
                done = min(start + batch_size, len(texts))
                print(f"    encoded {done}/{len(texts)}", flush=True)
        if not vecs:
            return np.zeros((0, self.hidden_size), dtype=np.float32)
        return np.concatenate(vecs, axis=0).astype(np.float32)

    @property
    def hidden_size(self) -> int:
        return self.d_model
