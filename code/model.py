"""
SCRUB: Self-supervised Contrastive Record Understanding, from-scratch.

The from-scratch instruction-conditioned encoder for this project. No
pretrained backbone, no external weights, no HF `transformers` dependency.
Every trainable parameter here starts from random initialisation and is
learned only from this project's own contrastive training signal - including
the word embeddings, which is what lets a literal-text instruction (rather
than a learned indicator vector) still work without any pretraining: the model
has to learn what "abbreviation" or "capitalisation" predict purely from
co-occurring with those noise patterns in the training pairs. That is a much
narrower kind of "understanding" than a pretrained language model has, and
results should be read with that in mind.

Architecture, per record:

    word ids   -> trainable word embedding  ---\
                                                  concat -> +pos embed -> N x
    char ids   -> char embedding -> char-CNN --/           (pre-norm
                  (case/punctuation-preserving,              self-attention
                   gives OOV & typo robustness                + FFN) blocks
                   independent of word identity)            -> masked mean
                                                                 pool
                                                              -> L2 normalise

Instruction and content are tokenised separately so the record always gets a
guaranteed `min_content_tokens` budget (see `_sequence_budget`), and pooling
masks out the instruction positions by default so a long instruction can't
dominate the pooled vector - the same asymmetric design as before, just
implemented over a vocabulary this project built itself instead of a
pretrained tokenizer.
"""
from __future__ import annotations

import sys
import warnings
from typing import Dict, List, Optional, Union

import numpy as np
import tensorflow as tf

from config import ExperimentConfig
from data import AddressData
from instructions import INSTRUCTION_BANK
from tokenizer import Tokenizer, Vocab


# --------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------
def _load_or_build_vocab(cfg: ExperimentConfig, data: AddressData) -> Vocab:
    import os

    cfg.make_dirs()
    if os.path.exists(cfg.vocab_path):
        vocab = Vocab.load(cfg.vocab_path)
        print(f"[vocab] loaded {len(vocab)} tokens from {cfg.vocab_path}")
        return vocab

    train_split = data.splits["train"]
    corpus_texts = train_split.dirty_texts + train_split.clean_texts
    # The full instruction bank, not just this run's variant: a shared vocab
    # has to cover every instruction key so it can serve any variant (e.g. a
    # frozen encoder reused across a zero-shot ablation sweep).
    instruction_texts = list(INSTRUCTION_BANK.values())

    vocab = Vocab.build(corpus_texts, instruction_texts,
                        min_freq=cfg.vocab_min_freq)
    vocab.save(cfg.vocab_path)
    print(f"[vocab] built {len(vocab)} tokens from {len(corpus_texts)} "
          f"training records + {len(instruction_texts)} instruction "
          f"variants -> {cfg.vocab_path}")
    return vocab


# --------------------------------------------------------------------------
# Encoder
# --------------------------------------------------------------------------
class _TransformerBlock(tf.keras.layers.Layer):
    """Pre-norm self-attention + FFN block. Plain Keras layers, no HF code."""

    def __init__(self, d_model: int, num_heads: int, ffn_dim: int,
                dropout: float, **kwargs):
        super().__init__(**kwargs)
        self.attn_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout
        )
        self.attn_drop = tf.keras.layers.Dropout(dropout)

        self.ffn_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ffn_dim, activation="gelu"),
            tf.keras.layers.Dense(d_model),
        ])
        self.ffn_drop = tf.keras.layers.Dropout(dropout)

    def call(self, x, key_mask, training: bool = False):
        h = self.attn_norm(x)
        attn_out = self.attn(query=h, value=h, key=h,
                             attention_mask=key_mask, training=training)
        x = x + self.attn_drop(attn_out, training=training)

        h = self.ffn_norm(x)
        x = x + self.ffn_drop(self.ffn(h, training=training), training=training)
        return x


# --------------------------------------------------------------------------
# v3 block: RMSNorm + RoPE attention + SwiGLU FFN (LLaMA-style pre-norm block)
# --------------------------------------------------------------------------
class RMSNorm(tf.keras.layers.Layer):
    """Root-mean-square layer norm: no mean-centering, no bias - cheaper than
    LayerNorm and the standard choice in RoPE-based Transformers (LLaMA, T5)."""

    def __init__(self, d_model: int, eps: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.scale = self.add_weight(
            name="scale", shape=(d_model,), initializer="ones", trainable=True
        )

    def call(self, x):
        x = tf.cast(x, tf.float32)
        variance = tf.reduce_mean(tf.square(x), axis=-1, keepdims=True)
        x = x * tf.math.rsqrt(variance + self.eps)
        return x * self.scale


def _rope_sin_cos(seq_len: int, head_dim: int, base: float = 10000.0):
    """Precompute the rotary sin/cos tables for a sequence of length seq_len.

    Standard RoPE: pairs up adjacent dims of each head and rotates them by an
    angle that grows with position, so relative position is encoded directly
    in the attention dot product instead of added to the input like a learned
    position embedding.
    """
    inv_freq = 1.0 / (base ** (tf.range(0, head_dim, 2, dtype=tf.float32) / head_dim))
    positions = tf.range(seq_len, dtype=tf.float32)
    freqs = tf.einsum("i,j->ij", positions, inv_freq)          # [T, head_dim/2]
    emb = tf.concat([freqs, freqs], axis=-1)                    # [T, head_dim]
    return tf.sin(emb), tf.cos(emb)


def _rotate_half(x):
    x1, x2 = tf.split(x, 2, axis=-1)
    return tf.concat([-x2, x1], axis=-1)


def _apply_rope(x, sin, cos):
    # x: [B, H, T, head_dim]; sin/cos: [T, head_dim] -> broadcast over B, H
    sin = sin[tf.newaxis, tf.newaxis, :, :]
    cos = cos[tf.newaxis, tf.newaxis, :, :]
    return x * cos + _rotate_half(x) * sin


class _RoPESelfAttention(tf.keras.layers.Layer):
    """Multi-head self-attention with rotary position embeddings applied to
    Q/K, implemented by hand since tf.keras.layers.MultiHeadAttention has no
    hook to inject RoPE before the dot product."""

    def __init__(self, d_model: int, num_heads: int, dropout: float, **kwargs):
        super().__init__(**kwargs)
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = tf.keras.layers.Dense(d_model, use_bias=False, name="q_proj")
        self.k_proj = tf.keras.layers.Dense(d_model, use_bias=False, name="k_proj")
        self.v_proj = tf.keras.layers.Dense(d_model, use_bias=False, name="v_proj")
        self.out_proj = tf.keras.layers.Dense(d_model, use_bias=False, name="out_proj")
        self.drop = tf.keras.layers.Dropout(dropout)

    def _split_heads(self, x, B, T):
        x = tf.reshape(x, [B, T, self.num_heads, self.head_dim])
        return tf.transpose(x, [0, 2, 1, 3])                    # [B,H,T,hd]

    def call(self, x, key_mask, sin, cos, training: bool = False):
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]

        q = self._split_heads(self.q_proj(x), B, T)
        k = self._split_heads(self.k_proj(x), B, T)
        v = self._split_heads(self.v_proj(x), B, T)

        q = _apply_rope(q, sin, cos)
        k = _apply_rope(k, sin, cos)

        scale = tf.cast(1.0 / tf.sqrt(tf.cast(self.head_dim, tf.float32)), q.dtype)
        logits = tf.matmul(q, k, transpose_b=True) * scale       # [B,H,T,T]

        # key_mask: [B,1,T] boolean, True = attend. Broadcasts over query dim.
        # -1e9 overflows float16's range (~65504 max), which mixed_precision
        # can put logits.dtype in - use the dtype's own -max instead so the
        # mask still zeroes softmax weight without producing inf/NaN.
        mask = key_mask[:, :, tf.newaxis, :]                     # [B,1,1,T]
        neg_fill = tf.constant(logits.dtype.min, dtype=logits.dtype)
        logits = tf.where(mask, logits, neg_fill)

        weights = tf.nn.softmax(logits, axis=-1)
        weights = self.drop(weights, training=training)

        out = tf.matmul(weights, v)                              # [B,H,T,hd]
        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, [B, T, self.num_heads * self.head_dim])
        return self.out_proj(out)


class _SwiGLU_FFN(tf.keras.layers.Layer):
    """SwiGLU feed-forward: gate = Swish(x W_gate), out = (gate * x W_up) W_down.
    Standard in modern (LLaMA/PaLM-style) Transformers as a GELU-FFN swap-in;
    the hidden width is shrunk ~2/3 to keep parameter count comparable to a
    plain 1x-expansion GELU FFN of the same nominal `ffn_dim`."""

    def __init__(self, d_model: int, ffn_dim: int, **kwargs):
        super().__init__(**kwargs)
        hidden = max(8, int(ffn_dim * 2 / 3))
        self.gate_proj = tf.keras.layers.Dense(hidden, use_bias=False, name="gate_proj")
        self.up_proj = tf.keras.layers.Dense(hidden, use_bias=False, name="up_proj")
        self.down_proj = tf.keras.layers.Dense(d_model, use_bias=False, name="down_proj")

    def call(self, x):
        gate = tf.nn.silu(self.gate_proj(x))
        return self.down_proj(gate * self.up_proj(x))


class _RoPETransformerBlock(tf.keras.layers.Layer):
    """Pre-norm block: RMSNorm -> RoPE attention -> residual,
    RMSNorm -> SwiGLU FFN -> residual. The v3 architecture variant."""

    def __init__(self, d_model: int, num_heads: int, ffn_dim: int,
                dropout: float, **kwargs):
        super().__init__(**kwargs)
        self.attn_norm = RMSNorm(d_model)
        self.attn = _RoPESelfAttention(d_model, num_heads, dropout, name="self_attention")
        self.attn_drop = tf.keras.layers.Dropout(dropout)

        self.ffn_norm = RMSNorm(d_model)
        self.ffn = _SwiGLU_FFN(d_model, ffn_dim, name="swiglu_ffn")
        self.ffn_drop = tf.keras.layers.Dropout(dropout)

    def call(self, x, key_mask, sin, cos, training: bool = False):
        h = self.attn_norm(x)
        attn_out = self.attn(h, key_mask, sin, cos, training=training)
        x = x + self.attn_drop(attn_out, training=training)

        h = self.ffn_norm(x)
        x = x + self.ffn_drop(self.ffn(h), training=training)
        return x


class InstructionEncoder(tf.keras.Model):
    """Encodes text with an optional prepended instruction.

    Same public surface as the pipeline expects (`tokenize`, `__call__`,
    `encode`, `hidden_size`, `.trainable_variables`, checkpoint-compatible),
    so train.py / evaluate.py / run_ablation.py need no changes to use it.
    """

    def __init__(self, cfg: ExperimentConfig, vocab: Vocab, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.vocab = vocab
        self.tok = Tokenizer(vocab, max_chars_per_token=cfg.max_chars_per_token)

        char_output_dim = cfg.char_cnn_filters * len(cfg.char_cnn_kernel_sizes)
        concat_dim = cfg.token_embed_dim + char_output_dim

        # v3 architecture (use_rope/use_rmsnorm/use_swiglu): the word+char
        # concat is projected to a separate model_dim so the block width isn't
        # tied to token_embed_dim + char_cnn output size. v1/v2 (all three
        # flags False, the default) keep the original concat-is-d_model
        # behaviour so existing configs/checkpoints are unaffected.
        self.use_rope = cfg.use_rope
        self.use_rmsnorm = cfg.use_rmsnorm
        self.use_swiglu = cfg.use_swiglu
        self._is_v3 = self.use_rope or self.use_rmsnorm or self.use_swiglu

        if self._is_v3:
            self.d_model = cfg.model_dim or concat_dim
            self.input_proj = tf.keras.layers.Dense(
                self.d_model, name="input_projection"
            )
        else:
            self.d_model = concat_dim
            self.input_proj = None

        if self.d_model % cfg.num_heads != 0:
            raise ValueError(
                f"resolved d_model ({self.d_model}) must be divisible by "
                f"num_heads ({cfg.num_heads})"
            )

        self.word_embed = tf.keras.layers.Embedding(
            len(vocab), cfg.token_embed_dim, name="word_embedding"
        )
        self.char_embed = tf.keras.layers.Embedding(
            vocab.num_chars, cfg.char_embed_dim, name="char_embedding"
        )
        # One Conv1D per kernel size, run in parallel and concatenated - a
        # learned analogue of TF-IDF's 3-5 character n-gram range, rather than
        # a single kernel size guessing at one n-gram length.
        self.char_convs = [
            tf.keras.layers.Conv1D(
                cfg.char_cnn_filters, kernel_size=k, padding="same",
                activation="relu", name=f"char_cnn_k{k}"
            )
            for k in cfg.char_cnn_kernel_sizes
        ]
        self.char_pool = tf.keras.layers.GlobalMaxPooling1D()

        # Longest possible sequence budget: every instruction's token count
        # plus the full content budget. Used to size the learned position
        # embedding (v1/v2) or precompute the RoPE sin/cos table (v3).
        self._max_positions = cfg.max_seq_length + max(
            (len(self.tok.tokenize_words(t)) for t in INSTRUCTION_BANK.values()),
            default=0,
        ) + 8

        if self.use_rope:
            self.pos_embed = None
            head_dim = self.d_model // cfg.num_heads
            sin, cos = _rope_sin_cos(self._max_positions, head_dim)
            self._rope_sin = tf.constant(sin)
            self._rope_cos = tf.constant(cos)
        else:
            self.pos_embed = tf.keras.layers.Embedding(
                self._max_positions, self.d_model, name="position_embedding"
            )

        self.input_drop = tf.keras.layers.Dropout(cfg.dropout)

        if self._is_v3:
            self.blocks = [
                _RoPETransformerBlock(self.d_model, cfg.num_heads, cfg.ffn_dim,
                                      cfg.dropout, name=f"block_{i}")
                for i in range(cfg.num_layers)
            ]
        else:
            self.blocks = [
                _TransformerBlock(self.d_model, cfg.num_heads, cfg.ffn_dim,
                                  cfg.dropout, name=f"block_{i}")
                for i in range(cfg.num_layers)
            ]
        self.final_norm = (RMSNorm(self.d_model) if self.use_rmsnorm
                           else tf.keras.layers.LayerNormalization(epsilon=1e-6))

        self._instruction_cache: Dict[str, Dict[str, List]] = {}
        self._budget_warned: set = set()

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------
    def _instruction_encoded(self, instruction: str) -> Dict[str, List]:
        if instruction not in self._instruction_cache:
            self._instruction_cache[instruction] = (
                self.tok.encode(instruction) if instruction
                else {"word_ids": [], "char_ids": [], "words": []}
            )
        return self._instruction_cache[instruction]

    def _sequence_budget(self, inst_len: int) -> int:
        """Total sequence length for this instruction.

        max_seq_length is a floor, not a cap: the window grows so the record
        always keeps at least `min_content_tokens`, so a long instruction
        can't crowd out the address and confound the length-vs-content
        comparison the ablation depends on.
        """
        required = inst_len + self.cfg.min_content_tokens
        budget = max(self.cfg.max_seq_length, required)
        if budget > self.cfg.max_seq_length and budget not in self._budget_warned:
            self._budget_warned.add(budget)
            warnings.warn(
                f"Instruction is {inst_len} tokens, so the sequence window was "
                f"grown from {self.cfg.max_seq_length} to {budget} to keep "
                f"{self.cfg.min_content_tokens} tokens for the record.",
                RuntimeWarning,
            )
        return budget

    def tokenize(self,
                texts: List[str],
                instruction: Union[str, List[str]] = "") -> Dict[str, tf.Tensor]:
        """Build padded word_ids / char_ids / attention_mask / pool_mask.

        pool_mask is attention_mask with the instruction positions zeroed
        (unless pool_over_instruction=True), so mean pooling averages content
        tokens only.

        `instruction` is either one string shared by every row in `texts`
        (the original behaviour: every noise_* variant uses this), or a list
        of one instruction PER ROW, same length as `texts` (used by the
        record_specific / record_shuffled variants in instructions.py, whose
        instruction depends on which noise types are actually present in
        that specific row - see AddressData.per_record_instructions). The
        sequence budget is computed from the longest instruction in this
        call either way, so every row in the batch still gets at least
        `min_content_tokens` for its own record text.
        """
        if isinstance(instruction, str):
            instructions = [instruction] * len(texts)
        else:
            instructions = list(instruction)
            if len(instructions) != len(texts):
                raise ValueError(
                    f"per-row instruction list has {len(instructions)} "
                    f"entries but there are {len(texts)} texts"
                )

        insts = [self._instruction_encoded(ins) for ins in instructions]
        inst_lens = [len(i["word_ids"]) for i in insts]
        max_inst_len = max(inst_lens, default=0)
        budget = self._sequence_budget(max_inst_len)
        C = self.cfg.max_chars_per_token

        rows_words, rows_chars, rows_pool = [], [], []
        for text, inst, inst_len in zip(texts, insts, inst_lens):
            content_budget = max(8, budget - inst_len)
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
                if pool.sum() == 0:                # degenerate: empty text
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

        word_vec = self.word_embed(word_ids)                    # [B,T,E]

        char_in = self.char_embed(char_ids)                     # [B,T,C,Ec]
        char_in = tf.reshape(char_in, [B * T, C, self.cfg.char_embed_dim])
        char_feats = []
        for conv in self.char_convs:
            feat = conv(char_in)                                # [B*T,C,F]
            feat = self.char_pool(feat)                         # [B*T,F]
            char_feats.append(feat)
        char_vec = tf.concat(char_feats, axis=-1)                # [B*T, F*len(kernels)]
        char_vec = tf.reshape(
            char_vec,
            [B, T, self.cfg.char_cnn_filters * len(self.cfg.char_cnn_kernel_sizes)]
        )

        x = tf.concat([word_vec, char_vec], axis=-1)             # [B,T,concat_dim]
        if self.input_proj is not None:
            x = self.input_proj(x)                                # [B,T,d_model]

        if self.use_rope:
            sin = self._rope_sin[:T]
            cos = self._rope_cos[:T]
        else:
            positions = tf.range(T)[tf.newaxis, :]
            x = x + self.pos_embed(positions)
        x = self.input_drop(x, training=training)

        key_mask = tf.cast(attention_mask, tf.bool)[:, tf.newaxis, :]  # [B,1,T]
        for block in self.blocks:
            if self._is_v3:
                x = block(x, key_mask, sin, cos, training=training)
            else:
                x = block(x, key_mask, training=training)
        hidden = self.final_norm(x)

        mask = tf.cast(inputs["pool_mask"], hidden.dtype)[..., None]
        summed = tf.reduce_sum(hidden * mask, axis=1)
        counts = tf.maximum(tf.reduce_sum(mask, axis=1), 1e-9)
        pooled = summed / counts

        if self.cfg.normalize_embeddings:
            pooled = tf.math.l2_normalize(pooled, axis=-1)
        return pooled

    # ------------------------------------------------------------------
    # Convenience: encode a python list of strings end to end
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
        return self.d_model


# --------------------------------------------------------------------------
# Devices (generic TF setup, unrelated to the encoder architecture)
# --------------------------------------------------------------------------
_WINDOWS_GPU_NOTE = (
    "TensorFlow dropped native Windows GPU support after 2.10, so any "
    "tensorflow >= 2.11 on native Windows is CPU-only no matter what is "
    "installed. To use an NVIDIA GPU from Windows, run this project inside "
    "WSL2 (`wsl --install`, then `uv sync --extra gpu` in the WSL shell)."
)


def configure_devices(cfg: ExperimentConfig) -> list:
    """Report and configure compute devices.

    Prints what TF can actually see, so a run that silently fell back to CPU is
    obvious in the log rather than only in the wall-clock time.
    """
    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        if cfg.gpu_memory_growth:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError:
                    pass                       # already initialised; harmless
        for i, gpu in enumerate(gpus):
            try:
                details = tf.config.experimental.get_device_details(gpu)
                name = details.get("device_name", "unknown")
                cc = details.get("compute_capability")
                suffix = f", compute capability {cc[0]}.{cc[1]}" if cc else ""
            except Exception:
                name, suffix = "unknown", ""
            print(f"[device] GPU {i}: {name}{suffix}")
        print(f"[device] using {len(gpus)} GPU(s), "
              f"memory_growth={cfg.gpu_memory_growth}")
    else:
        message = "[device] no GPU visible to TensorFlow - running on CPU"
        if sys.platform.startswith("win"):
            message += f"\n[device] {_WINDOWS_GPU_NOTE}"
        print(message)
        if cfg.require_gpu:
            raise RuntimeError(
                "require_gpu=true but TensorFlow sees no GPU.\n"
                + (_WINDOWS_GPU_NOTE if sys.platform.startswith("win") else
                   "On Linux install the CUDA wheels with "
                   "`uv sync --extra gpu`, and check `nvidia-smi` works.")
            )
    return gpus


def build_encoder(cfg: ExperimentConfig, data: AddressData) -> InstructionEncoder:
    """Build (or restore the vocabulary for) a from-scratch encoder.

    `data` is required: the vocabulary is built from the training split's
    own text plus the instruction bank, and is cached to `cfg.vocab_path` so a
    resumed run or a later `evaluate.py --run <name>` reload the exact
    vocabulary a checkpoint was trained with, rather than silently rebuilding
    a different one.
    """
    configure_devices(cfg)

    if cfg.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    tf.random.set_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if cfg.encoder_type == "pretrained":
        from model_pretrained import PretrainedInstructionEncoder
        encoder = PretrainedInstructionEncoder(cfg)
    else:
        vocab = _load_or_build_vocab(cfg, data)
        if cfg.encoder_type == "snn":
            from model_snn import SCRUBSNNEncoder
            encoder = SCRUBSNNEncoder(cfg, vocab)
        elif cfg.encoder_type == "transformer":
            encoder = InstructionEncoder(cfg, vocab)
        else:
            raise ValueError(
                f"Unknown encoder_type '{cfg.encoder_type}'. "
                f"Available: 'transformer', 'snn', 'pretrained'"
            )

    # One tiny forward pass so all variables exist before a checkpoint is
    # restored into them.
    warm = encoder.tokenize(["name: warmup | road: main street"], "")
    encoder(warm, training=False)

    # The pretrained-backbone encoder's own trainable_variables can be
    # unreliable (see model_pretrained.py's module docstring on the Keras
    # 3 / tf_keras auto-tracking mismatch for a from_config()-built,
    # pretrained_random_init backbone) - go straight to the backbone.
    trainable_vars = (encoder.backbone.trainable_variables
                      if cfg.encoder_type == "pretrained"
                      else encoder.trainable_variables)
    n_params = sum(int(tf.size(v)) for v in trainable_vars)
    if cfg.encoder_type == "pretrained":
        print(f"[model] encoder       : pretrained backbone "
              f"'{cfg.pretrained_model_name}' (trainable={cfg.pretrained_trainable}) "
              f"- side experiment, not SCRUB itself")
        print(f"[model] hidden size   : {encoder.hidden_size}")
        print(f"[model] parameters    : {n_params:,} "
              f"({'all trainable' if cfg.pretrained_trainable else 'backbone frozen'})")
        print(f"[model] pooling       : "
              f"{'all tokens' if cfg.pool_over_instruction else 'content tokens only'}")
        return encoder
    if cfg.encoder_type == "snn":
        dyn = ("learnable per-channel, init " if cfg.snn_learnable_dynamics
              else "fixed ")
        print(f"[model] encoder       : SCRUB-SNN, from-scratch, {cfg.num_layers} "
              f"spiking LIF blocks ({dyn}decay={cfg.snn_decay}, "
              f"threshold={cfg.snn_threshold})")
    elif encoder._is_v3:
        print(f"[model] encoder       : SCRUB v3, from-scratch, {cfg.num_layers} "
              f"RoPE+RMSNorm+SwiGLU blocks, {cfg.num_heads} heads")
    else:
        print(f"[model] encoder       : SCRUB, from-scratch, {cfg.num_layers} layers, "
              f"{cfg.num_heads} heads")
    char_dim = cfg.char_cnn_filters * len(cfg.char_cnn_kernel_sizes)
    concat_dim = cfg.token_embed_dim + char_dim
    if cfg.encoder_type == "transformer" and encoder._is_v3:
        print(f"[model] hidden size   : {encoder.hidden_size} "
              f"(projected from {cfg.token_embed_dim} word + {char_dim} "
              f"char-CNN = {concat_dim}, kernels={cfg.char_cnn_kernel_sizes})")
    else:
        print(f"[model] hidden size   : {encoder.hidden_size} "
              f"({cfg.token_embed_dim} word + {char_dim} char-CNN, "
              f"kernels={cfg.char_cnn_kernel_sizes})")
    print(f"[model] vocabulary    : {len(vocab)} word tokens, "
          f"{vocab.num_chars} chars")
    print(f"[model] parameters    : {n_params:,} (all trainable, all random-init)")
    print(f"[model] pooling       : "
          f"{'all tokens' if cfg.pool_over_instruction else 'content tokens only'}")
    return encoder
