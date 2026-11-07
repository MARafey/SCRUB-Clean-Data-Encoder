"""
From-scratch vocabulary and tokenizer.

No pretrained tokenizer, no external corpus. The vocabulary is built only from
this project's own data: the training split's record text plus every word that
appears in the instruction bank. Instruction words are kept regardless of
frequency, because they are rare in the address corpus by construction but the
model has no other way to ever learn what they mean - if they are dropped as
"too rare", literal-text prompt conditioning cannot work at all.

Tokenisation is case- and punctuation-preserving. Normalising here (as
data.classify_cell does for *labelling* noise) would erase the exact surface
signal - case, punctuation, spacing - the model is supposed to learn to handle.
Case/punctuation robustness instead comes from the character-CNN half of the
encoder in model.py, which sees the raw token string.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Dict, List, Sequence

PAD, UNK = "[PAD]", "[UNK]"

# A "word" is a run of letters/digits; anything else (punctuation, symbols) is
# its own single-character token. This keeps the exact character that carries
# punctuation-noise information (an apostrophe, a hyphen) visible as a token,
# rather than merging it into an adjoining word or dropping it.
_TOKEN_RE = re.compile(r"[^\W\d_]+|\d+|[^\s\w]", re.UNICODE)


def split_words(text: str) -> List[str]:
    return _TOKEN_RE.findall(text or "")


class Vocab:
    """Word -> id lookup, plus the character alphabet for the char-CNN."""

    def __init__(self, token_to_id: Dict[str, int], char_to_id: Dict[str, int]):
        self.token_to_id = token_to_id
        self.char_to_id = char_to_id
        self.id_to_token = {i: t for t, i in token_to_id.items()}

    def __len__(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK]

    @property
    def char_pad_id(self) -> int:
        return self.char_to_id[PAD]

    @property
    def num_chars(self) -> int:
        return len(self.char_to_id)

    def encode_word(self, word: str) -> int:
        return self.token_to_id.get(word.lower(), self.unk_id)

    def encode_chars(self, word: str, max_chars: int) -> List[int]:
        ids = [self.char_to_id.get(c, self.char_to_id[UNK]) for c in word[:max_chars]]
        ids += [self.char_pad_id] * (max_chars - len(ids))
        return ids

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"token_to_id": self.token_to_id,
                      "char_to_id": self.char_to_id}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "Vocab":
        with open(path, "r", encoding="utf-8") as f:
            blob = json.load(f)
        return cls(blob["token_to_id"], blob["char_to_id"])

    @classmethod
    def build(cls,
             corpus_texts: Sequence[str],
             instruction_texts: Sequence[str],
             min_freq: int = 2) -> "Vocab":
        """Build a vocabulary from this project's own data only.

        corpus_texts: training-split record text (dirty + clean). Words below
            `min_freq` are dropped - typos and one-off tokens fall back to
            [UNK], which is fine because the char-CNN still sees them.
        instruction_texts: every string in instructions.INSTRUCTION_BANK.
            Kept unconditionally: these words are what makes a prompt legible
            to the model at all, and by construction they are far too rare in
            the address corpus to survive a frequency cutoff on their own.
        """
        counts = Counter()
        for text in corpus_texts:
            counts.update(w.lower() for w in split_words(text))

        instruction_words = set()
        for text in instruction_texts:
            instruction_words.update(w.lower() for w in split_words(text))

        kept = {w for w, c in counts.items() if c >= min_freq}
        kept |= instruction_words

        specials = [PAD, UNK]
        token_to_id = {t: i for i, t in enumerate(specials)}
        for w in sorted(kept):                    # sorted: deterministic ids
            token_to_id[w] = len(token_to_id)

        chars = set()
        for text in list(corpus_texts) + list(instruction_texts):
            chars.update(text)
        char_specials = [PAD, UNK]
        char_to_id = {c: i for i, c in enumerate(char_specials)}
        for c in sorted(chars):
            char_to_id[c] = len(char_to_id)

        return cls(token_to_id, char_to_id)


class Tokenizer:
    """Turns raw text into word ids + per-word character ids."""

    def __init__(self, vocab: Vocab, max_chars_per_token: int = 16):
        self.vocab = vocab
        self.max_chars_per_token = max_chars_per_token

    def tokenize_words(self, text: str) -> List[str]:
        return split_words(text)

    def encode(self, text: str) -> Dict[str, List]:
        """One text -> {word_ids, char_ids} without padding to a batch."""
        words = self.tokenize_words(text)
        return {
            "word_ids": [self.vocab.encode_word(w) for w in words],
            "char_ids": [self.vocab.encode_chars(w, self.max_chars_per_token)
                        for w in words],
            "words": words,
        }
