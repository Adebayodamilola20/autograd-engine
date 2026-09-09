r"""Turning text into integers, and back.

A language model never sees text. It sees a sequence of integers, each one an
index into an embedding table, and every property of the model is downstream
of how that mapping was chosen.

Two tokenizers live here, and the choice between them is a real trade-off
rather than a matter of taste.

``CharTokenizer``
    One id per distinct character. The vocabulary is tiny (about 65 for
    English prose), nothing is ever out-of-vocabulary, and the code is short
    enough to verify by reading. The cost is that a sequence of ``T`` tokens
    covers only ``T`` characters, so a 128-token context window sees roughly
    twenty words. Attention cost grows with :math:`T^2`, so buying more
    context is expensive.

``BPETokenizer``
    Byte Pair Encoding, the algorithm behind GPT-2, GPT-4 and Llama. Starts
    from raw bytes and repeatedly merges the most frequent adjacent pair into
    a new token, so common sequences (``the``, `` and``) collapse to single
    ids while rare words still decompose into pieces. Typically three to four
    characters per token, so the same context window reaches three to four
    times further into the text.

Why byte-level matters
----------------------
``BPETokenizer`` starts from the 256 possible *bytes*, not from characters.
This is the trick that makes GPT-2's tokenizer total rather than partial: any
byte sequence whatsoever decodes, so emoji, accented text, and mojibake are
all representable without an out-of-vocabulary token. A character-level
vocabulary built from a training corpus has no id for a character it never
saw, and must either crash or silently substitute.

The decode path is where byte-level gets subtle. A multi-byte UTF-8 character
can be split across two tokens, so decoding token-by-token would produce
invalid fragments. Bytes are therefore accumulated across the whole sequence
and decoded once at the end, with ``errors="replace"`` for the genuinely
incomplete tail.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

__all__ = ["CharTokenizer", "BPETokenizer"]


class CharTokenizer:
    """One token per distinct character, in sorted order.

    Sorted rather than first-seen so that the same corpus always produces the
    same vocabulary: a checkpoint is worthless if the ids it was trained on
    shift when the text is reordered.

    Examples
    --------
    >>> tok = CharTokenizer.from_text("hello")
    >>> tok.vocab_size
    4
    >>> tok.decode(tok.encode("hell"))
    'hell'
    """

    def __init__(self, characters: Sequence[str]) -> None:
        self.itos = list(characters)
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        """Unknown characters are dropped rather than crashing a long run."""
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {"kind": "char", "itos": self.itos}

    @classmethod
    def from_dict(cls, payload: dict) -> "CharTokenizer":
        return cls(payload["itos"])


class BPETokenizer:
    r"""Byte Pair Encoding over raw bytes.

    Training
        Start with every byte as its own token (ids 0..255). Count adjacent
        pairs, merge the most frequent into a new id, and repeat until the
        vocabulary reaches the requested size. Each merge is recorded in
        order, because *decoding the same way requires replaying them in
        exactly that order*.

    Encoding
        Apply the learned merges to new text, lowest-numbered merge first.
        Merge order is priority order: a merge learned early was more frequent
        in training, and applying a later merge first would produce a
        different, unlearned segmentation.

    The cost is :math:`O(n)` per merge over the corpus, so training on a few
    megabytes with a few thousand merges is a minute or two. That is fine: it
    happens once, and the result is saved with the model.

    ``vocab_size`` is a ceiling, not a promise. Merging stops early once no
    pair repeats, because a merge seen once is memorising the corpus rather
    than learning its structure. A small sample therefore yields a smaller
    vocabulary than requested, and that is the correct behaviour.

    Examples
    --------
    >>> tok = BPETokenizer.train("the theme of the thesis" * 20, vocab_size=280)
    >>> tok.decode(tok.encode("the thesis")) == "the thesis"
    True
    >>> tok.vocab_size <= 280
    True
    >>> BPETokenizer.train("abc", vocab_size=300).vocab_size   # nothing repeats
    256
    """

    def __init__(self, merges: dict[tuple[int, int], int]) -> None:
        self.merges = merges
        # The inverse table: every id expanded back to the bytes it stands for.
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for (a, b), new_id in merges.items():
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]

    # ------------------------------------------------------------------

    @staticmethod
    def _pair_counts(ids: Sequence[int]) -> Counter:
        return Counter(zip(ids, ids[1:]))

    @staticmethod
    def _merge(ids: Sequence[int], pair: tuple[int, int], new_id: int) -> list[int]:
        out: list[int] = []
        i = 0
        while i < len(ids):
            # `i < len(ids) - 1` guards the final element, which has no
            # successor and therefore cannot start a pair.
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    @classmethod
    def train(cls, text: str, vocab_size: int = 512, *, verbose: bool = False):
        """Learn merges from ``text`` until the vocabulary reaches ``vocab_size``."""
        if vocab_size < 256:
            raise ValueError(
                f"vocab_size must be at least 256 to cover every byte, got {vocab_size}"
            )
        ids = list(text.encode("utf-8"))
        merges: dict[tuple[int, int], int] = {}

        for new_id in range(256, vocab_size):
            counts = cls._pair_counts(ids)
            if not counts:
                break                      # corpus collapsed to a single token
            pair, count = counts.most_common(1)[0]
            if count < 2:
                break                      # nothing repeats; further merges are noise
            ids = cls._merge(ids, pair, new_id)
            merges[pair] = new_id
            if verbose and (new_id - 255) % 100 == 0:
                print(f"  merge {new_id - 255}: {pair} -> {new_id} ({count} times)")

        return cls(merges)

    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        # Repeatedly apply whichever learned merge has the lowest id, which is
        # the one learned earliest and therefore the most frequent.
        while len(ids) >= 2:
            counts = self._pair_counts(ids)
            candidates = [p for p in counts if p in self.merges]
            if not candidates:
                break
            pair = min(candidates, key=lambda p: self.merges[p])
            ids = self._merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        # Concatenate all the bytes first, decode once. Decoding per token
        # would split multi-byte UTF-8 characters and fail on the halves.
        raw = b"".join(self.vocab[int(i)] for i in ids)
        return raw.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        # JSON keys must be strings, and tuples are not valid keys at all.
        return {
            "kind": "bpe",
            "merges": [[a, b, new] for (a, b), new in self.merges.items()],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "BPETokenizer":
        return cls({(a, b): new for a, b, new in payload["merges"]})


# ======================================================================


def load_tokenizer(payload: dict):
    """Rebuild whichever tokenizer produced ``payload``."""
    kind = payload.get("kind")
    if kind == "char":
        return CharTokenizer.from_dict(payload)
    if kind == "bpe":
        return BPETokenizer.from_dict(payload)
    raise ValueError(f"unknown tokenizer kind {kind!r}")


def encode_corpus(tokenizer, text: str) -> np.ndarray:
    """Encode a whole document to a flat array of ids, ready for batching."""
    return np.asarray(tokenizer.encode(text), dtype=np.intp)


def save_tokenizer(tokenizer, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(tokenizer.to_dict()), encoding="utf-8")
    return p
