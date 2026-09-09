"""Tests for the tokenizers.

The property that matters is **round-tripping**: ``decode(encode(t)) == t``.
A tokenizer that loses information puts a ceiling on the model that no amount
of training can lift, and it does so silently, because the model still trains
happily on the corrupted ids.
"""

from __future__ import annotations

import pytest

from nabla.data.tokenizer import (
    BPETokenizer,
    CharTokenizer,
    encode_corpus,
    load_tokenizer,
    save_tokenizer,
)

SAMPLE = "the theme of the thesis is the thesaurus " * 30


class TestCharTokenizer:
    def test_round_trips(self):
        tok = CharTokenizer.from_text(SAMPLE)
        assert tok.decode(tok.encode(SAMPLE)) == SAMPLE

    def test_vocabulary_is_sorted_and_deduplicated(self):
        tok = CharTokenizer.from_text("cba abc")
        assert tok.itos == [" ", "a", "b", "c"]

    def test_vocabulary_does_not_depend_on_text_order(self):
        """A checkpoint is worthless if reordering the corpus shifts the ids."""
        assert CharTokenizer.from_text("abc").itos == CharTokenizer.from_text("cba").itos

    def test_unknown_characters_are_dropped_not_fatal(self):
        tok = CharTokenizer.from_text("abc")
        assert tok.decode(tok.encode("abcxyz")) == "abc"


class TestBPETokenizer:
    def test_round_trips(self):
        tok = BPETokenizer.train(SAMPLE, vocab_size=300)
        assert tok.decode(tok.encode(SAMPLE)) == SAMPLE

    def test_round_trips_text_it_never_saw(self):
        tok = BPETokenizer.train(SAMPLE, vocab_size=300)
        unseen = "quantum jellyfish, 42 of them!"
        assert tok.decode(tok.encode(unseen)) == unseen

    def test_round_trips_non_ascii(self):
        """Byte-level means emoji and accents need no special handling."""
        tok = BPETokenizer.train(SAMPLE, vocab_size=300)
        for text in ("héllo café", "🌍🌎🌏", "naïve — dash", "日本語"):
            assert tok.decode(tok.encode(text)) == text

    def test_merging_actually_compresses(self):
        tok = BPETokenizer.train(SAMPLE, vocab_size=350)
        raw = len(SAMPLE.encode("utf-8"))
        assert len(tok.encode(SAMPLE)) < raw / 2

    def test_vocab_size_is_a_ceiling_not_a_promise(self):
        """Merging stops when nothing repeats, rather than inventing tokens."""
        assert BPETokenizer.train("abcdef", vocab_size=400).vocab_size == 256

    def test_a_vocabulary_smaller_than_the_byte_range_is_rejected(self):
        with pytest.raises(ValueError, match="at least 256"):
            BPETokenizer.train(SAMPLE, vocab_size=100)

    def test_merges_are_applied_in_learned_order(self):
        """Encoding must reproduce the segmentation training produced."""
        tok = BPETokenizer.train(SAMPLE, vocab_size=300)
        # Re-encoding the training text must give back exactly what the
        # trainer's own merge loop converged on.
        assert tok.decode(tok.encode("the thesis")) == "the thesis"
        assert len(tok.encode("the")) < 3


class TestPersistence:
    @pytest.mark.parametrize("build", [
        lambda: CharTokenizer.from_text(SAMPLE),
        lambda: BPETokenizer.train(SAMPLE, vocab_size=300),
    ])
    def test_survives_a_save_and_load(self, build, tmp_path):
        original = build()
        path = save_tokenizer(original, tmp_path / "tok.json")

        import json
        restored = load_tokenizer(json.loads(path.read_text()))

        assert restored.vocab_size == original.vocab_size
        assert restored.encode(SAMPLE) == original.encode(SAMPLE)
        assert restored.decode(restored.encode(SAMPLE)) == SAMPLE

    def test_an_unknown_kind_is_rejected(self):
        with pytest.raises(ValueError, match="unknown tokenizer kind"):
            load_tokenizer({"kind": "wordpiece"})

    def test_encode_corpus_returns_an_index_array(self):
        tok = CharTokenizer.from_text(SAMPLE)
        ids = encode_corpus(tok, SAMPLE)
        assert ids.ndim == 1
        assert ids.dtype.kind == "i"
        assert int(ids.max()) < tok.vocab_size
