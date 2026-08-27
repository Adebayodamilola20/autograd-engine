"""Tests for the data pipeline: ``Dataset``, ``DataLoader``, splitting, one-hot.

This module had no tests until Phase 19, which is exactly why the performance
bug lived there. The engine was covered in obsessive detail and the plumbing
around it was not -- so the plumbing was where 39% of every epoch was being
spent, unnoticed.

The correctness properties that matter here are not mathematical, they are
*bookkeeping*: every sample appears exactly once per epoch, batches do not
alias the dataset, shuffling is reproducible under a seed, and a stratified
split really does preserve class balance. Each of those, if broken, produces a
model that trains but is quietly wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from nabla.data import DataLoader, train_val_split
from nabla.data.dataset import Dataset, one_hot


@pytest.fixture
def data():
    """40 samples, 3 features, 4 balanced classes."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 3))
    y = np.repeat(np.arange(4), 10)
    return Dataset(x, y, name="fixture")


# ======================================================================
# Dataset
# ======================================================================


class TestDataset:
    def test_reports_length_features_and_classes(self, data):
        assert len(data) == 40
        assert data.n_features == 3
        assert data.n_classes == 4

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="rows"):
            Dataset(np.zeros((5, 2)), np.zeros(4))

    def test_rejects_non_2d_features(self):
        with pytest.raises(ValueError, match="2-D"):
            Dataset(np.zeros(10), np.zeros(10))

    def test_getitem_returns_plain_python(self, data):
        """The scalar engine wants floats and ints, not NumPy scalars."""
        xs, y = data[0]
        assert isinstance(xs, list)
        assert all(isinstance(v, float) for v in xs)
        assert isinstance(y, int)

    def test_coerces_features_to_float64(self):
        d = Dataset(np.ones((4, 2), dtype=np.int8), np.zeros(4))
        assert d.x.dtype == np.float64

    def test_class_counts_are_correct(self, data):
        assert data.class_counts() == {0: 10, 1: 10, 2: 10, 3: 10}

    def test_subset_is_smaller_and_reproducible(self, data):
        a = data.subset(12, seed=1)
        b = data.subset(12, seed=1)
        assert len(a) == 12
        assert np.array_equal(a.x, b.x)

    def test_subset_larger_than_dataset_returns_everything(self, data):
        assert data.subset(999) is data

    def test_subset_draws_without_replacement(self, data):
        """Sampling with replacement would silently duplicate training samples."""
        sub = data.subset(30, seed=3)
        # Every row of the fixture is distinct, so duplicates are detectable.
        assert len({tuple(row) for row in sub.x.tolist()}) == 30


# ======================================================================
# DataLoader
# ======================================================================


class TestDataLoader:
    def test_batch_count_covers_every_sample(self, data):
        loader = DataLoader(data, batch_size=16)
        assert len(loader) == 3  # 16 + 16 + 8
        assert sum(len(bx) for bx, _ in loader) == 40

    def test_drop_last_discards_the_partial_batch(self, data):
        loader = DataLoader(data, batch_size=16, drop_last=True)
        assert len(loader) == 2
        assert all(len(bx) == 16 for bx, _ in loader)

    def test_every_sample_appears_exactly_once_per_epoch(self, data):
        """The property that matters most, and the easiest to break by an
        off-by-one in the slicing."""
        loader = DataLoader(data, batch_size=7, seed=0)
        seen = [tuple(row) for bx, _ in loader for row in bx]
        assert len(seen) == 40
        assert len(set(seen)) == 40

    def test_features_stay_paired_with_their_labels(self, data):
        """Shuffling x and y independently would be catastrophic and silent:
        the model would train happily and learn nothing."""
        lookup = {tuple(row): int(label) for row, label in zip(data.x, data.y)}
        loader = DataLoader(data, batch_size=6, seed=2)
        for bx, by in loader:
            for row, label in zip(bx, by):
                assert lookup[tuple(row)] == int(label)

    def test_same_seed_gives_the_same_order(self, data):
        a = [list(b) for b, _ in DataLoader(data, batch_size=8, seed=7)]
        b = [list(b) for b, _ in DataLoader(data, batch_size=8, seed=7)]
        assert a == b

    def test_shuffle_false_preserves_dataset_order(self, data):
        loader = DataLoader(data, batch_size=10, shuffle=False)
        first_batch = next(iter(loader))[0]
        assert np.allclose(np.asarray(first_batch), data.x[:10])

    def test_reshuffles_between_epochs(self, data):
        """A loader that returns the same order every epoch defeats the point
        of shuffling, and the bug survives any single-epoch test."""
        loader = DataLoader(data, batch_size=40, seed=0)
        first = next(iter(loader))[0]
        second = next(iter(loader))[0]
        assert not np.allclose(np.asarray(first), np.asarray(second))

    def test_rejects_a_nonsense_batch_size(self, data):
        with pytest.raises(ValueError, match="batch_size"):
            DataLoader(data, batch_size=0)

    def test_batch_size_is_clamped_to_the_dataset(self, data):
        assert DataLoader(data, batch_size=1000).batch_size == 40


class TestAsArrays:
    """Phase 19: the fast path must be a pure representation change."""

    def test_default_yields_plain_python_lists(self, data):
        bx, by = next(iter(DataLoader(data, batch_size=4, shuffle=False)))
        assert isinstance(bx, list) and isinstance(bx[0], list)
        assert isinstance(bx[0][0], float)
        assert isinstance(by, list)

    def test_as_arrays_yields_ndarrays(self, data):
        bx, by = next(
            iter(DataLoader(data, batch_size=4, shuffle=False, as_arrays=True))
        )
        assert isinstance(bx, np.ndarray) and isinstance(by, np.ndarray)
        assert bx.shape == (4, 3)

    def test_both_forms_carry_identical_numbers(self, data):
        """The whole justification for the optimisation: same data, cheaper
        representation. If this ever fails, the speedup was a bug."""
        lists = DataLoader(data, batch_size=6, seed=11)
        arrays = DataLoader(data, batch_size=6, seed=11, as_arrays=True)
        for (lx, ly), (ax, ay) in zip(lists, arrays):
            assert np.allclose(np.asarray(lx), ax)
            assert np.array_equal(np.asarray(ly), ay)

    def test_batches_do_not_alias_the_dataset(self, data):
        """Fancy indexing copies, but if it ever returned a view, mutating a
        batch would corrupt the dataset for every later epoch."""
        original = data.x.copy()
        bx, _ = next(iter(DataLoader(data, batch_size=4, shuffle=False, as_arrays=True)))
        bx[0, 0] = 999.0
        assert np.array_equal(data.x, original)

    def test_len_works_on_both_forms(self, data):
        """The Trainer calls len(batch_x) for its sample counter."""
        for flag in (True, False):
            bx, _ = next(iter(DataLoader(data, batch_size=5, as_arrays=flag)))
            assert len(bx) == 5

    def test_repr_says_which_form_it_yields(self, data):
        assert "arrays" in repr(DataLoader(data, as_arrays=True))
        assert "lists" in repr(DataLoader(data, as_arrays=False))


# ======================================================================
# splitting
# ======================================================================


class TestTrainValSplit:
    def test_split_sizes_add_up(self, data):
        """Nothing is lost or duplicated by the split.

        The per-class cut is ``round(10 * 0.25)``, and Python rounds halves to
        even -- so that is 2 per class, not 3, giving 8 rather than 10. Worth
        pinning: it is the kind of off-by-one that only shows up on small
        validation sets, which is precisely where it does damage.
        """
        train, val = train_val_split(data, val_fraction=0.25, seed=0)
        assert len(train) + len(val) == len(data)
        assert len(val) == 8
        assert len(train) == 32

    def test_the_two_halves_are_disjoint(self, data):
        """Leakage here inflates validation accuracy and invalidates every
        conclusion drawn from it."""
        train, val = train_val_split(data, val_fraction=0.25, seed=0)
        train_rows = {tuple(r) for r in train.x.tolist()}
        val_rows = {tuple(r) for r in val.x.tolist()}
        assert train_rows.isdisjoint(val_rows)

    def test_stratified_split_preserves_class_balance(self, data):
        train, val = train_val_split(data, val_fraction=0.25, seed=0, stratify=True)
        assert val.class_counts() == {0: 2, 1: 2, 2: 2, 3: 2}
        assert train.class_counts() == {0: 8, 1: 8, 2: 8, 3: 8}

    def test_stratification_keeps_rare_classes_present(self):
        """The case stratification exists for: a plain random 10% split of an
        imbalanced set can miss a class entirely."""
        x = np.arange(100, dtype=float).reshape(100, 1)
        y = np.array([0] * 95 + [1] * 5)
        _, val = train_val_split(Dataset(x, y), val_fraction=0.1, seed=0)
        assert 1 in val.class_counts()

    def test_features_stay_paired_after_splitting(self, data):
        lookup = {tuple(r): int(l) for r, l in zip(data.x, data.y)}
        train, val = train_val_split(data, val_fraction=0.25, seed=5)
        for part in (train, val):
            for row, label in zip(part.x, part.y):
                assert lookup[tuple(row)] == int(label)

    def test_same_seed_reproduces_the_split(self, data):
        a, _ = train_val_split(data, val_fraction=0.2, seed=3)
        b, _ = train_val_split(data, val_fraction=0.2, seed=3)
        assert np.array_equal(a.x, b.x)

    @pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
    def test_rejects_out_of_range_fractions(self, data, fraction):
        with pytest.raises(ValueError, match="val_fraction"):
            train_val_split(data, val_fraction=fraction)


# ======================================================================
# one-hot
# ======================================================================


class TestOneHot:
    def test_shape_and_contents(self):
        out = one_hot([0, 2, 1], 3)
        assert out.shape == (3, 3)
        assert out.tolist() == [[1, 0, 0], [0, 0, 1], [0, 1, 0]]

    def test_each_row_sums_to_one(self):
        out = one_hot([0, 1, 2, 3, 0], 4)
        assert np.allclose(out.sum(axis=1), 1.0)

    def test_handles_a_single_label(self):
        assert one_hot([1], 2).tolist() == [[0, 1]]

    def test_wider_than_the_labels_used(self):
        """10 classes with only digit 3 present must still produce 10 columns."""
        assert one_hot([3], 10).shape == (1, 10)
