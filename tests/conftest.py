"""Shared test helpers.

Adds the repository root to ``sys.path`` so the suite runs against the working
tree without requiring an install, and defines the tolerance helpers used
throughout.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Analytic derivatives should agree with hand-computed values to near machine
# precision -- these are the *same* arithmetic, so any disagreement beyond
# round-off is a real bug. Finite-difference comparisons (Phase 5) get a much
# looser tolerance, for the reasons in docs/01-*.md section 3.
EXACT = 1e-12
TIGHT = 1e-9


def close(a: float, b: float, tol: float = TIGHT) -> bool:
    """Absolute-or-relative closeness, tolerant of large magnitudes."""
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


@pytest.fixture
def approx_exact():
    """``pytest.approx`` configured for exact-arithmetic comparisons."""
    return lambda x: pytest.approx(x, abs=EXACT, rel=EXACT)
