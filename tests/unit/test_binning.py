import numpy as np
import pytest

from online_moments.binning import (
    d_mod,
    find_bin,
    find_mod_bin,
    get_bin,
    in_range,
)


@pytest.fixture
def edges():
    return np.linspace(0.0, 4.0, 5)  # 4 bins: [0,1), [1,2), [2,3), [3,4]


def test_in_range_endpoints(edges):
    assert in_range(edges, 0.0)
    assert in_range(edges, 4.0)
    assert in_range(edges, 2.0)
    assert not in_range(edges, -0.001)
    assert not in_range(edges, 4.001)


def test_find_bin_interior(edges):
    assert find_bin(edges, 0.5) == 0
    assert find_bin(edges, 1.5) == 1
    assert find_bin(edges, 2.5) == 2
    assert find_bin(edges, 3.5) == 3


def test_find_bin_internal_edges_go_upper(edges):
    """Half-open: a value exactly on edge[i] (i in interior) lands in bin i."""
    assert find_bin(edges, 1.0) == 1
    assert find_bin(edges, 2.0) == 2
    assert find_bin(edges, 3.0) == 3


def test_find_bin_left_edge(edges):
    assert find_bin(edges, 0.0) == 0


def test_find_bin_right_edge(edges):
    """Closed last bin: value equal to edges[-1] lands in the last bin."""
    assert find_bin(edges, 4.0) == 3


def test_find_bin_matches_searchsorted(edges):
    rng = np.random.default_rng(42)
    xs = rng.uniform(0.0, 4.0 - 1e-9, size=200)
    expected = np.clip(np.searchsorted(edges, xs, side="right") - 1, 0, len(edges) - 2)
    actual = np.array([find_bin(edges, x) for x in xs])
    assert np.array_equal(actual, expected)


def test_get_bin_oob(edges):
    assert get_bin(edges, -1.0) == -1
    assert get_bin(edges, 5.0) == -1
    assert get_bin(edges, 1.5) == 1


def test_d_mod_basic():
    assert d_mod(0.3, 1.0) == pytest.approx(0.3)
    assert d_mod(-0.3, 1.0) == pytest.approx(0.3)
    assert d_mod(0.7, 1.0) == pytest.approx(0.3)
    assert d_mod(1.3, 1.0) == pytest.approx(0.3)


def test_d_mod_zero():
    assert d_mod(0.0, 1.0) == 0.0
    assert d_mod(1.0, 1.0) == 0.0


def test_find_mod_bin_translational_invariance():
    edges = np.linspace(0.0, 1.0, 5)
    period = 1.0
    rng = np.random.default_rng(0)
    xs = rng.uniform(0.0, 1.0, size=50)
    for x in xs:
        for k in (-3, -1, 0, 2, 7):
            shifted = (x + k * period) % period
            assert find_mod_bin(edges, period, x) == find_mod_bin(edges, period, shifted)


def test_find_mod_bin_returns_minus1_for_uncovered_region():
    # Edges only cover half the period; the other half maps to -1.
    edges = np.array([0.0, 0.25, 0.5])
    period = 1.0
    assert find_mod_bin(edges, period, 0.1) >= 0
    assert find_mod_bin(edges, period, 0.4) >= 0
    assert find_mod_bin(edges, period, 0.7) == -1
