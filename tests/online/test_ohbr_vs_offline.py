"""OHBR online ↔ HBR offline equivalence.

Both code paths apply the same Welford-style updates in the same order, so
results are bit-identical. This is the strongest correctness test in the
suite — any divergence here means the streaming code does different math
from the offline reference.
"""
import numpy as np
import pytest

from online_moments import OHBR, hbr_moments


@pytest.mark.parametrize("moment_form", ["variance", "raw"])
def test_ohbr_matches_hbr_exactly(X_small, tau_indices_small, x_edges_small, moment_form):
    online = OHBR(x_edges_small, tau_indices_small, moment_form=moment_form)
    online.update_batch(X_small)

    M1_off, M2_off = hbr_moments(
        X_small, tau_indices=tau_indices_small, edges=x_edges_small,
        moment_form=moment_form,
    )
    assert np.array_equal(online.M1, M1_off), "OHBR.M1 must equal hbr_moments M1 exactly"
    assert np.array_equal(online.M2, M2_off), "OHBR.M2 must equal hbr_moments M2 exactly"


def test_ohbr_streaming_matches_batch(X_small, x_edges_small):
    tau = np.array([1, 2, 3], dtype=np.int64)
    a = OHBR(x_edges_small, tau)
    a.update_batch(X_small)

    b = OHBR(x_edges_small, tau)
    for x in X_small:
        b.update(float(x))

    assert np.array_equal(a.M1, b.M1)
    assert np.array_equal(a.M2, b.M2)
    assert np.array_equal(a.counts, b.counts)
    assert a.n_processed == b.n_processed == len(X_small)


def test_ohbr_single_tau(X_small, x_edges_small):
    tau = np.array([1], dtype=np.int64)
    o = OHBR(x_edges_small, tau)
    o.update_batch(X_small)
    M1_off, M2_off = hbr_moments(X_small, tau_indices=tau, edges=x_edges_small)
    assert np.array_equal(o.M1, M1_off)
    assert np.array_equal(o.M2, M2_off)
