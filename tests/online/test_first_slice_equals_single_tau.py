"""For tau_indices=[1, 2, 3], row 0 of multi-τ M1 must equal an independently-run
single-τ-{1} estimator. Detects τ-indexing bugs.
"""
import numpy as np

from online_moments import OHBR, OKBR, Epanechnikov


def test_ohbr_first_slice(X_small):
    edges = np.linspace(0.0, 0.5, 6)
    multi = OHBR(edges, np.array([1, 2, 3], dtype=np.int64))
    multi.update_batch(X_small)

    single = OHBR(edges, np.array([1], dtype=np.int64))
    single.update_batch(X_small)

    np.testing.assert_array_equal(multi.M1[0], single.M1[0])
    np.testing.assert_array_equal(multi.M2[0], single.M2[0])


def test_okbr_first_slice(X_small):
    x_eval = np.linspace(0.0, 0.5, 20)
    multi = OKBR(x_eval, np.array([1, 2, 3], dtype=np.int64),
                 kernel=Epanechnikov(), bandwidth=0.05)
    multi.update_batch(X_small)

    single = OKBR(x_eval, np.array([1], dtype=np.int64),
                  kernel=Epanechnikov(), bandwidth=0.05)
    single.update_batch(X_small)

    np.testing.assert_allclose(multi.M1[0], single.M1[0], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(multi.M2[0], single.M2[0], rtol=1e-12, atol=1e-14)
