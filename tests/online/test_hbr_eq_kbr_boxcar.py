"""HBR with edges [a, a+h, a+2h, ...] should match KBR with Boxcar at bin centers, h/2.

Mirrors the Julia ``compare_online.jl`` test: histogram binning is recoverable
as boxcar-kernel KBR with the kernel covering exactly one bin.

Caveat: the Boxcar kernel uses strict ``|u| < 1`` (open both sides) while HBR
bins are half-open ``[a, b)``. A data point that lands *exactly* on a bin edge
gets binned by HBR but excluded by KBR-Boxcar. The test therefore uses
synthetic data positioned strictly interior to its bin.
"""
import numpy as np

from online_moments import OHBR, OKBR, Boxcar


def test_ohbr_matches_okbr_boxcar():
    h = 0.1
    edges = np.array([0.0, h, 2 * h, 3 * h, 4 * h])  # 4 bins
    centers = (edges[:-1] + edges[1:]) / 2.0
    tau = np.array([1, 2], dtype=np.int64)

    # Synthetic data: 200 points uniformly distributed inside [edges[0]+ε, edges[-1]-ε],
    # with no chance of landing exactly on an internal edge.
    rng = np.random.default_rng(20240101)
    X = rng.uniform(edges[0] + 1e-3, edges[-1] - 1e-3, size=200)

    ohbr = OHBR(edges, tau)
    ohbr.update_batch(X)

    okbr = OKBR(centers, tau, kernel=Boxcar(), bandwidth=h / 2.0)
    okbr.update_batch(X)

    np.testing.assert_allclose(ohbr.M1, okbr.M1, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(ohbr.M2, okbr.M2, rtol=1e-12, atol=1e-14)
