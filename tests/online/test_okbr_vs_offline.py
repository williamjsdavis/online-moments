"""OKBR online ↔ KBR offline equivalence.

The offline reference accumulates ``Σ K dx`` and divides at the end; the
online estimator uses a kernel-weighted Welford recurrence. They are
mathematically equivalent but accumulate floating-point error differently —
``np.allclose(rtol=1e-10)`` is the right bar.
"""
import numpy as np
import pytest

from online_moments import OKBR, Boxcar, Epanechnikov, kbr_moments


@pytest.fixture
def x_eval():
    return np.linspace(0.0, 0.5, 20)


@pytest.fixture(params=[Boxcar(), Epanechnikov()], ids=["boxcar", "epanechnikov"])
def kernel(request):
    return request.param


@pytest.mark.parametrize("moment_form", ["variance", "raw"])
def test_okbr_matches_kbr(X_small, tau_indices_small, x_eval, kernel, moment_form):
    bandwidth = 0.05
    online = OKBR(
        x_eval, tau_indices_small, kernel=kernel, bandwidth=bandwidth,
        moment_form=moment_form,
    )
    online.update_batch(X_small)

    M1_off, M2_off = kbr_moments(
        X_small, tau_indices=tau_indices_small, x_eval=x_eval,
        kernel=kernel, bandwidth=bandwidth, moment_form=moment_form,
    )
    np.testing.assert_allclose(online.M1, M1_off, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(online.M2, M2_off, rtol=1e-10, atol=1e-12)


def test_okbr_streaming_matches_batch(X_small, x_eval):
    tau = np.array([1, 3], dtype=np.int64)
    a = OKBR(x_eval, tau, kernel=Epanechnikov(), bandwidth=0.05)
    a.update_batch(X_small)

    b = OKBR(x_eval, tau, kernel=Epanechnikov(), bandwidth=0.05)
    for x in X_small:
        b.update(float(x))

    np.testing.assert_allclose(a.M1, b.M1, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(a.M2, b.M2, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(a.W, b.W, rtol=1e-12, atol=1e-14)
