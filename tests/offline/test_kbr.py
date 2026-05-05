import numpy as np
import pytest

from online_moments import Boxcar, Epanechnikov, kbr_moments


@pytest.mark.parametrize("kernel", [Boxcar(), Epanechnikov()])
def test_kbr_shapes(X_small, tau_indices_small, kernel):
    x_eval = np.linspace(0.0, 0.5, 20)
    M1, M2 = kbr_moments(
        X_small, tau_indices=tau_indices_small, x_eval=x_eval,
        kernel=kernel, bandwidth=0.05,
    )
    assert M1.shape == (len(tau_indices_small), len(x_eval))
    assert M2.shape == M1.shape


def test_kbr_variance_nonnegative(X_small, tau_indices_small):
    x_eval = np.linspace(0.0, 0.5, 20)
    _, M2 = kbr_moments(
        X_small, tau_indices=tau_indices_small, x_eval=x_eval,
        kernel=Epanechnikov(), bandwidth=0.05, moment_form="variance",
    )
    assert (M2 >= -1e-12).all()


def test_kbr_with_zero_weight_returns_zero(X_small, tau_indices_small):
    """Evaluation point far from data, kernel weight is zero everywhere ⇒ M1=M2=0."""
    x_eval = np.array([100.0])
    M1, M2 = kbr_moments(
        X_small, tau_indices=tau_indices_small, x_eval=x_eval,
        kernel=Epanechnikov(), bandwidth=0.05,
    )
    assert (M1 == 0).all()
    assert (M2 == 0).all()
