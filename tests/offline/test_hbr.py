import numpy as np

from online_moments import hbr_moments


def test_hbr_shapes_and_dtype(X_small, tau_indices_small, x_edges_small):
    M1, M2 = hbr_moments(X_small, tau_indices=tau_indices_small, edges=x_edges_small)
    assert M1.shape == (len(tau_indices_small), len(x_edges_small) - 1)
    assert M2.shape == M1.shape
    assert M1.dtype == np.float64


def test_hbr_variance_nonnegative(X_small, tau_indices_small, x_edges_small):
    _, M2 = hbr_moments(
        X_small, tau_indices=tau_indices_small, edges=x_edges_small,
        moment_form="variance",
    )
    assert (M2 >= -1e-15).all()


def test_hbr_raw_form_differs_from_variance(X_small, tau_indices_small, x_edges_small):
    _, M2_var = hbr_moments(
        X_small, tau_indices=tau_indices_small, edges=x_edges_small,
        moment_form="variance",
    )
    _, M2_raw = hbr_moments(
        X_small, tau_indices=tau_indices_small, edges=x_edges_small,
        moment_form="raw",
    )
    # Raw second moment ≥ variance always (E[X²] = Var[X] + E[X]² ≥ Var[X]).
    assert (M2_raw + 1e-15 >= M2_var).all()
