import numpy as np
import pytest

from online_moments.kernels import Boxcar, Epanechnikov, apply_kernel


def test_boxcar_value_at_zero():
    assert Boxcar()(0.0) == 0.5


def test_boxcar_support():
    k = Boxcar()
    assert k(-0.999) == 0.5
    assert k(0.999) == 0.5
    assert k(-1.0) == 0.0
    assert k(1.0) == 0.0
    assert k(2.0) == 0.0


def test_epanechnikov_value_at_zero():
    assert Epanechnikov()(0.0) == pytest.approx(0.75)


def test_epanechnikov_support():
    k = Epanechnikov()
    assert k(0.999) > 0
    assert k(-0.999) > 0
    assert k(1.0) == 0.0
    assert k(-1.0) == 0.0
    assert k(1.5) == 0.0


@pytest.mark.parametrize("kernel", [Boxcar(), Epanechnikov()])
def test_kernel_integrates_to_one(kernel):
    x = np.linspace(-2.0, 2.0, 100_001)
    y = np.array([kernel(xi) for xi in x])
    integral = np.trapezoid(y, x)
    assert abs(integral - 1.0) < 1e-3


@pytest.mark.parametrize("kernel", [Boxcar(), Epanechnikov()])
@pytest.mark.parametrize("h", [0.1, 0.5, 1.0, 2.0])
def test_bandwidth_scaling_preserves_normalization(kernel, h):
    hinv = 1.0 / h
    x = np.linspace(-3.0 * h, 3.0 * h, 200_001)
    y = np.array([apply_kernel(xi, kernel, hinv) for xi in x])
    integral = np.trapezoid(y, x)
    assert abs(integral - 1.0) < 1e-3


def test_epanechnikov_second_moment():
    k = Epanechnikov()
    x = np.linspace(-1.0, 1.0, 100_001)
    y = np.array([k(xi) for xi in x])
    second_moment = np.trapezoid(x * x * y, x)
    # Standard convention: variance is 1/5 = 0.2
    assert abs(second_moment - 0.2) < 1e-3
