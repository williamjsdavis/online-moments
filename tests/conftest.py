from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def X_small() -> np.ndarray:
    """Small fixture (100 floats), ported from OnlineMoments.jl/test/X_data_small.jld2."""
    return np.load(DATA_DIR / "X_data_small.npy")


@pytest.fixture(scope="session")
def x_edges_small() -> np.ndarray:
    """Bin edges used by the Julia tests (4 bins over [0.01, 0.41])."""
    return np.linspace(0.01, 0.41, 5)


@pytest.fixture(scope="session")
def tau_indices_small() -> np.ndarray:
    return np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
