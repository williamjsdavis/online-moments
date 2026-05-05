"""Shared utilities for validation drivers.

Provides Euler-Maruyama simulators for the SDEs in the paper, and a helper
that lays Python output next to the corresponding paper figure for visual
side-by-side comparison.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from numba import njit


REPO_ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = REPO_ROOT / "paper"
FIGURES_DIR = REPO_ROOT / "validation" / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def euler_maruyama(
    drift: Callable[[float], float],
    diffusion: Callable[[float], float],
    *,
    x0: float,
    dt: float,
    n_steps: int,
    seed: int = 0,
) -> np.ndarray:
    """Euler-Maruyama integration of dX = f(X) dt + sqrt(2 g(X)) dW.

    Note the factor of √2: the paper's convention is ``dX/dt = f + g·Γ`` with
    ``g(x) = sqrt(2 D^(2)(x))``, so the noise increment is ``sqrt(2·D^(2)·dt)·N(0,1)``.
    """
    rng = np.random.default_rng(seed)
    X = np.empty(n_steps, dtype=np.float64)
    X[0] = x0
    sqrt_dt = np.sqrt(dt)
    noise = rng.standard_normal(n_steps - 1)
    x = x0
    for k in range(1, n_steps):
        f = drift(x)
        g = np.sqrt(2.0 * diffusion(x))
        x = x + f * dt + g * sqrt_dt * noise[k - 1]
        X[k] = x
    return X


def euler_maruyama_chunked(
    drift: Callable[[float], float],
    diffusion: Callable[[float], float],
    *,
    x0: float,
    dt: float,
    n_steps: int,
    seed: int = 0,
    chunk: int = 10_000_000,
):
    """Chunked simulator for very large N — yields chunks instead of allocating
    a single array. Used by the scaling driver."""
    rng = np.random.default_rng(seed)
    sqrt_dt = np.sqrt(dt)
    x = x0
    remaining = n_steps
    first = True
    while remaining > 0:
        c = min(remaining, chunk)
        out = np.empty(c, dtype=np.float64)
        if first:
            out[0] = x
            start = 1
            first = False
        else:
            start = 0
        # Generate noise for c steps if first, else c steps.
        if start == 1:
            noise = rng.standard_normal(c - 1)
            for k in range(start, c):
                f = drift(x)
                g = np.sqrt(2.0 * diffusion(x))
                x = x + f * dt + g * sqrt_dt * noise[k - 1]
                out[k] = x
        else:
            noise = rng.standard_normal(c)
            for k in range(c):
                f = drift(x)
                g = np.sqrt(2.0 * diffusion(x))
                x = x + f * dt + g * sqrt_dt * noise[k]
                out[k] = x
        yield out
        remaining -= c


# SDE specifications from the paper

def ou_drift(x: float) -> float:
    return -x


def ou_diffusion(x: float) -> float:
    return 1.0


def tristable_drift(x: float) -> float:
    return -x + 27.0 * x**3 - 26.0 * x**5


def tristable_diffusion(x: float) -> float:
    return 0.7


# Numba-jitted simulators for the paper's two SDEs (much faster at large N).
#
# Two flavors per SDE:
#   - simulate_ou(n_steps, dt, x0, seed) returns one array of length n_steps.
#     Convenient for n_steps up to ~1e8 (~800 MB at float64).
#   - simulate_ou_chunk(out, dt, x_init) writes a pre-allocated ``out`` chunk
#     in place using Numba's persistent global RNG state, returning the final
#     x value. Caller seeds once via ``_seed_rng(seed)`` and then calls the
#     chunk function repeatedly. Enables N >= 1e9 streaming runs without
#     allocating a single huge array.


@njit(cache=True)
def _seed_rng(seed: int) -> None:
    np.random.seed(seed)


@njit(cache=True)
def simulate_ou(n_steps: int, dt: float, x0: float, seed: int) -> np.ndarray:
    """Ornstein-Uhlenbeck: dX = -X dt + sqrt(2) dW."""
    np.random.seed(seed)
    sqrt_2dt = np.sqrt(2.0 * dt)
    X = np.empty(n_steps, dtype=np.float64)
    X[0] = x0
    x = x0
    for k in range(1, n_steps):
        x = x + (-x) * dt + sqrt_2dt * np.random.randn()
        X[k] = x
    return X


@njit(cache=True)
def simulate_ou_chunk(out: np.ndarray, dt: float, x_init: float) -> float:
    """Continuation-style OU simulator. Writes ``out`` in place, returns last x.

    Caller is responsible for seeding the RNG via ``_seed_rng`` before the
    first call. RNG state persists across calls (Numba's per-thread state).
    """
    sqrt_2dt = np.sqrt(2.0 * dt)
    x = x_init
    for k in range(out.shape[0]):
        x = x + (-x) * dt + sqrt_2dt * np.random.randn()
        out[k] = x
    return x


@njit(cache=True)
def simulate_tristable(n_steps: int, dt: float, x0: float, seed: int) -> np.ndarray:
    """Tri-stable: dX = (-x + 27 x^3 - 26 x^5) dt + sqrt(2*0.7) dW."""
    np.random.seed(seed)
    sqrt_2g_dt = np.sqrt(2.0 * 0.7 * dt)
    X = np.empty(n_steps, dtype=np.float64)
    X[0] = x0
    x = x0
    for k in range(1, n_steps):
        f = -x + 27.0 * x**3 - 26.0 * x**5
        x = x + f * dt + sqrt_2g_dt * np.random.randn()
        X[k] = x
    return X


@njit(cache=True)
def simulate_tristable_chunk(out: np.ndarray, dt: float, x_init: float) -> float:
    sqrt_2g_dt = np.sqrt(2.0 * 0.7 * dt)
    x = x_init
    for k in range(out.shape[0]):
        f = -x + 27.0 * x**3 - 26.0 * x**5
        x = x + f * dt + sqrt_2g_dt * np.random.randn()
        out[k] = x
    return x


def stream_simulator(
    chunk_fn,
    *,
    n_total: int,
    dt: float,
    x0: float,
    seed: int,
    chunk_size: int = 10_000_000,
):
    """Yield successive chunks of a long simulation, sized to ``chunk_size``.

    Maintains continuity through Numba's persistent RNG state and a running
    ``x`` value. ``chunk_fn`` is one of ``simulate_ou_chunk`` or
    ``simulate_tristable_chunk``.
    """
    _seed_rng(seed)
    x = float(x0)
    remaining = n_total
    buf = np.empty(chunk_size, dtype=np.float64)
    while remaining > 0:
        c = min(remaining, chunk_size)
        view = buf[:c] if c < chunk_size else buf
        x = chunk_fn(view, dt, x)
        yield view
        remaining -= c


def render_paper_page_as_image(pdf_path: Path, dpi: int = 150) -> np.ndarray | None:
    """Render the first page of a single-page PDF to a NumPy RGB array.

    Returns None if pdf2image / poppler is not available — drivers should fall
    back to plotting only the Python output in that case.
    """
    try:
        from pdf2image import convert_from_path
    except ImportError:
        return None
    try:
        pages = convert_from_path(str(pdf_path), dpi=dpi)
    except Exception:
        return None
    if not pages:
        return None
    return np.asarray(pages[0])
