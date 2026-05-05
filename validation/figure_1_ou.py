"""Reproduce Figure 1 of the paper (Ornstein-Uhlenbeck process).

Drift D^(1)(x) = -x, diffusion D^(2)(x) = 1. Conditional moments at 26 evenly
spaced x in [-5, 5], bandwidth h = 0.4, single τ = Δt with Δt = 1e-3.

The paper's Figure 1 has THREE traces:
    (1) KBR offline at N₁ (the largest N that fits in memory),
    (2) OKBR streaming at N₁ (overlays exactly with (1) — demonstrates equivalence),
    (3) OKBR streaming at N₂ ≫ N₁ (impossible to compute with (1) — shows the
        scientific gain of streaming).

The paper uses N₁ = 10⁷ and N₂ = 10¹⁰. The smoke run uses N₁ = 10⁶ and
N₂ = 10⁸ to show the same shape in ~30 seconds.

The N₂ run uses a chunked simulator so the time series is never fully
materialised in memory — important at N₂ = 10¹⁰ (80 GB if held as one array).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from online_moments import OKBR, Epanechnikov, kbr_moments

from _common import (
    FIGURES_DIR,
    simulate_ou,
    simulate_ou_chunk,
    stream_simulator,
)


def run(n_small: int, n_large: int, dt: float = 1e-3, seed: int = 1) -> dict:
    x_eval = np.linspace(-5.0, 5.0, 26)
    tau_indices = np.array([1], dtype=np.int64)
    h = 0.4
    kernel = Epanechnikov()

    # Small run: KBR offline + OKBR streaming on the SAME trajectory.
    print(f"  small N={n_small:.0e} ...")
    t0 = time.perf_counter()
    X_small = simulate_ou(n_small, dt, x0=0.0, seed=seed)
    t_sim_small = time.perf_counter() - t0
    print(f"    sim done in {t_sim_small:.1f}s")

    okbr_small = OKBR(x_eval, tau_indices, kernel=kernel, bandwidth=h)
    t0 = time.perf_counter()
    okbr_small.update_batch(X_small)
    t_okbr_small = time.perf_counter() - t0
    D1_okbr_small, D2_okbr_small = okbr_small.drift_diffusion(dt, fit="direct")
    print(f"    OKBR(small) done in {t_okbr_small:.1f}s")

    t0 = time.perf_counter()
    M1_off, M2_off = kbr_moments(
        X_small, tau_indices=tau_indices, x_eval=x_eval,
        kernel=kernel, bandwidth=h,
    )
    t_kbr_small = time.perf_counter() - t0
    D1_kbr = M1_off[0] / dt
    D2_kbr = M2_off[0] / (2.0 * dt)
    print(f"    KBR(small) done in {t_kbr_small:.1f}s")

    # Free the small trajectory before the large run.
    del X_small

    # Large run: OKBR streaming via chunked simulator. Never materialises the
    # full trajectory.
    print(f"  large N={n_large:.0e} ...  (chunked)")
    okbr_large = OKBR(x_eval, tau_indices, kernel=kernel, bandwidth=h)
    chunks_seen = 0
    samples_seen = 0
    t0 = time.perf_counter()
    chunk_size = min(10_000_000, max(n_large // 10, 1))
    for chunk in stream_simulator(
        simulate_ou_chunk, n_total=n_large, dt=dt, x0=0.0, seed=seed + 1000,
        chunk_size=chunk_size,
    ):
        okbr_large.update_batch(chunk)
        chunks_seen += 1
        samples_seen += chunk.size
        if chunks_seen % max(1, n_large // chunk_size // 10) == 0:
            print(f"      {samples_seen / n_large * 100:.0f}% "
                  f"({samples_seen:,} / {n_large:,})")
    t_okbr_large = time.perf_counter() - t0
    D1_okbr_large, D2_okbr_large = okbr_large.drift_diffusion(dt, fit="direct")
    print(f"    OKBR(large) done in {t_okbr_large:.1f}s")

    return {
        "x_eval": x_eval,
        "n_small": n_small,
        "n_large": n_large,
        "D1_kbr": D1_kbr,
        "D2_kbr": D2_kbr,
        "D1_okbr_small": D1_okbr_small,
        "D2_okbr_small": D2_okbr_small,
        "D1_okbr_large": D1_okbr_large,
        "D2_okbr_large": D2_okbr_large,
    }


def _fmt(n: int) -> str:
    return f"$10^{int(np.log10(n))}$" if n in (10**k for k in range(15)) else f"{n:.0e}"


def plot(result: dict, out_path: Path) -> Path:
    x = result["x_eval"]
    n_small = result["n_small"]
    n_large = result["n_large"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.axhline(0, color="lightgray", lw=0.5)
    ax.plot(x, -x, "k--", lw=1.2, label=r"true $D^{(1)}(x) = -x$")
    ax.plot(x, result["D1_kbr"], "rx", ms=10, mew=1.5,
            label=f"KBR, N={_fmt(n_small)}")
    ax.plot(x, result["D1_okbr_small"], "g+", ms=10, mew=1.5,
            label=f"OKBR, N={_fmt(n_small)}")
    ax.plot(x, result["D1_okbr_large"], "o", ms=7, mfc="none", mec="C0",
            label=f"OKBR, N={_fmt(n_large)}")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"drift $D^{(1)}(x)$")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(-7.5, 7.5)

    ax = axes[1]
    ax.axhline(1.0, color="k", linestyle="--", lw=1.2, label=r"true $D^{(2)}(x) = 1$")
    ax.plot(x, result["D2_kbr"], "rx", ms=10, mew=1.5,
            label=f"KBR, N={_fmt(n_small)}")
    ax.plot(x, result["D2_okbr_small"], "g+", ms=10, mew=1.5,
            label=f"OKBR, N={_fmt(n_small)}")
    ax.plot(x, result["D2_okbr_large"], "o", ms=7, mfc="none", mec="C0",
            label=f"OKBR, N={_fmt(n_large)}")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"diffusion $D^{(2)}(x)$")
    ax.legend(fontsize=9, loc="lower center")
    ax.set_ylim(-0.2, 2.0)

    fig.suptitle("Figure 1 reproduction: Ornstein-Uhlenbeck (Python)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-small", type=int, default=None,
                        help="N for the KBR + OKBR-equivalence trace")
    parser.add_argument("--n-large", type=int, default=None,
                        help="N for the OKBR-only large trace")
    parser.add_argument("--full", action="store_true",
                        help="paper sizes: N_small=1e7, N_large=1e10")
    parser.add_argument("--smoke", action="store_true",
                        help="CI sizes: N_small=1e6, N_large=1e8")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    if args.full:
        n_small, n_large = 10_000_000, 10_000_000_000
    else:
        n_small, n_large = 1_000_000, 100_000_000
    if args.n_small is not None:
        n_small = args.n_small
    if args.n_large is not None:
        n_large = args.n_large

    result = run(n_small, n_large, seed=args.seed)
    out_solo = plot(result, FIGURES_DIR / "figure_1_python.png")
    print(f"  wrote {out_solo}")


if __name__ == "__main__":
    main()
