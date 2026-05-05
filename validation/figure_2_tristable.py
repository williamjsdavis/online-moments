"""Reproduce Figure 2 of the paper (tri-stable system).

Drift D^(1)(x) = -x + 27 x^3 - 26 x^5 (three attractors at x = -1, 0, +1, the
middle one weak). Diffusion D^(2)(x) = 0.7. Δt = 1e-4. Conditional moments at
45 evenly spaced x in [-1.4, 1.4], bandwidth h = 0.03, τ = [Δt..4Δt].

The paper's Figure 2 shows TWO traces:
    (1) KBR offline at N₁ = 5×10⁷ — fails to resolve the weak attractor at x=0,
    (2) OKBR streaming at N₂ = 10¹⁰ — resolves it cleanly.

This is the headline scientific point of the paper: the streaming algorithm
unlocks resolution at sample sizes that simply don't fit in memory.

Smoke run uses N₁ = 5×10⁶ and N₂ = 5×10⁸ to capture the same shape in ~3 min.
The N₂ run is chunked so the trajectory is never fully materialised.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from online_moments import OKBR, Epanechnikov, kbr_moments
from online_moments.reductions import kramers_moyal_regression

from _common import (
    FIGURES_DIR,
    simulate_tristable,
    simulate_tristable_chunk,
    stream_simulator,
)


def true_drift(x):
    return -x + 27.0 * x**3 - 26.0 * x**5


def run(n_kbr: int, n_okbr_large: int, dt: float = 1e-4, seed: int = 7) -> dict:
    x_eval = np.linspace(-1.4, 1.4, 45)
    tau_indices = np.array([1, 2, 3, 4], dtype=np.int64)
    h = 0.03
    kernel = Epanechnikov()

    print(f"  KBR run, N={n_kbr:.0e} ...")
    t0 = time.perf_counter()
    X_kbr = simulate_tristable(n_kbr, dt, x0=1.0, seed=seed)
    t_sim_kbr = time.perf_counter() - t0
    print(f"    sim done in {t_sim_kbr:.1f}s")

    t0 = time.perf_counter()
    M1_off, M2_off = kbr_moments(
        X_kbr, tau_indices=tau_indices, x_eval=x_eval,
        kernel=kernel, bandwidth=h,
    )
    t_kbr = time.perf_counter() - t0
    D1_kbr, D2_kbr = kramers_moyal_regression(M1_off, M2_off, tau_indices, dt)
    print(f"    KBR done in {t_kbr:.1f}s")

    del X_kbr  # free before the streaming run

    # OKBR run at the large N — chunked, never holds the full trajectory.
    print(f"  OKBR run, N={n_okbr_large:.0e} ...  (chunked)")
    okbr = OKBR(x_eval, tau_indices, kernel=kernel, bandwidth=h)
    chunk_size = min(10_000_000, max(n_okbr_large // 10, 1))
    t0 = time.perf_counter()
    samples_seen = 0
    chunks_seen = 0
    report_every = max(1, n_okbr_large // chunk_size // 10)
    for chunk in stream_simulator(
        simulate_tristable_chunk, n_total=n_okbr_large, dt=dt, x0=1.0,
        seed=seed + 1000, chunk_size=chunk_size,
    ):
        okbr.update_batch(chunk)
        chunks_seen += 1
        samples_seen += chunk.size
        if chunks_seen % report_every == 0:
            print(f"      {samples_seen / n_okbr_large * 100:.0f}% "
                  f"({samples_seen:,} / {n_okbr_large:,})")
    t_okbr = time.perf_counter() - t0
    D1_okbr, D2_okbr = okbr.drift_diffusion(dt, fit="regression")
    print(f"    OKBR done in {t_okbr:.1f}s")

    return {
        "x_eval": x_eval,
        "n_kbr": n_kbr,
        "n_okbr": n_okbr_large,
        "D1_kbr": D1_kbr,
        "D2_kbr": D2_kbr,
        "D1_okbr": D1_okbr,
        "D2_okbr": D2_okbr,
    }


def _fmt(n: int) -> str:
    if n in (10**k for k in range(15)):
        return f"$10^{int(np.log10(n))}$"
    # fall through for things like 5e7, 5e8
    exp = int(np.floor(np.log10(n)))
    mant = n / 10**exp
    if mant == int(mant):
        return rf"${int(mant)}\times10^{exp}$"
    return f"{n:.0e}"


def plot(result: dict, out_path: Path) -> Path:
    x = result["x_eval"]
    n_kbr = result["n_kbr"]
    n_okbr = result["n_okbr"]
    fig, axes = plt.subplots(2, 1, figsize=(7, 8.5), sharex=True)

    ax = axes[0]
    x_dense = np.linspace(-1.4, 1.4, 400)
    ax.plot(x_dense, true_drift(x_dense), "k--", lw=1.2,
            label="true $D^{(1)}(x) = -x + 27x^3 - 26x^5$")
    ax.axhline(0, color="lightgray", lw=0.5)
    ax.plot(x, result["D1_kbr"], "rx", ms=10, mew=1.5,
            label=f"KBR, N={_fmt(n_kbr)}")
    ax.plot(x, result["D1_okbr"], "g+", ms=10, mew=1.5,
            label=f"OKBR, N={_fmt(n_okbr)}")
    ax.set_ylabel(r"drift $D^{(1)}(x)$")
    ax.legend(fontsize=9, loc="upper center")
    ax.set_title("Figure 2 reproduction: tri-stable system (Python)")
    ax.set_ylim(-3.5, 3.5)

    # Inset zoom of the central weak-attractor region (paper's inset)
    inset = ax.inset_axes([0.55, 0.05, 0.4, 0.32])
    inset.plot(x_dense, true_drift(x_dense), "k--", lw=1.0)
    inset.axhline(0, color="lightgray", lw=0.5)
    mask = (x >= -0.5) & (x <= 0.5)
    inset.plot(x[mask], result["D1_kbr"][mask], "rx", ms=8, mew=1.2)
    inset.plot(x[mask], result["D1_okbr"][mask], "g+", ms=8, mew=1.2)
    inset.set_xlim(-0.5, 0.5)
    inset.set_ylim(-1.0, 1.0)
    inset.set_title("zoom: central weak attractor", fontsize=8)
    inset.tick_params(labelsize=7)

    ax = axes[1]
    ax.axhline(0.7, color="k", linestyle="--", lw=1.2,
               label=r"true $D^{(2)}(x) = 0.7$")
    ax.plot(x, result["D2_kbr"], "rx", ms=10, mew=1.5,
            label=f"KBR, N={_fmt(n_kbr)}")
    ax.plot(x, result["D2_okbr"], "g+", ms=10, mew=1.5,
            label=f"OKBR, N={_fmt(n_okbr)}")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"diffusion $D^{(2)}(x)$")
    ax.legend(fontsize=9, loc="upper center")
    ax.set_ylim(0.0, 1.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-kbr", type=int, default=None,
                        help="N for the offline KBR run")
    parser.add_argument("--n-okbr", type=int, default=None,
                        help="N for the OKBR streaming run")
    parser.add_argument("--full", action="store_true",
                        help="paper sizes: N_kbr=5e7, N_okbr=1e10")
    parser.add_argument("--smoke", action="store_true",
                        help="CI sizes: N_kbr=5e6, N_okbr=5e8")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.full:
        n_kbr, n_okbr = 50_000_000, 10_000_000_000
    else:
        n_kbr, n_okbr = 5_000_000, 500_000_000
    if args.n_kbr is not None:
        n_kbr = args.n_kbr
    if args.n_okbr is not None:
        n_okbr = args.n_okbr

    result = run(n_kbr, n_okbr, seed=args.seed)
    out_solo = plot(result, FIGURES_DIR / "figure_2_python.png")
    print(f"  wrote {out_solo}")


if __name__ == "__main__":
    main()
