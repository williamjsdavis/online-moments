"""Reproduce the paper's Table 1 (time + space scaling) as a figure.

Runs OKBR over a range of N values up to N=10^10 (the paper's largest size),
records wall-clock time and the OKBR accumulator footprint. For N <= 10^7 the
trajectory is materialised as a single array; for larger N the chunked
simulator streams 10^7-sized chunks straight into ``OKBR.update_batch`` so
the time series is never fully materialised in memory (a 10^10 float64 array
would be 80 GB).

Smoke default: N = 1e4 ... 1e7 (~30 s).
``--full``:    N = 1e4 ... 1e10 (~30-60 min depending on hardware).
"""
from __future__ import annotations

import argparse
import time
import tracemalloc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from online_moments import OKBR, Epanechnikov

from _common import (
    FIGURES_DIR,
    simulate_ou,
    simulate_ou_chunk,
    stream_simulator,
)


# Above this threshold we use the chunked streaming simulator to avoid
# allocating an array of size n. 1e7 floats = 80 MB; 1e8 = 800 MB; 1e9 = 8 GB
# (uncomfortable); 1e10 = 80 GB (impossible).
_CHUNK_THRESHOLD = 50_000_000


def benchmark_one(n: int, seed: int) -> dict:
    dt = 1e-3
    x_eval = np.linspace(-5.0, 5.0, 26)
    tau = np.array([1], dtype=np.int64)
    h = 0.4

    print(f"  N={n:.0e} ...", end=" ", flush=True)

    tracemalloc.start()
    okbr = OKBR(x_eval, tau, kernel=Epanechnikov(), bandwidth=h)
    okbr_state_bytes = (
        okbr.W.nbytes + okbr.M1.nbytes + okbr.M2.nbytes
        + okbr._ring.nbytes + okbr.x_eval.nbytes + okbr.tau_indices.nbytes
    )

    t0 = time.perf_counter()
    if n <= _CHUNK_THRESHOLD:
        X = simulate_ou(n, dt, x0=0.0, seed=seed)
        okbr.update_batch(X)
    else:
        chunk_size = 10_000_000
        for chunk in stream_simulator(
            simulate_ou_chunk, n_total=n, dt=dt, x0=0.0, seed=seed,
            chunk_size=chunk_size,
        ):
            okbr.update_batch(chunk)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"{elapsed:.2f}s  state={okbr_state_bytes}B  trace_peak={peak}B")
    return dict(N=n, time_s=elapsed, okbr_state_bytes=okbr_state_bytes,
                tracemalloc_peak=peak)


def run(n_values: list[int], seed: int = 1) -> list[dict]:
    print("warming JIT (small run)...")
    benchmark_one(10_000, seed=seed)  # JIT warmup; result discarded
    print("running benchmark...")
    return [benchmark_one(n, seed=seed) for n in n_values]


def plot(rows: list[dict], out_path: Path) -> Path:
    Ns = np.array([r["N"] for r in rows], dtype=np.float64)
    times = np.array([r["time_s"] for r in rows], dtype=np.float64)
    state = np.array([r["okbr_state_bytes"] for r in rows], dtype=np.float64)

    log_n = np.log10(Ns)
    log_t = np.log10(times)
    slope = np.polyfit(log_n, log_t, 1)[0]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.loglog(Ns, times, "o-", color="C0", label=f"OKBR (slope={slope:.2f})")
    # Reference O(N) line through the second point (skip the first which has JIT noise)
    ref_idx = 1 if len(Ns) > 1 else 0
    ax.loglog(Ns, times[ref_idx] * Ns / Ns[ref_idx], "k--", lw=0.7,
              label="O(N) reference")
    ax.set_xlabel("N (samples)")
    ax.set_ylabel("wall-clock time (s)")
    ax.set_title("Time scaling — should be linear (slope ≈ 1.0)")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    ax.semilogx(Ns, state / 1024, "o-", color="C2",
                label="OKBR accumulator state")
    ax.set_xlabel("N (samples)")
    ax.set_ylabel("OKBR state size (KB)")
    ax.set_title("Memory scaling — should be flat (O(1) in N)")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(0, max(state) / 1024 * 1.5 + 1)

    fig.suptitle(
        f"Reproduction of paper Table 1: OKBR scaling (N up to {int(Ns.max()):.0e})",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="run up to N=1e10 (slow)")
    parser.add_argument("--smoke", action="store_true",
                        help="N=1e4..1e7 (CI default)")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    if args.full:
        ns = [10**k for k in range(4, 11)]  # 1e4 ... 1e10
    else:
        ns = [10**k for k in range(4, 8)]   # 1e4 ... 1e7

    rows = run(ns, seed=args.seed)
    csv_path = FIGURES_DIR / "scaling.csv"
    with open(csv_path, "w") as f:
        f.write("N,time_s,okbr_state_bytes,tracemalloc_peak\n")
        for r in rows:
            f.write(f"{r['N']},{r['time_s']:.6g},"
                    f"{r['okbr_state_bytes']},{r['tracemalloc_peak']}\n")
    print(f"wrote {csv_path}")

    out = plot(rows, FIGURES_DIR / "scaling.png")
    print(f"wrote {out}")

    Ns = np.array([r["N"] for r in rows])
    times = np.array([r["time_s"] for r in rows])
    slope = np.polyfit(np.log10(Ns), np.log10(times), 1)[0]
    print(f"\nlog-log time slope: {slope:.2f} (expect ~1.0)")
    state = [r["okbr_state_bytes"] for r in rows]
    print(f"OKBR state bytes across run: min={min(state)} max={max(state)}")
    print("(state size should be identical across N — O(1) memory.)")


if __name__ == "__main__":
    main()
