# Differences from `OnlineMoments.jl`

This Python port is a *behavioral* port, not a bit-for-bit clone. The list below
is the complete set of intentional differences. None of them affect the
mathematical correctness of the result — they affect API ergonomics or
numerical conventions where the Julia code's choice was non-obvious.

## 1. Epanechnikov kernel convention

| | Form | Support |
|---|---|---|
| Julia (`src/kernels.jl`) | `(3·√5 / 100) · (5 - x²)` | `x² < 5` |
| Python (`kernels.py`) | `(3/4) · (1 - x²)` | `|x| < 1` |

The Julia form is variance-1 normalised (kernel σ² = 1). The Python form is the
textbook one used by `scipy.stats.epanechnikov`, every kernel-density-estimation
reference, and the original Epanechnikov 1969 paper.

**Practical consequence**: outputs at fixed `h` differ. To compare numerically,
use `h_python = h_julia / √5`. Documented here once; not a bug.

## 2. Non-consecutive `tau_indices` are supported

The Julia types declare `tau_i::UnitRange{Int}` and look up lagged samples by
buffer position (`mem[i_tau]`), which silently misbehaves for any
`tau_indices` that isn't `[1, 2, …, N_τ]`. The codebase has a `#TODO: generalize
tau_i` left at the top of `OHBR_multiple` etc.

The Python implementation sizes the ring buffer to `max(tau_indices)` and looks
up `ring[tau - 1]` for each individual lag. Arbitrary positive ascending integer
lags work. `tests/online/test_non_consecutive_tau.py` is a regression for this
fix.

## 3. Bin-edge inclusivity

Julia `find_bin` rounds via `floor`, so a value exactly on `edges[-1]` falls
into bin `N_x + 1` (one past the last). The implementation never trips on this
because `in_range` excludes such points anyway, but it's a quiet trap.

Python `find_bin` is half-open `[edge_i, edge_{i+1})` for all bins except the
last, which is closed `[edge_{N_x-1}, edge_{N_x}]`. Implemented as
`np.searchsorted(edges, x, side="right") - 1` clamped at the right edge. Tested
explicitly.

## 4. Streaming entry point: `update`, not `add_data!`

Convention is `update(x)`. Mass-update is `update_batch(X)`. Bang suffix is
inappropriate in Python (mutation is the default for methods that don't return
`self`).

## 5. Single class for single-τ and multi-τ

Julia has separate `OHBR_single` / `OHBR_multiple` types because Julia's type
system makes scalar-vs-vector dispatch cheap. In Python, a single class with
`tau_indices = np.array([k])` for the single-τ case is clearer and removes a
class-explosion problem. Single-τ is the special case `len(tau_indices) == 1`;
the row-0 of multi-τ M1 is identical to a single-τ run with `tau_indices=[1]`.

## 6. Cold-start sentinel

Julia uses `NaN` in the ring-buffer entries to mark "no data yet". Python uses
an integer counter `_n_pushed` and skips updates while `_n_pushed < tau_i`.
Cleaner; no risk of NaN propagating.

## 7. Out-of-bin sentinel

Julia returns `0` (Julia is 1-indexed; `0` is "no bin"). Python returns `-1`,
which is the more conventional Python sentinel. Internal only; not user-visible.

## 8. Algorithm A/B/C variants are collapsed into one offline reference

Julia ships `HBR_moments_A`, `HBR_moments_B`, `HBR_moments_C`, `HBR_moments_C2`
— four equivalent implementations cross-checked against each other. The Python
port keeps a single `hbr_moments` (Algorithm C: streaming-mean / streaming-variance
in one pass over X), which is the same arithmetic the streaming `OHBR` performs.
The online-vs-offline test in Tier 2 is the cross-check.

## 9. Welford-S form is a private detail

Julia exports `OHBR_welford_single` as a separate user-facing type. The Python
port does not — `moment_form="variance"` *uses* the Welford recurrence
internally, but exposes only `M2 = S/N`. The Welford accumulator is verified in
`tests/unit/test_statistics.py::test_update_var_matches_welford_form`.

## 10. Turbulence variants are deferred to a future v2

`OXBR_turbulence.jl` is not ported. Out of scope for v1.

## 11. Lehle–Peinke correlated-noise inversion is not ported

That machinery (used to reproduce paper Figure 3) belongs to a different paper
(Lehle & Peinke 2018) and is out of scope for an `OnlineMoments` port.
