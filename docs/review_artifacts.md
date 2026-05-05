# Review artifacts

This document is a navigation index for an independent post-hoc code review.
A reviewing agent should be able to start here and reach every artifact needed
to evaluate the implementation, with no prior conversation context.

## What this package is

A Python port of [`OnlineMoments.jl`](../OnlineMoments.jl/), implementing the
streaming `O(1)`-space estimator for conditional moments of stochastic-process
increments described in *Davis 2023, PRE 108 054110* (arXiv:2307.00445).

## Reference materials

- **Paper PDF**: `paper/2307.00445v3.pdf`
- **Paper LaTeX source**: `paper/manuscript.tex`
- **Algorithm summary** (written to support the port): `paper/UNDERSTANDING.md`
- **Theory restated as code**: `docs/theory.md`
- **Intentional differences from Julia**: `docs/compared_to_julia.md`

## Implementations

- **Julia reference**: `OnlineMoments.jl/src/`
  - Streaming kernel-based: `OKBR.jl`, `OKBR_uncorrected.jl`
  - Streaming histogram-based: `OHBR.jl`, `OHBR_uncorrected.jl`, `OHBR_welford.jl`
  - Offline references: `KBR.jl`, `KBR_uncorrected.jl`, `HBR.jl`, `HBR_uncorrected.jl`
  - Periodic state space: `XKBR_mod.jl`, `XKBR_mod_uncorrected.jl`
  - Primitives: `kernels.jl`, `statistics.jl`, `utils.jl`
- **Python port**: `src/online_moments/`
  - Streaming: `online/okbr.py`, `online/ohbr.py`, hot loop in `online/_inner_loop.py`
  - Offline: `offline/kbr.py`, `offline/hbr.py`
  - Periodic: `modulo/okbr.py`, `modulo/ohbr.py`, `modulo/_inner_loop.py`
  - Autocorrelation: `autocorr.py`
  - Primitives: `kernels.py`, `binning.py`, `ringbuffer.py`, `statistics.py`
  - Reductions to `D^(k)`: `reductions.py`

## Tests

- `tests/unit/` — primitives (kernels, binning, ringbuffer, statistics, autocorrelation, reductions)
- `tests/offline/` — offline `hbr_moments` and `kbr_moments` shape and basic-property tests
- `tests/online/` — the headline equivalence tests:
  - `test_ohbr_vs_offline.py` — `OHBR.update_batch(X) == hbr_moments(X)` element-wise (exact)
  - `test_okbr_vs_offline.py` — `OKBR.update_batch(X) ≈ kbr_moments(X)` to `rtol=1e-10`
  - `test_hbr_eq_kbr_boxcar.py` — HBR matches KBR-Boxcar at bin centers
  - `test_first_slice_equals_single_tau.py` — multi-τ row 0 matches single-τ run
  - `test_non_consecutive_tau.py` — regression test for the Julia `tau_i` bug
- `tests/modulo/` — periodic state space translational-invariance tests

Test fixture: `tests/data/X_data_small.npy` (100 floats, ported from
`OnlineMoments.jl/test/X_data_small.jld2`).

## Visual evidence

- `validation/figures/figure_1_python.png` — Python OU-process result (drift, diffusion)
- `validation/figures/figure_1a_side_by_side.png` — Python panel a next to `paper/figure-1a.pdf`
- `validation/figures/figure_1b_side_by_side.png` — Python panel b next to `paper/figure-1b.pdf`
- `validation/figures/figure_2_python.png` — Python tri-stable result
- `validation/figures/figure_2a_side_by_side.png` — Python panel a next to `paper/figure-2a.pdf`
- `validation/figures/figure_2b_side_by_side.png` — Python panel b next to `paper/figure-2b.pdf`
- `validation/figures/scaling.png` — Python reproduction of the paper's Table 1 (time + memory vs N)
- `validation/figures/scaling.csv` — raw timing/memory numbers backing `scaling.png`

These are produced by `bash validation/run_smoke.sh` (CI-sized; ~3 min) or by
running the individual drivers with `--full` for paper-sized N (manual; minutes
to hours per driver depending on N).

## Recommended review prompt

A fresh reviewing agent should be given roughly:

> Compare `src/online_moments/online/okbr.py` and
> `src/online_moments/online/_inner_loop.py::okbr_update` against
> `OnlineMoments.jl/src/OKBR.jl` and §III of `paper/2307.00445v3.pdf`.
>
> Verify:
> (a) the kernel-weighted update of `M^(k)_{ij}` matches paper eq. 14;
> (b) the variance accumulator update matches paper eq. 16 (the
>     kernel-weighted Welford recurrence with both old and new means);
> (c) the multi-τ ring-buffer logic correctly looks up `X_{N - τ}` for each τ
>     in `tau_indices` (this is the Julia-bug fix described in
>     `docs/compared_to_julia.md` §2);
> (d) `tests/online/test_okbr_vs_offline.py` actually exercises the streaming
>     update path (not a tautology calling the same function);
> (e) `validation/figures/figure_1_side_by_side.png`,
>     `figure_2_side_by_side.png`, and `scaling.png` visually match the
>     corresponding paper panels and the paper's Table 1 — flag any
>     systematic deviation.
>
> Report: substantive issues (correctness, scientific accuracy) first;
> stylistic suggestions second; positive observations last. If the
> implementation is correct, say so plainly.
