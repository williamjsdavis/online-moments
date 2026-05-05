# Validation drivers

Reproduce the figures and Table 1 from *Davis 2023, PRE 108 054110*.

Each driver runs end-to-end: simulate → stream through OKBR → compute drift
and diffusion → save a PNG, plus a side-by-side comparison against the paper's
PDF panel.

## Quick smoke (CI sizes, ~3 min total)

```bash
bash validation/run_smoke.sh
```

Generates:

- `figures/figure_1_python.png` (OU process drift+diffusion at N=1e6)
- `figures/figure_1a_side_by_side.png` and `figures/figure_1b_side_by_side.png`
- `figures/figure_2_python.png` (tri-stable system at N=5e6)
- `figures/figure_2a_side_by_side.png` and `figures/figure_2b_side_by_side.png`
- `figures/scaling.png` (time + memory vs N, N=1e4..1e7)
- `figures/scaling.csv` (raw timings backing `scaling.png`)

## Full-sized reproductions (manual, longer)

```bash
python validation/figure_1_ou.py        --full   # N=1e9, ~10 min
python validation/figure_2_tristable.py --full   # N=5e7, ~5 min
python validation/figure_table_1_scaling.py --full   # N=1e4..1e9, ~30 min
```

## Notes on what to look for

- **Figure 1 (OU)**: drift recovers `D^(1)(x) = -x` exactly in the central
  region; diffusion is flat at 1.0. Edges are noisy at small N because the
  process rarely visits |x| > 4.
- **Figure 2 (tri-stable)**: diffusion is flat at 0.7 across [-1.2, 1.2].
  Drift is well-resolved at the wells x = ±1; the central x ≈ 0 is sparsely
  sampled at any N below ~10⁹ (this is the headline scientific point of the
  paper — only OKBR with very large N resolves the weak attractor).
- **scaling.png**: time should be linear in N (slope ≈ 1.0 in log-log);
  memory should be flat (the OKBR accumulator state is `O(1)` in N).
- **Side-by-side panels**: the visual shape of the OKBR (green +) cloud
  should match the paper's. KBR (red ×) overlays OKBR exactly when both are
  run — that's the streaming-vs-offline equivalence visualised.

## Dependencies

- `numpy`, `numba`, `matplotlib` (always required)
- `pdf2image` plus `poppler` (for the side-by-side PDF rendering). On macOS:
  `brew install poppler`. If unavailable, the drivers skip the side-by-side
  panel and still produce the standalone Python figure.
