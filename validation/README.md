# Validation drivers

Reproduce the figures and Table 1 from *Davis 2023, PRE 108 054110*
([DOI:10.1103/PhysRevE.108.054110](https://doi.org/10.1103/PhysRevE.108.054110)).

Each driver runs end-to-end: simulate → stream through OKBR → compute drift
and diffusion → save a PNG.

## Quick smoke (CI sizes, ~3 min total)

```bash
bash validation/run_smoke.sh
```

Generates:

- `figures/figure_1_python.png` (OU process drift+diffusion at N=1e6 + N=1e8)
- `figures/figure_2_python.png` (tri-stable system at N=5e6 + N=5e8)
- `figures/scaling.png` (time + memory vs N, N=1e4..1e7)
- `figures/scaling.csv` (raw timings backing `scaling.png`)

## Full-sized reproductions (manual, longer)

```bash
python validation/figure_1_ou.py        --full   # N=1e7 + N=1e10, ~10 min
python validation/figure_2_tristable.py --full   # N=5e7 + N=1e10, ~40 min
python validation/figure_table_1_scaling.py --full   # N=1e4..1e10, ~10 min
```

The `N=1e10` traces use a chunked simulator so the time series is never fully
materialised in memory (a 1e10 float64 array would be 80 GB).

## Notes on what to look for

- **Figure 1 (OU)**: drift recovers $D^{(1)}(x) = -x$ exactly in the central
  region; diffusion is flat at 1.0. The small-N traces (KBR and OKBR overlaid)
  fail outside $|x| \gtrsim 4$ — the large-N OKBR trace fills in those tails.
- **Figure 2 (tri-stable)**: diffusion is flat at 0.7 across $[-1.2, 1.2]$.
  KBR-N=5×10⁷ scatters in the central $|x| < 0.5$ region; OKBR-N=10¹⁰ traces
  the weak central attractor cleanly. This is the headline scientific point
  of the paper.
- **scaling.png**: time linear in N (slope ≈ 1.0 in log-log); memory flat
  (the OKBR accumulator state is $\mathcal{O}(1)$ in N).

## Dependencies

`numpy`, `numba`, `matplotlib` are required. No other dependencies.
