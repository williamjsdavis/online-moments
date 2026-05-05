# online_moments

A Python port of [`OnlineMoments.jl`](https://github.com/williamjsdavis/OnlineMoments.jl) — streaming, `O(1)`-space estimators
for conditional moments of stochastic-process increments. Recovers drift `D^(1)(x)` and
diffusion `D^(2)(x)` of a Langevin SDE from time-series data of arbitrary length, including
data streams that don't fit in memory.

Implements the method of *Davis 2023, PRE 108 054110* (`paper/2307.00445v3.pdf`).

## Quick start

```python
import numpy as np
from online_moments import OKBR, Epanechnikov

okbr = OKBR(
    x_eval=np.linspace(-5, 5, 26),
    tau_indices=np.array([1]),
    kernel=Epanechnikov(),
    bandwidth=0.4,
)

# stream
for x in data_stream:
    okbr.update(x)

# or batch
okbr.update_batch(X)

# recover drift and diffusion (Langevin / Kramers-Moyal coefficients)
D1, D2 = okbr.drift_diffusion(dt=1e-3, fit="regression")
```

## Install

```bash
pip install -e .[dev]
```

Requires Python 3.11+. The streaming inner loop uses Numba; first call has a
~3 second JIT cold-start, cached thereafter.

## Layout

- `src/online_moments/` — library
- `tests/` — unit tests, offline tests, online-vs-offline equivalence tests
- `validation/` — paper-figure reproduction (Figure 1 OU, Figure 2 tri-stable, scaling
  benchmark). Each driver renders a side-by-side PNG against the corresponding paper figure.
- `paper/UNDERSTANDING.md` — written summary of the algorithm
- `docs/compared_to_julia.md` — intentional differences from the Julia reference
- `OnlineMoments.jl/` — the Julia reference implementation (gitignored)

## Verification

```bash
pytest tests/ -v                          # unit + equivalence tests
bash validation/run_smoke.sh              # generates side-by-side figures (CI sizes)
python validation/figure_1_ou.py --full   # full N=1e9 reproduction (manual)
```
