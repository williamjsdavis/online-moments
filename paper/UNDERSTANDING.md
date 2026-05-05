# Understanding of OnlineMoments / OKBR

This document summarizes my understanding of the paper *Reconstruction of Stochastic Dynamics from Large Streamed Datasets* (Davis 2023, PRE 108 054110, arXiv:2307.00445v3) and the accompanying `OnlineMoments.jl` reference implementation. The goal is to record the algorithmic and implementation context that the planned Python rewrite will draw on.

## 1. The problem

Given a single time series `X(t)` sampled at regular interval `Δt`, model it as the scalar Langevin SDE

```
dX/dt = f(X) + g(X) Γ(t),     ⟨Γ⟩ = 0,  ⟨Γ(t)Γ(t')⟩ = δ(t - t')
```

(Itô interpretation). The transition density satisfies the Fokker–Planck equation, whose coefficients are the Kramers–Moyal (KM) coefficients

```
D^(k)(x) = lim_{τ→0}  (1 / k! τ)  ∫ (x' - x)^k  p(x', t+τ | x, t)  dx'
```

with `f(x) = D^(1)(x)` (drift) and `g(x) = √(2 D^(2)(x))` (diffusion). The aim is to recover non-parametric estimates of `D^(1)` and `D^(2)` on a user-chosen grid of evaluation points `𝒳 = [x_1, …, x_{N_x}]`.

The bridge between data and KM coefficients is the *finite-time conditional moment*

```
M^(k)(τ, x) = ∫ (x' - x)^k  p(x', t+τ | x, t)  dx'
```

evaluated on a τ-grid `𝒯 = [Δt, 2Δt, …, N_τ Δt]`. The result is a `N_τ × N_x` matrix `M̂^(k)`.

KM coefficients are then read off by:
- *Direct estimation* — `D̂^(k) = M̂^(k) / (k! Δt)` for a single small τ;
- *Linear regression in τ* — `D̂^(k) = argmin ‖M̂^(k) − 𝒯 D̂^(k)‖²`, taking the `τ → 0` slope. This is preferred when several τ values are sampled (used in the tri-stable and multiplicative-noise examples).

A correlated-noise variant ("Lehle–Peinke") fits `M̂^(k)` to a basis `r_i(τ; θ)` instead of a straight line, recovering `D^(k)` from the leading basis coefficient. This requires *both* the conditional mean (`k=1`) and the conditional variance (`k=2*`).

## 2. Two estimator families: HBR and KBR

Both families produce the same shape of output (`M̂^(1)`, `M̂^(2)`, both `N_τ × N_x`) but differ in how they condition on `x`.

### Histogram-Based Regression (HBR)

Hard-bin `X_n` into one of `N_x` bins defined by an edge vector. For each `(τ_i, bin_j)`, average increments `(X_{n+i} − X_n)^k` over the data points whose `X_n` falls in bin `j`.

```
M̂^(k)_{ij} = (1 / N_{ij}) Σ_{n: X_n ∈ bin_j, n ≤ N − i}  (X_{n+i} − X_n)^k
```

### Kernel-Based Regression (KBR)

Replace the hard bin indicator with a kernel weight `K_h(𝒳_j − X_n) = K((𝒳_j − X_n)/h) / h`:

```
M̂^(k)_{ij} = [ Σ K_h(𝒳_j − X_n) (X_{n+i} − X_n)^k ]  /  [ Σ K_h(𝒳_j − X_n) ]
```

Two kernels are supported in the Julia code:
- **Epanechnikov** `K(x) = (3/4)(1 − x²)` for `|x| < 1` (default; numerically favorable);
- **Boxcar** `K(x) = 1/2` for `|x| < 1`.

Note: the Julia Epanechnikov is rescaled — the support written in code is `x² < 5` with `epan_scale = 3√5 / 100` rather than the standard `|x|<1, 3/4(1−x²)`. The Python rewrite should pick a single convention and document it.

### Conditional variance vs. raw second moment

The paper distinguishes two flavors of the second moment:

| Quantity | Formula | Code suffix |
|---|---|---|
| Raw second moment `M̂^(2)` | `E[ ΔX² ]` (centered at zero) | "uncorrected" (`HBRu`, `KBRu`, `OHBRu`, `OKBRu`) |
| Conditional variance `M̂^(2*)` | `E[ (ΔX − M̂^(1))² ]` (centered at running mean) | default (`HBR`, `KBR`, `OHBR`, `OKBR`) |

The "uncorrected" variants are simpler (one accumulator, no mean coupling) but estimate a different quantity. The Lehle–Peinke correlated-noise method needs the *variance* form.

## 3. The contribution: online updating formulas

The paper's central result is incremental update rules that turn the offline KBR/HBR estimators into `O(1)`-space streaming algorithms.

### Cumulative weights

```
W_{ij}|_N  =  W_{ij}|_{N-1}  +  K_h(𝒳_j − X_{N−i})
```

(For HBR, replace `K_h(...)` with the indicator that `X_{N−i}` falls in bin `j`, i.e. `W` becomes an integer count `N_{ij}`.)

### Conditional moment (mean form, k=1, 2)

```
M̂^(k)_{ij}|_N  =  M̂^(k)_{ij}|_{N-1}
              +  K_h(𝒳_j − X_{N−i}) · ( (X_N − X_{N−i})^k − M̂^(k)_{ij}|_{N-1} ) / W_{ij}|_N
```

This is the kernel-weighted analog of the textbook running-mean update `μ_n = μ_{n-1} + (x_n − μ_{n-1}) / n`.

### Conditional variance (k = 2*)

Maintain a sum-of-squared-deviations accumulator `S_{ij}|_N`, with `M̂^(2*)_{ij} = S_{ij} / W_{ij}`. The update (a kernel-weighted Welford / West formula) is:

```
S_{ij}|_N  =  S_{ij}|_{N-1}
           +  K_h(𝒳_j − X_{N−i})
              · ( (X_N − X_{N−i}) − M̂^(1)_{ij}|_{N-1} )
              · ( (X_N − X_{N−i}) − M̂^(1)_{ij}|_N     )
```

i.e. the increment is multiplied by the *old* and *new* mean residuals — the standard trick for numerically stable streaming variance.

The Julia code also carries an alternate "Welford" form (`OHBR_welford.jl`) using `S += (ΔX − M1_old)(ΔX − M1_new)` as a sanity check.

### Why this works, conceptually

Each `(τ_i, j)` slot is a kernel-weighted running average over a particular conditional sub-population (points whose `X_{n−i}` is near `𝒳_j`). The update rules are the standard streaming-mean and streaming-variance recurrences with the per-sample weight `K_h(𝒳_j − X_{N−i})` instead of `1`. Numerator and denominator decouple cleanly, and only the running statistics need to be stored.

### Lookback memory

The update at step `N` references `X_{N−i}` for every `i ∈ [1, N_τ]`. The streaming implementation therefore keeps a small ring buffer of the last `N_τ` `X` values (and, for HBR, the last `N_τ` bin indices). This is `O(N_τ)`, independent of `N`.

## 4. Complexity

| | KBR / HBR (offline) | OKBR / OHBR (online) |
|---|---|---|
| Space | `O(N)` (must hold all `X_n`) | `O(N_τ · N_x)` — independent of `N` |
| Time | `O(N · N_τ · N_x)` | `O(N · N_τ · N_x)` |

The paper's benchmarks (Table I) confirm linear time scaling for both methods, and constant memory for OKBR up to `N = 10¹⁰` (where KBR is no longer feasible).

The big practical consequence: OKBR resolves rarely-sampled tails of state space (transient states, heavy-tailed processes) by simply running on much larger datasets than KBR can hold in memory. This is shown in:
- the Ornstein–Uhlenbeck example (`D^(1) = −x`, `D^(2) = 1`) at `N = 10¹⁰`;
- the tri-stable polynomial system, where the weak attractor at `x = 0` is invisible at `N = 5·10⁷` and clearly visible at `N = 10¹⁰`;
- the multiplicative + Ornstein–Uhlenbeck-noise system, where the quartic coefficient of `D^(2)` is unresolvable at `N = 10⁷` but accurately recovered at `N = 5·10⁹`.

A turbulence application (Renner et al. air-jet dataset, hot-wire velocity) demonstrates a different application: conditional moments are taken not over the value `X` but over *velocity increments at scale r* — a two-scale, scale-cascade variant of the same machinery.

## 5. Implementation in `OnlineMoments.jl`

The Julia package is organized as a tree of variants. Names are systematic:

```
[O]  [H|K]  BR  [u]  [_mod | _turb]  [_single | _multiple]
 │    │     │    │    │                │
 │    │     │    │    │                └─ τ-grid: single τ=Δt vs. vector of τ
 │    │     │    │    └─ topology: linear, periodic (mod), or two-scale (turb)
 │    │     │    └─ "uncorrected" → raw second moment instead of variance
 │    │     └─ Based-Regression (always present)
 │    └─ kernel: H = histogram (hard bins), K = kernel (smooth)
 └─ optional "O" prefix → online (streaming) variant
```

So e.g. `OKBR_multiple` = online, kernel-based, multi-τ; `HBRu_mod_single` = offline, histogram, raw second moment, periodic state space, single τ.

### Module structure (`src/`)

| File | Contents |
|---|---|
| `OnlineMoments.jl` | module entry point, exports |
| `utils.jl` | bin lookup (`find_bin`, `in_range`), modulo bin helpers, ring-buffer `update_mem!` |
| `kernels.jl` | `Kernel` abstract type, `Epaneknikov`, `Boxcar`, `apply_kernel` |
| `statistics.jl` | streaming-mean / streaming-variance / weighted-mean / weighted-variance / weighted-sum-of-squares / Welford updates |
| `HBR.jl`, `HBR_uncorrected.jl` | offline HBR (variance and raw forms, three implementation variants A/B/C) |
| `KBR.jl`, `KBR_uncorrected.jl` | offline KBR |
| `OHBR.jl`, `OHBR_uncorrected.jl`, `OHBR_welford.jl` | online HBR estimator structs + `add_data!` |
| `OKBR.jl`, `OKBR_uncorrected.jl` | online KBR estimator structs + `add_data!` |
| `XKBR_mod.jl`, `XKBR_mod_uncorrected.jl` | periodic-state-space variants (HBR + KBR, both online and offline grouped together) |
| `OXBR_turbulence.jl` | turbulence two-scale increment variants |
| `autocorr.jl` | offline autocorrelation + thin wrapper around `OnlineStats.AutoCov` |

### Online estimator interface

Every online estimator is a small mutable struct holding accumulators plus a memory buffer, and exposes a single mutation entry point `add_data!(estimator, X_new)`. The single-τ form holds a scalar `mem::Float64`; the multi-τ form holds `mem::Vector{Float64}` of length `N_τ` (a stack/ring buffer maintained by `update_mem!`, plus, for HBR, a parallel `bin_mem` of cached bin indices to avoid recomputation).

KBR additionally caches `hinv = 1/h` so the kernel can be applied as `K_h(x) = hinv · K(hinv · x)` without per-step division.

A small post-hoc helper `M1τ`, `M2τ` divides the accumulated moments by `Δt · τ_i` for direct estimation; the user does the linear-regression-in-τ step themselves outside the estimator.

### Edge cases handled

- **Out-of-range `X_n`**: skipped (HBR returns bin `0`; KBR weight is zero on disjoint kernel support).
- **Cold start**: `mem` is initialized to `NaN`; the update is a no-op until enough data has been buffered. Multi-τ variants warm up over the first `N_τ` steps.
- **Periodic state space**: `find_mod_bin` and `d_mod` handle wraparound for angular/circular variables. The kernel form scales the kernel argument by the modulo distance.
- **Turbulence form**: instead of `(X_{n+i} − X_n)`, both axes are *velocity increments* `ξ_{n,i} − ξ_{n,0}`; the conditioning is on `ξ_{n,0}` rather than on `X_n`. Same update structure, different feature extraction.

## 6. What the Python rewrite needs to deliver (not yet planned, just noted)

The minimum useful surface area, judging from the paper's examples and the Julia API:
1. Two kernels (Epanechnikov, boxcar) with a clean bandwidth scaling.
2. Online estimators for both HBR and KBR, in single-τ and multi-τ forms, in both "variance" and "raw second moment" flavors.
3. Both an `add_data(x)` streaming API and a convenience batch entry point.
4. Reduction helpers: scaled moments `M^(k) / (k! · τ)` and a τ-regression for `D^(k)`.
5. (Probably scope-deferrable) modulo and turbulence variants.

Numerical-stability decisions worth carrying over: the kernel-weighted Welford-style variance update (the paper's `S_{ij}` recurrence), and caching `1/h` once per estimator.
