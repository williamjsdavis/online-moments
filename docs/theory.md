# Theory: streaming conditional moments and Kramers–Moyal coefficients

A concise restatement of the mathematics implemented in this package, with
references back to *Davis 2023, PRE 108 054110* (`paper/2307.00445v3.pdf`) and
to a longer summary at `paper/UNDERSTANDING.md`.

## 1. Setup

Given a sample `X(t)` of a scalar process satisfying the Itô-interpreted
Langevin SDE

$$
\frac{dX}{dt} = f(X) + g(X)\,\Gamma(t),
\qquad \langle\Gamma(t)\Gamma(t')\rangle = \delta(t-t'),
$$

the Fokker–Planck equation has Kramers–Moyal coefficients

$$
D^{(k)}(x) = \lim_{\tau \to 0} \frac{1}{k!\,\tau} \int (x' - x)^k\, p(x', t+\tau \mid x, t)\, dx'.
$$

`f(x) = D^(1)(x)` (drift), `g(x) = √(2 D^(2)(x))` (diffusion). The package
estimates `D^(k)` non-parametrically on a user-chosen evaluation grid `𝒳 = [x_1, …, x_{N_x}]`.

## 2. Conditional moments

The bridge between data and `D^(k)` is the conditional moment

$$
M^{(k)}(\tau, x) = \int (x' - x)^k\, p(x', t+\tau \mid x, t)\, dx',
$$

evaluated on a τ-grid `𝒯 = [Δt, 2Δt, …, N_τ Δt]`. `M̂^{(k)}` is therefore a
`(N_τ × N_x)` matrix.

For τ small, `M^{(k)} ≈ k! · D^{(k)} · τ`. We recover `D^{(k)}` either by
direct estimation (single τ, divide) or OLS regression in τ (zero-intercept
slope).

## 3. Kernel-based estimator

Conditioning on `x = 𝒳_j` is achieved with a kernel `K_h(·)`:

$$
\hat{M}^{(k)}_{ij} = \frac{\sum_{n=1}^{N-i} K_h(\mathcal{X}_j - X_n)\,(X_{n+i} - X_n)^k}{\sum_{n=1}^{N-i} K_h(\mathcal{X}_j - X_n)}.
$$

`K_h(x) = K(x/h)/h`. Replacing `K_h` with a hard bin indicator gives the
histogram form (HBR). The conditional variance is

$$
\hat{M}^{(2^*)}_{ij} = \frac{\sum K_h(\cdot) \, (\Delta X - \hat{M}^{(1)})^2}{\sum K_h(\cdot)}.
$$

## 4. Streaming update rules (paper §III)

The contribution of the paper is `O(1)`-space updating: each cell maintains
cumulative weight `W_{ij}`, conditional mean `M̂^{(k)}_{ij}`, and
sum-of-squared-deviations `S_{ij}`.

Cumulative weight (paper eq. 13):

$$
W_{ij}\big|_N = W_{ij}\big|_{N-1} + K_h(\mathcal{X}_j - X_{N-i}).
$$

Conditional mean (paper eq. 14, kernel-weighted Welford):

$$
\hat{M}^{(k)}_{ij}\big|_N = \hat{M}^{(k)}_{ij}\big|_{N-1} + K_h(\cdot)\,\frac{(X_N - X_{N-i})^k - \hat{M}^{(k)}_{ij}\big|_{N-1}}{W_{ij}\big|_N}.
$$

Conditional variance (paper eq. 16, kernel-weighted Welford):

$$
S_{ij}\big|_N = S_{ij}\big|_{N-1} + K_h(\cdot)\,((X_N - X_{N-i}) - \hat{M}^{(1)}\big|_{N-1})\,((X_N - X_{N-i}) - \hat{M}^{(1)}\big|_N),
$$

with `M̂^(2*) = S / W`.

For HBR, replace `K_h(·)` by the indicator `[X_{N-i} ∈ bin_j]` and `W` by an
integer count `N_{ij}`. All updates collapse to standard streaming-mean /
streaming-variance recurrences.

## 5. Where each piece lives in the code

| Math | Code |
|---|---|
| `K_h(x)`, kernel definitions | `online_moments/kernels.py` (Python), `online/_inner_loop.py::_kernel_value` (Numba) |
| Bin lookup `[X ∈ bin_j]` | `online_moments/binning.py::find_bin`, `online/_inner_loop.py::_find_bin` |
| Cumulative weight `W_{ij}` | `OKBR.W`, updated in `okbr_update` |
| Conditional mean update | `OHBR.M1` / `OKBR.M1` updated in inner-loop functions |
| Welford `S_{ij}` for variance | Embedded in `okbr_update` (variance branch) and `ohbr_update` |
| Direct estimation `D^{(k)} = M̂^{(k)} / (k! · τ · Δt)` | `reductions.kramers_moyal_regression` (single-τ path) |
| Regression τ → 0 | `reductions.kramers_moyal_regression` (multi-τ path) |

## 6. Numerical guarantees

- HBR (online and offline) uses identical Welford updates in identical order,
  so `OHBR.update_batch(X)` and `hbr_moments(X)` produce **bit-identical**
  accumulator state. This is asserted by `tests/online/test_ohbr_vs_offline.py`.
- KBR's offline path accumulates `Σ K dx` and divides at the end; the online
  path is a kernel-weighted Welford. The two are mathematically equal but
  differ at last-bit precision (`rtol=1e-10` is the test bar).
- `fastmath=False` is enforced in all `@njit` decorators because the variance
  recurrence requires the new mean to be computed before the variance step.
