# Convergence & Cost

How to read what the engine reports: what "converged" means for each problem class, which
stopping criterion actually certifies it, how to estimate an empirical convergence rate
from the logs, and which cost currency makes a comparison fair. This is the cross-cutting
companion to [Stopping Criteria](modules/stopping-criteria.md) and the per-method specs.

## What convergence means here

Two notions are tracked, and they are not interchangeable:

- **Stationarity** — a first-order optimality condition on the *current* iterate. For a
  smooth unconstrained problem this is `∇f(x) = 0`. For the composite problem
  `min_x f(x) + g(x)` it is `0 ∈ ∇f(x) + ∂g(x)`, which the smooth gradient `∇f(x)` alone
  does **not** capture — `∇f` is generally nonzero at the composite minimum.
- **Distance to a known optimum** — `‖x − x*‖`, available only when the problem generator
  supplies `x_opt` (the runner then fills `dist_to_opt`).

### The gradient mapping (composite problems)

The correct stationarity measure for `f + g` is the **gradient mapping**

    G_γ(x) = ( x − prox_{γg}( x − γ∇f(x) ) ) / γ

`x` is stationary for `f + g` iff `G_γ(x) = 0`. A proximal-gradient step is exactly
`x⁺ = x − γ·G_γ(x)`, so the step displacement `‖x⁺ − x‖ = γ·‖G_γ(x)‖` is a scaled
gradient-mapping norm — it vanishes precisely at a composite stationary point. This is why
`ProximalGradient` reports `step_norm = ‖x⁺ − x‖` as its convergence proxy and why its
`gradient_norm` (the smooth part `‖∇f(y)‖`, kept only because it is free) is *not* a
stationarity certificate. See the [metric-fields note](@ref metric-fields).

## Which stopping criterion certifies convergence

The engine's criteria fall into two classes: **budgets** (always safe to include; they
bound work, not optimality) and **convergence tests** (valid only for the problem classes
whose optimality condition they actually measure).

| Criterion | Measures | Valid convergence test for | Notes |
| --- | --- | --- | --- |
| `MaxIterations` | iteration count | — (budget) | safety cap; always safe to include |
| `TimeLimit` | accumulated **core** time | — (budget) | core-time, never wall-clock |
| `GradientTolerance` | `‖∇f(x)‖` (smooth part) | **smooth, unconstrained** only | misleading on composite problems — `‖∇f‖` need not vanish at the optimum. (For `SteihaugCG` it reads the model **residual**, which is its intended use.) |
| `StepTolerance` | `‖x_{k+1} − x_k‖` | **composite** (gradient-mapping proxy) and smooth | the right default for `ProximalGradient`; on smooth problems can trip early during a slow crawl |
| `DistanceToOptimal` | `‖x − x*‖` | any problem **with known `x_opt`** | inert (`Inf`) when `x_opt` is unset; see the lasso caveat below |
| `ObjectiveStagnation` | objective change over a window | weak / auxiliary | a plateau is not a proof of optimality (flat regions trigger it); pair with another test |

**Rules of thumb.**

- Smooth unconstrained (Rosenbrock, least squares): `GradientTolerance` and/or
  `DistanceToOptimal`.
- Composite (`f + g`, e.g. lasso): `StepTolerance` (gradient mapping) — *not*
  `GradientTolerance`.
- Always pair a convergence test with a `MaxIterations` (and optionally `TimeLimit`)
  budget via `stop_when_any(...)`.

**`x_opt` is not always the minimizer.** For the lasso family `x_opt` is the *planted*
signal, not the lasso solution, so `DistanceToOptimal` will not reach `0`. Use it as a
recovery reference, and converge on `StepTolerance` or an `f − f*` estimate instead.

## Reading an empirical convergence rate

Rates are read off the logs as slopes, not asserted:

- **Rate vs. conditioning** (the `ls2` conditioning sweep). Plot iterations-to-tolerance
  against `κ` on **log–log** axes; the slope is the exponent. Slope ≈ 1 means `O(κ)`
  (Fixed / Armijo / Cauchy on a quadratic); slope ≈ ½ means `O(√κ)` (Barzilai–Borwein) —
  the slope *difference* is the result.
- **Rate vs. iteration** (ISTA → FISTA on the lasso). Plot `f(x_k) − f*` on a **log-y**
  axis against `k`; a `−1` log–log slope is `O(1/k)` (ISTA), a `−2` slope is `O(1/k²)`
  (FISTA). Because `f*` is unknown, estimate it from a long reference run (do **not** use
  `f(x_star)`), then plot the gap.
- **Caveat — fit the asymptotic regime.** Early iterations are pre-asymptotic; estimate
  the slope on the straight portion of the curve, after transients and before the
  floating-point floor.

## Cost model: three currencies

"Is it faster?" has three honest answers, and they can disagree:

| Currency | What it captures | Caveat |
| --- | --- | --- |
| **Iterations** | algorithmic progress per step | unfair across methods with different per-step cost (Armijo does several `f`-evals per step; BB does one `∇f`) |
| **Core time** (`core_time_ns`) | actual compute in the math kernel | the engine's `@core_timed` measure; grows with problem size, so compare within a fixed problem |
| **Oracle calls** | `value` / `grad!` / Hessian-vector evaluations | the optimization-native unit; **opt-in** via `ExperimentConfig.count_oracles` → cumulative `:n_value` / `:n_grad` / `:n_hvp` in each log entry (line-search `f`-evals are also tracked separately via `n_linesearch_evals`) |

**Fairness guidance.** When methods differ in per-step oracle cost, compare by **core
time** or by **evaluations**, never by raw iteration count — an iteration of Armijo-GD and
an iteration of BB are not the same unit of work. The portfolio figures plot convergence
against iterations, core time, *and* (where relevant) evaluations for exactly this reason.

Set `ExperimentConfig.count_oracles = true` to record cumulative `value` / `grad!` /
Hessian-vector counts (`:n_value` / `:n_grad` / `:n_hvp`) in each log entry; the counter is
transparent and captures nested-solver work too (see
[oracle counting](modules/problem-interface.md#Oracle-counting-(opt-in-instrumentation))). It is opt-in so the default core-time
path stays unperturbed. Plot any convergence metric against `:n_grad` (or the sum) for an
implementation-independent comparison.

## On the core-time measurement

`core_time_ns` is the `@core_timed` kernel measure, and the reported `core_time / wall_time`
ratio is itself a result. Two confounds matter when interpreting it — garbage-collection /
allocation inside a timed block, and first-call JIT compilation — both detailed under
**Caveats** in [Algorithm, Core Timing & the Runner](modules/algorithm-core.md).
