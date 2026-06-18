# ProximalGradient — ISTA / FISTA

A composite-objective method for problems

    min_x  f(x) + g(x)

with `f` smooth (gradient available) and `g` "simple" (proximal operator
available). Lights up the composite branch of the framework: `prox` dispatch,
the `f + g` sum in `total_objective`, and the `Extrapolation` hierarchy.

## 1. Method

```julia
@kwdef struct ProximalGradient <: IterativeMethod
    step_size    :: StepSize    = FixedStep(α = 1e-3)   # use FixedStep(α = 1/L)
    extrapolation :: Extrapolation = NoExtrapolation()       # NesterovStep() ⇒ FISTA
end
```

- `extrapolation = NoExtrapolation()` ⇒ **ISTA** (plain proximal gradient).
- `extrapolation = NesterovStep()`  ⇒ **FISTA** (accelerated, `O(1/k²)`).
- `g = ZeroRegularizer` (or no regularizer) ⇒ reduces to (accelerated)
  gradient descent on the smooth `f`, so the same method also tells the
  smooth-acceleration story.

## 2. Iteration

Each `step!`, in order:

1. **Extrapolate.** `y = extrapolate(extrapolation, x, x_prev, t)`
   — FISTA: `y = x + β(x − x_prev)`, `β = (t−1)/t_next`; ISTA: `y = x`.
2. **Gradient of the smooth part at `y`.** `g = ∇f(y)` (one gradient eval/step).
3. **Step size.** `γ = compute_step_size(step_size, …)`. The supported rule is
   `FixedStep(α = 1/L)` with `L` the Lipschitz constant of `∇f` (the lasso
   generator stores `L = ‖A‖² ` in `meta[:L]`). Backtracking line search here
   would need a *proximal* sufficient-decrease test — not implemented; future
   work.
4. **Proximal step.** `xⁿ = prox(g_reg, y − γ·∇f(y), γ)` (identity if no
   regularizer). For `L1Norm(λ)` this is soft-thresholding at level `γλ`.
5. **Shift history.** `x_prev ← x`, `x ← xⁿ`.
6. **Advance momentum.** `t ← advance_momentum(extrapolation, t)`.

Exactly **one** gradient evaluation and **one** `prox` call per step.

## 3. State

`ProximalGradientState` composes the canonical `IterateGroup` / `MetricsGroup`
/ `TimingGroup` with `ProximalGradientNumerics`:

| field | role |
|---|---|
| `t` | FISTA momentum parameter `t_k` (init `1.0`) |
| `α_k` | last step size, logged |
| `direction` | `−∇f(y)`, handed to the step-size rule |
| `x_trial`, `grad_prev`, `n_linesearch_evals` | buffers the `StepSize` rules expect |

## 4. Implementation contract

The three dispatch points (`init_state` / `step!` / `extract_log_entry`) over the
shared algorithm interface (`docs/src/modules/algorithm-core.md`).

**`init_state`.** Copies `x₀`, evaluates `∇f(x₀)` and `total_objective(x₀)` once, and
sets `t = 1.0`. It enforces the one-regularizer restriction (§6) *up front*: more than
one `g` raises an `ArgumentError` before any iteration runs. `dist_to_opt` is left
`Inf` for the runner to fill from `problem.x_opt`.

**`step!` — timing discipline.** Two `@core_timed` blocks bracket the math; everything
else is bookkeeping the clock never sees:

| Operation | Timed? |
|---|---|
| `extrapolate` → `y`, `grad(f, y)`, `direction = −g` | yes |
| `compute_step_size` | inside the rule (self-timed) — `step!` does not wrap it |
| forward step `y − γ∇f(y)`, `prox`, history shift, `total_objective` | yes |
| `advance_momentum`, `gradient_norm`, metric writes | no |

Postconditions mirror the GD contract: `x` holds `x_{k+1}`, `x_prev` holds `x_k`, and
`α_k` / `t` are stored for logging. The reported **gradient vector** `state.iterate.gradient`
is the smooth part at the evaluation point `y` (reused from the step's single eval); the
reported **`gradient_norm`** is the gradient-mapping norm `‖G_γ(y)‖`, a valid composite
stationarity residual (see §5).

**`extract_log_entry`.** Adds `:step_size` and `:t` to `extras`, plus `:x_iter` (a copy
of the iterate) when `dim ≤ 2`, for trajectory plots.

## 5. Metrics & conventions

- `objective = total_objective(p, x) = f(x) + g(x)` (the composite value).
- `gradient_norm = ‖G_γ(y_k)‖ = ‖(y_k − xⁿ)/γ‖` — the norm of the (prox-)gradient
  **mapping** at the evaluation point. This is the proper composite-stationarity
  residual: `G_γ(y) → 0` iff `0 ∈ ∇f(y) + ∂g(y)`, so `GradientTolerance` is a valid
  stopping test even on composite problems. Computed from `y`, `xⁿ`, `γ` already in
  hand — **no extra gradient or prox eval**. With no regularizer the prox is the
  identity and `G_γ(y) = ∇f(y)`, recovering the smooth-case `‖∇f(y)‖` exactly.
- `step_norm = ‖xⁿ − x‖` (for ISTA, `= γ‖G_γ(x_k)‖`; an equivalent stationarity proxy).

## 6. Restrictions

- Supports **0 or 1** regularizer. A sum of several nonsmooth terms needs
  operator splitting and raises an `ArgumentError`.

## 7. Win conditions (lasso experiment)

- `prox` called once per step with step `γ`; `total_objective` sums `f + g`.
- FISTA's `f − f*` curve visibly beats ISTA's (acceleration). On this well-conditioned
  instance both converge linearly once the support is identified; the sublinear
  `O(1/k)`-vs-`O(1/k²)` slope separation is measured on a dedicated non-strongly-convex
  instance in `test/test_proximal_gradient.jl`.
- At sufficiently large `λ`, recovered support ⊆ true support; off-support
  coordinates are within one soft-threshold (`|x_i| ≤ γλ`) of zero.

These are executed, not narrated: the experiment's `validate()` asserts the decomposition,
the acceleration, and exact support recovery, and the load-bearing claims — decomposition,
acceleration, one-prox-per-step, support recovery, the gradient-mapping stop, and the
`O(1/k)`-vs-`O(1/k²)` rate separation — are pinned in `test/test_proximal_gradient.jl`.
