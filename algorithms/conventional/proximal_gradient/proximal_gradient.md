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

## 4. Metrics & conventions

- `objective = total_objective(p, x) = f(x) + g(x)` (the composite value).
- `gradient_norm = ‖∇f(y_k)‖` — the smooth-part gradient at the **evaluation
  point** (cheap; no extra eval). Not a composite stationarity certificate;
  use `step_norm` (the gradient-mapping proxy `‖xⁿ − x‖`) or
  `DistanceToOptimal` for convergence in experiments.
- `step_norm = ‖xⁿ − x‖`.

## 5. Restrictions

- Supports **0 or 1** regularizer. A sum of several nonsmooth terms needs
  operator splitting and raises an `ArgumentError`.

## 6. Win conditions (lasso experiment)

- `prox` called once per step with step `γ`; `total_objective` sums `f + g`.
- FISTA's `f − f*` curve visibly beats ISTA's (`O(1/k²)` vs `O(1/k)`).
- At sufficiently large `λ`, recovered support ⊆ true support; off-support
  coordinates are within one soft-threshold (`|x_i| ≤ γλ`) of zero.
