# PreconditionedGradient (experimental)

A first-order method whose descent direction is **preconditioned**:

    x_{k+1} = x_k + α_k · d_k,    d_k = −M⁻¹ ∇f(x_k)

It crosses a **preconditioner** axis (`M⁻¹`) with a **step-size** axis (`α_k` along `d_k`),
and is the framework's showcase of the *signature workflow*: define **one** experimental
method, sweep its variants in a `VariantGrid` with `role = :experimental`, and let
`resolve_methods` route them into the *experimental* bucket against a baseline — all in a
single `run_experiment`. It drives dual-bucket routing, a genuine ≥2-axis Cartesian
grid, `DiagonalHessian`, and the Jacobi preconditioner.

## 1. Method

```julia
@kwdef struct PreconditionedGradient <: IterativeMethod
    preconditioner :: Preconditioner = IdentityPreconditioner()
    step_size      :: StepSize        = ArmijoLS()
end
```

- `preconditioner = IdentityPreconditioner()` ⇒ `M⁻¹ = I`, so `d = −∇f` — plain gradient
  descent.
- `preconditioner = JacobiPreconditioner()` ⇒ `d = −diag(∇²f)⁻¹ ∇f` — **exact Newton on a
  diagonal Hessian** (§6).
- `step_size` — any shared `StepSize` rule applied along `d` (see the BB caveat in §5).

Both fields are abstract, so any concrete `Preconditioner` / `StepSize` composes without
touching `step!`. See [preconditioners.md](../components/preconditioners.md) and
[step_sizes.md](../components/step_sizes.md).

## 2. Iteration

Each `step!`, in order:

1. **Preconditioned direction.** `d = −precondition(M, ∇f(x), problem, x) = −M⁻¹∇f(x)`.
2. **Step size.** `α = compute_step_size(step_size, …)` along `d`.
3. **Update.** `x ← x + α·d`.
4. **Refresh.** recompute `f(x)` and `∇f(x)` at the new iterate.

Exactly **one** gradient evaluation and **one** `precondition` call per step. The order
deliberately mirrors [`GradientDescent`](../gradient_descent/gradient_descent.md) — the
iterate and gradient histories (`x_prev`, `grad_prev`) are saved at the same points — so
the shared secant-based step sizes see a valid `(x_prev, grad_prev)` pair.

## 3. State

`PreconditionedGradientState` composes the canonical `IterateGroup` / `MetricsGroup` /
`TimingGroup` with `PreconditionedGradientNumerics`, whose fields are **the same shape as
`GradientDescentNumerics`** (`direction`, `α_k`, `n_linesearch_evals`, `grad_prev`,
`x_trial`). That is intentional: it lets the shared `StepSize` rules (Armijo, Cauchy, BB)
run against this method unchanged — only the direction computation differs from GD.

## 4. Implementation contract

The three dispatch points over the shared interface
(`docs/src/modules/algorithm-core.md`).

**`init_state`.** Copies `x₀`, evaluates `∇f(x₀)` and `total_objective(x₀)` once, sizes
the `x_trial` scratch buffer to `x₀`, and leaves `dist_to_opt = Inf` for the runner to
fill from `problem.x_opt`.

**`step!` — timing discipline.** Three `@core_timed` blocks bracket the math; the rest is
bookkeeping the clock never sees:

| Operation | Timed? |
|---|---|
| `precondition(M, …)` → `M⁻¹∇f`, `direction = −M⁻¹∇f` | yes (a pure kernel — see preconditioners.md) |
| `compute_step_size` | inside the rule (self-timed); `step!` does not wrap it |
| iterate update `x .+= α·d` | yes |
| objective + gradient refresh at `x_{k+1}` | yes |
| `x_prev` / `grad_prev` saves, `gradient_norm`, `step_norm` | no |

Ordering is load-bearing: `x_prev` is saved **after** `compute_step_size` and before the
update; `grad_prev` is saved **after** the update and before the gradient refresh — exactly
the GD ordering, so a `BarzilaiBorwein` secant pair would be correct (subject to §5).

**`extract_log_entry`.** Adds `:n_linesearch_evals` and `:step_size` to `extras`, plus
`:x_iter` (a copy of the iterate) when `dim ≤ 2`, for trajectory plots.

## 5. Relationship to GradientDescent, and the BB caveat

With `IdentityPreconditioner` this method *is* `GradientDescent` — identical iterate
sequence — which makes it a clean baseline-vs-experimental control. The only structural
difference is the preconditioned direction.

- **Barzilai–Borwein is excluded on a preconditioned direction.** The BB step `sᵀs / sᵀy`
  is derived for the steepest-descent direction `d = −g`; applied along `d = −M⁻¹g` it is
  ill-posed and diverges. The showcase grid therefore sweeps only `Fixed` / `Armijo` /
  `Cauchy`.
- **Cauchy composes** — it is the exact line search *along `d`*; on a true quadratic pass
  `CauchyStep(α_max = Inf)` so the trust-radius cap doesn't throttle the exact step.

## 6. Why Jacobi ≈ Newton

For a separable quadratic `f(x) = ½ Σᵢ dᵢ xᵢ²` the Hessian is `diag(d)`, so
`M⁻¹∇f = D⁻¹(D x) = x` and a unit step gives `x − x = 0` — the minimizer — in **one
iteration**, independent of the condition number `κ = max d / min d`. Plain gradient
descent on the same problem crawls at rate `1 − 1/κ`. Full derivation in
[preconditioners.md](../components/preconditioners.md) and
[separable_quadratic.md](../../problems/separable_quadratic/separable_quadratic.md).

## 7. Win conditions (preconditioning experiment)

The `exp_precond_grid.jl` portfolio experiment sweeps
`preconditioner ∈ {Identity, Jacobi} × step_size ∈ {Fixed, Armijo, Cauchy}` (a 2×3 grid,
`role = :experimental`) against a `GradientDescent` baseline on the `:separable_quadratic`
family (`n = 50`, `κ = 1e4`):

- **Dual-bucket routing.** `resolve_methods` routes the grid's 6 expanded specs to the
  *experimental* bucket and the lone `GradientDescent` to the *baseline* bucket — by the
  grid's `role`, not by method type (asserted before the run).
- **Cartesian expansion.** the two axes expand to exactly 6 `VariantSpec`s.
- **Jacobi ≈ Newton.** every Jacobi variant reaches `DistanceToOptimal(1e-8)` in ~1
  iteration regardless of `κ`; every Identity variant and the baseline crawl at `O(κ)`.
  The ~5-orders-of-magnitude iteration gap is the result.
