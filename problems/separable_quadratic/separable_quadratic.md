# `:separable_quadratic` — diagonal quadratic

```text
f(x) = ½ Σᵢ dᵢ xᵢ²,   dᵢ > 0
```

The smallest interesting problem with a non-trivial, *diagonal* Hessian. Its
purpose is to exercise `DiagonalHessian` and give the Jacobi preconditioner a
showcase where it is exactly Newton.

## Objective

```julia
struct SeparableQuadratic <: Objective; d::Vector{Float64}; end
value(f, x)    = ½ Σ dᵢ xᵢ²
grad!(g, f, x) = (g .= d .* x)
hessian(f, x)  = DiagonalHessian(d)        # constant
```

Unique minimizer `x_opt = 0`, `f* = 0` — stop on gradient/distance, not f-value.

## `:separable_quadratic` generator

| param | default | meaning |
|---|---|---|
| `n` | 50 | dimension |
| `condition_number` | 1e4 | `κ = max d / min d` |

Curvatures span **`[1/κ, 1]`** (descending from 1), *not* `[1, κ]`. The choice is
deliberate: with `λ_max = 1`,

- `FixedStep(α = 1)` is **stable** for the unpreconditioned method (`α ≤ 2/L`)
  yet converges slowly along the `1/κ` direction (`~κ` iters), and
- it is **exactly the Newton step** for Jacobi (`M⁻¹∇f = x` ⇒ `x − x = 0` in one
  step).

So the step-size axis (`Fixed` / `Armijo` / `BB1`) stays fair across both
preconditioners — no single global `α` is secretly tuned for one of them.

`x0 = ones(n)`; `meta[:condition_number] = κ`, `meta[:L] = max(d) = 1`.

## Win condition (preconditioning experiment)

Jacobi-preconditioned variants converge in `O(1–2)` iterations regardless of `κ`;
Identity variants take `O(κ)` (Fixed/Armijo) or `O(√κ)` (BB). The two-decade gap
in iteration count is the validation. Choose `κ ≥ 1e3` so the unpreconditioned
method is *visibly* slow.
