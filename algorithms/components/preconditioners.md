# Preconditioner components

A `Preconditioner` supplies `M⁻¹` for the preconditioned gradient direction

    d = −M⁻¹ ∇f(x)

It is one axis of `PreconditionedGradient` (the experimental method), crossed
with a step-size axis. The engine grid machinery is preconditioner-agnostic.

## Contract

```julia
abstract type Preconditioner end
precondition(M::Preconditioner, g, problem, x) -> M⁻¹·g
```

## Concretes

| type | `M⁻¹` | reduces to |
|---|---|---|
| `IdentityPreconditioner` | `I` | plain gradient descent |
| `JacobiPreconditioner` | `diag(∇²f)⁻¹` | Newton, when `∇²f` is diagonal |

## The `diagonal` contract — a feature, not a limitation

`JacobiPreconditioner` reads `diagonal(hessian(f, x))`. `diagonal` is an
**optional** part of the `Hessian` contract — declared only by the
representations that can supply it:

| Hessian | `diagonal`? | Jacobi |
|---|---|---|
| `DiagonalHessian` | ✓ | exact Newton (separable quadratic) |
| `MatrixHessian` | ✓ | reads `diag(AᵀA)` |
| `OperatorHessian` | ✗ | **correctly inapplicable** |

On an `OperatorHessian` (matvec-only, e.g. the `:linear_ls` family) Jacobi is
*correctly inapplicable* and raises a clean `ArgumentError` — never a silent
fallback that would quietly degrade to un-preconditioned steps. This is the
"each `Hessian` declares which operations it supports" design made operational.

Detection uses a small explicit trait `_supports_diagonal(::Hessian)` rather
than `applicable(diagonal, H)`: the engine's throwing fallback
`diagonal(::Hessian)` *is* applicable to every subtype, so `applicable` can't
tell a real method from the fallback.

## Why Jacobi ≈ Newton on a diagonal Hessian

For `f(x) = ½ xᵀ D x` with `D = diag(d)`: `∇f = D x`, `diag(∇²f) = d`, so
`M⁻¹∇f = D⁻¹ D x = x` and a unit step gives `x − x = x* = 0` in **one
iteration**, independent of the condition number `κ = max d / min d`. Plain
gradient descent on the same problem crawls at rate `1 − 1/κ`. That gap — orders
of magnitude in iteration count — is what the preconditioning experiment shows.
