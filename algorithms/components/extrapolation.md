# Extrapolation components

An `Extrapolation` is a **post-step correction**: it chooses the point at which a
method evaluates the gradient, then advances any momentum parameter after the iterate
update. It is one axis of `ProximalGradient`, crossed with a step-size axis — the axis
that turns ISTA into FISTA. The engine grid machinery is extrapolation-agnostic; this
vocabulary is content.

## Contract

```julia
abstract type Extrapolation end
extrapolate(mu::Extrapolation, x, x_prev, t) -> y      # gradient-evaluation point this step
advance_momentum(mu::Extrapolation, t)       -> t_next # momentum update after the iterate step
```

A composing method calls `extrapolate` *before* the gradient step to pick `y`, then
`advance_momentum` *after* the iterate update. On the first step `x_prev` is empty
(the sentinel), so every variant returns `y = x` — there is nothing to extrapolate
from yet.

Timing: `extrapolate` is part of the kernel that produces the next iterate, so the
consuming method times it inside `@core_timed` (the same way `ProximalGradient` times
`grad!`). `advance_momentum` is an O(1) scalar update and is left untimed as
bookkeeping.

## Concretes

| type | `extrapolate` → `y` | `advance_momentum` | reduces to |
|---|---|---|---|
| `NoExtrapolation` | `x` | `t` (unchanged) | ISTA / plain (proximal) gradient |
| `NesterovStep` | `x + β·(x − x_prev)` | `t_next = (1 + √(1 + 4t²))/2` | FISTA (`O(1/k²)`) |
| `MomentumStep` | `x + α·(x − x_prev)`, fixed `α` | `t` (unchanged) | heavy-ball momentum |

For `NesterovStep` the coefficient is `β = (t − 1)/t_next` with the **same** `t_next`
recurrence that `advance_momentum` applies, so passing the *current* `t` to both keeps
`β` and the momentum advance consistent. The momentum parameter starts at `t = 1.0`
(in `ProximalGradientNumerics`), which makes `β = 0` on the first extrapolated step.

`MomentumStep` carries a fixed coefficient `α` (default `0.1`); `NesterovStep`'s
coefficient comes entirely from the `t` recurrence, not from a tunable field.

## The FISTA story — why this is its own axis

Swapping `NoExtrapolation` for `NesterovStep` is the *only* change that turns plain
proximal gradient (ISTA, `O(1/k)`) into accelerated proximal gradient (FISTA,
`O(1/k²)`) — no change to the step-size rule, the prox, or the problem. Making
extrapolation a first-class component (rather than a flag inside the method) is what
lets a single `ProximalGradient` method tell the whole acceleration story by sweeping
one axis. And because the same extrapolation applies when the regularizer is zero or
absent, the method also demonstrates smooth-function acceleration (accelerated
gradient descent) for free. See
[proximal_gradient.md](../proximal_gradient/proximal_gradient.md) and the
flagship lasso experiment.

## Extension — adding a new extrapolation

1. Add a concrete struct subtyping `Extrapolation` in `extrapolation.jl`.
2. Implement `extrapolate(::YourStep, x, x_prev, t)` (return `copy(x)` when `x_prev`
   is empty) and `advance_momentum(::YourStep, t)`.
3. If the rule needs per-iteration state beyond `t`, add the field to the consuming
   method's `Numerics` and document the update ordering (see `step_sizes.md` §5.6 for
   the pattern).
4. Register a short name for legends/filenames via `register_abbreviation!`.
5. Pass the new rule as the `extrapolation` field in `ProximalGradient(...)`. No
   changes to `step!`, the runner, logging, or stopping criteria are required.

## References

- Beck, A. & Teboulle, M. (2009). *A Fast Iterative Shrinkage-Thresholding Algorithm
  for Linear Inverse Problems.* SIAM J. Imaging Sciences, 2(1), 183–202. (FISTA — the
  `t` recurrence and `O(1/k²)` rate.)
- Nesterov, Y. (1983). *A method for solving the convex programming problem with
  convergence rate O(1/k²).* Soviet Math. Dokl., 27(2), 372–376.
