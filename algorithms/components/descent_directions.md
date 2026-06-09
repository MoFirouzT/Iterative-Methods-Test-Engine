# Descent Directions — Specification

> This file specifies the `DescentDirection` abstraction and its concrete
> implementations. It is shared across all gradient-based methods in the framework.
> Add new directions here and reference this file when implementing algorithms
> that use pluggable descent directions.

---

## 1. Abstraction

A **descent direction** at iterate $x_k$ is any vector $d_k \in \mathbb{R}^n$
satisfying:

$$\nabla f(x_k)^T d_k < 0$$

This inner product condition guarantees that moving from $x_k$ along $d_k$
decreases $f$ for sufficiently small step sizes.

### Julia Type Hierarchy

```julia
# In: algorithms/conventional/gradient_descent/components/descent_directions.jl

abstract type DescentDirection end
```

### Dispatch Interface

Every concrete direction implements exactly one method:

```julia
"""
    compute_direction(dir, state, problem) -> Vector{Float64}

Returns the descent direction d_k at the current iterate.
The returned vector is NOT normalized.
Normalization, if desired, is the responsibility of the step-size rule.

Preconditions (guaranteed by step! on entry):
  - state.iterate.x         holds the current iterate x_k
  - state.iterate.gradient  holds ∇f(x_k) (must be current — compute inside step!)
  - state.metrics.objective holds f(x_k) (must be current — compute inside step!)
"""
function compute_direction(dir::DescentDirection, state, problem)::Vector{Float64} end
```

> **Convention.** `compute_direction` reads from `state` but must **not** mutate it.
> The caller (`step!`) writes the result into `state.numerics.direction`.

---

## 2. Steepest Descent

### 2.1 Mathematics

The **steepest descent direction** (also called the negative gradient direction) is:

$$d_k^{SD} = -\nabla f(x_k)$$

It is the direction of maximum local decrease of $f$ with respect to the Euclidean
norm, i.e. it solves:

$$d_k^{SD} = \arg\min_{\|d\| = 1} \nabla f(x_k)^T d = -\frac{\nabla f(x_k)}{\|\nabla f(x_k)\|}$$

In practice the normalization is omitted and absorbed into the step-size:

$$d_k^{SD} = -\nabla f(x_k)$$

### 2.2 Properties

| Property | Value |
|---|---|
| Descent guarantee | Yes (whenever $\nabla f(x_k) \neq 0$) |
| Memory per iteration | None — stateless |
| Extra problem calls | None (gradient already computed in `step!`) |

### 2.3 Julia Struct

```julia
struct SteepestDescent <: DescentDirection end
```

No parameters. The struct is a singleton dispatch token.

### 2.4 Implementation

```julia
function compute_direction(::SteepestDescent, state, problem)::Vector{Float64}
    return -state.iterate.gradient    # d_k = -∇f(x_k); gradient is current on entry
end
```

> **Do not call `grad(problem.f, state.iterate.x)` here.** By the time
> `compute_direction` is called inside `step!`, the caller has already computed
> and stored the gradient in `state.iterate.gradient`. Reading it directly avoids
> a redundant evaluation.

### 2.5 Variable Mapping

| Math symbol        | Julia expression            | Type               |
|--------------------|-----------------------------|--------------------|
| $\nabla f(x_k)$   | `state.iterate.gradient`    | `Vector{Float64}`  |
| $d_k^{SD}$         | return value; stored in `state.numerics.direction` | `Vector{Float64}` |

---

## 3. Extension — Adding a New Direction

To add a new direction (e.g. Newton, conjugate gradient, L-BFGS):

1. Add a concrete struct subtyping `DescentDirection` in `descent_directions.jl`.
2. Implement `compute_direction(::YourDirection, state, problem)`.
3. The `GradientDescentNumerics` struct already holds
   `direction :: Vector{Float64}`; no structural changes are needed for the
   direction itself. If your direction maintains additional state (e.g. an
   approximate Hessian for quasi-Newton, or a previous-direction buffer for CG),
   add the corresponding field to `GradientDescentNumerics` and document the
   update ordering — see `step_sizes.md` §5.6 for the BB pattern as a reference.
4. Pass the new direction as the `direction` field in `GradientDescent(...)`.

No changes to `step!`, the runner, logging, or stopping criteria are required.

---

## 4. References

- Nocedal & Wright (2006). *Numerical Optimization* §2.2 — Steepest descent,
  convergence rate, zigzag behaviour on ill-conditioned quadratics.
- Boyd & Vandenberghe (2004). *Convex Optimization* §9.3 — Gradient descent
  and the steepest descent interpretation.
