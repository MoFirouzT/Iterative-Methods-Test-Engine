# Step-Size Rules — Specification

> This file specifies the `StepSizeRule` abstraction and all concrete implementations currently available.
> Step-size rules are shared across any gradient-based algorithm in the framework.
> Add new rules here and reference this file when implementing algorithms that use pluggable step sizes.

---

## 1. Abstraction

Given the current iterate $x_k$ and a descent direction $d_k$, a **step-size rule** produces a scalar $\alpha_k > 0$ such that the update

$$x_{k+1} = x_k + \alpha_k\, d_k$$

makes sufficient progress. 
Different rules trade exactness for cost.

### Julia Type Hierarchy

```julia
# In: algorithms/conventional/components/step_sizes.jl

abstract type StepSizeRule end
```

### Dispatch Interface

Every concrete rule implements exactly one method:

```julia
"""
    compute_step(rule, state, problem, direction) -> Float64

Returns the step size α_k > 0.

Arguments:
  rule      — the concrete StepSizeRule instance (carries hyperparameters)
  state     — current algorithm state; provides x_k, ∇f(x_k), f(x_k), x_{k-1}, ∇f(x_{k-1})
  problem   — the Problem struct; rules may call value() and grad() for line search evaluations
  direction — the descent direction d_k (already computed and stored in state.numerics.direction)

Preconditions (guaranteed by step! before calling compute_step):
  - state.iterate.x        = x_k        (current iterate)
  - state.iterate.gradient = ∇f(x_k)   (current gradient)
  - state.metrics.objective = f(x_k)   (current objective value)
  - state.iterate.x_prev   = x_{k-1}   (previous iterate; may be empty at k=1)
  - state.numerics.direction = d_k      (current descent direction)

Postconditions (guaranteed by compute_step):
  - returned α_k > 0
  - rules that call value() or grad() do so OUTSIDE @core_timed — these evaluations
    are counted separately in state.numerics.n_linesearch_evals
"""
function compute_step(rule::StepSizeRule, state, problem,
                      direction::Vector{Float64})::Float64 end
```

> **Timing convention.** The `@core_timed` block in `step!` covers direction
> computation and the final update $x_{k+1} = x_k + \alpha_k d_k$ only.
> Line search function evaluations inside `compute_step` are **not** wrapped in
> `@core_timed` — they are bookkeeping, tracked in `n_linesearch_evals` instead.

---

## 2. Fixed Step Size

### 2.1 Mathematics

$$\alpha_k = \alpha \quad \forall\, k$$

The step size is constant throughout the run. Convergence for smooth strongly convex
$f$ with Lipschitz gradient constant $L$ requires:

$$0 < \alpha < \frac{2}{L}$$

For the Rosenbrock function $L$ is the largest eigenvalue of $\nabla^2 f(x)$, which
varies across the domain. A safe conservative choice near the minimum is:

$$\alpha_{\max} \approx \frac{1}{\lambda_{\max}(\nabla^2 f(x^*))} = \frac{1}{1202} \approx 8.3 \times 10^{-4} \quad (\rho = 100)$$

### 2.2 Julia Struct

```julia
@kwdef struct FixedStep <: StepSizeRule
    α :: Float64 = 1e-3    # step size; must satisfy 0 < α < 2/L for convergence
end
```

### 2.3 Implementation

```julia
function compute_step(rule::FixedStep, state, problem,
                      direction::Vector{Float64})::Float64
    return rule.α
end
```

### 2.4 Variable Mapping

| Math symbol | Julia expression | Type      | Notes                    |
|-------------|-----------------|-----------|--------------------------|
| $\alpha$    | `rule.α`         | `Float64` | constant step size       |
| $\alpha_k$  | return value     | `Float64` | same every iteration     |

---

## 3. Cauchy Step (Exact Quadratic Line Search)

### 3.1 Mathematics

The **Cauchy step** minimizes the second-order Taylor expansion of $f$ along $d_k$:

$$\phi(\alpha) = f(x_k) + \alpha\, \nabla f(x_k)^T d_k + \frac{\alpha^2}{2}\, d_k^T \nabla^2 f(x_k)\, d_k$$

Setting $\phi'(\alpha) = 0$:

$$\alpha_k^C = -\frac{\nabla f(x_k)^T d_k}{d_k^T \nabla^2 f(x_k)\, d_k}$$

For the steepest descent direction $d_k = -\nabla f(x_k)$, this simplifies to:

$$\boxed{\alpha_k^C = \frac{g_k^T g_k}{g_k^T H_k g_k}}$$

where $g_k = \nabla f(x_k)$ and $H_k = \nabla^2 f(x_k)$.

> **Validity condition.** The denominator $g_k^T H_k g_k > 0$ requires $H_k$ to be
> positive definite along $d_k$. For the Rosenbrock function this may fail far from
> the minimum. The implementation must guard against non-positive denominators.

This is an **exact** line search only when $f$ is a quadratic. For the (non-quadratic)
Rosenbrock function it is an approximation — it minimizes the local quadratic model
rather than $f$ itself.

### 3.2 Julia Struct

```julia
@kwdef struct CauchyStep <: StepSizeRule
    fallback_α :: Float64 = 1e-3    # used when denominator ≤ ε_denom
    ε_denom    :: Float64 = 1e-14   # numerical zero threshold for denominator
end
```

### 3.3 Implementation

```julia
function compute_step(rule::CauchyStep, state, problem,
                      direction::Vector{Float64})::Float64
    g   = state.iterate.gradient              # g_k = ∇f(x_k)
    Hd  = hessian_vec(problem.f, state.iterate.x, direction)   # H_k d_k
    num = dot(g, direction)                   # g_k^T d_k  (negative for descent)
    den = dot(direction, Hd)                  # d_k^T H_k d_k

    if den <= rule.ε_denom
        # Non-positive curvature along direction — fall back to safe default
        return rule.fallback_α
    end

    return -num / den    # α_k^C = -(g_k^T d_k) / (d_k^T H_k d_k)
end
```

> For steepest descent $d_k = -g_k$, so `num = dot(g, -g) = -‖g‖²` and the
> formula reduces to `‖g‖² / (g^T H g)` as in the boxed formula above.

### 3.4 Variable Mapping

| Math symbol             | Julia expression                           | Type               | Notes                        |
|-------------------------|--------------------------------------------|--------------------|------------------------------|
| $g_k = \nabla f(x_k)$ | `state.iterate.gradient`                   | `Vector{Float64}`  | read only                    |
| $H_k d_k$               | `hessian_vec(problem.f, state.iterate.x, direction)` | `Vector{Float64}` | one Hessian-vec product |
| $g_k^T d_k$             | `dot(g, direction)`                        | `Float64`          | numerator (negative)         |
| $d_k^T H_k d_k$         | `dot(direction, Hd)`                       | `Float64`          | denominator; must be > 0     |
| $\alpha_k^C$            | `-num / den`                               | `Float64`          | Cauchy step size             |

### 3.5 Cost

One call to `hessian_vec` per iteration. For `RosenbrockFidelity` this is $O(n)$ and
does not require forming the full Hessian matrix.

---

## 4. Armijo Backtracking Line Search

### 4.1 Mathematics

The **Armijo condition** (also called the sufficient decrease condition) requires:

$$f(x_k + \alpha\, d_k) \leq f(x_k) + c_1\, \alpha\, \nabla f(x_k)^T d_k$$

where $c_1 \in (0, 1)$ is the sufficient decrease parameter (typically $c_1 = 10^{-4}$).

The **backtracking algorithm** starts at an initial trial step $\alpha_0$ and repeatedly
reduces it by a contraction factor $\beta \in (0, 1)$ until the Armijo condition holds:

$$\alpha \leftarrow \beta \, \alpha \quad \text{until } f(x_k + \alpha d_k) \leq f(x_k) + c_1 \alpha \nabla f(x_k)^T d_k$$

**Termination guarantee.** Because $d_k$ is a descent direction ($\nabla f(x_k)^T d_k < 0$),
the right-hand side is below $f(x_k)$ for all $\alpha > 0$, so the condition is
eventually satisfied by any $\alpha$ small enough. The algorithm always terminates in
finite iterations.

**Reduction in progress.** At termination, the accepted step satisfies:

$$\alpha_k \geq \beta \cdot \frac{\alpha_0 \cdot c_1 \cdot |\nabla f(x_k)^T d_k|}{L \|d_k\|^2}$$

which is bounded away from zero whenever the gradient is non-zero.

### 4.2 Julia Struct

```julia
@kwdef struct ArmijoLS <: StepSizeRule
    α₀  :: Float64 = 1.0      # initial trial step size
    β   :: Float64 = 0.5      # contraction factor ∈ (0, 1)
    c₁  :: Float64 = 1e-4     # sufficient decrease constant ∈ (0, 1)
    max_iter :: Int = 50       # safety cap on backtracking iterations
end
```

### 4.3 Implementation

```julia
function compute_step(rule::ArmijoLS, state, problem,
                      direction::Vector{Float64})::Float64
    x_k   = state.iterate.x
    f_k   = state.metrics.objective               # f(x_k) — already computed
    slope = dot(state.iterate.gradient, direction) # ∇f(x_k)^T d_k — must be < 0

    α = rule.α₀
    for _ in 1:rule.max_iter
        x_trial = x_k + α .* direction
        f_trial = objective(problem, x_trial)     # one function evaluation per step

        if f_trial ≤ f_k + rule.c₁ * α * slope
            return α                              # Armijo condition satisfied
        end
        α *= rule.β                              # backtrack
    end

    # max_iter reached: return the last α regardless (very small, but non-zero)
    return α
end
```

> **`objective(problem, x_trial)`** calls the framework's composite objective
> `f(x) + Σgᵢ(x)`. For the pure Rosenbrock problem (no regularizer) this reduces
> to `value(problem.f, x_trial)`.
> Track the number of these evaluations in `state.numerics.n_linesearch_evals`
> for fair comparisons.

### 4.4 Variable Mapping

| Math symbol                  | Julia expression                          | Type               | Notes                    |
|------------------------------|-------------------------------------------|--------------------|--------------------------|
| $\alpha_0$                   | `rule.α₀`                                | `Float64`          | initial trial step       |
| $\beta$                      | `rule.β`                                 | `Float64`          | contraction factor       |
| $c_1$                        | `rule.c₁`                                | `Float64`          | sufficient decrease const|
| $f(x_k)$                     | `state.metrics.objective`                 | `Float64`          | reused; not recomputed   |
| $\nabla f(x_k)^T d_k$       | `dot(state.iterate.gradient, direction)`  | `Float64`          | slope; must be < 0       |
| $f(x_k + \alpha d_k)$        | `objective(problem, x_trial)`             | `Float64`          | one eval per backtrack   |
| $\alpha_k$                   | return value                              | `Float64`          | accepted step size       |

### 4.5 Cost

One call to `objective` per backtracking trial. For well-behaved problems (small $\kappa$),
typical backtracking count is 1–3. For the Rosenbrock function deep inside the valley,
more steps may be needed when the initial $\alpha_0 = 1$ is too large.

---

## 5. Barzilai-Borwein Step Size

### 5.1 Mathematics

The **Barzilai-Borwein (BB) method** approximates the step size using curvature
information derived from two consecutive iterates, without requiring a Hessian
evaluation. Let:

$$s_{k-1} = x_k - x_{k-1}, \qquad y_{k-1} = \nabla f(x_k) - \nabla f(x_{k-1})$$

These satisfy the **secant equation** $H_k s_{k-1} \approx y_{k-1}$. Two BB variants
arise from different least-squares interpretations of this equation:

**BB1 — Long step** (minimizes $\|\alpha^{-1} s_{k-1} - y_{k-1}\|$):

$$\boxed{\alpha_k^{BB1} = \frac{s_{k-1}^T s_{k-1}}{s_{k-1}^T y_{k-1}}}$$

**BB2 — Short step** (minimizes $\|s_{k-1} - \alpha y_{k-1}\|$):

$$\boxed{\alpha_k^{BB2} = \frac{s_{k-1}^T y_{k-1}}{y_{k-1}^T y_{k-1}}}$$

The two variants satisfy $\alpha_k^{BB2} \leq \alpha_k^{BB1}$ (by Cauchy-Schwarz).

**First iteration.** At $k = 1$ the previous iterate $x_0$ and gradient $g_0$ are
available from `init_state`. However, $s_0 = x_1 - x_0$ and $y_0 = g_1 - g_0$ are
computed at the end of the first step. Therefore BB is only applied from $k = 2$
onward. At $k = 1$ a fallback (e.g. Cauchy or fixed step) is used.

**Curvature condition.** If $s_{k-1}^T y_{k-1} \leq 0$, the secant equation has no
positive solution and the BB step is undefined. The implementation must fall back to
`fallback_α` in this case.

**Non-monotone behaviour.** Unlike Armijo, BB does not guarantee $f(x_{k+1}) < f(x_k)$.
This is expected and desired — BB deliberately accepts temporary increases to avoid
the slow convergence of steepest descent on ill-conditioned problems.

### 5.2 Convergence

For strongly convex quadratics, BB1 converges **superlinearly** in the two-dimensional
case and **R-linearly** in higher dimensions. For non-quadratic functions like
Rosenbrock, convergence is not guaranteed without additional safeguards (e.g.
non-monotone Armijo). In practice BB performs very well on smooth problems.

### 5.3 Julia Struct

```julia
@kwdef struct BarzilaiBorwein <: StepSizeRule
    variant    :: Symbol  = :BB1       # :BB1 (long) or :BB2 (short)
    fallback_α :: Float64 = 1e-3       # step used at k=1 or when s^T y ≤ 0
    ε_denom    :: Float64 = 1e-14      # numerical zero threshold
end
```

### 5.4 Required Additional State

BB needs the **previous gradient** $\nabla f(x_{k-1})$ in addition to the previous
iterate $x_{k-1}$ (already in `state.iterate.x_prev`). Add a field to
`GradientDescentNumerics`:

```julia
@kwdef mutable struct GradientDescentNumerics
    direction          :: Vector{Float64} = Float64[]
    n_linesearch_evals :: Int             = 0
    grad_prev          :: Vector{Float64} = Float64[]   # ∇f(x_{k-1}); needed by BB
end
```

`grad_prev` is initialized to `Float64[]` (empty). The `compute_step` method checks
`isempty(state.numerics.grad_prev)` to detect the first iteration.

### 5.5 Implementation

```julia
function compute_step(rule::BarzilaiBorwein, state, problem,
                      direction::Vector{Float64})::Float64
    g_curr = state.iterate.gradient
    g_prev = state.numerics.grad_prev
    x_curr = state.iterate.x
    x_prev = state.iterate.x_prev

    # First iteration: previous data not yet available
    if isempty(g_prev) || isempty(x_prev)
        return rule.fallback_α
    end

    s = x_curr .- x_prev    # s_{k-1} = x_k - x_{k-1}
    y = g_curr .- g_prev    # y_{k-1} = ∇f(x_k) - ∇f(x_{k-1})

    sy = dot(s, y)           # s^T y — must be positive for valid BB step

    if sy <= rule.ε_denom
        return rule.fallback_α    # curvature condition violated
    end

    if rule.variant == :BB1
        return dot(s, s) / sy    # α^BB1 = ‖s‖² / (s^T y)
    else  # :BB2
        return sy / dot(y, y)    # α^BB2 = (s^T y) / ‖y‖²
    end
end
```

### 5.6 State Update Responsibility

The caller (`step!`) is responsible for updating `grad_prev` **after** using it:

```julia
# Inside step! (GradientDescent), after compute_step and before returning:
state.numerics.grad_prev = copy(state.iterate.gradient)
# (x_prev is already managed by the runner convention via state.iterate.x_prev)
```

This ordering is critical: `compute_step` reads the *current* gradient as $g_k$
and `grad_prev` as $g_{k-1}$. Updating `grad_prev` before calling `compute_step`
would corrupt the secant pair.

### 5.7 Variable Mapping

| Math symbol                  | Julia expression                          | Type               | Notes                             |
|------------------------------|-------------------------------------------|--------------------|-----------------------------------|
| $x_k$                        | `state.iterate.x`                         | `Vector{Float64}`  | current iterate                   |
| $x_{k-1}$                   | `state.iterate.x_prev`                    | `Vector{Float64}`  | previous iterate                  |
| $\nabla f(x_k)$             | `state.iterate.gradient`                  | `Vector{Float64}`  | current gradient                  |
| $\nabla f(x_{k-1})$         | `state.numerics.grad_prev`                | `Vector{Float64}`  | previous gradient; updated in `step!` after compute_step |
| $s_{k-1}$                    | `x_curr .- x_prev`                       | `Vector{Float64}`  | step difference                   |
| $y_{k-1}$                    | `g_curr .- g_prev`                       | `Vector{Float64}`  | gradient difference               |
| $s_{k-1}^T y_{k-1}$         | `dot(s, y)`                              | `Float64`          | curvature; must be > 0            |
| $\alpha_k^{BB1}$             | `dot(s,s) / dot(s,y)`                   | `Float64`          | long step                         |
| $\alpha_k^{BB2}$             | `dot(s,y) / dot(y,y)`                   | `Float64`          | short step                        |

---

## 6. Comparison at a Glance

| Rule | Cost per iter | Monotone | Requires Hessian | Parameters |
|------|---|---|---|---|
| `FixedStep` | $O(1)$ | Only if $\alpha < 2/L$ | No | $\alpha$ |
| `CauchyStep` | $O(n)$ Hessian-vec | Yes (approx) | Yes (Hessian-vec) | `fallback_α`, `ε_denom` |
| `ArmijoLS` | $O(n)$ per backtrack | Yes | No | $\alpha_0$, $\beta$, $c_1$ |
| `BarzilaiBorwein` | $O(n)$ dot products | **No** | No | `variant`, `fallback_α` |

---

## 7. Extension — Adding a New Step-Size Rule

1. Add a concrete struct subtyping `StepSizeRule` in `step_sizes.jl`.
2. Implement `compute_step(::YourRule, state, problem, direction)`.
3. If the rule requires additional per-iteration state (as BB requires `grad_prev`),
   add the field to `GradientDescentNumerics` and document the update responsibility
   (who writes it, when, and why the ordering matters).
4. Pass the new rule as the `step_rule` field in `GradientDescent(...)`.

No changes to `step!`, the runner, logging, or stopping criteria are required.

---

## 8. References

- Armijo, L. (1966). *Minimization of functions having Lipschitz continuous first
  partial derivatives.* Pacific Journal of Mathematics, 16(1), 1–3.
- Barzilai, J. & Borwein, J.M. (1988). *Two-point step size gradient methods.*
  IMA Journal of Numerical Analysis, 8(1), 141–148.
- Nocedal & Wright (2006). *Numerical Optimization* §3.1 (Armijo), §3.3 (BB).
- Dai, Y.H. & Liao, L.Z. (2002). *R-linear convergence of the Barzilai and Borwein
  gradient method.* IMA Journal of Numerical Analysis, 22(1), 1–10.
