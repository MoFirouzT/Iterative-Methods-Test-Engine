# Step-Size Rules

> This file specifies the `StepSize` abstraction and all concrete implementations
> currently available, including the `LineSearch` sub-hierarchy. Step-size rules
> are shared across any gradient-based algorithm in the framework. Add new rules
> here and reference this file when implementing algorithms that use pluggable
> step sizes.

---

## 1. Abstraction

Given the current iterate $x_k$ and a descent direction $d_k$, a **step-size rule**
produces a scalar $\alpha_k > 0$ such that the update

$$x_{k+1} = x_k + \alpha_k\, d_k$$

makes sufficient progress. Different rules trade exactness for cost.

### Type Hierarchy — `StepSize` and `LineSearch`

The framework distinguishes two kinds of step-size rules:

- **Direct rules** compute $\alpha_k$ in closed form from current state — no
  trial points are tested. Examples: `FixedStep`, `CauchyStep`, `BarzilaiBorwein`.
- **Line searches** test trial points along the ray $x_k + \alpha\, d_k$ until a
  sufficient-decrease (or curvature) condition holds. Examples: `ArmijoLS`.

Both are valid step-size strategies, so they share an umbrella abstract type
`StepSize`. `LineSearch <: StepSize` is the sub-hierarchy of genuine 1D searches:

```julia
# In: algorithms/components/step_sizes.jl

abstract type StepSize end
abstract type LineSearch <: StepSize end    # subset that performs an actual 1D search
```

> **Why the sub-hierarchy?** It is a real dispatch handle. Code that needs to
> reason about line searches specifically — for example, future stopping logic
> that watches the line-search evaluation budget, or trust-region wrappers that
> insert sufficient-decrease guards — can dispatch on `::LineSearch` directly.
> Code that does not care simply dispatches on `::StepSize` and accepts any rule.

### Dispatch Interface

Every concrete rule (direct or line search) implements exactly one method:

```julia
"""
    compute_step_size(rule, state, problem, direction) -> Float64

Returns the step size α_k > 0.

Arguments:
  rule      — the concrete StepSize instance (carries hyperparameters)
  state     — current algorithm state; provides x_k, ∇f(x_k), f(x_k),
              x_{k-1}, ∇f(x_{k-1}) and the timing accumulator
  problem   — the Problem struct; search-based rules may call value() and grad!()
  direction — the descent direction d_k (already stored in state.numerics.direction)

Preconditions (guaranteed by step! before calling compute_step_size):
  - state.iterate.x          = x_k         (current iterate)
  - state.iterate.gradient   = ∇f(x_k)     (current gradient)
  - state.metrics.objective  = f(x_k)      (current objective value)
  - state.iterate.x_prev     = x_{k-1}     (previous iterate; empty at k=1)
  - state.numerics.direction = d_k         (current descent direction)

Postcondition:
  - returned value α_k > 0

Timing responsibility:
  Each implementation wraps its own core mathematical computations in
  `@core_timed state`. Function evaluations are part of that core computation
  and are therefore timed as well. The caller (step!) needs no knowledge of
  how timing is handled inside any rule — it calls compute_step_size uniformly
  and the accumulation into state.timing.core_time_ns happens automatically.
"""
function compute_step_size(rule::StepSize, state, problem,
                           direction::Vector{Float64})::Float64 end
```

---

## 2. Fixed Step Size  *(direct — `<: StepSize`)*

### 2.1 Mathematics

$$\alpha_k = \alpha \quad \forall\, k$$

The step size is constant throughout the run.

### 2.2 Julia Struct

```julia
@kwdef struct FixedStep <: StepSize
    α :: Float64 = 1e-3
end
```

### 2.3 Implementation

```julia
function compute_step_size(rule::FixedStep, state, problem,
                           direction::Vector{Float64})::Float64
    return rule.α
end
```

> No variable-mapping table: $\alpha_k = \alpha = $ `rule.α` is direct enough that
> the code is its own mapping.

---

## 3. Cauchy Step (Exact Quadratic Line Search)  *(direct — `<: StepSize`)*

> Despite the historical name "exact line search," the Cauchy step is a
> closed-form rule, not a search — no trial points are tested. It therefore
> subtypes `StepSize` directly, **not** `LineSearch`.

### 3.1 Mathematics

The **Cauchy step** minimizes the second-order Taylor expansion of $f$ along $d_k$:

$$\phi(\alpha) = f(x_k) + \alpha\, \nabla f(x_k)^T d_k
               + \frac{\alpha^2}{2}\, d_k^T \nabla^2 f(x_k)\, d_k$$

Setting $\phi'(\alpha) = 0$:

$$\alpha_k^C = -\frac{\nabla f(x_k)^T d_k}{d_k^T \nabla^2 f(x_k)\, d_k}$$

> **Validity condition.** The denominator $d_k^T \nabla^2 f(x_k)\, d_k > 0$
> requires positive curvature along $d_k$. The implementation guards against
> non-positive curvature with a **scale-relative** test (see §3.3).

This is an **exact** line search only when $f$ is quadratic. On a genuine
quadratic set `α_max = Inf` — the trust-radius cap below exists only to tame the
non-quadratic case (Rosenbrock), and would otherwise throttle the legitimate
exact step (which routinely exceeds 1).

### 3.2 Julia Struct

```julia
@kwdef struct CauchyStep <: StepSize
    fallback_α :: Float64 = 1e-3    # used when curvature is non-positive
    ε_denom    :: Float64 = 1e-14   # RELATIVE threshold: den ≤ ε_denom · ‖d‖²
    α_min      :: Float64 = 0.0     # reject negative steps from sign pathologies
    α_max      :: Float64 = 1.0     # trust-radius cap; set Inf on a true quadratic
end
```

### 3.3 Implementation

The Hessian is obtained as a `Hessian` object via `hessian(problem.f, x)` and
$\nabla^2 f(x)\, d$ is computed by `apply(H, d)`. Both calls and the dot products
are core mathematical computation — wrapped in `@core_timed`. The guard, clamp,
and division are bookkeeping.

```julia
function compute_step_size(rule::CauchyStep, state, problem,
                           direction::Vector{Float64})::Float64
    @core_timed state begin
        H   = hessian(problem.f, state.iterate.x)         # ∇²f(x_k) — a Hessian object
        Hd  = apply(H, direction)                         # H_k · d_k
        num = dot(state.iterate.gradient, direction)      # g_k^T d_k
        den = dot(direction, Hd)                          # d_k^T H_k d_k
        den <= rule.ε_denom * dot(direction, direction) && return rule.fallback_α
        return clamp(-num / den, rule.α_min, rule.α_max)
    end
end
```

> **Scale-relative curvature guard.** `den = dᵀ∇²f d` is the unnormalized
> Rayleigh quotient; near convergence `‖d‖ = ‖∇f‖ → 0`, so `den → 0`
> quadratically. An *absolute* guard (`den ≤ ε_denom`) then misfires even though
> `−num/den` is perfectly well-defined, collapsing Cauchy to the tiny
> `fallback_α` — on a consistent least-squares system (κ = 100) this turned a
> ~650-iteration solve into ~240 000. The relative form `den ≤ ε_denom · ‖d‖²`
> (Rayleigh quotient ≤ ε_denom) fires only for genuinely flat/negative
> curvature, independent of gradient scale. This mirrors the `BarzilaiBorwein`
> guard, which was fixed the same way earlier; `CauchyStep` carried the latent
> bug until the `:linear_ls` family exercised the small-gradient regime.

> **Hessian abstraction.** `hessian(problem.f, x)` returns a `Hessian` object
> (see `docs/src/modules/problem-interface.md`). For Rosenbrock this is a `MatrixHessian`
> wrapping a 2×2 matrix; for higher-dimensional problems it may be an
> `OperatorHessian` or a quasi-Newton approximation. `apply(H, d)` is the
> required operation across all subtypes — `CauchyStep` does not need to know
> the concrete representation.

### 3.4 Variable Mapping

| Math symbol              | Julia expression                                   | Type               | Notes                    |
|--------------------------|----------------------------------------------------|--------------------|--------------------------|
| $g_k$                    | `state.iterate.gradient`                           | `Vector{Float64}`  | read only                |
| $H_k$                    | `hessian(problem.f, state.iterate.x)`              | `Hessian`          | concrete subtype varies  |
| $H_k d_k$                | `apply(H, direction)`                              | `Vector{Float64}`  | one Hessian-vec product  |
| $g_k^T d_k$              | `dot(state.iterate.gradient, direction)`           | `Float64`          | numerator (negative)     |
| $d_k^T H_k d_k$          | `dot(direction, Hd)`                               | `Float64`          | denominator; must be > 0 |
| $\alpha_k^C$             | `-num / den`                                       | `Float64`          | Cauchy step size         |

### 3.5 Cost

One call to `hessian(problem.f, ·)` and one `apply(H, ·)` per iteration. For
problems where `hessian` is recomputed at each $x$ (typical for explicit-Hessian
problems like Rosenbrock), the cost is dominated by the Hessian construction
itself; for problems backed by an `OperatorHessian` closure or a quasi-Newton
approximation, only the `apply` call counts.

---

## 4. Armijo Backtracking Line Search  *(`<: LineSearch`)*

### 4.1 Mathematics

The **Armijo condition** (sufficient decrease) requires:

$$f(x_k + \alpha\, d_k) \leq f(x_k) + c_1\, \alpha\, \nabla f(x_k)^T d_k$$

where $c_1 \in (0, 1)$ is the sufficient decrease parameter (typically
$c_1 = 10^{-4}$).

The **backtracking algorithm** starts at an initial trial step $\alpha_0$ and
repeatedly reduces it by a contraction factor $\beta \in (0, 1)$ until the above
condition holds.

**Termination guarantee.**
Because $\nabla f(x_k)^T d_k < 0$ for any descent direction, the right-hand side
is below $f(x_k)$ for all $\alpha > 0$, so the condition is eventually satisfied.
The algorithm always terminates in finite steps.

### 4.2 Julia Struct

```julia
@kwdef struct ArmijoLS <: LineSearch
    α₀       :: Float64 = 1.0     # initial trial step size
    β        :: Float64 = 0.5     # contraction factor ∈ (0, 1)
    c₁       :: Float64 = 1e-4    # sufficient decrease constant ∈ (0, 1)
    max_iter :: Int     = 50      # safety cap on backtracking iterations
end
```

### 4.3 Implementation

Armijo performs function evaluations, and those evaluations are part of the core
mathematical computation in the sense of the framework. `@core_timed` is used
around the trial evaluations; `state.numerics.n_linesearch_evals` tracks how many
trial evaluations were needed. The accepted/rejected trial point is stored in
`state.numerics.x_trial`, which lets backtracking reuse a scratch buffer instead
of allocating a fresh vector on every reduction.

```julia
function compute_step_size(rule::ArmijoLS, state, problem,
                           direction::Vector{Float64})::Float64
    f_xk     = state.metrics.objective                    # f(x_k) — already current
    slope    = dot(state.iterate.gradient, direction)     # ∇f(x_k)^T d_k (< 0)
    α        = rule.α₀
    rhs_coef = rule.c₁ * slope                            # used in Armijo RHS

    @core_timed state begin
        for _ in 1:rule.max_iter
            state.numerics.x_trial .= state.iterate.x .+ α .* direction
            f_trial = total_objective(problem, state.numerics.x_trial)
            state.numerics.n_linesearch_evals += 1
            if f_trial <= f_xk + α * rhs_coef
                return α
            end
            α *= rule.β
        end
    end

    return α    # max_iter reached; return last α (very small but non-zero)
end
```

> `total_objective(problem, x_trial)` evaluates the composite objective
> $f(x) + \sum_i g_i(x)$. For the pure Rosenbrock problem (no regularizer)
> this reduces to `value(problem.f, x_trial)` because `problem.gs` is empty.
> See `docs/src/modules/problem-interface.md` for the helper definition.

### 4.4 Variable Mapping

| Math symbol              | Julia expression                                 | Type               | Notes                      |
|--------------------------|--------------------------------------------------|--------------------|----------------------------|
| $\alpha_0$               | `rule.α₀`                                        | `Float64`          | initial trial step         |
| $\beta$                  | `rule.β`                                         | `Float64`          | contraction factor         |
| $c_1$                    | `rule.c₁`                                        | `Float64`          | sufficient decrease const  |
| $f(x_k)$                 | `state.metrics.objective`                        | `Float64`          | reused; not recomputed     |
| $\nabla f(x_k)^T d_k$   | `dot(state.iterate.gradient, direction)`         | `Float64`          | slope; must be < 0         |
| $f(x_k + \alpha d_k)$   | `total_objective(problem, state.numerics.x_trial)` | `Float64`          | one eval per backtrack; `x_trial` reused |
| $\alpha_k$               | return value                                     | `Float64`          | accepted step size         |

### 4.5 Cost

One call to `total_objective` per backtracking trial. For well-behaved problems,
typical backtracking count is 1–3.

---

## 5. Barzilai-Borwein Step Size  *(direct — `<: StepSize`)*

### 5.1 Mathematics

The **Barzilai-Borwein (BB) method** approximates the step size using curvature
information derived from two consecutive iterates, without requiring a Hessian
evaluation. Let:

$$s_{k-1} = x_k - x_{k-1}, \qquad y_{k-1} = \nabla f(x_k) - \nabla f(x_{k-1})$$

These satisfy the **secant equation** $H_k s_{k-1} \approx y_{k-1}$.
Two BB variants arise from different least-squares interpretations:

**BB1 — Long step** (minimizes $\|\alpha^{-1} s_{k-1} - y_{k-1}\|$):

$$\boxed{\alpha_k^{BB1} = \frac{s_{k-1}^T s_{k-1}}{s_{k-1}^T y_{k-1}}}$$

**BB2 — Short step** (minimizes $\|s_{k-1} - \alpha\, y_{k-1}\|$):

$$\boxed{\alpha_k^{BB2} = \frac{s_{k-1}^T y_{k-1}}{y_{k-1}^T y_{k-1}}}$$

The two variants satisfy $\alpha_k^{BB2} \leq \alpha_k^{BB1}$ (by Cauchy-Schwarz).

**First iteration.** At $k = 1$, `x_prev` and `grad_prev` are not yet available
(`Float64[]`). The implementation falls back to `fallback_α`.

**Curvature condition.** If $s_{k-1}^T y_{k-1} \leq 0$, the BB step is undefined.
The implementation falls back to `fallback_α`.

**Non-monotone behaviour.** Unlike Armijo, BB does not guarantee
$f(x_{k+1}) < f(x_k)$. This is expected and desired — BB deliberately accepts
temporary increases to escape the slow convergence of steepest descent on
ill-conditioned problems.

### 5.2 Convergence

For strongly convex quadratics, BB1 converges superlinearly in the
two-dimensional case and R-linearly in higher dimensions. For non-quadratic
functions convergence is not guaranteed without additional safeguards, but in
practice BB performs very well on smooth problems.

The `[α_min, α_max]` clamp is **not** that safeguard. An empirical `α_max`
sweep on Rosenbrock from x₀ = (−1.2, 1) shows the clamp has no useful
problem-independent default:

| `α_max`              | n_iters     | result                                   |
|----------------------|-------------|------------------------------------------|
| 10                   | 2000 (DNF)  | clamp active — **breaks convergence**    |
| 30, 50, 100, 1e6, ∞  | 76          | clamp transparent — BB1 converges        |

BB1 takes a step at iter 6 with α ≈ 28 that sends `f` to ~10⁹, then contracts
and recovers over ~40 iters. That excursion is **load-bearing** — clipping it
interrupts the recovery and BB1 gets stuck. Tight enough to catch the spike
breaks BB; loose enough to let BB work catches nothing. Hence the default
`α_max = 1e6` is a pure overflow net, not a convergence safeguard.

**The principled safeguard is a nonmonotone line search** (Grippo–Lampariello–
Lucidi 1986; Raydan 1997 for BB). It permits non-monotone `f` but requires, for
a window of size `M` (typically 5–10) and small `γ` (typically 1e-4):

$$f_k \le \max\bigl(f_{k-M}, \dots, f_{k-1}\bigr) - \gamma\,\alpha\,\lVert \nabla f_k \rVert^2$$

and backtracks only when violated — so BB's recovering excursions survive while
genuinely diverging ones are caught. *Sketch:* give `GradientDescentNumerics` a
`nonmonotone_window` circular buffer of recent `f` values, let
`compute_step_size` for BB peek at it and the candidate `α`, and backtrack
inside the rule when the condition fails — or wrap BB in a higher-order
`NonmonotoneSafeguard(BarzilaiBorwein(...))` rule. Either is ~30 lines; not yet
implemented.

### 5.3 Julia Struct

```julia
@kwdef struct BarzilaiBorwein <: StepSize
    variant    :: Symbol  = :BB1      # :BB1 (long step) or :BB2 (short step)
    fallback_α :: Float64 = 1e-3      # used at k=1 or when s^T y ≤ 0
    ε_denom    :: Float64 = 1e-14     # numerical zero threshold
end
```

### 5.4 Required Additional State

BB requires the **previous gradient** $\nabla f(x_{k-1})$ in addition to the
previous iterate $x_{k-1}$ (already in `state.iterate.x_prev`). One field is
added to `GradientDescentNumerics`:

```julia
@kwdef mutable struct GradientDescentNumerics
    direction          :: Vector{Float64} = Float64[]
    n_linesearch_evals :: Int             = 0
    grad_prev          :: Vector{Float64} = Float64[]   # ∇f(x_{k-1}); required by BB
end
```

`grad_prev` is initialized to `Float64[]`. The implementation detects the first
iteration via `isempty(state.numerics.grad_prev)`.

### 5.5 Implementation

The secant vector constructions and dot products are core mathematical
computation — wrapped in `@core_timed`. The validity guard and fallback are
bookkeeping.

```julia
function compute_step_size(rule::BarzilaiBorwein, state, problem,
                           direction::Vector{Float64})::Float64
    # First iteration: previous data not yet available
    if isempty(state.numerics.grad_prev) || isempty(state.iterate.x_prev)
        return rule.fallback_α
    end

    local s, y, sy, α
    @core_timed state begin
        s  = state.iterate.x        .- state.iterate.x_prev      # s_{k-1} = x_k - x_{k-1}
        y  = state.iterate.gradient .- state.numerics.grad_prev  # y_{k-1} = g_k - g_{k-1}
        sy = dot(s, y)
        sy <= rule.ε_denom && return rule.fallback_α   # curvature condition violated
        α = if rule.variant == :BB1
            dot(s, s) / sy       # α^BB1 = ‖s‖² / (s^T y)
        else
            sy / dot(y, y)       # α^BB2 = (s^T y) / ‖y‖²
        end
    end

    return α
end
```

### 5.6 State Update Responsibility

`step!` must update `grad_prev` **after** calling `compute_step_size` and
**before** returning. This ordering is critical:
`compute_step_size` reads `grad_prev` as $\nabla f(x_{k-1})$ and
`state.iterate.gradient` as $\nabla f(x_k)$. Updating before the call would
corrupt the secant pair.

```julia
# Inside step!, after compute_step_size and the iterate update:
state.numerics.grad_prev = copy(state.iterate.gradient)
# x_prev is written by step! at the start (before the gradient recompute) and is
# therefore already correct on the next call.
```

### 5.7 Variable Mapping

| Math symbol               | Julia expression                              | Type               | Notes                                  |
|---------------------------|-----------------------------------------------|--------------------|----------------------------------------|
| $x_k$                     | `state.iterate.x`                             | `Vector{Float64}`  | current iterate                        |
| $x_{k-1}$                | `state.iterate.x_prev`                        | `Vector{Float64}`  | previous iterate; set by `step!`       |
| $\nabla f(x_k)$          | `state.iterate.gradient`                      | `Vector{Float64}`  | current gradient                       |
| $\nabla f(x_{k-1})$      | `state.numerics.grad_prev`                    | `Vector{Float64}`  | updated in `step!` after `compute_step_size`|
| $s_{k-1}$                 | `state.iterate.x .- state.iterate.x_prev`     | `Vector{Float64}`  | step difference                        |
| $y_{k-1}$                 | `state.iterate.gradient .- grad_prev`         | `Vector{Float64}`  | gradient difference                    |
| $s_{k-1}^T y_{k-1}$      | `dot(s, y)`                                   | `Float64`          | curvature; must be > 0                 |
| $\alpha_k^{BB1}$          | `dot(s, s) / dot(s, y)`                       | `Float64`          | long step                              |
| $\alpha_k^{BB2}$          | `dot(s, y) / dot(y, y)`                       | `Float64`          | short step                             |

---

## 6. Extension — Adding a New Step-Size Rule

1. Decide whether your rule is a **direct** rule or a **line search**:
   - Direct rule (closed-form $\alpha_k$ from current state, no trial points
     tested) — subtype `StepSize` directly.
   - Line search (tests trial points along $d_k$ until a sufficient-decrease or
     curvature condition holds) — subtype `LineSearch`.

2. Add a numbered section to this file following the template above:
   derivation, Julia struct, implementation, cost. Add a variable-mapping table
   only when the symbol↔code mapping isn't obvious from the annotated code (as for
   `CauchyStep` and `BarzilaiBorwein`); a direct rule like `FixedStep` needs none.

3. Add the concrete struct and `compute_step_size` method to
   `components/step_sizes.jl`.

4. Inside `compute_step_size`, wrap only the mathematically core operations in
   `@core_timed state`. Use the following as a guide:

   | Operation type | Inside `@core_timed`? |
   |---|---|
   | Dot products, norms, linear algebra on $O(n)$ vectors | Yes |
   | `apply(H, d)` calls on a `Hessian` object | Yes |
   | `hessian(problem.f, x)` construction | Yes |
  | Calls to `total_objective(problem, ...)`, `value(problem.f, ...)`, or `grad!(..., problem.f, ...)` | Yes |
   | Guard checks, fallback returns, counter increments | No — bookkeeping |

5. If the rule requires additional per-iteration state (as BB requires
   `grad_prev`), add the field to `GradientDescentNumerics`, initialize it in
   `init_state`, and document the update ordering explicitly — who writes it,
   when, and why the ordering relative to `compute_step_size` matters.

6. Pass the new rule as the `step_size` field in `GradientDescent(...)`.
   No changes to `step!`, the runner, logging, or stopping criteria are required.

7. If you want a custom short name in plot legends and filenames, register an
   abbreviation:
   ```julia
   register_abbreviation!("YourStepSize", "Yrs")
   ```

---

## 7. References

- Armijo, L. (1966). *Minimization of functions having Lipschitz continuous first
  partial derivatives.* Pacific Journal of Mathematics, 16(1), 1–3.
- Barzilai, J. & Borwein, J.M. (1988). *Two-point step size gradient methods.*
  IMA Journal of Numerical Analysis, 8(1), 141–148.
- Nocedal, J. & Wright, S.J. (2006). *Numerical Optimization* (2nd ed.). Springer.
  §3.1 (Armijo), §3.3 (Barzilai-Borwein).
- Dai, Y.H. & Liao, L.Z. (2002). *R-linear convergence of the Barzilai and Borwein
  gradient method.* IMA Journal of Numerical Analysis, 22(1), 1–10.
