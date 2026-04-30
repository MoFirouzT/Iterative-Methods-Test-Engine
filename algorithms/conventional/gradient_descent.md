# Gradient Descent — Implementation Specification

> **Related specs** (read these first):
> - `problems/rosenbrock/SPEC.md` — problem interface, `DataFidelity`, `grad`, `objective`
> - `algorithms/conventional/gradient_descent/DIRECTIONS.md` — `DescentDirection` abstraction
> - `algorithms/conventional/gradient_descent/STEP_SIZES.md` — `StepSizeRule` abstraction
> - `src/ARCHITECTURE.md` — state groups, runner contract, `@core_timed`, logger injection

---

## 1. Method Overview

**Gradient descent** generates a sequence $\{x_k\}$ by repeatedly moving from the
current iterate $x_k$ along a descent direction $d_k$, scaled by a step size $\alpha_k$:

$$x_{k+1} = x_k + \alpha_k\, d_k$$

The descent direction and step-size rule are **pluggable components**: they are
specified at construction time and dispatched via the `DescentDirection` and
`StepSizeRule` abstractions. Adding a new direction or step-size rule requires no
changes to `step!` or any other part of the algorithm.

This is a **conventional method** in the framework: it is a standard baseline against
which experimental methods are benchmarked.

---

## 2. Mathematical Formulation

### 2.1 Iteration

Given $x_k \in \mathbb{R}^n$:

1. Compute gradient: $g_k = \nabla f(x_k)$
2. Compute descent direction: $d_k = \texttt{compute\_direction}(\text{dir},\, x_k,\, g_k)$
3. Compute step size: $\alpha_k = \texttt{compute\_step}(\text{rule},\, x_k,\, g_k,\, d_k)$
4. Update: $x_{k+1} = x_k + \alpha_k\, d_k$

For the steepest descent direction ($d_k = -g_k$) this is:

$$x_{k+1} = x_k - \alpha_k\, \nabla f(x_k)$$

### 2.2 Convergence (Steepest Descent, Smooth Strongly Convex $f$)

For $f$ that is $\mu$-strongly convex with $L$-Lipschitz gradient:

$$\|x_{k+1} - x^*\|^2 \leq \left(1 - \frac{2\mu\alpha_k}{1 + \mu\alpha_k L}\right) \|x_k - x^*\|^2$$

With the optimal fixed step $\alpha = \frac{2}{\mu + L}$, this gives the linear rate:

$$\|x_{k+1} - x^*\|^2 \leq \left(\frac{\kappa - 1}{\kappa + 1}\right)^2 \|x_k - x^*\|^2, \qquad \kappa = \frac{L}{\mu}$$

For the Rosenbrock problem $\kappa \approx 2508$ near $x^*$, yielding a very slow
convergence rate. This is the motivation for using better-conditioned step-size rules
(BB, Cauchy) or directions (Newton, L-BFGS).

---

## 3. Julia Structs

### 3.1 Method Struct

```julia
# In: algorithms/conventional/gradient_descent/gradient_descent.jl

@kwdef struct GradientDescent <: ConventionalMethod
    direction  :: DescentDirection = SteepestDescent()
    step_rule  :: StepSizeRule     = ArmijoLS()
end
```

Both fields are abstract — any concrete `DescentDirection` or `StepSizeRule` can be
swapped in without touching any other code.

### 3.2 State — Shared Groups

```julia
@kwdef mutable struct GradientDescentState
    # ── Shared groups (framework convention — all methods carry these) ──────────
    iterate  :: IterateGroup     # x, gradient, x_prev
    metrics  :: MetricsGroup     # objective, gradient_norm, step_norm, dist_to_opt
    timing   :: TimingGroup      # core_time_ns (reset by runner before each step!)

    # ── Method-specific numerics ─────────────────────────────────────────────────
    numerics :: GradientDescentNumerics
    # Note: NO _logger field — logger is injected as a step! parameter by the runner
end
```

### 3.3 Method-Specific Numerics

```julia
@kwdef mutable struct GradientDescentNumerics
    direction          :: Vector{Float64} = Float64[]  # d_k; computed by compute_direction
    n_linesearch_evals :: Int             = 0          # cumulative line search f() calls
    grad_prev          :: Vector{Float64} = Float64[]  # ∇f(x_{k-1}); required by BB only
end
```

> **Field ownership.**
> - `direction` — written by `step!` via `compute_direction`; read by `compute_step`.
> - `n_linesearch_evals` — incremented by `step!`; exposed in `extras` for logging.
> - `grad_prev` — written by `step!` *after* `compute_step` returns; read by BB variant
>   on the *next* call to `compute_step`. Ordering is critical (see §5).

---

## 4. `init_state` Contract

**Signature:**

```julia
function init_state(method::GradientDescent, problem, rng::AbstractRNG)::GradientDescentState
```

**What to compute:**

| Field | How to initialize | Notes |
|-------|-------------------|-------|
| `iterate.x` | `copy(problem.x0)` | copy — never alias the problem's vector |
| `iterate.gradient` | `grad(problem.f, problem.x0)` | compute at $x_0$ |
| `iterate.x_prev` | `Float64[]` | empty; filled after first step! call |
| `metrics.objective` | `objective(problem, problem.x0)` | compute at $x_0$ |
| `metrics.gradient_norm` | `norm(iterate.gradient)` | from above |
| `metrics.step_norm` | `0.0` | no step taken yet |
| `metrics.dist_to_opt` | `isnothing(problem.x_opt) ? Inf : norm(problem.x0 .- problem.x_opt)` | runner will update each step |
| `timing.core_time_ns` | `0` | reset by runner before each `step!` anyway |
| `numerics.direction` | `Float64[]` | filled on first `step!` |
| `numerics.n_linesearch_evals` | `0` | cumulative counter |
| `numerics.grad_prev` | `Float64[]` | empty; used only by BB from k≥2 |

**Implementation:**

```julia
function init_state(method::GradientDescent, problem,
                    rng::AbstractRNG)::GradientDescentState
    x0 = copy(problem.x0)
    g0 = grad(problem.f, x0)
    f0 = objective(problem, x0)

    GradientDescentState(
        iterate  = IterateGroup(x=x0, gradient=g0, x_prev=Float64[]),
        metrics  = MetricsGroup(
                       objective     = f0,
                       gradient_norm = norm(g0),
                       step_norm     = 0.0,
                       dist_to_opt   = isnothing(problem.x_opt) ?
                                       Inf : norm(x0 .- problem.x_opt)
                   ),
        timing   = TimingGroup(core_time_ns=0),
        numerics = GradientDescentNumerics(),
    )
end
```

> `rng` is accepted as part of the framework interface but not used here — gradient
> descent is deterministic given $x_0$.

---

## 5. `step!` Contract

**Signature:**

```julia
function step!(method::GradientDescent, state::GradientDescentState,
               problem, iter::Int, logger::Logger, rng::AbstractRNG)
```

**Preconditions** (guaranteed by runner on entry):

- `state.timing.core_time_ns == 0` — reset by runner immediately before this call.
- `state.iterate.x` holds the current $x_k$.
- `state.iterate.gradient` holds $\nabla f(x_k)$ from the *previous* step (stale — must recompute).
- `state.metrics.objective` holds $f(x_k)$ from the *previous* step (stale — must recompute).
- `logger` is the injected logger for this method+run — forward to `run_sub_method` if needed.
- `rng` is the method's reproducible RNG stream — forward to `run_sub_method` if needed.

**Postconditions** (expected by runner on exit):

- `state.iterate.x` holds $x_{k+1}$.
- `state.iterate.x_prev` holds $x_k$ (the iterate on entry).
- `state.iterate.gradient` holds $\nabla f(x_{k+1})$.
- `state.metrics.objective` holds $f(x_{k+1})$.
- `state.metrics.gradient_norm` holds $\|\nabla f(x_{k+1})\|_2$.
- `state.metrics.step_norm` holds $\|\alpha_k d_k\|_2 = \|x_{k+1} - x_k\|_2$.
- `state.metrics.dist_to_opt` is updated by the **runner** after `step!` returns —
  the algorithm must not set it.
- `state.timing.core_time_ns` has been incremented by the time spent inside `@core_timed`.
- `state.numerics.grad_prev` holds $\nabla f(x_k)$ (the gradient on entry to this call).

**Timing discipline:**

| Operation | Inside `@core_timed`? | Reason |
|-----------|----------------------|--------|
| `grad(problem.f, x_k)` | **Yes** | Core mathematical computation |
| `compute_direction(...)` | **Yes** | Core mathematical computation |
| `compute_step(...)` with FixedStep / BB / Cauchy | **Yes** | $O(n)$ dot products only |
| `compute_step(...)` with ArmijoLS | **Yes** | Function evaluations are core computation |
| `x_{k+1} = x_k + α d_k` | **Yes** | Core update |
| `norm(...)`, metric updates | **No** | Bookkeeping |

> **ArmijoLS timing.** Line search calls to `objective(problem, ...)` are
> part of the core computation and should be wrapped in `@core_timed`. The count
> is tracked in `state.numerics.n_linesearch_evals`.
> All other step-size rules (FixedStep, Cauchy, BB) contain only $O(n)$ dot
> products and Hessian-vector products, which also count as core computation.

**Implementation:**

```julia
function step!(method::GradientDescent, state::GradientDescentState,
               problem, iter::Int, logger::Logger, rng::AbstractRNG)

    # ── Save previous iterate ───────────────────────────────────────────────────
    x_prev = copy(state.iterate.x)   # needed for x_prev and for BB's s_{k-1}

    # ── Core: gradient and descent direction ────────────────────────────────────
    @core_timed state begin
        g_k = grad(problem.f, state.iterate.x)          # ∇f(x_k)
        state.iterate.gradient = g_k

        d_k = compute_direction(method.direction, state, problem)  # e.g. -g_k
        state.numerics.direction = d_k
    end

    # ── Step-size selection ─────────────────────────────────────────────────────
    # ArmijoLS: function evals excluded from core timing — tracked separately
    # All other rules: wrapped in @core_timed below
    if method.step_rule isa ArmijoLS
        n_before = state.numerics.n_linesearch_evals
        α_k      = compute_step(method.step_rule, state, problem, d_k)
        # compute_step updates internal eval counter via the problem calls it makes;
        # track here by counting objective calls (ArmijoLS exposes this via return metadata
        # or by wrapping objective — implementation choice)
    else
        local α_k
        @core_timed state begin
            α_k = compute_step(method.step_rule, state, problem, d_k)
        end
    end

    # ── Core: iterate update ─────────────────────────────────────────────────────
    @core_timed state begin
        step_vec           = α_k .* d_k
        state.iterate.x   .+= step_vec               # x_{k+1} = x_k + α_k d_k
    end

    # ── Bookkeeping (outside @core_timed) ────────────────────────────────────────
    state.iterate.x_prev          = x_prev            # store x_k for BB and logging
    state.numerics.grad_prev      = copy(g_k)         # store ∇f(x_k) for BB (next iter)

    state.metrics.objective       = objective(problem, state.iterate.x)   # f(x_{k+1})
    state.iterate.gradient        = grad(problem.f, state.iterate.x)      # ∇f(x_{k+1})
    state.metrics.gradient_norm   = norm(state.iterate.gradient)
    state.metrics.step_norm       = norm(step_vec)
    # dist_to_opt is computed and set by the runner — do not set here
end
```

---

## 6. `extract_log_entry` Contract

**Signature:**

```julia
function extract_log_entry(method::GradientDescent, state::GradientDescentState,
                            iter::Int)::IterationLog
```

The default framework fields are populated automatically from `state.metrics`.
Override this function only to add algorithm-specific fields to `extras`:

```julia
function extract_log_entry(method::GradientDescent, state::GradientDescentState,
                            iter::Int)::IterationLog
    IterationLog(
        iter           = iter,
        core_time_ns   = state.timing.core_time_ns,
        objective      = state.metrics.objective,
        gradient_norm  = state.metrics.gradient_norm,
        step_norm      = state.metrics.step_norm,
        dist_to_opt    = state.metrics.dist_to_opt,    # Inf when x_opt not set
        extras         = Dict{Symbol,Any}(
            :n_linesearch_evals => state.numerics.n_linesearch_evals,
            :step_size          => norm(state.metrics.step_norm) /
                                   max(norm(state.numerics.direction), 1e-16),
            # α_k recovered from ‖α d‖ / ‖d‖ — exact when d is not rescaled
        ),
    )
end
```

---

## 7. Full Variable Mapping

| Math symbol                  | Julia expression                              | Type                | Notes                              |
|------------------------------|-----------------------------------------------|---------------------|------------------------------------|
| $x_k$                        | `state.iterate.x`                             | `Vector{Float64}`   | current iterate                    |
| $x_{k-1}$                   | `state.iterate.x_prev`                        | `Vector{Float64}`   | set in `step!` at end of each step |
| $x^*$                        | `problem.x_opt`                               | `Vector{Float64}` or `nothing` | known minimizer        |
| $f(x_k)$                     | `state.metrics.objective`                     | `Float64`           | stale on entry; refreshed in step! |
| $\nabla f(x_k)$ (fresh)     | `grad(problem.f, state.iterate.x)`           | `Vector{Float64}`   | computed inside `@core_timed`      |
| $\nabla f(x_k)$ (stored)    | `state.iterate.gradient`                      | `Vector{Float64}`   | written at end of step!            |
| $\nabla f(x_{k-1})$         | `state.numerics.grad_prev`                    | `Vector{Float64}`   | written after compute_step         |
| $d_k$                        | `state.numerics.direction`                    | `Vector{Float64}`   | from `compute_direction`           |
| $\alpha_k$                   | local `α_k` in `step!`                        | `Float64`           | from `compute_step`                |
| $\alpha_k d_k$               | local `step_vec`                              | `Vector{Float64}`   | stored implicitly in x diff        |
| $\|\nabla f(x_k)\|$         | `state.metrics.gradient_norm`                 | `Float64`           | used by GradientTolerance          |
| $\|x_{k+1} - x_k\|$        | `state.metrics.step_norm`                     | `Float64`           | used by StepTolerance              |
| $\|x_k - x^*\|$             | `state.metrics.dist_to_opt`                   | `Float64`           | set by runner, not step!           |

---

## 8. Convergence & Edge Cases

| Condition | Consequence | Safeguard |
|-----------|-------------|-----------|
| $\|\nabla f(x_k)\| = 0$ | `compute_direction` returns zero vector | `GradientTolerance` stopping criterion fires first |
| `CauchyStep`: $d_k^T H_k d_k \leq 0$ | Denominator non-positive | `fallback_α` used; see `STEP_SIZES.md §3` |
| `BarzilaiBorwein`: $s^T y \leq 0$ | Secant curvature violated | `fallback_α` used; see `STEP_SIZES.md §5` |
| `ArmijoLS`: `max_iter` backtracking steps reached | Very small $\alpha_k$ accepted | Log via `n_linesearch_evals`; consider reducing `α₀` |
| BB at $k = 1$ | `grad_prev` is empty | `fallback_α` used; see `STEP_SIZES.md §5.5` |
| `FixedStep`: $\alpha \geq 2/L$ | Divergence | User responsibility; document valid range in config |

---

## 9. Integration Example

```julia
# Conventional GD with Armijo line search on Rosenbrock (classical starting point)
gd_armijo = GradientDescent(
    direction = SteepestDescent(),
    step_rule = ArmijoLS(α₀=1.0, β=0.5, c₁=1e-4),
)

# GD with Barzilai-Borwein (long step)
gd_bb1 = GradientDescent(
    direction = SteepestDescent(),
    step_rule = BarzilaiBorwein(variant=:BB1, fallback_α=1e-3),
)

config = ExperimentConfig(
    name                 = "GD step-size comparison — Rosenbrock",
    problem_spec         = AnalyticProblem(name=:rosenbrock,
                               params=(rho=100.0, x0=[-1.2, 1.0])),
    conventional_methods = [gd_armijo, gd_bb1,
                             GradientDescent(step_rule=CauchyStep()),
                             GradientDescent(step_rule=FixedStep(α=8e-4))],
    stopping_criteria    = stop_when_any(
                               MaxIterations(n=5000),
                               DistanceToOptimal(tol=1e-8),
                               GradientTolerance(tol=1e-9),
                           ),
    n_runs = 1,
    seed   = 42,
)
```

---

## 10. References

- Nocedal, J. & Wright, S.J. (2006). *Numerical Optimization* (2nd ed.). Springer.
  - §2.2 — Steepest descent convergence rate, condition number dependence.
  - §3.1 — Armijo condition and backtracking line search.
  - §3.3 — Barzilai-Borwein step sizes.
- Rosenbrock, H.H. (1960). *An automatic method for finding the greatest or least value
  of a function.* The Computer Journal, 3(3), 175–184.
- Barzilai, J. & Borwein, J.M. (1988). *Two-point step size gradient methods.*
  IMA Journal of Numerical Analysis, 8(1), 141–148.
- See also: `DIRECTIONS.md` and `STEP_SIZES.md` in this directory for the full
  mathematical derivation of each pluggable component.
