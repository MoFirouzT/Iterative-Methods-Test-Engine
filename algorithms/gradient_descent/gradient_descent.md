# Gradient Descent

> **Related specs** (read these first):
> - `problems/rosenbrock/rosenbrock.md` — problem interface, `Objective`, `grad!`, `hessian`
> - `algorithms/components/descent_directions.md` — `DescentDirection` abstraction
> - `algorithms/components/step_sizes.md` — `StepSize` / `LineSearch` abstractions
> - `docs/src/modules/algorithm-core.md` — state groups, runner contract, `@core_timed`, logger injection

---

## 1. Method Overview

**Gradient descent** generates a sequence $\{x_k\}$ by repeatedly moving from the
current iterate $x_k$ along a descent direction $d_k$, scaled by a step size
$\alpha_k$:

$$x_{k+1} = x_k + \alpha_k\, d_k$$

The descent direction and step-size rule are **pluggable components**: they are
specified at construction time and dispatched via the `DescentDirection` and
`StepSize` abstractions. Adding a new direction or step-size rule requires no
changes to `step!` or any other part of the algorithm.

In comparisons it typically serves as a **baseline** — the standard reference
against which experimental methods are benchmarked. That role is declared per
experiment (via `baseline_methods` or a grid's `role`), not encoded in the
method's type.

---

## 2. Mathematical Formulation

### 2.1 Iteration

Given $x_k \in \mathbb{R}^n$:

1. Compute gradient: $g_k = \nabla f(x_k)$
2. Compute descent direction: $d_k = \texttt{compute\_direction}(\text{dir},\, x_k,\, g_k)$
3. Compute step size: $\alpha_k = \texttt{compute\_step\_size}(\text{rule},\, x_k,\, g_k,\, d_k)$
4. Update: $x_{k+1} = x_k + \alpha_k\, d_k$

For the steepest descent direction ($d_k = -g_k$) this is:

$$x_{k+1} = x_k - \alpha_k\, \nabla f(x_k)$$

### 2.2 Convergence (Steepest Descent, Smooth Strongly Convex $f$)

For $f$ that is $\mu$-strongly convex with $L$-Lipschitz gradient:

$$\|x_{k+1} - x^*\|^2 \leq \left(1 - \frac{2\mu\alpha_k}{1 + \mu\alpha_k L}\right) \|x_k - x^*\|^2$$

With the optimal fixed step $\alpha = \frac{2}{\mu + L}$, this gives the linear rate:

$$\|x_{k+1} - x^*\|^2 \leq \left(\frac{\kappa - 1}{\kappa + 1}\right)^2 \|x_k - x^*\|^2, \qquad \kappa = \frac{L}{\mu}$$

For the Rosenbrock problem $\kappa \approx 2508$ near $x^*$, yielding a very slow
convergence rate. This is the motivation for using better-conditioned step-size
rules (BB, Cauchy) or richer descent directions (Newton, L-BFGS) — though only
the step-size axis is exercised in the current setup.

---

## 3. Julia Structs

### 3.1 Method Struct

```julia
# In: algorithms/gradient_descent/gradient_descent.jl

@kwdef struct GradientDescent <: IterativeMethod
    direction :: DescentDirection = SteepestDescent()
    step_size :: StepSize         = ArmijoLS()
end
```

Both fields are abstract — any concrete `DescentDirection` or `StepSize` can be
swapped in without touching any other code. (`ArmijoLS <: LineSearch <: StepSize`,
so it satisfies the field type.)

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
    direction          :: Vector{Float64} = Float64[]   # d_k; written by compute_direction
    α_k                :: Float64         = 0.0         # last step size; written by step!
    n_linesearch_evals :: Int             = 0           # cumulative line-search f() calls
    grad_prev          :: Vector{Float64} = Float64[]   # ∇f(x_{k-1}); required by BB
    x_trial            :: Vector{Float64} = Float64[]   # scratch buffer for line-search trials
end
```

> **Field ownership.**
> - `direction` — written by `step!` via `compute_direction`; read by
>   `compute_step_size`.
> - `α_k` — written by `step!` immediately after `compute_step_size` returns;
>   read by `extract_log_entry`.
> - `n_linesearch_evals` — incremented by `compute_step_size` for line searches
>   (currently `ArmijoLS`); exposed in `extras` for logging.
> - `grad_prev` — written by `step!` *after* the iterate update and *before* the
>   gradient refresh, using the empty-vector sentinel to detect the first
>   iteration. Read by `BarzilaiBorwein` on the *next* call to
>   `compute_step_size`. Ordering is critical (see §5).
> - `x_trial` — scratch buffer for line-search trial points. Sized in
>   `init_state` to `similar(problem.x0)`. Used by `ArmijoLS`; available to any
>   rule that needs a per-iteration trial-point buffer without allocating.

---

## 4. `init_state` Contract

**Signature:**

```julia
function init_state(method::GradientDescent, problem,
                    rng::AbstractRNG)::GradientDescentState
```

**What to compute:**

| Field | How to initialize | Notes |
|-------|-------------------|-------|
| `iterate.x` | `copy(problem.x0)` | copy — never alias the problem's vector |
| `iterate.gradient` | `g0 = similar(x0); grad!(g0, problem.f, x0)` | compute at $x_0$, in place |
| `iterate.x_prev` | `Float64[]` | empty; sentinel — filled in-place after first step! call |
| `metrics.objective` | `total_objective(problem, problem.x0)` | reduces to `value(problem.f, x_0)` when `gs` is empty |
| `metrics.gradient_norm` | `norm(iterate.gradient)` | from above |
| `metrics.step_norm` | `0.0` | no step taken yet |
| `metrics.dist_to_opt` | `isnothing(problem.x_opt) ? Inf : norm(problem.x0 .- problem.x_opt)` | runner will update each step |
| `timing.core_time_ns` | `0` | reset by runner before each `step!` anyway |
| `numerics.direction` | `Float64[]` | filled on first `step!` |
| `numerics.α_k` | `0.0` | written by first `step!` |
| `numerics.n_linesearch_evals` | `0` | cumulative counter |
| `numerics.grad_prev` | `Float64[]` | empty; sentinel — filled in-place after first step! |
| `numerics.x_trial` | `similar(problem.x0)` | preallocated; required by `ArmijoLS` |

**Implementation:**

```julia
function init_state(method::GradientDescent, problem,
                    rng::AbstractRNG)::GradientDescentState
    x0 = copy(problem.x0)
    g0 = similar(x0)
    grad!(g0, problem.f, x0)
    f0 = total_objective(problem, x0)

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
        numerics = GradientDescentNumerics(x_trial = similar(problem.x0)),
    )
end
```

> `rng` is accepted as part of the framework interface but not used here —
> gradient descent is deterministic given $x_0$.

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
- `state.iterate.gradient` holds $\nabla f(x_k)$ from the *previous* step.
    The runner / `init_state` is responsible for providing this fresh gradient at
    entry so that `step!` can avoid recomputing it. If the runner does not
    provide a fresh gradient, `step!` must compute it before calling
    `compute_direction`.
- `state.metrics.objective` holds $f(x_k)$ from the *previous* step (stale — must
  be recomputed).
- `logger` is the injected logger for this method+run — forward to
  `run_sub_method` if needed.
- `rng` is the method's reproducible RNG stream — forward to `run_sub_method`
  if needed.

**Postconditions** (expected by runner on exit):

- `state.iterate.x` holds $x_{k+1}$.
- `state.iterate.x_prev` holds $x_k$ (the iterate on entry).
- `state.iterate.gradient` holds $\nabla f(x_{k+1})$ — **freshly recomputed** at
  the new iterate.
- `state.metrics.objective` holds $f(x_{k+1})$ — freshly recomputed.
- `state.metrics.gradient_norm` and `state.metrics.step_norm` are updated.
- `state.metrics.dist_to_opt` is **not** set here (runner sets it).
- `state.numerics.direction` holds the descent direction used for this step.
- `state.numerics.α_k` holds the step size $\alpha_k$ used for this step.
- `state.numerics.grad_prev` holds $\nabla f(x_k)$ (the gradient at the iterate
  *before* the step) — required for BB's next-iteration secant pair.

**Timing discipline:**

| Operation | Inside `@core_timed`? | Reason |
|-----------|----------------------|--------|
| `grad!(state.iterate.gradient, ...)` | **Yes** | Core mathematical computation |
| `compute_direction(...)` | **Yes** | Core mathematical computation |
| `compute_step_size(...)` for any rule | **Inside the rule** | Each rule wraps its own kernel |
| `x_{k+1} = x_k + α_k d_k` (broadcast) | **Yes** | Core update |
| `total_objective(problem, x_{k+1})`, `grad!(...)` (refresh) | **Yes** | Core computation at new iterate |
| `copyto!(...)` for `x_prev` / `grad_prev` saves | **No** | Bookkeeping (state shuffling) |
| `norm(...)`, metric updates | **No** | Bookkeeping |

> **Step-size timing is owned by the rule.** Each `compute_step_size`
> implementation calls `@core_timed state begin … end` around its own kernel
> (see `step_sizes.md`). `step!` therefore does **not** wrap the
> `compute_step_size` call in `@core_timed` — doing so would double-count.
> For `ArmijoLS`, function evaluations are part of the kernel and are counted;
> `state.numerics.n_linesearch_evals` records how many trial evaluations were
> needed, which is logged in `extras` but is independent of the timing.

**In-place state saves.** Both `state.iterate.x_prev` and
`state.numerics.grad_prev` use the empty-vector sentinel (`Float64[]`) to mark
"no previous data." On the first iteration the buffer is allocated via `copy`;
on every subsequent iteration the existing buffer is reused via `copyto!`.
This eliminates per-step allocation while preserving the sentinel that BB
relies on (see `step_sizes.md` §5.5).

**No `step_vec` temporary.** The displacement $\alpha_k d_k$ is applied
directly to `state.iterate.x` via fused broadcast (`.+=`). The step norm is
computed analytically as $|\alpha_k| \cdot \|d_k\|$, avoiding a per-step
allocation.

**Implementation:**

```julia
function step!(method::GradientDescent, state::GradientDescentState,
               problem, iter::Int, logger::Logger, rng::AbstractRNG)

    # ── Save x_k into x_prev (allocate once on iter 1, then reuse) ──────────────
    if isempty(state.iterate.x_prev)
        state.iterate.x_prev = copy(state.iterate.x)
    else
        copyto!(state.iterate.x_prev, state.iterate.x)
    end

    # ── Core: descent direction at x_k (∇f(x_k) already computed in init/last step) ───
    @core_timed state begin
        state.numerics.direction = compute_direction(method.direction,
                                                    state, problem)          # d_k
    end

    # ── Step-size selection ─────────────────────────────────────────────────────
    # Each StepSize rule wraps its own core operations in @core_timed; do NOT
    # wrap this call here.
    α_k = compute_step_size(method.step_size, state, problem,
                            state.numerics.direction)
    state.numerics.α_k = α_k                                  # store for logging

    # ── Core: iterate update (fused broadcast, no temporary) ────────────────────
    @core_timed state begin
        state.iterate.x .+= α_k .* state.numerics.direction   # x_{k+1} = x_k + α_k d_k
    end

    # ── Save ∇f(x_k) into grad_prev (allocate once on iter 1, then reuse) ───────
    # MUST happen before the gradient refresh below — at this point
    # state.iterate.gradient still holds ∇f(x_k).
    if isempty(state.numerics.grad_prev)
        state.numerics.grad_prev = copy(state.iterate.gradient)
    else
        copyto!(state.numerics.grad_prev, state.iterate.gradient)
    end

    # ── Core: refresh objective and gradient at x_{k+1} ─────────────────────────
    @core_timed state begin
        state.metrics.objective = total_objective(problem, state.iterate.x)
        grad!(state.iterate.gradient, problem.f, state.iterate.x)
    end

    # ── Bookkeeping (outside @core_timed) ────────────────────────────────────────
    state.metrics.gradient_norm = norm(state.iterate.gradient)
    state.metrics.step_norm     = abs(α_k) * norm(state.numerics.direction)
    # dist_to_opt is computed and set by the runner — do not set here.
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
Override this function only to add algorithm-specific fields to `extras`. The
step size is read directly from `state.numerics.α_k` — no reconstruction from
norms is needed, since `step!` stores the value produced by `compute_step_size`.

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
            :step_size          => state.numerics.α_k,
        ),
    )
end
```

---

## 7. Full Variable Mapping

| Math symbol                  | Julia expression                              | Type                | Notes                              |
|------------------------------|-----------------------------------------------|---------------------|------------------------------------|
| $x_k$                        | `state.iterate.x`                             | `Vector{Float64}`   | current iterate                    |
| $x_{k-1}$                    | `state.iterate.x_prev`                        | `Vector{Float64}`   | written in-place by `step!` at start of each step (allocated once, reused thereafter) |
| $x^*$                        | `problem.x_opt`                               | `Vector{Float64}` or `nothing` | known minimizer        |
| $f(x_k)$                     | `state.metrics.objective`                     | `Float64`           | refreshed at end of `step!`        |
| $\nabla f(x_k)$ (fresh at $x_k$) | `grad!(state.iterate.gradient, problem.f, state.iterate.x)` | (writes `Vector{Float64}`) | computed inside `@core_timed`; in-place |
| $\nabla f(x_k)$ (stored)    | `state.iterate.gradient`                      | `Vector{Float64}`   | refreshed at end of `step!`        |
| $\nabla f(x_{k-1})$         | `state.numerics.grad_prev`                    | `Vector{Float64}`   | written in-place before metric refresh (allocated once, reused thereafter) |
| $\nabla^2 f(x_k)$           | `hessian(problem.f, state.iterate.x)`         | `Hessian` object    | used by `CauchyStep`               |
| $d_k$                        | `state.numerics.direction`                    | `Vector{Float64}`   | from `compute_direction`           |
| $\alpha_k$                   | `state.numerics.α_k`                          | `Float64`           | written by `step!` from `compute_step_size`; read by `extract_log_entry` |
| $\alpha_k d_k$ (displacement) | `α_k .* state.numerics.direction` (fused into update) | (no buffer)  | applied via `state.iterate.x .+=`; no temporary allocated |
| $\|\nabla f(x_k)\|$         | `state.metrics.gradient_norm`                 | `Float64`           | used by `GradientTolerance`        |
| $\|x_{k+1} - x_k\|$        | `state.metrics.step_norm`                     | `Float64`           | computed as `abs(α_k) * norm(d_k)`; used by `StepTolerance` |
| $\|x_k - x^*\|$             | `state.metrics.dist_to_opt`                   | `Float64`           | set by runner, not `step!`         |

---

## 8. Convergence & Edge Cases

| Condition | Consequence | Safeguard |
|-----------|-------------|-----------|
| $\|\nabla f(x_k)\| = 0$ | `compute_direction` returns zero vector | `GradientTolerance` stopping criterion fires first |
| `CauchyStep`: $d_k^T H_k d_k \leq 0$ | Denominator non-positive | `fallback_α` used; see `step_sizes.md` §3 |
| `BarzilaiBorwein`: $s^T y \leq 0$ | Secant curvature violated | `fallback_α` used; see `step_sizes.md` §5 |
| `ArmijoLS`: `max_iter` backtracking steps reached | Very small $\alpha_k$ accepted | Logged via `n_linesearch_evals`; consider reducing `α₀` |
| BB at $k = 1$ | `grad_prev` is empty | `fallback_α` used; see `step_sizes.md` §5.5 |
| `FixedStep`: $\alpha \geq 2/L$ | Divergence | User responsibility; document a valid range in config |

---

## 9. Integration Example

```julia
# Conventional GD with Armijo line search on Rosenbrock (classical starting point)
gd_armijo = GradientDescent(
    direction = SteepestDescent(),
    step_size = ArmijoLS(α₀=1.0, β=0.5, c₁=1e-4),
)

# GD with Barzilai-Borwein (long step)
gd_bb1 = GradientDescent(
    direction = SteepestDescent(),
    step_size = BarzilaiBorwein(variant=:BB1, fallback_α=1e-3),
)

# GD with Cauchy step
gd_cauchy = GradientDescent(
    direction = SteepestDescent(),
    step_size = CauchyStep(),
)

# GD with a fixed step
gd_fixed = GradientDescent(
    direction = SteepestDescent(),
    step_size = FixedStep(α=8e-4),
)

config = ExperimentConfig(
    name                 = "GD step-size comparison — Rosenbrock",
    problem_spec         = AnalyticProblem(name=:rosenbrock,
                               params=(rho=100.0, x0=[-1.2, 1.0])),
    baseline_methods     = [gd_armijo, gd_bb1, gd_cauchy, gd_fixed],
    stopping_criteria    = stop_when_any(
                               MaxIterations(n=5000),
                               DistanceToOptimal(tol=1e-8),
                               GradientTolerance(tol=1e-9),
                           ),
    n_runs = 1,
    seed   = 42,
)
```

The same comparison expressed as a single **variant grid** (recommended when the
comparison is purely along one axis):

```julia
step_size_axis = VariantAxis(:step_size,
    FixedStep(α=8e-4)              => "Fixed",
    ArmijoLS(α₀=1.0, β=0.5, c₁=1e-4) => "Armijo",
    BarzilaiBorwein(variant=:BB1)  => "BB1",
    BarzilaiBorwein(variant=:BB2)  => "BB2",
    CauchyStep()                   => "Cauchy",
)

grid = VariantGrid(
    base_name = "GradientDescent",
    axes      = [step_size_axis],
    builder   = (; step_size, kwargs...) ->
                    GradientDescent(direction=SteepestDescent(), step_size=step_size),
    role      = :baseline,
)

config = ExperimentConfig(
    name              = "GD step-size sweep — Rosenbrock",
    problem_spec      = AnalyticProblem(name=:rosenbrock,
                            params=(rho=100.0, x0=[-1.2, 1.0])),
    variant_grids     = [grid],
    stopping_criteria = stop_when_any(
                            MaxIterations(n=5000),
                            DistanceToOptimal(tol=1e-8),
                            GradientTolerance(tol=1e-9),
                        ),
    n_runs = 1,
    seed   = 42,
)
```

`resolve_methods` routes each grid output to the **baseline** bucket because
the grid declares `role = :baseline`.

---

## 10. Win Conditions

Gradient descent is the framework's **baseline**; "winning" here means it behaves
exactly as first-order theory predicts, so the experimental methods have an honest
reference to beat.

- **Step-size sweep (Rosenbrock, `stages/stage1.jl`, `stage2.jl`).** All five
  variants (`Fixed`, `Armijo`, `Cauchy`, `BB1`, `BB2`) trace the curved valley from
  `x₀ = (−1.2, 1)` and reduce `f`; `Cauchy` and `BB` reach high accuracy in far fewer
  iterations than `Fixed`, while `Armijo`'s `n_linesearch_evals` stays in the 1–3
  range per step.
- **Conditioning (least squares, `exp_ls2_conditioning.jl`).** Iters-to-tolerance vs
  `κ` on a log-log axis: `Fixed`/`Armijo`/`Cauchy` scale `O(κ)` (slope ≈ 1) while
  `BB` is markedly flatter (`O(√κ)`) — the slope difference is the validation.
- **Timing pillar (least squares, `exp_ls1_dimension.jl`).** The 2-D Rosenbrock
  kernel sits below the timing floor, but at `n = 1000` the `O(mn)` matvec dominates
  and the `core_time / wall_time` ratio climbs into a meaningful band — the honest
  ratio is itself the reported result.

## 11. References

- Nocedal, J. & Wright, S.J. (2006). *Numerical Optimization* (2nd ed.). Springer.
  - §2.2 — Steepest descent convergence rate, condition number dependence.
  - §3.1 — Armijo condition and backtracking line search.
  - §3.3 — Barzilai-Borwein step sizes.
- Rosenbrock, H.H. (1960). *An automatic method for finding the greatest or least
  value of a function.* The Computer Journal, 3(3), 175–184.
- Barzilai, J. & Borwein, J.M. (1988). *Two-point step size gradient methods.*
  IMA Journal of Numerical Analysis, 8(1), 141–148.
- See also: `descent_directions.md` and `step_sizes.md` in this directory for the
  full mathematical derivation of each pluggable component.
