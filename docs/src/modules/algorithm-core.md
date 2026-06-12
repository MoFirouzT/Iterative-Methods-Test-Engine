# Algorithm Abstraction & Core Timing

## Type Hierarchy

Every algorithm, conventional or experimental, is a concrete subtype of
`IterativeMethod`. The hierarchy separates conventional baselines from the experimental
family — the methods you are developing and testing in-engine — enabling dispatch and
experiment-level filtering.

```julia
abstract type IterativeMethod end
abstract type ConventionalMethod <: IterativeMethod end
abstract type ExperimentalMethod <: IterativeMethod end
```

Each method is a `@kwdef` struct carrying its own fixed hyperparameters.
**Stopping criteria** are supplied separately at experiment definition time
(see Module 3). This keeps the algorithm struct a pure description of the method,
not of how long to run it.

```julia
@kwdef struct GradientDescent <: ConventionalMethod
    direction :: DescentDirection = SteepestDescent()
    step_size :: StepSize         = ArmijoLS()
end
```

## State Parameter Groups — Composable Modules

A method's state can carry many heterogeneous fields: the current iterate, convergence
metrics, timing, and references to sub-solver states. The solution is to partition
state into **reusable, independently typed modules** that compose together.

**Three canonical shared modules** (identical across all methods):

```julia
# The optimization variables
@kwdef mutable struct IterateGroup
    x        :: Vector{Float64}              # current iterate
    gradient :: Vector{Float64}
    x_prev   :: Vector{Float64} = Float64[]  # previous iterate
end

# Scalar convergence metrics — mirrors the fixed fields of IterationLog
@kwdef mutable struct MetricsGroup
    objective     :: Float64 = Inf
    gradient_norm :: Float64 = Inf
    step_norm     :: Float64 = Inf
    dist_to_opt   :: Float64 = Inf  # ‖x − x*‖; set by runner when x_opt is known
end

# Per-step core computation time; reset by runner before each step!
@kwdef mutable struct TimingGroup
    core_time_ns :: Int64 = 0
end
```

**Method-specific numerics module** (one per concrete method):

```julia
# Example for GradientDescent: the descent direction buffer and any
# component-specific scratch state.
@kwdef mutable struct GradientDescentNumerics
    direction          :: Vector{Float64} = Float64[]
    n_linesearch_evals :: Int             = 0
    grad_prev          :: Vector{Float64} = Float64[]   # required by BB step size
end
```

> **Convention — no field duplication across groups.** A `Numerics` struct must never
> declare a field already present in a shared group. In particular, do **not** add a
> `gradient` or `objective` field to `Numerics`; always read and write these through
> `state.iterate.gradient` and `state.metrics.objective`. If the method needs a
> *separate* gradient buffer (e.g. for a sub-problem or a previous-step copy), name
> the field explicitly: `sub_gradient`, `grad_prev`, etc.

**Optional sub-solver modules** — embed a concrete sub-solver state directly:

```julia
@kwdef mutable struct InnerCGModule
    solver_state       :: ConjugateGradientState  # concrete type — never Any
    subproblem_iterate :: IterateGroup
end
```

**Concrete composed state struct**:

```julia
@kwdef mutable struct GradientDescentState
    iterate  :: IterateGroup
    metrics  :: MetricsGroup
    timing   :: TimingGroup
    numerics :: GradientDescentNumerics
    # Note: no _logger field — logger is passed explicitly via step! parameter
end
```

The state struct carries **no logger reference**. The runner injects the logger as an
explicit parameter to `step!` on every call, keeping algorithm code free of
logging infrastructure.

**Sub-routine state reuse.** When a method uses a nested solver, the outer state
struct includes a field typed as the sub-solver's concrete state type. However,
`run_sub_method` **creates and manages a fresh sub-state instance** each time it is
called in `step!` — the outer state can optionally store the final sub-state for
inspection, but it is **not used to initialize or control** the sub-run.

Each state (outer and sub) has its own independent `TimingGroup`. The sub-solver's
accumulated `core_time_ns` is reported in `SubResult.core_time_ns` and tracked
separately from the outer timing.

**Core-time attribution convention (settled by `TrustRegion`, Item 4).** When an
outer `step!` runs a sub-solver, the recommended convention is: **fold the inner
solve's total core time into the outer step's `core_time_ns`** (so cumulative-core
plots reflect *all* real work), **and** also expose it per-step in the log extras
(`:inner_core_ns`) for an inner/outer breakdown. Concretely the outer `step!` adds
`sub.core_time_ns` to `state.timing.core_time_ns` directly — it does **not** wrap
`run_sub_method` in `@core_timed`, which would wrongly count the inner *wall*
(scaffolding) time. `TrustRegion` does exactly this, and `test_trust_region.jl`
asserts `outer.core_time_ns ≥ inner_core_ns > 0`. Per-outer-iteration inner traces
are attached to each entry's `extras[:sub_logs]` by the outer method (rather than
relying on `finalize!`, which only attaches the *last* pending sub-log batch).

## `extract_log_entry` — Default Implementation

Because `state.metrics` mirrors `IterationLog`'s fixed fields, the default
implementation is trivial:

```julia
function extract_log_entry(method::IterativeMethod, state, iter::Int)::IterationLog
    IterationLog(
        iter          = iter,
        core_time_ns  = state.timing.core_time_ns,
        objective     = state.metrics.objective,
        gradient_norm = state.metrics.gradient_norm,
        step_norm     = state.metrics.step_norm,
        dist_to_opt   = state.metrics.dist_to_opt,
    )
end
# Methods override this to additionally populate the extras dict.
```

## The Three Dispatch Points

These three functions are the only interface an algorithm must implement:

```julia
# Called once before the loop; returns a mutable state object.
function init_state(method::IterativeMethod, problem, rng::AbstractRNG) end

# Called every iteration; mutates state in place.
# logger and rng are injected by the runner — algorithms may forward them
# to run_sub_method but must not store them in state.
# Core computation must be wrapped in @core_timed (see below).
function step!(method::IterativeMethod, state, problem, iter::Int,
               logger::Logger, rng::AbstractRNG) end

# Called to extract a log entry; dispatches on method type.
function extract_log_entry(method::IterativeMethod, state, iter::Int)::IterationLog end
```

## Core Timing — `@core_timed`

Scientific timing measures **only the mathematical kernel** of each step.
The `@core_timed` macro is the single entry point for this.
It accumulates elapsed nanoseconds into `state.timing.core_time_ns` and is
**exception-safe**: if the wrapped expression throws, the elapsed time up to the
exception is still recorded before the error is re-thrown.

```julia
"""
    @core_timed state expr

Wraps `expr` in a high-resolution timer.
Elapsed nanoseconds are **added** to `state.timing.core_time_ns`,
so multiple disjoint mathematical kernels in a single step are all counted
without timing the bookkeeping between them.
Exception-safe: partial time is recorded even if `expr` throws.
"""
macro core_timed(state, expr)
    quote
        _t0 = time_ns()
        try
            $(esc(expr))
        finally
            $(esc(state)).timing.core_time_ns += time_ns() - _t0
        end
    end
end
```

Usage inside `step!` — the algorithm author controls exactly what counts:

```julia
function step!(m::GradientDescent, state, problem, iter, logger, rng)
    @core_timed state begin
        grad!(state.iterate.gradient, problem.f, state.iterate.x)
        d = compute_direction(m.direction, state, problem)
        state.numerics.direction = d
    end

    # Step-size rule may itself call @core_timed for its core operations:
    α = compute_step_size(m.step_size, state, problem, d)

    @core_timed state begin
        state.iterate.x .+= α .* d
    end
    state.metrics.step_norm = norm(α .* d)
end
```

Every concrete state struct contains a `TimingGroup` field named `timing`. The runner
resets `state.timing.core_time_ns = 0` before each `step!` call so the logger always
sees a single-step measurement.

## The Generic Runner

The runner owns the loop. Algorithms never hold a logger reference — the logger is
passed as an explicit parameter to `step!` on every iteration. A `StoppingCriteria`
object controls termination (see Module 3). If `problem.x_opt` is set, the runner
computes `dist_to_opt` after each step and stores it in `state.metrics` before
`extract_log_entry` is called, so algorithms remain unaware of the optimal point.
Debug checks run after logging, before the stopping check.

```julia
function run_method(method   :: IterativeMethod,
                    problem,
                    criteria :: StoppingCriteria,
                    logger   :: Logger,
                    rng      :: AbstractRNG,
                    debug    :: DebugConfig = DebugConfig())

    state = init_state(method, problem, rng)
    log_init!(logger, method, state)
    iter       = 0
    prev_entry = nothing

    while true
        iter += 1
        state.timing.core_time_ns = 0           # reset per-step accumulator

        local entry
        try
            step!(method, state, problem, iter, logger, rng)
            # ↑ logger and rng forwarded; only this call contributes to core_time_ns

            # Runner computes dist_to_opt — algorithms never access problem.x_opt
            if !isnothing(problem.x_opt)
                state.metrics.dist_to_opt = norm(state.iterate.x .- problem.x_opt)
            end

            entry = extract_log_entry(method, state, iter)
            log_iter!(logger, entry)
            run_debug_checks!(debug, logger, state, problem, entry, prev_entry, iter)

        catch e
            log_event!(logger, :step_error, iter)
            break
        end

        prev_entry = entry

        # Stopping check happens AFTER logging — never timed
        stop, reason = should_stop(criteria, state, iter, logger)
        if stop
            log_event!(logger, reason, iter)
            break
        end
    end

    return finalize!(logger, method, state)     # returns MethodResult{typeof(state)}
end
```

---

