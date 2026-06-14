# Algorithm Abstraction, Core Timing & the Runner

## Method Type

Every algorithm is a concrete subtype of the single category `IterativeMethod`.
A method's **comparison role** — baseline vs experimental — is deliberately *not*
part of its type. It is experiment-level metadata declared when you assemble the
experiment (see [Experiment Orchestration](@ref)), so the same algorithm can be a
baseline in one experiment and the method under study in another.

```julia
abstract type IterativeMethod end
```

Each method is a `@kwdef` struct carrying its own fixed hyperparameters.
**Stopping criteria** are supplied separately at experiment definition time (see [Stopping Criteria](@ref)).
This keeps the algorithm struct a pure description of the method, not of how long to run it.

```julia
@kwdef struct GradientDescent <: IterativeMethod
    direction :: DescentDirection = SteepestDescent()
    step_size :: StepSize         = ArmijoLS()
end
```

## State Parameter Groups — Composable Modules

A method's state can carry many heterogeneous fields:
the current iterate, convergence metrics, timing, and references to sub-solver states.
The solution is to partition state into **reusable, independently typed modules** that compose together.

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
    dist_to_opt   :: Float64 = Inf
end

# Per-step core computation time; reset by runner before each step!
@kwdef mutable struct TimingGroup
    core_time_ns :: Int64 = 0
end
```

`TimingGroup` is kept separate from `MetricsGroup` even though it holds a single
field — the reason is categorical, not structural. `MetricsGroup` holds *convergence
mathematics the algorithm produces* (objective, gradient norm, …); `core_time_ns` is
a *measurement*, and honest measurement is this framework's whole thesis, so it earns
its own home rather than being folded into the math. The two also have different
lifecycles: the runner **resets** `timing` to zero before every `step!`, whereas
metrics are overwritten in place. (`IterationLog` likewise keeps `core_time_ns`
distinct from its metric fields.)

**Method-specific numerics module** (one per concrete method). "Numerics" is a
*naming convention*, not a type: there is no abstract `Numerics` supertype. Each
method defines its own concrete struct named `<Method>Numerics` (e.g.
`GradientDescentNumerics`) and stores it in the state's `numerics` field.

```julia
# Example for GradientDescent: the descent direction buffer and any
# component-specific scratch state.
@kwdef mutable struct GradientDescentNumerics
    direction          :: Vector{Float64} = Float64[]
    n_linesearch_evals :: Int             = 0
    grad_prev          :: Vector{Float64} = Float64[]   # required by BB step size
end
```

> **Convention — no field duplication across groups.**
> A numerics struct must never declare a field already present in a shared group.
> In particular, do **not** add a `gradient` or `objective` field to `Numerics`; always read and write these through `state.iterate.gradient` and `state.metrics.objective`.
> If the method needs a *separate* gradient buffer (e.g. for a sub-problem or a previous-step copy), name the field explicitly: `sub_gradient`, `grad_prev`, etc.

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

**LOGGING:**
The state struct carries **no logger reference**.
The runner injects the logger as an explicit parameter to `step!` on every call, keeping algorithm code free of logging infrastructure.

**Sub-routine state reuse.**
When a method uses a nested solver, the outer state struct includes a field typed as the sub-solver's concrete state type.
However, `run_sub_method` **creates and manages a fresh sub-state instance** each time it is
called in `step!` —
the outer state can optionally store the final sub-state for inspection, but it is **not used to initialize or control** the sub-run.

Each state (outer and sub) has its own independent `TimingGroup`.
The sub-solver's accumulated `core_time_ns` is reported in `SubResult.core_time_ns`
and tracked separately from the outer timing. How an outer step *attributes* that
inner time to its own clock is a timing rule — see **Timing a nested solve** in the
Core Timing section below.

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

`extract_log_entry` has a trivial default — it copies `state.metrics` and
`state.timing.core_time_ns` into an `IterationLog` — so a method implements it only to
add entries to the `extras` dict. That default, and the `IterationLog` shape it
targets, are documented in [Logging & Verbosity](logging.md).

## Core Timing — `@core_timed`

Scientific timing measures **only the mathematical kernel** of each step.
The `@core_timed` macro is the single entry point for this.
It accumulates elapsed nanoseconds into `state.timing.core_time_ns` and is **exception-safe**:
if the wrapped expression throws, the elapsed time up to the exception is still recorded before the error is re-thrown.

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

**What counts.** The rule of thumb: **time the mathematics that produces the next
iterate; do not time quantities computed only to populate metrics for logging or
stopping.** A norm is timed when the *update* uses it (a normalized-gradient step
divides by ‖g‖) and left untimed when it is computed solely for a convergence metric.

**Components self-time; `step!` does not wrap them.** A descent direction
(`compute_direction`) and a step size (`compute_step_size`) each wrap their own core
operations in `@core_timed`, so a component with internal bookkeeping — a line
search's trial loop, a quasi-Newton two-loop recursion — counts only its kernel, not
its scaffolding. The caller must therefore **not** wrap these calls (that would
double-count). `grad!` and the iterate update have no such component, so `step!`
times them directly:

```julia
function step!(m::GradientDescent, state, problem, iter, logger, rng)
    @core_timed state begin
        grad!(state.iterate.gradient, problem.f, state.iterate.x)   # gradient kernel — timed by step!
    end

    d = compute_direction(m.direction, state, problem)   # direction rule self-times
    state.numerics.direction = d

    α = compute_step_size(m.step_size, state, problem, d) # step-size rule self-times

    @core_timed state begin
        state.iterate.x .+= α .* d                        # iterate update — timed by step!
    end
    state.metrics.step_norm = norm(α .* d)                # metric only — NOT timed
end
```

**Timing a nested solve.** When a `step!` runs a sub-solver via `run_sub_method`, fold
the sub-solver's reported core time into the outer step directly —
`state.timing.core_time_ns += sub.core_time_ns` — and do **not** wrap the
`run_sub_method` call in `@core_timed`, which would count the sub-run's wall /
scaffolding time instead of its measured kernel. Each state keeps its own
`TimingGroup`, so the inner core time also stays available separately for an
inner/outer breakdown.

Every concrete state struct contains a `TimingGroup` field named `timing`.
The runner resets `state.timing.core_time_ns = 0` before each `step!` call so the logger always sees a single-step measurement.

## The Generic Runner

The runner owns the loop. *"The runner"* (`run_method`) is the per-method iteration
loop that drives **one** method to termination — not to be confused with **a run**,
one seeded repetition of a whole experiment indexed by `run_id` (see
[Experiment Orchestration](orchestration.md)).

Algorithms never hold a logger reference — the logger is passed as an explicit
parameter to `step!` on every iteration. A `StoppingCriterion` object controls
termination (see [Stopping Criteria](@ref)). If `problem.x_opt` is set, the runner
computes `dist_to_opt` — for the initial snapshot and after each step — and stores it
in `state.metrics` before `extract_log_entry`, so algorithms never read
`problem.x_opt`. Debug checks run after logging, before the stopping check. A `step!`
error is **not** caught: it propagates out of `run_method` as a clean failure.

```julia
function run_method(method   :: IterativeMethod,
                    problem,
                    criteria,                      # untyped: stopping.jl loads after core.jl
                    logger   :: Logger,
                    rng      :: AbstractRNG;
                    debug    = nothing)            # untyped: debug.jl loads after core.jl

    state = init_state(method, problem, rng)

    # Runner owns dist_to_opt — algorithms never access problem.x_opt.
    if !isnothing(problem.x_opt)
        state.metrics.dist_to_opt = norm(state.iterate.x .- problem.x_opt)
    end
    log_init!(logger, method, state)

    iter       = 0
    prev_entry = nothing

    while true
        iter += 1
        state.timing.core_time_ns = 0           # reset per-step accumulator

        step!(method, state, problem, iter, logger, rng)   # logger & rng forwarded
        # ↑ an error here propagates out (clean failure, not a silent fallback)

        if !isnothing(problem.x_opt)
            state.metrics.dist_to_opt = norm(state.iterate.x .- problem.x_opt)
        end

        entry = extract_log_entry(method, state, iter)
        log_iter!(logger, entry)

        if debug !== nothing && debug.enabled
            run_debug_checks!(debug, logger, state, problem, entry, prev_entry, iter)
        end
        prev_entry = entry

        # Stopping check happens AFTER logging — never timed.
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
