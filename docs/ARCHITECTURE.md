# Test Engine for Iterative Methods — Architecture

> A Julia-based experimentation framework for benchmarking conventional solvers against
> multiple variants of an under-development iterative method, with composable stopping
> criteria, nested algorithm support, a structured problem factory, comprehensive logging,
> a flexible multi-figure plotting pipeline, optional warm-up initialization, known-optimum
> tracking, and a structured debug mode.

**Audience.** This document is the contributor / maintainer reference. If you only want
to use the framework, start with the user guide and `gradient_descent.md`.

**Terminology.** The framework is organized into a number of cohesive *modules*.
Earlier drafts used the word "layer," but the structure is hub-and-spoke (the
orchestrator is the hub; everything else is a peer module) rather than strictly
stacked. Module numbering below follows dependency order — read top-down without
forward references.

---

## Table of Contents

1. [Design Philosophy](#1-design-philosophy)
2. [High-Level Module Map](#2-high-level-module-map)
3. [Module 1 — Problem Interface](#3-module-1--problem-interface)
4. [Module 2 — Algorithm Abstraction & Core Timing](#4-module-2--algorithm-abstraction--core-timing)
5. [Module 3 — Stopping Criteria](#5-module-3--stopping-criteria)
6. [Module 4 — Variant Grid Engine](#6-module-4--variant-grid-engine)
7. [Module 5 — Nested Algorithm Infrastructure](#7-module-5--nested-algorithm-infrastructure)
8. [Module 6 — Logging & Verbosity](#8-module-6--logging--verbosity)
9. [Module 7 — Experiment Orchestration](#9-module-7--experiment-orchestration)
10. [Module 8 — Persistence & Experiment Naming](#10-module-8--persistence--experiment-naming)
11. [Module 9 — Debug Mode](#11-module-9--debug-mode)
12. [Module 10 — Analysis & Plotting](#12-module-10--analysis--plotting)
13. [Directory & File Structure](#13-directory--file-structure)
14. [Data Flow Diagram](#14-data-flow-diagram)
15. [Key Architectural Decisions](#15-key-architectural-decisions)
16. [Extension Guide](#16-extension-guide)

---

## 1. Design Philosophy

The framework is built on four Julia-native principles:

- **Multiple dispatch over class hierarchies.** Every algorithm, component, stopping
  criterion, and problem is a dispatch point. Adding a new variant never requires
  touching existing code.
- **Separation of concerns across modules.** Algorithms know nothing about logging.
  Loggers know nothing about plotting. Stopping criteria know nothing about algorithms.
  Each module communicates through well-defined data structures.
- **Declarative experiment definition.** An experiment is a plain, serializable data
  structure (`ExperimentConfig`). Running it, saving it, and reloading it are separate,
  independent operations.
- **Scientific measurement discipline.** Timing records only the core mathematical
  computation inside each step, accumulated per iteration and summed across iterations.
  All bookkeeping (logging, stopping criterion checks, verbosity output) is
  deliberately excluded from measured time.
- **Specification-driven implementation.** Every problem and every algorithm is
  accompanied by a `.md` file co-located with its source. Each spec is the
  single source of truth for the mathematical formulation, variable mapping, and
  implementation contracts (`init_state`, `step!`, `extract_log_entry`). Pluggable
  components (descent directions, step-size rules, ...) get their own dedicated spec
  file when they are shared across algorithms.

**Additional invariants enforced by the framework:**

- **Logger purity.** Algorithms never hold a reference to the logger. The logger is
  passed as an explicit parameter to `step!` and `run_sub_method` by the runner.
  Algorithm code is free of logging concerns.
- **Reproducibility.** Every source of randomness — data generation, warm-up,
  initial point `x0`, stochastic algorithmic steps, and nested sub-solver calls — is
  derived from a single `ExperimentConfig.seed` via deterministic, session-stable
  hashing. Sub-solver RNGs are child streams derived from the outer method's RNG.
- **Debug orthogonality.** The debug mode is an optional layer activated at experiment
  level. It adds diagnostic computations after each step (e.g. gradient checks,
  objective monotonicity) without touching algorithm or logging code.

---

## 2. High-Level Module Map

| # | File | Responsibility |
|---|------|----------------|
| 1 | `problems.jl` | `Objective`, `Regularizer`, `Hessian`, `Problem`, `ProblemSpec` hierarchy, problem factory |
| 2 | `core.jl` | Type hierarchy, state groups, algorithm interface, `@core_timed`, run loop, nested infrastructure |
| 3 | `stopping.jl` | Stopping criteria abstraction, composites, `should_stop` dispatch |
| 4 | `variants.jl` | Component abstractions, Cartesian grid expansion, auto-naming |
| 5 | `core.jl` | Nested algorithm infrastructure (`SubRunConfig`, `run_sub_method`) |
| 6 | `logging.jl` | Per-iteration capture, core-time accumulation, event logging, sub-logs, verbosity |
| 7 | `experiment.jl` | Experiment config, result types, warm-up, orchestration, multi-run management |
| 8 | `persistence.jl` | JLD2 binary + CSV sidecar + JSON manifest; date/counter naming |
| 9 | `debug.jl` | `DebugConfig`, `DebugCheck` hierarchy, `run_debug_checks!`, diagnostic helpers |
| 10 | `analysis.jl` | DataFrame pipeline, color registry, flexible multi-figure layout |

Modules 2 and 5 are co-located in `core.jl` to avoid circular includes: both depend on
the same base types and the nested infrastructure (`run_sub_method`) calls `init_state`
and `step!` defined in Module 2.

---

## 3. Module 1 — Problem Interface

Every problem has an **objective** `f(x)`, optionally augmented by one or more
**regularizers** `gᵢ(x)`. The total objective is `f(x) + Σ gᵢ(x)`.
Algorithms interact with the problem exclusively through this interface.

### Objective

```julia
abstract type Objective end

# Required dispatch for every concrete subtype:
#   value(f, x)    → scalar objective value of f at x
#   grad(f, x)     → gradient vector ∇f(x)
#   hessian(f, x)  → a Hessian object representing ∇²f(x); see below
```

`Objective` was previously called `DataFidelity`. The rename reflects that not every
problem has data — the abstraction is about which mathematical operations are
available, not about inverse-problems vocabulary.

### Hessian

The Hessian at a point is itself an object, not a single function. This unifies exact
Hessians, full quasi-Newton approximations (BFGS, SR1), limited-memory variants
(L-BFGS), and structured forms (diagonal, Kronecker, ...) under one dispatch surface.

```julia
abstract type Hessian end

# Required of every concrete Hessian:
apply(H::Hessian, d::Vector{Float64}) :: Vector{Float64}        # H · d

# Optional, defined when feasible:
materialize(H::Hessian) :: Matrix{Float64}                      # full matrix
diagonal(H::Hessian)    :: Vector{Float64}                      # diag(H)
```

Built-in concrete types:

```julia
# Explicit matrix; cheap apply and materialize. Use when the matrix is small or
# already computed.
struct MatrixHessian <: Hessian
    H :: Matrix{Float64}
end
apply(H::MatrixHessian, d)    = H.H * d
materialize(H::MatrixHessian) = H.H

# Closure-based; only Hv products are available. Use for very large problems
# where forming the matrix is infeasible.
struct OperatorHessian <: Hessian
    apply_fn :: Function       # d -> H·d
    n        :: Int
end
apply(H::OperatorHessian, d)  = H.apply_fn(d)
# materialize intentionally NOT defined — calling it raises a clear MethodError.

# Pure-diagonal Hessian (or diagonal preconditioner). Both apply and materialize
# are cheap.
struct DiagonalHessian <: Hessian
    d :: Vector{Float64}
end
apply(H::DiagonalHessian, v)  = H.d .* v
materialize(H::DiagonalHessian) = Diagonal(H.d)
diagonal(H::DiagonalHessian)  = H.d
```

A method that only needs Hessian-vector products calls `apply`; a method that needs
the full matrix calls `materialize`; a method that needs the diagonal calls
`diagonal`. Each concrete `Hessian` subtype declares which of these are available.

> **Forward-compatibility with quasi-Newton methods.** Stateful Hessian
> approximations (BFGS, SR1, L-BFGS) plug into the same hierarchy: each is a
> `Hessian` subtype with an internal `update!(H, s, y)` method called by the
> algorithm after each step. L-BFGS, in particular, never defines `materialize` —
> it stores `(s, y)` history and computes `apply` via the two-loop recursion.
> Methods consuming the Hessian are unaware of the internal mechanism.

### Regularizer

```julia
abstract type Regularizer end

# Required dispatch:
#   value(g, x)    → scalar regularizer value
#   prox(g, x, γ)  → argmin_u { g(u) + 1/(2γ)‖u−x‖² }
```

### Problem

```julia
struct Problem
    f     :: Objective
    gs    :: Vector{Regularizer}              # may be empty; total = f + Σ gᵢ
    x0    :: Vector{Float64}                  # initial point
    n     :: Int                              # problem dimension
    meta  :: Dict{Symbol, Any}                # optional metadata
    x_opt :: Union{Nothing, Vector{Float64}}  # known optimal point; nothing if unavailable
end
```

Convenience constructors keep simple problems simple:

```julia
# Pure (non-composite) problem — no regularizer, no x_opt:
Problem(f, x0)

# Pure problem with a known optimum:
Problem(f, x0; x_opt = ...)

# Composite with one regularizer:
Problem(f, g::Regularizer, x0; x_opt = nothing)

# Composite with multiple regularizers:
Problem(f, gs::Vector{Regularizer}, x0; x_opt = nothing)
```

The total objective and gradient are framework-provided helpers — algorithms use
them rather than reaching into `gs` directly:

```julia
total_objective(p::Problem, x) = value(p.f, x) + sum(value(g, x) for g in p.gs; init=0.0)
```

`x_opt` is set by the problem generator when the true minimizer is known
(famous test problems, synthetic data with planted solution, quadratics with closed
form). Algorithms never access `x_opt` — the runner computes
`state.metrics.dist_to_opt` after each step.

### ProblemSpec — Declarative Construction

`ProblemSpec` is the serializable description used inside an `ExperimentConfig`.
`make_problem(spec, rng)` dispatches on the spec type to construct a concrete
`Problem`.

```julia
abstract type ProblemSpec end
make_problem(spec::ProblemSpec, rng::AbstractRNG) :: Problem
```

#### Analytic problems

```julia
@kwdef struct AnalyticProblem <: ProblemSpec
    name   :: Symbol
    params :: NamedTuple = (;)
    dim    :: Int = 2
end

const ANALYTIC_PROBLEMS = Dict{Symbol, Function}()

register_problem!(:rosenbrock, (params, rng) -> ...)
register_problem!(:quadratic,  (params, rng) -> ...)

make_problem(s::AnalyticProblem, rng::AbstractRNG) =
    ANALYTIC_PROBLEMS[s.name](s.params, rng)
```

#### Data-driven problems

Loaders are referenced by symbol so `FileProblem` stays JLD2-serializable:

```julia
const FILE_LOADERS = Dict{Symbol, Function}()
register_file_loader!(name::Symbol, f::Function) = (FILE_LOADERS[name] = f)

@kwdef struct FileProblem <: ProblemSpec
    path        :: String
    loader_name :: Symbol      # key into FILE_LOADERS
    description :: String = ""
end

make_problem(s::FileProblem, rng::AbstractRNG) = FILE_LOADERS[s.loader_name](s.path)
```

#### Random problems

```julia
@kwdef struct RandomProblem <: ProblemSpec
    name   :: Symbol
    params :: NamedTuple = (;)
end

const RANDOM_GENERATORS = Dict{Symbol, Function}()
register_random_problem!(name::Symbol, gen::Function) =
    (RANDOM_GENERATORS[name] = gen)

make_problem(s::RandomProblem, rng::AbstractRNG) =
    RANDOM_GENERATORS[s.name](rng, s.params)
```

---

## 4. Module 2 — Algorithm Abstraction & Core Timing

### Type Hierarchy

Every algorithm, conventional or experimental, is a concrete subtype of
`IterativeMethod`. The hierarchy separates baselines from the under-development
family, enabling dispatch and experiment-level filtering.

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

### State Parameter Groups — Composable Modules

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

### `extract_log_entry` — Default Implementation

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

### The Three Dispatch Points

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

### Core Timing — `@core_timed`

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
        catch _e
            $(esc(state)).timing.core_time_ns += time_ns() - _t0
            rethrow(_e)
        end
        $(esc(state)).timing.core_time_ns += time_ns() - _t0
    end
end
```

Usage inside `step!` — the algorithm author controls exactly what counts:

```julia
function step!(m::GradientDescent, state, problem, iter, logger, rng)
    @core_timed state begin
        g = grad(problem.f, state.iterate.x)
        state.iterate.gradient = g
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

### The Generic Runner

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
            run_debug_checks!(debug, state, problem, entry, prev_entry, iter)

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

## 5. Module 3 — Stopping Criteria

The runner uses a `while true` loop controlled entirely by `StoppingCriteria`.
This gives full, composable control over how many steps any algorithm takes:
by count, time, proximity to solution, or any user-defined condition.

### Type Hierarchy

```julia
abstract type StoppingCriteria end

# Terminate after a fixed number of steps
@kwdef struct MaxIterations <: StoppingCriteria
    n :: Int = 1000
end

# Terminate when accumulated **core computation** time exceeds the budget.
# Time is measured as the sum of per-iteration core_time_ns values recorded
# in the logger — wall-clock and bookkeeping time are never counted.
@kwdef struct TimeLimit <: StoppingCriteria
    seconds :: Float64 = 60.0
end

# Terminate when gradient norm falls below threshold
@kwdef struct GradientTolerance <: StoppingCriteria
    tol :: Float64 = 1e-6
end

# Terminate when objective change over last `window` iters is below threshold
@kwdef struct ObjectiveStagnation <: StoppingCriteria
    tol    :: Float64 = 1e-8
    window :: Int     = 10
end

# Terminate when step norm falls below threshold
@kwdef struct StepTolerance <: StoppingCriteria
    tol :: Float64 = 1e-8
end

# Terminate when ‖x − x*‖ falls below threshold.
# Only fires when problem.x_opt is set; always returns (false, :none) otherwise.
@kwdef struct DistanceToOptimal <: StoppingCriteria
    tol :: Float64 = 1e-6
end

# Combine multiple criteria: :any (first satisfied wins) or :all (all must hold)
@kwdef struct CompositeCriteria <: StoppingCriteria
    criteria :: Vector{StoppingCriteria}
    mode     :: Symbol = :any
end

# Convenience constructors
stop_when_any(cs...) = CompositeCriteria(criteria=collect(cs), mode=:any)
stop_when_all(cs...) = CompositeCriteria(criteria=collect(cs), mode=:all)
```

### The `should_stop` Interface

```julia
# Returns (stop::Bool, reason::Symbol)
function should_stop(c::StoppingCriteria, state, iter::Int, logger::Logger) end

function should_stop(c::MaxIterations, state, iter, logger)
    iter >= c.n ? (true, :max_iterations) : (false, :none)
end

function should_stop(c::TimeLimit, state, iter, logger)
    elapsed_core_s(logger) >= c.seconds ? (true, :time_limit) : (false, :none)
end

function should_stop(c::GradientTolerance, state, iter, logger)
    state.metrics.gradient_norm <= c.tol ? (true, :gradient_converged) : (false, :none)
end

function should_stop(c::StepTolerance, state, iter, logger)
    state.metrics.step_norm <= c.tol ? (true, :step_converged) : (false, :none)
end

function should_stop(c::ObjectiveStagnation, state, iter, logger)
    iter < c.window && return (false, :none)
    first_obj = logger.iter_logs[end - c.window + 1].objective
    last_obj  = logger.iter_logs[end].objective
    abs(first_obj - last_obj) <= c.tol ? (true, :objective_stagnated) : (false, :none)
end

function should_stop(c::DistanceToOptimal, state, iter, logger)
    # state.metrics.dist_to_opt is Inf when x_opt is not provided → never fires
    state.metrics.dist_to_opt <= c.tol ? (true, :optimal_reached) : (false, :none)
end

function should_stop(c::CompositeCriteria, state, iter, logger)
    results = [should_stop(sub, state, iter, logger) for sub in c.criteria]
    if c.mode == :any
        idx = findfirst(r -> r[1], results)
        isnothing(idx) ? (false, :none) : results[idx]
    else  # :all
        all(r -> r[1], results) ? (true, :all_criteria_met) : (false, :none)
    end
end
```

### Usage at Experiment Definition

```julia
default_stop = stop_when_any(
    MaxIterations(n=2000),
    GradientTolerance(tol=1e-7),
    TimeLimit(seconds=120.0),
)

# When x_opt is known, add distance-to-optimal as an additional trigger
exact_stop = stop_when_any(
    MaxIterations(n=5000),
    DistanceToOptimal(tol=1e-8),
    GradientTolerance(tol=1e-9),
)
```

---

## 6. Module 4 — Variant Grid Engine

This module models each **dimension of variation** as a typed component,
then constructs all valid combinations automatically. Grids work uniformly over
any `IterativeMethod` — conventional or experimental — and the routing into the
right bucket happens later in `resolve_methods`.

### Component Abstractions

Each variation axis is an abstract type with concrete implementations. For the
current simplified setup, the framework ships with two pluggable component
hierarchies, both consumed by `GradientDescent`:

```julia
# Descent direction (see descent_directions.md)
abstract type DescentDirection end
struct SteepestDescent <: DescentDirection end

# Step-size rule (see step_sizes.md)
abstract type StepSize end
abstract type LineSearch <: StepSize end       # subset that performs an actual 1D search

# Direct step-size rules (closed-form α from current state):
@kwdef struct FixedStep       <: StepSize ; α :: Float64 = 1e-3 end
@kwdef struct CauchyStep      <: StepSize ; fallback_α :: Float64 = 1e-3
                                            ε_denom    :: Float64 = 1e-14 end
@kwdef struct BarzilaiBorwein <: StepSize ; variant    :: Symbol  = :BB1
                                            fallback_α :: Float64 = 1e-3
                                            ε_denom    :: Float64 = 1e-14 end

# Genuine line searches (test trial points until a sufficient-decrease condition holds):
@kwdef struct ArmijoLS <: LineSearch
    α₀       :: Float64 = 1.0
    β        :: Float64 = 0.5
    c₁       :: Float64 = 1e-4
    max_iter :: Int     = 50
end
```

Other component hierarchies (Hessian approximations, minor updates, ...) can be added
later. They follow the same pattern: an abstract type, concrete subtypes, a single
dispatched function on the abstract.

### VariantAxis and VariantGrid

```julia
# One axis of variation: a parameter name, a list of values, and a label per value
struct VariantAxis
    param  :: Symbol
    values :: Vector{Any}
    labels :: Vector{String}
end

# Convenience constructor using Pair syntax: value => "label"
function VariantAxis(param::Symbol, labeled_values::Pair...)
    VariantAxis(param,
        [p.first  for p in labeled_values],
        [p.second for p in labeled_values])
end

# The full grid: axes + a builder function + optional exclusion filters
@kwdef struct VariantGrid
    base_name     :: String
    axes          :: Vector{VariantAxis}
    builder       :: Function          # (;param=value, ...) -> IterativeMethod
    filters       :: Vector{Function}  = []  # [(combo::NamedTuple) -> Bool]
    shared_params :: NamedTuple        = (;)
end
```

> **Builder return type is `IterativeMethod`, not `ExperimentalMethod`.** Grids
> can produce conventional methods (e.g. exploring step-size variants of
> `GradientDescent`) just as readily as experimental ones. `resolve_methods` (Module 7)
> sorts each produced method into the conventional or experimental bucket based on
> its concrete type.

### Grid Expansion

`expand(grid)` takes the Cartesian product of all axes, applies filters, builds each
method instance, and attaches auto-generated names. This function operates on a
**single** `VariantGrid` and is independently callable and unit-testable.

```julia
struct VariantSpec
    name       :: String              # full human-readable name
    short_name :: String              # compact legend label
    params     :: NamedTuple          # the exact parameter combination
    method     :: IterativeMethod     # ready-to-run instance
end

function expand(grid::VariantGrid)::Vector{VariantSpec}
    # Cartesian product → filter → build → name
end
```

### Naming Convention

| Format | Example |
|--------|---------|
| Full (logging, filenames) | `GradientDescent[step_size=Armijo]` |
| Short (plot legends) | `GD/Arm` |

```julia
const ABBREVIATIONS = Dict(
    "GradientDescent" => "GD",
    "SteepestDescent" => "SD",
    "FixedStep"       => "Fix",
    "CauchyStep"      => "Cau",
    "BarzilaiBorwein" => "BB",
    "ArmijoLS"        => "Arm",
)

# Register abbreviations for user-defined components
register_abbreviation!(long::String, short::String) = (ABBREVIATIONS[long] = short)
```

`register_abbreviation!` must be called for any user-defined component name before
`expand` is first called. It is documented in the Extension Guide.

### Defining a Grid (Usage Example)

```julia
step_size_axis = VariantAxis(:step_size,
    FixedStep(α=1e-3)              => "Fixed",
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
)
# 5 combinations, all conventional GradientDescent variants.
```

---

## 7. Module 5 — Nested Algorithm Infrastructure

Some algorithms run another iterative method as a **sub-routine** inside their own
`step!`. Examples: trust-region methods that solve an inner subproblem iteratively,
bi-level methods, inner-loop methods that refine a correction, or meta-algorithms
that call multiple sub-solvers per outer step.

This module provides the infrastructure to make nested invocation clean, safe, and
fully logged. **No concrete user of nested algorithms is shipped in the simplified
setup**; the infrastructure is described here because it is part of the framework
and reserved for upcoming experimental methods.

### Design Principle

An algorithm struct holds a **sub-method slot** typed concretely via a parametric
`SubRunConfig{M}`. During `step!`, the algorithm calls `run_sub_method(...)`, passing
the logger and rng it received from the outer runner. `run_sub_method` derives a
child RNG from the outer rng — deterministically — ensuring sub-runs are fully
reproducible. Sub-iteration logs are attached to the current outer iteration's log
entry under `extras`. The sub-runner never writes to disk or console independently.

### Infrastructure Types

```julia
# Parametric over the inner method type — enables type-stable state inference
@kwdef struct SubRunConfig{M <: IterativeMethod}
    method        :: M
    criteria      :: StoppingCriteria
    log_sub_iters :: Bool            = true
    verbosity     :: VerbosityConfig = VerbosityConfig(level=SILENT)
end

# Parametric result: final_state is typed as the concrete sub-state S
struct SubResult{S}
    converged    :: Bool
    stop_reason  :: Symbol
    n_iters      :: Int
    final_state  :: S            # concrete type → type-stable field access in step!
    iter_logs    :: Vector{IterationLog}
    core_time_ns :: Int64        # total core time across all sub-iterations
end
```

### `run_sub_method`

```julia
"""
    run_sub_method(config, problem, outer_logger, outer_rng)

Runs the sub-algorithm described by `config`. A child RNG is derived from
`outer_rng` deterministically, ensuring reproducibility regardless of how many
times this function is called per outer step. Attaches sub-iteration logs to the
outer logger's current pending entry if `config.log_sub_iters` is true.
Returns a `SubResult{S}` where S is the concrete sub-state type.
"""
function run_sub_method(config       :: SubRunConfig{M},
                        problem,
                        outer_logger :: Logger,
                        outer_rng    :: AbstractRNG)::SubResult where M

    # Derive a child RNG — deterministic, independent of outer rng's future draws
    sub_rng    = Xoshiro(rand(outer_rng, UInt64))
    sub_state  = init_state(config.method, problem, sub_rng)
    sub_logger = make_sub_logger(config.verbosity)
    iter       = 0

    while true
        iter += 1
        sub_state.timing.core_time_ns = 0
        step!(config.method, sub_state, problem, iter, sub_logger, sub_rng)

        entry = extract_log_entry(config.method, sub_state, iter)
        log_iter!(sub_logger, entry)

        stop, reason = should_stop(config.criteria, sub_state, iter, sub_logger)
        if stop
            if config.log_sub_iters
                attach_sub_logs!(outer_logger, sub_logger.iter_logs)
            end
            total_core = sum(e.core_time_ns for e in sub_logger.iter_logs; init=0)
            return SubResult(is_converged_reason(reason), reason, iter,
                             sub_state, sub_logger.iter_logs, total_core)
        end
    end
end
```

### Schematic Use in an Outer Algorithm

```julia
# Sketch — the outer algorithm holds a sub-config, calls run_sub_method inside step!,
# and reads the concrete final_state to extract whatever it needs.

@kwdef struct SomeOuterMethod <: ExperimentalMethod
    inner_sub :: SubRunConfig{<:IterativeMethod} = ...
    # ... outer hyperparameters ...
end

function step!(m::SomeOuterMethod, state, problem, iter, logger, rng)
    @core_timed state begin
        sub_problem = build_subproblem(state, problem)
    end
    sub_result = run_sub_method(m.inner_sub, sub_problem, logger, rng)
    # sub_result.final_state has the concrete sub-state type — type-stable.

    @core_timed state begin
        Δx = extract_direction(sub_result.final_state)
        state.iterate.x .-= Δx
    end
end
```

The outer `IterationLog.extras` for that iteration will contain:

```julia
extras = Dict(
    :sub_iters        => sub_result.n_iters,
    :sub_converged    => sub_result.converged,
    :sub_core_time_ns => sub_result.core_time_ns,
    :sub_logs         => sub_result.iter_logs,   # full inner history if log_sub_iters=true
)
```

---

## 8. Module 6 — Logging & Verbosity

The logger is external to all algorithms. It is injected by the runner and captures
data through three hooks: `log_init!`, `log_iter!`, and `log_event!`. Verbosity is
co-located with logging because both share the same `Logger` struct.

### IterationLog

```julia
@kwdef mutable struct IterationLog
    iter           :: Int
    core_time_ns   :: Int64            # nanoseconds of core computation this step
    objective      :: Float64
    gradient_norm  :: Float64
    step_norm      :: Float64
    dist_to_opt    :: Float64 = Inf    # ‖x − x*‖; Inf when x_opt not provided
    extras         :: Dict{Symbol,Any} = Dict()  # algorithm-specific & sub-logs
end
```

`dist_to_opt` is `Inf` by default. It is updated by the runner (never by the
algorithm) when `problem.x_opt` is non-`nothing`. Analysis code can test
`isfinite(entry.dist_to_opt)` to determine whether optimality tracking was active.

The `extras` dict carries algorithm-specific fields and, when nested algorithms are
used, `:sub_logs` containing the full `Vector{IterationLog}` from each sub-method run.

### Logger

```julia
mutable struct Logger
    method_name      :: String
    run_id           :: Int
    exp_path         :: String
    verbosity_config :: VerbosityConfig
    iter_logs        :: Vector{IterationLog}
    events           :: Vector{NamedTuple}     # :converged, :stopped, :warning
    metadata         :: Dict{Symbol,Any}
    start_wall_time  :: Float64                # wall clock at log_init! — informational
    total_core_ns    :: Int64                  # accumulated core nanoseconds across iters
    pending_sub_logs :: Vector{IterationLog}   # buffer for attach_sub_logs!
end

# Core computation elapsed — the authoritative timing used by TimeLimit
elapsed_core_s(logger::Logger) = logger.total_core_ns / 1e9

# Wall-clock elapsed — informational only, never a stopping criterion
elapsed_wall_s(logger::Logger) = time() - logger.start_wall_time
```

### `log_iter!`

`log_iter!` is the single point where `entry.core_time_ns` is accumulated:

```julia
function log_iter!(logger::Logger, entry::IterationLog)
    push!(logger.iter_logs, entry)
    logger.total_core_ns += entry.core_time_ns   # feeds elapsed_core_s → TimeLimit
    maybe_print(logger, entry)
end
```

`extract_log_entry(method, state, iter)` dispatches on the method type, allowing each
algorithm to populate `extras` while sharing the common log schema.

### Verbosity Levels

Verbosity is a first-class, orthogonal concern — not scattered `if verbose` checks.
It is **independent of debug mode** (Module 9): verbosity controls what is printed
from normal iteration data; debug mode controls diagnostic calculations triggered by
threshold violations.

```julia
@enum VerbosityLevel begin
    SILENT    = 0    # no output
    MILESTONE = 1    # start, end, and stop events only
    SUMMARY   = 2    # every N iterations (configurable)
    DETAILED  = 3    # every iteration, compact single line
    VERBOSE   = 4    # every iteration with full extras dict
end
```

### VerbosityConfig

```julia
@kwdef mutable struct VerbosityConfig
    level       :: VerbosityLevel               = SUMMARY
    print_every :: Int                          = 10
    fields      :: Vector{Symbol}               = [:iter, :objective, :gradient_norm]
    color       :: Bool                         = true
    io          :: IO                           = stdout
    iter_range  :: Union{Nothing,UnitRange{Int}} = nothing
    # iter_range = 100:200 → DETAILED output only for iterations 100–200
    # iter_range = nothing → apply level uniformly
end
```

### Range-Gated Output

`maybe_print(logger, entry)` is the single gating function. It evaluates:

1. Is `entry.iter` inside `iter_range` (if set)?
   — If yes: apply `DETAILED` regardless of the configured `level`.
   — If no and `iter_range` is set: suppress output for that iteration.
   — If `iter_range` is `nothing`: apply `level` uniformly.

2. Within the applicable level, apply the `print_every` stride.

```julia
function maybe_print(logger::Logger, entry::IterationLog)
    cfg = logger.verbosity_config
    effective_level = if !isnothing(cfg.iter_range) && entry.iter in cfg.iter_range
        DETAILED
    elseif !isnothing(cfg.iter_range)
        SILENT
    else
        cfg.level
    end

    effective_level == SILENT && return
    effective_level == MILESTONE && return   # handled by log_event! separately

    if effective_level >= SUMMARY
        entry.iter % cfg.print_every == 0 || effective_level >= DETAILED || return
        format_and_print(cfg, entry, effective_level)
    end
end
```

Usage example — print every iteration between 100 and 200 only:

```julia
verbosity = VerbosityConfig(
    level      = MILESTONE,
    iter_range = 100:200,
    fields     = [:iter, :objective, :gradient_norm, :dist_to_opt, :core_time_ns],
)
```

---

## 9. Module 7 — Experiment Orchestration

### Experiment Naming

Each experiment is stored under a two-level path: a **date folder** and a
**zero-padded sequential counter** that resets each day.

```
logs/
└── 20260417/          ← date folder (YYYYMMDD)
    ├── 001/           ← first experiment of this day
    │   ├── manifest.json
    │   ├── result.jld2
    │   └── ...
    ├── 002/
    └── 003/
```

The counter is determined at save time by atomically creating the directory — the
`mkdir` call fails if the path already exists, avoiding the TOCTOU race condition
inherent in a scan-then-create approach:

```julia
function next_experiment_path(log_root::String)::String
    date_str = Dates.format(today(), "yyyymmdd")
    day_dir  = joinpath(log_root, date_str)
    mkpath(day_dir)
    existing = filter(isdir, readdir(day_dir; join=true))
    nums     = [parse(Int, basename(d)) for d in existing
                if occursin(r"^\d{3,}$", basename(d))]
    next_num = isempty(nums) ? 1 : maximum(nums) + 1
    while true
        path = joinpath(day_dir, lpad(next_num, 3, '0'))
        try
            mkdir(path)   # atomic on POSIX — throws if already exists
            return path
        catch e
            e isa SystemError && e.errnum == Base.UV_EEXIST || rethrow(e)
            next_num += 1
        end
    end
end
```

The human-readable `name` field from `ExperimentConfig` is stored inside
`manifest.json`, not in the path.

### Warm-up Infrastructure

A warm-up is an optional, **shared** pre-run initialization step. It executes once
per run before any method starts, and its output — a new initial point `x0_warm` —
replaces `problem.x0` for all methods in that run. Methods cannot distinguish between
a warm-up start and a cold start; the problem interface is identical.

```julia
abstract type WarmupStrategy end

# No warm-up — use problem.x0 as-is (default)
struct NoWarmup <: WarmupStrategy end

# Run an iterative method as warm-up; use its final iterate as x0
@kwdef struct IterativeWarmup <: WarmupStrategy
    method    :: IterativeMethod
    criteria  :: StoppingCriteria
    verbosity :: VerbosityConfig = VerbosityConfig(level=MILESTONE)
end

# Apply a registered pure function to produce x0
struct FunctionWarmup <: WarmupStrategy
    name :: Symbol    # key into WARMUP_FUNCTIONS registry — serialization-safe
end

const WARMUP_FUNCTIONS = Dict{Symbol, Function}()
register_warmup!(name::Symbol, gen::Function) = (WARMUP_FUNCTIONS[name] = gen)
```

`run_warmup` dispatches on the strategy and returns the new `x0`:

```julia
function run_warmup(w::NoWarmup, problem, rng, debug)::Vector{Float64}
    return problem.x0
end

function run_warmup(w::IterativeWarmup, problem, rng, debug)::Vector{Float64}
    warmup_logger = Logger("__warmup__", 0, "", w.verbosity)
    result        = run_method(w.method, problem, w.criteria, warmup_logger, rng, debug)
    return copy(result.final_state.iterate.x)   # universal iterate.x convention
end

function run_warmup(w::FunctionWarmup, problem, rng, debug)::Vector{Float64}
    return WARMUP_FUNCTIONS[w.name](problem, rng)
end
```

### ExperimentConfig

```julia
@kwdef struct ExperimentConfig
    name                 :: String
    problem_spec         :: ProblemSpec
    conventional_methods :: Vector{ConventionalMethod} = []
    experimental_methods :: Vector{ExperimentalMethod} = []
    variant_grids        :: Vector{VariantGrid}        = []
    stopping_criteria    :: StoppingCriteria           = stop_when_any(
                                MaxIterations(1000), GradientTolerance(1e-6))
    method_criteria      :: Dict{String, StoppingCriteria} = Dict()
    warmup               :: WarmupStrategy             = NoWarmup()
    n_runs               :: Int                        = 1
    seed                 :: Union{Int,Nothing}         = 42
    tags                 :: Dict{String,Any}           = Dict()
    debug                :: DebugConfig                = DebugConfig()
end
```

`method_criteria` lets specific methods use different stopping budgets within the
same experiment.

### Result Types

```julia
# Outcome of running one method on one problem instance (one run_id).
# Parametric over the concrete state type S for type-stable field access.
struct MethodResult{S}
    method_name  :: String
    iter_logs    :: Vector{IterationLog}
    final_state  :: S          # concrete state type — no Any in the hot path
    stop_reason  :: Symbol
    n_iters      :: Int
end

# All method outcomes for a single run
struct RunResult
    run_id         :: Int
    method_results :: Dict{String, Any}   # values are MethodResult{S} for various S
end

# Full experiment outcome, wrapping all runs
struct ExperimentResult
    config          :: ExperimentConfig
    experiment_path :: String
    timestamp       :: DateTime
    host            :: String
    run_results     :: Vector{RunResult}
end
```

`finalize!(logger, method, state)` returns a `MethodResult{typeof(state)}`, preserving
the concrete state type through the parametric wrapper.

### Result Hierarchy (Overview)

```
ExperimentResult
    ├── config          :: ExperimentConfig
    ├── experiment_path :: String
    ├── timestamp       :: DateTime
    ├── host            :: String
    └── run_results[]   :: RunResult
            ├── run_id
            └── method_results :: Dict{String, Any}   # MethodResult{S} values
                    ├── method_name  :: String
                    ├── iter_logs    :: Vector{IterationLog}
                    ├── final_state  :: S
                    ├── stop_reason  :: Symbol
                    └── n_iters      :: Int
```

### Orchestration Loop

RNG streams are separated by concern — data generation, warm-up, and per-method
computation each draw from independent, deterministically-derived Xoshiro streams.
Per-method RNGs are additionally keyed by method name so that adding or removing a
method from the config does not alter other methods' streams.

```julia
function run_experiment(config    :: ExperimentConfig,
                        log_root  :: String = "logs";
                        verbosity :: VerbosityConfig = VerbosityConfig())

    exp_path = next_experiment_path(log_root)
    mkpath(exp_path)

    conventional, experimental = resolve_methods(config)
    results = RunResult[]

    for run_id in 1:config.n_runs
        seed = something(config.seed, rand(UInt64))

        rng_data   = Xoshiro(hash((seed, run_id, :data)))
        rng_warmup = Xoshiro(hash((seed, run_id, :warmup)))

        problem = make_problem(config.problem_spec, rng_data)

        if !isa(config.warmup, NoWarmup)
            x0_warm = run_warmup(config.warmup, problem, rng_warmup, config.debug)
            problem  = Problem(problem.f, problem.gs, x0_warm,
                               problem.n, problem.meta, problem.x_opt)
        end

        method_results = Dict{String, Any}()
        for (name, method) in [conventional; experimental]
            method_rng = Xoshiro(hash((seed, run_id, name)))
            criteria   = get(config.method_criteria, name, config.stopping_criteria)
            logger     = Logger(name, run_id, exp_path, verbosity)
            result     = run_method(method, problem, criteria, logger,
                                    method_rng, config.debug)
            method_results[name] = result
        end
        push!(results, RunResult(run_id, method_results))
    end

    exp_result = ExperimentResult(config, exp_path, now(), gethostname(), results)
    save_experiment(exp_result)
    return exp_result
end
```

### `resolve_methods`

`resolve_methods(config)` flattens the three method sources — `conventional_methods`,
`experimental_methods`, and the `expand`-ed output of every entry in `variant_grids` —
and routes each produced method into the conventional or experimental bucket based on
its concrete type.

```julia
function resolve_methods(config::ExperimentConfig)
    conventional = Tuple{String, ConventionalMethod}[]
    experimental = Tuple{String, ExperimentalMethod}[]

    for m in config.conventional_methods
        push!(conventional, (string(typeof(m)), m))
    end
    for m in config.experimental_methods
        push!(experimental, (string(typeof(m)), m))
    end
    for grid in config.variant_grids
        for spec in expand(grid)
            if spec.method isa ConventionalMethod
                push!(conventional, (spec.name, spec.method))
            else
                push!(experimental, (spec.name, spec.method))
            end
        end
    end

    return conventional, experimental
end
```

This is the single point that uses the `ConventionalMethod` / `ExperimentalMethod`
distinction: grids and other config sources never need to know which bucket their
output belongs in.

---

## 10. Module 8 — Persistence & Experiment Naming

Two formats are always written together for every experiment:

| Format | Purpose |
|--------|---------|
| `result.jld2` | Full binary — preserves all Julia types, fast reload |
| `run{N}_{MethodName}.csv` | Per-method per-run CSV — human-readable, grep-able |
| `manifest.json` | Experiment metadata and human name; no binary load needed |

### File Layout on Disk

```
logs/
└── 20260417/
    ├── 001/
    │   ├── manifest.json            ← {name, timestamp, host, methods, n_runs, tags}
    │   ├── result.jld2
    │   ├── run1_GradientDescent[step_size=Armijo].csv
    │   ├── run1_GradientDescent[step_size=BB1].csv
    │   └── run2_GradientDescent[step_size=Armijo].csv
    └── 002/
        ├── manifest.json
        └── ...
```

### API

```julia
# Save — called automatically at the end of run_experiment
save_experiment(result::ExperimentResult)

# Reload for analysis — one call restores everything
result = load_experiment("logs/20260417/001/")

# Quick metadata access without loading the binary
manifest = load_manifest("logs/20260417/001/manifest.json")

# List all experiments across all days
list_experiments(log_root="logs") :: Vector{NamedTuple}
# Returns [{path, date, number, name, timestamp, n_methods, n_runs}, ...]
```

---

## 11. Module 9 — Debug Mode

The debug mode is an optional, experiment-level diagnostic layer. When activated,
the runner performs additional computations after each step — computations that may
be expensive (such as numerical gradient checks) and are never run in normal
operation. When a check's condition triggers, a configurable action is taken:
a warning is printed, the error is raised, or the event is recorded silently.

Debug mode is **orthogonal to verbosity** (Module 6): verbosity controls what is
printed from normal iteration data; debug mode controls diagnostic calculations
triggered by threshold violations. Both can be active simultaneously at independent
levels.

### DebugConfig

```julia
@kwdef struct DebugConfig
    enabled    :: Bool               = false
    checks     :: Vector{DebugCheck} = DebugCheck[
                      CheckObjectiveMonotonicity(),
                      CheckGradientNormBound(),
                  ]
    on_trigger :: Symbol             = :warn   # :warn | :error | :log
    io         :: IO                 = stderr
end
```

`on_trigger` controls what happens when any check fires:
- `:warn` — print a formatted warning to `cfg.io` and continue.
- `:error` — print the warning and throw an `ErrorException` (stops the run).
- `:log` — record silently in `logger.events` without printing.

### DebugCheck Hierarchy

```julia
abstract type DebugCheck end

# Warn if the objective increases by more than `tolerance` between consecutive iters.
@kwdef struct CheckObjectiveMonotonicity <: DebugCheck
    tolerance :: Float64 = 1e-10
end

# Warn if the gradient norm exceeds `max_norm` — detects divergence early.
@kwdef struct CheckGradientNormBound <: DebugCheck
    max_norm :: Float64 = 1e8
end

# Warn if ‖x_k+1 − x_k‖ has not decreased over the last `window` iterations.
@kwdef struct CheckStepDecay <: DebugCheck
    window :: Int = 20
end

# Expensive: compute a numerical gradient and compare with state.iterate.gradient.
@kwdef struct CheckNumericalGradient <: DebugCheck
    epsilon   :: Float64 = 1e-7
    max_error :: Float64 = 1e-4
end
```

### `run_debug_checks!` and `debug_check!` Dispatch

The runner calls `run_debug_checks!` after `log_iter!` on every iteration:

```julia
function run_debug_checks!(cfg        :: DebugConfig,
                           state, problem,
                           entry      :: IterationLog,
                           prev_entry :: Union{Nothing, IterationLog},
                           iter       :: Int)
    cfg.enabled || return
    for check in cfg.checks
        debug_check!(check, cfg, state, problem, entry, prev_entry, iter)
    end
end
```

Each `debug_check!` method implements one check:

```julia
function debug_check!(c::CheckObjectiveMonotonicity, cfg, state, problem,
                      entry, prev, iter)
    isnothing(prev) && return
    increase = entry.objective - prev.objective
    if increase > c.tolerance
        trigger_debug!(cfg, iter,
            "Objective increased by $(increase) " *
            "(prev=$(prev.objective), curr=$(entry.objective))")
    end
end

function debug_check!(c::CheckGradientNormBound, cfg, state, problem,
                      entry, prev, iter)
    if entry.gradient_norm > c.max_norm
        trigger_debug!(cfg, iter,
            "Gradient norm $(entry.gradient_norm) exceeds bound $(c.max_norm)")
    end
end

function debug_check!(c::CheckNumericalGradient, cfg, state, problem,
                      entry, prev, iter)
    g_analytical = state.iterate.gradient
    g_numerical  = numerical_gradient(problem.f, state.iterate.x, c.epsilon)
    denom        = max(norm(g_analytical), 1.0)
    rel_error    = norm(g_analytical .- g_numerical) / denom
    if rel_error > c.max_error
        trigger_debug!(cfg, iter,
            "Gradient check failed: relative error = $(rel_error) " *
            "(threshold=$(c.max_error))")
    end
end
```

### `trigger_debug!` and Diagnostic Helpers

```julia
function trigger_debug!(cfg::DebugConfig, iter::Int, msg::String)
    full_msg = "[DEBUG iter=$iter] $msg"
    if cfg.on_trigger in (:warn, :error)
        println(cfg.io, full_msg)
    end
    cfg.on_trigger == :error && error(full_msg)
end

# Central-difference numerical gradient — used by CheckNumericalGradient
function numerical_gradient(f::Objective, x::Vector{Float64},
                            ε::Float64)::Vector{Float64}
    n  = length(x)
    g  = zeros(n)
    xp = copy(x)
    xm = copy(x)
    for i in 1:n
        xp[i] += ε;  xm[i] -= ε
        g[i]   = (value(f, xp) - value(f, xm)) / (2ε)
        xp[i]  = x[i];  xm[i] = x[i]
    end
    return g
end
```

### Integration Example

```julia
config = ExperimentConfig(
    name         = "Debug run — gradient check active",
    problem_spec = AnalyticProblem(name=:rosenbrock, params=(rho=100.0,)),
    conventional_methods = [GradientDescent(step_size=ArmijoLS())],
    n_runs = 1,
    seed   = 42,
    debug  = DebugConfig(
        enabled    = true,
        checks     = [
            CheckObjectiveMonotonicity(tolerance=0.0),
            CheckNumericalGradient(epsilon=1e-6, max_error=1e-5),
        ],
        on_trigger = :warn,
        io         = stderr,
    ),
)
```

---

## 12. Module 10 — Analysis & Plotting

The analysis module has two roles:

1. **DataFrame pipeline** — load a saved experiment, then answer any question by
   filtering, aggregating, and transforming the data.
2. **Figure layout system** — compose any number of plots in any formation and
   render them to a single PDF or image file.

There is no grid-aware analysis layer. Because variant names embed axis information
(e.g. `GradientDescent[step_size=Armijo]`), the user can always parse or filter on
names as plain strings if needed.

### Loading and Transforming

```julia
result = load_experiment("logs/20260417/001/")

# Convert all iteration logs to a flat DataFrame
# Columns: :run_id, :method_name, :iter, :objective, :gradient_norm,
#          :step_norm, :dist_to_opt, :core_time_ns, + any extras keys
df = to_dataframe(result)

df = filter_methods(df, ["GradientDescent[step_size=Armijo]",
                          "GradientDescent[step_size=BB1]"])
df = aggregate_runs(df, :median)    # :all | :mean | :median
```

User transforms are plain `DataFrame -> DataFrame` functions:

```julia
transforms = [
    df -> @transform(df, :log_obj      = log10.(:objective)),
    df -> @transform(df, :log_dist     = log10.(max.(:dist_to_opt, 1e-16))),
    df -> @subset(df,   :iter .< 500),
    df -> @transform(df, :core_time_ms = :core_time_ns ./ 1e6),
]
for t in transforms; df = t(df); end
```

### MethodStyle — Per-Method Visual Properties

```julia
@kwdef struct MethodStyle
    color     :: Any
    linestyle :: Symbol    = :solid
    linewidth :: Float64   = 2.0
    marker    :: Union{Nothing, Symbol} = nothing
    label     :: Union{Nothing, String} = nothing
end
```

### Method Color Registry

Colors are **deterministic and visually appealing** by default. A fixed curated
palette (Wong colorblind-safe + Tableau extensions) is assigned to method names via a
stable hash — the same method name always maps to the same color regardless of
experiment or run order.

```julia
const METHOD_PALETTE = [
    "#0072B2", "#E69F00", "#009E73", "#CC79A7",
    "#56B4E9", "#D55E00", "#F0E442", "#000000",
]

const METHOD_COLOR_REGISTRY = Dict{String, String}()

function method_color(name::String)::String
    METHOD_PALETTE[(hash(name) % length(METHOD_PALETTE)) + 1]
end

register_method_color!(name::String, color::String) =
    (METHOD_COLOR_REGISTRY[name] = color)

get_method_color(name::String)::String =
    get(METHOD_COLOR_REGISTRY, name, method_color(name))
```

### PlotSpec — Describing a Single Plot

```julia
@kwdef struct PlotSpec
    data          :: DataFrame
    x             :: Symbol          = :iter
    y             :: Symbol          = :objective
    group_by      :: Symbol          = :method_name
    title         :: String          = ""
    xlabel        :: String          = ""
    ylabel        :: String          = ""
    yscale        :: Symbol          = :linear
    xscale        :: Symbol          = :linear
    xlim          :: Union{Nothing,Tuple} = nothing
    ylim          :: Union{Nothing,Tuple} = nothing
    legend        :: Bool            = true
    method_styles :: Dict{String, MethodStyle} = Dict()
    extra_kwargs  :: Dict            = Dict()
end
```

### FigureLayout — Composing Multiple Plots

```julia
@kwdef struct FigureLayout
    plots       :: Matrix{Union{PlotSpec,Nothing}}
    figure_size :: Tuple{Int,Int} = (1200, 900)
    title       :: String = ""
    padding     :: Int = 20
end

function render_figure(layout::FigureLayout)::Makie.Figure
    fig = Figure(resolution=layout.figure_size)
    for row in 1:size(layout.plots, 1), col in 1:size(layout.plots, 2)
        spec = layout.plots[row, col]
        isnothing(spec) && continue
        ax = Axis(fig[row, col],
            title  = spec.title,
            xlabel = isempty(spec.xlabel) ? string(spec.x) : spec.xlabel,
            ylabel = isempty(spec.ylabel) ? string(spec.y) : spec.ylabel,
            yscale = spec.yscale == :log10 ? log10 : identity,
            xscale = spec.xscale == :log10 ? log10 : identity,
        )
        _render_lines!(ax, spec)
        !isnothing(spec.xlim) && xlims!(ax, spec.xlim...)
        !isnothing(spec.ylim) && ylims!(ax, spec.ylim...)
    end
    isempty(layout.title) || Label(fig[0, :], layout.title, fontsize=18)
    return fig
end

function save_figure(fig::Makie.Figure, path::String)
    save(path, fig)
end
```

### End-to-End Plotting Example

```julia
result  = load_experiment("logs/20260417/001/")
df_all  = to_dataframe(result) |> df -> aggregate_runs(df, :median)

styles = Dict(
    "GradientDescent[step_size=Armijo]" => MethodStyle(color="#0072B2", linewidth=2.5),
    "GradientDescent[step_size=BB1]"    => MethodStyle(color="#E69F00", linewidth=2.5),
    "GradientDescent[step_size=Cauchy]" => MethodStyle(color="#009E73", linewidth=2.5),
)

layout = FigureLayout(
    figure_size = (1600, 900),
    title       = "Experiment 001 — Rosenbrock, GD step-size comparison",
    plots = [
        PlotSpec(data=df_all, x=:iter,        y=:objective,     yscale=:log10,
                 title="Objective vs iter", method_styles=styles)   PlotSpec(data=df_all, x=:iter, y=:dist_to_opt, yscale=:log10,
                 title="‖x − x*‖ vs iter",   method_styles=styles);
        PlotSpec(data=df_all, x=:core_time_ns, y=:objective, yscale=:log10,
                 xlabel="Cumulative core time (ns)",
                 title="Objective vs core time")                    nothing
    ],
)

fig = render_figure(layout)
save_figure(fig, "logs/20260417/001/convergence_overview.pdf")
```

### Plotting Across Multiple Experiments

```julia
df1 = to_dataframe(load_experiment("logs/20260417/001/")) |> d -> @transform(d, :exp = "exp1")
df2 = to_dataframe(load_experiment("logs/20260417/002/")) |> d -> @transform(d, :exp = "exp2")
df  = vcat(df1, df2)
# Then build PlotSpec(data=df, group_by=:exp, ...) as normal
```

---

## 13. Directory & File Structure

The source tree is consolidated into **9 files**. Each file groups tightly related
concerns; none is so large as to become unwieldy.

```
TestEngine.jl/
├── src/
│   ├── TestEngine.jl     # Module entry; includes all src files; exports public API
│   │
│   ├── problems.jl       # Objective abstract type & built-ins; Hessian abstraction
│   │                     #   (MatrixHessian, OperatorHessian, DiagonalHessian);
│   │                     #   Regularizer; Problem (with optional gs and x_opt) +
│   │                     #   convenience constructors; total_objective; ProblemSpec
│   │                     #   hierarchy (AnalyticProblem, FileProblem with FILE_LOADERS,
│   │                     #   RandomProblem); make_problem dispatch; register_problem!,
│   │                     #   register_file_loader!, register_random_problem!
│   │
│   ├── core.jl           # Abstract types & type hierarchy; state groups
│   │                     #   (IterateGroup, MetricsGroup, TimingGroup); algorithm
│   │                     #   interface (init_state, step!, extract_log_entry);
│   │                     #   @core_timed macro; generic runner (run_method);
│   │                     #   nested infrastructure (SubRunConfig{M}, SubResult{S},
│   │                     #   run_sub_method)
│   │
│   ├── stopping.jl       # StoppingCriteria hierarchy; should_stop dispatch;
│   │                     #   CompositeCriteria; stop_when_any / stop_when_all;
│   │                     #   DistanceToOptimal
│   │
│   ├── variants.jl       # Pluggable component abstractions (DescentDirection,
│   │                     #   StepSize / LineSearch); concrete subtypes; VariantAxis,
│   │                     #   VariantGrid, VariantSpec; expand(); ABBREVIATIONS;
│   │                     #   register_abbreviation!; build_names()
│   │
│   ├── logging.jl        # IterationLog (incl. dist_to_opt); Logger; log_init!,
│   │                     #   log_iter!, log_event!, attach_sub_logs!, finalize!;
│   │                     #   elapsed_core_s, elapsed_wall_s; VerbosityLevel,
│   │                     #   VerbosityConfig, maybe_print()
│   │
│   ├── experiment.jl     # ExperimentConfig; ExperimentResult / RunResult /
│   │                     #   MethodResult{S}; WarmupStrategy (NoWarmup,
│   │                     #   IterativeWarmup, FunctionWarmup); run_warmup();
│   │                     #   resolve_methods(); run_experiment();
│   │                     #   next_experiment_path()
│   │
│   ├── persistence.jl    # save_experiment(); load_experiment(); load_manifest();
│   │                     #   list_experiments(); CSV sidecar writer
│   │
│   ├── debug.jl          # DebugConfig; DebugCheck hierarchy
│   │                     #   (CheckObjectiveMonotonicity, CheckGradientNormBound,
│   │                     #   CheckStepDecay, CheckNumericalGradient);
│   │                     #   run_debug_checks!; debug_check! dispatch;
│   │                     #   trigger_debug!; numerical_gradient()
│   │
│   └── analysis.jl       # to_dataframe(); filter_methods(); aggregate_runs();
│                         #   MethodStyle; METHOD_PALETTE; METHOD_COLOR_REGISTRY;
│                         #   get_method_color(); register_method_color!;
│                         #   PlotSpec; FigureLayout; render_figure(); save_figure()
│
├── problems/
│   └── rosenbrock/
│       ├── rosenbrock.md          # spec — see rosenbrock.md
│       └── rosenbrock.jl
│
├── algorithms/
│   ├── conventional/
│   │   └── gradient_descent/
│   │       ├── gradient_descent.md       # spec — see gradient_descent.md
│   │       ├── gradient_descent.jl
│   │       └── components/
│   │           ├── descent_directions.md  # see descent_directions.md
│   │           ├── descent_directions.jl
│   │           ├── step_sizes.md          # see step_sizes.md (covers StepSize/LineSearch)
│   │           └── step_sizes.jl
│   └── experimental/
│       └── (added later)
│
├── experiments/
│   └── exp_baseline.jl           # GradientDescent step-size sweep on Rosenbrock
│
├── logs/                         # Git-ignored; written at runtime
│   └── 20260417/
│       ├── 001/
│       │   ├── manifest.json
│       │   ├── result.jld2
│       │   └── run1_GradientDescent[step_size=Armijo].csv
│       └── 002/
│
└── test/
    ├── test_problems.jl          # Objective, Hessian, Problem interface, x_opt,
    │                             #   make_problem, seed propagation
    ├── test_core.jl              # runner, state groups, @core_timed, exception safety
    ├── test_stopping.jl          # all StoppingCriteria subtypes, DistanceToOptimal,
    │                             #   TimeLimit
    ├── test_variants.jl          # expand(), naming, filters, abbreviations
    ├── test_warmup.jl            # NoWarmup, IterativeWarmup, FunctionWarmup;
    │                             #   x0 propagation
    ├── test_debug.jl             # all DebugCheck subtypes, trigger modes,
    │                             #   numerical_gradient
    ├── test_analysis.jl          # to_dataframe (incl. dist_to_opt col),
    │                             #   aggregate_runs, PlotSpec
    └── test_integration.jl       # full run_experiment on Rosenbrock; serialize+reload
```

---

## 14. Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│  DEFINITION PHASE                                                       │
│                                                                         │
│  problems.jl        Objective / Hessian / Regularizer / Problem        │
│       │             AnalyticProblem / FileProblem / RandomProblem       │
│       │                                                                 │
│  variants.jl        VariantAxis(:step_size, FixedStep=>"Fix", ...)     │
│       │                   │                                             │
│       └──────────► VariantGrid → expand() → [VariantSpec, ...]         │
│                                     │                                   │
│  stopping.jl        StoppingCriteria (per-experiment or per-method)    │
│                                     │                                   │
│                         ExperimentConfig                                │
│                    (+ warmup :: WarmupStrategy)                         │
│                    (+ debug  :: DebugConfig)                            │
└─────────────────────────────────────┬───────────────────────────────────┘
                                      │
┌─────────────────────────────────────▼───────────────────────────────────┐
│  EXECUTION PHASE                                                        │
│                                                                         │
│         run_experiment(config, log_root)                                │
│                  │                                                      │
│         next_experiment_path()  →  logs/YYYYMMDD/NNN/ (atomic mkdir)   │
│         resolve_methods()       →  routes by ConventionalMethod /      │
│                                     ExperimentalMethod                  │
│                  │                                                      │
│   ┌──── WARM-UP (once per run, if configured) ──────────────────────┐   │
│   │  rng_warmup = Xoshiro(hash((seed, run_id, :warmup)))            │   │
│   │  x0_warm   = run_warmup(config.warmup, problem, rng_warmup, …)  │   │
│   │  problem   = Problem(…, x0=x0_warm, …)   ← shared by all       │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                  │                                                      │
│        for each run × method:                                           │
│         method_rng = Xoshiro(hash((seed, run_id, method_name)))         │
│                  │                                                      │
│      ┌───────────▼────────────┐                                         │
│      │    run_method()        │  ◄── Logger + Criteria + DebugConfig    │
│      │  init_state(…, rng)    │                                         │
│      │  while true:           │                                         │
│      │    core_time_ns = 0    │                                         │
│      │    step!(…, logger,    │  ◄── logger & rng explicit params       │
│      │          rng)          │  ◄── @core_timed inside step!           │
│      │    dist_to_opt         │  ──► runner computes from x_opt         │
│      │    extract_log_entry() │  ──► entry.dist_to_opt copied           │
│      │    log_iter!()         │  ──► logger.total_core_ns accumulated   │
│      │    run_debug_checks!() │  ──► checks fire → warn/error/log       │
│      │    should_stop()       │  ◄── DistanceToOptimal reads dist_to_opt│
│      └───────────┬────────────┘                                         │
│                  │  (if nested algorithm used)                          │
│      ┌───────────▼────────────┐                                         │
│      │  run_sub_method()      │  ◄── SubRunConfig{M}                    │
│      │  sub_rng = Xoshiro(    │  ──► child RNG derived from outer_rng   │
│      │    rand(outer_rng,…))  │  ──► independent core time tracking     │
│      │  step!(…, sub_logger,  │  ──► sub logs → outer IterationLog      │
│      │         sub_rng)       │                                         │
│      └───────────┬────────────┘                                         │
│                  │                                                      │
│           finalize!() → MethodResult{S}                                 │
│           collected into RunResult → ExperimentResult                   │
└─────────────────────────────────────┬───────────────────────────────────┘
                                      │
┌─────────────────────────────────────▼───────────────────────────────────┐
│  PERSISTENCE PHASE                                                      │
│                                                                         │
│   save_experiment()                                                     │
│       ├── result.jld2              (full binary, fast reload)           │
│       ├── run{N}_{method}.csv      (per-method, human-readable,        │
│       │                            includes dist_to_opt column)        │
│       └── manifest.json           (name, metadata, no binary needed)   │
└─────────────────────────────────────┬───────────────────────────────────┘
                                      │
┌─────────────────────────────────────▼───────────────────────────────────┐
│  ANALYSIS PHASE                                                         │
│                                                                         │
│   load_experiment()  ──►  to_dataframe()  (incl. :dist_to_opt col)     │
│                                │                                        │
│                      filter_methods() / aggregate_runs()                │
│                                │                                        │
│                      user transforms   (DataFrame -> DataFrame)         │
│                                │                                        │
│                       METHOD_COLOR_REGISTRY + MethodStyle               │
│                       PlotSpec / FigureLayout                           │
│                                │                                        │
│                      render_figure()  ──►  save_figure(.pdf / .png)    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 15. Key Architectural Decisions

| Decision | Rationale |
|----------|-----------|
| `Objective` abstract type (renamed from `DataFidelity`) | Honest about what the abstraction is — the (typically smooth) main objective term — without inverse-problems jargon that misleads on problems like Rosenbrock |
| `gs :: Vector{Regularizer}` with empty default + convenience constructors | Pure (non-composite) problems remain expressible as `Problem(f, x0)`; composite problems carry explicit `gs`; no `Union{Nothing,Vector}` indirection |
| `Hessian` abstract type with `apply` / `materialize` / `diagonal` interface | Unifies exact Hessians, full quasi-Newton (BFGS, SR1), L-BFGS, and structured forms under one dispatch surface; each concrete type declares which operations are available |
| `StepSize` umbrella with `LineSearch <: StepSize` subset | Type-honest: closed-form rules (Fixed, BB, Cauchy) and genuine 1D searches (Armijo, Wolfe) are both valid step-size strategies; the subset gives a real dispatch handle for code that needs to specifically target line searches |
| `StoppingCriteria` hierarchy replaces `converged` + `for` loop | Full control over termination: count, time, tolerance, composites, all independently testable |
| Stopping criteria separated from algorithm struct | Same algorithm, different run budgets across experiments; no code changes required |
| `@core_timed` in algorithm code, exception-safe, accumulates into `state.timing.core_time_ns` | Scientific discipline: only the kernel is measured; bookkeeping invisible to the clock; error recovery does not corrupt timing |
| `log_iter!` accumulates `entry.core_time_ns` into `logger.total_core_ns` | `TimeLimit` reads `elapsed_core_s(logger)` — per-iteration core time is logged, then summed; wall-clock never used as a stopping criterion |
| Logger passed as explicit `step!` parameter — not stored in state | Algorithm code is pure: no logging infrastructure in state structs; logger strategy controlled entirely by the runner |
| `step!(method, state, problem, iter, logger, rng)` extended signature | Logger and rng are injected by the runner and forwarded by algorithms to `run_sub_method` — clean, testable, no hidden state |
| Four canonical state groups (`IterateGroup`, `MetricsGroup`, `TimingGroup`, method-specific `Numerics`) | Clean separation of concerns; sub-routines can receive independent groups; `extract_log_entry` default is trivial; no field duplication permitted |
| `VariantGrid.builder` returns `IterativeMethod` (not `ExperimentalMethod`) | Grids work uniformly for conventional and experimental methods; `resolve_methods` routes each produced method to the right bucket based on its concrete type |
| `SubRunConfig{M}` parametric over method type | Type-stable `init_state` → `SubResult{S}` with concrete `S` → type-stable `final_state` access in `step!` |
| Child RNG `Xoshiro(rand(outer_rng, UInt64))` in `run_sub_method` | Deterministic, reproducible sub-runs; independent of outer rng's future draws |
| Per-method RNG `Xoshiro(hash((seed, run_id, name)))` | Adding or removing a method does not alter any other method's RNG stream; full between-run and between-method independence |
| `MethodResult{S}` parametric over state type | Concrete `final_state` type preserved through the result hierarchy; `finalize!` is type-stable; warm-up can access `result.final_state.iterate.x` without Any-dispatch |
| `Problem.x_opt` set by generator; `dist_to_opt` computed by runner | Algorithms are unaware of optimality tracking; `DistanceToOptimal` criterion and `dist_to_opt` logging activate automatically when `x_opt` is non-nothing |
| `DistanceToOptimal` returns `(false, :none)` when `x_opt` is nothing | Criterion is safe to include in any stopping config regardless of problem type; never fires spuriously |
| `WarmupStrategy` hierarchy with `run_warmup` dispatch | Warm-up is optional, declarative, and serializable; warm-up result (x0) is shared across all methods in a run |
| `IterativeWarmup` calls `run_method` and extracts `final_state.iterate.x` | Warm-up reuses the full runner machinery (logging, stopping, debug); relies on universal `iterate :: IterateGroup` convention |
| `FunctionWarmup` uses `name :: Symbol` + `WARMUP_FUNCTIONS` registry | Pure-function warm-ups remain JLD2-serializable |
| `FileProblem.loader_name :: Symbol` + `FILE_LOADERS` registry | Raw functions are not JLD2-serializable; symbol reference is |
| `DebugConfig` in `ExperimentConfig`; checks run by runner after `log_iter!` | Debug mode is orthogonal to algorithms, verbosity, and logging; disabled by default; zero cost when `enabled = false` |
| `DebugCheck` dispatch with `prev_entry` parameter | Checks that require two consecutive entries (e.g. `CheckObjectiveMonotonicity`) receive both; first-iteration check is a no-op via `isnothing(prev)` guard |
| `ObjectiveStagnation` uses direct index access instead of slice | Eliminates per-check array allocation |
| `next_experiment_path` uses atomic `mkdir` | Eliminates TOCTOU race when two processes write to the same log root simultaneously |
| Verbosity colocated with logging in `logging.jl` | Both share the `Logger` struct; separating into its own file would create artificial coupling |
| `aggregate_runs` modes `:all`, `:mean`, `:median` | `:all` preserves every run for full distribution; `:mean`/`:median` reduce to a representative curve; `:best` omitted — cherry-picking runs has no sound benchmarking interpretation |
| `FigureLayout` as `Matrix{Union{PlotSpec,Nothing}}` | Any grid formation expressible as a Julia matrix literal; blank cells are `nothing`; arbitrary sizes |
| Transforms as `DataFrame -> DataFrame` | No DSL to learn; composable with DataFramesMeta; independently unit-testable |

---

## 16. Extension Guide

### Adding a new conventional baseline

Create `algorithms/conventional/<name>/<name>.jl`. Define the struct, implement
`init_state` (using `IterateGroup`, `MetricsGroup`, `TimingGroup`), `step!` with
the full signature `step!(method, state, problem, iter, logger, rng)` (use
`@core_timed state begin ... end` around the kernel), and `extract_log_entry`.
Add it to an `ExperimentConfig`. The runner, logger, stopping criteria, and plots
all pick it up automatically.

Before writing any code, create `algorithms/conventional/<name>/<name>.md`
following the structure of `algorithms/conventional/gradient_descent/gradient_descent.md`:
problem statement, iteration formula, Julia structs, `init_state` / `step!` /
`extract_log_entry` contracts, and a full variable mapping table. If the method
has pluggable components (directions, step-size rules, ...), give each a dedicated
`<component>.md` file in the components directory.

### Adding an algorithm that uses a sub-algorithm

Embed a `SubRunConfig{M}` field in the outer algorithm struct (typed concretely for
type stability). Call `run_sub_method` inside `step!`, forwarding the `logger` and
`rng` that the runner injected. See Module 5 for the schematic.

### Adding a new stopping criterion

Add a struct subtyping `StoppingCriteria` and a `should_stop` method to `stopping.jl`.
It can immediately be used standalone or composed inside `CompositeCriteria`. Access
state quantities via `state.metrics.*`.

```julia
@kwdef struct RelativeObjectiveDecrease <: StoppingCriteria
    tol :: Float64 = 1e-6
end

function should_stop(c::RelativeObjectiveDecrease, state, iter, logger)
    iter < 2 && return (false, :none)
    prev = logger.iter_logs[end-1].objective
    curr = logger.iter_logs[end].objective
    rel  = abs(prev - curr) / max(abs(prev), 1.0)
    rel <= c.tol ? (true, :relative_obj_converged) : (false, :none)
end
```

### Adding a warm-up

**Iterative warm-up** (run a cheap method to find a better x0):

```julia
config = ExperimentConfig(
    ...,
    warmup = IterativeWarmup(
        method   = GradientDescent(step_size=FixedStep(α=0.1)),
        criteria = MaxIterations(n=100),
    ),
)
```

**Function warm-up** (closed-form initialization):

```julia
register_warmup!(:custom_init, (problem, rng) -> begin
    # ... return Vector{Float64}
end)

config = ExperimentConfig(..., warmup = FunctionWarmup(:custom_init))
```

### Adding a known optimal point to a problem

When registering a problem generator, embed `x_opt` in the returned `Problem`:

```julia
register_random_problem!(:quadratic, (rng, p) -> begin
    A     = randn(rng, p.n, p.n); A = A'A + p.μ * I    # positive definite
    b     = randn(rng, p.n)
    x_opt = A \ b                                       # known minimizer
    x0    = zeros(p.n)
    Problem(QuadraticObjective(A, b), x0; x_opt = x_opt)
end)
```

`DistanceToOptimal` and the `dist_to_opt` log column then activate automatically.

### Adding a new debug check

Add a struct subtyping `DebugCheck` and a `debug_check!` method to `debug.jl`.

```julia
@kwdef struct CheckHessianPositiveDefiniteness <: DebugCheck
    sample_directions :: Int = 5
end

function debug_check!(c::CheckHessianPositiveDefiniteness, cfg, state, problem,
                      entry, prev, iter)
    H = hessian(problem.f, state.iterate.x)
    for _ in 1:c.sample_directions
        d = randn(length(state.iterate.x))
        curvature = dot(d, apply(H, d))
        if curvature < 0
            trigger_debug!(cfg, iter,
                "Hessian is not positive definite: d'Hd = $(curvature)")
            return
        end
    end
end
```

### Adding a new problem

Create `problems/<name>/<name>.md` following `problems/rosenbrock/rosenbrock.md`:

1. State the optimization problem in standard form.
2. Derive and document $\nabla f$ and the Hessian (full matrix or H·d).
3. Provide the known minimizer `x_opt` if it exists analytically; document why it
   is `nothing` if it does not.
4. Include the variable mapping table and the `register_problem!` call.

Then implement `problems/<name>/<name>.jl` with `value`, `grad`, `hessian`
(returning a `Hessian` object), and the registration call. The `<name>.md` is the
contract; the `.jl` is the implementation.

### Adding a new logged field

Add the field to `IterationLog` or to `extras` in `extract_log_entry`. The CSV
sidecar picks up all `extras` keys automatically via `to_dataframe()`.

### Adding a new step-size rule (or line search)

See `step_sizes.md` for the full template. In summary:

1. Decide whether the rule is a closed-form `StepSize` or a genuine `LineSearch`
   (subtyping `LineSearch` enables future code that targets searches specifically).
2. Add the concrete struct and `compute_step_size` method to
   `components/step_sizes.jl`.
3. Wrap only the mathematically core operations in `@core_timed state`.
4. Add the abbreviation via `register_abbreviation!` if you want a custom short name.
5. Pass the new rule as the `step_size` field in `GradientDescent(...)`.

No changes to `step!`, the runner, logging, or stopping criteria are required.

### Adding a fixed color for a method across all plots

```julia
register_method_color!("GradientDescent[step_size=Armijo]", "#0072B2")
```

Call once at session startup. Per-plot `method_styles` in `PlotSpec` take precedence
over the registry when both are present.

### Plotting across multiple experiments

```julia
df1 = to_dataframe(load_experiment("logs/20260417/001/")) |> d -> @transform(d, :exp = "exp1")
df2 = to_dataframe(load_experiment("logs/20260417/002/")) |> d -> @transform(d, :exp = "exp2")
df  = vcat(df1, df2)
PlotSpec(data=df, group_by=:exp, ...)
```
