# Test Engine for Iterative Methods — Architecture (v3)

> A Julia-based experimentation framework for benchmarking conventional solvers against
> multiple variants of an under-development iterative method, with composable stopping
> criteria, nested algorithm support, a structured problem factory, comprehensive logging,
> and a flexible multi-figure plotting pipeline.

---

## Table of Contents

1. [Design Philosophy](#1-design-philosophy)
2. [High-Level Layer Map](#2-high-level-layer-map)
3. [Layer 1 — Algorithm Abstraction & Core Timing](#3-layer-1--algorithm-abstraction--core-timing)
4. [Layer 2 — Variant Grid Engine](#4-layer-2--variant-grid-engine)
5. [Layer 3 — Stopping Criteria](#5-layer-3--stopping-criteria)
6. [Layer 4 — Nested Algorithm Infrastructure](#6-layer-4--nested-algorithm-infrastructure)
7. [Layer 5 — Experiment Orchestration](#7-layer-5--experiment-orchestration)
8. [Layer 6 — Logging System](#8-layer-6--logging-system)
9. [Layer 7 — Verbosity System](#9-layer-7--verbosity-system)
10. [Layer 8 — Persistence & Experiment Naming](#10-layer-8--persistence--experiment-naming)
11. [Layer 9 — Problem Factory](#11-layer-9--problem-factory)
12. [Layer 10 — Analysis & Plotting](#12-layer-10--analysis--plotting)
13. [Directory & Module Structure](#13-directory--module-structure)
14. [Data Flow Diagram](#14-data-flow-diagram)
15. [Key Architectural Decisions](#15-key-architectural-decisions)
16. [Extension Guide](#16-extension-guide)

---

## 1. Design Philosophy

The framework is built on four Julia-native principles:

- **Multiple dispatch over class hierarchies.** Every algorithm, component, stopping
  criterion, and problem is a dispatch point. Adding a new variant never requires
  touching existing code.
- **Separation of concerns across layers.** Algorithms know nothing about logging.
  Loggers know nothing about plotting. Stopping criteria know nothing about algorithms.
  Each layer communicates through well-defined data structures.
- **Declarative experiment definition.** An experiment is a plain, serializable data
  structure (`ExperimentConfig`). Running it, saving it, and reloading it are separate,
  independent operations.
- **Scientific measurement discipline.** Timing records only the core mathematical
  computation inside each step, accumulated per iteration and summed across iterations.
  All bookkeeping — logging, stopping criterion checks, verbosity output — is
  deliberately excluded from measured time.

The top-level concerns flow in one direction:

```
Problems  ──►  Algorithms  ──►  Experiments  ──►  Logging  ──►  Analysis / Plotting
```

---

## 2. High-Level Layer Map

| Layer | File | Responsibility |
|-------|------|----------------|
| 1 | `core.jl` | Type hierarchy, state groups, algorithm interface, `@core_timed`, run loop |
| 2 | `variants.jl` | Component abstractions, Cartesian grid expansion, auto-naming |
| 3 | `stopping.jl` | Stopping criteria abstraction, composites, `should_stop` dispatch |
| 4 | `core.jl` | Nested algorithm infrastructure (`SubRunConfig`, `run_sub_method`) |
| 5 | `experiment.jl` | Experiment config, result types, orchestration, multi-run management |
| 6 | `logging.jl` | Per-iteration capture, core-time accumulation, event logging, sub-logs |
| 7 | `logging.jl` | Verbosity levels, range-gated console output (co-located with logging) |
| 8 | `persistence.jl` | JLD2 binary + CSV sidecar + JSON manifest; date/counter naming |
| 9 | `problems.jl` | Problem interface, data fidelity, regularizers, problem factory |
| 10 | `analysis.jl` | DataFrame pipeline, color registry, flexible multi-figure layout |

---

## 3. Layer 1 — Algorithm Abstraction & Core Timing

### Type Hierarchy

Every algorithm — conventional or experimental — is a concrete subtype of
`IterativeMethod`. The hierarchy separates baselines from the under-development family,
enabling dispatch and experiment-level filtering.

```julia
abstract type IterativeMethod end
abstract type ConventionalMethod  <: IterativeMethod end
abstract type ExperimentalMethod  <: IterativeMethod end
```

Each method is a `@kwdef` struct carrying its own fixed hyperparameters.
**Stopping criteria are no longer embedded in the algorithm struct** — they are
supplied separately at experiment definition time (see Layer 3). This keeps the
algorithm struct a pure description of the method, not of how long to run it.

```julia
@kwdef struct GradientDescent <: ConventionalMethod
    step_size :: Float64 = 0.01
end

@kwdef struct MyMethod <: ExperimentalMethod
    step_size  :: Float64       = 0.01
    # Swappable component slots (see Layer 2)
    hessian    :: HessianApprox = BFGS()
    minor      :: MinorUpdate   = NoMinorUpdate()
    linesearch :: LineSearch    = ArmijoLS()
end
```

### State Parameter Groups

A method's state struct can carry many heterogeneous fields: the current iterate,
convergence metrics, timing, Hessian approximations, step vectors, curvature
scalars, flags that toggle subroutines, and handles to method-specific functions.
Keeping all of these flat becomes unmanageable for complex methods and makes it
impossible for a sub-routine to reuse a clean subset.

The solution is to partition every state struct into **four canonical groups** plus
one method-specific numerics group:

```julia
# --- Shared groups (identical structure across all methods) ---

# The optimization variables
@kwdef mutable struct IterateGroup
    x      :: Vector{Float64}              # current iterate
    x_prev :: Vector{Float64} = Float64[] # previous iterate (step norm, momentum, …)
end

# Scalar convergence metrics — mirrors the fixed fields of IterationLog
@kwdef mutable struct MetricsGroup
    objective     :: Float64 = Inf
    gradient_norm :: Float64 = Inf
    step_norm     :: Float64 = Inf
    residual      :: Float64 = Inf
end

# Per-step core computation time; reset by runner before each step!
@kwdef mutable struct TimingGroup
    core_time_ns :: Int64 = 0
end

# --- Method-specific numerics group (one per concrete method) ---

# Example for MyMethod: all scalars, vectors, matrices, and behavioral flags.
# Each flag enables a distinct subroutine or update path inside step!.
@kwdef mutable struct MyMethodNumerics
    # Scalars
    step_size     :: Float64 = 0.0
    curvature     :: Float64 = 0.0
    # Vectors
    gradient      :: Vector{Float64} = Float64[]
    direction     :: Vector{Float64} = Float64[]
    # Matrices
    H             :: Matrix{Float64} = Matrix{Float64}(undef, 0, 0)
    # Behavioral flags — each toggles a distinct code path in step!
    use_correction      :: Bool = false   # enable correction subroutine
    subproblem_solved   :: Bool = false   # set true when inner solve succeeds
    use_extra_x_update  :: Bool = false   # apply secondary iterate update after main step
end

# --- Concrete state struct ---

@kwdef mutable struct MyMethodState
    iterate  :: IterateGroup
    metrics  :: MetricsGroup
    timing   :: TimingGroup
    numerics :: MyMethodNumerics
    _logger  :: Union{Nothing, Logger} = nothing  # injected by runner; used for sub-calls
end
```

**Sub-routine states.** When a sub-routine solves a subproblem, it receives its own
state struct following the same four-group pattern. The outer algorithm passes an
`IterateGroup` (or a derived subproblem variable) as the sub-routine's initial iterate.
The sub-state has its own independent `TimingGroup`; its accumulated `core_time_ns` is
reported in `SubResult.core_time_ns` separately from the outer timing.

**`extract_log_entry` simplification.** Because `state.metrics` mirrors
`IterationLog`'s fixed fields, the default implementation is trivial:

```julia
function extract_log_entry(method::IterativeMethod, state, iter::Int)::IterationLog
    IterationLog(
        iter          = iter,
        core_time_ns  = state.timing.core_time_ns,
        objective     = state.metrics.objective,
        gradient_norm = state.metrics.gradient_norm,
        step_norm     = state.metrics.step_norm,
        residual      = state.metrics.residual,
    )
    # Methods override this to additionally populate the extras dict
end
```

**`should_stop` field access.** Stopping criteria that inspect state read from the
appropriate group, e.g. `state.metrics.gradient_norm` (not `state.gradient_norm`).

### The Three Dispatch Points

These three functions are the only interface an algorithm must implement:

```julia
# Called once before the loop; returns a mutable state object
function init_state(method::IterativeMethod, problem, rng::AbstractRNG) end

# Called every iteration; mutates state in place.
# Core computation must be wrapped in @core_timed (see below).
function step!(method::IterativeMethod, state, problem, iter::Int) end

# Called to extract a log entry; dispatches on method type.
# Allows algorithm-specific fields to populate the extras dict.
function extract_log_entry(method::IterativeMethod, state, iter::Int)::IterationLog end
```

### Core Timing — `@core_timed`

Scientific timing measures **only the mathematical kernel** of each step.
The `@core_timed` macro is the single entry point for this. It accumulates elapsed
nanoseconds into `state.timing.core_time_ns`. The runner resets that field to zero
before each `step!` call; after `step!` returns, `extract_log_entry` copies it into
`IterationLog.core_time_ns`; `log_iter!` then adds it to `logger.total_core_ns`.
The runner never wraps `step!` in a timer itself.

```julia
"""
    @core_timed state expr

Wraps `expr` in a high-resolution timer. Elapsed nanoseconds are **added** to
`state.timing.core_time_ns`, so multiple disjoint kernels in a single step are all
counted without timing the bookkeeping between them.
"""
macro core_timed(state, expr)
    quote
        _t0 = time_ns()
        $(esc(expr))
        $(esc(state)).timing.core_time_ns += time_ns() - _t0
    end
end
```

Usage inside `step!` — the algorithm author controls exactly what counts:

```julia
function step!(m::MyMethod, state, problem, iter)
    @core_timed state begin
        g  = grad(problem.f, state.iterate.x)
        H  = approximate_hessian(m.hessian, state)
        Δx = solve_direction(H, g)
    end

    # Line search and minor update may themselves call @core_timed if appropriate:
    α = search_step(m.linesearch, problem, state, Δx)
    state.iterate.x .-= α .* Δx
    state.metrics.step_norm = norm(α .* Δx)
    apply_minor_update!(m.minor, state, problem, iter)
end
```

Every concrete state struct contains a `TimingGroup` field named `timing`. The runner
resets `state.timing.core_time_ns = 0` before each `step!` call so the logger always
sees a single-step measurement.

### The Generic Runner

The runner owns the loop. Algorithms never call the logger directly — it is injected
by the runner. `converged` is gone; a `StoppingCriteria` object controls termination
(see Layer 3).

```julia
function run_method(method     :: IterativeMethod,
                    problem,
                    criteria   :: StoppingCriteria,
                    logger     :: Logger,
                    rng        :: AbstractRNG)

    state = init_state(method, problem, rng)
    state._logger = logger                      # inject logger reference for sub-calls
    log_init!(logger, method, state)
    iter  = 0

    while true
        iter += 1
        state.timing.core_time_ns = 0           # reset per-step accumulator
        step!(method, state, problem, iter)
        # ↑ only this call contributes to state.timing.core_time_ns

        entry = extract_log_entry(method, state, iter)
        log_iter!(logger, entry)                # also: logger.total_core_ns += entry.core_time_ns

        # Stopping check happens AFTER logging — never timed
        stop, reason = should_stop(criteria, state, iter, logger)
        if stop
            log_event!(logger, reason, iter)
            break
        end
    end

    return finalize!(logger, method, state)     # returns MethodResult
end
```

---

## 4. Layer 2 — Variant Grid Engine

This layer models each **dimension of variation** as a typed component, then
constructs all valid combinations automatically.

### Component Abstraction

Each variation axis is an abstract type with concrete implementations:

```julia
# Dimension 1: Hessian approximation strategy
abstract type HessianApprox end

struct FullHessian              <: HessianApprox end
struct BFGS                     <: HessianApprox end
struct SR1                      <: HessianApprox end
@kwdef struct LBFGS             <: HessianApprox;  m::Int = 5 end
@kwdef struct DiagBFGS          <: HessianApprox;  damped::Bool = false end

# Dimension 2: Minor correction applied after each main update
abstract type MinorUpdate end

struct NoMinorUpdate            <: MinorUpdate end
@kwdef struct MomentumStep      <: MinorUpdate;    α::Float64 = 0.1 end
@kwdef struct NesterovStep      <: MinorUpdate;    α::Float64 = 0.1 end
@kwdef struct CorrectionStep    <: MinorUpdate;    n_inner::Int = 3 end

# Dimension 3: Step-size selection
abstract type LineSearch end

@kwdef struct FixedStep         <: LineSearch;     α::Float64 = 0.01 end
struct ArmijoLS                 <: LineSearch end
struct WolfeLS                  <: LineSearch end
```

Each component is dispatched inside `step!`, keeping every piece independently testable:

```julia
function step!(m::MyMethod, state, problem, iter)
    @core_timed state begin
        g  = grad(problem.f, state.iterate.x)
        H  = approximate_hessian(m.hessian, state)    # dispatches on hessian type
        Δx = solve_direction(H, g)
        α  = search_step(m.linesearch, problem, state, Δx)
        state.iterate.x .-= α .* Δx
        apply_minor_update!(m.minor, state, problem, iter)
    end
end
```

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
    builder       :: Function          # (;param=value, ...) -> ExperimentalMethod
    filters       :: Vector{Function}  = []  # [(combo::NamedTuple) -> Bool]
    shared_params :: NamedTuple        = (;)
end
```

### Grid Expansion

`expand(grid)` takes the Cartesian product of all axes, applies filters, builds each
method instance, and attaches auto-generated names. This function operates on a
**single** `VariantGrid` and is independently callable and unit-testable.

```julia
struct VariantSpec
    name       :: String               # full human-readable name
    short_name :: String               # compact legend label
    params     :: NamedTuple           # the exact parameter combination
    method     :: ExperimentalMethod   # ready-to-run instance
end

function expand(grid::VariantGrid)::Vector{VariantSpec}
    # Cartesian product → filter → build → name
end
```

### Naming Convention

| Format | Example |
|--------|---------|
| Full (logging, filenames) | `MyMethod[hessian=BFGS,minor=Mom10,linesearch=Wolfe]` |
| Short (plot legends) | `MM/BFGS/Mom10/Wlf` |

```julia
const ABBREVIATIONS = Dict(
    "MyMethod"  => "MM",    "BFGS"     => "BFGS",
    "SR1"       => "SR1",   "LBFGS"    => "LBFG",
    "None"      => "∅",     "Momentum" => "Mom",
    "Nesterov"  => "Nest",  "Armijo"   => "Arm",
    "Wolfe"     => "Wlf",
)
```

### Defining a Grid (Usage Example)

```julia
hessian_axis = VariantAxis(:hessian,
    BFGS()                => "BFGS",
    SR1()                 => "SR1",
    LBFGS(m=5)            => "LBFGS5",
    LBFGS(m=10)           => "LBFGS10",
    DiagBFGS(damped=true) => "DiagBFGS",
)

minor_axis = VariantAxis(:minor,
    NoMinorUpdate()           => "None",
    MomentumStep(α=0.05)      => "Mom05",
    MomentumStep(α=0.1)       => "Mom10",
    NesterovStep(α=0.1)       => "Nest",
    CorrectionStep(n_inner=3) => "Corr3",
)

linesearch_axis = VariantAxis(:linesearch,
    ArmijoLS() => "Armijo",
    WolfeLS()  => "Wolfe",
)

grid = VariantGrid(
    base_name     = "MyMethod",
    axes          = [hessian_axis, minor_axis, linesearch_axis],
    builder       = (;hessian, minor, linesearch, kwargs...) ->
                        MyMethod(; hessian, minor, linesearch, step_size=0.01),
    filters       = [
        combo -> !(combo.hessian isa SR1      && combo.linesearch isa WolfeLS),
        combo -> !(combo.minor isa CorrectionStep && combo.hessian isa DiagBFGS),
    ],
    shared_params = (;),
)
# 5 × 5 × 2 = 50 combinations, minus filtered → ~44 named variants
```

---

## 5. Layer 3 — Stopping Criteria

The runner uses a **`while true` loop controlled entirely by `StoppingCriteria`**.
There is no `for iter in 1:max_iter`. This gives full, composable control over
how many steps any algorithm takes: by count, time, proximity to solution, or
any user-defined condition.

### Type Hierarchy

```julia
abstract type StoppingCriterion end

# Terminate after a fixed number of steps
@kwdef struct MaxIterations <: StoppingCriterion
    n :: Int = 1000
end

# Terminate when accumulated **core computation** time exceeds the budget.
# Time is measured as the sum of per-iteration core_time_ns values recorded
# in the logger — wall-clock and bookkeeping time are never counted.
@kwdef struct TimeLimit <: StoppingCriterion
    seconds :: Float64 = 60.0
end

# Terminate when gradient norm falls below threshold
@kwdef struct GradientTolerance <: StoppingCriterion
    tol :: Float64 = 1e-6
end

# Terminate when objective change over last `window` iters is below threshold
@kwdef struct ObjectiveStagnation <: StoppingCriterion
    tol    :: Float64 = 1e-8
    window :: Int     = 10
end

# Terminate when step norm falls below threshold
@kwdef struct StepTolerance <: StoppingCriterion
    tol :: Float64 = 1e-8
end

# Combine multiple criteria: :any (first satisfied wins) or :all (all must hold)
@kwdef struct CompositeCriterion <: StoppingCriterion
    criteria :: Vector{StoppingCriterion}
    mode     :: Symbol = :any    # :any | :all
end

# Convenience constructors
stop_when_any(cs...) = CompositeCriterion(criteria=collect(cs), mode=:any)
stop_when_all(cs...) = CompositeCriterion(criteria=collect(cs), mode=:all)
```

### The `should_stop` Interface

```julia
# Returns (stop::Bool, reason::Symbol)
function should_stop(c::StoppingCriterion, state, iter::Int, logger::Logger) end

function should_stop(c::MaxIterations, state, iter, logger)
    iter >= c.n ? (true, :max_iterations) : (false, :none)
end

function should_stop(c::TimeLimit, state, iter, logger)
    # Uses accumulated core computation time — never wall-clock
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
    recent = logger.iter_logs[end-c.window+1:end]
    Δ = abs(recent[1].objective - recent[end].objective)
    Δ <= c.tol ? (true, :objective_stagnated) : (false, :none)
end

function should_stop(c::CompositeCriterion, state, iter, logger)
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

Stopping criteria are supplied at experiment definition time, separately from the
algorithm itself:

```julia
# A common stopping configuration shared across methods
default_stop = stop_when_any(
    MaxIterations(n=2000),
    GradientTolerance(tol=1e-7),
    TimeLimit(seconds=120.0),
)

# A different budget for a fast baseline
quick_stop = MaxIterations(n=200)

# Assigned per-method in the experiment config (see Layer 5)
```

This separation means the same `GradientDescent` struct can be benchmarked for 500
iterations in one experiment and 5000 in another without any code changes.

---

## 6. Layer 4 — Nested Algorithm Infrastructure

Some algorithms run another iterative method as a **sub-routine inside their own
`step!`**. Examples: trust-region methods that solve an inner subproblem iteratively,
bi-level methods, inner loop methods that refine a correction, or meta-algorithms
that call multiple sub-solvers per outer step.

This layer provides the infrastructure to make nested invocation clean, safe, and
fully logged.

### Design Principle

An algorithm struct holds a **sub-method slot** typed as `IterativeMethod`. During
`step!`, the algorithm calls `run_sub_method(...)`, which is a lightweight runner
that returns a `SubResult`. Sub-iteration logs are attached to the current outer
iteration's log entry under `extras`. The outer logger coordinates this; the
sub-runner never writes to disk or console independently unless configured to do so.

### Infrastructure Types

```julia
# Holds configuration for a sub-algorithm invocation
@kwdef struct SubRunConfig
    method        :: IterativeMethod
    criteria      :: StoppingCriteria
    log_sub_iters :: Bool = true   # attach sub-iter logs to outer IterationLog.extras
    verbosity     :: VerbosityConfig = VerbosityConfig(level=SILENT)
end

# Result returned from a sub-method run
struct SubResult
    converged    :: Bool
    stop_reason  :: Symbol
    n_iters      :: Int
    final_state  :: Any
    iter_logs    :: Vector{IterationLog}
    core_time_ns :: Int64   # total core time across all sub-iterations
end
```

### `run_sub_method`

```julia
"""
    run_sub_method(config, problem, outer_logger)

Runs the sub-algorithm described by `config`. Attaches sub-iteration logs to the
outer logger's current pending entry if `config.log_sub_iters` is true.
Returns a `SubResult`. The sub-state has its own independent TimingGroup; its
accumulated core time is reported in SubResult.core_time_ns separately.
"""
function run_sub_method(config       :: SubRunConfig,
                        problem,
                        outer_logger :: Logger)::SubResult

    rng        = Random.TaskLocalRNG()   # sub-problems use local RNG; seeded by outer run
    sub_state  = init_state(config.method, problem, rng)
    sub_logger = make_sub_logger(config.verbosity)
    iter       = 0

    while true
        iter += 1
        sub_state.timing.core_time_ns = 0
        step!(config.method, sub_state, problem, iter)

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

### Usage in an Outer Algorithm

```julia
@kwdef struct MyOuterMethod <: ExperimentalMethod
    step_size   :: Float64         = 0.01
    inner_sub   :: SubRunConfig    = SubRunConfig(
                       method   = ConjugateGradient(),
                       criteria = stop_when_any(MaxIterations(50), GradientTolerance(1e-5)),
                   )
    hessian     :: HessianApprox   = BFGS()
    linesearch  :: LineSearch      = ArmijoLS()
end

function step!(m::MyOuterMethod, state, problem, iter)
    # Core outer computation
    @core_timed state begin
        g           = grad(problem.f, state.iterate.x)
        sub_problem = build_subproblem(state, g, problem)
    end

    # Run inner algorithm — its core time is logged separately in sub-logs
    sub_result = run_sub_method(m.inner_sub, sub_problem, state._logger)

    @core_timed state begin
        Δx = extract_direction(sub_result.final_state)
        α  = search_step(m.linesearch, problem, state, Δx)
        state.iterate.x .-= α .* Δx
        state.metrics.step_norm = norm(α .* Δx)
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

### Sub-Algorithm State Access

The outer algorithm's state carries a `_logger` reference (injected by the runner
before calling `step!`) so that `run_sub_method` can attach sub-logs without needing
the logger passed explicitly into `step!`:

```julia
state._logger = logger   # set once by runner before the loop; read in step! if needed
```

---

## 7. Layer 5 — Experiment Orchestration

### Experiment Naming

Each experiment is stored under a two-level path: a **date folder** and a
**zero-padded sequential counter** that resets each day. This produces a clean,
browsable log hierarchy:

```
logs/
└── 20260417/          ← date folder (YYYYMMDD)
    ├── 001/           ← first experiment of this day
    │   ├── manifest.json
    │   ├── result.jld2
    │   └── ...
    ├── 002/           ← second experiment of this day
    └── 003/
```

The counter is determined at save time by scanning the date folder for the highest
existing number and incrementing it. The human-readable `name` field from
`ExperimentConfig` is stored inside `manifest.json`, not in the path.

```julia
function next_experiment_path(log_root::String)::String
    date_str  = Dates.format(today(), "yyyymmdd")
    day_dir   = joinpath(log_root, date_str)
    mkpath(day_dir)
    existing  = filter(isdir, readdir(day_dir; join=true))
    nums      = [parse(Int, basename(d)) for d in existing
                 if occursin(r"^\d{3,}$", basename(d))]
    next_num  = isempty(nums) ? 1 : maximum(nums) + 1
    return joinpath(day_dir, lpad(next_num, 3, '0'))
end
```

### `expand` and `resolve_methods`

These two functions are **not** the same and must not share a name.

- `expand(grid::VariantGrid)` (Layer 2) operates on a **single** grid and returns a
  `Vector{VariantSpec}`. It is independently callable and unit-testable.
- `resolve_methods(config::ExperimentConfig)` is the **config-level aggregator**: it
  concatenates `config.conventional_methods`, `config.experimental_methods`, and the
  flattened output of calling `expand(grid)` on every entry in `config.variant_grids`.
  It returns two flat `Vector{Tuple{String, IterativeMethod}}` — one conventional,
  one experimental.

The separation keeps grid expansion a pure, reusable primitive while orchestration
logic stays in Layer 5.

### ExperimentConfig

```julia
@kwdef struct ExperimentConfig
    name                 :: String                     # human label stored in manifest
    problem_spec         :: ProblemSpec                # see Layer 9
    conventional_methods :: Vector{ConventionalMethod}
    experimental_methods :: Vector{ExperimentalMethod} = []
    variant_grids        :: Vector{VariantGrid}        = []
    stopping_criteria    :: StoppingCriteria           = stop_when_any(
                                MaxIterations(1000), GradientTolerance(1e-6))
    # Per-method override: method_name => StoppingCriteria
    method_criteria      :: Dict{String, StoppingCriteria} = Dict()
    n_runs               :: Int                        = 1
    seed                 :: Union{Int,Nothing}         = 42
    # Single seed governing ALL randomness: data generation, initial point x0,
    # and any stochastic algorithmic components. Per-run rng derived as
    # MersenneTwister(seed + run_id - 1). Set nothing to use global RNG.
    tags                 :: Dict{String,Any}           = Dict()
end
```

`method_criteria` lets specific methods use different stopping budgets within the
same experiment — e.g., a fast baseline gets `MaxIterations(100)` while the
experimental methods get a full composite criterion.

### Result Types

```julia
# Outcome of running one method on one problem instance (one run_id)
struct MethodResult
    method_name  :: String
    iter_logs    :: Vector{IterationLog}   # one entry per iteration
    final_state  :: Any                    # concrete state struct at termination
    stop_reason  :: Symbol                 # e.g. :max_iterations, :gradient_converged
    n_iters      :: Int
end

# All method outcomes for a single run
struct RunResult
    run_id         :: Int
    method_results :: Dict{String, MethodResult}   # keyed by method full name
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

`finalize!(logger, method, state)` (called by the runner at loop exit) assembles and
returns a `MethodResult`. The runner collects these into a `RunResult` per run and
wraps everything in `ExperimentResult` at the end of `run_experiment`.

### Result Hierarchy (Overview)

```
ExperimentResult
    ├── config          :: ExperimentConfig
    ├── experiment_path :: String
    ├── timestamp       :: DateTime
    ├── host            :: String
    └── run_results[]   :: RunResult
            ├── run_id
            └── method_results :: Dict{String, MethodResult}
                    ├── method_name  :: String
                    ├── iter_logs    :: Vector{IterationLog}
                    ├── final_state  :: Any
                    ├── stop_reason  :: Symbol
                    └── n_iters      :: Int
```

### Orchestration Loop

```julia
function run_experiment(config   :: ExperimentConfig,
                        log_root :: String = "logs";
                        verbosity :: VerbosityConfig = VerbosityConfig())

    exp_path = next_experiment_path(log_root)
    mkpath(exp_path)

    conventional, experimental = resolve_methods(config)   # calls expand() on each VariantGrid
    results = RunResult[]

    for run_id in 1:config.n_runs
        # One seed controls everything: data, x0, and any stochastic steps
        rng = isnothing(config.seed) ? Random.TaskLocalRNG() :
                  Random.MersenneTwister(config.seed + run_id - 1)
        problem = make_problem(config.problem_spec, rng)

        method_results = Dict{String, MethodResult}()
        for (name, method) in [conventional; experimental]
            criteria = get(config.method_criteria, name, config.stopping_criteria)
            logger   = Logger(name, run_id, exp_path, verbosity)
            result   = run_method(method, problem, criteria, logger, rng)
            method_results[name] = result
        end
        push!(results, RunResult(run_id, method_results))
    end

    exp_result = ExperimentResult(config, exp_path, now(), gethostname(), results)
    save_experiment(exp_result)
    return exp_result
end
```

---

## 8. Layer 6 — Logging System

The logger is external to all algorithms. It is injected by the runner and captures
data through three hooks: `log_init!`, `log_iter!`, and `log_event!`.

### IterationLog

```julia
@kwdef mutable struct IterationLog
    iter           :: Int
    core_time_ns   :: Int64            # nanoseconds of core computation this step
    objective      :: Float64
    gradient_norm  :: Float64
    step_norm      :: Float64
    residual       :: Float64
    extras         :: Dict{Symbol,Any} = Dict()  # algorithm-specific & sub-logs
end
```

The `extras` dict carries algorithm-specific fields (curvature estimates, inner
iteration counts) and, when nested algorithms are used, `:sub_logs` containing
the full `Vector{IterationLog}` from each sub-method run.

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
    start_wall_time  :: Float64                # wall clock at log_init! — informational only
    total_core_ns    :: Int64                  # accumulated core nanoseconds across all iters
    pending_sub_logs :: Vector{IterationLog}   # buffer for attach_sub_logs!
end

# Core computation elapsed — the authoritative timing used by TimeLimit
elapsed_core_s(logger::Logger) = logger.total_core_ns / 1e9

# Wall-clock elapsed — informational only, never a stopping criterion
elapsed_wall_s(logger::Logger) = time() - logger.start_wall_time
```

`log_iter!` is the single point where `entry.core_time_ns` is accumulated:

```julia
function log_iter!(logger::Logger, entry::IterationLog)
    push!(logger.iter_logs, entry)
    logger.total_core_ns += entry.core_time_ns   # feeds elapsed_core_s → TimeLimit
    maybe_print(logger, entry)
end
```

`extract_log_entry(method, state, iter)` dispatches on the method type, allowing
each algorithm to populate `extras` while sharing the common log schema.

---

## 9. Layer 7 — Verbosity System

Verbosity is a first-class, orthogonal concern — not scattered `if verbose` checks.
It lives in `logging.jl` alongside the logger, since they share the `Logger` struct.

### Verbosity Levels

```julia
@enum VerbosityLevel begin
    SILENT    = 0    # no output
    MILESTONE = 1    # start, end, and stop events only
    SUMMARY   = 2    # every N iterations (configurable)
    DETAILED  = 3    # every iteration, compact single line
    DEBUG     = 4    # every iteration with full extras dict
end
```

### VerbosityConfig

```julia
@kwdef mutable struct VerbosityConfig
    level       :: VerbosityLevel               = SUMMARY
    print_every :: Int                           = 10
    fields      :: Vector{Symbol}                = [:iter, :objective, :gradient_norm]
    color       :: Bool                          = true
    io          :: IO                            = stdout
    iter_range  :: Union{Nothing,UnitRange{Int}} = nothing
    # e.g. iter_range = 100:200 → DETAILED output only for iterations 100–200
    # iter_range = nothing → apply level uniformly to all iterations
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
        DETAILED                        # force detailed for range iterations
    elseif !isnothing(cfg.iter_range)
        SILENT                          # suppress everything outside the range
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
    level      = MILESTONE,    # normally only print start/end
    iter_range = 100:200,      # but DETAILED for this window
    fields     = [:iter, :objective, :gradient_norm, :core_time_ns],
)
```

---

## 10. Layer 8 — Persistence & Experiment Naming

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
    │   ├── run1_GradientDescent.csv
    │   ├── run1_MyMethod[hessian=BFGS,minor=Mom10,linesearch=Wolfe].csv
    │   └── run2_GradientDescent.csv
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

## 11. Layer 9 — Problem Factory

Problems are declared as typed `ProblemSpec` values. `make_problem(spec, rng)` dispatches
on the spec type to construct the problem. This provides a structured, serializable,
and reproducible system where every problem — analytic, file-based, or randomly
generated — has an identical interface.

### Problem Interface

Every problem has a **composite objective** `f(x) + g₁(x) + g₂(x) + …`, where `f`
is the data fidelity term and the `gᵢ` are regularizers. All algorithms interact
with the problem exclusively through this interface.

```julia
# --- Data Fidelity ---

abstract type DataFidelity end
# Required dispatch for every concrete subtype:
#   value(f, x)               → scalar objective value of f at x
#   grad(f, x)                → gradient vector ∇f(x)
#   hessian_vec(f, x, d)      → Hessian-vector product H_f(x)·d  (d is a direction)
#
# DataFidelity is backed by a kernel h. For LeastSquares, for example:
#   h(x) = 0.5‖Ax − b‖²,  ∇h(x) = Aᵀ(Ax−b),  H_h(x,d) = AᵀAd

# --- Regularizer ---

abstract type Regularizer end
# Required dispatch for every concrete subtype:
#   value(g, x)               → scalar regularizer value
#   prox(g, x, γ)             → proximal operator argmin_u { g(u) + 1/(2γ)‖u−x‖² }

# --- Composite Problem ---

struct Problem
    f    :: DataFidelity
    gs   :: Vector{Regularizer}    # may be empty; total = f + Σgᵢ
    x0   :: Vector{Float64}        # initial point (generated or loaded alongside data)
    n    :: Int                    # problem dimension
    meta :: Dict{Symbol, Any}      # optional: condition number, sparsity level, …
end

# Convenience constructor for the common single-regularizer case
Problem(f, g::Regularizer, x0) = Problem(f, [g], x0, length(x0), Dict())

# Total objective value (used by logging / stopping criteria)
objective(p::Problem, x) = value(p.f, x) + sum(value(g, x) for g in p.gs; init=0.0)
```

**Lasso example** (data fidelity = least squares, regularizer = ℓ₁ norm):

```julia
struct LeastSquaresKernel
    A :: Matrix{Float64}
    b :: Vector{Float64}
end

struct LeastSquares <: DataFidelity
    kernel :: LeastSquaresKernel
end
value(f::LeastSquares, x)          = 0.5 * norm(f.kernel.A * x - f.kernel.b)^2
grad(f::LeastSquares, x)           = f.kernel.A' * (f.kernel.A * x - f.kernel.b)
hessian_vec(f::LeastSquares, x, d) = f.kernel.A' * (f.kernel.A * d)

struct L1Norm <: Regularizer
    λ :: Float64
end
value(g::L1Norm, x)   = g.λ * norm(x, 1)
prox(g::L1Norm, x, γ) = sign.(x) .* max.(abs.(x) .- γ * g.λ, 0.0)  # soft-threshold

# Lasso:  min  0.5‖Ax−b‖² + λ‖x‖₁
# lasso_problem = Problem(LeastSquares(LeastSquaresKernel(A, b)), L1Norm(λ), x0)
```

### ProblemSpec Type Hierarchy

```julia
abstract type ProblemSpec end
```

All `make_problem` implementations share the same signature:

```julia
make_problem(spec::ProblemSpec, rng::AbstractRNG) :: Problem
```

#### Analytic Problems

Problems defined purely analytically — no external data required.

```julia
@kwdef struct AnalyticProblem <: ProblemSpec
    name   :: Symbol        # registered problem identifier, e.g. :rosenbrock
    params :: NamedTuple = (;)
    dim    :: Int = 2
end

const ANALYTIC_PROBLEMS = Dict{Symbol, Function}()

register_problem!(:rosenbrock, (params, rng) -> RosenbrockProblem(params))
register_problem!(:quadratic,  (params, rng) -> QuadraticProblem(params.A, params.b))
register_problem!(:logistic,   (params, rng) -> LogisticProblem(params))

make_problem(s::AnalyticProblem, rng::AbstractRNG) =
    ANALYTIC_PROBLEMS[s.name](s.params, rng)
```

#### File-Based Problems

Data is loaded from a file on disk. The loader is a user-supplied function.

```julia
@kwdef struct FileProblem <: ProblemSpec
    path        :: String
    loader      :: Function    # path::String -> Problem
    description :: String = ""
end

make_problem(s::FileProblem, rng::AbstractRNG) = s.loader(s.path)  # rng unused

# Usage:
FileProblem(
    path        = "data/regression_dataset.csv",
    loader      = p -> build_lasso_from_csv(p),
    description = "UCI regression dataset",
)
```

#### Randomly Generated Problems

A `RandomProblem` generates **data** (e.g. matrix `A` and vector `b` for Lasso)
using an RNG, but produces a `Problem` with **exactly the same interface** as one
built from a file. There is no separate "random problem interface" — algorithms
cannot tell the difference.

There is **no `seed` field** on `RandomProblem`. One seed — `ExperimentConfig.seed`
— controls everything: data generation, initial point `x0`, and any algorithmic
stochasticity. The experiment runner derives a per-run `rng` from this seed and
passes it to `make_problem`.

```julia
@kwdef struct RandomProblem <: ProblemSpec
    name   :: Symbol           # registered generator key, e.g. :lasso, :ridge
    params :: NamedTuple = (;) # e.g. (m=200, n=100, λ=0.1, condition_number=10.0)
end

const RANDOM_GENERATORS = Dict{Symbol, Function}()

register_random_problem!(name::Symbol, gen::Function) =
    (RANDOM_GENERATORS[name] = gen)

# gen signature: (rng::AbstractRNG, params::NamedTuple) -> Problem
# The generator builds A, b, x0, etc. — all from the single rng.
make_problem(s::RandomProblem, rng::AbstractRNG) =
    RANDOM_GENERATORS[s.name](rng, s.params)
```

**Registration example — random Lasso:**

```julia
register_random_problem!(:lasso, (rng, p) -> begin
    A  = randn(rng, p.m, p.n)
    b  = randn(rng, p.m)
    x0 = zeros(p.n)           # or: randn(rng, p.n) for random x0
    Problem(LeastSquares(LeastSquaresKernel(A, b)), L1Norm(p.λ), x0)
end)
```

Usage at experiment definition time (identical structure to `FileProblem`):

```julia
RandomProblem(
    name   = :lasso,
    params = (m=200, n=100, λ=0.05, condition_number=50.0),
)
```

### Using the Factory in ExperimentConfig

```julia
config = ExperimentConfig(
    name         = "Random Lasso sweep λ=0.05",
    problem_spec = RandomProblem(
        name   = :lasso,
        params = (m=200, n=100, λ=0.05, condition_number=50.0),
    ),
    conventional_methods = [GradientDescent(), ISTA()],
    variant_grids        = [grid],
    n_runs               = 10,
    seed                 = 42,   # one seed: data, x0, and any stochastic steps
)

# With n_runs=10 and seed=42, run k uses MersenneTwister(42 + k - 1).
# Each run gets a fresh independently seeded A, b, and x0 — all from
# the same single seed parameter, with no separate per-problem seed.
```

---

## 12. Layer 10 — Analysis & Plotting

The analysis layer has two roles:

1. **DataFrame pipeline** — load a saved experiment, then answer any question by
   filtering, aggregating, and transforming the data.
2. **Figure layout system** — compose any number of plots in any formation and
   render them to a single PDF or image file.

There is no grid-aware analysis layer. Because variant names embed axis information
(e.g. `MyMethod[hessian=BFGS,minor=Mom10,...]`), the user can always parse or filter
on names as plain strings if needed.

### Loading and Transforming

```julia
# Restore a saved experiment
result = load_experiment("logs/20260417/001/")

# Convert all iteration logs to a flat DataFrame
# Columns: :run_id, :method_name, :iter, :objective, :gradient_norm,
#          :step_norm, :residual, :core_time_ns, + any extras keys
df = to_dataframe(result)

# Standard filter/aggregate utilities (all DataFrame -> DataFrame)
df = filter_methods(df, ["GradientDescent", "MyMethod[hessian=BFGS,minor=None,linesearch=Wolfe]"])
df = aggregate_runs(df, :median)    # :all | :mean | :median
```

User transforms are plain `DataFrame -> DataFrame` functions:

```julia
transforms = [
    df -> @transform(df, :log_obj = log10.(:objective)),
    df -> @subset(df, :iter .< 500),
    df -> @transform(df, :core_time_ms = :core_time_ns ./ 1e6),
]
for t in transforms; df = t(df); end
```

### MethodStyle — Per-Method Visual Properties

The user specifies visual style per method, per plot. Styles are never inferred
automatically from data — they are an explicit declaration at plot-definition time.

```julia
@kwdef struct MethodStyle
    color      :: Any                            # any Makie-compatible color spec
    linestyle  :: Symbol    = :solid             # :solid | :dash | :dot | :dashdot
    linewidth  :: Float64   = 2.0
    marker     :: Union{Nothing, Symbol} = nothing
    label      :: Union{Nothing, String} = nothing  # override the legend label
end
```

### Method Color Registry

Colors are **deterministic and visually appealing** by default. A fixed curated
palette (Wong colorblind-safe + Tableau extensions) is assigned to method names
via a stable hash — the same method name always maps to the same color regardless
of experiment or run order. Users can override individual entries.

```julia
# Eight-color palette: colorblind-safe, perceptually distinct
const METHOD_PALETTE = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # teal
    "#CC79A7",  # pink
    "#56B4E9",  # sky blue
    "#D55E00",  # vermillion
    "#F0E442",  # yellow
    "#000000",  # black
]

const METHOD_COLOR_REGISTRY = Dict{String, String}()

# Stable hash-based assignment: same name → same color, no randomness
function method_color(name::String)::String
    METHOD_PALETTE[(hash(name) % length(METHOD_PALETTE)) + 1]
end

# Override a specific method's color across all plots in all experiments
register_method_color!(name::String, color::String) =
    (METHOD_COLOR_REGISTRY[name] = color)

# Lookup: explicit registry first, then stable hash fallback
get_method_color(name::String)::String =
    get(METHOD_COLOR_REGISTRY, name, method_color(name))
```

When `render_figure` draws a line group, it calls `get_method_color(method_name)`
unless a `MethodStyle` entry in the plot's `method_styles` dict overrides it.

### PlotSpec — Describing a Single Plot

```julia
@kwdef struct PlotSpec
    data          :: DataFrame                      # pre-processed DataFrame
    x             :: Symbol          = :iter
    y             :: Symbol          = :objective
    group_by      :: Symbol          = :method_name # color/line grouping
    title         :: String          = ""
    xlabel        :: String          = ""
    ylabel        :: String          = ""
    yscale        :: Symbol          = :linear      # :linear | :log10
    xscale        :: Symbol          = :linear
    xlim          :: Union{Nothing,Tuple} = nothing
    ylim          :: Union{Nothing,Tuple} = nothing
    legend        :: Bool            = true
    method_styles :: Dict{String, MethodStyle} = Dict()  # per-method style overrides
    extra_kwargs  :: Dict            = Dict()       # forwarded to Makie
end
```

### FigureLayout — Composing Multiple Plots

Plots are arranged in a `rows × cols` matrix. Any cell can be `nothing` (left blank).
This allows completely arbitrary layouts: 1×1, 2×3, irregular L-shapes, etc.

```julia
@kwdef struct FigureLayout
    plots        :: Matrix{Union{PlotSpec,Nothing}}   # [row, col] indexing
    figure_size  :: Tuple{Int,Int} = (1200, 900)      # pixels
    title        :: String = ""
    padding      :: Int = 20
end
```

```julia
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
        _render_lines!(ax, spec)   # uses method_styles and color registry internally
        !isnothing(spec.xlim) && xlims!(ax, spec.xlim...)
        !isnothing(spec.ylim) && ylims!(ax, spec.ylim...)
    end
    isempty(layout.title) || Label(fig[0, :], layout.title, fontsize=18)
    return fig
end

function save_figure(fig::Makie.Figure, path::String)
    # Extension determines format: .pdf, .png, .svg, .jpg
    save(path, fig)
end
```

### End-to-End Plotting Example

```julia
result = load_experiment("logs/20260417/001/")

# Shared data preparation
df_all  = to_dataframe(result) |> df -> aggregate_runs(df, :median)
df_fast = @subset(df_all, :iter .<= 200)
df_gd   = @subset(df_all, :method_name .== "GradientDescent")
df_exp  = @subset(df_all, startswith.(:method_name, "MyMethod"))

# Per-method style overrides — color, linestyle, width are user-controlled per plot
styles = Dict(
    "GradientDescent"                             => MethodStyle(color="#999999", linestyle=:dash),
    "MyMethod[hessian=BFGS,minor=None,linesearch=Wolfe]" => MethodStyle(color="#0072B2", linewidth=2.5),
)

layout = FigureLayout(
    figure_size = (1600, 1000),
    title       = "Experiment 001 — Lasso λ=0.05",
    plots       = [
        # Row 1: full convergence | early iters zoom
        PlotSpec(data=df_all,  x=:iter, y=:objective,     yscale=:log10,
                 title="All methods — full run",   method_styles=styles)   PlotSpec(data=df_fast, x=:iter, y=:objective, yscale=:log10,
                 title="First 200 iters",          method_styles=styles);
        # Row 2: gradient norm | core time per step
        PlotSpec(data=df_exp,  x=:iter, y=:gradient_norm, yscale=:log10,
                 title="Gradient norm (experimental)")                      PlotSpec(data=df_all,  x=:iter, y=:core_time_ns,
                 title="Core step time (ns)");
        # Row 3: single wide plot spanning both columns
        PlotSpec(data=df_all,  x=:core_time_ns, y=:objective, yscale=:log10,
                 xlabel="Cumulative core time (ns)",
                 title="Obj vs. cumulative core time")                      nothing
    ],
)

fig = render_figure(layout)
save_figure(fig, "logs/20260417/001/convergence_overview.pdf")
save_figure(fig, "logs/20260417/001/convergence_overview.png")
```

---

## 13. Directory & Module Structure

The source tree is consolidated into **8 files** instead of 15+. Each file groups
tightly related concerns; none is so large as to become unwieldy.

```
TestEngine.jl/
├── src/
│   ├── TestEngine.jl     # Module entry; includes all src files; exports public API
│   │
│   ├── core.jl           # Abstract types & type hierarchy; state groups (IterateGroup,
│   │                     #   MetricsGroup, TimingGroup); algorithm interface (init_state,
│   │                     #   step!, extract_log_entry); @core_timed macro; generic runner
│   │                     #   (run_method); nested infrastructure (SubRunConfig, SubResult,
│   │                     #   run_sub_method)
│   │
│   ├── stopping.jl       # StoppingCriterion hierarchy; should_stop dispatch;
│   │                     #   CompositeCriterion; stop_when_any / stop_when_all
│   │
│   ├── variants.jl       # Component abstract types & implementations (HessianApprox,
│   │                     #   MinorUpdate, LineSearch); VariantAxis, VariantGrid,
│   │                     #   VariantSpec; expand(); ABBREVIATIONS; build_names()
│   │
│   ├── experiment.jl     # ExperimentConfig; ExperimentResult / RunResult / MethodResult;
│   │                     #   resolve_methods(); run_experiment(); next_experiment_path()
│   │
│   ├── logging.jl        # IterationLog; Logger; log_init!, log_iter!, log_event!,
│   │                     #   attach_sub_logs!, finalize!; elapsed_core_s, elapsed_wall_s;
│   │                     #   VerbosityLevel, VerbosityConfig, maybe_print()
│   │
│   ├── persistence.jl    # save_experiment(); load_experiment(); load_manifest();
│   │                     #   list_experiments(); CSV sidecar writer
│   │
│   ├── problems.jl       # Problem interface (DataFidelity, Regularizer, Problem,
│   │                     #   objective); concrete types (LeastSquares, L1Norm, …);
│   │                     #   ProblemSpec hierarchy (AnalyticProblem, FileProblem,
│   │                     #   RandomProblem); make_problem(); register_problem!;
│   │                     #   register_random_problem!; built-in generators
│   │
│   └── analysis.jl       # to_dataframe(); filter_methods(); aggregate_runs();
│                         #   MethodStyle; METHOD_PALETTE; METHOD_COLOR_REGISTRY;
│                         #   get_method_color(); register_method_color!;
│                         #   PlotSpec; FigureLayout; render_figure(); save_figure()
│
├── algorithms/
│   ├── conventional/
│   │   ├── gradient_descent.jl   # struct + init_state + step! + extract_log_entry
│   │   └── conjugate_gradient.jl
│   └── experimental/
│       ├── mymethod.jl           # MyMethod + MyMethodState + MyMethodNumerics + step!
│       ├── my_outer_method.jl    # Algorithm using run_sub_method
│       └── components/
│           ├── hessian.jl        # approximate_hessian() per HessianApprox subtype
│           ├── minor_update.jl   # apply_minor_update!() per MinorUpdate subtype
│           └── linesearch.jl     # search_step() per LineSearch subtype
│
├── experiments/
│   ├── exp_baseline.jl           # Conventional methods; sanity-check run
│   └── exp_grid_sweep.jl         # VariantGrid definition and full run
│
├── logs/                         # Git-ignored; written at runtime
│   └── 20260417/
│       ├── 001/
│       │   ├── manifest.json
│       │   ├── result.jld2
│       │   └── run1_GradientDescent.csv
│       └── 002/
│
└── test/
    ├── test_core.jl              # runner, state groups, @core_timed, timing accumulation
    ├── test_stopping.jl          # all StoppingCriterion subtypes, should_stop, TimeLimit
    ├── test_variants.jl          # expand(), naming, filters, abbreviations
    ├── test_problems.jl          # Problem interface, make_problem, seed propagation
    └── test_analysis.jl          # to_dataframe, aggregate_runs, color registry, PlotSpec
```

---

## 14. Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│  DEFINITION PHASE                                                       │
│                                                                         │
│  problems.jl        DataFidelity / Regularizer / Problem                │
│       │             AnalyticProblem / FileProblem / RandomProblem       │
│       │                                                                 │
│  variants.jl        VariantAxis(:hessian, BFGS=>"BFGS", ...)           │
│       │             VariantAxis(:minor,   Mom=>"Mom",   ...)            │
│       │                   │                                             │
│       └──────────► VariantGrid → expand() → [VariantSpec, ...]         │
│                                     │                                   │
│  stopping.jl        StoppingCriteria (per-experiment or per-method)    │
│                                     │                                   │
│                         ExperimentConfig                                │
└─────────────────────────────────────┬───────────────────────────────────┘
                                      │
┌─────────────────────────────────────▼───────────────────────────────────┐
│  EXECUTION PHASE                                                        │
│                                                                         │
│         run_experiment(config, log_root)                                │
│                  │                                                      │
│         next_experiment_path()  →  logs/YYYYMMDD/NNN/                  │
│         resolve_methods()       →  calls expand() on each VariantGrid  │
│                  │                                                      │
│         rng = MersenneTwister(seed + run_id - 1)                       │
│         problem = make_problem(spec, rng)                               │
│                  │                                                      │
│        for each run × method:                                           │
│                  │                                                      │
│      ┌───────────▼────────────┐                                         │
│      │    run_method()        │  ◄── Logger + Criteria injected here    │
│      │  init_state(…, rng)    │                                         │
│      │  state._logger = lg    │                                         │
│      │  while true:           │                                         │
│      │    state.timing.       │                                         │
│      │      core_time_ns = 0  │                                         │
│      │    step!()             │  ◄── @core_timed inside step!           │
│      │    extract_log_entry() │  ──► entry.core_time_ns copied          │
│      │    log_iter!()         │  ──► logger.total_core_ns accumulated   │
│      │    should_stop()       │  ◄── TimeLimit reads elapsed_core_s()   │
│      └───────────┬────────────┘                                         │
│                  │  (if nested algorithm used)                          │
│      ┌───────────▼────────────┐                                         │
│      │  run_sub_method()      │  ◄── SubRunConfig                       │
│      │  own TimingGroup       │  ──► independent core time tracking     │
│      │  sub logs attached to  │  ──► outer IterationLog.extras          │
│      └───────────┬────────────┘                                         │
│                  │                                                      │
│           finalize!() → MethodResult                                    │
│           collected into RunResult → ExperimentResult                   │
└─────────────────────────────────────┬───────────────────────────────────┘
                                      │
┌─────────────────────────────────────▼───────────────────────────────────┐
│  PERSISTENCE PHASE                                                      │
│                                                                         │
│   save_experiment()                                                     │
│       ├── result.jld2              (full binary, fast reload)           │
│       ├── run{N}_{method}.csv      (per-method, human-readable)         │
│       └── manifest.json           (name, metadata, no binary needed)   │
└─────────────────────────────────────┬───────────────────────────────────┘
                                      │
┌─────────────────────────────────────▼───────────────────────────────────┐
│  ANALYSIS PHASE                                                         │
│                                                                         │
│   load_experiment()  ──►  to_dataframe()                                │
│                                │                                        │
│                      filter_methods()     (select methods)              │
│                                │                                        │
│                      aggregate_runs()     (mean/median)                 │
│                                │                                        │
│                      user transforms     (DataFrame -> DataFrame)       │
│                                │                                        │
│                       METHOD_COLOR_REGISTRY + MethodStyle               │
│                       PlotSpec (x, y, group_by, method_styles, …)      │
│                       FigureLayout (Matrix of PlotSpec)                 │
│                                │                                        │
│                      render_figure()  ──►  save_figure(.pdf / .png)    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 15. Key Architectural Decisions

| Decision | Rationale |
|----------|-----------|
| `StoppingCriteria` hierarchy replaces `converged` + `for` loop | Full control over termination: count, time, tolerance, composites, all independently testable |
| Stopping criteria separated from algorithm struct | Same algorithm, different run budgets across experiments; no code changes required |
| `@core_timed` in algorithm code, accumulates into `state.timing.core_time_ns` | Scientific discipline: only the kernel is measured; bookkeeping is invisible to the clock |
| `log_iter!` accumulates `entry.core_time_ns` into `logger.total_core_ns` | `TimeLimit` reads `elapsed_core_s(logger)` — per-iteration core time is logged, then summed; wall-clock never used as a stopping criterion |
| Four canonical state groups (`IterateGroup`, `MetricsGroup`, `TimingGroup`, method-specific `Numerics`) | Clean separation of concerns inside state; sub-routines can receive and own independent groups; `extract_log_entry` default is trivial |
| `SubRunConfig` + `run_sub_method` for nesting | Clean, logger-aware sub-invocation; independent `TimingGroup` per sub-state; sub-logs attach to parent iter entry; core times tracked separately |
| `Problem` interface: `DataFidelity` + `Vector{Regularizer}` | Uniform interface for f+g₁+…+gₙ objectives; separates kernel, gradient, Hessian-vec for f; prox for each g; algorithms never know if data was loaded or generated |
| `RandomProblem` with no seed field; single `ExperimentConfig.seed` | One seed controls data generation, `x0`, and stochastic steps — reproducibility is a property of the experiment, not the problem spec |
| `make_problem(spec, rng)` signature for all spec types | `FileProblem` ignores `rng`; `RandomProblem` and `AnalyticProblem` use it; uniform call site in the orchestration loop |
| `MethodStyle` + `METHOD_COLOR_REGISTRY` for plotting | Colors are stable-hash-deterministic and visually appealing by default; user overrides are explicit per-plot, not global mutations; no random color assignment |
| `expand(grid)` and `resolve_methods(config)` are distinct functions | `expand` is a pure, testable primitive on one grid; `resolve_methods` is the config-level aggregator; they must not share a name |
| `RunResult`, `MethodResult`, `ExperimentResult` fully defined | Typed result hierarchy enables pattern-matched dispatch, serialization, and the analysis pipeline without `Any`-typed fields in the hot path |
| `aggregate_runs` modes `:all`, `:mean`, `:median` | `:all` preserves every run for full distribution; `:mean`/`:median` reduce to a single representative curve; `:best` omitted — cherry-picking runs has no sound benchmarking interpretation |
| 8 source files instead of 15+ | Cohesive groupings (e.g. logging + verbosity, all problems in one file) reduce include-order friction and make the module boundary obvious without fragmenting related code |
| `FigureLayout` as `Matrix{Union{PlotSpec,Nothing}}` | Any grid formation expressible as a Julia matrix literal; blank cells are `nothing`; arbitrary sizes |
| `save_figure` format determined by file extension | Single call saves PDF or image; no API change needed |
| `logs/YYYYMMDD/NNN/` two-level naming | Date folders group experiments naturally; per-day counter resets cleanly; human name lives in manifest |
| Logger injected by runner, not called by algorithm | Algorithm code stays pure; logging strategy controlled externally |
| Transforms as `DataFrame -> DataFrame` | No DSL to learn; composable with DataFramesMeta; independently unit-testable |

---

## 16. Extension Guide

### Adding a new conventional baseline

Create `algorithms/conventional/my_baseline.jl`. Define the struct, implement
`init_state` (using `IterateGroup`, `MetricsGroup`, `TimingGroup`), `step!` (use
`@core_timed state begin ... end` around the kernel), and `extract_log_entry`.
Add it to an `ExperimentConfig`. The runner, logger, stopping criteria, and plots
all pick it up automatically.

### Adding a new stopping criterion

Add a struct subtyping `StoppingCriterion` and a `should_stop` method to `stopping.jl`.
It can immediately be used standalone or composed inside `CompositeCriterion`. Access
state quantities via `state.metrics.*`.

### Adding an algorithm that uses a sub-algorithm

Embed a `SubRunConfig` field in the outer algorithm struct. Call `run_sub_method`
inside `step!`, passing `state._logger` as the outer logger. The sub-iteration logs
appear automatically in the outer `IterationLog.extras[:sub_logs]` and in the CSV
sidecar. The sub-state has its own independent `TimingGroup`.

### Adding a new problem type

For a random problem, call `register_random_problem!(:my_problem, (rng, p) -> ...)`
in `problems.jl`. The generator receives the experiment-level `rng` and must build
the full `Problem` (data fidelity, regularizers, and `x0`) from it. For a file-based
problem, construct a `FileProblem` inline at experiment definition time.

### Adding a new logged field

Add the field to `IterationLog` or to `extras` in `extract_log_entry`. The CSV
sidecar picks up all `extras` keys automatically via `to_dataframe()`.

### Adding a new Hessian variant

Add one line to the relevant `VariantAxis`:

```julia
VariantAxis(:hessian,
    ...
    MyNewHessian(param=3.0) => "NewH",   # ← one new line
)
```

Implement `approximate_hessian(::MyNewHessian, state)` in `hessian.jl`.
Naming, filenames, CSV columns, and plot labels all update automatically.

### Adding a fixed color for a method across all plots

```julia
register_method_color!("GradientDescent", "#999999")
register_method_color!("MyMethod[hessian=BFGS,minor=None,linesearch=Wolfe]", "#0072B2")
```

Call this once at session startup (e.g. in `experiments/exp_baseline.jl`). All
subsequent `render_figure` calls respect the registry. Per-plot `method_styles` in
`PlotSpec` take precedence over the registry when both are present.

### Plotting across multiple experiments

`to_dataframe` accepts a single `ExperimentResult`. To compare across experiments,
concatenate the DataFrames with a tag column:

```julia
df1 = to_dataframe(load_experiment("logs/20260417/001/")) |> d -> @transform(d, :exp = "exp1")
df2 = to_dataframe(load_experiment("logs/20260417/002/")) |> d -> @transform(d, :exp = "exp2")
df  = vcat(df1, df2)
# Then build PlotSpec(data=df, group_by=:exp, ...) as normal
```
