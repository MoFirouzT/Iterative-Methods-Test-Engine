"""
    Module 3 — Stopping Criteria

Defines the StoppingCriterion type hierarchy and the `should_stop` dispatch interface.
Provides composable criteria: by iteration count, core computation time, tolerance,
or user-defined conditions.

The runner uses a `while true` loop controlled entirely by StoppingCriteria.
This gives full, composable control over how many steps any algorithm takes.
"""

# ─────────────────────────────────────────────────────────────────────────
# Abstract Base Type
# ─────────────────────────────────────────────────────────────────────────

"""
    abstract type StoppingCriterion

Base type for all stopping criteria. Each concrete subtype must implement:
    should_stop(criterion, state, iter::Int, logger::Logger) -> (stop::Bool, reason::Symbol)
"""
abstract type StoppingCriterion end


# ─────────────────────────────────────────────────────────────────────────
# Concrete Stopping Criteria
# ─────────────────────────────────────────────────────────────────────────

"""
    MaxIterations(; n::Int = 1000)

Terminate after a fixed number of iterations.
"""
@kwdef struct MaxIterations <: StoppingCriterion
    n :: Int = 1000
end


"""
    TimeLimit(; seconds::Float64 = 60.0)

Terminate when accumulated **core computation** time exceeds the budget.
Time is measured as the sum of per-iteration core_time_ns values recorded
in the logger — wall-clock and bookkeeping time are never counted.
"""
@kwdef struct TimeLimit <: StoppingCriterion
    seconds :: Float64 = 60.0
end


"""
    GradientTolerance(; tol::Float64 = 1e-6)

Terminate when gradient norm falls below threshold.
"""
@kwdef struct GradientTolerance <: StoppingCriterion
    tol :: Float64 = 1e-6
end


"""
    ObjectiveStagnation(; tol::Float64 = 1e-8, window::Int = 10)

Terminate when objective change over last `window` iterations is below threshold.
"""
@kwdef struct ObjectiveStagnation <: StoppingCriterion
    tol    :: Float64 = 1e-8
    window :: Int     = 10
end


"""
    StepTolerance(; tol::Float64 = 1e-8)

Terminate when step norm falls below threshold.
"""
@kwdef struct StepTolerance <: StoppingCriterion
    tol :: Float64 = 1e-8
end


"""
    CompositeCriterion(; criteria::Vector{StoppingCriterion}, mode::Symbol = :any)

Combine multiple criteria: :any (first satisfied wins) or :all (all must hold).

# Modes
- `:any` — terminate when any criterion is satisfied (default)
- `:all` — terminate when all criteria are satisfied
"""
@kwdef struct CompositeCriterion <: StoppingCriterion
    criteria :: Vector{StoppingCriterion}
    mode     :: Symbol = :any    # :any | :all
end


# ─────────────────────────────────────────────────────────────────────────
# Convenience Constructors
# ─────────────────────────────────────────────────────────────────────────

"""
    stop_when_any(cs...)

Convenience constructor for CompositeCriterion with mode=:any.
Terminates when the first criterion is satisfied.

# Example
```julia
stop_when_any(MaxIterations(1000), GradientTolerance(1e-6))
```
"""
function stop_when_any(cs...)
    CompositeCriterion(criteria=collect(cs), mode=:any)
end


"""
    stop_when_all(cs...)

Convenience constructor for CompositeCriterion with mode=:all.
Terminates when all criteria are satisfied.

# Example
```julia
stop_when_all(MaxIterations(1000), GradientTolerance(1e-6))
```
"""
function stop_when_all(cs...)
    CompositeCriterion(criteria=collect(cs), mode=:all)
end


# ─────────────────────────────────────────────────────────────────────────
# The `should_stop` Interface
# ─────────────────────────────────────────────────────────────────────────

"""
    should_stop(criterion::StoppingCriterion, state, iter::Int, logger::Logger) -> (stop::Bool, reason::Symbol)

Dispatch point for all stopping criteria. Returns a tuple (stop, reason) where:
- `stop::Bool` — whether the algorithm should terminate
- `reason::Symbol` — the reason for stopping, or `:none` if not stopping

Each concrete StoppingCriterion subtype must implement this function.
"""
function should_stop end


# MaxIterations dispatch
function should_stop(c::MaxIterations, state, iter::Int, logger)
    iter >= c.n ? (true, :max_iterations) : (false, :none)
end


# TimeLimit dispatch
# Uses accumulated core computation time — never wall-clock
function should_stop(c::TimeLimit, state, iter::Int, logger)
    elapsed_core_s(logger) >= c.seconds ? (true, :time_limit) : (false, :none)
end


# GradientTolerance dispatch
function should_stop(c::GradientTolerance, state, iter::Int, logger)
    state.metrics.gradient_norm <= c.tol ? (true, :gradient_converged) : (false, :none)
end


# StepTolerance dispatch
function should_stop(c::StepTolerance, state, iter::Int, logger)
    state.metrics.step_norm <= c.tol ? (true, :step_converged) : (false, :none)
end


# ObjectiveStagnation dispatch
function should_stop(c::ObjectiveStagnation, state, iter::Int, logger)
    iter < c.window && return (false, :none)
    recent = logger.iter_logs[end-c.window+1:end]
    Δ = abs(recent[1].objective - recent[end].objective)
    Δ <= c.tol ? (true, :objective_stagnated) : (false, :none)
end


# CompositeCriterion dispatch
function should_stop(c::CompositeCriterion, state, iter::Int, logger)
    results = [should_stop(sub, state, iter, logger) for sub in c.criteria]
    if c.mode == :any
        idx = findfirst(r -> r[1], results)
        isnothing(idx) ? (false, :none) : results[idx]
    else  # :all
        all(r -> r[1], results) ? (true, :all_criteria_met) : (false, :none)
    end
end
