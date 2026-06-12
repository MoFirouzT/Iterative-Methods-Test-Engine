# Stopping Criteria

The runner uses a `while true` loop controlled entirely by `StoppingCriteria`.
This gives full, composable control over how many steps any algorithm takes:
by count, time, proximity to solution, or any user-defined condition.

## Type Hierarchy

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

## The `should_stop` Interface

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

## Usage at Experiment Definition

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

