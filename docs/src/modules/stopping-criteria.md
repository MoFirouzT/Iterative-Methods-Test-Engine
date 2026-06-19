# Stopping Criteria

The runner uses a `while true` loop controlled entirely by `StoppingCriterion`.
This gives full, composable control over how many steps any algorithm takes:
by count, time, proximity to solution, or any user-defined condition.

## Type Hierarchy

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

# Terminate when ‖x − x*‖ falls below threshold.
# Only fires when problem.x_opt is set; always returns (false, :none) otherwise.
@kwdef struct DistanceToOptimal <: StoppingCriterion
    tol :: Float64 = 1e-6
end

# Combine multiple criteria: :any (first satisfied wins) or :all (all must hold)
@kwdef struct CompositeCriterion <: StoppingCriterion
    criteria :: Vector{StoppingCriterion}
    mode     :: Symbol = :any
end

# Convenience constructors
stop_when_any(cs...) = CompositeCriterion(criteria=collect(cs), mode=:any)
stop_when_all(cs...) = CompositeCriterion(criteria=collect(cs), mode=:all)
```

> **Choosing a *valid* convergence test.**
> These split into **budgets** (`MaxIterations`, `TimeLimit` — they bound work, not optimality, and are always safe to include) and **convergence tests** (the rest — sound only for the problem class whose optimality condition they actually measure).
> In particular `GradientTolerance` reads `‖∇f‖`, the *smooth-part* gradient, so it certifies convergence only for **smooth, unconstrained** problems;
> on a composite `f + g` the smooth gradient need not vanish at the optimum, so use `StepTolerance` (the gradient-mapping proxy) or `DistanceToOptimal` instead.
> The full criterion-by-problem-class matrix is in [Convergence & Cost](../convergence-and-cost.md).

## The `should_stop` Interface

```julia
# Returns (stop::Bool, reason::Symbol)
function should_stop(c::StoppingCriterion, state, iter::Int, logger::Logger) end

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

# Short-circuits without materializing a per-criterion result vector — this is
# the default `stopping_criteria` and runs every iteration, so it stays allocation-free.
function should_stop(c::CompositeCriterion, state, iter, logger)
    if c.mode == :any
        for sub in c.criteria
            stop, reason = should_stop(sub, state, iter, logger)
            stop && return (true, reason)
        end
        return (false, :none)
    else  # :all — an empty criteria list is never "all met"
        isempty(c.criteria) && return (false, :none)
        for sub in c.criteria
            should_stop(sub, state, iter, logger)[1] || return (false, :none)
        end
        return (true, :all_criteria_met)
    end
end
```

## Extending — method-specific criteria

The criteria above are the engine's generic set.
Because `StoppingCriterion` is a plain dispatch point, **content can define its own criteria** by subtyping it and adding a `should_stop` method — exactly how a method defines `step!`.
These stay out of the engine.
For example, `TrustRegion`'s inner truncated-CG solve ships `NegativeCurvature` and `TrustRegionBoundary` (in `algorithms/trust_region/`), which read a method-specific inner-status accessor; being trust-region-specific, they live with the method, not here.

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
