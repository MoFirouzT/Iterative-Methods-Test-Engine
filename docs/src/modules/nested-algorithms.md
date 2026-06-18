# Nested Algorithm Infrastructure

Some algorithms run another iterative method as a **sub-routine** inside their own `step!`.
Examples: trust-region methods that solve an inner subproblem iteratively, bi-level methods, inner-loop methods that refine a correction, or meta-algorithms that call multiple sub-solvers per outer step.

This module provides the infrastructure to make nested invocation clean, safe, and fully logged.
The shipped consumer is **`TrustRegion`**, whose Steihaug truncated-CG inner solve runs as a genuine sub-method on a genuine `Problem` via `run_sub_method` —
it folds the inner solve's core time into the outer step and attaches the full inner CG trace to each outer log entry.
The generic schematic below is the pattern `TrustRegion` follows;
see [trust_region.md](https://github.com/MoFirouzT/Iterative-Methods-Test-Engine/blob/main/algorithms/trust_region/trust_region.md).

## Design Principle

An algorithm struct holds a **sub-method slot** typed concretely via a parametric
`SubRunConfig{M}`.
During `step!`, the algorithm calls `run_sub_method(...)`, passing the logger and rng it received from the outer runner.
`run_sub_method` derives a child RNG from the outer rng — deterministically — ensuring sub-runs are fully reproducible.
Sub-iteration logs are attached to the current outer iteration's log entry under `extras`.
The sub-runner never writes to disk or console independently.

## Infrastructure Types

```julia
# Parametric over the inner method type — enables type-stable state inference
@kwdef struct SubRunConfig{M <: IterativeMethod}
    method        :: M
    criteria      :: StoppingCriterion
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

## `run_sub_method`

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

## Schematic Use in an Outer Algorithm

```julia
# Sketch — the outer algorithm holds a sub-config, calls run_sub_method inside step!,
# and reads the concrete final_state to extract whatever it needs.

@kwdef struct SomeOuterMethod <: IterativeMethod
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
