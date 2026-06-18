# Debug Mode

The debug mode is an optional, experiment-level diagnostic layer.
When activated, the runner performs additional computations after each step — computations that may be expensive (such as numerical gradient checks) and are never run in normal operation.
When a check's condition triggers, a configurable action is taken:
a warning is printed, the error is raised, or the event is recorded silently.

Debug mode is **orthogonal to verbosity** (see [Logging & Verbosity](@ref)):
verbosity controls what is printed from normal iteration data;
debug mode controls diagnostic calculations triggered by threshold violations.
Both can be active simultaneously at independent levels.

## DebugConfig

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

## DebugCheck Hierarchy

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

## `run_debug_checks!` and `debug_check!` Dispatch

The runner calls `run_debug_checks!` after `log_iter!` on every iteration and passes the logger explicitly so history-based checks do not need state mutation:

```julia
function run_debug_checks!(cfg        :: DebugConfig,
                           logger     :: Union{Nothing, Logger},
                           state, problem,
                           entry      :: IterationLog,
                           prev_entry :: Union{Nothing, IterationLog},
                           iter       :: Int)
    cfg.enabled || return
    for check in cfg.checks
        debug_check!(check, cfg, logger, state, problem, entry, prev_entry, iter)
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

## `trigger_debug!` and Diagnostic Helpers

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

Note on silent logging (`:log`):
when `cfg.on_trigger == :log`, callers may pass a `logger` instance to `trigger_debug!` (or ensure `state` carries a reference to the current logger) so the event can be recorded in `logger.events` rather than printed.
This allows debug checks to be recorded without altering console output;
the runner or caller should supply the logger when invoking debug helpers in order to enable this mode.

## Integration Example

```julia
config = ExperimentConfig(
    name         = "Debug run — gradient check active",
    problem_spec = AnalyticProblem(name=:rosenbrock, params=(rho=100.0,)),
    baseline_methods = [GradientDescent(step_size=ArmijoLS())],
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
