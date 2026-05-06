"""
    Module 9 — Debug Mode

Provides `DebugConfig`, the `DebugCheck` hierarchy, `run_debug_checks!`,
`debug_check!` dispatch for several checks, `trigger_debug!`, and a
central-difference numerical gradient helper.
"""

@kwdef struct DebugConfig
    enabled    :: Bool               = false
    checks     :: Vector{Any} = Any[
                      nothing,
                  ]
    on_trigger :: Symbol             = :warn   # :warn | :error | :log
    io         :: IO                 = stderr
end

abstract type DebugCheck end

@kwdef struct CheckObjectiveMonotonicity <: DebugCheck
    tolerance :: Float64 = 1e-10
end

@kwdef struct CheckGradientNormBound <: DebugCheck
    max_norm :: Float64 = 1e8
end

@kwdef struct CheckStepDecay <: DebugCheck
    window :: Int = 20
end

@kwdef struct CheckNumericalGradient <: DebugCheck
    epsilon   :: Float64 = 1e-7
    max_error :: Float64 = 1e-4
end

function run_debug_checks!(cfg::DebugConfig,
                           logger::Union{Nothing,Logger},
                           state, problem,
                           entry::IterationLog,
                           prev_entry::Union{Nothing,IterationLog},
                           iter::Int)
    cfg.enabled || return
    for check in cfg.checks
        debug_check!(check, cfg, logger, state, problem, entry, prev_entry, iter)
    end
end

function debug_check!(c::CheckObjectiveMonotonicity, cfg, logger, state, problem,
                      entry::IterationLog, prev::Union{Nothing,IterationLog}, iter::Int)
    isnothing(prev) && return
    increase = entry.objective - prev.objective
    if increase > c.tolerance
        trigger_debug!(cfg, iter,
            "Objective increased by $(increase) " *
            "(prev=$(prev.objective), curr=$(entry.objective))")
    end
end

function debug_check!(c::CheckGradientNormBound, cfg, logger, state, problem,
                      entry::IterationLog, prev::Union{Nothing,IterationLog}, iter::Int)
    if entry.gradient_norm > c.max_norm
        trigger_debug!(cfg, iter,
            "Gradient norm $(entry.gradient_norm) exceeds bound $(c.max_norm)")
    end
end

function debug_check!(c::CheckStepDecay, cfg, logger, state, problem,
                      entry::IterationLog, prev::Union{Nothing,IterationLog}, iter::Int)
    # Requires access to historical iter_logs; logger must be injected explicitly.
    if logger === nothing || length(logger.iter_logs) < c.window
        return
    end
    idx_first = length(logger.iter_logs) - c.window + 1
    old_step = logger.iter_logs[idx_first].step_norm
    if entry.step_norm >= old_step
        trigger_debug!(cfg, iter,
            "Step norm did not decrease over last $(c.window) iters " *
            "(old=$(old_step), curr=$(entry.step_norm))",
            logger=logger)
    end
end

function debug_check!(c::CheckNumericalGradient, cfg, logger, state, problem,
                      entry::IterationLog, prev::Union{Nothing,IterationLog}, iter::Int)
    # state.iterate.gradient is expected to exist on well-formed states
    g_analytical = getproperty(state.iterate, :gradient)
    g_numerical  = numerical_gradient(problem.f, state.iterate.x, c.epsilon)
    denom        = max(norm(g_analytical), 1.0)
    rel_error    = norm(g_analytical .- g_numerical) / denom
    if rel_error > c.max_error
        trigger_debug!(cfg, iter,
            "Gradient check failed: relative error = $(rel_error) " *
            "(threshold=$(c.max_error))")
    end
end

function trigger_debug!(cfg::DebugConfig, iter::Int, msg::String; logger::Union{Nothing,Logger}=nothing)
    full_msg = "[DEBUG iter=$iter] $msg"
    if cfg.on_trigger in (:warn, :error)
        println(cfg.io, full_msg)
    end
    if cfg.on_trigger == :error
        error(full_msg)
    elseif cfg.on_trigger == :log
        if logger !== nothing
            push!(logger.events, (kind = :debug, iter = iter, msg = full_msg))
        else
            # no logger available — fallback to printing a warning so user notices
            println(cfg.io, full_msg)
        end
    end
end

function numerical_gradient(f::Any, x::Vector{Float64}, ε::Float64)::Vector{Float64}
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
