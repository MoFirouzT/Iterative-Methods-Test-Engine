"""
    Layer 6 — Logging System

Provides per-iteration capture, core-time accumulation, event logging, and sub-logs.
The logger is external to all algorithms and injected by the runner.

Core logging functions:
- log_init!  : Initialize logger before the run loop
- log_iter!  : Record one iteration's results
- log_event! : Record a named event (convergence, stopping reason, warnings)
- finalize!  : Finalize the logger and return a MethodResult
"""

using Dates


# ─────────────────────────────────────────────────────────────────────────
# Layer 7 — Verbosity System
# ─────────────────────────────────────────────────────────────────────────

@enum VerbosityLevel begin
    SILENT = 0
    MILESTONE = 1
    SUMMARY = 2
    DETAILED = 3
    DEBUG = 4
end


"""
    VerbosityConfig

Controls console output from the logger.
"""
@kwdef mutable struct VerbosityConfig
    level::VerbosityLevel = SUMMARY
    print_every::Int = 10
    fields::Vector{Symbol} = [:iter, :objective, :gradient_norm]
    color::Bool = true
    io::IO = stdout
    iter_range::Union{Nothing,UnitRange{Int}} = nothing
end


# ─────────────────────────────────────────────────────────────────────────
# IterationLog — Per-iteration record
# ─────────────────────────────────────────────────────────────────────────

"""
    IterationLog

Record of a single iteration. Mirrors the canonical MetricsGroup structure
so that extract_log_entry has a trivial default implementation.

# Fields
- `iter::Int` — iteration number
- `core_time_ns::Int64` — nanoseconds of core computation in this step
- `objective::Float64` — objective function value at current iterate
- `gradient_norm::Float64` — gradient norm (or other convergence metric)
- `step_norm::Float64` — norm of the step taken in this iteration
- `dist_to_opt::Float64` — distance to known optimum (Inf if unavailable)
- `extras::Dict{Symbol,Any}` — algorithm-specific fields and nested sub-logs
"""
@kwdef mutable struct IterationLog
    iter           :: Int
    core_time_ns   :: Int64            # nanoseconds of core computation this step
    objective      :: Float64
    gradient_norm  :: Float64
    step_norm      :: Float64
    dist_to_opt    :: Float64 = Inf    # ‖x − x*‖; Inf when x_opt not provided
    extras         :: Dict{Symbol,Any} = Dict()  # algorithm-specific & sub-logs
end


# ─────────────────────────────────────────────────────────────────────────
# Logger — External logging controller
# ─────────────────────────────────────────────────────────────────────────

"""
    Logger

External logging system injected by the runner. Records iteration logs,
events, and accumulated timing information. Never called directly by algorithms.

# Fields
- `method_name::String` — name of the algorithm being run
- `run_id::Int` — which run within the experiment (1, 2, ...)
- `exp_path::String` — path to experiment directory
- `verbosity_config::VerbosityConfig` — controls console output (see Layer 7)
- `iter_logs::Vector{IterationLog}` — all iteration records
- `events::Vector{NamedTuple}` — named events (convergence, stop reasons, warnings)
- `metadata::Dict{Symbol,Any}` — custom metadata
- `start_wall_time::Float64` — wall-clock time at log_init! (informational)
- `total_core_ns::Int64` — accumulated core nanoseconds across all iterations
- `pending_sub_logs::Vector{IterationLog}` — buffer for attach_sub_logs!
"""
mutable struct Logger
    method_name      :: String
    run_id           :: Int
    exp_path         :: String
    verbosity_config :: Any  # VerbosityConfig (forward ref to Layer 7)
    iter_logs        :: Vector{IterationLog}
    events           :: Vector{NamedTuple}
    metadata         :: Dict{Symbol,Any}
    start_wall_time  :: Float64
    total_core_ns    :: Int64
    pending_sub_logs :: Vector{IterationLog}
end


_verbosity_config(logger::Logger) = logger.verbosity_config isa VerbosityConfig ? logger.verbosity_config : nothing


function _entry_field(entry::IterationLog, field::Symbol)
    if field === :iter
        return entry.iter
    elseif field === :core_time_ns
        return entry.core_time_ns
    elseif field === :objective
        return entry.objective
    elseif field === :gradient_norm
        return entry.gradient_norm
    elseif field === :step_norm
        return entry.step_norm
    else
        return get(entry.extras, field, missing)
    end
end


function _fmt_field(value)
    if value isa AbstractFloat
        return string(round(value; sigdigits = 6))
    end
    return string(value)
end


function format_and_print(cfg::VerbosityConfig, logger::Logger, entry::IterationLog, level::VerbosityLevel)
    parts = String[string(field, "=", _fmt_field(_entry_field(entry, field))) for field in cfg.fields]
    line = string("[", logger.method_name, "|run ", logger.run_id, "] ", join(parts, " "))
    if level == DEBUG
        line = string(line, " extras=", entry.extras)
    end
    println(cfg.io, line)
end


function _print_milestone(logger::Logger, message::String)
    cfg = _verbosity_config(logger)
    isnothing(cfg) && return
    if cfg.level >= MILESTONE
        println(cfg.io, "[", logger.method_name, "|run ", logger.run_id, "] ", message)
    end
end


# ─────────────────────────────────────────────────────────────────────────
# Helper Functions — Timing Queries
# ─────────────────────────────────────────────────────────────────────────

"""
    elapsed_core_s(logger::Logger) :: Float64

Return accumulated core computation time in seconds.
This is the **authoritative** timing used by TimeLimit stopping criteria.
Wall-clock time is never used as a stopping criterion.
"""
function elapsed_core_s(logger::Logger)
    logger.total_core_ns / 1e9
end


"""
    elapsed_wall_s(logger::Logger) :: Float64

Return wall-clock elapsed time since log_init! in seconds.
Informational only — never used for stopping decisions.
"""
function elapsed_wall_s(logger::Logger)
    time() - logger.start_wall_time
end


# ─────────────────────────────────────────────────────────────────────────
# Core Logging Functions
# ─────────────────────────────────────────────────────────────────────────

"""
    log_init!(logger::Logger, method, state)

Initialize logger before the run loop. Called once per method per run.
Records the start wall-time and any initial state information.

# Arguments
- `logger::Logger` — the logger to initialize
- `method::IterativeMethod` — the algorithm being initialized
- `state` — the initial algorithm state
"""
function log_init!(logger::Logger, method, state)
    logger.start_wall_time = time()
    logger.total_core_ns = 0
    logger.iter_logs = IterationLog[]
    logger.events = NamedTuple[]
    logger.pending_sub_logs = IterationLog[]
    _print_milestone(logger, "start")
    # Subclasses or extensions can override to capture method-specific metadata
end


"""
    log_iter!(logger::Logger, entry::IterationLog)

Record one iteration's results. Accumulates core time and may print to console.
Called after each step!, after extract_log_entry, but before should_stop.

# Arguments
- `logger::Logger` — the logger
- `entry::IterationLog` — the log entry from this iteration
"""
function log_iter!(logger::Logger, entry::IterationLog)
    push!(logger.iter_logs, entry)
    logger.total_core_ns += entry.core_time_ns   # feeds elapsed_core_s() → TimeLimit
    maybe_print(logger, entry)  # forward ref to Layer 7
end


"""
    log_event!(logger::Logger, reason::Symbol, iter::Int)

Record a named event: convergence, stopping reason, or warning.
Called after should_stop returns true (never timed).

# Arguments
- `logger::Logger` — the logger
- `reason::Symbol` — the stopping or event reason (e.g., :max_iterations, :gradient_converged)
- `iter::Int` — the iteration number when the event occurred
"""
function log_event!(logger::Logger, reason::Symbol, iter::Int)
    event = (reason=reason, iter=iter, timestamp=now())
    push!(logger.events, event)
    _print_milestone(logger, string("event=", reason, " iter=", iter))
end


"""
    attach_sub_logs!(logger::Logger, sub_logs::Vector{IterationLog})

Attach sub-iteration logs from a nested algorithm to the current pending entry.
Called by run_sub_method when log_sub_iters=true.

# Arguments
- `logger::Logger` — the outer logger
- `sub_logs::Vector{IterationLog}` — logs from the sub-algorithm
"""
function attach_sub_logs!(logger::Logger, sub_logs::Vector{IterationLog})
    # Store in pending buffer; finalize! will attach to the current iteration
    logger.pending_sub_logs = sub_logs
end


"""
    finalize!(logger::Logger, method, state) :: MethodResult

Finalize the logger and return a MethodResult. Called after the run loop exits.

# Returns
A `MethodResult` containing:
- method_name, iter_logs, final_state, stop_reason, n_iters

# Arguments
- `logger::Logger` — the completed logger
- `method::IterativeMethod` — the algorithm
- `state` — the final algorithm state
"""
function finalize!(logger::Logger, method, state)
    # Extract stop reason from the last event, or default to :unknown
    stop_reason = isempty(logger.events) ? :unknown : logger.events[end].reason
    
    # If there are pending sub-logs from the final iteration, attach them
    if !isempty(logger.pending_sub_logs) && !isempty(logger.iter_logs)
        last_entry = logger.iter_logs[end]
        last_entry.extras[:sub_logs] = logger.pending_sub_logs
    end
    
    n_iters = length(logger.iter_logs)
    _print_milestone(logger, string("finalize stop_reason=", stop_reason, " n_iters=", n_iters))
    
    # Return typed MethodResult when Layer 5 is loaded, otherwise a compatible NamedTuple.
    if @isdefined(MethodResult)
        return MethodResult(
            logger.method_name,
            logger.iter_logs,
            state,
            stop_reason,
            n_iters,
        )
    end

    return (
        method_name = logger.method_name,
        iter_logs = logger.iter_logs,
        final_state = state,
        stop_reason = stop_reason,
        n_iters = n_iters,
    )
end


# ─────────────────────────────────────────────────────────────────────────
# Placeholder for Layer 7 (Verbosity) Functions
# ─────────────────────────────────────────────────────────────────────────

"""
    maybe_print(logger::Logger, entry::IterationLog)

Placeholder for verbosity-gated console output (Layer 7).
Will be implemented with VerbosityConfig and range-gated printing.
"""
function maybe_print(logger::Logger, entry::IterationLog)
    cfg = _verbosity_config(logger)
    isnothing(cfg) && return

    effective_level = if !isnothing(cfg.iter_range) && entry.iter in cfg.iter_range
        DETAILED
    elseif !isnothing(cfg.iter_range)
        SILENT
    else
        cfg.level
    end

    effective_level == SILENT && return
    effective_level == MILESTONE && return

    if effective_level >= SUMMARY
        if !(effective_level >= DETAILED || entry.iter % cfg.print_every == 0)
            return
        end
        format_and_print(cfg, logger, entry, effective_level)
    end
end
