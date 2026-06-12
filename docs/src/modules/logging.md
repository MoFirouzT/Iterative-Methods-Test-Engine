# Logging & Verbosity

The logger is external to all algorithms. It is injected by the runner and captures
data through three hooks: `log_init!`, `log_iter!`, and `log_event!`. Verbosity is
co-located with logging because both share the same `Logger` struct.

## IterationLog

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

## Logger

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

## `log_iter!`

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

## Verbosity Levels

Verbosity is a first-class, orthogonal concern — not scattered `if verbose` checks.
It is **independent of debug mode** (see [Debug Mode](@ref)): verbosity controls what is printed
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

## VerbosityConfig

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

## Range-Gated Output

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

