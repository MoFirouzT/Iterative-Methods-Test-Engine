# Experiment Orchestration

## Experiment Naming

Each experiment is stored under a two-level path:
a **date folder** and a **zero-padded sequential counter** that resets each day.

```text
logs/
└── 20260417/          ← date folder (YYYYMMDD)
    ├── 001/           ← first experiment of this day
    │   ├── manifest.json
    │   ├── result.jld2
    │   └── ...
    ├── 002/
    └── 003/
```

The counter is determined at save time by atomically creating the directory —
the `mkdir` call fails if the path already exists, avoiding the TOCTOU race condition inherent in a scan-then-create approach:

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

The human-readable `name` field from `ExperimentConfig` is stored inside `manifest.json`, not in the path.

## Warm-up Infrastructure

A warm-up is an optional, **shared** pre-run initialization step.
It executes once per run before any method starts, and its output — a new initial point `x0_warm` — replaces `problem.x0` for all methods in that run.
Methods cannot distinguish between a warm-up start and a cold start;
the problem interface is identical.

```julia
abstract type WarmupStrategy end

# No warm-up — use problem.x0 as-is (default)
struct NoWarmup <: WarmupStrategy end

# Run an iterative method as warm-up; use its final iterate as x0
@kwdef struct IterativeWarmup <: WarmupStrategy
    method    :: IterativeMethod
    criteria  :: StoppingCriterion
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

## ExperimentConfig

```julia
@kwdef struct ExperimentConfig
    name                 :: String
    problem_spec         :: ProblemSpec
    baseline_methods     :: Vector{IterativeMethod}    = []
    experimental_methods :: Vector{IterativeMethod}    = []
    variant_grids        :: Vector{VariantGrid}        = []
    stopping_criteria    :: StoppingCriterion           = stop_when_any(
                                MaxIterations(1000), GradientTolerance(1e-6))
    method_criteria      :: Dict{String, StoppingCriterion} = Dict()
    warmup               :: WarmupStrategy             = NoWarmup()
    n_runs               :: Int                        = 1
    seed                 :: Union{Int,Nothing}         = 42
    tags                 :: Dict{String,Any}           = Dict()
    debug                :: DebugConfig                = DebugConfig()
    count_oracles        :: Bool                       = false
    persist              :: PersistPolicy              = PersistPolicy()
end
```

`method_criteria` lets specific methods use different stopping budgets within the same experiment.

`count_oracles` (default `false`) turns on **oracle counting**:
the runner wraps each method's `problem.f` in a `CountingObjective` (with a fresh `OracleCounts`) before the run, and surfaces the cumulative `:n_value` / `:n_grad` / `:n_hvp` counts in every `IterationLog`'s `extras`.
It is opt-in precisely so the default path — and the core-time measurement it produces — is untouched;
the wrapper is installed per method, *after* warm-up, so warm-up evaluations are excluded from the measured run.
See [oracle counting](problem-interface.md#Oracle-counting-(opt-in-instrumentation)) for the wrapper and [Convergence & Cost](../convergence-and-cost.md) for using the counts.

## Result Types

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

`finalize!(logger, method, state)` returns a `MethodResult{typeof(state)}`, preserving the concrete state type through the parametric wrapper.

`iter_logs` begins with an `iter=0` snapshot recorded by `log_init!` (so trajectory plots and warm-up x₀ invariants can see the starting point), followed by one entry per iteration.
`n_iters` counts only the actual iterations — entries with `iter > 0`, excluding that init snapshot.
(`run_sub_method` counts the same way via its loop counter.)

## Result Hierarchy (Overview)

```text
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

## Orchestration Loop

RNG streams are separated by concern —
data generation, warm-up, and per-method computation each draw from independent, deterministically-derived Xoshiro streams.
Per-method RNGs are additionally keyed by method name so that adding or removing a method from the config does not alter other methods' streams.

**Per-role deterministic RNG derivation:**
The framework derives all active RNGs from a single root seed to guarantee reproducibility regardless of method ordering or composition.
Concretely:

- `root_seed` is `config.seed` when provided, otherwise a freshly sampled `UInt64`.
- `rng_problem = Xoshiro(hash((root_seed, run_id, :data)))` — RNG used to build the problem instance.
- `method_rng  = Xoshiro(hash((root_seed, run_id, method_name)))` — RNG passed to each method's run.
- For nested sub-runs, `sub_rng = Xoshiro(rand(outer_rng, UInt64))` is derived deterministically from the outer RNG.

This guarantees that adding or removing a method (or re-ordering methods) does not shift the RNG streams of other methods.

```julia
function run_experiment(config    :: ExperimentConfig,
                        log_root  :: String = "logs";
                        verbosity :: VerbosityConfig = VerbosityConfig())

    exp_path = next_experiment_path(log_root)
    mkpath(exp_path)

    baseline, experimental = resolve_methods(config)
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
        for (name, method) in [baseline; experimental]
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

## `resolve_methods`

`resolve_methods(config)` flattens the three method sources —
`baseline_methods`, `experimental_methods`, and the `expand`-ed output of every entry in `variant_grids` — into two buckets.
The bucket is decided by **role metadata**, never by the method's type:
direct methods go to the bucket whose config field they were listed in, and a grid's expanded specs all go to the bucket named by the grid's `role`.

```julia
function resolve_methods(config::ExperimentConfig)
    baseline     = Tuple{String, IterativeMethod}[]
    experimental = Tuple{String, IterativeMethod}[]

    for m in config.baseline_methods
        push!(baseline, (string(typeof(m)), m))
    end
    for m in config.experimental_methods
        push!(experimental, (string(typeof(m)), m))
    end
    for grid in config.variant_grids
        bucket = grid.role === :baseline ? baseline : experimental
        for spec in expand(grid)
            push!(bucket, (spec.name, spec.method))
        end
    end

    return baseline, experimental
end
```

This is the single point that interprets the baseline / experimental distinction:
it lives entirely in config-level role metadata, so the method types themselves stay role-agnostic.

---
