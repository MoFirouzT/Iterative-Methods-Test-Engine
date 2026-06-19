"""
	Experiment Orchestration

Defines experiment configuration and typed results, resolves methods from direct
lists + variant grids, and provides the multi-run `run_experiment` driver.
"""

using Dates
using Random
using Sockets: gethostname


"""
	WarmupStrategy

Abstract type for optional, **shared** per-run warm-up steps. A warm-up runs
once per run before any method starts; its output is a new initial point
`x0_warm` that replaces `problem.x0` for every method in that run. Methods
cannot distinguish a warm-up start from a cold start.

Concrete strategies: [`NoWarmup`](@ref), [`IterativeWarmup`](@ref),
[`FunctionWarmup`](@ref).
"""
abstract type WarmupStrategy end


"""
	NoWarmup <: WarmupStrategy

Default — leaves `problem.x0` untouched.
"""
struct NoWarmup <: WarmupStrategy end


"""
	IterativeWarmup <: WarmupStrategy

Run an iterative method as warm-up; the final iterate becomes the shared `x0`.
Uses the universal `result.final_state.iterate.x` convention.
"""
@kwdef struct IterativeWarmup <: WarmupStrategy
	method::IterativeMethod
	criteria::StoppingCriterion
	verbosity::VerbosityConfig = VerbosityConfig(level = MILESTONE)
end


"""
	FunctionWarmup <: WarmupStrategy

Apply a registered pure function `(problem, rng) -> Vector{Float64}` to
produce `x0`. The function lives in [`WARMUP_FUNCTIONS`](@ref); the strategy
itself only holds the `name` so it stays serialization-safe.
"""
struct FunctionWarmup <: WarmupStrategy
	name::Symbol
end


"""
	WARMUP_FUNCTIONS

Registry of `FunctionWarmup` generators. Maps `Symbol` → `(problem, rng) -> Vector{Float64}`.
"""
const WARMUP_FUNCTIONS = Dict{Symbol,Function}()


"""
	register_warmup!(name::Symbol, gen::Function)

Register a function-warmup generator with signature
`(problem::Problem, rng::AbstractRNG) -> Vector{Float64}`.
"""
function register_warmup!(name::Symbol, gen::Function)
	WARMUP_FUNCTIONS[name] = gen
	return nothing
end


"""
	run_warmup(strategy, problem, rng) -> Vector{Float64}

Returns the warm-started initial point. Dispatches on the strategy.
"""
function run_warmup(::NoWarmup, problem, ::AbstractRNG)
	return copy(problem.x0)
end

function run_warmup(w::IterativeWarmup, problem, rng::AbstractRNG)
	warmup_logger = make_logger("__warmup__", 0, "", w.verbosity)
	result = run_method(w.method, problem, w.criteria, warmup_logger, rng)
	return copy(result.final_state.iterate.x)
end

function run_warmup(w::FunctionWarmup, problem, rng::AbstractRNG)
	haskey(WARMUP_FUNCTIONS, w.name) ||
		throw(KeyError("FunctionWarmup :$(w.name) not registered"))
	return copy(WARMUP_FUNCTIONS[w.name](problem, rng))
end


"""
	_with_x0(problem, x0_new) -> Problem

Return a copy of `problem` whose `x0` is `x0_new` (and whose `n` is updated to
match). All other fields are preserved by reference. Used by `run_experiment`
to plumb the warm-up output back into the Problem record without mutating the
caller's `problem_spec`.
"""
function _with_x0(problem::Problem, x0_new::Vector{Float64})
	return Problem(problem.f, problem.gs, x0_new, length(x0_new),
		problem.meta, problem.x_opt)
end


"""
	PersistPolicy

Save-time control over which per-iteration `extras` are written to
`result.jld2`. The scalar metric columns (`objective`, `gradient_norm`,
`step_norm`, `dist_to_opt`, `core_time_ns`) are **never** affected — they are
always persisted at full resolution. Only named `extras` keys are pruned.

- `drop::Vector{Symbol}` — keys removed from the JLD2 payload entirely. The
  usual target is `:x_iter` (full-iterate snapshots) or `:sub_logs`, which
  dominate the on-disk size on high-dimensional problems but are only needed
  for trajectory plots.
- `decimate::Dict{Symbol,Int}` — `key => k` keeps that key only on iters where
  `iter % k == 0` (plus iter 0), dropping it elsewhere. Use this to thin a
  trajectory without losing it entirely.

The default keeps everything, so existing behaviour is unchanged. CSV sidecars
are independent of this policy — they only ever carry scalar extras and are
written from the full result. What was pruned is recorded in `manifest.json`.
"""
@kwdef struct PersistPolicy
	drop::Vector{Symbol} = Symbol[]
	decimate::Dict{Symbol,Int} = Dict{Symbol,Int}()
end


"""
	ExperimentConfig

Declarative experiment definition.
"""
@kwdef struct ExperimentConfig
	name::String
	problem_spec::ProblemSpec
	# Methods are split by their comparison **role**, not by type: a method is a
	# `baseline` (reference) or `experimental` (under study) purely by which list
	# it is placed in. `variant_grids` route by each grid's own `role`. None is
	# individually required (at least one must produce a non-empty method list),
	# so all default to empty.
	baseline_methods::Vector{IterativeMethod} = IterativeMethod[]
	experimental_methods::Vector{IterativeMethod} = IterativeMethod[]
	variant_grids::Vector{VariantGrid} = VariantGrid[]
	stopping_criteria::StoppingCriterion = stop_when_any(
		MaxIterations(n = 1000),
		GradientTolerance(tol = 1e-6),
	)
	method_criteria::Dict{String,StoppingCriterion} = Dict{String,StoppingCriterion}()
	warmup::WarmupStrategy = NoWarmup()
	n_runs::Int = 1
	seed::Union{Int,Nothing} = 42
	tags::Dict{String,Any} = Dict{String,Any}()
	debug::DebugConfig = DebugConfig()
	# Opt-in oracle counting: wrap each method's problem.f in a CountingObjective and
	# surface cumulative :n_value/:n_grad/:n_hvp in the logs. Off by default so the
	# core-time measurement path is unperturbed.
	count_oracles::Bool = false
	# Save-time pruning of heavy per-iteration extras from result.jld2 (see
	# PersistPolicy). Default keeps everything; set drop=[:x_iter] (or decimate)
	# to shrink the binary on high-dimensional problems.
	persist::PersistPolicy = PersistPolicy()
end


"""
	MethodResult

Outcome of running one method on one problem instance.

`events` carries any named events the logger recorded — typically the
stopping reason emitted by `log_event!`, plus any debug events captured via
`on_trigger = :log`. Without this field, the debug `:log` payload would be
dropped at `finalize!` (Stage 8 covers the roundtrip explicitly).
"""
struct MethodResult
	method_name::String
	iter_logs::Vector{IterationLog}
	final_state::Any
	stop_reason::Symbol
	n_iters::Int
	events::Vector{NamedTuple}
end

MethodResult(name, iter_logs, final_state, stop_reason, n_iters) =
	MethodResult(name, iter_logs, final_state, stop_reason, n_iters, NamedTuple[])


"""
	RunResult

All method outcomes for one run.
"""
struct RunResult
	run_id::Int
	method_results::Dict{String,MethodResult}
end


"""
	ExperimentResult

Top-level experiment outcome across all runs.
"""
struct ExperimentResult
	config::ExperimentConfig
	experiment_path::String
	timestamp::DateTime
	host::String
	run_results::Vector{RunResult}
end


_method_name(method::IterativeMethod) = string(typeof(method))


"""
	resolve_methods(config::ExperimentConfig)

Returns `(baseline, experimental)` where each element is a vector of
`(name, method)` pairs. Direct methods are placed by which config list they
came from (`baseline_methods` / `experimental_methods`); each variant grid is
expanded and all of its produced specs are routed into the bucket named by the
grid's `role` (`:baseline` or `:experimental`). Role is experiment-level
metadata, never inferred from the method's type.
"""
function resolve_methods(config::ExperimentConfig)
	baseline = Tuple{String,IterativeMethod}[
		(_method_name(method), method) for method in config.baseline_methods
	]

	experimental = Tuple{String,IterativeMethod}[
		(_method_name(method), method) for method in config.experimental_methods
	]

	for grid in config.variant_grids
		bucket = grid.role === :baseline     ? baseline     :
				 grid.role === :experimental ? experimental :
				 throw(ArgumentError("VariantGrid role must be :baseline or " *
					":experimental; got :$(grid.role)"))
		for spec in expand(grid)
			push!(bucket, (spec.name, spec.method))
		end
	end

	return baseline, experimental
end


"""
	next_experiment_path(log_root::String) -> String

Atomically reserves and returns a date/counter path like `logs/YYYYMMDD/001`.

The numbered directory is created with `mkdir` (not `mkpath`), which throws on
`EEXIST`. Two processes that compute the same next number therefore can't both
win: the loser catches the collision, re-scans, and takes the next free slot.
The returned path is guaranteed to exist and be owned by this caller.
"""
function next_experiment_path(log_root::String)::String
	date_str = Dates.format(today(), "yyyymmdd")
	day_dir = joinpath(log_root, date_str)
	mkpath(day_dir)

	while true
		nums = Int[]
		for name in readdir(day_dir)
			if occursin(r"^\d{3,}$", name)
				push!(nums, parse(Int, name))
			end
		end

		next_num = isempty(nums) ? 1 : maximum(nums) + 1
		candidate = joinpath(day_dir, lpad(next_num, 3, '0'))

		try
			mkdir(candidate)            # atomic reservation; throws on EEXIST
			return candidate
		catch err
			# Lost the race for this number — another writer created it between
			# our readdir and mkdir. Re-scan and retry; any other error is real.
			(err isa Base.IOError && err.code == Base.UV_EEXIST) || rethrow()
		end
	end
end


function _to_method_result(name::String, result)
	if result isa MethodResult
		return result
	end

	MethodResult(
		name,
		result.iter_logs,
		result.final_state,
		result.stop_reason,
		result.n_iters,
		hasproperty(result, :events) ? result.events : NamedTuple[],
	)
end


"""
	run_experiment(config::ExperimentConfig, log_root::String = "logs"; verbosity = VerbosityConfig())

Runs all configured methods for `n_runs`, creates per-run RNGs from
`config.seed`, and returns an `ExperimentResult`.
"""
function run_experiment(config::ExperimentConfig,
						log_root::String = "logs";
						verbosity = VerbosityConfig())
	exp_path = next_experiment_path(log_root)
	mkpath(exp_path)

	baseline, experimental = resolve_methods(config)
	all_methods = vcat(
		Tuple{String,IterativeMethod}[(name, method) for (name, method) in baseline],
		Tuple{String,IterativeMethod}[(name, method) for (name, method) in experimental],
	)

	run_results = RunResult[]

	for run_id in 1:config.n_runs
		root_seed = isnothing(config.seed) ? rand(UInt64) : UInt64(config.seed)

		# Per-role deterministic RNGs derived from the root seed
		rng_problem = Xoshiro(hash((root_seed, run_id, :data)))
		problem = make_problem(config.problem_spec, rng_problem)

		# Shared per-run warm-up: a single call whose output replaces problem.x0
		# for every method in this run. Skipped (no-op) when warmup is NoWarmup.
		if !(config.warmup isa NoWarmup)
			rng_warmup = Xoshiro(hash((root_seed, run_id, :warmup)))
			x0_warm = run_warmup(config.warmup, problem, rng_warmup)
			problem = _with_x0(problem, x0_warm)
		end

		method_results = Dict{String,MethodResult}()
		for (name, method) in all_methods
			criteria = get(config.method_criteria, name, config.stopping_criteria)
			logger = make_logger(name, run_id, exp_path, verbosity)
			method_rng = Xoshiro(hash((root_seed, run_id, name)))
			# Opt-in oracle counting: wrap per method so each gets a fresh tally
			# (after warm-up, so warm-up evaluations are excluded from the measured run).
			method_problem = config.count_oracles ? with_oracle_counting(problem) : problem
			raw_result = run_method(method, method_problem, criteria, logger, method_rng;
									debug = config.debug)
			method_results[name] = _to_method_result(name, raw_result)
		end

		push!(run_results, RunResult(run_id, method_results))
	end

	exp_result = ExperimentResult(config, exp_path, now(), gethostname(), run_results)
	# `save_experiment` lives in persistence.jl, included after this file; the
	# global is resolved at call time, by which point the module is fully loaded.
	save_experiment(exp_result; persist = config.persist)
	return exp_result
end
