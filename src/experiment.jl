"""
	Layer 5 — Experiment Orchestration

Defines experiment configuration and typed results, resolves methods from direct
lists + variant grids, and provides the multi-run `run_experiment` driver.
"""

using Dates
using Random
using Sockets: gethostname


"""
	ExperimentConfig

Declarative experiment definition.
"""
@kwdef struct ExperimentConfig
	name::String
	problem_spec::ProblemSpec
	conventional_methods::Vector{ConventionalMethod}
	experimental_methods::Vector{ExperimentalMethod} = ExperimentalMethod[]
	variant_grids::Vector{VariantGrid} = VariantGrid[]
	stopping_criteria::StoppingCriterion = stop_when_any(
		MaxIterations(n = 1000),
		GradientTolerance(tol = 1e-6),
	)
	method_criteria::Dict{String,StoppingCriterion} = Dict{String,StoppingCriterion}()
	n_runs::Int = 1
	seed::Union{Int,Nothing} = 42
	tags::Dict{String,Any} = Dict{String,Any}()
end


"""
	MethodResult

Outcome of running one method on one problem instance.
"""
struct MethodResult
	method_name::String
	iter_logs::Vector{IterationLog}
	final_state::Any
	stop_reason::Symbol
	n_iters::Int
end


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

Returns `(conventional, experimental)` where each element is a vector of
`(name, method)` pairs. Variant grids are expanded and appended to
`experimental`.
"""
function resolve_methods(config::ExperimentConfig)
	conventional = Tuple{String,ConventionalMethod}[
		(_method_name(method), method) for method in config.conventional_methods
	]

	experimental = Tuple{String,ExperimentalMethod}[
		(_method_name(method), method) for method in config.experimental_methods
	]

	for grid in config.variant_grids
		specs = expand(grid)
		append!(experimental, [(spec.name, spec.method) for spec in specs])
	end

	return conventional, experimental
end


"""
	next_experiment_path(log_root::String) -> String

Returns a date/counter path like `logs/YYYYMMDD/001`.
"""
function next_experiment_path(log_root::String)::String
	date_str = Dates.format(today(), "yyyymmdd")
	day_dir = joinpath(log_root, date_str)
	mkpath(day_dir)

	entries = readdir(day_dir)
	nums = Int[]
	for name in entries
		if occursin(r"^\d{3,}$", name)
			push!(nums, parse(Int, name))
		end
	end

	next_num = isempty(nums) ? 1 : maximum(nums) + 1
	joinpath(day_dir, lpad(next_num, 3, '0'))
end


function _make_logger(method_name::String, run_id::Int, exp_path::String, verbosity)
	Logger(
		method_name,
		run_id,
		exp_path,
		verbosity,
		IterationLog[],
		NamedTuple[],
		Dict{Symbol,Any}(),
		0.0,
		0,
		IterationLog[],
	)
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
	)
end


function _save_experiment_if_available(result::ExperimentResult)
	if @isdefined(save_experiment)
		save_experiment(result)
	end
	return nothing
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

	conventional, experimental = resolve_methods(config)
	all_methods = vcat(
		Tuple{String,IterativeMethod}[(name, method) for (name, method) in conventional],
		Tuple{String,IterativeMethod}[(name, method) for (name, method) in experimental],
	)

	run_results = RunResult[]

	for run_id in 1:config.n_runs
		root_seed = isnothing(config.seed) ? rand(UInt64) : UInt64(config.seed)

		# Per-role deterministic RNGs derived from the root seed
		rng_problem = Xoshiro(hash((root_seed, run_id, :data)))
		problem = make_problem(config.problem_spec, rng_problem)

		method_results = Dict{String,MethodResult}()
		for (name, method) in all_methods
			criteria = get(config.method_criteria, name, config.stopping_criteria)
			logger = _make_logger(name, run_id, exp_path, verbosity)
			method_rng = Xoshiro(hash((root_seed, run_id, name)))
			raw_result = run_method(method, problem, criteria, logger, method_rng)
			method_results[name] = _to_method_result(name, raw_result)
		end

		push!(run_results, RunResult(run_id, method_results))
	end

	exp_result = ExperimentResult(config, exp_path, now(), gethostname(), run_results)
	_save_experiment_if_available(exp_result)
	return exp_result
end
