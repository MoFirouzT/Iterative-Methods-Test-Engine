"""
	Layer 1 — Algorithm Abstraction & Core Timing

Defines the core method hierarchy, canonical state parameter groups, required
dispatch points, `@core_timed`, and the generic `run_method` loop.
"""

using Random: AbstractRNG, TaskLocalRNG


# ─────────────────────────────────────────────────────────────────────────
# Method Hierarchy
# ─────────────────────────────────────────────────────────────────────────

"""
	abstract type IterativeMethod

Base type for all iterative algorithms.
"""
abstract type IterativeMethod end


"""
	abstract type ConventionalMethod <: IterativeMethod

Baseline/reference methods.
"""
abstract type ConventionalMethod <: IterativeMethod end


"""
	abstract type ExperimentalMethod <: IterativeMethod

Under-development and variant-driven methods.
"""
abstract type ExperimentalMethod <: IterativeMethod end


# ─────────────────────────────────────────────────────────────────────────
# Canonical State Groups
# ─────────────────────────────────────────────────────────────────────────

"""
	IterateGroup

Shared iterate-related fields used by all methods.
"""
@kwdef mutable struct IterateGroup
	x :: Vector{Float64}
	gradient :: Vector{Float64}
	x_prev :: Vector{Float64} = Float64[]
end


"""
	MetricsGroup

Scalar convergence metrics. Mirrors the fixed fields of IterationLog.
"""
@kwdef mutable struct MetricsGroup
	objective :: Float64 = Inf
	gradient_norm :: Float64 = Inf
	step_norm :: Float64 = Inf
	dist_to_opt :: Float64 = Inf
end


"""
	TimingGroup

Per-step core computation time accumulator in nanoseconds.
The runner resets this before each step! call.
"""
@kwdef mutable struct TimingGroup
	core_time_ns :: Int64 = 0
end


# ─────────────────────────────────────────────────────────────────────────
# Algorithm Interface Dispatch Points
# ─────────────────────────────────────────────────────────────────────────

"""
	init_state(method::IterativeMethod, problem, rng::AbstractRNG)

Create and return mutable algorithm state before the iteration loop.
"""
function init_state end


"""
	step!(method::IterativeMethod, state, problem, iter::Int, logger::Logger, rng::AbstractRNG)

Advance one iteration by mutating state in place.
Core mathematical kernels inside this function should use @core_timed.
Logger and RNG are injected by the runner for use in nested sub-methods.
"""
function step! end


"""
	extract_log_entry(method::IterativeMethod, state, iter::Int)

Build one IterationLog from the current state.
Methods can overload this to add algorithm-specific extras.
"""
function extract_log_entry end


# Helpful default errors for algorithms that forgot to implement the interface.
function init_state(method::IterativeMethod, problem, rng::AbstractRNG)
	throw(MethodError(init_state, (method, problem, rng)))
end

function step!(method::IterativeMethod, state, problem, iter::Int, logger::Logger, rng::AbstractRNG)
	throw(MethodError(step!, (method, state, problem, iter, logger, rng)))
end


"""
	extract_log_entry(method::IterativeMethod, state, iter::Int)

Default extraction based on canonical MetricsGroup + TimingGroup.
Override in concrete methods to populate extras.
"""
function extract_log_entry(method::IterativeMethod, state, iter::Int)
	IterationLog(
		iter = iter,
		core_time_ns = state.timing.core_time_ns,
		objective = state.metrics.objective,
		gradient_norm = state.metrics.gradient_norm,
		step_norm = state.metrics.step_norm,
		dist_to_opt = state.metrics.dist_to_opt,
		extras = Dict{Symbol,Any}(),
	)
end


# ─────────────────────────────────────────────────────────────────────────
# Core Timing Macro
# ─────────────────────────────────────────────────────────────────────────

"""
	@core_timed state expr

Evaluate expr and add elapsed nanoseconds to state.timing.core_time_ns.
Multiple @core_timed blocks per step accumulate naturally.
"""
macro core_timed(state, expr)
	quote
		local _t0 = time_ns()
		local _ret = $(esc(expr))
		$(esc(state)).timing.core_time_ns += time_ns() - _t0
		_ret
	end
end


# ─────────────────────────────────────────────────────────────────────────
# Nested Algorithm Infrastructure (Layer 4)
# ─────────────────────────────────────────────────────────────────────────

"""
	SubRunConfig

Configuration for a nested algorithm invocation.
"""
@kwdef struct SubRunConfig
	method::IterativeMethod
	criteria
	log_sub_iters::Bool = true
	verbosity::Any = nothing
end


"""
	SubResult

Result summary for one nested algorithm run.
"""
struct SubResult
	converged::Bool
	stop_reason::Symbol
	n_iters::Int
	final_state::Any
	iter_logs::Vector{Any}
	core_time_ns::Int64
end


"""
	is_converged_reason(reason::Symbol) -> Bool

Classifies stop reasons that indicate successful convergence.
"""
function is_converged_reason(reason::Symbol)
	reason in (:gradient_converged, :step_converged, :objective_stagnated, :all_criteria_met)
end


"""
	make_sub_logger(method, outer_logger, verbosity) -> Logger

Constructs an in-memory logger for nested runs.
"""
function make_sub_logger(method::IterativeMethod, outer_logger::Logger, verbosity)
	Logger(
		string(typeof(method)),
		outer_logger.run_id,
		outer_logger.exp_path,
		verbosity,
		IterationLog[],
		NamedTuple[],
		Dict{Symbol,Any}(),
		0.0,
		0,
		IterationLog[],
	)
end


"""
	run_sub_method(config, problem, outer_logger) -> SubResult

Runs a nested method with its own state and logger. If configured, sub-iteration
logs are attached to the outer logger via `attach_sub_logs!`.
"""
function run_sub_method(config::SubRunConfig, problem, outer_logger::Logger)::SubResult
	rng = TaskLocalRNG()
	sub_state = init_state(config.method, problem, rng)

	if hasproperty(sub_state, :_logger)
		setproperty!(sub_state, :_logger, outer_logger)
	end

	sub_logger = make_sub_logger(config.method, outer_logger, config.verbosity)
	log_init!(sub_logger, config.method, sub_state)

	iter = 0
	while true
		iter += 1

		sub_state.timing.core_time_ns = 0
		step!(config.method, sub_state, problem, iter, sub_logger, rng)

		entry = extract_log_entry(config.method, sub_state, iter)
		log_iter!(sub_logger, entry)

		stop, reason = should_stop(config.criteria, sub_state, iter, sub_logger)
		if stop
			if config.log_sub_iters
				attach_sub_logs!(outer_logger, sub_logger.iter_logs)
			end

			total_core = sum(e.core_time_ns for e in sub_logger.iter_logs; init = Int64(0))
			return SubResult(
				is_converged_reason(reason),
				reason,
				iter,
				sub_state,
				sub_logger.iter_logs,
				total_core,
			)
		end
	end
end


# ─────────────────────────────────────────────────────────────────────────
# Generic Runner
# ─────────────────────────────────────────────────────────────────────────

"""
	run_method(method, problem, criteria, logger, rng)

Generic iterative loop driven by stopping criteria.

Flow per iteration:
1. Reset state.timing.core_time_ns
2. step!
3. extract_log_entry
4. log_iter!
5. should_stop

On stop, records event and returns finalize!(logger, method, state).
"""
function run_method(method::IterativeMethod, problem, criteria, logger, rng::AbstractRNG)
	state = init_state(method, problem, rng)

	# Runner injects logger reference so nested routines can access it when needed.
	if hasproperty(state, :_logger)
		setproperty!(state, :_logger, logger)
	end

	log_init!(logger, method, state)
	iter = 0

	while true
		iter += 1

		state.timing.core_time_ns = 0
		step!(method, state, problem, iter, logger, rng)

		entry = extract_log_entry(method, state, iter)
		log_iter!(logger, entry)

		stop, reason = should_stop(criteria, state, iter, logger)
		if stop
			log_event!(logger, reason, iter)
			break
		end
	end

	finalize!(logger, method, state)
end
