"""
	Gradient Descent Algorithm

A simple conventional first-order method using constant step size.
Demonstrates the composable state module pattern.
"""

using Random: AbstractRNG


# ─────────────────────────────────────────────────────────────────────────
# Method Definition
# ─────────────────────────────────────────────────────────────────────────

"""
	GradientDescent <: ConventionalMethod

Constant step-size gradient descent.

Fields:
- step_size::Float64 — fixed step size (learning rate)
"""
@kwdef struct GradientDescent <: ConventionalMethod
	step_size::Float64 = 0.01
end


# ─────────────────────────────────────────────────────────────────────────
# Method-Specific Numerics Module
# ─────────────────────────────────────────────────────────────────────────

"""
	GradientDescentNumerics

Working storage for gradient descent: gradient vector and step direction.
"""
@kwdef mutable struct GradientDescentNumerics
	gradient::Vector{Float64} = Float64[]
	direction::Vector{Float64} = Float64[]
end


# ─────────────────────────────────────────────────────────────────────────
# Composed State
# ─────────────────────────────────────────────────────────────────────────

"""
	GradientDescentState

Complete mutable state for gradient descent.

Composes:
- IterateGroup (shared): current x, gradient, previous iterate
- MetricsGroup (shared): objective, gradient_norm, step_norm
- TimingGroup (shared): core_time_ns accumulator
- GradientDescentNumerics (method-specific): working vectors
- _logger (injected): reference to the logging system
"""
@kwdef mutable struct GradientDescentState
	iterate::IterateGroup
	metrics::MetricsGroup
	timing::TimingGroup
	numerics::GradientDescentNumerics
	_logger::Union{Nothing, Any} = nothing
end


# ─────────────────────────────────────────────────────────────────────────
# Dispatch Point: Initialization
# ─────────────────────────────────────────────────────────────────────────

"""
	init_state(method::GradientDescent, problem, rng::AbstractRNG)

Initialize gradient descent state by setting x₀ from the problem,
pre-allocating working vectors, and computing the initial gradient.
"""
function init_state(method::GradientDescent, problem, rng::AbstractRNG)
	n = problem.n
	
	x = copy(problem.x0)
	gradient = zeros(Float64, n)
	
	# Compute initial gradient
	grad!(gradient, problem.f, x)
	
	return GradientDescentState(
		iterate = IterateGroup(
			x = x,
			gradient = gradient,
			x_prev = Float64[]
		),
		metrics = MetricsGroup(
			objective = objective(problem, x),
			gradient_norm = norm(gradient),
			step_norm = 0.0
		),
		timing = TimingGroup(core_time_ns = 0),
		numerics = GradientDescentNumerics(
			gradient = gradient,
			direction = similar(gradient)
		),
		_logger = nothing
	)
end


# ─────────────────────────────────────────────────────────────────────────
# Dispatch Point: Step
# ─────────────────────────────────────────────────────────────────────────

"""
	step!(method::GradientDescent, state::GradientDescentState, problem, iter::Int)

One iteration of gradient descent: x ← x - step_size · ∇f(x)

The core computation (gradient evaluation, step) is timed.
Metrics update is not timed (bookkeeping).
"""
function step!(method::GradientDescent, state::GradientDescentState, problem, iter::Int)
	# Core kernel: gradient evaluation and step
	@core_timed state begin
		# Compute gradient at current x
		grad!(state.numerics.gradient, problem.f, state.iterate.x)
		
		# Gradient descent direction: -∇f
		copyto!(state.numerics.direction, state.numerics.gradient)
		rmul!(state.numerics.direction, -method.step_size)
		
		# Update iterate
		state.iterate.x_prev = copy(state.iterate.x)
		state.iterate.x .+= state.numerics.direction
	end
	
	# Update metrics (not timed: bookkeeping)
	state.metrics.objective = objective(problem, state.iterate.x)
	grad!(state.iterate.gradient, problem.f, state.iterate.x)
	state.metrics.gradient_norm = norm(state.iterate.gradient)
	state.metrics.step_norm = norm(state.numerics.direction)
end


# ─────────────────────────────────────────────────────────────────────────
# Dispatch Point: Log Entry Extraction
# ─────────────────────────────────────────────────────────────────────────

"""
	extract_log_entry(method::GradientDescent, state::GradientDescentState, iter::Int)

Build log entry from metrics. Uses default implementation (no algorithm-specific extras).
"""
function extract_log_entry(method::GradientDescent, state::GradientDescentState, iter::Int)
	IterationLog(
		iter = iter,
		core_time_ns = state.timing.core_time_ns,
		objective = state.metrics.objective,
		gradient_norm = state.metrics.gradient_norm,
		step_norm = state.metrics.step_norm,
		extras = Dict{Symbol,Any}()
	)
end
