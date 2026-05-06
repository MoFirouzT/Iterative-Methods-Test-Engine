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
	direction::DescentDirection = SteepestDescent()
	step_size::StepSize = ArmijoLS()
end


# ─────────────────────────────────────────────────────────────────────────
# Method-Specific Numerics Module
# ─────────────────────────────────────────────────────────────────────────

"""
	GradientDescentNumerics

Working storage for gradient descent: gradient vector and step direction.
"""
@kwdef mutable struct GradientDescentNumerics
	direction::Vector{Float64} = Float64[]
	n_linesearch_evals::Int = 0
	grad_prev::Vector{Float64} = Float64[]
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
"""
@kwdef mutable struct GradientDescentState
	iterate::IterateGroup
	metrics::MetricsGroup
	timing::TimingGroup
	numerics::GradientDescentNumerics
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
	x0 = copy(problem.x0)
	g0 = grad(problem.f, x0)
	f0 = total_objective(problem, x0)

	return GradientDescentState(
		iterate = IterateGroup(
			x = x0,
			gradient = g0,
			x_prev = Float64[]
		),
		metrics = MetricsGroup(
			objective = f0,
			gradient_norm = norm(g0),
			step_norm = 0.0,
			dist_to_opt = isnothing(problem.x_opt) ? Inf : norm(x0 .- problem.x_opt)
		),
		timing = TimingGroup(core_time_ns = 0),
		numerics = GradientDescentNumerics(
			direction = Float64[],
			n_linesearch_evals = 0,
			grad_prev = Float64[]
		),
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
function step!(method::GradientDescent, state::GradientDescentState, problem::Problem, iter::Int, logger::Logger, rng::AbstractRNG)

	# Save previous iterate for x_prev and BB s_{k-1}
	x_prev = copy(state.iterate.x)

	# Core: compute gradient at x_k and descent direction
	@core_timed state begin
		g_k = grad(problem.f, state.iterate.x)
		state.iterate.gradient = g_k

		d_k = compute_direction(method.direction, state, problem)
		state.numerics.direction = d_k
	end

	# Step-size selection (each rule handles its own timing)
	α_k = compute_step_size(method.step_size, state, problem, state.numerics.direction)

	# Core: apply step
	local step_vec
	@core_timed state begin
		step_vec = α_k .* state.numerics.direction
		state.iterate.x .+= step_vec
	end

	# Bookkeeping required for BB and logging
	state.iterate.x_prev = x_prev
	state.numerics.grad_prev = copy(state.iterate.gradient)

	# Core: refresh objective and gradient at new iterate
	@core_timed state begin
		state.metrics.objective = total_objective(problem, state.iterate.x)
		state.iterate.gradient = grad(problem.f, state.iterate.x)
	end

	# Bookkeeping (outside timed region)
	state.metrics.gradient_norm = norm(state.iterate.gradient)
	state.metrics.step_norm = norm(step_vec)
end


# ─────────────────────────────────────────────────────────────────────────
# Dispatch Point: Log Entry Extraction
# ─────────────────────────────────────────────────────────────────────────

"""
	extract_log_entry(method::GradientDescent, state::GradientDescentState, iter::Int)

Build log entry from metrics. Uses default implementation (no algorithm-specific extras).
"""
function extract_log_entry(method::GradientDescent, state::GradientDescentState, iter::Int)
	α_k_recovered = state.metrics.step_norm / max(norm(state.numerics.direction), 1e-16)
	IterationLog(
		iter = iter,
		core_time_ns = state.timing.core_time_ns,
		objective = state.metrics.objective,
		gradient_norm = state.metrics.gradient_norm,
		step_norm = state.metrics.step_norm,
		dist_to_opt = state.metrics.dist_to_opt,
		extras = Dict{Symbol,Any}(
			:n_linesearch_evals => state.numerics.n_linesearch_evals,
			:step_size => α_k_recovered,
		)
	)
end
