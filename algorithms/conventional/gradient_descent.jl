"""
    Gradient Descent Algorithm

A conventional first-order method with pluggable descent direction and
step-size rule. Demonstrates the composable state module pattern.
"""

using Random: AbstractRNG


# ─────────────────────────────────────────────────────────────────────────
# Method Definition
# ─────────────────────────────────────────────────────────────────────────

"""
    GradientDescent <: ConventionalMethod

Gradient descent with pluggable descent direction and step-size rule.

Fields:
- direction :: DescentDirection — descent direction strategy (default: SteepestDescent)
- step_size :: StepSize         — step-size rule (default: ArmijoLS)
"""
@kwdef struct GradientDescent <: ConventionalMethod
	direction::DescentDirection = SteepestDescent()
	step_size::StepSize         = ArmijoLS()
end


# ─────────────────────────────────────────────────────────────────────────
# Method-Specific Numerics Module
# ─────────────────────────────────────────────────────────────────────────

"""
    GradientDescentNumerics

Working storage for gradient descent.

Fields:
- direction          : d_k buffer; written by `compute_direction`.
- α_k                : last step size produced by `compute_step_size`; logged.
- n_linesearch_evals : cumulative line-search trial evaluations.
- grad_prev          : ∇f(x_{k-1}); written before gradient refresh, read by BB.
                       Empty until first step completes (sentinel for BB).
- x_trial            : scratch buffer for line-search trial points.
                       Sized in `init_state`; required by Armijo and Cauchy paths.
"""
@kwdef mutable struct GradientDescentNumerics
	direction::Vector{Float64}          = Float64[]
	α_k::Float64                        = 0.0
	n_linesearch_evals::Int             = 0
	grad_prev::Vector{Float64}          = Float64[]
	x_trial::Vector{Float64}            = Float64[]
end


# ─────────────────────────────────────────────────────────────────────────
# Composed State
# ─────────────────────────────────────────────────────────────────────────

"""
    GradientDescentState

Complete mutable state for gradient descent.

Composes:
    - IterateGroup  (shared) : x, gradient, x_prev
    - MetricsGroup  (shared) : objective, gradient_norm, step_norm, dist_to_opt
    - TimingGroup   (shared) : core_time_ns accumulator (reset by runner each step)
    - GradientDescentNumerics (method-specific): see above
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

Initialize state at x₀: copy iterate, compute initial gradient and objective,
preallocate the line-search trial buffer. `x_prev` and `grad_prev` are kept
empty as the "no previous data" sentinel for BB and the in-place save path.
"""
function init_state(method::GradientDescent, problem, rng::AbstractRNG)
	x0 = copy(problem.x0)
	g0 = similar(x0)
	grad!(g0, problem.f, x0)
	f0 = total_objective(problem, x0)

	return GradientDescentState(
		iterate = IterateGroup(x = x0, gradient = g0, x_prev = Float64[]),
		metrics = MetricsGroup(
			objective     = f0,
			gradient_norm = norm(g0),
			step_norm     = 0.0,
			dist_to_opt   = isnothing(problem.x_opt) ? Inf : norm(x0 .- problem.x_opt),
		),
		timing  = TimingGroup(core_time_ns = 0),
		numerics = GradientDescentNumerics(x_trial = similar(problem.x0)),
	)
end


# ─────────────────────────────────────────────────────────────────────────
# Dispatch Point: Step
# ─────────────────────────────────────────────────────────────────────────

"""
    step!(method::GradientDescent, state, problem, iter, logger, rng)

One iteration: x_{k+1} = x_k + α_k · d_k.

Order is critical for BB: `grad_prev ← ∇f(x_k)` is written **after** the
update and **before** the gradient refresh at x_{k+1}, so the next call's
secant pair uses (x_{k-1}, ∇f(x_{k-1})) correctly.
"""
function step!(method::GradientDescent, state::GradientDescentState,
               problem::Problem, iter::Int, logger::Logger, rng::AbstractRNG)

	# ── Save x_k into x_prev (allocate once on iter 1, then reuse) ────────────
	if isempty(state.iterate.x_prev)
		state.iterate.x_prev = copy(state.iterate.x)
	else
		copyto!(state.iterate.x_prev, state.iterate.x)
	end

	# ── Core: gradient and descent direction at x_k ───────────────────────────
	# 	 ∇f(x_k) already computed in init/last step
	@core_timed state begin
		state.numerics.direction = compute_direction(method.direction, state, problem)
	end

	# ── Step-size selection (rule wraps its own kernel in @core_timed) ────────
	α_k = compute_step_size(method.step_size, state, problem, state.numerics.direction)
	state.numerics.α_k = α_k

	# ── Core: iterate update (no temporary; broadcast fuses into x in place) ──
	@core_timed state begin
		state.iterate.x .+= α_k .* state.numerics.direction
	end

	# ── Save ∇f(x_k) into grad_prev (allocate once on iter 1, then reuse) ─────
	#    Must happen BEFORE the gradient refresh below.
	if isempty(state.numerics.grad_prev)
		state.numerics.grad_prev = copy(state.iterate.gradient)
	else
		copyto!(state.numerics.grad_prev, state.iterate.gradient)
	end

	# ── Core: refresh objective and gradient at x_{k+1} ───────────────────────
	@core_timed state begin
		state.metrics.objective = total_objective(problem, state.iterate.x)
		grad!(state.iterate.gradient, problem.f, state.iterate.x)
	end

	# ── Bookkeeping (untimed) ─────────────────────────────────────────────────
	state.metrics.gradient_norm = norm(state.iterate.gradient)
	state.metrics.step_norm     = abs(α_k) * norm(state.numerics.direction)
	state.metrics.dist_to_opt   = isnothing(problem.x_opt) ? Inf : norm(state.iterate.x .- problem.x_opt)
end


# ─────────────────────────────────────────────────────────────────────────
# Dispatch Point: Log Entry Extraction
# ─────────────────────────────────────────────────────────────────────────

"""
    extract_log_entry(method::GradientDescent, state, iter)

Build IterationLog. `α_k` is read directly from `state.numerics.α_k` —
no reconstruction from norms.
"""
function extract_log_entry(method::GradientDescent, state::GradientDescentState, iter::Int)
	IterationLog(
		iter          = iter,
		core_time_ns  = state.timing.core_time_ns,
		objective     = state.metrics.objective,
		gradient_norm = state.metrics.gradient_norm,
		step_norm     = state.metrics.step_norm,
		dist_to_opt   = state.metrics.dist_to_opt,
		extras = Dict{Symbol,Any}(
			:n_linesearch_evals => state.numerics.n_linesearch_evals,
			:step_size          => state.numerics.α_k,
		),
	)
end