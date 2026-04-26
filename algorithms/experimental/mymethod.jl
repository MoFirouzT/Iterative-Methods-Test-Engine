"""
	MyMethod — Under-Development Experimental Algorithm

A demonstrative experimental method that uses:
- Swappable components (Hessian approximation, minor updates, line search)
- Multiple behavioral flags controlling subroutines in step!
- Optional nested sub-solver (e.g., LBFGS for a subproblem)

Illustrates the composable state module pattern with method-specific numerics,
optional sub-solver modules, and parameter-driven control flow.
"""

using LinearAlgebra: I, norm
using Random: AbstractRNG


# ─────────────────────────────────────────────────────────────────────────
# Method Definition
# ─────────────────────────────────────────────────────────────────────────

"""
	MyMethod <: ExperimentalMethod

Under-development experimental method with swappable components.

Fields:
- step_size::Float64 — base step size or learning rate
- hessian::HessianApprox — Hessian approximation strategy (BFGS)
- minor::MinorUpdate — minor update strategy
- linesearch::LineSearch — line search strategy
"""
@kwdef struct MyMethod <: ExperimentalMethod
	step_size::Float64 = 0.01
	hessian::HessianApprox = BFGS()
	minor::MinorUpdate = NoMinorUpdate()
	linesearch::LineSearch = ArmijoLS()
end


# ─────────────────────────────────────────────────────────────────────────
# Method-Specific Numerics Module
# ─────────────────────────────────────────────────────────────────────────

"""
	MyMethodNumerics

Working storage for MyMethod: scalars, vectors, matrices, and behavioral flags.

Scalars:
- step_size: current working step size (may vary by line search)
- curvature: estimated curvature or damping parameter

Vectors:
- gradient: workspace for gradient evaluations
- direction: search direction

Matrices:
- H: approximate Hessian matrix (or preconditioner)

Behavioral flags (each toggles a distinct code path in step!):
- use_correction: enable curvature correction
- subproblem_solved: track if nested solver converged
- use_extra_x_update: enable additional corrective step
"""
@kwdef mutable struct MyMethodNumerics
	# Scalars
	step_size::Float64 = 0.0
	curvature::Float64 = 0.0
	
	# Vectors
	gradient::Vector{Float64} = Float64[]
	direction::Vector{Float64} = Float64[]
	
	# Matrices
	H::Matrix{Float64} = Matrix{Float64}(undef, 0, 0)
	
	# Behavioral flags
	use_correction::Bool = false
	subproblem_solved::Bool = false
	use_extra_x_update::Bool = false
end


# ─────────────────────────────────────────────────────────────────────────
# Optional Sub-Solver Module
# ─────────────────────────────────────────────────────────────────────────

"""
	InnerLBFGSModule

Optional sub-solver module for nested LBFGS solver.
Wraps the sub-solver state and optional shared iterate.

This is attached to MyMethodState only if the method uses a nested LBFGS
to solve a subproblem (e.g., a proximal subproblem or inner optimization).
"""
@kwdef mutable struct InnerLBFGSModule
	# Reused or wrapped sub-solver state type
	# (In a real implementation, would reference actual LBFGS state)
	solver_state::Union{Nothing, Any} = nothing
	
	# Optional: shared iterate or subproblem-specific variables
	subproblem_iterate::Union{Nothing, IterateGroup} = nothing
end


# ─────────────────────────────────────────────────────────────────────────
# Composed State
# ─────────────────────────────────────────────────────────────────────────

"""
	MyMethodState

Complete mutable state for MyMethod.

Composes:
- IterateGroup (shared): current x, gradient, previous iterate
- MetricsGroup (shared): objective, gradient_norm, step_norm
- TimingGroup (shared): core_time_ns accumulator
- MyMethodNumerics (method-specific): scalars, vectors, matrices, flags
- inner_solver (optional sub-solver module): Union{Nothing, InnerLBFGSModule}
- _logger (injected): reference to the logging system

The inner_solver field is optional (None by default) and is only instantiated
if the method calls a nested solver. The outer runner does not manage it;
`run_sub_method` creates and runs fresh sub-states independently.
"""
@kwdef mutable struct MyMethodState
	iterate::IterateGroup
	metrics::MetricsGroup
	timing::TimingGroup
	numerics::MyMethodNumerics
	inner_solver::Union{Nothing, InnerLBFGSModule} = nothing
	_logger::Union{Nothing, Any} = nothing
end


# ─────────────────────────────────────────────────────────────────────────
# Dispatch Point: Initialization
# ─────────────────────────────────────────────────────────────────────────

"""
	init_state(method::MyMethod, problem, rng::AbstractRNG)

Initialize MyMethod state by setting x₀ from the problem,
pre-allocating working vectors and matrices, initializing behavioral flags,
and computing the initial gradient and Hessian approximation.
"""
function init_state(method::MyMethod, problem, rng::AbstractRNG)
	n = problem.n
	
	x = copy(problem.x0)
	gradient = zeros(Float64, n)
	
	# Compute initial gradient
	grad!(gradient, problem, x)
	
	return MyMethodState(
		iterate = IterateGroup(
			x = x,
			gradient = gradient,
			x_prev = Float64[]
		),
		metrics = MetricsGroup(
			objective = f(problem, x),
			gradient_norm = norm(gradient),
			step_norm = 0.0
		),
		timing = TimingGroup(core_time_ns = 0),
		numerics = MyMethodNumerics(
			step_size = method.step_size,
			curvature = 1.0,
			gradient = gradient,
			direction = similar(gradient),
			H = Matrix{Float64}(I, n, n),  # Start with identity
			use_correction = true,
			subproblem_solved = false,
			use_extra_x_update = false
		),
		inner_solver = nothing,  # Optional; only instantiate if needed
		_logger = nothing
	)
end


# ─────────────────────────────────────────────────────────────────────────
# Dispatch Point: Step
# ─────────────────────────────────────────────────────────────────────────

"""
	step!(method::MyMethod, state::MyMethodState, problem, iter::Int)

One iteration of MyMethod:

1. Compute gradient at current x
2. Solve for direction using (approximate) Hessian: H·d = -∇f
3. Apply line search if enabled
4. Apply curvature correction if flag is set
5. Optionally call sub-solver for additional refinement
6. Update iterate and metrics

The core computation is wrapped in @core_timed; bookkeeping and metrics
updates are not timed.
"""
function step!(method::MyMethod, state::MyMethodState, problem, iter::Int)
	@core_timed state begin
		# Compute gradient at current x
		grad!(state.numerics.gradient, problem, state.iterate.x)
		
		# Solve for direction: H·d = -∇f (simplified: d = -∇f)
		# In a real implementation, would solve H·d = -∇f using the Hessian approximation
		copyto!(state.numerics.direction, state.numerics.gradient)
		rmul!(state.numerics.direction, -state.numerics.step_size)
		
		# Apply curvature correction if flag is set
		if state.numerics.use_correction
			rmul!(state.numerics.direction, state.numerics.curvature)
		end
		
		# Update iterate
		state.iterate.x_prev = copy(state.iterate.x)
		state.iterate.x .+= state.numerics.direction
	end
	
	# Metrics update (not timed: bookkeeping)
	state.metrics.objective = f(problem, state.iterate.x)
	grad!(state.iterate.gradient, problem, state.iterate.x)
	state.metrics.gradient_norm = norm(state.iterate.gradient)
	state.metrics.step_norm = norm(state.numerics.direction)
	
	# Reset behavioral flags for next iteration
	state.numerics.use_correction = false
	state.numerics.subproblem_solved = false
	state.numerics.use_extra_x_update = false
end


# ─────────────────────────────────────────────────────────────────────────
# Dispatch Point: Log Entry Extraction
# ─────────────────────────────────────────────────────────────────────────

"""
	extract_log_entry(method::MyMethod, state::MyMethodState, iter::Int)

Build log entry from metrics. Can optionally populate extras with
method-specific data (e.g., curvature, Hessian rank, sub-solver status).
"""
function extract_log_entry(method::MyMethod, state::MyMethodState, iter::Int)
	extras = Dict{Symbol,Any}(
		:curvature => state.numerics.curvature,
		:subproblem_solved => state.numerics.subproblem_solved,
	)
	
	IterationLog(
		iter = iter,
		core_time_ns = state.timing.core_time_ns,
		objective = state.metrics.objective,
		gradient_norm = state.metrics.gradient_norm,
		step_norm = state.metrics.step_norm,
		extras = extras
	)
end
