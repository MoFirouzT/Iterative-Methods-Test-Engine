"""
    Trust Region with a Steihaug truncated-CG inner solve.

The consumer that lights up the nested-optimization subsystem (`run_sub_method`,
`SubRunConfig`, `SubResult`, `attach_sub_logs!`): each OUTER trust-region step
builds a quadratic model `m(p)` at the current iterate and solves
`min_{‖p‖≤Δ} m(p)` approximately with Steihaug-CG run as a GENUINE sub-method on
a GENUINE `Problem`. See `trust_region.md`.

This file defines three things:
  • `QuadraticModel <: Objective` — the model sub-problem objective.
  • `SteihaugCG`                  — the truncated-CG inner solver (one CG iter / step!).
  • `TrustRegion`                 — the outer method (defined below the inner solver).
"""

using Random: AbstractRNG
using LinearAlgebra: dot, norm
using .TestEngine
import .TestEngine: Objective, Hessian, value, grad!, hessian,
	init_state, step!, extract_log_entry, _tr_status


# ═════════════════════════════════════════════════════════════════════════
# Quadratic model objective:  m(p) = gᵀp + ½ pᵀ H p   (with m(0) = 0)
# The minimization variable is the STEP p; H is carried as a Hessian object
# (an OperatorHessian, so no matrix is ever formed).
# ═════════════════════════════════════════════════════════════════════════

struct QuadraticModel <: Objective
	g::Vector{Float64}    # ∇f at the outer iterate x
	H::Hessian            # ∇²f at x (operator / matrix / diagonal)
end

value(m::QuadraticModel, p::Vector{Float64}) = dot(m.g, p) + 0.5 * dot(p, apply(m.H, p))

function grad!(out::Vector{Float64}, m::QuadraticModel, p::Vector{Float64})::Vector{Float64}
	out .= m.g .+ apply(m.H, p)        # ∇m(p) = g + H p
	return out
end

hessian(m::QuadraticModel, p::Vector{Float64})::Hessian = m.H


# ═════════════════════════════════════════════════════════════════════════
# Steihaug truncated-CG inner solver
#
# Minimizes m(p) within the trust region ‖p‖ ≤ Δ. One CG iteration per `step!`,
# so the inner trace is a real iteration log. Terminates (via composable
# StoppingCriteria) on: residual tol (GradientTolerance on ‖r‖), max inner iters,
# negative curvature, or hitting the trust-region boundary.
# ═════════════════════════════════════════════════════════════════════════

@kwdef struct SteihaugCG <: ConventionalMethod
	Δ::Float64 = 1.0       # trust radius for this solve
end

@kwdef mutable struct SteihaugCGNumerics
	d::Vector{Float64} = Float64[]     # current CG search direction
	Δ::Float64         = 1.0           # trust radius
	status::Symbol     = :running      # :running | :boundary | :negative_curvature
	rTr::Float64       = 0.0           # cached ‖r‖²
end

@kwdef mutable struct SteihaugCGState
	iterate::IterateGroup              # x = p (the step), gradient = r (model residual)
	metrics::MetricsGroup
	timing::TimingGroup
	numerics::SteihaugCGNumerics
end

# Expose the inner status to the NegativeCurvature / TrustRegionBoundary criteria.
_tr_status(s::SteihaugCGState) = s.numerics.status


function init_state(method::SteihaugCG, problem, rng::AbstractRNG)
	p = zeros(problem.n)
	r = similar(p); grad!(r, problem.f, p)   # ∇m(0) = g
	return SteihaugCGState(
		iterate = IterateGroup(x = p, gradient = r, x_prev = Float64[]),
		metrics = MetricsGroup(objective = total_objective(problem, p),
			gradient_norm = norm(r), step_norm = 0.0, dist_to_opt = Inf),
		timing  = TimingGroup(core_time_ns = 0),
		numerics = SteihaugCGNumerics(d = -r, Δ = method.Δ, status = :running, rTr = dot(r, r)),
	)
end

# Positive root τ ≥ 0 of ‖p + τ d‖ = Δ  (the forward step to the boundary).
function _tau_to_boundary(p::Vector{Float64}, d::Vector{Float64}, Δ::Float64)
	a = dot(d, d)
	b = 2.0 * dot(p, d)
	c = dot(p, p) - Δ^2
	disc = b^2 - 4a * c
	return (-b + sqrt(max(disc, 0.0))) / (2a)
end

function step!(method::SteihaugCG, st::SteihaugCGState, problem::Problem,
               iter::Int, logger::Logger, rng::AbstractRNG)
	nu = st.numerics
	p  = st.iterate.x
	r  = st.iterate.gradient
	Δ  = nu.Δ
	H  = hessian(problem.f, p)          # constant model Hessian

	@core_timed st begin
		Hd  = apply(H, nu.d)
		dHd = dot(nu.d, Hd)
		rTr = nu.rTr
		p_old = copy(p)

		if dHd <= 0.0
			# Negative curvature: the model is unbounded along d — go to the boundary.
			τ = _tau_to_boundary(p, nu.d, Δ)
			st.iterate.x .= p .+ τ .* nu.d
			nu.status = :negative_curvature
			grad!(r, problem.f, st.iterate.x)
		else
			α = rTr / dHd
			if norm(p .+ α .* nu.d) >= Δ
				# Unconstrained CG step would leave the region — truncate to the boundary.
				τ = _tau_to_boundary(p, nu.d, Δ)
				st.iterate.x .= p .+ τ .* nu.d
				nu.status = :boundary
				grad!(r, problem.f, st.iterate.x)
			else
				st.iterate.x .= p .+ α .* nu.d
				r .= r .+ α .* Hd               # r_{k+1} = r_k + α H d
				rTr_new = dot(r, r)
				β = rTr_new / rTr
				nu.d .= -r .+ β .* nu.d         # d_{k+1} = -r_{k+1} + β d_k
				nu.rTr = rTr_new
				nu.status = :running
			end
		end

		st.metrics.step_norm     = norm(st.iterate.x .- p_old)
		st.metrics.objective     = total_objective(problem, st.iterate.x)
		st.metrics.gradient_norm = norm(r)       # ‖residual‖ → GradientTolerance = residual tol
	end
end

function extract_log_entry(method::SteihaugCG, st::SteihaugCGState, iter::Int)
	entry = IterationLog(
		iter = iter, core_time_ns = st.timing.core_time_ns,
		objective = st.metrics.objective, gradient_norm = st.metrics.gradient_norm,
		step_norm = st.metrics.step_norm, dist_to_opt = st.metrics.dist_to_opt,
		extras = Dict{Symbol,Any}(:status => st.numerics.status, :radius => st.numerics.Δ),
	)
	length(st.iterate.x) <= 2 && (entry.extras[:p_iter] = copy(st.iterate.x))
	return entry
end


# ═════════════════════════════════════════════════════════════════════════
# Trust-region outer method
#
# Each step: build m(p) at x, solve min_{‖p‖≤Δ} m(p) with SteihaugCG via
# run_sub_method, then accept/reject by the actual/predicted reduction ratio ρ
# and update Δ. The inner solve's CORE time is folded into the outer step's
# core_time_ns (so cumulative-core plots reflect ALL work) and also exposed
# per-step via the log extras — see trust_region.md / docs/src/modules/persistence.md.
# ═════════════════════════════════════════════════════════════════════════

@kwdef struct TrustRegion <: ConventionalMethod
	Δ0::Float64        = 1.0       # initial trust radius
	Δmax::Float64      = 100.0     # max trust radius
	η::Float64         = 0.1       # accept the step when ρ > η
	max_inner::Int     = 0         # inner CG cap (0 ⇒ problem dimension + 1)
	inner_tol::Float64 = 1e-8      # inner residual tolerance
end

@kwdef mutable struct TrustRegionNumerics
	Δ::Float64           = 1.0
	ρ::Float64           = 0.0
	accepted::Bool       = false
	n_inner::Int         = 0
	inner_core_ns::Int64 = 0
	inner_stop::Symbol   = :none
	sub_logs::Vector{Any} = Any[]   # inner CG trace for THIS outer iteration
end

@kwdef mutable struct TrustRegionState
	iterate::IterateGroup
	metrics::MetricsGroup
	timing::TimingGroup
	numerics::TrustRegionNumerics
end

function init_state(method::TrustRegion, problem, rng::AbstractRNG)
	x0 = copy(problem.x0)
	g0 = similar(x0); grad!(g0, problem.f, x0)
	f0 = total_objective(problem, x0)
	return TrustRegionState(
		iterate = IterateGroup(x = x0, gradient = g0, x_prev = Float64[]),
		metrics = MetricsGroup(objective = f0, gradient_norm = norm(g0), step_norm = 0.0,
			dist_to_opt = isnothing(problem.x_opt) ? Inf : norm(x0 .- problem.x_opt)),
		timing  = TimingGroup(core_time_ns = 0),
		numerics = TrustRegionNumerics(Δ = method.Δ0),
	)
end

function step!(method::TrustRegion, st::TrustRegionState, problem::Problem,
               iter::Int, logger::Logger, rng::AbstractRNG)
	nu = st.numerics
	g  = copy(st.iterate.gradient)            # ∇f(x_k)
	H  = hessian(problem.f, st.iterate.x)     # ∇²f(x_k) (operator / matrix)

	# ── Inner solve: approximately minimize the model in the trust region ─────
	maxinner = method.max_inner > 0 ? method.max_inner : problem.n + 1
	inner_criteria = stop_when_any(
		MaxIterations(n = maxinner),
		GradientTolerance(tol = method.inner_tol),
		NegativeCurvature(),
		TrustRegionBoundary(),
	)
	model = Problem(QuadraticModel(g, H), zeros(problem.n))
	sub = run_sub_method(SubRunConfig(method = SteihaugCG(Δ = nu.Δ),
	                                  criteria = inner_criteria, log_sub_iters = true),
	                     model, logger, rng)
	# Fold the inner solve's CORE time into this outer step (convention: §10).
	st.timing.core_time_ns += sub.core_time_ns
	p = sub.final_state.iterate.x

	# ── Actual vs predicted reduction ─────────────────────────────────────────
	local fxp, pred, actual, ρ
	@core_timed st begin
		x_trial = st.iterate.x .+ p
		fxp     = total_objective(problem, x_trial)
		mp      = dot(g, p) + 0.5 * dot(p, apply(H, p))   # m(p); m(0)=0
		pred    = -mp                                     # predicted reduction
		actual  = st.metrics.objective - fxp              # f(x_k) − f(x_k+p)
		ρ       = pred > 0 ? actual / pred : (actual > 0 ? Inf : 0.0)
	end

	accepted     = (pred > 0) && (ρ > method.η)
	hit_boundary = sub.stop_reason in (:boundary_reached, :negative_curvature)

	# ── Accept / reject ───────────────────────────────────────────────────────
	if accepted
		if isempty(st.iterate.x_prev)
			st.iterate.x_prev = copy(st.iterate.x)
		else
			copyto!(st.iterate.x_prev, st.iterate.x)
		end
		@core_timed st begin
			st.iterate.x .+= p
			st.metrics.objective = fxp
			grad!(st.iterate.gradient, problem.f, st.iterate.x)
		end
		st.metrics.step_norm = norm(p)
	else
		st.metrics.step_norm = 0.0            # x, objective, gradient unchanged
	end

	# ── Trust-radius update ────────────────────────────────────────────────────
	if ρ < 0.25
		nu.Δ *= 0.25
	elseif ρ > 0.75 && hit_boundary
		nu.Δ = min(2 * nu.Δ, method.Δmax)
	end

	# ── Bookkeeping ────────────────────────────────────────────────────────────
	st.metrics.gradient_norm = norm(st.iterate.gradient)
	st.metrics.dist_to_opt   = isnothing(problem.x_opt) ? Inf : norm(st.iterate.x .- problem.x_opt)
	nu.ρ = ρ; nu.accepted = accepted
	nu.n_inner = sub.n_iters; nu.inner_core_ns = sub.core_time_ns
	nu.inner_stop = sub.stop_reason; nu.sub_logs = sub.iter_logs
end

function extract_log_entry(method::TrustRegion, st::TrustRegionState, iter::Int)
	nu = st.numerics
	entry = IterationLog(
		iter = iter, core_time_ns = st.timing.core_time_ns,
		objective = st.metrics.objective, gradient_norm = st.metrics.gradient_norm,
		step_norm = st.metrics.step_norm, dist_to_opt = st.metrics.dist_to_opt,
		extras = Dict{Symbol,Any}(
			:radius => nu.Δ, :rho => nu.ρ, :accepted => nu.accepted,
			:n_inner => nu.n_inner, :inner_core_ns => nu.inner_core_ns,
			:inner_stop => nu.inner_stop, :sub_logs => nu.sub_logs,
		),
	)
	length(st.iterate.x) <= 2 && (entry.extras[:x_iter] = copy(st.iterate.x))
	return entry
end
