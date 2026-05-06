"""
    Step-size rule components for conventional methods.

Defines the shared `StepSize` abstraction and the concrete rules used by
conventional algorithms and variant grids. `LineSearch` is a sub-hierarchy
for rules that perform actual 1D searches along the descent direction.

Concrete rules:
  - FixedStep        — constant α (StepSize)
  - ArmijoLS         — backtracking sufficient-decrease search (LineSearch)
  - CauchyStep       — exact quadratic line-search via local Hessian (StepSize)
  - BarzilaiBorwein  — secant-curvature step (StepSize)

State convention: this file assumes the state carries `iterate :: IterateGroup`,
`metrics :: MetricsGroup`, `timing :: TimingGroup`, and a `numerics` field
providing `x_trial :: Vector{Float64}`, `n_linesearch_evals :: Int`, and
`grad_prev :: Vector{Float64}`. These are guaranteed by
`GradientDescentNumerics`; see `gradient_descent.md` §3.3. `x_trial` MUST
be initialized in `init_state` (`similar(problem.x0)`); the line-search
rules use it as a non-allocating scratch buffer.
"""

using Base: @kwdef
using LinearAlgebra: dot


# ─────────────────────────────────────────────────────────────────────────
# Abstraction
# ─────────────────────────────────────────────────────────────────────────

abstract type StepSize end
abstract type LineSearch <: StepSize end


# ─────────────────────────────────────────────────────────────────────────
# Concrete Rules
# ─────────────────────────────────────────────────────────────────────────

"""
    FixedStep <: StepSize

Constant step size α applied at every iteration.
"""
@kwdef struct FixedStep <: StepSize
	α::Float64 = 1e-3
end

"""
    ArmijoLS <: LineSearch

Backtracking sufficient-decrease search. Starts at α₀, contracts by β until
the Armijo condition holds: f(x + αd) ≤ f(x) + c₁·α·∇f(x)ᵀd.
"""
@kwdef struct ArmijoLS <: LineSearch
	α₀::Float64   = 1.0
	β::Float64    = 0.5
	c₁::Float64   = 1e-4
	max_iter::Int = 50
end

"""
    CauchyStep <: StepSize

Closed-form step minimizing the quadratic Taylor expansion along d_k:
α = −∇f(x)ᵀd / dᵀ∇²f(x)d. Falls back to `fallback_α` when curvature is
non-positive (denominator ≤ ε_denom).
"""
@kwdef struct CauchyStep <: StepSize
	fallback_α::Float64 = 1e-3
	ε_denom::Float64    = 1e-14
end

"""
    BarzilaiBorwein <: StepSize

Secant-curvature step from two consecutive iterates. Two variants:
:BB1 (long step) α = sᵀs / sᵀy, :BB2 (short step) α = sᵀy / yᵀy, with
s = x_k − x_{k−1} and y = ∇f(x_k) − ∇f(x_{k−1}). Falls back to
`fallback_α` at k=1 (no previous data) or when sᵀy ≤ ε_denom.
"""
@kwdef struct BarzilaiBorwein <: StepSize
	variant::Symbol     = :BB1
	fallback_α::Float64 = 1e-3
	ε_denom::Float64    = 1e-14
end


# ─────────────────────────────────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────────────────────────────────

"""
    compute_step_size(rule, state, problem, direction) -> Float64

Compute α_k for the update x_{k+1} = x_k + α_k·d_k. Each concrete rule
wraps its own core mathematical computation in `@core_timed state`. See
`step_sizes.md` for preconditions, timing discipline, and per-rule
derivations.
"""
function compute_step_size(rule::StepSize, state, problem, direction::Vector{Float64})::Float64
	throw(MethodError(compute_step_size, (rule, state, problem, direction)))
end


# ── FixedStep ────────────────────────────────────────────────────────────
compute_step_size(rule::FixedStep, state, problem, direction::Vector{Float64})::Float64 = rule.α


# ── ArmijoLS ─────────────────────────────────────────────────────────────
function compute_step_size(rule::ArmijoLS, state, problem, direction::Vector{Float64})::Float64
	x_k     = state.iterate.x
	f_k     = state.metrics.objective
	x_trial = state.numerics.x_trial
	α       = rule.α₀

	@core_timed state begin
		slope = dot(state.iterate.gradient, direction)         # ∇f(x_k)ᵀd_k < 0
		for _ in 1:rule.max_iter
			x_trial .= x_k .+ α .* direction
			f_trial  = total_objective(problem, x_trial)
			state.numerics.n_linesearch_evals += 1
			f_trial <= f_k + rule.c₁ * α * slope && return α
			α *= rule.β
		end
	end
	return α   # max_iter exhausted; return last (very small) α
end


# ── CauchyStep ───────────────────────────────────────────────────────────
function compute_step_size(rule::CauchyStep, state, problem, direction::Vector{Float64})::Float64
	@core_timed state begin
		H   = hessian(problem.f, state.iterate.x)              # ∇²f(x_k) — Hessian object
		Hd  = apply(H, direction)                              # ∇²f(x_k)·d_k
		num = dot(state.iterate.gradient, direction)           # ∇f(x_k)ᵀd_k
		den = dot(direction, Hd)                               # d_kᵀ∇²f(x_k)d_k
		den <= rule.ε_denom && return rule.fallback_α          # non-positive curvature
		return -num / den
	end
end


# ── BarzilaiBorwein ──────────────────────────────────────────────────────
function compute_step_size(rule::BarzilaiBorwein, state, problem, direction::Vector{Float64})::Float64
	rule.variant ∈ (:BB1, :BB2) || throw(ArgumentError(
		"unknown BarzilaiBorwein variant $(rule.variant); expected :BB1 or :BB2"))

	# First iteration: no previous iterate/gradient yet
	if isempty(state.iterate.x_prev) || isempty(state.numerics.grad_prev)
		return rule.fallback_α
	end

	@core_timed state begin
		s  = state.iterate.x        .- state.iterate.x_prev      # s_{k-1}
		y  = state.iterate.gradient .- state.numerics.grad_prev  # y_{k-1}
		sy = dot(s, y)
		sy <= rule.ε_denom && return rule.fallback_α             # curvature guard (BEFORE division)
		return rule.variant === :BB1 ? dot(s, s) / sy : sy / dot(y, y)
	end
end