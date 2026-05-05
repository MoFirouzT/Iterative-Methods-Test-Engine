"""
	Step-size rule components for conventional methods.

Defines the shared `StepSizeRule` abstraction and the concrete rules used by
conventional algorithms and variant grids.
"""

using Base: @kwdef
using LinearAlgebra: dot


# ─────────────────────────────────────────────────────────────────────────
# Abstraction
# ─────────────────────────────────────────────────────────────────────────

"""
	abstract type StepSize end

Base type for step-size rules. `LineSearch` is a subtype for rules that
perform actual 1D searches along the descent direction.
"""
abstract type StepSize end
abstract type LineSearch <: StepSize end


# ─────────────────────────────────────────────────────────────────────────
# Concrete Rules
# ─────────────────────────────────────────────────────────────────────────

"""
	struct FixedStep <: StepSizeRule

Constant step size.
"""
@kwdef struct FixedStep <: StepSize
	α::Float64 = 1e-3
end

"""
	struct ArmijoLS <: StepSizeRule

Armijo backtracking search.
"""
@kwdef struct ArmijoLS <: LineSearch
	α₀::Float64 = 1.0
	β::Float64 = 0.5
	c₁::Float64 = 1e-4
	max_iter::Int = 50
end

"""
	struct WolfeLS <: StepSizeRule

Wolfe line search.
"""
@kwdef struct WolfeLS <: LineSearch
	α₀::Float64 = 1.0
	β::Float64 = 0.5
	c₁::Float64 = 1e-4
	c₂::Float64 = 0.9
	max_iter::Int = 50
end

"""
	struct CauchyStep <: StepSizeRule

Exact quadratic line-search step based on the local Hessian model.
"""
@kwdef struct CauchyStep <: StepSize
	fallback_α::Float64 = 1e-3
	ε_denom::Float64 = 1e-14
end

"""
	struct BarzilaiBorwein <: StepSizeRule

Curvature-based step size from two consecutive iterates.
"""
@kwdef struct BarzilaiBorwein <: StepSize
	variant::Symbol = :BB1
	fallback_α::Float64 = 1e-3
	ε_denom::Float64 = 1e-14
end


# ─────────────────────────────────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────────────────────────────────

function compute_step_size(rule::StepSize, state, problem, direction::Vector{Float64})
	throw(MethodError(compute_step_size, (rule, state, problem, direction)))
end

compute_step_size(rule::FixedStep, state, problem, direction::Vector{Float64}) = rule.α

@inline function _increment_linesearch_evals!(state)
	hasproperty(state, :numerics) && hasproperty(state.numerics, :n_linesearch_evals) || return
	state.numerics.n_linesearch_evals += 1
end

@inline function _objective_eval(state, problem, x::Vector{Float64})
	@core_timed state begin
		total_objective(problem, x)
	end
end

function compute_step_size(rule::ArmijoLS, state, problem, direction::Vector{Float64})::Float64
	x_k = state.iterate.x
	f_k = state.metrics.objective
	slope = dot(state.iterate.gradient, direction)

	α = rule.α₀
	for _ in 1:rule.max_iter
		f_trial = _objective_eval(state, problem, x_k .+ α .* direction)
		_increment_linesearch_evals!(state)

		f_trial <= f_k + rule.c₁ * α * slope && return α
		α *= rule.β
	end

	return α
end

function compute_step_size(rule::WolfeLS, state, problem, direction::Vector{Float64})::Float64
	x_k = state.iterate.x
	g_k = state.iterate.gradient
	f_k = state.metrics.objective
	slope_0 = dot(g_k, direction)
	trial_gradient = similar(g_k)

	α = rule.α₀
	for _ in 1:rule.max_iter
		x_trial = x_k .+ α .* direction
		f_trial = _objective_eval(state, problem, x_trial)
		_increment_linesearch_evals!(state)

		if f_trial > f_k + rule.c₁ * α * slope_0
			α *= rule.β
			continue
		end

		@core_timed state begin
			grad!(trial_gradient, problem.f, x_trial)
		end
		if abs(dot(trial_gradient, direction)) <= rule.c₂ * abs(slope_0)
			return α
		end

		α *= rule.β
	end

	return α
end

function compute_step_size(rule::CauchyStep, state, problem, direction::Vector{Float64})::Float64
	local Hd, num, den
	@core_timed state begin
		H = hessian(problem.f, state.iterate.x)
		Hd = apply(H, direction)
		num = dot(state.iterate.gradient, direction)
		den = dot(direction, Hd)
	end

	den <= rule.ε_denom && return rule.fallback_α
	return -num / den
end

function compute_step_size(rule::BarzilaiBorwein, state, problem, direction::Vector{Float64})::Float64
	if !hasproperty(state, :iterate) || !hasproperty(state, :numerics) ||
	   !hasproperty(state.iterate, :x_prev) || !hasproperty(state.numerics, :grad_prev) ||
	   isempty(state.iterate.x_prev) || isempty(state.numerics.grad_prev)
		return rule.fallback_α
	end

	local s, y, sy, α
	@core_timed state begin
		s = state.iterate.x .- state.iterate.x_prev
		y = state.iterate.gradient .- state.numerics.grad_prev
		sy = dot(s, y)

		if rule.variant == :BB1
			α = dot(s, s) / sy
		elseif rule.variant == :BB2
			α = sy / dot(y, y)
		else
			throw(ArgumentError("unknown BarzilaiBorwein variant $(rule.variant); expected :BB1 or :BB2"))
		end
	end

	sy <= rule.ε_denom && return rule.fallback_α
	return α
end
