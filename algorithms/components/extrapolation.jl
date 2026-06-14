"""
    extrapolation.jl — Extrapolation (post-step correction) components (content)

Shared `Extrapolation` vocabulary that methods compose — e.g. `NesterovStep`
turns ProximalGradient (ISTA) into FISTA. The engine machinery never
references these; they belong to the method-construction layer.
"""

using Base: @kwdef
using .TestEngine: register_abbreviation!


"""
	abstract type Extrapolation

Base type for post-step correction strategies.
"""
abstract type Extrapolation end

"""
	struct NoExtrapolation <: Extrapolation

No-op correction.
"""
struct NoExtrapolation <: Extrapolation end

"""
	struct MomentumStep <: Extrapolation

Simple momentum correction.
"""
@kwdef struct MomentumStep <: Extrapolation
	α::Float64 = 0.1
end

"""
	struct NesterovStep <: Extrapolation

Nesterov-style correction.
"""
@kwdef struct NesterovStep <: Extrapolation
	α::Float64 = 0.1
end


# ─────────────────────────────────────────────────────────────────────────
# Behavior — consumed by composing methods (e.g. ProximalGradient)
#
# A method calls `extrapolate` to choose the gradient-evaluation point, then
# `advance_momentum` after the iterate update. The no-op pair below makes a
# composing method behave as its plain (un-accelerated) variant.
# ─────────────────────────────────────────────────────────────────────────

"""
	extrapolate(mu::Extrapolation, x, x_prev, t) -> y

The point at which the gradient is evaluated this step.
- `NoExtrapolation` ⇒ `y = x` (plain proximal gradient / ISTA).
- `NesterovStep`  ⇒ FISTA extrapolation `y = x + β·(x − x_prev)` with
  `β = (t − 1)/t_next`, `t_next = (1 + √(1 + 4t²))/2`. The same `t_next`
  formula is used by `advance_momentum`, so passing the *current* `t` to both
  keeps `β` and the momentum advance consistent.
- `MomentumStep`  ⇒ heavy-ball `y = x + α·(x − x_prev)` with fixed `α`.

`x_prev` empty (the first step's sentinel) ⇒ `y = x` for every variant.
"""
extrapolate(::NoExtrapolation, x::Vector{Float64}, x_prev::Vector{Float64}, t::Float64) = copy(x)

function extrapolate(::NesterovStep, x::Vector{Float64}, x_prev::Vector{Float64}, t::Float64)
	isempty(x_prev) && return copy(x)
	t_next = (1 + sqrt(1 + 4t^2)) / 2
	β = (t - 1) / t_next
	return x .+ β .* (x .- x_prev)
end

function extrapolate(mu::MomentumStep, x::Vector{Float64}, x_prev::Vector{Float64}, t::Float64)
	isempty(x_prev) && return copy(x)
	return x .+ mu.α .* (x .- x_prev)
end


"""
	advance_momentum(mu::Extrapolation, t) -> t_next

Advance the momentum parameter after the iterate update. FISTA advances
`t_next = (1 + √(1 + 4t²))/2`; the no-op and the fixed-`α` heavy-ball leave
`t` unchanged.
"""
advance_momentum(::NoExtrapolation, t::Float64) = t
advance_momentum(::NesterovStep, t::Float64) = (1 + sqrt(1 + 4t^2)) / 2
advance_momentum(::MomentumStep, t::Float64) = t


# Friendly short names for variant-grid short_name / plot legends — this content
# registers its own vocabulary with the engine's abbreviation table on load.
register_abbreviation!("NoExtrapolation", "∅")
register_abbreviation!("Momentum",      "Mom")
register_abbreviation!("MomentumStep",  "Mom")
register_abbreviation!("Nesterov",      "Nest")
register_abbreviation!("NesterovStep",  "Nest")
