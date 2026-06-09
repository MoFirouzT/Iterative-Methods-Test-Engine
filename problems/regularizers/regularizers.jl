"""
    regularizers.jl — Concrete regularizers (content, not engine)

Concrete `Regularizer` implementations that plug into the engine through the
`value` / `prox` contract. These live in the content layer so the engine core
stays free of specific penalty definitions — and free of whatever library is
used to implement them.
"""

import .TestEngine: Regularizer, value, prox
using Base: @kwdef
using LinearAlgebra: norm


"""
	L1Norm <: Regularizer

ℓ₁ regularization: g(x) = λ ‖x‖₁
"""
@kwdef struct L1Norm <: Regularizer
	λ::Float64 = 0.01
end


function value(g::L1Norm, x::Vector{Float64})
	g.λ * norm(x, 1)
end


function prox(g::L1Norm, x::Vector{Float64}, γ::Float64)
	sign.(x) .* max.(abs.(x) .- γ * g.λ, 0.0)
end


"""
	L2Norm <: Regularizer

ℓ₂ (ridge) regularization: g(x) = λ ‖x‖²
"""
@kwdef struct L2Norm <: Regularizer
	λ::Float64 = 0.01
end


function value(g::L2Norm, x::Vector{Float64})
	g.λ * norm(x)^2
end


function prox(g::L2Norm, x::Vector{Float64}, γ::Float64)
	x ./ (1.0 + 2.0 * γ * g.λ)
end


"""
	ZeroRegularizer <: Regularizer

No-op regularizer (always zero).
"""
struct ZeroRegularizer <: Regularizer end


function value(g::ZeroRegularizer, x::Vector{Float64})
	0.0
end


function prox(g::ZeroRegularizer, x::Vector{Float64}, γ::Float64)
	copy(x)
end
