"""
    regularizers.jl — Concrete regularizers (content, not engine)

Concrete `Regularizer` implementations that plug into the engine through the
`value` / `prox` contract. These live in the content layer so the engine core
stays free of specific penalty definitions — and free of whatever library is
used to implement them.

**Backend.** The proximal/value math is delegated to
[`ProximalOperators.jl`](https://github.com/JuliaFirstOrder/ProximalOperators.jl)
rather than hand-rolled. Each regularizer is a thin wrapper: it keeps a tidy
public face (`L1Norm(λ)`, the `.λ` field, the engine's `value`/`prox` contract
returning plain `Vector{Float64}`) and forwards the actual computation to a
stored ProximalOperators function. Consumers (`ProximalGradient`, experiments,
tests) never see the backend — swapping it out again would touch only this file.

An inner constructor builds the backend operator from `λ`, so the two can never
drift apart.

**No co-located design note.** Unlike the problem families alongside this
directory, regularizers are *components* — the `g(x)` vocabulary of the composite
objective `f + Σ gᵢ`, not a problem family with its own math or win conditions.
The `Regularizer`/`value`/`prox` contract lives in the engine and is documented
in `docs/src/modules/problem-interface.md` (§Regularizer); the consumer behavior
(soft-thresholding, ridge shrinkage, zero ⇒ plain gradient descent) is in
`algorithms/proximal_gradient/proximal_gradient.md`.
"""

import .TestEngine: Regularizer, value, prox
import ProximalOperators as PO
using LinearAlgebra: norm


"""
    L1Norm <: Regularizer

ℓ₁ regularization: g(x) = λ ‖x‖₁. Backed by `ProximalOperators.NormL1(λ)`;
its `prox` is soft-thresholding at level `γλ`.
"""
struct L1Norm <: Regularizer
    λ::Float64
    op::PO.NormL1{Float64}
    L1Norm(λ::Real) = new(Float64(λ), PO.NormL1(Float64(λ)))
end
L1Norm(; λ::Real = 0.01) = L1Norm(λ)


value(g::L1Norm, x::Vector{Float64}) = g.op(x)
prox(g::L1Norm, x::Vector{Float64}, γ::Float64) = first(PO.prox(g.op, x, γ))


"""
    L2Norm <: Regularizer

ℓ₂ (ridge) regularization: g(x) = λ ‖x‖². Backed by
`ProximalOperators.SqrNormL2(2λ)` (which encodes `(2λ/2)‖x‖² = λ‖x‖²`), so its
`prox` is the shrinkage `x / (1 + 2γλ)`.
"""
struct L2Norm <: Regularizer
    λ::Float64
    op::PO.SqrNormL2{Float64}
    L2Norm(λ::Real) = new(Float64(λ), PO.SqrNormL2(2 * Float64(λ)))
end
L2Norm(; λ::Real = 0.01) = L2Norm(λ)


value(g::L2Norm, x::Vector{Float64}) = g.op(x)
prox(g::L2Norm, x::Vector{Float64}, γ::Float64) = first(PO.prox(g.op, x, γ))


"""
    ZeroRegularizer <: Regularizer

No-op regularizer (always zero). Backed by `ProximalOperators.Zero`, whose
`prox` is the identity — so `ProximalGradient` with a `ZeroRegularizer` reduces
to (accelerated) gradient descent on the smooth part.
"""
struct ZeroRegularizer <: Regularizer
    op::PO.Zero
    ZeroRegularizer() = new(PO.Zero())
end


value(g::ZeroRegularizer, x::Vector{Float64}) = g.op(x)
prox(g::ZeroRegularizer, x::Vector{Float64}, γ::Float64) = first(PO.prox(g.op, x, γ))
