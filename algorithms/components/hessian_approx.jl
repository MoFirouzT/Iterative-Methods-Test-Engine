"""
    hessian_approx.jl — Hessian-approximation components (content)

Shared `HessianApprox` vocabulary for quasi-Newton-style methods. No consumer
ships yet (these are roadmap prune-candidates); they live here so the engine
core stays free of method-construction vocabulary.
"""

using Base: @kwdef


"""
	abstract type HessianApprox

Base type for Hessian approximation strategies.
"""
abstract type HessianApprox end

"""
	struct FullHessian <: HessianApprox

Dense exact Hessian.
"""
struct FullHessian <: HessianApprox end

"""
	struct BFGS <: HessianApprox

BFGS rank-2 update.
"""
struct BFGS <: HessianApprox end

"""
	struct SR1 <: HessianApprox

Symmetric rank-1 update.
"""
struct SR1 <: HessianApprox end

"""
	struct LBFGS <: HessianApprox

Limited-memory BFGS update.
"""
@kwdef struct LBFGS <: HessianApprox
	m::Int = 5
end

"""
	struct DiagBFGS <: HessianApprox

Diagonal BFGS approximation.
"""
@kwdef struct DiagBFGS <: HessianApprox
	damped::Bool = false
end
