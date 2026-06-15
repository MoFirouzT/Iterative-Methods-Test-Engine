"""
    preconditioners.jl — Preconditioner components (content)

A `Preconditioner` supplies `M⁻¹` for a preconditioned gradient direction
`d = −M⁻¹ ∇f(x)`. Composed by `PreconditionedGradient` (the experimental
method) crossed with a step-size axis. The engine machinery never references
these; they belong to the method-construction layer.
"""

import .TestEngine: Hessian, MatrixHessian, DiagonalHessian, CountingHessian, hessian, diagonal
using LinearAlgebra: norm


# ─────────────────────────────────────────────────────────────────────────
# Abstraction
# ─────────────────────────────────────────────────────────────────────────

"""
    abstract type Preconditioner

Base type for preconditioners. Each concrete subtype implements
`precondition(M, g, problem, x) -> M⁻¹·g`.
"""
abstract type Preconditioner end


"""
    precondition(M::Preconditioner, g, problem, x) -> Vector{Float64}

Apply the inverse preconditioner to the gradient: return `M⁻¹·g`. The
preconditioned descent direction is then `−M⁻¹·g`.

A pure kernel (no internal control flow): `PreconditionedGradient.step!` times it
inside `@core_timed`, like `grad!`, rather than the kernel self-timing.
"""
function precondition end


# ─────────────────────────────────────────────────────────────────────────
# Diagonal-availability predicate
#
# `diagonal(H)` is an OPTIONAL part of the Hessian contract — declared only by
# the Hessian types that can supply it. The engine's fallback `diagonal(::Hessian)`
# throws, and `applicable(diagonal, H)` can't distinguish "has a real method"
# from "matches the throwing fallback", so we trait it explicitly here.
# ─────────────────────────────────────────────────────────────────────────

_supports_diagonal(::Hessian)         = false
_supports_diagonal(::DiagonalHessian) = true
_supports_diagonal(::MatrixHessian)   = true
# Stay transparent under opt-in oracle counting: forward through the wrapper.
_supports_diagonal(H::CountingHessian) = _supports_diagonal(H.inner)


# ─────────────────────────────────────────────────────────────────────────
# Identity — no preconditioning (M⁻¹ = I); reduces to plain gradient descent.
# ─────────────────────────────────────────────────────────────────────────

struct IdentityPreconditioner <: Preconditioner end

precondition(::IdentityPreconditioner, g::Vector{Float64}, problem, x::Vector{Float64}) = g


# ─────────────────────────────────────────────────────────────────────────
# Jacobi — M⁻¹ = diag(∇²f)⁻¹.
#
# On a problem whose Hessian is diagonal (DiagonalHessian) this is EXACTLY
# Newton's method: d = −diag(H)⁻¹ ∇f, and a unit step lands on the minimizer.
# It requires the Hessian to expose `diagonal`; that is a feature of the
# "each Hessian declares which operations it supports" contract — Jacobi is
# *correctly inapplicable* on an OperatorHessian, and says so with a clean
# error rather than a silent fallback.
# ─────────────────────────────────────────────────────────────────────────

struct JacobiPreconditioner <: Preconditioner end

function precondition(::JacobiPreconditioner, g::Vector{Float64}, problem, x::Vector{Float64})
	H = hessian(problem.f, x)
	_supports_diagonal(H) || throw(ArgumentError(
		"JacobiPreconditioner requires a Hessian exposing `diagonal` " *
		"(e.g. DiagonalHessian, MatrixHessian); got $(typeof(H)), which does not. " *
		"Use IdentityPreconditioner, or a problem whose hessian is diagonal-capable."))
	d = diagonal(H)
	return g ./ d
end
