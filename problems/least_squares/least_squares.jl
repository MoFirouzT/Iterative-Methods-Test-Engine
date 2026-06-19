"""
    least_squares.jl — Linear least-squares objective (content, not engine)

    f(x) = ½‖Ax − b‖²

Plugs into the engine through the `value` / `grad!` / `hessian` contract.
"""

import .TestEngine: Objective, Hessian, MatrixHessian, OperatorHessian, Problem,
    value, grad!, hessian, register_analytic_problem!, register_random_problem!
using LinearAlgebra: norm, mul!, adjoint, qr, Diagonal, I
using Random: randn


"""
    LeastSquaresKernel

Encapsulates the data matrix A and vector b for least-squares data fidelity.

`residual` is a preallocated length-m scratch buffer (`m = length(b)`) reused by
`value`/`grad!` so the per-step kernel allocates nothing — the matvecs write
through `mul!` instead of allocating `A*x` and `A*x − b` each call. It is a
single-threaded scratch: every call fully overwrites it from `x` before reading,
so sequential reuse (the engine runs methods sequentially) is safe; do not share
one kernel across threads.

`AtA` memoizes `AᵀA` for the `:matrix`-mode Hessian: `A` is fixed, so the n×n
product is constant and formed at most once (lazily, on the first matrix-mode
`hessian` call) instead of on every call. It stays `nothing` in `:operator`
mode, which never materializes it. This is why the kernel is `mutable`.
"""
mutable struct LeastSquaresKernel
    A::Matrix{Float64}
    b::Vector{Float64}
    residual::Vector{Float64}
    AtA::Union{Nothing,Matrix{Float64}}
end

# Convenience constructor: allocate the scratch buffer once. All call sites use
# this 2-arg form; the 4-arg inner constructor is for completeness.
LeastSquaresKernel(A::Matrix{Float64}, b::Vector{Float64}) =
    LeastSquaresKernel(A, b, similar(b), nothing)


"""
    LeastSquares <: Objective

Least-squares objective: f(x) = 0.5 ‖Ax − b‖².

The Hessian ∇²f = AᵀA is constant. Its **representation is selectable** via
`hessian_mode` so the same objective can serve both regimes:

- `:matrix`   — materialize `MatrixHessian(AᵀA)`. The default (preserves all
  prior behavior); needed where a method reads `materialize`/`diagonal`
  (e.g. a Jacobi preconditioner). Cost: forms the n×n matrix once per call.
- `:operator` — `OperatorHessian(d -> Aᵀ(A d), n)`, never materialized. The
  scalable mode for large n (`apply` is two O(mn) matvecs); used by the
  `:linear_ls` conditioning family. `CauchyStep` only calls `apply`, so it
  works in either mode.
"""
struct LeastSquares <: Objective
    kernel::LeastSquaresKernel
    hessian_mode::Symbol
end

# Default to :matrix so existing call sites (lasso, :quadratic, tests) are unchanged.
LeastSquares(kernel::LeastSquaresKernel) = LeastSquares(kernel, :matrix)


function value(f::LeastSquares, x::Vector{Float64})
    r = f.kernel.residual
    mul!(r, f.kernel.A, x)        # r ← A x
    r .-= f.kernel.b              # r ← A x − b   (in place, no temporary)
    return 0.5 * sum(abs2, r)
end


function grad!(g::Vector{Float64}, f::LeastSquares, x::Vector{Float64})::Vector{Float64}
    r = f.kernel.residual
    mul!(r, f.kernel.A, x)             # r ← A x
    r .-= f.kernel.b                   # r ← A x − b
    mul!(g, adjoint(f.kernel.A), r)    # g ← Aᵀ r      (residual buffer reused, no alloc)
    return g
end


function hessian(f::LeastSquares, x::Vector{Float64})::Hessian
    k = f.kernel
    A = k.A
    if f.hessian_mode === :operator
        n  = size(A, 2)
        Ad = similar(k.b)                          # m-vector scratch, reused across applies of THIS Hessian
        # AᵀA d, never materialized; mul! into Ad avoids the A*d temporary each apply.
        # The result is a fresh n-vector (callers use it immediately, never aliased).
        return OperatorHessian(d -> A' * mul!(Ad, A, d), n)
    elseif f.hessian_mode === :matrix
        k.AtA === nothing && (k.AtA = A' * A)      # constant; form at most once
        return MatrixHessian(k.AtA)
    else
        throw(ArgumentError("LeastSquares hessian_mode must be :matrix or :operator, got :$(f.hessian_mode)"))
    end
end


# ─────────────────────────────────────────────────────────────────────────
# Built-in registration: a generic quadratic least-squares family `:quadratic`
# (a quadratic f(x)=½‖Ax−b‖² is just a LeastSquares instance, so it lives here).
# ─────────────────────────────────────────────────────────────────────────

register_analytic_problem!(:quadratic, (params, rng) -> begin
    A  = get(params, :A, Matrix{Float64}(I, 2, 2))
    b  = get(params, :b, zeros(2))
    x0 = get(params, :x0, zeros(length(b)))
    Problem(LeastSquares(LeastSquaresKernel(A, b)), x0)
end)


# ─────────────────────────────────────────────────────────────────────────
# `:linear_ls` — conditioning- and dimension-parametrized least squares.
#
# Parametrized by the HESSIAN condition number κ = cond(AᵀA), which is what
# governs the GD convergence rate — NOT cond(A). Since cond(AᵀA) = cond(A)²,
# we set A's singular values to span 1 → κ^(-1/2), giving cond(A) = √κ and thus
# cond(AᵀA) = κ. The system is CONSISTENT (b = A·x_star), so the minimizer is
# unique, x_opt = x_star, and f(x_opt) = 0 — stop on gradient/distance, never on
# an f-value (relative-f is degenerate when f* = 0). Uses the :operator Hessian
# mode so OperatorHessian is exercised (CauchyStep applies it as two matvecs).
# ─────────────────────────────────────────────────────────────────────────

register_random_problem!(:linear_ls, (rng, p) -> begin
    n = get(p, :n, 100)
    m = get(p, :m, 2n)
    κ = get(p, :condition_number, 1.0e3)               # κ = cond(AᵀA), the rate-driver

    s = exp10.(range(0, -0.5 * log10(κ); length = n))  # σ: 1 → κ^(-1/2) ⇒ cond(A)=√κ
    U = Matrix(qr(randn(rng, m, n)).Q)[:, 1:n]         # m×n orthonormal columns (thin factor)
    V = Matrix(qr(randn(rng, n, n)).Q)                 # n×n orthonormal
    A = U * Diagonal(s) * V'

    x_star = randn(rng, n)
    b = A * x_star                                     # consistent ⇒ x_opt = x_star, f* = 0

    Problem(
        LeastSquares(LeastSquaresKernel(A, b), :operator),
        zeros(n);
        meta  = Dict{Symbol,Any}(:condition_number => κ, :L => maximum(s)^2, :m => m),
        x_opt = x_star,
    )
end)
