"""
    lasso.jl — sparse recovery problem family `:lasso` (content, not engine)

    min_x  ½‖A x − b‖²  +  λ‖x‖₁

Composes existing content — `LeastSquares` data fidelity (problems/least_squares)
and `L1Norm` (problems/regularizers) — so this file only registers a random
generator. See `lasso.md` for the regime and the `x_opt = x_star` caveat.
"""

import .TestEngine: Problem, Regularizer, register_random_problem!
using Random: randn, randperm
using LinearAlgebra: opnorm


register_random_problem!(:lasso, (rng, p) -> begin
    m = get(p, :m, 128)
    n = get(p, :n, 256)
    k = get(p, :k, 10)
    λ = get(p, :λ, 0.1)

    A = randn(rng, m, n) ./ sqrt(m)                 # benign Gaussian, near-isometric columns
    supp = randperm(rng, n)[1:k]                    # true support
    x_star = zeros(n)
    x_star[supp] = sign.(randn(rng, k))             # ±1 planted signal
    b = A * x_star .+ 0.01 .* randn(rng, m)         # noisy measurements

    L = opnorm(A)^2                                 # Lipschitz const of ∇(½‖A·−b‖²); feeds FixedStep(1/L)

    Problem(
        LeastSquares(LeastSquaresKernel(A, b)),
        Regularizer[L1Norm(λ)],                     # NB: typed Regularizer[...], not a bare [...], so the field eltype stays invariant
        zeros(n);
        meta  = Dict{Symbol,Any}(:L => L, :support => sort(supp), :λ => λ),
        x_opt = x_star,                             # the PLANTED signal, not the lasso minimizer (see lasso.md)
    )
end)
