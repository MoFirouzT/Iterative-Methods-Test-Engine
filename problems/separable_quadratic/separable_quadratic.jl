"""
    separable_quadratic.jl — Separable (diagonal) quadratic (content, not engine)

    f(x) = ½ Σᵢ dᵢ xᵢ²,   dᵢ > 0

The Hessian is the constant diagonal `diag(d)`, exposed as a `DiagonalHessian`
— so this family lights up `DiagonalHessian` and is the natural showcase for the
Jacobi preconditioner (on a diagonal Hessian, Jacobi is exactly Newton).
"""

import .TestEngine: Objective, Hessian, DiagonalHessian, Problem,
    value, grad!, hessian, register_random_problem!


"""
    SeparableQuadratic <: Objective

`f(x) = ½ Σ dᵢ xᵢ²`, with per-coordinate curvatures `d` (all positive). The
unique minimizer is `0`, with `f(0) = 0`.
"""
struct SeparableQuadratic <: Objective
    d::Vector{Float64}
end

value(f::SeparableQuadratic, x::Vector{Float64}) = 0.5 * sum(f.d .* x .^ 2)

function grad!(g::Vector{Float64}, f::SeparableQuadratic, x::Vector{Float64})::Vector{Float64}
    g .= f.d .* x
    return g
end

hessian(f::SeparableQuadratic, x::Vector{Float64})::Hessian = DiagonalHessian(f.d)


# ─────────────────────────────────────────────────────────────────────────
# `:separable_quadratic` — diagonal quadratic with controllable conditioning.
#
# Curvatures span [1/κ, 1] (NOT [1, κ]): with λ_max = 1, `FixedStep(α = 1)` is
# both stable for the unpreconditioned method (slow, ~κ iters) AND exactly the
# Newton step for the Jacobi preconditioner (1 iter) — so the step-size axis
# stays fair across preconditioners. x0 = ones(n); minimizer x_opt = 0.
# ─────────────────────────────────────────────────────────────────────────

register_random_problem!(:separable_quadratic, (rng, p) -> begin
    n = get(p, :n, 50)
    κ = get(p, :condition_number, 1.0e4)
    d = exp10.(range(0, -log10(κ); length = n))      # 1 → 1/κ  ⇒  λ_max = 1, cond = κ
    Problem(
        SeparableQuadratic(d),
        ones(n);
        meta  = Dict{Symbol,Any}(:condition_number => κ, :L => maximum(d)),
        x_opt = zeros(n),
    )
end)
