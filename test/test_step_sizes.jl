using Test
using LinearAlgebra: I, norm

include(joinpath(@__DIR__, "..", "experiments", "_bootstrap.jl"))
include(joinpath(@__DIR__, "testutils.jl"))

# Minimal mutable state mimicking a real method's state — just enough for the
# step-size rules to read/write: the iterate, its metrics, a timing slot, and the
# line-search scratch (eval counter + trial buffer the ArmijoLS path writes into).
@kwdef mutable struct StepSizeNumerics
    n_linesearch_evals::Int = 0
    grad_prev::Vector{Float64} = Float64[]
    x_trial::Vector{Float64} = Float64[]
end

@kwdef mutable struct StepSizeState
    iterate::IterateGroup
    metrics::MetricsGroup
    timing::TimingGroup
    numerics::StepSizeNumerics
end

@testset "Step size rules" begin
    a = Matrix{Float64}(I, 2, 2)
    problem = Problem(LeastSquares(LeastSquaresKernel(a, zeros(2))), zeros(2))

    x = [2.0, -1.0]
    gradient = zeros(2)
    grad!(gradient, problem.f, x)
    direction = -gradient

    state = StepSizeState(
        iterate = IterateGroup(x = copy(x), gradient = copy(gradient), x_prev = Float64[]),
        metrics = MetricsGroup(objective = objective(problem, x), gradient_norm = norm(gradient), step_norm = 0.0),
        timing = TimingGroup(core_time_ns = 0),
        numerics = StepSizeNumerics(x_trial = similar(x)),
    )

    @test compute_step_size(FixedStep(α = 0.25), state, problem, direction) == 0.25

    state.timing.core_time_ns = 0
    state.numerics.n_linesearch_evals = 0
    α_armijo = compute_step_size(ArmijoLS(α₀ = 1.0, β = 0.5, c₁ = 1e-4, max_iter = 10), state, problem, direction)
    @test α_armijo ≈ 1.0
    @test state.numerics.n_linesearch_evals == 1
    @test state.timing.core_time_ns > 0

    state.timing.core_time_ns = 0
    α_cauchy = compute_step_size(CauchyStep(), state, problem, direction)
    @test α_cauchy ≈ 1.0
    @test state.timing.core_time_ns > 0

    state.iterate.x_prev = [1.0, 0.0]
    state.numerics.grad_prev = [1.0, 0.0]
    state.iterate.x = [2.0, 0.0]
    grad!(state.iterate.gradient, problem.f, state.iterate.x)

    α_bb1 = compute_step_size(BarzilaiBorwein(variant = :BB1), state, problem, direction)
    α_bb2 = compute_step_size(BarzilaiBorwein(variant = :BB2), state, problem, direction)
    @test α_bb1 ≈ 1.0
    @test α_bb2 ≈ 1.0
end
