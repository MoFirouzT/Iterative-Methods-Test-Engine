using Test
using LinearAlgebra: I, norm
using Random: AbstractRNG, default_rng

include(joinpath(@__DIR__, "..", "src", "logging.jl"))
include(joinpath(@__DIR__, "..", "src", "core.jl"))
include(joinpath(@__DIR__, "..", "src", "stopping.jl"))
include(joinpath(@__DIR__, "..", "algorithms", "conventional", "components", "step_sizes.jl"))
include(joinpath(@__DIR__, "..", "src", "problems.jl"))
include(joinpath(@__DIR__, "..", "src", "variants.jl"))

struct DummyProblem
    n::Int
    x0::Vector{Float64}
end

f(problem::DummyProblem, x::Vector{Float64}) = sum(abs2, x)

function grad!(gradient::Vector{Float64}, problem::DummyProblem, x::Vector{Float64})
    @assert length(gradient) == problem.n
    for index in eachindex(gradient, x)
        gradient[index] = 2.0 * x[index]
    end
    gradient
end

function make_logger()
    Logger(
        "Dummy",
        1,
        "",
        nothing,
        IterationLog[],
        NamedTuple[],
        Dict{Symbol,Any}(),
        0.0,
        0,
        IterationLog[],
    )
end

struct SimpleMethod <: IterativeMethod
end

struct DummyMethod <: ExperimentalMethod
end

@kwdef mutable struct StepSizeNumerics
    n_linesearch_evals::Int = 0
    grad_prev::Vector{Float64} = Float64[]
end

@kwdef mutable struct StepSizeState
    iterate::IterateGroup
    metrics::MetricsGroup
    timing::TimingGroup
    numerics::StepSizeNumerics
end

@kwdef mutable struct SimpleState
    iterate::IterateGroup
    metrics::MetricsGroup
    timing::TimingGroup
    _logger::Union{Nothing,Any} = nothing
end

function init_state(::SimpleMethod, problem::DummyProblem, rng::AbstractRNG)
    x = copy(problem.x0)
    gradient = zeros(Float64, problem.n)
    grad!(gradient, problem, x)
    SimpleState(
        iterate = IterateGroup(x = x, gradient = gradient, x_prev = Float64[]),
        metrics = MetricsGroup(objective = f(problem, x), gradient_norm = norm(gradient), step_norm = 0.0),
        timing = TimingGroup(core_time_ns = 0),
    )
end

function step!(::SimpleMethod, state::SimpleState, problem::DummyProblem, iter::Int)
    @core_timed state begin
        acc = 0.0
        for value in 1:50_000
            acc += sqrt(float(value))
        end
        state.iterate.x_prev = copy(state.iterate.x)
        state.iterate.x .-= 0.1 .* state.iterate.x
        state.iterate.gradient .= 2.0 .* state.iterate.x
        state.metrics.objective = f(problem, state.iterate.x)
        state.metrics.gradient_norm = norm(state.iterate.gradient)
        state.metrics.step_norm = norm(state.iterate.x_prev .- state.iterate.x)
        state.metrics.objective += acc * 0.0
    end
end

struct UnimplementedMethod <: IterativeMethod end


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
        numerics = StepSizeNumerics(),
    )

    @test compute_step(FixedStep(α = 0.25), state, problem, direction) == 0.25

    state.timing.core_time_ns = 0
    state.numerics.n_linesearch_evals = 0
    α_armijo = compute_step(ArmijoLS(α₀ = 1.0, β = 0.5, c₁ = 1e-4, max_iter = 10), state, problem, direction)
    @test α_armijo ≈ 1.0
    @test state.numerics.n_linesearch_evals == 1
    @test state.timing.core_time_ns > 0

    state.timing.core_time_ns = 0
    state.numerics.n_linesearch_evals = 0
    α_wolfe = compute_step(WolfeLS(α₀ = 1.0, β = 0.5, c₁ = 1e-4, c₂ = 0.9, max_iter = 10), state, problem, direction)
    @test α_wolfe ≈ 1.0
    @test state.numerics.n_linesearch_evals == 1
    @test state.timing.core_time_ns > 0

    state.timing.core_time_ns = 0
    α_cauchy = compute_step(CauchyStep(), state, problem, direction)
    @test α_cauchy ≈ 1.0
    @test state.timing.core_time_ns > 0

    state.iterate.x_prev = [1.0, 0.0]
    state.numerics.grad_prev = [1.0, 0.0]
    state.iterate.x = [2.0, 0.0]
    grad!(state.iterate.gradient, problem.f, state.iterate.x)

    α_bb1 = compute_step(BarzilaiBorwein(variant = :BB1), state, problem, direction)
    α_bb2 = compute_step(BarzilaiBorwein(variant = :BB2), state, problem, direction)
    @test α_bb1 ≈ 1.0
    @test α_bb2 ≈ 1.0
end


@testset "Layer 1 core abstraction" begin
    problem = DummyProblem(3, [2.0, -1.0, 0.5])

    @test IterativeMethod <: Any
    @test ConventionalMethod <: IterativeMethod
    @test ExperimentalMethod <: IterativeMethod

    empty_logger = make_logger()
    empty_state = SimpleState(
        iterate = IterateGroup(x = copy(problem.x0), gradient = zeros(3), x_prev = Float64[]),
        metrics = MetricsGroup(),
        timing = TimingGroup(),
    )
    entry = extract_log_entry(SimpleMethod(), empty_state, 7)
    @test entry.iter == 7
    @test entry.objective == empty_state.metrics.objective
    @test entry.gradient_norm == empty_state.metrics.gradient_norm
    @test entry.step_norm == empty_state.metrics.step_norm
    @test isempty(entry.extras)

    @test_throws MethodError init_state(UnimplementedMethod(), problem, default_rng())
    @test_throws MethodError step!(UnimplementedMethod(), empty_state, problem, 1)

    result = run_method(SimpleMethod(), problem, MaxIterations(n = 1), empty_logger, default_rng())

    @test result.stop_reason == :max_iterations
    @test result.n_iters == 1
    @test length(result.iter_logs) == 1
    @test result.iter_logs[1].core_time_ns > 0
    @test result.iter_logs[1].objective == result.final_state.metrics.objective
    @test result.final_state.timing.core_time_ns > 0
    @test empty_logger.total_core_ns == result.iter_logs[1].core_time_ns
end

@testset "Layer 4 nested runner skeleton" begin
    problem = DummyProblem(2, [1.0, -0.5])
    outer_logger = make_logger()

    cfg = SubRunConfig(
        method = SimpleMethod(),
        criteria = MaxIterations(n = 2),
        log_sub_iters = true,
    )

    sub = run_sub_method(cfg, problem, outer_logger, default_rng())

    @test sub.stop_reason == :max_iterations
    @test sub.converged == false
    @test sub.n_iters == 2
    @test length(sub.iter_logs) == 2
    @test sub.core_time_ns == sum(log.core_time_ns for log in sub.iter_logs)
    @test length(outer_logger.pending_sub_logs) == 2

    outer_logger_2 = make_logger()
    cfg_no_attach = SubRunConfig(
        method = SimpleMethod(),
        criteria = MaxIterations(n = 1),
        log_sub_iters = false,
    )

    sub_no_attach = run_sub_method(cfg_no_attach, problem, outer_logger_2, default_rng())

    @test sub_no_attach.n_iters == 1
    @test isempty(outer_logger_2.pending_sub_logs)
end

@testset "Variant grid expansion" begin
    axis1 = VariantAxis(:hessian,
        BFGS() => "BFGS",
        SR1() => "SR1",
    )

    axis2 = VariantAxis(:linesearch,
        ArmijoLS() => "Armijo",
        WolfeLS() => "Wolfe",
    )

    grid = VariantGrid(
        base_name = "MyMethod",
        axes = [axis1, axis2],
        builder = (; hessian, linesearch, step_size) -> DummyMethod(),
        shared_params = (; step_size = 0.01),
    )

    specs = expand(grid)

    @test length(specs) == 4
    @test specs[1].name == "MyMethod[hessian=BFGS,linesearch=Armijo,step_size=0.01]"
    @test specs[1].short_name == "MM/BFGS/Arm"
    @test specs[1].params == (; hessian = BFGS(), linesearch = ArmijoLS(), step_size = 0.01)
    @test specs[1].method isa DummyMethod
end

include(joinpath(@__DIR__, "test_layer5.jl"))
include(joinpath(@__DIR__, "test_layer7.jl"))
include(joinpath(@__DIR__, "test_layer8.jl"))