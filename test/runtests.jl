using Test
using LinearAlgebra: I, norm
using Random: AbstractRNG, default_rng

# Engine + all content via the shared bootstrap (idempotent).
include(joinpath(@__DIR__, "..", "experiments", "_bootstrap.jl"))
import .TestEngine: grad!, init_state, step!   # engine dispatch points these tests extend

# Content conformance harness — runs FIRST so its completeness guard sees only
# registered content (before any test file registers a throwaway problem).
include(joinpath(@__DIR__, "test_problem_contract.jl"))

struct DummyProblem
    n::Int
    x0::Vector{Float64}
    x_opt::Union{Nothing,Vector{Float64}}
end
# Back-compatible 2-arg form; the runner now reads problem.x_opt (no known optimum here).
DummyProblem(n::Int, x0::Vector{Float64}) = DummyProblem(n, x0, nothing)

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
        VerbosityConfig(),
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
    x_trial::Vector{Float64} = Float64[]   # scratch buffer the ArmijoLS path writes into
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

function step!(::SimpleMethod, state::SimpleState, problem::DummyProblem, iter::Int, logger::Logger, rng::AbstractRNG)
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


@testset "Module 1 core abstraction" begin
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
    @test_throws MethodError step!(UnimplementedMethod(), empty_state, problem, 1, empty_logger, default_rng())

    result = run_method(SimpleMethod(), problem, MaxIterations(n = 1), empty_logger, default_rng())

    @test result.stop_reason == :max_iterations
    @test result.n_iters == 1
    # iter_logs[1] is the iter=0 init snapshot (core_time 0); the real iteration is iter_logs[end].
    @test length(result.iter_logs) == 2
    @test result.iter_logs[end].core_time_ns > 0
    @test result.iter_logs[end].objective == result.final_state.metrics.objective
    @test result.final_state.timing.core_time_ns > 0
    @test empty_logger.total_core_ns == result.iter_logs[end].core_time_ns
end

@testset "Module 4 nested runner skeleton" begin
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
    @test length(sub.iter_logs) == 3   # iter=0 init entry + 2 iterations
    @test sub.core_time_ns == sum(log.core_time_ns for log in sub.iter_logs)
    @test length(outer_logger.pending_sub_logs) == 3

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
    register_abbreviation!("MyMethod", "MM")   # test fixture's own short name
    axis1 = VariantAxis(:preconditioner,
        IdentityPreconditioner() => "Identity",
        JacobiPreconditioner()   => "Jacobi",
    )

    axis2 = VariantAxis(:linesearch,
        ArmijoLS() => "Armijo",
    )

    grid = VariantGrid(
        base_name = "MyMethod",
        axes = [axis1, axis2],
        builder = (; preconditioner, linesearch, step_size) -> DummyMethod(),
        shared_params = (; step_size = 0.01),
    )

    specs = expand(grid)

    @test length(specs) == 2   # 2 preconditioner values × 1 linesearch value
    @test specs[1].name == "MyMethod[preconditioner=Identity,linesearch=Armijo,step_size=0.01]"
    @test specs[1].short_name == "MM/Identity/Arm"
    @test specs[1].params == (; preconditioner = IdentityPreconditioner(), linesearch = ArmijoLS(), step_size = 0.01)
    @test specs[1].method isa DummyMethod
end

include(joinpath(@__DIR__, "test_module5.jl"))
include(joinpath(@__DIR__, "test_module7.jl"))
include(joinpath(@__DIR__, "test_module8.jl"))
include(joinpath(@__DIR__, "test_module9.jl"))   # Problem Factory: exercises the moved LeastSquares/regularizer content
include(joinpath(@__DIR__, "test_proximal_gradient.jl"))   # ProximalGradient: ISTA↔GD reduction, FISTA acceleration, one-prox-per-step
include(joinpath(@__DIR__, "test_least_squares.jl"))       # LeastSquares Hessian modes, :linear_ls conditioning, Cauchy-guard regression
include(joinpath(@__DIR__, "test_preconditioned_gradient.jl"))   # PreconditionedGradient: Jacobi=Newton, dual-bucket routing, diagonal contract
include(joinpath(@__DIR__, "test_external_validation.jl"))       # cross-check converged solutions vs Optim.jl + ProximalAlgorithms.jl
include(joinpath(@__DIR__, "test_trust_region.jl"))              # TrustRegion + Steihaug-CG: branches, nesting, core-time attribution