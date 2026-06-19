using Test
using LinearAlgebra: norm
using Random: AbstractRNG, default_rng

include(joinpath(@__DIR__, "..", "experiments", "_bootstrap.jl"))
include(joinpath(@__DIR__, "testutils.jl"))
import .TestEngine: grad!, init_state, step!   # engine dispatch points these fixtures extend

# ── Dummy problem + methods: the minimal fixtures the core abstractions need ──
struct DummyProblem
    n::Int
    x0::Vector{Float64}
    x_opt::Union{Nothing,Vector{Float64}}
end
# Back-compatible 2-arg form; the runner reads problem.x_opt (no known optimum here).
DummyProblem(n::Int, x0::Vector{Float64}) = DummyProblem(n, x0, nothing)

f(problem::DummyProblem, x::Vector{Float64}) = sum(abs2, x)

function grad!(gradient::Vector{Float64}, problem::DummyProblem, x::Vector{Float64})
    @assert length(gradient) == problem.n
    for index in eachindex(gradient, x)
        gradient[index] = 2.0 * x[index]
    end
    gradient
end

struct SimpleMethod <: IterativeMethod end
struct DummyMethod <: IterativeMethod end

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
        for value in 1:50_000        # measurable work so the timed region is non-trivial
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


@testset "Core abstraction" begin
    problem = DummyProblem(3, [2.0, -1.0, 0.5])

    @test IterativeMethod <: Any
    @test DummyMethod <: IterativeMethod   # single method category; role is experiment-level metadata

    empty_logger = silent_logger("Dummy")
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
    # iter_logs[1] is the iter=0 init snapshot (untimed); the real iteration is iter_logs[end].
    @test length(result.iter_logs) == 2
    @test result.iter_logs[end].objective == result.final_state.metrics.objective

    # Core-time plumbing contract (not a wall-clock magnitude bet): the init
    # snapshot is untimed, the real iteration carries the measured work, and the
    # logger's running total is exactly the sum of the per-iteration core times.
    @test result.iter_logs[1].core_time_ns == 0
    @test result.iter_logs[end].core_time_ns > 0
    @test result.final_state.timing.core_time_ns > 0
    @test empty_logger.total_core_ns == sum(e.core_time_ns for e in result.iter_logs)
    @test empty_logger.total_core_ns == result.iter_logs[end].core_time_ns
end

@testset "Nested runner" begin
    problem = DummyProblem(2, [1.0, -0.5])
    outer_logger = silent_logger("Dummy")

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

    outer_logger_2 = silent_logger("Dummy")
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
