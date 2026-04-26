using Test
using LinearAlgebra: I, norm
using Random: AbstractRNG

if !@isdefined(IterationLog)
    include(joinpath(@__DIR__, "..", "src", "logging.jl"))
end
if !@isdefined(IterativeMethod)
    include(joinpath(@__DIR__, "..", "src", "core.jl"))
end
if !@isdefined(StoppingCriterion)
    include(joinpath(@__DIR__, "..", "src", "stopping.jl"))
end
if !@isdefined(VariantGrid)
    include(joinpath(@__DIR__, "..", "src", "variants.jl"))
end
if !@isdefined(Problem)
    include(joinpath(@__DIR__, "..", "src", "problems.jl"))
end
if !@isdefined(ExperimentConfig)
    include(joinpath(@__DIR__, "..", "src", "experiment.jl"))
end

@kwdef struct TinyGD <: ConventionalMethod
    step_size::Float64 = 0.2
end

@kwdef struct TinyExp <: ExperimentalMethod
    step_size::Float64 = 0.2
end

@kwdef mutable struct TinyState
    iterate::IterateGroup
    metrics::MetricsGroup
    timing::TimingGroup
    _logger::Union{Nothing,Any} = nothing
end

function init_state(method::Union{TinyGD,TinyExp}, problem::Problem, rng::AbstractRNG)
    x = copy(problem.x0)
    gradient = zeros(problem.n)
    grad!(gradient, problem.f, x)
    TinyState(
        iterate = IterateGroup(x = x, gradient = gradient, x_prev = Float64[]),
        metrics = MetricsGroup(
            objective = objective(problem, x),
            gradient_norm = norm(gradient),
            step_norm = 0.0,
        ),
        timing = TimingGroup(),
    )
end

function step!(method::Union{TinyGD,TinyExp}, state::TinyState, problem::Problem, iter::Int)
    @core_timed state begin
        state.iterate.x_prev = copy(state.iterate.x)
        state.iterate.x .-= method.step_size .* state.iterate.x
        grad!(state.iterate.gradient, problem.f, state.iterate.x)
        state.metrics.objective = objective(problem, state.iterate.x)
        state.metrics.gradient_norm = norm(state.iterate.gradient)
        state.metrics.step_norm = norm(state.iterate.x_prev .- state.iterate.x)
    end
end

@testset "Layer 5 experiment orchestration" begin
    quad_spec = AnalyticProblem(
        name = :quadratic,
        params = (
            A = Matrix{Float64}(I, 2, 2),
            b = zeros(2),
            x0 = [1.0, -1.0],
        ),
    )

    cfg_resolve = ExperimentConfig(
        name = "resolve check",
        problem_spec = quad_spec,
        conventional_methods = [TinyGD()],
        variant_grids = [VariantGrid(
            base_name = "TinyExp",
            axes = [VariantAxis(:step_size, 0.1 => "s1", 0.2 => "s2")],
            builder = (; step_size) -> TinyExp(step_size = step_size),
        )],
        n_runs = 1,
    )

    conventional, experimental = resolve_methods(cfg_resolve)
    @test length(conventional) == 1
    @test length(experimental) == 2
    @test conventional[1][1] == "TinyGD"
    @test startswith(experimental[1][1], "TinyExp[")

    cfg_run = ExperimentConfig(
        name = "layer5 run",
        problem_spec = quad_spec,
        conventional_methods = [TinyGD()],
        experimental_methods = [TinyExp()],
        stopping_criteria = MaxIterations(n = 3),
        method_criteria = Dict("TinyGD" => MaxIterations(n = 1)),
        n_runs = 2,
        seed = 7,
    )

    mktempdir() do tmpdir
        p1 = next_experiment_path(tmpdir)
        mkpath(p1)
        p2 = next_experiment_path(tmpdir)
        @test basename(p1) == "001"
        @test basename(p2) == "002"

        result = run_experiment(cfg_run, tmpdir)
        @test result isa ExperimentResult
        @test result.config.name == "layer5 run"
        @test isdir(result.experiment_path)
        @test length(result.run_results) == 2

        for rr in result.run_results
            @test rr isa RunResult
            @test haskey(rr.method_results, "TinyGD")
            @test haskey(rr.method_results, "TinyExp")
            @test rr.method_results["TinyGD"].n_iters == 1
            @test rr.method_results["TinyExp"].n_iters == 3
        end
    end

    cfg_default = ExperimentConfig(
        name = "default method_criteria",
        problem_spec = quad_spec,
        conventional_methods = [TinyGD()],
        n_runs = 1,
    )
    @test isempty(cfg_default.method_criteria)
end
