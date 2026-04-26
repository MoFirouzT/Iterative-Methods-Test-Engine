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
if !@isdefined(save_experiment)
    include(joinpath(@__DIR__, "..", "src", "persistence.jl"))
end

@kwdef struct PersistMethod <: ConventionalMethod
    step_size::Float64 = 0.3
end

@kwdef mutable struct PersistState
    iterate::IterateGroup
    metrics::MetricsGroup
    timing::TimingGroup
    _logger::Union{Nothing,Any} = nothing
end

function init_state(method::PersistMethod, problem::Problem, rng::AbstractRNG)
    x = copy(problem.x0)
    g = zeros(problem.n)
    grad!(g, problem.f, x)
    PersistState(
        iterate = IterateGroup(x = x, gradient = g),
        metrics = MetricsGroup(
            objective = objective(problem, x),
            gradient_norm = norm(g),
            step_norm = 0.0,
        ),
        timing = TimingGroup(),
    )
end

function step!(method::PersistMethod, state::PersistState, problem::Problem, iter::Int)
    @core_timed state begin
        state.iterate.x_prev = copy(state.iterate.x)
        state.iterate.x .-= method.step_size .* state.iterate.x
        grad!(state.iterate.gradient, problem.f, state.iterate.x)
        state.metrics.objective = objective(problem, state.iterate.x)
        state.metrics.gradient_norm = norm(state.iterate.gradient)
        state.metrics.step_norm = norm(state.iterate.x_prev .- state.iterate.x)
    end
end

@testset "Layer 8 persistence" begin
    spec = AnalyticProblem(
        name = :quadratic,
        params = (
            A = Matrix{Float64}(I, 2, 2),
            b = zeros(2),
            x0 = [1.0, -1.0],
        ),
    )

    cfg = ExperimentConfig(
        name = "persistence-check",
        problem_spec = spec,
        conventional_methods = [PersistMethod()],
        stopping_criteria = MaxIterations(n = 2),
        n_runs = 2,
        seed = 123,
        tags = Dict("suite" => "layer8"),
    )

    mktempdir() do tmpdir
        result = run_experiment(cfg, tmpdir; verbosity = VerbosityConfig(level = SILENT))

        @test result isa ExperimentResult
        @test isdir(result.experiment_path)

        manifest_path = joinpath(result.experiment_path, "manifest.json")
        jld_path = joinpath(result.experiment_path, "result.jld2")
        csv_path = joinpath(result.experiment_path, "run1_PersistMethod.csv")

        @test isfile(manifest_path)
        @test isfile(jld_path)
        @test isfile(csv_path)

        manifest = load_manifest(manifest_path)
        @test String(manifest.name) == "persistence-check"
        @test Int(manifest.n_runs) == 2
        @test Int(manifest.n_methods) == 1

        loaded = load_experiment(result.experiment_path)
        @test loaded isa ExperimentResult
        @test loaded.config.name == result.config.name
        @test length(loaded.run_results) == 2
        @test loaded.run_results[1].method_results["PersistMethod"].n_iters == 2

        listed = list_experiments(tmpdir)
        @test length(listed) >= 1
        @test listed[end].path == result.experiment_path
        @test listed[end].name == "persistence-check"
        @test listed[end].n_runs == 2
    end
end
