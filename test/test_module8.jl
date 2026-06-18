using Test
using LinearAlgebra: I, norm
using Random: AbstractRNG
using Dates: now

include(joinpath(@__DIR__, "..", "experiments", "_bootstrap.jl"))
import .TestEngine: init_state, step!   # engine dispatch points these tests extend

@kwdef struct PersistMethod <: IterativeMethod
    step_size::StepSize = FixedStep(α = 0.3)
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

function step!(method::PersistMethod, state::PersistState, problem::Problem, iter::Int, logger::Logger, rng::AbstractRNG)
    @core_timed state begin
        state.iterate.x_prev = copy(state.iterate.x)
        α = compute_step_size(method.step_size, state, problem, state.iterate.x)
        state.iterate.x .-= α .* state.iterate.x
        grad!(state.iterate.gradient, problem.f, state.iterate.x)
        state.metrics.objective = objective(problem, state.iterate.x)
        state.metrics.gradient_norm = norm(state.iterate.gradient)
        state.metrics.step_norm = norm(state.iterate.x_prev .- state.iterate.x)
    end
end

@testset "Module 8 persistence" begin
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
        baseline_methods = [PersistMethod()],
        stopping_criteria = MaxIterations(n = 2),
        n_runs = 2,
        seed = 123,
        tags = Dict("suite" => "module8"),
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


# Hand-build an ExperimentResult with rich extras so we exercise the columnar
# (struct-of-arrays) JLD2 shim and the save-time PersistPolicy directly.
function _rich_result(path)
    spec = AnalyticProblem(
        name = :quadratic,
        params = (A = Matrix{Float64}(I, 2, 2), b = zeros(2), x0 = [1.0, -1.0]),
    )
    cfg = ExperimentConfig(name = "rich", problem_spec = spec,
                           baseline_methods = [PersistMethod()])
    logs = IterationLog[]
    for i in 0:5
        extras = Dict{Symbol,Any}(
            :x_iter => Float64[i, -i],   # dense vector extra
            :n_grad => i,                # dense scalar extra
        )
        i == 5 && (extras[:sub_logs] = [IterationLog(iter = 1, core_time_ns = 7,
            objective = 0.1, gradient_norm = 0.2, step_norm = 0.3)])  # sparse
        i == 2 && (extras[:maybe] = missing)  # genuine missing value (not "absent")
        push!(logs, IterationLog(iter = i, core_time_ns = i * 10, objective = 1.0 / (i + 1),
            gradient_norm = 2.0 / (i + 1), step_norm = 0.01 * i, dist_to_opt = 3.0 / (i + 1),
            extras = extras))
    end
    mres = MethodResult("PersistMethod", logs, nothing, :converged, 5)
    ExperimentResult(cfg, path, now(), "testhost", [RunResult(1, Dict("PersistMethod" => mres))])
end

@testset "Module 8 columnar round-trip + selective saving" begin
    mktempdir() do tmpdir
        # ── full round-trip through the columnar shim ──────────────────────
        full_path = joinpath(tmpdir, "full")
        save_experiment(_rich_result(full_path))
        loaded = load_experiment(full_path).run_results[1].method_results["PersistMethod"]
        logs = loaded.iter_logs

        @test [e.iter for e in logs] == collect(0:5)
        @test [e.objective for e in logs] ≈ [1.0 / (i + 1) for i in 0:5]
        @test logs[1].extras[:x_iter] == Float64[0, 0]
        @test logs[6].extras[:x_iter] == Float64[5, -5]
        @test all(haskey(e.extras, :n_grad) for e in logs)          # dense key everywhere
        @test logs[6].extras[:sub_logs][1].objective == 0.1         # sparse key only on last
        @test !haskey(logs[1].extras, :sub_logs)
        # absent vs genuine `missing`: only iter 2 carries :maybe, and as missing.
        @test haskey(logs[3].extras, :maybe) && ismissing(logs[3].extras[:maybe])
        @test !any(haskey(e.extras, :maybe) for e in logs[[1, 2, 4, 5, 6]])

        # ── drop policy: :x_iter gone from JLD2, scalars intact ────────────
        drop_path = joinpath(tmpdir, "drop")
        save_experiment(_rich_result(drop_path); persist = PersistPolicy(drop = [:x_iter]))
        dlogs = load_experiment(drop_path).run_results[1].method_results["PersistMethod"].iter_logs
        @test !any(haskey(e.extras, :x_iter) for e in dlogs)
        @test all(haskey(e.extras, :n_grad) for e in dlogs)
        @test [e.objective for e in dlogs] ≈ [1.0 / (i + 1) for i in 0:5]  # metrics untouched
        dmani = load_manifest(joinpath(drop_path, "manifest.json"))
        @test "x_iter" in String.(dmani.persist_dropped_extras)
        # dropping is binary-only: the JLD2 shrinks vs the full save.
        @test filesize(joinpath(drop_path, "result.jld2")) <
              filesize(joinpath(full_path, "result.jld2"))

        # ── decimate policy: keep :x_iter on iter 0 and every 3rd iter ─────
        dec_path = joinpath(tmpdir, "dec")
        save_experiment(_rich_result(dec_path); persist = PersistPolicy(decimate = Dict(:x_iter => 3)))
        declogs = load_experiment(dec_path).run_results[1].method_results["PersistMethod"].iter_logs
        kept = [e.iter for e in declogs if haskey(e.extras, :x_iter)]
        @test kept == [0, 3]   # iter 0 always kept; then every 3rd
        @test all(haskey(e.extras, :n_grad) for e in declogs)   # untouched key stays
    end
end
