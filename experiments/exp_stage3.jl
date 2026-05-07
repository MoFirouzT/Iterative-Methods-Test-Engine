# experiments/exp_stage3.jl
#
# Stage 3 — Persistence + roundtrip.
#
# Goal: prove that an ExperimentResult survives a save_experiment →
# load_experiment cycle byte-for-byte (in JLD2; the CSV sidecar is best-effort
# for vector-valued extras — see the note below). Plotting now goes
# *exclusively* through the persistence layer:
#     run → save → load → to_dataframe → plot
# never directly from the in-memory result.
#
# Validation workflow:
#   1)   julia --project=. experiments/exp_stage3.jl
#        → runs five methods, writes logs/YYYYMMDD/NNN/{result.jld2,
#          run1_*.csv, manifest.json}, then loads back from disk and plots
#          into the same directory. An assertion block confirms the
#          DataFrame round-trips identically.
#
#   2)   <quit Julia, restart fresh>
#        julia --project=. -e 'include("experiments/exp_stage3.jl"); \
#                              replot("logs/YYYYMMDD/NNN/")'
#        → loads from disk, regenerates the same two figures. They must be
#          visually identical to the ones from step 1; the underlying CSVs
#          must be line-for-line identical.
#
# CSV vector-extras decision (the staged plan flags this explicitly)
# ------------------------------------------------------------------
# extras[:x_iter] is Vector{Float64} per row. CSV doesn't represent that
# cleanly. Two reasonable choices in your CSV writer:
#   (a) skip vector-valued extras, note their omission in manifest.json;
#   (b) JSON-encode them as strings.
# JLD2 handles vectors natively either way, so the load-and-replot path
# below is unaffected. Pick (a) or (b) once and stick with it — retrofitting
# the writer after a few weeks of accumulated logs is unpleasant.
#
# To run, from project root:
#     julia --project=. experiments/exp_stage3.jl

include("../src/TestEngine.jl")
using .TestEngine
using Random
using Dates
using DataFrames
using CairoMakie

# ---------------------------------------------------------------------------
# Configuration — same problem, methods, seed, and stopping rule as Stages 1+2.
# ---------------------------------------------------------------------------

const SEED   = 42
const RUN_ID = 1

const PLOT_ORDER = ["Fixed", "Armijo", "Cauchy", "BB1", "BB2"]
const COLORS = Dict(
    "Fixed"  => "#000000",
    "Armijo" => "#0072B2",
    "Cauchy" => "#009E73",
    "BB1"    => "#E69F00",
    "BB2"    => "#D55E00",
)

function build_methods()
    [
        "Fixed"  => GradientDescent(direction = SteepestDescent(),
                                    step_size = FixedStep(α = 8e-4)),
        "Armijo" => GradientDescent(direction = SteepestDescent(),
                                    step_size = ArmijoLS()),
        "Cauchy" => GradientDescent(direction = SteepestDescent(),
                                    step_size = CauchyStep()),
        "BB1"    => GradientDescent(direction = SteepestDescent(),
                                    step_size = BarzilaiBorwein(variant = :BB1)),
        "BB2"    => GradientDescent(direction = SteepestDescent(),
                                    step_size = BarzilaiBorwein(variant = :BB2)),
    ]
end

# Compact ExperimentConfig used as the manifest record. We do *not* drive the
# orchestrator with this — Stage 6 introduces VariantGrid + run_experiment.
# Everything Stage 3 needs is below: a record of intent (problem, seed,
# stopping criteria) the loader can use to rebuild the problem.
function build_config()
    ExperimentConfig(
        name              = "stage3_rosenbrock_gd_stepsize_sweep",
        problem_spec      = AnalyticProblem(name = :rosenbrock,
                                            params = (rho = 100.0, dim = 2)),
        conventional_methods = [m for (_, m) in build_methods()],
        stopping_criteria = stop_when_any(
            MaxIterations(n   = 2000),
            GradientTolerance(tol = 1e-9),
        ),
        n_runs = 1,
        seed   = SEED,
        tags   = Dict("stage" => 3, "purpose" => "persistence roundtrip"),
    )
end

# ---------------------------------------------------------------------------
# Run + save (imperative path; matches Stages 1 and 2)
# ---------------------------------------------------------------------------

function run_and_save(; log_root::String = "logs")
    config = build_config()

    # Allocate the experiment directory atomically. next_experiment_path uses
    # mkdir, which throws on EEXIST — safe under concurrent writes.
    exp_path = next_experiment_path(log_root)
    @info "Allocated experiment directory" path = exp_path

    # Reproduce the same RNG derivation the orchestrator will use at Stage 6,
    # so iter_logs are byte-comparable across stages for any randomness-free
    # method.
    rng_data = Xoshiro(hash((SEED, RUN_ID, :data)))
    problem  = make_problem(config.problem_spec, rng_data)

    @info "Running" problem = "Rosenbrock(ρ=100)" x0 = problem.x0 x_opt = problem.x_opt

    method_results = Dict{String, Any}()    # MethodResult{S} for various S → Any
    for (name, method) in build_methods()
        method_rng = Xoshiro(hash((SEED, RUN_ID, name)))
        logger     = make_logger(name, RUN_ID, exp_path,
                                 VerbosityConfig(level = MILESTONE))
        result     = run_method(method, problem,
                                config.stopping_criteria, logger, method_rng)
        method_results[name] = result

        last_entry = result.iter_logs[end]
        @info "[$name] done" iters       = result.n_iters
                              stop_reason = result.stop_reason
                              f_final     = last_entry.objective
                              dist_to_opt = last_entry.dist_to_opt
    end

    run_result = RunResult(RUN_ID, method_results)
    exp_result = ExperimentResult(
        config,
        exp_path,
        now(),
        gethostname(),
        [run_result],
    )

    save_experiment(exp_result)
    @info "Saved experiment" path = exp_path
    return exp_result, exp_path
end

# ---------------------------------------------------------------------------
# Plotting — both panels now consume a DataFrame, not raw results.
# Identical to Stages 1+2 in everything but the data source.
# ---------------------------------------------------------------------------

function plot_convergence_panel(df::DataFrame;
                                outpath::String,
                                title_suffix::String = "")
    fig = Figure(size = (1400, 850))

    panels = (
        (1, 1, :objective,     "f(xₖ)",          "f(x)"),
        (1, 2, :gradient_norm, "‖∇f(xₖ)‖",       "‖∇f(x)‖"),
        (2, 1, :dist_to_opt,   "‖xₖ − x*‖",      "‖x − x*‖"),
        (2, 2, :step_size,     "αₖ (step size)", "α"),
    )

    for (row, col, ycol, title, ylabel) in panels
        ax = Axis(fig[row, col],
            xlabel = "iteration",
            ylabel = ylabel,
            yscale = log10,
            title  = title,
        )
        for name in PLOT_ORDER
            sub = filter(:method_name => ==(name), df)
            isempty(sub) && continue
            # iter=0 has α=0 by default — invalid on log scale.
            ycol === :step_size && (sub = filter(:iter => >(0), sub))
            ys = max.(sub[!, ycol], 1e-16)
            lines!(ax, sub.iter, ys; color = COLORS[name], linewidth = 2.0)
        end
    end

    legend_elems = [LineElement(color = COLORS[name], linewidth = 2.5)
                    for name in PLOT_ORDER]
    Legend(fig[1:2, 3], legend_elems, PLOT_ORDER, "step size";
           framevisible = true, tellwidth = true)

    Label(fig[0, :],
          "Stage 3 — GradientDescent on Rosenbrock(ρ=100)" * title_suffix,
          fontsize = 16, font = :bold)

    save(outpath, fig)
    @info "Saved figure" path = outpath
    return fig
end

function plot_trajectories(df::DataFrame, problem;
                           outpath::String,
                           title_suffix::String = "")
    fig = Figure(size = (1000, 900))
    ax  = Axis(fig[1, 1],
        xlabel = "x₁",
        ylabel = "x₂",
        title  = "Stage 3 — Rosenbrock(ρ=100): trajectories" * title_suffix,
        aspect = DataAspect(),
    )

    # Contour grid (closed-form Rosenbrock; ρ from problem.meta with fallback).
    ρ = Float64(get(problem.meta, :rho, 100.0))
    rosen(x, y) = (1.0 - x)^2 + ρ * (y - x^2)^2

    xs = range(-2.0, 2.0, length = 400)
    ys = range(-1.0, 3.0, length = 400)
    zs = [rosen(x, y) for x in xs, y in ys]
    levels = 10.0 .^ range(-1.0, 3.5, length = 15)
    contour!(ax, xs, ys, zs;
        levels    = levels,
        color     = (:gray, 0.55),
        linewidth = 0.6,
    )

    # Trajectories from the DataFrame's :x_iter column.
    for name in PLOT_ORDER
        sub = filter(:method_name => ==(name), df)
        sub = filter(:x_iter => v -> !ismissing(v) && length(v) == 2, sub)
        isempty(sub) && continue
        traj_x = [v[1] for v in sub.x_iter]
        traj_y = [v[2] for v in sub.x_iter]
        lines!(ax, traj_x, traj_y;
            color     = COLORS[name],
            linewidth = 1.8,
            label     = name,
        )
    end

    scatter!(ax, [problem.x0[1]], [problem.x0[2]];
        color = :black, marker = :circle, markersize = 14,
        strokecolor = :white, strokewidth = 1.5, label = "x₀")
    if !isnothing(problem.x_opt)
        scatter!(ax, [problem.x_opt[1]], [problem.x_opt[2]];
            color = :red, marker = :star5, markersize = 20,
            strokecolor = :white, strokewidth = 1.5, label = "x*")
    end

    xlims!(ax, -2.0, 2.0)
    ylims!(ax, -1.0, 3.0)
    axislegend(ax;
        position = :rt, framevisible = true,
        backgroundcolor = (:white, 0.85), nbanks = 1)

    save(outpath, fig)
    @info "Saved figure" path = outpath
    return fig
end

# ---------------------------------------------------------------------------
# Roundtrip integrity check — the actual Stage 3 validator.
# Compares the DataFrame derived from the in-memory result against the one
# derived from the just-loaded copy. Anything divergent here is a bug in
# save_experiment / load_experiment / to_dataframe.
# ---------------------------------------------------------------------------

function assert_roundtrip(df_mem::DataFrame, df_disk::DataFrame)
    @assert nrow(df_mem) == nrow(df_disk) (
        "Row count mismatch: mem=$(nrow(df_mem)) disk=$(nrow(df_disk))")
    @assert sort(unique(df_mem.method_name)) == sort(unique(df_disk.method_name)) (
        "Method names differ on roundtrip")

    for name in PLOT_ORDER
        sub_mem  = filter(:method_name => ==(name), df_mem)
        sub_disk = filter(:method_name => ==(name), df_disk)
        @assert sub_mem.iter      == sub_disk.iter      "iter mismatch for $name"
        @assert sub_mem.objective == sub_disk.objective "objective mismatch for $name"
        @assert sub_mem.gradient_norm == sub_disk.gradient_norm "‖∇f‖ mismatch for $name"
        @assert sub_mem.dist_to_opt   == sub_disk.dist_to_opt   "dist_to_opt mismatch for $name"

        # Vector-valued extras are the most likely place for a serializer to go
        # wrong. Compare element-wise, tolerating Missing on either side.
        if hasproperty(sub_mem, :x_iter) && hasproperty(sub_disk, :x_iter)
            for (vm, vd) in zip(sub_mem.x_iter, sub_disk.x_iter)
                if ismissing(vm) || ismissing(vd)
                    @assert ismissing(vm) && ismissing(vd) "x_iter missingness differs for $name"
                else
                    @assert vm == vd "x_iter values differ for $name"
                end
            end
        end
    end
    @info "Roundtrip integrity ✓" rows = nrow(df_disk)
end

# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

function main()
    exp_result, exp_path = run_and_save()

    # Roundtrip in the same process: load what we just wrote and verify
    # the DataFrames are identical.
    exp_disk = load_experiment(exp_path)
    df_mem   = to_dataframe(exp_result)
    df_disk  = to_dataframe(exp_disk)
    assert_roundtrip(df_mem, df_disk)

    # Plot from the *loaded* copy, not the in-memory one. This is what makes
    # the Stage 3 deliverable "all plotting goes through disk".
    rng_data = Xoshiro(hash((exp_disk.config.seed, RUN_ID, :data)))
    problem  = make_problem(exp_disk.config.problem_spec, rng_data)

    plot_convergence_panel(df_disk;
        outpath      = joinpath(exp_path, "convergence.pdf"),
        title_suffix = " — replotted from disk")
    plot_trajectories(df_disk, problem;
        outpath      = joinpath(exp_path, "trajectories.pdf"),
        title_suffix = " — replotted from disk")

    println()
    println("Experiment saved to: ", exp_path)
    println("Cold-restart validation:")
    println("  1) quit Julia")
    println("  2) julia --project=. -e 'include(\"experiments/exp_stage3.jl\"); replot(\"$exp_path\")'")
    println("     → produces convergence.pdf / trajectories.pdf identical to the ones above.")
    println()
    return exp_result
end

# Cold-restart entry point — load + plot only; no run.
function replot(path::String)
    exp_result = load_experiment(path)
    df         = to_dataframe(exp_result)
    @info "Loaded" path                = path
                   n_methods           = length(unique(df.method_name))
                   n_rows              = nrow(df)
                   has_x_iter_column   = hasproperty(df, :x_iter)

    rng_data = Xoshiro(hash((exp_result.config.seed, RUN_ID, :data)))
    problem  = make_problem(exp_result.config.problem_spec, rng_data)

    plot_convergence_panel(df;
        outpath      = joinpath(path, "convergence.pdf"),
        title_suffix = " — cold-restart replot")
    plot_trajectories(df, problem;
        outpath      = joinpath(path, "trajectories.pdf"),
        title_suffix = " — cold-restart replot")
    return df
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

# ---------------------------------------------------------------------------
# Validation criteria
# ---------------------------------------------------------------------------
# After running main() once, then restarting Julia and calling replot(path):
#   • assert_roundtrip passes inside main() — DataFrames identical mem ↔ disk.
#   • The two replot()s produce visually identical figures across the cold
#     restart. Diff the per-method CSVs for line-for-line equality if you want
#     a hard test (PDF byte-equality is fragile because PDFs embed a
#     timestamp; CSVs do not).
#   • If assert_roundtrip fails on :x_iter specifically, the JLD2 writer is
#     dropping or corrupting vector-valued extras — likely the most common
#     persistence bug at this stage.
