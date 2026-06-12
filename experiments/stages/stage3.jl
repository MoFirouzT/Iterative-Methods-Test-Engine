# experiments/stages/stage3.jl
#
# Stage 3 — Persistence + roundtrip.
#
# Goal: prove that an ExperimentResult survives a save_experiment →
# load_experiment cycle byte-for-byte (in JLD2; the CSV sidecar is best-effort
# for vector-valued extras — see the note below).
# Plotting now goes *exclusively* through the persistence layer:
#     run → save → load → to_dataframe → plot
# never directly from the in-memory result.
#
# Validation workflow:
#   1)   julia --project=. experiments/stages/stage3.jl
#        → runs five methods, writes logs/YYYYMMDD/NNN/{result.jld2,
#          run1_*.csv, manifest.json}, then loads back from disk and plots
#          into the same directory. An assertion block confirms the
#          DataFrame round-trips identically.
#
#   2)   <quit Julia, restart fresh>
#        julia --project=. -e 'include("experiments/stages/stage3.jl"); \
#                              replot("logs/YYYYMMDD/NNN/")'
#        → loads from disk, regenerates the same two figures. They must be
#          visually identical to the ones from step 1; the underlying CSVs
#          must be line-for-line identical.
#
# CSV vector-extras decision (the staged plan flags this explicitly)
# ------------------------------------------------------------------
# extras[:x_iter] is Vector{Float64} per row.
# CSV doesn't represent that cleanly.
# Two reasonable choices in your CSV writer:
#   (a) skip vector-valued extras, note their omission in manifest.json;
#   (b) JSON-encode them as strings.
# JLD2 handles vectors natively either way, so the load-and-replot path
# below is unaffected. Pick (a) or (b) once and stick with it — retrofitting
# the writer after a few weeks of accumulated logs is unpleasant.
#
# To run, from project root:
#     julia --project=. experiments/stages/stage3.jl

include("../_bootstrap.jl")   # engine + all content (problems, methods, components)
using Random
using Dates
using DataFrames
using CairoMakie

# Canonical trajectory-plot recipe, shared with Stage 2.
include("../_shared.jl")

# ---------------------------------------------------------------------------
# Configuration — same problem, methods, seed, and stopping rule as Stages 1+2.
# ---------------------------------------------------------------------------

const SEED   = 42
const RUN_ID = 1

# PLOT_ORDER, COLORS, build_standard_methods live in _shared.jl.
const build_methods = build_standard_methods

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
        @info("[$name] done",
              iters       = result.n_iters,
              stop_reason = result.stop_reason,
              f_final     = last_entry.objective,
              dist_to_opt = last_entry.dist_to_opt)
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

    # Document the CSV vector-extras policy at runtime — so any Stage 3 log
    # directory is self-describing without anyone having to inspect a CSV.
    # The decision (omit vector extras / JSON-encode / etc.) is made inside
    # save_experiment; we just observe and record what actually ended up on
    # disk.
    csv_files = filter(f -> startswith(f, "run") && endswith(f, ".csv"),
                       readdir(exp_path))
    if !isempty(csv_files)
        header = split(readline(joinpath(exp_path, first(csv_files))), ',')
        @info("CSV vector-extras policy",
              sample_csv     = first(csv_files),
              has_x_iter_col = "x_iter" in header,
              csv_columns    = header)
    end

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
            if ycol === :step_size
                # αₖ is per-iter discrete (Armijo: βʲ rungs; Fixed: a constant
                # line; BB: a continuous sequence). Drawing markers + thin line
                # is more honest than plain lines, which smear the discrete
                # rungs into diagonals on log-y.
                scatterlines!(ax, sub.iter, ys;
                    color      = COLORS[name],
                    linewidth  = 0.8,
                    markersize = 3,
                )
            else
                lines!(ax, sub.iter, ys; color = COLORS[name], linewidth = 2.0)
            end
        end
    end

    legend_elems = [LineElement(color = COLORS[name], linewidth = 2.5)
                    for name in PLOT_ORDER]
    Legend(fig[1:2, 3], legend_elems, PLOT_ORDER, "method";
           framevisible = true, tellwidth = true)

    Label(fig[0, :],
          "Stage 3 — GradientDescent on Rosenbrock(ρ=100)" * title_suffix,
          fontsize = 16, font = :bold)

    save(outpath, fig)
    @info "Saved figure" path = outpath
    return fig
end

# Trajectory plot is rendered by the shared recipe in experiments/_shared.jl
# (`plot_trajectories(df, problem; outpath, plot_order, colors, title, ...)`),
# the same one Stage 2 uses. Keeping the recipe in one place stops the two
# stages from drifting and means improvements ship to both at once.

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

        # Base columns — every method should have these.
        @assert sub_mem.iter          == sub_disk.iter          "iter mismatch for $name"
        @assert sub_mem.objective     == sub_disk.objective     "objective mismatch for $name"
        @assert sub_mem.gradient_norm == sub_disk.gradient_norm "‖∇f‖ mismatch for $name"
        @assert sub_mem.step_norm     == sub_disk.step_norm     "step_norm mismatch for $name"
        @assert sub_mem.dist_to_opt   == sub_disk.dist_to_opt   "dist_to_opt mismatch for $name"
        @assert sub_mem.core_time_ns  == sub_disk.core_time_ns  "core_time_ns mismatch for $name"

        # step_size from extras — may carry NaN at iter=0; use isequal so NaN
        # compares equal to NaN. (== on NaN is false, which would falsely
        # fail every roundtrip.)
        if hasproperty(sub_mem, :step_size) && hasproperty(sub_disk, :step_size)
            @assert all(isequal.(sub_mem.step_size, sub_disk.step_size)) (
                "step_size mismatch for $name")
        end

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
    @info("Roundtrip integrity ✓",
          rows    = nrow(df_disk),
          columns = collect(propertynames(df_disk)))
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
        outpath    = joinpath(exp_path, "trajectories.pdf"),
        plot_order = PLOT_ORDER,
        colors     = COLORS,
        title      = "Stage 3 — Rosenbrock(ρ=100): trajectories — replotted from disk",
    )

    println()
    println("Experiment saved to: ", exp_path)
    println("Cold-restart validation:")
    println("  1) quit Julia")
    println("  2) julia --project=. -e 'include(\"experiments/stages/stage3.jl\"); replot(\"$exp_path\")'")
    println("     → produces convergence.pdf / trajectories.pdf identical to the ones above.")
    println()
    return exp_result
end

# Cold-restart entry point — load + plot only; no run.
function replot(path::String)
    exp_result = load_experiment(path)
    df         = to_dataframe(exp_result)
    @info("Loaded",
          path              = path,
          n_methods         = length(unique(df.method_name)),
          n_rows            = nrow(df),
          has_x_iter_column = hasproperty(df, :x_iter))

    # Cold-restart byte-equality: re-save the loaded experiment to a tmp
    # directory and diff each per-method CSV against the original. This
    # actually tests the load → save round trip across a fresh process, not
    # just "load → plot doesn't crash" (which is what previous replot did).
    # Only CSVs are diffed; result.jld2 + manifest.json embed timestamps and
    # would always differ.
    tmp = mktempdir(; cleanup = true)
    exp_replay = ExperimentResult(
        exp_result.config,
        tmp,
        now(),
        gethostname(),
        exp_result.run_results,
    )
    save_experiment(exp_replay)
    csv_files = filter(f -> startswith(f, "run") && endswith(f, ".csv"),
                       readdir(path))
    for fname in csv_files
        @assert read(joinpath(path, fname)) == read(joinpath(tmp, fname)) (
            "CSV byte-mismatch across cold restart: $fname")
    end
    @info("Cold-restart CSV byte equality ✓",
          n_csvs = length(csv_files),
          tmp    = tmp)

    rng_data = Xoshiro(hash((exp_result.config.seed, RUN_ID, :data)))
    problem  = make_problem(exp_result.config.problem_spec, rng_data)

    plot_convergence_panel(df;
        outpath      = joinpath(path, "convergence.pdf"),
        title_suffix = " — cold-restart replot")
    plot_trajectories(df, problem;
        outpath    = joinpath(path, "trajectories.pdf"),
        plot_order = PLOT_ORDER,
        colors     = COLORS,
        title      = "Stage 3 — Rosenbrock(ρ=100): trajectories — cold-restart replot",
    )
    return df
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

# ---------------------------------------------------------------------------
# Validation criteria
# ---------------------------------------------------------------------------
# After running main() once, then restarting Julia and calling replot(path):
#   • assert_roundtrip passes inside main() — DataFrames identical mem ↔ disk,
#     across :iter, :objective, :gradient_norm, :step_norm, :dist_to_opt,
#     :core_time_ns, :step_size, and the vector-valued :x_iter.
#   • replot() automates the cold-restart CSV byte-equality test — it
#     re-saves the loaded experiment to a tmp dir and diffs each per-method
#     CSV against the original. (JLD2 + manifest.json embed timestamps and
#     would always differ; that's intentional.)
#   • If assert_roundtrip fails on :x_iter specifically, the JLD2 writer is
#     dropping or corrupting vector-valued extras — likely the most common
#     persistence bug at this stage.
