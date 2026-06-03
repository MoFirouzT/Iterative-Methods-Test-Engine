# experiments/exp_stage4.jl
#
# Stage 4 — Stopping-criteria coverage.
#
# Goal: exercise the full StoppingCriteria machinery — DistanceToOptimal,
# CompositeCriteria(:any), the runner-side dist_to_opt update, and stop_reason
# propagation through MethodResult — by running each method to a tight
# multi-criterion stop and visualizing convergence speed.
#
# Stopping rule:  stop_when_any(MaxIterations(20000),
#                                DistanceToOptimal(1e-8),
#                                GradientTolerance(1e-10))
#
# Two distinct quantities are reported:
#   • n_iters / stop_reason   — from MethodResult, i.e. when the run stopped
#                               and which criterion fired.
#   • iters_to_milestone      — from findfirst(d ≤ 1e-6, dist_to_opt_trace),
#                               i.e. how long it took to first reach the
#                               1e-6 milestone (independent of the stop tol).
# These are *different numbers*. A method that hits :gradient_converged at
# iter 5000 may have crossed 1e-6 long before; a method that hits
# :max_iterations may never cross 1e-6 at all (= DNF on the bar chart).
#
# Expected ordering (revised from the original staged-plan numbers, which
# had Cauchy and BB transposed):
#   BB1, BB2          — fastest; should hit :optimal_reached, low thousands
#   Armijo            — middle; usually :optimal_reached or :gradient_converged
#   Cauchy            — slow on Rosenbrock; may hit :max_iterations
#   Fixed (α=8e-4)    — DNF; :max_iterations
#
# To run, from project root:
#     julia --project=. experiments/exp_stage4.jl

include("../src/TestEngine.jl")
using .TestEngine
using Random
using Dates
using DataFrames
using Printf
using CairoMakie

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

const SEED   = 42
const RUN_ID = 1

# Stopping budgets and bar-chart milestone. Note: DIST_MILESTONE (1e-6) is the
# bar-chart threshold and is *looser* than DIST_TOL (1e-8, the actual stop
# tolerance). The bar measures "first time we crossed 1e-6", not "where we
# stopped". Named explicitly to avoid shadowing TestEngine's `MILESTONE`
# verbosity-level export.
const MAX_ITERS       = 20_000
const DIST_TOL        = 1e-8
const GRAD_TOL        = 1e-10
const DIST_MILESTONE  = 1e-6

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

function build_config()
    ExperimentConfig(
        name = "stage4_rosenbrock_stopping_coverage",
        problem_spec = AnalyticProblem(name = :rosenbrock,
                                       params = (rho = 100.0, dim = 2)),
        conventional_methods = [m for (_, m) in build_methods()],
        stopping_criteria = stop_when_any(
            MaxIterations(n   = MAX_ITERS),
            DistanceToOptimal(tol = DIST_TOL),
            GradientTolerance(tol = GRAD_TOL),
        ),
        n_runs = 1,
        seed   = SEED,
        tags   = Dict("stage" => 4, "purpose" => "stopping criteria coverage"),
    )
end

# ---------------------------------------------------------------------------
# Run + save (Stage 3 pattern: imperative orchestration, manual
# ExperimentResult construction, all plotting goes through disk afterward)
# ---------------------------------------------------------------------------

function run_and_save(; log_root::String = "logs")
    config   = build_config()
    exp_path = next_experiment_path(log_root)
    @info "Allocated experiment directory" path = exp_path

    rng_data = Xoshiro(hash((SEED, RUN_ID, :data)))
    problem  = make_problem(config.problem_spec, rng_data)
    @info "Running" problem = "Rosenbrock(ρ=100)" x0 = problem.x0 x_opt = problem.x_opt

    # ── Warm-up: trigger JIT so wall time on the timed runs reflects steady
    #    state, not one-shot compilation. Without this the core/wall check
    #    below is dominated by compile cost on the first method.
    let warmup_method = build_methods()[1].second
        run_method(warmup_method, problem, MaxIterations(n = 1),
                   make_logger("warmup", 0, "", VerbosityConfig(level = SILENT)),
                   Xoshiro(0))
    end

    method_results = Dict{String, Any}()
    wall_ns        = Dict{String, Int64}()
    for (name, method) in build_methods()
        method_rng = Xoshiro(hash((SEED, RUN_ID, name)))
        # MILESTONE here is the verbosity level imported from TestEngine.
        logger     = make_logger(name, RUN_ID, exp_path,
                                 VerbosityConfig(level = MILESTONE))
        wall_t0    = time_ns()
        result     = run_method(method, problem,
                                config.stopping_criteria, logger, method_rng)
        wall_ns[name]        = Int64(time_ns() - wall_t0)
        method_results[name] = result

        last_entry = result.iter_logs[end]
        @info "[$name] done" iters       = result.n_iters
                              stop_reason = result.stop_reason
                              dist_final  = last_entry.dist_to_opt
                              grad_final  = last_entry.gradient_norm
    end

    run_result = RunResult(RUN_ID, method_results)
    exp_result = ExperimentResult(
        config, exp_path, now(), gethostname(), [run_result],
    )
    save_experiment(exp_result)
    @info "Saved experiment" path = exp_path
    return exp_result, exp_path, wall_ns
end

# ---------------------------------------------------------------------------
# Per-method summary (from MethodResult — n_iters, stop_reason, finals)
# ---------------------------------------------------------------------------

function summarize_methods(exp_result::ExperimentResult)
    out = Dict{String, NamedTuple}()
    for run_result in exp_result.run_results
        for (name, mres) in run_result.method_results
            last_entry = mres.iter_logs[end]
            out[name] = (
                n_iters     = mres.n_iters,
                stop_reason = mres.stop_reason,
                f_final     = last_entry.objective,
                grad_final  = last_entry.gradient_norm,
                dist_final  = last_entry.dist_to_opt,
            )
        end
    end
    return out
end

# ---------------------------------------------------------------------------
# Per-method milestone iters (from DataFrame iter-by-iter trace)
# ---------------------------------------------------------------------------

"""
    compute_milestone_iters(df; milestone) -> Dict{String, Union{Int, Nothing}}

For each method, returns the iter at which `dist_to_opt` first drops below
`milestone`, or `nothing` if it never does within the recorded trace.
Robust to Inf entries (when x_opt is not tracked) and to missing values.
"""
function compute_milestone_iters(df::DataFrame; milestone::Float64)
    out = Dict{String, Union{Int, Nothing}}()
    for name in unique(df.method_name)
        sub = sort(filter(:method_name => ==(name), df), :iter)
        idx = findfirst(d -> !ismissing(d) && isfinite(d) && d <= milestone,
                        sub.dist_to_opt)
        out[name] = isnothing(idx) ? nothing : Int(sub.iter[idx])
    end
    return out
end

# ---------------------------------------------------------------------------
# Printed summary table — exercises the "Log n_iters and stop_reason per
# method" requirement of the staged plan.
# ---------------------------------------------------------------------------

function print_summary_table(summary::Dict, milestone_iters::Dict;
                              milestone::Float64)
    println()
    println("Stage 4 — termination summary")
    println("─" ^ 100)
    @printf("%-8s | %8s | %-22s | %12s | %12s | %12s | %s\n",
            "method", "n_iters", "stop_reason",
            "f_final", "‖∇f‖_final", "dist_final",
            "iters→$(milestone)")
    println("─" ^ 100)
    for name in PLOT_ORDER
        s = summary[name]
        m = milestone_iters[name]
        m_str = isnothing(m) ? "DNF" : string(m)
        @printf("%-8s | %8d | %-22s | %12.4e | %12.4e | %12.4e | %s\n",
                name, s.n_iters, ":" * string(s.stop_reason),
                s.f_final, s.grad_final, s.dist_final, m_str)
    end
    println("─" ^ 100)
    println()
end

# ---------------------------------------------------------------------------
# Timing report — sums per-iter core_time_ns vs. captured wall-time and checks
# the doc's 50–110% band (basic_experiments.md, Stage 0 rationale). Stage 4
# is the right place for this assertion: 20_000 iters / method amortizes the
# per-iter scaffolding (extract_log_entry, should_stop, dispatch) and JIT is
# already warm by the time we enter the timed loop.
# ---------------------------------------------------------------------------

const CORE_WALL_LO = 0.50
const CORE_WALL_HI = 1.10

function print_timing_table(exp_result::ExperimentResult, wall_ns::Dict{String, Int64})
    println()
    println("Stage 4 — core / wall timing")
    println("─" ^ 78)
    @printf("%-8s | %10s | %10s | %10s | %8s | %s\n",
            "method", "iters", "core (ms)", "wall (ms)", "core/wall", "verdict")
    println("─" ^ 78)

    total_core = 0
    total_wall = 0
    for name in PLOT_ORDER
        result   = exp_result.run_results[1].method_results[name]
        core_ns  = sum(e.core_time_ns for e in result.iter_logs)
        wall_n   = wall_ns[name]
        ratio    = core_ns / wall_n
        verdict  = (CORE_WALL_LO ≤ ratio ≤ CORE_WALL_HI) ? "✓" : "✗"

        total_core += core_ns
        total_wall += wall_n

        @printf("%-8s | %10d | %10.3f | %10.3f | %7.2f%% | %s\n",
                name, result.n_iters, core_ns / 1e6, wall_n / 1e6,
                100 * ratio, verdict)
    end
    println("─" ^ 78)
    total_ratio = total_core / total_wall
    total_ok    = CORE_WALL_LO ≤ total_ratio ≤ CORE_WALL_HI
    @printf("%-8s | %10s | %10.3f | %10.3f | %7.2f%% | %s\n",
            "TOTAL", "", total_core / 1e6, total_wall / 1e6,
            100 * total_ratio, total_ok ? "✓" : "✗")
    println("─" ^ 78)

    if !total_ok
        println()
        if total_ratio < CORE_WALL_LO
            println("  Aggregate core/wall below $(Int(100 * CORE_WALL_LO))% — the kernel is a small")
            println("  fraction of per-iter work. Candidates to investigate:")
            println("    • @core_timed scope in algorithms/conventional/gradient_descent.jl")
            println("      currently omits step!'s norm/copy bookkeeping (lines 128-132,")
            println("      151-155, 164-166). Widen the scope if these belong in 'kernel'.")
            println("    • problem dimension is small (2D Rosenbrock): kernel work doesn't")
            println("      scale with n here, so per-iter scaffolding dominates regardless.")
            println("      Re-run this check on a higher-dim problem to disambiguate.")
        else
            println("  Aggregate core/wall above $(Int(100 * CORE_WALL_HI))% — @core_timed likely sweeps")
            println("  in non-kernel work (logger, stopping check). Narrow its scope.")
        end
        println()
    end
    return total_ratio
end

# ---------------------------------------------------------------------------
# Bar chart — iters to milestone, log y-axis, faded bars for DNF.
# ---------------------------------------------------------------------------

function plot_milestone_bars(milestone_iters::Dict, summary::Dict;
                              outpath::String,
                              milestone::Float64,
                              budget::Int)
    fig = Figure(size = (1100, 700))
    ax  = Axis(fig[1, 1],
        ylabel = "iterations  (log scale)",
        title  = "Stage 4 — iterations to ‖x − x*‖ ≤ $(milestone)" *
                 "  (budget = $(budget))",
        yscale = log10,
    )

    n = length(PLOT_ORDER)
    heights     = Float64[]
    bar_colors  = Tuple{String, Float64}[]   # (color, alpha)
    annotations = String[]

    for name in PLOT_ORDER
        iters = milestone_iters[name]
        s     = summary[name]
        if isnothing(iters)
            # DNF: bar height = how far the run actually went (n_iters), not
            # the budget cap — so a method that aborted early is visually
            # distinct from one that ran the full budget.
            push!(heights,     Float64(s.n_iters))
            push!(bar_colors,  (COLORS[name], 0.35))
            push!(annotations, "DNF\n:$(s.stop_reason)\nn=$(s.n_iters)")
        else
            push!(heights,     Float64(iters))
            push!(bar_colors,  (COLORS[name], 1.0))
            push!(annotations, "$(iters) iters\n:$(s.stop_reason)")
        end
    end

    barplot!(ax, 1:n, heights;
        color       = bar_colors,
        strokewidth = 0.8,
        strokecolor = :black,
    )

    ax.xticks       = (1:n, PLOT_ORDER)
    ax.ygridvisible = true

    # Annotation above each bar (multiplicative offset works on log scale).
    for i in 1:n
        text!(ax, i, heights[i] * 1.18;
            text     = annotations[i],
            align    = (:center, :bottom),
            fontsize = 10,
        )
    end

    # Headroom for annotations.
    ylims!(ax, 1, budget * 6)

    save(outpath, fig)
    @info "Saved bar chart" path = outpath
    return fig
end

# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

function _replot_from_loaded(exp_result::ExperimentResult, exp_path::String)
    df              = to_dataframe(exp_result)
    summary         = summarize_methods(exp_result)
    milestone_iters = compute_milestone_iters(df; milestone = DIST_MILESTONE)

    print_summary_table(summary, milestone_iters; milestone = DIST_MILESTONE)

    plot_milestone_bars(milestone_iters, summary;
        outpath   = joinpath(exp_path, "milestone_bars.pdf"),
        milestone = DIST_MILESTONE,
        budget    = MAX_ITERS,
    )
    return df, summary, milestone_iters
end

function main()
    exp_mem, exp_path, wall_ns = run_and_save()

    # Stage 3 discipline: plot from disk, never directly from the in-memory
    # result. If something is wrong with the persistence layer, this is where
    # it shows up.
    exp_disk = load_experiment(exp_path)
    _replot_from_loaded(exp_disk, exp_path)

    # Timing report uses the in-memory result (wall_ns isn't persisted).
    print_timing_table(exp_mem, wall_ns)

    println("Experiment saved to: ", exp_path)
    println("Cold-restart validation:")
    println("  julia --project=. -e 'include(\"experiments/exp_stage4.jl\"); replot(\"$exp_path\")'")
    println()
    return exp_disk
end

# Cold-restart entry: load + replot, no run.
function replot(path::String)
    exp_result = load_experiment(path)
    return _replot_from_loaded(exp_result, path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

# ---------------------------------------------------------------------------
# What to look for in the output
# ---------------------------------------------------------------------------
# The printed summary table is the validation surface. Specifically:
#
#   • stop_reason for BB1, BB2: should read :optimal_reached (DistanceToOptimal
#     fires) — confirms DistanceToOptimal works and dist_to_opt is being
#     updated by the runner.
#
#   • stop_reason for Fixed: should read :max_iterations — confirms the
#     CompositeCriteria's :any mode works (other criteria didn't fire).
#
#   • iters→1e-6 column: BB1 and BB2 should be the smallest numbers; Armijo
#     larger; Cauchy larger still or DNF; Fixed DNF. If BB1/BB2 don't lead by
#     a clear margin, suspect the BB grad_prev bookkeeping in step! before
#     suspecting the plot.
#
#   • dist_final: for any row whose stop_reason is :optimal_reached or
#     :gradient_converged, dist_final should be ≤ DIST_TOL (1e-8) or near it.
#     If dist_final is large for those methods, the runner is not updating
#     state.metrics.dist_to_opt before the stopping check.
