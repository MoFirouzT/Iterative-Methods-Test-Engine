# experiments/stages/stage2.jl
#
# Stage 2 — the Rosenbrock-iconic plot.
# Same five GradientDescent variants as Stage 1, but instead of convergence
# curves we render the (x₁, x₂) trajectories overlaid on log-spaced contours of f.
#
# Exercises:
#   • extras plumbing all the way through to a DataFrame (here: :x_iter), and
#   • custom Makie code outside FigureLayout — trajectory plots are a different
#     kind of figure and trying to force them through the layout DSL hurts more
#     than it helps at this stage.
#
# To run, from project root:
#     julia --project=. experiments/stages/stage2.jl

include("../_bootstrap.jl")   # engine + all content (problems, methods, components)
using Random
using DataFrames
using CairoMakie

# Canonical trajectory-plot recipe, shared with Stage 3.
include("../_shared.jl")

# ---------------------------------------------------------------------------
# Configuration — mirrors Stage 1 so trajectories overlay the same runs we
# already plotted as convergence curves.
# ---------------------------------------------------------------------------

const SEED   = 42
const RUN_ID = 1

# PLOT_ORDER, COLORS, build_standard_methods live in _shared.jl.
const build_methods = build_standard_methods

# ---------------------------------------------------------------------------
# Driver — same as Stage 1, but returns the problem too (we need x0, x_opt,
# and ρ for the contour grid and the markers).
# ---------------------------------------------------------------------------

function run_stage2(; seed::Int = SEED, run_id::Int = RUN_ID)
    problem_spec = AnalyticProblem(
        name   = :rosenbrock,
        params = (rho = 100.0, dim = 2),
    )

    criteria = stop_when_any(
        MaxIterations(n   = 700),
        GradientTolerance(tol = 1e-9),
    )

    rng_data = Xoshiro(hash((seed, run_id, :data)))
    problem  = make_problem(problem_spec, rng_data)

    @info "Stage 2 — running" problem = "Rosenbrock(ρ=100)" x0 = problem.x0 x_opt = problem.x_opt
    results = Pair{String, Any}[]
    for (name, method) in build_methods()
        method_rng = Xoshiro(hash((seed, run_id, name)))
        logger     = make_logger(name, run_id, "",
                                 VerbosityConfig(level = MILESTONE))
        result     = run_method(method, problem, criteria, logger, method_rng)
        push!(results, name => result)

        last_entry = result.iter_logs[end]
        @info("[$name] done",
              iters       = result.n_iters,
              stop_reason = result.stop_reason,
              f_final     = last_entry.objective,
              dist_to_opt = last_entry.dist_to_opt)
    end
    return results, problem
end

# ---------------------------------------------------------------------------
# DataFrame adapter — same schema as Stage 1, plus the new :x_iter column.
# This is the "extras plumbing through to DataFrame" exercise: we don't use
# the DataFrame for plotting (custom Makie code below), but we build it and
# sanity-check that x_iter survives the round trip.
# ---------------------------------------------------------------------------

function results_to_df(results::Vector; run_id::Int = RUN_ID)
    rows = NamedTuple[]
    for (name, res) in results
        for entry in res.iter_logs
            push!(rows, (
                run_id        = run_id,
                method_name   = name,
                iter          = entry.iter,
                objective     = entry.objective,
                gradient_norm = entry.gradient_norm,
                step_norm     = entry.step_norm,
                dist_to_opt   = entry.dist_to_opt,
                core_time_ns  = entry.core_time_ns,
                step_size     = get(entry.extras, :step_size, NaN),
                x_iter        = get(entry.extras, :x_iter, missing),
            ))
        end
    end
    return DataFrame(rows)
end

# ---------------------------------------------------------------------------
# Plotting — log-spaced contours + overlaid trajectories.
# Rendering itself lives in experiments/_shared.jl (the same recipe is also
# used by Stage 3); this file only owns the run-loop + DataFrame adapter.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

function main()
    results, problem = run_stage2()
    df               = results_to_df(results)

    # Sanity-check the extras → DataFrame plumbing: every method ran in 2D, so
    # every row should carry a non-missing :x_iter. If this fires, the package
    # change documented at the top of the file probably hasn't been applied.
    n_with_xiter = count(!ismissing, df.x_iter)
    n_total      = nrow(df)
    @info("extras plumbing", rows = n_total, with_x_iter = n_with_xiter)
    @assert n_with_xiter == n_total ("x_iter missing on $(n_total - n_with_xiter)/$(n_total) rows — " *
                                     "has the extract_log_entry patch been applied?")

    # PNG into stages/figures/ — this is the figure surfaced in the repo README's
    # "Going deeper" gateway, so it regenerates from this script alone.
    figdir = joinpath(@__DIR__, "figures")
    mkpath(figdir)
    plot_trajectories(df, problem;
        outpath    = joinpath(figdir, "stage2_trajectories.png"),
        plot_order = PLOT_ORDER,
        colors     = COLORS,
        title      = "Stage 2 — GradientDescent on Rosenbrock(ρ=100): trajectories from x₀=(−1.2, 1)",
    )
    return df
end

# Run only when executed as a script, not when included for interactive use.
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
