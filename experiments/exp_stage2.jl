# experiments/exp_stage2.jl
#
# Stage 2 — the Rosenbrock-iconic plot.
# Same five GradientDescent variants as Stage 1, but instead of convergence
# curves we render the (x₁, x₂) trajectories overlaid on log-spaced contours
# of f.  This is the "wow" plot.
#
# Exercises:
#   • extras plumbing all the way through to a DataFrame (here: :x_iter), and
#   • custom Makie code outside FigureLayout — trajectory plots are a different
#     kind of figure and trying to force them through the layout DSL hurts more
#     than it helps at this stage.
#
# To run, from project root:
#     julia --project=. experiments/exp_stage2.jl

include("../src/TestEngine.jl")
using .TestEngine
using Random
using DataFrames
using CairoMakie

# ---------------------------------------------------------------------------
# Configuration — mirrors Stage 1 so trajectories overlay the same runs we
# already plotted as convergence curves.
# ---------------------------------------------------------------------------

const SEED   = 42
const RUN_ID = 1

const PLOT_ORDER = ["Fixed", "Armijo", "Cauchy", "BB1", "BB2"]

# Wong colorblind-safe palette (consistent with METHOD_PALETTE in analysis.jl
# and with exp_stage1.jl).
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
        MaxIterations(n   = 2000),
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
        @info "[$name] done" iters       = result.n_iters
                              stop_reason = result.stop_reason
                              f_final     = last_entry.objective
                              dist_to_opt = last_entry.dist_to_opt
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
# Trajectory extraction — pull (x₁, x₂) sequences out of iter_logs.
# Returned as a vector of (xs, ys) pairs in PLOT_ORDER.
# ---------------------------------------------------------------------------

function extract_trajectories(results, problem)
    res_dict = Dict(results)
    out = Vector{Tuple{String, Vector{Float64}, Vector{Float64}}}()
    for name in PLOT_ORDER
        haskey(res_dict, name) || continue
        traj_x = Float64[problem.x0[1]]   # ← seed with x₀
        traj_y = Float64[problem.x0[2]]   # ←
        for entry in res_dict[name].iter_logs
            entry.iter == 0 && continue   # skip init (no :x_iter and would dup x₀)
            haskey(entry.extras, :x_iter) || continue
            x = entry.extras[:x_iter]
            length(x) == 2 || continue
            push!(traj_x, x[1])
            push!(traj_y, x[2])
        end
        push!(out, (name, traj_x, traj_y))
    end
    return out
end

# ---------------------------------------------------------------------------
# Plotting — log-spaced contours + overlaid trajectories.
# ---------------------------------------------------------------------------

function plot_stage2(results, problem; outpath::String = "stage2_trajectories.pdf")
    fig = Figure(size = (1000, 900))
    ax  = Axis(fig[1, 1],
        xlabel = "x₁",
        ylabel = "x₂",
        title  = "Stage 2 — GradientDescent on Rosenbrock(ρ=100): trajectories from x₀=(−1.2, 1)",
        aspect = DataAspect(),
    )

    # ── Contour grid ────────────────────────────────────────────────────────
    # Hardcoded Rosenbrock — the contour is reference geometry, not a result of
    # the framework's machinery, so reaching for `value(problem.f, ·)` here
    # would couple the figure unnecessarily to whichever Objective subtype
    # implements the spec. Pull ρ from problem.meta when available, fall back
    # to 100.0 otherwise.
    ρ = Float64(get(problem.meta, :rho, 100.0))
    rosen(x, y) = (1.0 - x)^2 + ρ * (y - x^2)^2

    xs = range(-2.0, 2.0, length = 400)
    ys = range(-1.0, 3.0, length = 400)
    zs = [rosen(x, y) for x in xs, y in ys]

    # Log-spaced contour levels: f ranges from 0 (at x*) to ~2.5×10³ on this box.
    # 15 levels uniform on log give nicely banded curvature, dense in the
    # narrow valley and sparse on the flatlands.
    levels = 10.0 .^ range(-1.0, 3.5, length = 15)

    contour!(ax, xs, ys, zs;
        levels    = levels,
        color     = (:gray, 0.55),    # uniform gray, no colormap → trajectories pop
        linewidth = 0.6,
    )

    # ── Trajectories ────────────────────────────────────────────────────────
    for (name, traj_x, traj_y) in extract_trajectories(results, problem)
        lines!(ax, traj_x, traj_y;
            color     = COLORS[name],
            linewidth = 1.8,
            label     = name,
        )
    end

    # ── Markers: x₀ (black filled circle) and x* (red star) ─────────────────
    scatter!(ax, [problem.x0[1]], [problem.x0[2]];
        color        = :black,
        marker       = :circle,
        markersize   = 14,
        strokecolor  = :white,
        strokewidth  = 1.5,
        label        = "x₀",
    )
    if !isnothing(problem.x_opt)
        scatter!(ax, [problem.x_opt[1]], [problem.x_opt[2]];
            color        = :red,
            marker       = :star5,
            markersize   = 20,
            strokecolor  = :white,
            strokewidth  = 1.5,
            label        = "x*",
        )
    end

    xlims!(ax, -2.0, 2.0)
    ylims!(ax, -1.0, 3.0)

    axislegend(ax;
        position        = :rt,
        framevisible    = true,
        backgroundcolor = (:white, 0.85),
        nbanks          = 1,
    )

    save(outpath, fig)
    @info "Saved figure" path = outpath
    return fig
end

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
    @info "extras plumbing" rows = n_total with_x_iter = n_with_xiter
    @assert n_with_xiter == n_total ("x_iter missing on $(n_total - n_with_xiter)/$(n_total) rows — " *
                                     "has the extract_log_entry patch been applied?")

    plot_stage2(results, problem; outpath = "stage2_trajectories.pdf")
    return df
end

# Run only when executed as a script, not when included for interactive use.
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
