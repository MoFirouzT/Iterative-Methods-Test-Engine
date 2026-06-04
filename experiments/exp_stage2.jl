# experiments/exp_stage2.jl
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
    fig = Figure(size = (1100, 950))
    ax  = Axis(fig[1, 1],
        xlabel = "x₁",
        ylabel = "x₂",
        title  = "Stage 2 — GradientDescent on Rosenbrock(ρ=100): trajectories from x₀=(−1.2, 1)",
        aspect = DataAspect(),
    )

    # Iteration counts → legend labels (e.g. "BB1 (n=47)").
    iter_counts = Dict(name => res.n_iters for (name, res) in results)

    trajectories = extract_trajectories(results, problem)

    # ── Main-plot extent ───────────────────────────────────────────────────
    # BB1's worst spikes go very far off; capturing all of them shrinks the
    # interesting valley behavior to nothing. Fix the window instead — wide
    # enough to show most of the BB wandering, tight enough to keep the
    # valley legible.
    x_min, x_max = -3.0, 3.0
    y_min, y_max = -3.0,  4.0

    # ── Contour grid ────────────────────────────────────────────────────────
    # Hardcoded Rosenbrock — the contour is reference geometry, not a result of
    # the framework's machinery, so reaching for `value(problem.f, ·)` here
    # would couple the figure unnecessarily to whichever Objective subtype
    # implements the spec. Pull ρ from problem.meta when available, fall back
    # to 100.0 otherwise.
    ρ = Float64(get(problem.meta, :rho, 100.0))
    rosen(x, y) = (1.0 - x)^2 + ρ * (y - x^2)^2

    xs = range(x_min, x_max, length = 500)
    ys = range(y_min, y_max, length = 500)
    zs = [rosen(x, y) for x in xs, y in ys]

    # Log-spaced contour levels: f ranges from 0 (at x*) to ~2.5×10³ on this box.
    # 15 levels uniform on log give nicely banded curvature, dense in the
    # narrow valley and sparse on the flatlands.
    levels = 10.0 .^ range(-1.0, 3.5, length = 15)

    # Low-chroma reversed bone — light in the valley (trajectories pop), darker
    # on the cliffs. Avoids competing with any of the five trajectory hues.
    heatmap!(ax, xs, ys, log10.(max.(zs, 1e-2));
        colormap = Reverse(:bone),
        alpha    = 0.35,
    )
    contour!(ax, xs, ys, zs;
        levels    = levels,
        color     = (:gray, 0.55),
        linewidth = 0.6,
    )

    # ── Trajectories (alpha 0.7 so valley overlaps shade rather than overwrite) ─
    for (name, traj_x, traj_y) in trajectories
        lines!(ax, traj_x, traj_y;
            color     = (COLORS[name], 0.7),
            linewidth = 1.8,
            label     = "$name (n=$(iter_counts[name]))",
        )
        # Cadence dots — ~12 per trajectory, regardless of length.
        # Lets you see Fixed's slow march vs. BB1's few violent steps at a glance.
        stride = max(1, length(traj_x) ÷ 12)
        idx    = 1:stride:length(traj_x)
        scatter!(ax, traj_x[idx], traj_y[idx];
            color       = COLORS[name],
            marker      = :circle,
            markersize  = 7,
            strokecolor = :white,
            strokewidth = 0.7,
        )
        # Endpoint marker — diamond at the terminal iterate, with strong stroke
        # so it pops out of the trajectory's own line color.
        scatter!(ax, [traj_x[end]], [traj_y[end]];
            color       = COLORS[name],
            marker      = :diamond,
            markersize  = 13,
            strokecolor = :black,
            strokewidth = 1.2,
        )
    end

    # ── Markers: x₀ (black filled circle) and x* (red star) ─────────────────
    scatter!(ax, [problem.x0[1]], [problem.x0[2]];
        color        = :black,
        marker       = :circle,
        markersize   = 18,
        strokecolor  = :white,
        strokewidth  = 2,
        label        = "x₀",
    )
    text!(ax, problem.x0[1], problem.x0[2];
        text     = "  x₀",
        align    = (:left, :center),
        fontsize = 14,
        font     = :bold,
    )
    if !isnothing(problem.x_opt)
        scatter!(ax, [problem.x_opt[1]], [problem.x_opt[2]];
            color        = :red,
            marker       = :star5,
            markersize   = 22,
            strokecolor  = :white,
            strokewidth  = 2,
            label        = "x*",
        )
    end

    # ── Inset-region indicator (dashed box on main plot) ────────────────────
    # Tight window around x* — shows the "convergence dance" of the methods
    # that actually arrived. Methods whose endpoints fall outside the box
    # (notably Fixed, which is far from x*) are already legible on the main
    # plot via their endpoint diamonds; the inset is dedicated to the close-in
    # landing comparison.
    ins_xlo, ins_xhi = 0.85, 1.10
    ins_ylo, ins_yhi = 0.85, 1.10
    lines!(ax,
        [ins_xlo, ins_xhi, ins_xhi, ins_xlo, ins_xlo],
        [ins_ylo, ins_ylo, ins_yhi, ins_yhi, ins_ylo];
        color     = :red,
        linewidth = 2.0,
        linestyle = :dash,
    )

    xlims!(ax, x_min, x_max)
    ylims!(ax, y_min, y_max)

    axislegend(ax;
        position        = :rt,
        framevisible    = true,
        backgroundcolor = (:white, 0.85),
        nbanks          = 1,
    )

    # ── Inset axis: zoom near x* ────────────────────────────────────────────
    # Anchor the inset to the main axis's viewport (not its GridLayout cell),
    # so it sits inside the axis frame rather than overlapping the margin/labels.
    inset_bbox = lift(ax.scene.viewport) do vp
        w   = 0.32 * vp.widths[1]
        h   = 0.32 * vp.widths[2]
        pad = 0.025 * min(vp.widths[1], vp.widths[2])
        x   = vp.origin[1] + pad
        y   = vp.origin[2] + pad
        Rect2f(Vec2f(x, y), Vec2f(w, h))
    end
    ax_inset = Axis(fig.scene,
        bbox                = inset_bbox,
        aspect              = DataAspect(),
        backgroundcolor     = (:white, 0.95),
        title               = "zoom near x*",
        titlesize           = 11,
        xlabelvisible       = false,
        ylabelvisible       = false,
        xticklabelsvisible  = false,
        yticklabelsvisible  = false,
        xticksvisible       = false,
        yticksvisible       = false,
        # Red frame matches the inset-region box on the main plot.
        leftspinecolor      = :red,
        rightspinecolor     = :red,
        topspinecolor       = :red,
        bottomspinecolor    = :red,
        spinewidth          = 1.5,
    )
    translate!(ax_inset.blockscene, 0, 0, 1000)  # ensure inset sits above main

    xs_i = range(ins_xlo, ins_xhi, length = 250)
    ys_i = range(ins_ylo, ins_yhi, length = 250)
    zs_i = [rosen(x, y) for x in xs_i, y in ys_i]
    contour!(ax_inset, xs_i, ys_i, zs_i;
        levels    = levels,
        color     = (:gray, 0.55),
        linewidth = 0.6,
    )
    for (name, traj_x, traj_y) in trajectories
        lines!(ax_inset, traj_x, traj_y;
            color     = (COLORS[name], 0.85),
            linewidth = 1.6,
        )
        scatter!(ax_inset, [traj_x[end]], [traj_y[end]];
            color       = COLORS[name],
            marker      = :diamond,
            markersize  = 11,
            strokecolor = :black,
            strokewidth = 1.0,
        )
    end
    if !isnothing(problem.x_opt)
        scatter!(ax_inset, [problem.x_opt[1]], [problem.x_opt[2]];
            color       = :red,
            marker      = :star5,
            markersize  = 14,
            strokecolor = :white,
            strokewidth = 1.5,
        )
    end
    xlims!(ax_inset, ins_xlo, ins_xhi)
    ylims!(ax_inset, ins_ylo, ins_yhi)

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
