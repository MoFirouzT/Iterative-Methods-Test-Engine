# experiments/_shared.jl
#
# Shared plotting recipes consumed by experiment stages.
# Not a runnable experiment — leading underscore signals "helper".
#
# Currently exposes:
#   • plot_trajectories(df, problem; outpath, plot_order, colors, ...)
#       The canonical Rosenbrock-2D trajectory figure: heatmap shading of
#       log₁₀(f), gray contour lines, per-method trajectories with cadence
#       dots and endpoint diamonds, an inset zoomed near x*, and x₀ / x*
#       markers. Stage 2 and Stage 3 both call this; centralising it stops
#       the two stages from drifting in style.
#
# Schema expected on the DataFrame:
#   :method_name :: AbstractString
#   :iter        :: Integer
#   :x_iter      :: Vector{Float64} (length 2) or missing
# Other columns are ignored. The trajectory for each method is reconstructed
# by sorting on :iter and concatenating :x_iter values; x₀ is prepended from
# `problem.x0` so the trajectory always starts at the initial point even if
# the iter=0 row is missing :x_iter (different log verbosities may or may
# not emit it).

using DataFrames
using CairoMakie

# ---------------------------------------------------------------------------
# Shared constants — used by Stages 1–4 (Stages 5+ use the long
# "GradientDescent[step_size=Armijo]" form via the VariantGrid orchestrator,
# so they keep their own naming).
# ---------------------------------------------------------------------------

const PLOT_ORDER = ["Fixed", "Armijo", "Cauchy", "BB1", "BB2"]

# Wong colorblind-safe palette, consistent with METHOD_PALETTE in analysis.jl.
const COLORS = Dict(
    "Fixed"  => "#000000",
    "Armijo" => "#0072B2",
    "Cauchy" => "#009E73",
    "BB1"    => "#E69F00",
    "BB2"    => "#D55E00",
)

"""
    build_standard_methods()

The five-method baseline used across Stages 1–4: SteepestDescent with
FixedStep / ArmijoLS / CauchyStep / BB1 / BB2 step-size rules. Returned
as a `Vector{Pair{String, GradientDescent}}` in `PLOT_ORDER`.
"""
function build_standard_methods()
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

"""
    build_ls_methods(L)

The same five step-size rules as `build_standard_methods`, tuned for a linear
least-squares quadratic with Lipschitz constant `L = σ_max(A)²`:
- `FixedStep(α = 1/L)` — the stable fixed step for a quadratic (the Rosenbrock
  default `8e-4` would be far too small here).
- `CauchyStep(α_max = Inf)` — the quadratic model is exact, so no trust-radius
  cap; this is genuine exact-line-search steepest descent.
Reuses `PLOT_ORDER` / `COLORS`. Returned in `PLOT_ORDER`.
"""
function build_ls_methods(L)
    [
        "Fixed"  => GradientDescent(direction = SteepestDescent(),
                                    step_size = FixedStep(α = 1 / L)),
        "Armijo" => GradientDescent(direction = SteepestDescent(),
                                    step_size = ArmijoLS()),
        "Cauchy" => GradientDescent(direction = SteepestDescent(),
                                    step_size = CauchyStep(α_max = Inf)),
        "BB1"    => GradientDescent(direction = SteepestDescent(),
                                    step_size = BarzilaiBorwein(variant = :BB1)),
        "BB2"    => GradientDescent(direction = SteepestDescent(),
                                    step_size = BarzilaiBorwein(variant = :BB2)),
    ]
end

# ---------------------------------------------------------------------------
# Trajectory extraction from DataFrame
# ---------------------------------------------------------------------------

function _extract_trajectories_from_df(df::DataFrame, plot_order, problem)
    out = Tuple{String, Vector{Float64}, Vector{Float64}}[]
    for name in plot_order
        sub = filter(:method_name => ==(name), df)
        isempty(sub) && continue
        sub = sort(sub, :iter)
        xs = Float64[problem.x0[1]]
        ys = Float64[problem.x0[2]]
        for row in eachrow(sub)
            row.iter == 0 && continue   # skip init; we already seeded x₀
            v = row.x_iter
            ismissing(v) && continue
            length(v) == 2 || continue
            push!(xs, v[1]); push!(ys, v[2])
        end
        length(xs) > 1 || continue
        push!(out, (name, xs, ys))
    end
    return out
end

# ---------------------------------------------------------------------------
# Trajectory plot
# ---------------------------------------------------------------------------

"""
    plot_trajectories(df, problem; outpath, plot_order, colors,
                      title        = "Rosenbrock(ρ=100): trajectories",
                      main_extent  = (-3.0, 3.0, -3.0, 4.0),
                      inset_extent = (0.85, 1.10, 0.85, 1.10))

Render the canonical 2D Rosenbrock trajectory figure used by Stages 2 and 3.

Required keyword arguments:
- `outpath`      — file path to save the figure to.
- `plot_order`   — `Vector{String}` of method names in legend/draw order.
- `colors`       — `Dict{String,String}` mapping method name → hex color.

Optional:
- `title`        — full axis title.
- `main_extent`  — `(xlo, xhi, ylo, yhi)` of the main plot window.
- `inset_extent` — `(xlo, xhi, ylo, yhi)` of the zoom-near-x* inset window.

Rosenbrock geometry is closed-form here (not pulled from `problem.f`) — the
contour is *reference* geometry, not a result of the framework. `ρ` is read
from `problem.meta[:rho]`, falling back to 100.0.
"""
function plot_trajectories(df::DataFrame, problem;
                           outpath::String,
                           plot_order,
                           colors,
                           title::String = "Rosenbrock(ρ=100): trajectories",
                           main_extent  = (-3.0, 3.0, -3.0, 4.0),
                           inset_extent = (0.85, 1.10, 0.85, 1.10))
    x_min, x_max, y_min, y_max         = main_extent
    ins_xlo, ins_xhi, ins_ylo, ins_yhi = inset_extent

    fig = Figure(size = (1100, 950))
    ax  = Axis(fig[1, 1],
        xlabel = "x₁",
        ylabel = "x₂",
        title  = title,
        aspect = DataAspect(),
    )

    trajectories = _extract_trajectories_from_df(df, plot_order, problem)
    iter_counts  = Dict{String, Int}()
    for name in plot_order
        sub = filter(:method_name => ==(name), df)
        isempty(sub) && continue
        iter_counts[name] = maximum(sub.iter)
    end

    # ── Contour grid (closed-form Rosenbrock) ──────────────────────────────
    ρ = Float64(get(problem.meta, :rho, 100.0))
    rosen(x, y) = (1.0 - x)^2 + ρ * (y - x^2)^2

    xs = range(x_min, x_max, length = 500)
    ys = range(y_min, y_max, length = 500)
    zs = [rosen(x, y) for x in xs, y in ys]
    levels = 10.0 .^ range(-1.0, 3.5, length = 15)

    # Low-chroma reversed bone — light in the valley, darker on the cliffs.
    heatmap!(ax, xs, ys, log10.(max.(zs, 1e-2));
        colormap = Reverse(:bone),
        alpha    = 0.35,
    )
    contour!(ax, xs, ys, zs;
        levels    = levels,
        color     = (:gray, 0.55),
        linewidth = 0.6,
    )

    # ── Trajectories + cadence dots + endpoint diamonds ────────────────────
    for (name, traj_x, traj_y) in trajectories
        lines!(ax, traj_x, traj_y;
            color     = (colors[name], 0.7),
            linewidth = 1.8,
            label     = "$name (n=$(get(iter_counts, name, length(traj_x)-1)))",
        )
        stride = max(1, length(traj_x) ÷ 12)
        idx    = 1:stride:length(traj_x)
        scatter!(ax, traj_x[idx], traj_y[idx];
            color       = colors[name],
            marker      = :circle,
            markersize  = 7,
            strokecolor = :white,
            strokewidth = 0.7,
        )
        scatter!(ax, [traj_x[end]], [traj_y[end]];
            color       = colors[name],
            marker      = :diamond,
            markersize  = 13,
            strokecolor = :black,
            strokewidth = 1.2,
        )
    end

    # ── x₀ and x* markers ──────────────────────────────────────────────────
    scatter!(ax, [problem.x0[1]], [problem.x0[2]];
        color       = :black,
        marker      = :circle,
        markersize  = 18,
        strokecolor = :white,
        strokewidth = 2,
        label       = "x₀",
    )
    text!(ax, problem.x0[1], problem.x0[2];
        text     = "  x₀",
        align    = (:left, :center),
        fontsize = 14,
        font     = :bold,
    )
    if !isnothing(problem.x_opt)
        scatter!(ax, [problem.x_opt[1]], [problem.x_opt[2]];
            color       = :red,
            marker      = :star5,
            markersize  = 22,
            strokecolor = :white,
            strokewidth = 2,
            label       = "x*",
        )
    end

    # ── Inset-region box on main plot ──────────────────────────────────────
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

    # ── Inset axis: zoom near x*, anchored inside the main axis viewport ───
    inset_bbox = lift(ax.scene.viewport) do vp
        w   = 0.32 * vp.widths[1]
        h   = 0.32 * vp.widths[2]
        pad = 0.025 * min(vp.widths[1], vp.widths[2])
        x   = vp.origin[1] + pad
        y   = vp.origin[2] + pad
        Rect2f(Vec2f(x, y), Vec2f(w, h))
    end
    ax_inset = Axis(fig.scene,
        bbox               = inset_bbox,
        aspect             = DataAspect(),
        backgroundcolor    = (:white, 0.95),
        title              = "zoom near x*",
        titlesize          = 11,
        xlabelvisible      = false,
        ylabelvisible      = false,
        xticklabelsvisible = false,
        yticklabelsvisible = false,
        xticksvisible      = false,
        yticksvisible      = false,
        leftspinecolor     = :red,
        rightspinecolor    = :red,
        topspinecolor      = :red,
        bottomspinecolor   = :red,
        spinewidth         = 1.5,
    )
    translate!(ax_inset.blockscene, 0, 0, 1000)

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
            color     = (colors[name], 0.85),
            linewidth = 1.6,
        )
        scatter!(ax_inset, [traj_x[end]], [traj_y[end]];
            color       = colors[name],
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
