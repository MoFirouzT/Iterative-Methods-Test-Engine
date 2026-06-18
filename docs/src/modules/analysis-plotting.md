# Analysis & Plotting

The analysis module has two roles:

1. **DataFrame pipeline** — load a saved experiment, then answer any question by
   filtering, aggregating, and transforming the data.
2. **Figure layout system** — compose any number of plots in any formation and
   render them to a single PDF or image file.

The *rendering* layer is deliberately grid-blind: `PlotSpec` / `render_plot!` consume a flat DataFrame grouped by `:method_name` plus a `method_styles` dict, and know nothing about grids.
Grid structure stays recoverable, though — variant names embed axis information (e.g. `GradientDescent[step_size=Armijo]`), so you can filter on names as plain strings, and the opt-in `grid_styles` helper (below) turns a `VariantGrid`'s axes directly into a `method_styles` dict with no name string-matching at the plot site.

## Loading and Transforming

```julia
result = load_experiment("logs/20260417/001/")

# Convert all iteration logs to a flat DataFrame
# Columns: :run_id, :method_name, :iter, :objective, :gradient_norm,
#          :step_norm, :dist_to_opt, :core_time_ns, + any extras keys
df = to_dataframe(result)

df = filter_methods(df, ["GradientDescent[step_size=Armijo]",
                          "GradientDescent[step_size=BB1]"])
df = aggregate_runs(df, :median)    # :all | :mean | :median
```

**Aggregation Semantics:**
When `aggregate_runs(df, mode)` is called with `:mean` or `:median`, the DataFrame is grouped by `(:method_name, :iter)` pairs to collapse multiple runs into a representative curve per method at each iteration.
Numeric columns (objective, gradient_norm, step_norm, core_time_ns, dist_to_opt, and any numeric extras) are aggregated using the specified mode;
non-numeric columns (strings, booleans, etc.) are preserved from the first row of each group.
The `:run_id` column is dropped during aggregation.

User transforms are plain `DataFrame -> DataFrame` functions:

```julia
transforms = [
    df -> @transform(df, :log_obj      = log10.(:objective)),
    df -> @transform(df, :log_dist     = log10.(max.(:dist_to_opt, 1e-16))),
    df -> @subset(df,   :iter .< 500),
    df -> @transform(df, :core_time_ms = :core_time_ns ./ 1e6),
]
for t in transforms; df = t(df); end
```

## MethodStyle — Per-Method Visual Properties

```julia
@kwdef struct MethodStyle
    color     :: Any
    linestyle :: Symbol    = :solid
    linewidth :: Float64   = 2.0
    marker    :: Union{Nothing, Symbol} = nothing
    label     :: Union{Nothing, String} = nothing
end
```

### Grid-aware styling

`PlotSpec.method_styles` is a `Dict{String,MethodStyle}` keyed by `:method_name`. You can write it by hand, but for variants from a `VariantGrid` the `grid_styles` helper derives it from the grid's axes — mapping one axis to color and an optional second to line style:

```julia
grid_styles(grid::VariantGrid; color_by, style_by = nothing,
            colors = nothing, linestyles = nothing,
            palette = METHOD_PALETTE,
            linestyle_cycle = [:solid, :dash, :dot, :dashdot]) :: Dict{String, MethodStyle}

# color the curves by the preconditioner axis, dash them by the step-size axis
styles = grid_styles(precond_grid(); color_by = :preconditioner, style_by = :step_size)
PlotSpec(data = df, y = :objective, yscale = :log10, method_styles = styles)
```

A visual channel encodes a *dimension of variation*: `color_by` / `style_by` must name axes the grid actually varies. A parameter held fixed across the grid is a constant, not a channel — it belongs in the title. The number of channels equals the number of varied axes: **one axis → color, two → color + line style, three or more → facet into panels** with `FigureLayout` rather than inventing a third channel. `grid_styles` only *produces* a `method_styles` dict; `render_plot!` itself stays grid-blind. Optional `colors` / `linestyles` `Dict("label" => value)` arguments override the defaults per axis label.

## Method Color Registry

Colors are **deterministic and visually appealing** by default.
A fixed curated palette (Wong colorblind-safe + Tableau extensions) is assigned to method names via a stable hash — the same method name always maps to the same color regardless of experiment or run order.

```julia
const METHOD_PALETTE = [
    "#0072B2", "#E69F00", "#009E73", "#CC79A7",
    "#56B4E9", "#D55E00", "#F0E442", "#000000",
]

const METHOD_COLOR_REGISTRY = Dict{String, String}()

function method_color(name::String)::String
    METHOD_PALETTE[(hash(name) % length(METHOD_PALETTE)) + 1]
end

register_method_color!(name::String, color::String) =
    (METHOD_COLOR_REGISTRY[name] = color)

get_method_color(name::String)::String =
    get(METHOD_COLOR_REGISTRY, name, method_color(name))
```

## PlotSpec — Describing a Single Plot

```julia
@kwdef struct PlotSpec
    data          :: DataFrame
    x             :: Symbol          = :iter
    y             :: Symbol          = :objective
    group_by      :: Symbol          = :method_name
    title         :: String          = ""
    xlabel        :: String          = ""
    ylabel        :: String          = ""
    yscale        :: Symbol          = :linear
    xscale        :: Symbol          = :linear
    xlim          :: Union{Nothing,Tuple} = nothing
    ylim          :: Union{Nothing,Tuple} = nothing
    legend        :: Bool            = true
    method_styles :: Dict{String, MethodStyle} = Dict()
    extra_kwargs  :: Dict            = Dict()
end
```

## FigureLayout — Composing Multiple Plots

```julia
@kwdef struct FigureLayout
    plots       :: Matrix{Union{PlotSpec,Nothing}}
    figure_size :: Tuple{Int,Int} = (1200, 900)
    title       :: String = ""
    padding     :: Int = 20
end

# Composable building block: render one PlotSpec into a grid position and
# return the Axis. render_figure is assembled from this.
function render_plot!(gridpos, spec::PlotSpec)
    ax = Axis(gridpos,
        title  = spec.title,
        xlabel = isempty(spec.xlabel) ? string(spec.x) : spec.xlabel,
        ylabel = isempty(spec.ylabel) ? string(spec.y) : spec.ylabel,
        yscale = spec.yscale == :log10 ? log10 : identity,
        xscale = spec.xscale == :log10 ? log10 : identity,
    )
    _render_lines!(ax, spec)
    !isnothing(spec.xlim) && xlims!(ax, spec.xlim...)
    !isnothing(spec.ylim) && ylims!(ax, spec.ylim...)
    return ax
end

function render_figure(layout::FigureLayout)::Makie.Figure
    fig = Figure(size=layout.figure_size)
    for row in 1:size(layout.plots, 1), col in 1:size(layout.plots, 2)
        spec = layout.plots[row, col]
        isnothing(spec) && continue
        render_plot!(fig[row, col], spec)
    end
    isempty(layout.title) || Label(fig[0, :], layout.title, fontsize=18)
    return fig
end

function save_figure(fig::Makie.Figure, path::String)
    save(path, fig)
end
```

`render_figure` handles the all-standard case (a grid of `PlotSpec`s).
When a figure mixes a standard convergence panel with a **bespoke** one —
e.g. the flagship lasso figure pairs an `f − f*`-vs-iteration panel with a hand-rolled support-recovery stem plot —
build the `Figure` yourself, call `render_plot!` for the standard panel(s), and draw the custom panel into the same figure.
`render_plot!` returns the `Axis`, so the engine's plotting layer stays the one clean consumer for convergence curves without forcing every panel through a general API (see `experiments/exp_lasso_ista_fista.jl`).

## End-to-End Plotting Example

```julia
result  = load_experiment("logs/20260417/001/")
df_all  = to_dataframe(result) |> df -> aggregate_runs(df, :median)

styles = Dict(
    "GradientDescent[step_size=Armijo]" => MethodStyle(color="#0072B2", linewidth=2.5),
    "GradientDescent[step_size=BB1]"    => MethodStyle(color="#E69F00", linewidth=2.5),
    "GradientDescent[step_size=Cauchy]" => MethodStyle(color="#009E73", linewidth=2.5),
)

layout = FigureLayout(
    figure_size = (1600, 900),
    title       = "Experiment 001 — Rosenbrock, GD step-size comparison",
    plots = [
        PlotSpec(data=df_all, x=:iter,        y=:objective,     yscale=:log10,
                 title="Objective vs iter", method_styles=styles)   PlotSpec(data=df_all, x=:iter, y=:dist_to_opt, yscale=:log10,
                 title="‖x − x*‖ vs iter",   method_styles=styles);
        PlotSpec(data=df_all, x=:core_time_ns, y=:objective, yscale=:log10,
                 xlabel="Cumulative core time (ns)",
                 title="Objective vs core time")                    nothing
    ],
)

fig = render_figure(layout)
save_figure(fig, "logs/20260417/001/convergence_overview.pdf")
```

## Plotting Across Multiple Experiments

```julia
df1 = to_dataframe(load_experiment("logs/20260417/001/")) |> d -> @transform(d, :exp = "exp1")
df2 = to_dataframe(load_experiment("logs/20260417/002/")) |> d -> @transform(d, :exp = "exp2")
df  = vcat(df1, df2)
# Then build PlotSpec(data=df, group_by=:exp, ...) as normal
```

---
