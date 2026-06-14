# experiments/stages/stage5.jl
# =============================================================================
# Stage 5 — Orchestrator debut: run_experiment + VariantGrid + fair-comparison
#                                plots
# -----------------------------------------------------------------------------
# First experiment that drives the framework through its actual user-facing
# entry point — `run_experiment` — rather than a hand-rolled loop. The same five
# GradientDescent step-size variants from Stage 1 are defined here via a single
# VariantAxis on the `:step_size` parameter, expanded by the grid engine, routed
# into the baseline bucket by `resolve_methods` (the grid declares
# `role = :baseline`), and run through the orchestrator.
#
# Exercises
#   * VariantAxis, VariantGrid, expand, VariantSpec (Cartesian product +
#     auto-naming).
#   * ABBREVIATIONS registry + register_abbreviation!.
#   * resolve_methods routing by the grid's declared role.
#   * run_experiment's deterministic per-(seed, run_id, name) rng derivation.
#   * n_linesearch_evals accounting inside Armijo's backtracking loop.
#   * per-iter core_time_ns accumulation in Logger.total_core_ns.
#
# Validates
#   * Five methods complete; BB1/BB2 fastest; Armijo step-size column shows
#     discrete β^j values; Fixed's step-size column is constant.
#   * Optional byte-identical assertion against Stage 1's iter logs on the
#     four randomness-free columns. The assertion passes only because no
#     `step!` in this grid draws from rng — it would silently become vacuous
#     if a stochastic step-size component were ever added to the grid.
#
# Plots — Stage 1's four panels reproduced three times, with the x-axis being:
#   1. iter                       — baseline, identical to Stage 1's figure.
#   2. cumulative function evals  — 1 per iter, plus n_linesearch_evals from
#                                   the backtracking line search.
#   3. cumulative core_time_ns    — sanity-check only on 2D Rosenbrock; the
#                                   kernel is tens of nanoseconds and OS jitter
#                                   shifts cumulative time by 5–10%, so this
#                                   plot is not authoritative for ordering.
# =============================================================================

# Match the Stages 1–4 include pattern. TestEngine is a local module under
# src/, not a registered package; `using TestEngine` would fail.
include("../_bootstrap.jl")   # engine + all content (problems, methods, components)
using DataFrames, DataFramesMeta
using CairoMakie
using Statistics
using Printf

# COLORS / PLOT_ORDER live here; Stage 5 reuses them so legend colors stay
# consistent across Stages 1–5.
include("../_shared.jl")

# ─── Setup ───────────────────────────────────────────────────────────────────

# Abbreviations must be registered BEFORE expand() runs so that the
# short_name of each generated VariantSpec uses the friendly form rather than
# the long type names (see Module 4 — naming convention).
register_abbreviation!("GradientDescent", "GD")
register_abbreviation!("FixedStep",       "Fix")
register_abbreviation!("ArmijoLS",        "Arm")
register_abbreviation!("CauchyStep",      "Cau")
register_abbreviation!("BarzilaiBorwein", "BB")

# ─── Grid ────────────────────────────────────────────────────────────────────
#
# Single axis along the :step_size parameter. The same five values used in
# Stage 1, expressed declaratively. expand(grid) produces 5 VariantSpec
# instances; because the grid declares `role = :baseline`, resolve_methods
# routes them into the baseline bucket.

step_size_axis = VariantAxis(:step_size,
    FixedStep(α=8e-4)                                  => "Fixed",
    ArmijoLS(α₀=1.0, β=0.5, c₁=1e-4, max_iter=50)      => "Armijo",
    CauchyStep()                                       => "Cauchy",
    BarzilaiBorwein(variant=:BB1)                      => "BB1",
    BarzilaiBorwein(variant=:BB2)                      => "BB2",
)

grid = VariantGrid(
    base_name = "GradientDescent",
    axes      = [step_size_axis],
    builder   = (; step_size, kwargs...) ->
                    GradientDescent(direction=SteepestDescent(), step_size=step_size),
    role = :baseline,
)

# ─── Experiment ──────────────────────────────────────────────────────────────

criteria = stop_when_any(
    MaxIterations(n=2000),
    GradientTolerance(tol=1e-9),
)

config = ExperimentConfig(
    name              = "Stage 5 — VariantGrid + run_experiment + fair-comparison plots",
    problem_spec      = AnalyticProblem(
                            name   = :rosenbrock,
                            params = (ρ = 100.0, x0 = [-1.2, 1.0]),
                        ),
    variant_grids     = [grid],
    stopping_criteria = criteria,
    n_runs            = 1,
    seed              = 42,
    tags              = Dict{String,Any}("stage" => "5", "problem" => "rosenbrock"),
)

verbosity = VerbosityConfig(level=MILESTONE)
println("\n=== Stage 5 — running orchestrator ===\n")
result = run_experiment(config; verbosity=verbosity)
println("\nExperiment saved to: ", result.experiment_path)

# ─── Optional byte-identity check vs Stage 1 ─────────────────────────────────
#
# Stage 1 used the short name (e.g. "Armijo") as the per-method rng key.
# Stage 5 uses the long form ("GradientDescent[step_size=Armijo]"), so the
# rng streams differ. The deterministic iter logs are still identical because
# no `step!` in this grid actually consumes from rng.
#
# Bug surface: introduce any stochastic step-size component and this
# assertion silently becomes vacuous — comment kept in the file so the next
# person extending the grid sees the warning.

function assert_stage1_identity(result5, stage1_path::AbstractString)
    isdir(stage1_path) || error("Stage 1 result directory not found: $stage1_path")
    result1 = load_experiment(stage1_path)

    df1 = to_dataframe(result1)
    df5 = to_dataframe(result5)

    name_map = Dict(
        "Fixed"  => "GradientDescent[step_size=Fixed]",
        "Armijo" => "GradientDescent[step_size=Armijo]",
        "Cauchy" => "GradientDescent[step_size=Cauchy]",
        "BB1"    => "GradientDescent[step_size=BB1]",
        "BB2"    => "GradientDescent[step_size=BB2]",
    )

    for (short, long) in name_map
        d1 = @subset(df1, :method_name .== short)
        d5 = @subset(df5, :method_name .== long)
        nrow(d1) == nrow(d5) ||
            error("Row count mismatch for $short ↔ $long: $(nrow(d1)) vs $(nrow(d5))")
        for col in (:iter, :objective, :gradient_norm, :dist_to_opt)
            (col in propertynames(d1) && col in propertynames(d5)) ||
                error("Column $col missing in one of the DataFrames")
            isequal(d1[!, col], d5[!, col]) ||
                error("Column $col differs for $short ↔ $long")
        end
    end
    println("✓ Stage 1 byte-identity check passed for all 5 methods.")
end

let stage1_path = get(ENV, "STAGE1_PATH", "")
    if !isempty(stage1_path)
        println("\n=== Stage 5 ↔ Stage 1 byte-identity check ===")
        try
            assert_stage1_identity(result, stage1_path)
        catch e
            @error "Stage 1 byte-identity check failed" exception=e
        end
    else
        @info "Skipping Stage 1 byte-identity check (set STAGE1_PATH=logs/.../001 to enable)."
    end
end

# ─── DataFrame + cumulative columns ──────────────────────────────────────────

df = to_dataframe(result)
sort!(df, [:method_name, :iter])

# Cumulative function evaluations at iter k =
#   k            (one base gradient/value call per iter)
# + ls[k]        (cumulative line-search trial count up to and including iter k)
# where ls[k] is read directly from :n_linesearch_evals — the framework already
# stores the running total per docstring in gradient_descent.jl. Summing it
# again per row (as an earlier draft did) double-counts catastrophically and
# inflates Armijo's reported Σevals to ~20M instead of ~20k.
function _attach_cumulative!(df)
    has_ls = :n_linesearch_evals in propertynames(df)
    df.cum_core_time_ns = Vector{Float64}(undef, nrow(df))
    df.cum_n_evals      = Vector{Int}(undef, nrow(df))
    for sub in groupby(df, :method_name)
        idx = parentindices(sub)[1]
        df.cum_core_time_ns[idx] = cumsum(Float64.(sub.core_time_ns))
        ls = has_ls ? coalesce.(sub.n_linesearch_evals, 0) : zeros(Int, nrow(sub))
        df.cum_n_evals[idx]      = sub.iter .+ ls
    end
    df.cum_core_time_ms = df.cum_core_time_ns ./ 1e6
    return df
end
_attach_cumulative!(df)

# ─── Display name rename + COLORS lookup ─────────────────────────────────────
#
# The orchestrator stores method_name as e.g. "GradientDescent[step_size=Armijo]"
# — that's the canonical full name and it's what disk artifacts and the
# byte-identity check above use. For figures/legends/tables it's just noise:
# every entry says "GradientDescent[step_size=…]" and the only thing that
# varies is the label after `=`. We rename to the compact `GD[ss=Armijo]` form
# here for everything *downstream of the byte-identity check*. The on-disk
# manifest/CSVs/JLD2 keep the canonical name.
#
# The label inside the brackets ("Armijo", "BB1", ...) is exactly the key used
# by COLORS / PLOT_ORDER in _shared.jl, so palette lookup is a direct match.

const _LONG_NAME_RE = r"^GradientDescent\[step_size=(.+)\]$"

function _short_label(long_name::AbstractString)
    m = match(_LONG_NAME_RE, long_name)
    isnothing(m) ? String(long_name) : String(m.captures[1])
end

_display_name(long_name::AbstractString) = "GD[ss=$(_short_label(long_name))]"

# Build the per-method MethodStyle dict keyed by the *display* name, using the
# Stages-1–4 _shared.jl COLORS palette so every figure across the stages
# assigns each method the same hue. The framework's default
# `get_method_color` hashes the method name, which would produce *different*
# colors here than in Stages 1–4. Bypassing it keeps the cross-stage color
# promise.
original_long_names = sort(unique(df.method_name))
styles = Dict(
    _display_name(long) => MethodStyle(
        color     = get(COLORS, _short_label(long), :gray),
        linewidth = 2.0,
    )
    for long in original_long_names
)

# Rename in place, after the byte-identity check is done.
df.method_name = _display_name.(df.method_name)

# ─── Validation prints ───────────────────────────────────────────────────────

println("\n=== Per-method final state ===")
for sub in groupby(df, :method_name)
    last_row = sub[end, :]
    # n_linesearch_evals is cumulative — last row already has the grand total.
    nls = :n_linesearch_evals in propertynames(sub) ?
            Int(coalesce(last(sub.n_linesearch_evals), 0)) : 0
    @printf("  %-18s iter=%5d  f=%.3e  ‖∇f‖=%.3e  ‖x-x*‖=%.3e  Σevals=%d\n",
            sub.method_name[1], last_row.iter, last_row.objective,
            last_row.gradient_norm, last_row.dist_to_opt, last_row.iter + nls)
end

# Sanity check on Armijo's line-search accounting: total evals should be
# meaningfully larger than the iter count. If they're ≈ equal,
# n_linesearch_evals is not being incremented inside the backtracking loop.
let armijo_name = "GD[ss=Armijo]"
    sub = @subset(df, :method_name .== armijo_name)
    if nrow(sub) > 0 && :n_linesearch_evals in propertynames(sub)
        n_iter  = nrow(sub)
        # Last row carries the cumulative line-search trial count.
        n_extra = Int(coalesce(last(sub.n_linesearch_evals), 0))
        ratio   = (n_iter + n_extra) / max(n_iter, 1)
        @printf("\n  Armijo eval/iter ratio = %.2f (≈1 ⇒ line-search accounting is broken)\n",
                ratio)
        ratio > 1.05 || @warn "Armijo's eval/iter ratio looks suspiciously low"
    else
        @warn ":n_linesearch_evals not present in DataFrame — Armijo " *
              "line-search accounting may not be logged."
    end
end

method_names = sort(unique(df.method_name))

# ─── Plotting ────────────────────────────────────────────────────────────────

"""
    four_panel(df, xcol, xlabel; xscale, title)

Build a 2×2 FigureLayout: f(x), ‖∇f(x)‖, ‖x − x*‖, αₖ — all log-y, vs an
arbitrary x column. αₖ comes from the `:step_size` extras column populated by
extract_log_entry(::GradientDescent, ...). For Fixed it is constant; for
Armijo it is discrete β^j values; for BB it varies continuously.
"""
function four_panel(df, xcol::Symbol, xlabel::String;
                    xscale::Symbol=:linear, title::String="")
    p_obj  = TestEngine.PlotSpec(data=df, x=xcol, y=:objective,     yscale=:log10, xscale=xscale,
                      title="f(x)",     xlabel=xlabel, ylabel="f(x)",
                      method_styles=styles)
    p_grad = TestEngine.PlotSpec(data=df, x=xcol, y=:gradient_norm, yscale=:log10, xscale=xscale,
                      title="‖∇f(x)‖",  xlabel=xlabel, ylabel="‖∇f(x)‖",
                      method_styles=styles)
    p_dist = TestEngine.PlotSpec(data=df, x=xcol, y=:dist_to_opt,   yscale=:log10, xscale=xscale,
                      title="‖x − x*‖", xlabel=xlabel, ylabel="‖x − x*‖",
                      method_styles=styles)
    p_step = TestEngine.PlotSpec(data=df, x=xcol, y=:step_size,     yscale=:log10, xscale=xscale,
                      title="αₖ",       xlabel=xlabel, ylabel="αₖ",
                      method_styles=styles)
    plots = Union{TestEngine.PlotSpec,Nothing}[p_obj p_grad ; p_dist p_step]
    return render_figure(FigureLayout(
        figure_size = (1400, 1000),
        title       = title,
        plots       = plots,
    ))
end

fig_iter = four_panel(df, :iter, "iteration";
                      title = "Stage 5 — convergence vs iter")
save_figure(fig_iter, joinpath(result.experiment_path, "stage5_vs_iter.pdf"))

# vs_evals and vs_coretime: x-axis on log10 so the ~10⁰…~10⁴ range
# (BB1 ~76 evals up to Fixed ~20 000) shows every method with comparable
# resolution. On a linear x-axis the entire BB1/BB2/Cauchy story collapses to
# the first few pixels and one outlier method dominates.
#
# Filter iter ≥ 1 so cum_n_evals > 0 and cum_core_time_ms > 0 — the iter=0
# init entry has both at zero, which would be log10(0) = -Inf and break the
# axis transform.
df_pos = @subset(df, :iter .>= 1)

fig_evals = four_panel(df_pos, :cum_n_evals, "cumulative function evaluations";
                       xscale = :log10,
                       title  = "Stage 5 — convergence vs cumulative function evaluations")
save_figure(fig_evals, joinpath(result.experiment_path, "stage5_vs_evals.pdf"))

fig_time = four_panel(df_pos, :cum_core_time_ms, "cumulative core time (ms)";
                      xscale = :log10,
                      title  = "Stage 5 — convergence vs cumulative core time " *
                              "[SANITY-CHECK ONLY; OS jitter dominates on 2D Rosenbrock]")
save_figure(fig_time, joinpath(result.experiment_path, "stage5_vs_coretime.pdf"))

println("\n=== Figures saved ===")
println("  ", joinpath(result.experiment_path, "stage5_vs_iter.pdf"))
println("  ", joinpath(result.experiment_path, "stage5_vs_evals.pdf"))
println("  ", joinpath(result.experiment_path, "stage5_vs_coretime.pdf"))
println("""

  Note on the core-time axis:
    The kernel is tens of nanoseconds per iter on 2D Rosenbrock, so OS jitter
    shifts cumulative time by 5–10%. This axis is a sanity check, NOT
    authoritative for ordering. Use the eval-count axis for fair comparison.
""")
