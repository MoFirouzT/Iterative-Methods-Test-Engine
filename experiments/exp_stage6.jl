# experiments/exp_stage6.jl
# =============================================================================
# Stage 6 — Multi-run with randomized x₀ + warm-up
# -----------------------------------------------------------------------------
# The same five GradientDescent step-size variants on Rosenbrock, but now x₀
# is sampled uniformly in [−2, 2]² and the experiment runs n_runs = 20 times.
# Curves are summarised as a median trace with a shaded 25–75% IQR band. One
# of the two experiments wraps everything in an IterativeWarmup that runs 50
# iterations of FixedStep gradient descent first; the warm-started x₀ is then
# shared across all five methods within each run.
#
# Exercises
#   * RandomProblem + register_random_problem! (problem-side rng).
#   * run_experiment's per-run data rng: Xoshiro(hash((seed, run_id, :data))).
#   * aggregate_runs(df, :median) and :all modes.
#   * NoWarmup vs IterativeWarmup dispatch in run_warmup.
#   * The universal `result.final_state.iterate.x` convention used by
#     IterativeWarmup to read off the warm-started x₀.
#
# Validates
#   * Same seed → byte-identical DataFrame on a second invocation
#     (run twice → diff).
#   * Different seed → distribution that visibly tightens for stable methods
#     (Armijo) and widens for sensitive ones (BB on starts that miss the
#     valley).
#   * Warm-up x₀ is SHARED across all methods within a run_id. Concrete
#     invariant: `extras[:x_iter]` at `iter == 0` is identical across all
#     five methods for any single run_id when warm-up is active. This is the
#     only test that actually proves the warm-up output is shared rather than
#     each method warming itself up.
# =============================================================================

import Pkg
Pkg.activate(dirname(@__DIR__))

using TestEngine
using DataFrames, DataFramesMeta
using CairoMakie
using Statistics
using LinearAlgebra
using Printf
using Random

# ─── Setup ───────────────────────────────────────────────────────────────────

register_abbreviation!("GradientDescent", "GD")
register_abbreviation!("FixedStep",       "Fix")
register_abbreviation!("ArmijoLS",        "Arm")
register_abbreviation!("CauchyStep",      "Cau")
register_abbreviation!("BarzilaiBorwein", "BB")

# ─── Problem registration ────────────────────────────────────────────────────
#
# A RandomProblem variant of Rosenbrock that resamples x₀ ~ U([-2, 2]²) on
# every run. The signature follows architecture.md Module 1:
#   register_random_problem!(name, (rng, params) -> Problem)
# Note the (rng, params) order — random generators get rng first; analytic
# problems are (params, rng).
#
# The minimiser is at x* = (1, 1) for any ρ > 0 — no random component to
# x_opt, so it can be set unconditionally.

register_random_problem!(:rosenbrock_random_x0, (rng, params) -> begin
    ρ  = Float64(get(params, :ρ, 100.0))
    lo = Float64(get(params, :lo, -2.0))
    hi = Float64(get(params, :hi,  2.0))
    x0 = lo .+ (hi - lo) .* rand(rng, 2)
    f  = RosenbrockObjective(ρ)
    return Problem(f, x0;
                   x_opt = [1.0, 1.0],
                   meta  = Dict{Symbol,Any}(:ρ => ρ, :sampled_x0 => true))
end)

# ─── Grid ────────────────────────────────────────────────────────────────────

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
)

# ─── Common stopping criteria ────────────────────────────────────────────────
# We use a generous max_iters because Rosenbrock from x₁ < 0 wanders for a
# long time before finding the valley — BB methods particularly so.

criteria = stop_when_any(
    MaxIterations(n=5000),
    GradientTolerance(tol=1e-8),
    DistanceToOptimal(tol=1e-8),
)

# ─── Experiments — no warm-up vs IterativeWarmup ─────────────────────────────

problem_spec = RandomProblem(name=:rosenbrock_random_x0, params=(ρ=100.0,))

config_cold = ExperimentConfig(
    name              = "Stage 6 — random x₀, cold start",
    problem_spec      = problem_spec,
    variant_grids     = [grid],
    stopping_criteria = criteria,
    warmup            = NoWarmup(),
    n_runs            = 20,
    seed              = 42,
    tags              = Dict{String,Any}("stage" => "6", "warmup" => "none"),
)

config_warm = ExperimentConfig(
    name              = "Stage 6 — random x₀, FixedStep warm-up",
    problem_spec      = problem_spec,
    variant_grids     = [grid],
    stopping_criteria = criteria,
    warmup            = IterativeWarmup(
                            method   = GradientDescent(direction=SteepestDescent(),
                                                       step_size=FixedStep(α=1e-3)),
                            criteria = MaxIterations(n=50),
                        ),
    n_runs            = 20,
    seed              = 42,
    tags              = Dict{String,Any}("stage" => "6", "warmup" => "iterative_50"),
)

verbosity = VerbosityConfig(level=MILESTONE)

println("\n=== Stage 6 — running cold-start experiment ===")
result_cold = run_experiment(config_cold; verbosity=verbosity)
println("    saved to: ", result_cold.experiment_path)

println("\n=== Stage 6 — running warm-up experiment ===")
result_warm = run_experiment(config_warm; verbosity=verbosity)
println("    saved to: ", result_warm.experiment_path)

# ─── Validation 1 — same-seed determinism ───────────────────────────────────
#
# Re-run config_cold once more and assert that the resulting DataFrame is
# byte-identical to result_cold's. This is the load-bearing reproducibility
# check: per-run rng = Xoshiro(hash((seed, run_id, :data))) for the problem
# stream, plus Xoshiro(hash((seed, run_id, name))) for each method's stream.

println("\n=== Validation 1 — same-seed determinism ===")
let
    result_cold2 = run_experiment(config_cold;
                                  verbosity=VerbosityConfig(level=SILENT))
    df1 = to_dataframe(result_cold)
    df2 = to_dataframe(result_cold2)

    sort!(df1, [:method_name, :run_id, :iter])
    sort!(df2, [:method_name, :run_id, :iter])

    @assert nrow(df1) == nrow(df2) "row count differs across same-seed runs"
    for col in (:iter, :objective, :gradient_norm, :step_norm, :dist_to_opt)
        col in propertynames(df1) || continue
        isequal(df1[!, col], df2[!, col]) ||
            error("Column $col differs across same-seed runs — seed propagation broken")
    end
    println("✓ same-seed determinism passes on ",
            length(unique(df1.method_name)), " methods × ",
            length(unique(df1.run_id)),     " runs.")
end

# ─── Validation 2 — warm-up x₀ is shared across methods within a run ────────
#
# extras[:x_iter] at iter == 0 (or the smallest iter present, which is the
# initial point captured by log_init!) must be identical across all five
# methods for any given run_id when warmup is active. If methods see
# different starting points, the warm-up has been re-run per-method instead
# of being shared at the run level.

println("\n=== Validation 2 — warm-up x₀ shared across methods ===")
let
    df = to_dataframe(result_warm)
    @assert :x_iter in propertynames(df)  "extras[:x_iter] missing — needed for this check"

    failures = 0
    for run_sub in groupby(df, :run_id)
        # For this run_id, gather (method_name → captured x₀) by taking the
        # earliest log entry per method. log_init! is expected to create an
        # iter=0 entry holding the starting point; if it doesn't, argmin
        # picks the next earliest available, in which case the check
        # compares post-first-step states and would fail spuriously — that
        # would itself be a useful signal.
        per_method = Dict{String, Vector{Float64}}()
        for method_sub in groupby(run_sub, :method_name)
            j = argmin(method_sub.iter)
            per_method[method_sub.method_name[1]] = method_sub.x_iter[j]
        end
        xs  = collect(values(per_method))
        ref = xs[1]
        if !all(x -> isequal(x, ref), xs)
            failures += 1
            @warn "run_id=$(run_sub.run_id[1]): not all methods saw the same x₀" per_method
        end
    end
    failures == 0 || error("$failures runs had inconsistent warm-up x₀ across methods")
    println("✓ all $(length(unique(df.run_id))) runs see a single warm-up x₀ ",
            "shared across the 5 methods.")
end

# Quick sanity that warmup actually moved x₀ vs the cold spec.
let
    df_w = to_dataframe(result_warm)
    df_c = to_dataframe(result_cold)
    @assert :x_iter in propertynames(df_w) && :x_iter in propertynames(df_c)

    fixed = "GradientDescent[step_size=Fixed]"
    w_rows = @subset(df_w, :run_id .== 1, :method_name .== fixed)
    c_rows = @subset(df_c, :run_id .== 1, :method_name .== fixed)
    if nrow(w_rows) > 0 && nrow(c_rows) > 0
        x0_w = w_rows.x_iter[argmin(w_rows.iter)]
        x0_c = c_rows.x_iter[argmin(c_rows.iter)]
        diff = norm(x0_w .- x0_c)
        @printf("  warm-up displacement on run_id=1: ‖x0_warm − x0_cold‖ = %.4f\n", diff)
        diff > 1e-6 || @warn "warm-up did not visibly move x₀ — is the warm-up actually running?"
    else
        @warn "could not locate run_id=1 entry for $fixed in one of the experiments"
    end
end

# ─── Plotting — median curves with IQR shading ───────────────────────────────
#
# aggregate_runs gives medians (`:median`) and the raw frame (`:all`). For
# IQR we need quantiles, which the framework's aggregate_runs does not
# expose. We compute q25/q50/q75 directly via DataFramesMeta — falling under
# the architecture's "user transforms are plain DataFrame -> DataFrame
# functions" pattern.
#
# We render via custom Makie code (precedent: Stage 2's trajectory plot)
# rather than PlotSpec, because the band fill requires a primitive that
# PlotSpec does not expose.

const _SAFE_MIN_OBJ  = 1e-16
const _SAFE_MIN_GRAD = 1e-16
const _SAFE_MIN_DIST = 1e-16

"""
    quantile_summary(df, ycol; safe_floor)

For each (:method_name, :iter), compute q25, q50, q75 of `ycol`. Values are
floored at `safe_floor` so log-y plots stay finite. Returns a long DataFrame.
"""
function quantile_summary(df::AbstractDataFrame, ycol::Symbol;
                          safe_floor::Float64 = 0.0)
    return combine(groupby(df, [:method_name, :iter])) do g
        v = collect(skipmissing(g[!, ycol]))
        isempty(v) && return DataFrame()
        v = safe_floor > 0 ? max.(v, safe_floor) : v
        DataFrame(q25 = quantile(v, 0.25),
                  q50 = quantile(v, 0.50),
                  q75 = quantile(v, 0.75))
    end
end

method_names = sort(unique(to_dataframe(result_cold).method_name))
styles = Dict(name => MethodStyle(color=get_method_color(name), linewidth=2.0)
              for name in method_names)

function _axis_with_iqr!(ax, df_quantiles, label_for; alpha::Float64 = 0.18)
    for sub in groupby(df_quantiles, :method_name)
        name  = sub.method_name[1]
        col   = get(styles, name, MethodStyle(color="#888888")).color
        sort!(sub, :iter)
        # band! requires sorted x; use the q25/q75 as lower/upper.
        band!(ax, sub.iter, sub.q25, sub.q75; color=(col, alpha), label=nothing)
        lines!(ax, sub.iter, sub.q50; color=col, linewidth=2.0, label=label_for(name))
    end
end

function _label_short(name::String)
    # Match the short-name convention from variants.jl. We pull anything
    # between the first '=' and the closing ']' as the value label, falling
    # back to the full name for atypical strings.
    m = match(r"\[step_size=([^\]]+)\]", name)
    m === nothing ? name : "GD/" * m.captures[1]
end

function plot_iqr_panel(df_raw::DataFrame; title::String, savepath::String)
    q_obj  = quantile_summary(df_raw, :objective;     safe_floor=_SAFE_MIN_OBJ)
    q_grad = quantile_summary(df_raw, :gradient_norm; safe_floor=_SAFE_MIN_GRAD)
    q_dist = quantile_summary(df_raw, :dist_to_opt;   safe_floor=_SAFE_MIN_DIST)

    fig = Figure(size = (1400, 800))
    Label(fig[0, :], title, fontsize=18)

    ax1 = Axis(fig[1, 1]; title="f(x)",     xlabel="iter", ylabel="f(x)",
               yscale=log10)
    ax2 = Axis(fig[1, 2]; title="‖∇f(x)‖",  xlabel="iter", ylabel="‖∇f(x)‖",
               yscale=log10)
    ax3 = Axis(fig[1, 3]; title="‖x − x*‖", xlabel="iter", ylabel="‖x − x*‖",
               yscale=log10)

    _axis_with_iqr!(ax1, q_obj,  _label_short)
    _axis_with_iqr!(ax2, q_grad, _label_short)
    _axis_with_iqr!(ax3, q_dist, _label_short)

    # Single legend for the figure, attached to the rightmost axis.
    Legend(fig[1, 4], ax1, "method"; tellwidth=true)

    save(savepath, fig)
    return fig
end

# Step-size panel intentionally dropped from the multi-run figure: median of
# Armijo's discrete β^j values is nonsense; median of Fixed's constant is a
# constant; aggregating BB's values smooths out the very non-monotonicity
# that makes BB BB. See basic_experiments.md Stage 6 caveats.

println("\n=== Plotting median + IQR ===")
plot_iqr_panel(to_dataframe(result_cold);
               title    = "Stage 6 — cold start (n_runs = 20), median + 25–75% IQR",
               savepath = joinpath(result_cold.experiment_path,
                                   "stage6_cold_iqr.pdf"))

plot_iqr_panel(to_dataframe(result_warm);
               title    = "Stage 6 — IterativeWarmup (50 FixedStep iters), median + 25–75% IQR",
               savepath = joinpath(result_warm.experiment_path,
                                   "stage6_warmup_iqr.pdf"))

# Per-run overlay for the cold experiment — shows individual trajectory
# variation that the IQR band collapses. Mirrors aggregate_runs(:all). One
# axis per method to keep it readable.
println("=== Plotting per-run overlay ===")
let df = to_dataframe(result_cold)
    fig = Figure(size = (1400, 1000))
    Label(fig[0, :], "Stage 6 — cold start, ‖x − x*‖ per run (n_runs = 20)",
          fontsize = 18)
    for (k, name) in enumerate(method_names)
        row, col = fldmod1(k, 3)
        ax = Axis(fig[row, col]; title = _label_short(name), xlabel = "iter",
                  ylabel = "‖x − x*‖", yscale = log10)
        sub = @subset(df, :method_name .== name)
        c   = get(styles, name, MethodStyle(color="#888888")).color
        for run_sub in groupby(sub, :run_id)
            d = max.(run_sub.dist_to_opt, _SAFE_MIN_DIST)
            lines!(ax, run_sub.iter, d; color=(c, 0.35), linewidth=1.0)
        end
    end
    save(joinpath(result_cold.experiment_path, "stage6_cold_per_run.pdf"), fig)
end

println("\n=== Figures saved ===")
println("  ", joinpath(result_cold.experiment_path, "stage6_cold_iqr.pdf"))
println("  ", joinpath(result_warm.experiment_path, "stage6_warmup_iqr.pdf"))
println("  ", joinpath(result_cold.experiment_path, "stage6_cold_per_run.pdf"))
println("""

  Notes:
    * Rosenbrock from x₁ < 0 wanders for a long time before finding the
      valley; the IQR is wide for BB methods on those starts. This is
      correct, not a bug.
    * The step-size panel is omitted from these figures: medianing Armijo's
      discrete β^j values is meaningless; medianing Fixed's constant is
      pointless; medianing BB smooths out the very non-monotonicity that
      characterises BB. Inspect step sizes per-run via Stage 5 instead.
""")
