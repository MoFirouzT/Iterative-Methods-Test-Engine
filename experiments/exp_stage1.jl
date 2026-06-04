# experiments/exp_stage1.jl
#
# Stage 1 of the test plan — first comparison figure.
# Five GradientDescent variants on Rosenbrock (ρ=100, x₀=(-1.2, 1)),
# stopping on max-iter OR gradient tolerance.
#
# To run, from project root:
#     julia --project=. experiments/exp_stage1.jl

include("../src/TestEngine.jl")
using .TestEngine          # the framework — adjust the module name if yours differs
using Random
using DataFrames
using CairoMakie

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# RNG derivation here mirrors what run_experiment will use at Stage 6/8, so
# results stay byte-comparable across stages for any randomness-free method.
const SEED   = 42
const RUN_ID = 1

# Short labels for Stage 1. Stage 6's variant grid will produce full names like
# "GradientDescent[step_size=Fixed]"; that switch is intentional and isolated.
const PLOT_ORDER = ["Fixed", "Armijo", "Cauchy", "BB1", "BB2"]

# Wong colorblind-safe palette (consistent with METHOD_PALETTE in analysis.jl).
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
# Driver — runs each method, returns Vector{Pair{String, MethodResult}}
# ---------------------------------------------------------------------------

function run_stage1(; seed::Int = SEED, run_id::Int = RUN_ID)
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

    @info "Stage 1 — running" problem = "Rosenbrock(ρ=100)" x0 = problem.x0 x_opt = problem.x_opt
    results = Pair{String, Any}[]   # MethodResult{S} has varying S → Any here is fine
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
              grad_norm   = last_entry.gradient_norm,
              dist_to_opt = last_entry.dist_to_opt)
    end
    return results
end

# ---------------------------------------------------------------------------
# DataFrame adapter — schema chosen to match what to_dataframe(::ExperimentResult)
# will produce at Stage 3, so plot code is reusable.
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
            ))
        end
    end
    return DataFrame(rows)
end

# ---------------------------------------------------------------------------
# Plotting — 2×2 panel with shared legend on the right
# ---------------------------------------------------------------------------

function plot_stage1(df::DataFrame; outpath::String = "stage1_convergence.pdf")
    fig = Figure(size = (1400, 850))

    # (row, col, df-column, panel title, y-axis label)
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
            # The init entry has α_k = 0.0 by default — invalid on log scale.
            # Drop iter 0 from the α panel only; keep it on the other three.
            ycol === :step_size && (sub = filter(:iter => >(0), sub))
            ys = max.(sub[!, ycol], 1e-16)   # floor avoids log(0) for converged runs
            lines!(ax, sub.iter, ys;
                color     = COLORS[name],
                linewidth = 2.0,
            )
        end
    end

    # Single legend for the whole figure, in a dedicated 3rd column.
    legend_elements = [LineElement(color = COLORS[name], linewidth = 2.5)
                       for name in PLOT_ORDER]
    Legend(fig[1:2, 3], legend_elements, PLOT_ORDER, "step size";
           framevisible = true, tellwidth = true)

    Label(fig[0, :],
          "Stage 1 — GradientDescent on Rosenbrock (ρ=100, x₀=(−1.2, 1))",
          fontsize = 16, font = :bold)

    save(outpath, fig)
    @info "Saved figure" path = outpath
    return fig
end

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

function main()
    results = run_stage1()
    df      = results_to_df(results)
    plot_stage1(df; outpath = "stage1_convergence.pdf")
    return df
end

# Run only when executed as a script, not when included for interactive use.
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
