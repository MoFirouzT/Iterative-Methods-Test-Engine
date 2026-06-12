# experiments/exp_ls2_conditioning.jl
#
# Portfolio experiment: ls2 — conditioning sweep.
# Five GD variants on consistent linear least squares ½‖Ax−b‖², FIXED dimension
# n = 100, swept over the Hessian condition number κ = cond(AᵀA). Plot
# iters-to-tolerance vs κ on log-log. The validation IS the slope difference:
#
#   • Fixed / Armijo / Cauchy  → iters ~ O(κ),  slope ≈ 1
#   • BB1 / BB2                → markedly flatter, ~ O(√κ), slope ≈ 0.5
#
# κ range capped at 1e4: Fixed/Armijo are O(κ), so κ = 1e7 would need ~1e8
# iterations — infeasible. Three decades is plenty to read the slope.
#
# Run:  julia --project=. experiments/exp_ls2_conditioning.jl

include("_bootstrap.jl")
using Random
using DataFrames
using CairoMakie
using Printf
include("_shared.jl")

const SEED     = 42
const RUN_ID   = 1
const LS2_N    = 100
const LS2_KS   = [1.0e1, 1.0e2, 1.0e3, 1.0e4]
const LS2_TOL  = 1e-6
const LS2_CAP  = 300_000

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

function run_ls2(; seed::Int = SEED, n = LS2_N, kappas = LS2_KS, tol = LS2_TOL, cap = LS2_CAP)
    # JIT warm-up (smallest κ) so the first real run isn't penalized.
    let p = make_problem(RandomProblem(name = :linear_ls,
                                       params = (n = n, condition_number = kappas[1])), Xoshiro(seed))
        for (_, m) in build_ls_methods(p.meta[:L])
            run_method(m, p, MaxIterations(n = 20),
                       TestEngine.make_logger("warm", 0, "", VerbosityConfig(level = SILENT)), Xoshiro(0))
        end
    end

    rows = NamedTuple[]
    for κ in kappas
        p = make_problem(RandomProblem(name = :linear_ls,
                                       params = (n = n, condition_number = κ)),
                         Xoshiro(hash((seed, :data, κ))))
        crit = stop_when_any(MaxIterations(n = cap), DistanceToOptimal(tol = tol))
        for (name, method) in build_ls_methods(p.meta[:L])
            logger = TestEngine.make_logger(name, RUN_ID, "", VerbosityConfig(level = SILENT))
            res    = run_method(method, p, crit, logger, Xoshiro(hash((seed, name, κ))))
            push!(rows, (
                method_name = name, kappa = κ,
                iters = res.n_iters, converged = res.stop_reason == :optimal_reached,
            ))
            @info "[$name] κ=$(Int(κ))" iters=res.n_iters converged=(res.stop_reason==:optimal_reached)
        end
    end
    return DataFrame(rows)
end

# Least-squares slope of log10(iters) vs log10(κ).
function loglog_slope(κs, its)
    x = log10.(Float64.(κs)); y = log10.(Float64.(its))
    x̄ = sum(x)/length(x); ȳ = sum(y)/length(y)
    sum((x .- x̄) .* (y .- ȳ)) / sum((x .- x̄).^2)
end

# ---------------------------------------------------------------------------
# Figure: iters-to-tol vs κ (log-log), slopes annotated
# ---------------------------------------------------------------------------

function plot_ls2(df::DataFrame, slopes::Dict; outpath::String = "figures/ls2_conditioning.png")
    fig = Figure(size = (900, 640))
    ax  = Axis(fig[1, 1], xlabel = "condition number κ = cond(AᵀA)",
        ylabel = "iterations to ‖x−x*‖ ≤ 1e-6", xscale = log10, yscale = log10,
        title = "GD rate vs conditioning (n=$(LS2_N))")

    for name in PLOT_ORDER
        sub = sort(filter(:method_name => ==(name), df), :kappa)
        scatterlines!(ax, sub.kappa, sub.iters; color = COLORS[name], linewidth = 2.5,
            markersize = 11, label = @sprintf("%s  (slope %.2f)", name, slopes[name]))
    end
    axislegend(ax; position = :lt, framevisible = true)
    mkpath(dirname(outpath)); save(outpath, fig)
    @info "Saved figure" path = outpath
    return fig
end

# ---------------------------------------------------------------------------
# Win-condition checks
# ---------------------------------------------------------------------------

function validate_ls2(df::DataFrame)
    @assert all(df.converged) "not all runs converged: $(df[.!df.converged, [:method_name,:kappa]])"

    slopes = Dict{String,Float64}()
    @printf("\nlog-log slopes (iters ~ κ^slope):\n")
    for name in PLOT_ORDER
        sub = sort(filter(:method_name => ==(name), df), :kappa)
        slopes[name] = loglog_slope(sub.kappa, sub.iters)
        @printf("  %-7s slope=%.3f   iters=%s\n", name, slopes[name], string(sub.iters))
    end

    # O(κ) trio: slope ≈ 1.
    for name in ("Fixed", "Armijo", "Cauchy")
        @assert 0.85 <= slopes[name] <= 1.15 "$name slope $(slopes[name]) not ≈ 1 (expected O(κ))"
    end
    # BB markedly flatter than the O(κ) trio (the validation).
    for name in ("BB1", "BB2")
        @assert slopes[name] < 0.75 "$name slope $(slopes[name]) not markedly flatter than O(κ)"
        @assert slopes[name] < slopes["Fixed"] - 0.25 "$name not clearly flatter than Fixed"
    end
    return slopes
end

function main()
    df     = run_ls2()
    slopes = validate_ls2(df)
    plot_ls2(df, slopes)
    return df
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
