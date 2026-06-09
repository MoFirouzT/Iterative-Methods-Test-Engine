# experiments/exp_lasso1_ista_fista.jl
#
# Stage LASSO-1 — the flagship (portfolio Item 2).
# ISTA vs FISTA (ProximalGradient with NoMinorUpdate vs NesterovStep) on the
# sparse-recovery lasso  min_x ½‖Ax−b‖² + λ‖x‖₁, at the textbook step γ = 1/L.
#
# Two-panel money figure:
#   (left)  f − f*  vs iteration on log-y — FISTA's O(1/k²) visibly beats
#           ISTA's O(1/k).
#   (right) recovered x̂ (FISTA) vs the planted sparse signal x_star.
#
# Naming: there are two experiment tracks. `exp_stageN` is the GD-on-Rosenbrock
# narrative (see basic_experiments.md); `exp_<problem>N` (this file) is the
# portfolio-item track (see portfolio_roadmap.md). They coexist deliberately.
#
# To run, from project root:
#     julia --project=. experiments/exp_lasso1_ista_fista.jl

include("_bootstrap.jl")   # engine + all content (problems, methods, components)
using Random
using DataFrames
using CairoMakie
using LinearAlgebra: norm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

const SEED   = 42
const RUN_ID = 1

# Wong palette, consistent with the Stage figures.
const C_ISTA  = "#E69F00"   # orange
const C_FISTA = "#0072B2"   # blue
const PG_COLORS = Dict("ISTA" => C_ISTA, "FISTA" => C_FISTA)
const PG_ORDER  = ["ISTA", "FISTA"]

# Instance chosen so the O(1/k) vs O(1/k²) gap is visible before both curves
# reach machine precision (see the probe in the Item-2 write-up).
const LASSO_PARAMS = (m = 200, n = 512, k = 15, λ = 0.05)
const K_PLOT = 200       # iterations shown / run for the comparison
const K_REF  = 20_000    # long FISTA reference run to estimate f*

build_lasso_methods(L) = [
    "ISTA"  => ProximalGradient(step_size = FixedStep(α = 1/L), minor_update = NoMinorUpdate()),
    "FISTA" => ProximalGradient(step_size = FixedStep(α = 1/L), minor_update = NesterovStep()),
]

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

function run_lasso1(; seed::Int = SEED, run_id::Int = RUN_ID,
                      params = LASSO_PARAMS, K::Int = K_PLOT, K_ref::Int = K_REF)
    rng_data = Xoshiro(hash((seed, run_id, :data)))
    problem  = make_problem(RandomProblem(name = :lasso, params = params), rng_data)
    L = problem.meta[:L]

    @info "Stage LASSO-1 — running" m=params.m n=params.n k=params.k λ=params.λ L=L

    results = Pair{String, Any}[]
    for (name, method) in build_lasso_methods(L)
        rng    = Xoshiro(hash((seed, run_id, name)))
        logger = make_logger(name, run_id, "", VerbosityConfig(level = MILESTONE))
        res    = run_method(method, problem, MaxIterations(n = K), logger, rng)
        push!(results, name => res)
        @info("[$name] done", iters = res.n_iters,
              f_final = res.iter_logs[end].objective,
              step_norm = res.iter_logs[end].step_norm)
    end

    # f* from a long FISTA reference run (NOT f(x_star) — see lasso.md).
    fref    = ProximalGradient(step_size = FixedStep(α = 1/L), minor_update = NesterovStep())
    lg_ref  = make_logger("ref", run_id, "", VerbosityConfig(level = SILENT))
    res_ref = run_method(fref, problem, MaxIterations(n = K_ref), lg_ref,
                         Xoshiro(hash((seed, run_id, :ref))))
    fstar = minimum(e.objective for e in res_ref.iter_logs)

    return problem, results, fstar
end

# ---------------------------------------------------------------------------
# DataFrame adapter
# ---------------------------------------------------------------------------

function results_to_df(results::Vector, fstar::Float64; run_id::Int = RUN_ID)
    rows = NamedTuple[]
    for (name, res) in results
        for entry in res.iter_logs
            push!(rows, (
                run_id      = run_id,
                method_name = name,
                iter        = entry.iter,
                objective   = entry.objective,
                gap         = entry.objective - fstar,
                step_norm   = entry.step_norm,
            ))
        end
    end
    return DataFrame(rows)
end

# ---------------------------------------------------------------------------
# Money figure
# ---------------------------------------------------------------------------

function plot_lasso1(df::DataFrame, problem, results;
                     outpath::String = "figures/lasso_ista_fista.png")
    fig = Figure(size = (1280, 540))

    # ── Left: f − f* vs iteration (log-y) ──────────────────────────────────
    axL = Axis(fig[1, 1],
        xlabel = "iteration  k",
        ylabel = "f(xₖ) − f*",
        yscale = log10,
        title  = "Convergence: FISTA O(1/k²) vs ISTA O(1/k)",
    )
    for name in PG_ORDER
        sub = sort(filter(:method_name => ==(name), df), :iter)
        ys  = max.(sub.gap, 1e-15)   # floor for log scale once converged
        lines!(axL, sub.iter, ys; color = PG_COLORS[name], linewidth = 2.5, label = name)
    end
    axislegend(axL; position = :rt, framevisible = true)

    # ── Right: recovered support vs planted signal ─────────────────────────
    axR = Axis(fig[1, 2],
        xlabel = "coordinate index",
        ylabel = "value",
        title  = "Support recovery (FISTA)",
    )
    x_hat  = first(r for (nm, r) in results if nm == "FISTA").final_state.iterate.x
    x_star = problem.x_opt
    idx    = 1:length(x_hat)
    stem!(axR, idx, x_hat;
        color = C_FISTA, markersize = 5, stemcolor = (C_FISTA, 0.5),
        label = "recovered x̂")
    scatter!(axR, idx, x_star;
        color = (:red, 0.0), strokecolor = :red, strokewidth = 1.5,
        marker = :circle, markersize = 11, label = "planted x⋆")
    axislegend(axR; position = :rt, framevisible = true)

    Label(fig[0, :],
        "Lasso (m=$(LASSO_PARAMS.m), n=$(LASSO_PARAMS.n), k=$(LASSO_PARAMS.k) nonzeros, " *
        "λ=$(LASSO_PARAMS.λ); benign Gaussian A) — ISTA vs FISTA at γ = 1/L",
        fontsize = 16, font = :bold)

    mkpath(dirname(outpath))
    save(outpath, fig)
    @info "Saved money figure" path = outpath
    return fig
end

# ---------------------------------------------------------------------------
# Win-condition checks (printed; structural, not a unit test)
# ---------------------------------------------------------------------------

function validate(df::DataFrame, problem, results, fstar)
    # 1. total_objective decomposes as f + g on the final iterate.
    x_hat = first(r for (nm, r) in results if nm == "FISTA").final_state.iterate.x
    g_reg = problem.gs[1]
    decomp = value(problem.f, x_hat) + value(g_reg, x_hat)
    @assert isapprox(total_objective(problem, x_hat), decomp; rtol = 1e-12)

    # 2. FISTA is accelerated: strictly smaller gap than ISTA at a mid iteration.
    mid = 50
    gap_at(name) = let s = filter(r -> r.method_name == name && r.iter == mid, df)
        isempty(s) ? Inf : s.gap[1]
    end
    gi, gf = gap_at("ISTA"), gap_at("FISTA")
    @assert gf < gi "FISTA gap ($gf) not below ISTA gap ($gi) at iter $mid"

    # 3. Support recovery: the planted support is recovered.
    true_supp = problem.meta[:support]
    rec  = findall(>(0.1), abs.(x_hat))     # clear spikes
    spur = setdiff(rec, true_supp)
    recovered_ok = issubset(true_supp, rec)

    @info "Win conditions" objective_decomposes=true fista_gap=gf ista_gap=gi acceleration=(gf<gi)
    @info "Support recovery" true_support=sort(true_supp) recovered=sort(rec) spurious=sort(spur) all_planted_recovered=recovered_ok
    return nothing
end

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

function main()
    problem, results, fstar = run_lasso1()
    df = results_to_df(results, fstar)
    validate(df, problem, results, fstar)
    plot_lasso1(df, problem, results)
    return df
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
