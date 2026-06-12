# experiments/exp_tr_steihaug_cg.jl
#
# Portfolio experiment: trust region — nested optimization.
# TrustRegion with a Steihaug truncated-CG inner solve on Rosenbrock, compared to
# the five Stage-1 GD baselines. This is the consumer that lights up the
# nested-optimization subsystem (run_sub_method / SubResult / attach_sub_logs!):
# each OUTER step solves a quadratic trust-region model with CG as a genuine
# sub-method on a genuine Problem.
#
#   (left)  ‖∇f(xₖ)‖ vs outer iteration, log-log — trust-region-Newton reaches
#           high accuracy in ~25 outer iters; the first-order baselines crawl.
#   (right) the inner solver at work: trust radius Δ per outer step, marker
#           colored by the inner CG stop reason and sized by inner CG iterations.
#
# Run:  julia --project=. experiments/exp_tr_steihaug_cg.jl

include("_bootstrap.jl")
using Random
using DataFrames
using CairoMakie
using Printf
using LinearAlgebra: norm
include("_shared.jl")

const SEED = 42
const TR_COLOR = "#9467bd"   # purple — distinct from the GD palette
# Inner-stop → marker colour for the nesting panel.
const INNER_COLOR = Dict(:gradient_converged => "#0072B2", :boundary_reached => "#E69F00",
                         :negative_curvature => "#D55E00", :max_iterations => "#999999")

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

function run_tr(; seed::Int = SEED)
    p = make_problem(AnalyticProblem(name = :rosenbrock, params = (rho = 100.0,)),
                     Xoshiro(hash((seed, :data))))

    # Trust-region (the headliner).
    tr_logger = make_logger("TrustRegion", 1, "", VerbosityConfig(level = MILESTONE))
    tr = run_method(TrustRegion(Δ0 = 1.0), p,
                    stop_when_any(MaxIterations(n = 300), GradientTolerance(tol = 1e-8)),
                    tr_logger, Xoshiro(hash((seed, :tr))))
    @info "[TrustRegion] done" outer_iters=tr.n_iters stop=tr.stop_reason f=tr.final_state.metrics.objective

    # Five first-order baselines (Stage-1 methods), same problem.
    baselines = Pair{String,Any}[]
    for (name, method) in build_standard_methods()
        lg = make_logger(name, 1, "", VerbosityConfig(level = SILENT))
        res = run_method(method, p, stop_when_any(MaxIterations(n = 5000), GradientTolerance(tol = 1e-8)),
                         lg, Xoshiro(hash((seed, name))))
        push!(baselines, name => res)
    end

    # Secondary: an indefinite-Hessian start, to exercise the negative-curvature branch.
    p_indef = make_problem(AnalyticProblem(name = :rosenbrock, params = (rho = 100.0, x0 = [0.0, 2.0])),
                           Xoshiro(hash((seed, :indef))))
    tr_indef = run_method(TrustRegion(Δ0 = 1.0), p_indef,
                          stop_when_any(MaxIterations(n = 300), GradientTolerance(tol = 1e-8)),
                          make_logger("TR-indef", 1, "", VerbosityConfig(level = SILENT)),
                          Xoshiro(hash((seed, :tri))))

    return p, tr, tr_logger, baselines, tr_indef
end

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

function plot_tr(tr, baselines, tr_logger; outpath::String = "figures/tr_steihaug_cg.png")
    fig = Figure(size = (1280, 540))

    # ── Left: convergence vs outer iteration ──────────────────────────────────
    axL = Axis(fig[1, 1], xlabel = "iteration", ylabel = "‖∇f(xₖ)‖",
        xscale = log10, yscale = log10,
        title = "Convergence on Rosenbrock — trust-region-Newton vs first-order")
    for (name, res) in baselines
        reals = [e for e in res.iter_logs if e.iter > 0]
        lines!(axL, [e.iter for e in reals], [max(e.gradient_norm, 1e-12) for e in reals];
            color = (COLORS[name], 0.9), linewidth = 1.8, label = name)
    end
    trreals = [e for e in tr.iter_logs if e.iter > 0]
    scatterlines!(axL, [e.iter for e in trreals], [max(e.gradient_norm, 1e-12) for e in trreals];
        color = TR_COLOR, linewidth = 3, markersize = 7, label = "TrustRegion")
    axislegend(axL; position = :lb, framevisible = true, nbanks = 2, labelsize = 10)

    # ── Right: the nested inner solver at work ────────────────────────────────
    axR = Axis(fig[1, 2], xlabel = "outer iteration", ylabel = "trust radius Δ",
        yscale = log10, title = "Nested Steihaug-CG: radius Δ, stop reason, inner work")
    iters = [e.iter for e in trreals]
    radii = [e.extras[:radius] for e in trreals]
    lines!(axR, iters, radii; color = (:gray, 0.6), linewidth = 1.5)
    for stop in unique(get(e.extras, :inner_stop, :none) for e in trreals)
        idx = [i for (i, e) in enumerate(trreals) if get(e.extras, :inner_stop, :none) == stop]
        scatter!(axR, iters[idx], radii[idx];
            color = get(INNER_COLOR, stop, "#000000"),
            markersize = [6 + 3 * get(trreals[i].extras, :n_inner, 1) for i in idx],
            label = String(stop), strokecolor = :white, strokewidth = 0.5)
    end
    axislegend(axR, "inner CG stop (size ∝ #CG iters)"; position = :rt, framevisible = true, labelsize = 10)

    Label(fig[0, :], "TrustRegion + Steihaug-CG inner solve", fontsize = 16, font = :bold)
    mkpath(dirname(outpath)); save(outpath, fig)
    @info "Saved figure" path = outpath
    return fig
end

# ---------------------------------------------------------------------------
# Win-condition checks
# ---------------------------------------------------------------------------

function validate_tr(tr, tr_logger, tr_indef)
    trreals = [e for e in tr.iter_logs if e.iter > 0]

    # 1. Trust-region-Newton converges to high accuracy in O(20–30) outer iters.
    @assert tr.stop_reason == :gradient_converged "TR did not converge: $(tr.stop_reason)"
    @assert tr.n_iters <= 35 "TR took $(tr.n_iters) outer iters (expected O(20–30))"

    # 2. attach_sub_logs! populated; SubResult core time > 0; per-step inner trace present.
    @assert !isempty(tr_logger.pending_sub_logs) "pending_sub_logs empty — nesting not exercised"
    @assert any(e -> e.extras[:inner_core_ns] > 0, trreals) "no inner core time recorded"
    @assert all(haskey(e.extras, :sub_logs) for e in trreals) "inner CG trace missing from an outer step"

    # 3. Core-time attribution: inner core folded into outer core_time (outer ≥ inner > 0).
    e1 = trreals[1]
    @assert e1.core_time_ns >= e1.extras[:inner_core_ns] > 0 "core-time attribution wrong"

    # 4. The boundary branch fires (early Newton steps exceed Δ).
    n_boundary = count(e -> get(e.extras, :inner_stop, :none) == :boundary_reached, trreals)
    @assert n_boundary >= 1 "boundary branch never fired"

    # 5. The negative-curvature branch fires from the indefinite start.
    ireals = [e for e in tr_indef.iter_logs if e.iter > 0]
    n_negcurv = count(e -> get(e.extras, :inner_stop, :none) == :negative_curvature, ireals)
    @assert n_negcurv >= 1 "negative-curvature branch never fired (indefinite start)"

    total_inner = sum(get(e.extras, :n_inner, 0) for e in trreals)
    @info "Win conditions" outer_iters=tr.n_iters total_inner_CG=total_inner boundary_hits=n_boundary negcurv_hits_indef=n_negcurv
    @printf("\nTrustRegion: %d outer iters, %d total inner CG iters; boundary fired %d×, negative-curvature %d× (indefinite start).\n",
            tr.n_iters, total_inner, n_boundary, n_negcurv)
    return nothing
end

function main()
    p, tr, tr_logger, baselines, tr_indef = run_tr()
    validate_tr(tr, tr_logger, tr_indef)
    plot_tr(tr, baselines, tr_logger)
    return tr
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
