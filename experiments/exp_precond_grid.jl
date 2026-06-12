# experiments/exp_precond_grid.jl
#
# Portfolio experiment: preconditioning — the signature workflow.
# Define ONE experimental method (PreconditionedGradient), sweep its variants in
# a VariantGrid, and compare against a conventional baseline — all in a single
# `run_experiment`. This is the first exercise of:
#   • the experimental method bucket + dual routing in resolve_methods,
#   • a genuine ≥2-axis VariantGrid Cartesian product,
#   • DiagonalHessian (via the separable quadratic) and the Jacobi preconditioner.
#
# Grid: preconditioner ∈ {Identity, Jacobi} × step_size ∈ {Fixed, Armijo, Cauchy}
# (2×3 = 6 experimental variants) + a conventional GradientDescent baseline.
#
# step-size axis note: Fixed/Armijo/Cauchy all compose with an arbitrary descent
# direction (Cauchy = exact line search ALONG d). BarzilaiBorwein is deliberately
# excluded — its secant step `sᵀs/sᵀy` is derived for d = −g and is ill-posed (it
# diverges) on a preconditioned direction.
#
# On a diagonal quadratic the Jacobi preconditioner IS Newton: every Jacobi
# variant converges in ~1 iteration regardless of κ, while every Identity variant
# (and the baseline) crawls at O(κ). That ~5-orders-of-magnitude gap is the win.
#
# Run:  julia --project=. experiments/exp_precond_grid.jl

include("_bootstrap.jl")
using Random
using DataFrames
using CairoMakie
using Printf
include("_shared.jl")

const SEED   = 42
const PRECOND_KAPPA = 1.0e4
const PRECOND_N     = 50
const PRECOND_CAP   = 300_000

# Friendly short-names for the generated VariantSpec labels.
register_abbreviation!("PreconditionedGradient", "PCG")
register_abbreviation!("Identity", "Id")
register_abbreviation!("Jacobi",   "Jac")
register_abbreviation!("Fixed",    "Fix")
register_abbreviation!("Cauchy",   "Cau")   # "Armijo" ↦ "Arm" already registered

# ---------------------------------------------------------------------------
# Config: the VariantGrid + a conventional baseline
# ---------------------------------------------------------------------------

precond_grid() = VariantGrid(
    base_name = "PreconditionedGradient",
    axes = [
        VariantAxis(:preconditioner,
            IdentityPreconditioner() => "Identity",
            JacobiPreconditioner()   => "Jacobi"),
        VariantAxis(:step_size,
            FixedStep(α = 1.0)       => "Fixed",
            ArmijoLS()               => "Armijo",
            CauchyStep(α_max = Inf)  => "Cauchy"),
    ],
    builder = (; preconditioner, step_size) ->
        PreconditionedGradient(preconditioner = preconditioner, step_size = step_size),
)

function precond_config()
    ExperimentConfig(
        name = "exp_precond_grid",
        problem_spec = RandomProblem(name = :separable_quadratic,
                                     params = (n = PRECOND_N, condition_number = PRECOND_KAPPA)),
        conventional_methods = ConventionalMethod[GradientDescent(step_size = ArmijoLS())],
        variant_grids = VariantGrid[precond_grid()],
        stopping_criteria = stop_when_any(MaxIterations(n = PRECOND_CAP),
                                          DistanceToOptimal(tol = 1e-8)),
        n_runs = 1,
        seed = SEED,
    )
end

# ---------------------------------------------------------------------------
# Win condition 1+2: dual-bucket routing and grid expansion (assert BEFORE running)
# ---------------------------------------------------------------------------

function validate_routing(config)
    conventional, experimental = resolve_methods(config)
    conv_names = [n for (n, _) in conventional]
    exp_names  = [n for (n, _) in experimental]

    @info "resolve_methods buckets" conventional=conv_names n_experimental=length(exp_names)
    # The GradientDescent baseline → conventional; all 6 PreconditionedGradient
    # variants → experimental. This is the first real exercise of the dual routing.
    @assert conv_names == ["GradientDescent"] "baseline not routed to conventional: $conv_names"
    @assert length(exp_names) == 6 "expected 6 experimental variants, got $(length(exp_names))"
    @assert all(occursin("PreconditionedGradient", n) for n in exp_names)

    # expand() produces the full 2×3 product with abbreviated short-names.
    specs = expand(precond_grid())
    @assert length(specs) == 6 "grid did not expand to 2×3=6: $(length(specs))"
    shorts = Set(s.short_name for s in specs)
    for expected in ("PCG/Id/Fix", "PCG/Jac/Cau", "PCG/Jac/Arm")
        @assert expected in shorts "missing short-name $expected in $(sort(collect(shorts)))"
    end
    @info "grid expansion" short_names=sort(collect(shorts))
    return conventional, experimental
end

# ---------------------------------------------------------------------------
# Figure: f(xₖ) − f* vs iteration (log-log), Jacobi family vs Identity family
# ---------------------------------------------------------------------------

# Color by preconditioner family; line style by step-size rule.
const PRECOND_COLOR = Dict("Identity" => "#D55E00", "Jacobi" => "#0072B2", "baseline" => :black)
const STEP_STYLE    = Dict("Fixed" => :solid, "Armijo" => :dash, "Cauchy" => :dot)

# Downsample a long curve to ~400 log-spaced points for a light figure.
function _downsample(xs, ys; n = 400)
    length(xs) <= n && return (xs, ys)
    idx = unique(round.(Int, exp10.(range(0, log10(length(xs)); length = n))))
    return (xs[idx], ys[idx])
end

function plot_precond(results::Dict, conventional, experimental;
                       outpath::String = "figures/precond_grid.png")
    fig = Figure(size = (980, 660))
    ax  = Axis(fig[1, 1], xlabel = "iteration k", ylabel = "f(xₖ) − f*",
        xscale = log10, yscale = log10,
        title = "Jacobi preconditioning ≈ Newton on a diagonal quadratic (κ=$(Int(PRECOND_KAPPA)))")

    plot_curve!(name, color, style) = begin
        logs = results[name].iter_logs
        xs = [max(e.iter, 1) for e in logs]
        ys = [max(e.objective, 1e-16) for e in logs]   # f* = 0
        xd, yd = _downsample(xs, ys)
        scatterlines!(ax, xd, yd; color = color, linestyle = style, linewidth = 2,
            markersize = 6, label = abbreviate(split(name, '[')[1]) * label_suffix(name))
    end

    for (name, method) in experimental
        precond = occursin("preconditioner=Jacobi", name) ? "Jacobi" : "Identity"
        step    = occursin("step_size=Fixed", name) ? "Fixed" :
                  occursin("step_size=Armijo", name) ? "Armijo" : "Cauchy"
        plot_curve!(name, PRECOND_COLOR[precond], STEP_STYLE[step])
    end
    # conventional baseline
    plot_curve!(conventional[1][1], PRECOND_COLOR["baseline"], :dashdot)

    axislegend(ax; position = :lb, framevisible = true, nbanks = 2, labelsize = 11)
    mkpath(dirname(outpath)); save(outpath, fig)
    @info "Saved figure" path = outpath
    return fig
end

# Compact legend label from a variant's full name.
function label_suffix(name)
    occursin('[', name) || return ""   # baseline
    pc = occursin("Jacobi", name) ? "Jac" : "Id"
    ss = occursin("Fixed", name) ? "Fix" : occursin("Armijo", name) ? "Arm" : "Cau"
    return "/$pc/$ss"
end

# ---------------------------------------------------------------------------
# Win condition 3: Jacobi ≈ O(1), Identity ≈ O(κ)
# ---------------------------------------------------------------------------

function validate_iters(results::Dict, experimental)
    jac = Int[]; idn = Int[]
    for (name, _) in experimental
        (occursin("Jacobi", name) ? jac : idn) |> v -> push!(v, results[name].n_iters)
    end
    @info "iteration counts" jacobi=jac identity=idn
    @assert maximum(jac) <= 3   "Jacobi variants should converge in ~1 iter; got $jac"
    @assert minimum(idn) >= 100 * maximum(jac) "Identity variants not orders-of-magnitude slower: id=$idn jac=$jac"
    @printf("\nJacobi converges in ≤%d iters; Identity takes %d–%d — a %.0e× gap.\n",
            maximum(jac), minimum(idn), maximum(idn), minimum(idn) / max(maximum(jac), 1))
    return nothing
end

function main()
    config = precond_config()
    conventional, experimental = validate_routing(config)
    result = run_experiment(config)
    results = result.run_results[1].method_results
    validate_iters(results, experimental)
    plot_precond(results, conventional, experimental)
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
