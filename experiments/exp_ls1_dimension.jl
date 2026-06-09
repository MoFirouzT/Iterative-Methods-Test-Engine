# experiments/exp_ls1_dimension.jl
#
# Stage LS-1 — dimension scaling (portfolio Item 1).
# Five GD variants on consistent linear least squares ½‖Ax−b‖², FIXED condition
# number κ, swept over dimension n ∈ {10, 100, 1000} (m = 2n). Two things to see:
#
#   (left)  iters-to-tolerance vs n — ~FLAT: the rate is set by κ, not by n, so
#           no method has a hidden O(n)-iteration dependence.
#   (right) wall time vs n — grows ~O(mn) from the matvec kernel.
#
# Timing-pillar validation: at n = 1000 the O(mn) matvec finally dominates the
# per-iter scaffolding, so core_time/wall_time lands in [50%, 110%] — the 2-D
# Rosenbrock kernel was below the timing noise floor and could never show this.
#
# Run:  julia --project=. experiments/exp_ls1_dimension.jl

include("_bootstrap.jl")
using Random
using DataFrames
using CairoMakie
using Printf
using .TestEngine: elapsed_core_s
include("_shared.jl")

const SEED      = 42
const RUN_ID    = 1
const LS1_KAPPA = 1.0e3
const LS1_DIMS  = [10, 100, 1000]
const LS1_TOL   = 1e-6           # absolute ‖x − x*‖ tolerance (system is consistent)
const LS1_CAP   = 200_000

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

function run_ls1(; seed::Int = SEED, kappa = LS1_KAPPA, dims = LS1_DIMS,
                   tol = LS1_TOL, cap = LS1_CAP)
    # Warm up the JIT on the smallest dim so the timed runs are compile-free —
    # essential for an honest core_time/wall_time ratio.
    let p = make_problem(RandomProblem(name = :linear_ls,
                                       params = (n = dims[1], condition_number = kappa)),
                         Xoshiro(seed))
        for (_, m) in build_ls_methods(p.meta[:L])
            run_method(m, p, MaxIterations(n = 20),
                       TestEngine.make_logger("warm", 0, "", VerbosityConfig(level = SILENT)),
                       Xoshiro(0))
        end
    end

    rows = NamedTuple[]
    for n in dims
        p = make_problem(RandomProblem(name = :linear_ls,
                                       params = (n = n, condition_number = kappa)),
                         Xoshiro(hash((seed, :data, n))))
        L = p.meta[:L]; m = p.meta[:m]
        crit = stop_when_any(MaxIterations(n = cap), DistanceToOptimal(tol = tol))
        for (name, method) in build_ls_methods(L)
            logger = TestEngine.make_logger(name, RUN_ID, "", VerbosityConfig(level = SILENT))
            wall   = @elapsed res = run_method(method, p, crit, logger,
                                               Xoshiro(hash((seed, name, n))))
            core = elapsed_core_s(logger)
            push!(rows, (
                method_name = name, n = n, m = m,
                iters     = res.n_iters,
                converged = res.stop_reason == :optimal_reached,
                wall_s    = wall,
                core_s    = core,
                core_frac = core / wall,
                dist      = res.final_state.metrics.dist_to_opt,
            ))
            @info "[$name] n=$n" iters=res.n_iters wall_s=round(wall, sigdigits=3) core_frac=round(core/wall, sigdigits=3)
        end
    end
    return DataFrame(rows)
end

# ---------------------------------------------------------------------------
# Figure: iters-to-tol vs n (flat) + wall time vs n (∝ mn)
# ---------------------------------------------------------------------------

function plot_ls1(df::DataFrame; outpath::String = "figures/ls1_dimension.png")
    fig = Figure(size = (1280, 520))

    axL = Axis(fig[1, 1], xlabel = "dimension n", ylabel = "iterations to ‖x−x*‖ ≤ 1e-6",
        xscale = log10, yscale = log10, title = "Iterations vs n  (fixed κ=$(Int(LS1_KAPPA)) ⇒ ~flat)")
    axR = Axis(fig[1, 2], xlabel = "dimension n", ylabel = "wall time (s)",
        xscale = log10, yscale = log10, title = "Wall time vs n  (matvec O(mn))")

    for name in PLOT_ORDER
        sub = sort(filter(:method_name => ==(name), df), :n)
        scatterlines!(axL, sub.n, max.(sub.iters, 1); color = COLORS[name], linewidth = 2, markersize = 10, label = name)
        scatterlines!(axR, sub.n, sub.wall_s;          color = COLORS[name], linewidth = 2, markersize = 10, label = name)
    end
    Legend(fig[1, 3], axL, "step size"; framevisible = true)
    Label(fig[0, :], "Stage LS-1 — linear least squares, dimension scaling", fontsize = 16, font = :bold)

    mkpath(dirname(outpath)); save(outpath, fig)
    @info "Saved figure" path = outpath
    return fig
end

# ---------------------------------------------------------------------------
# Win-condition checks
# ---------------------------------------------------------------------------

function validate_ls1(df::DataFrame)
    # 1. Every variant reaches the optimum on the consistent system.
    @assert all(df.converged) "not all runs converged: $(df[.!df.converged, [:method_name,:n]])"

    # 2. Iters-to-tol does NOT scale with n (rate is κ-driven, not n-driven): a
    #    hidden O(n) dependence would show 10×/100× jumps across the 100× dim
    #    range. The O(κ) trio (Fixed/Armijo/Cauchy) is essentially flat; BB1/BB2
    #    are spectrum-sensitive, so their per-instance count carries more noise —
    #    a 2× band absorbs that while still excluding any real n-scaling.
    for name in PLOT_ORDER
        its = sort(filter(:method_name => ==(name), df), :n).iters
        ratio = maximum(its) / minimum(its)
        @info "iters spread (across 100× dim range)" method=name iters=its max_over_min=round(ratio, sigdigits=3)
        @assert ratio < 2.0 "iters for $name vary >2× across n (suggests n-scaling): $its"
    end

    # 3. Timing pillar: at the largest n, the matvec dominates scaffolding, so
    #    core/wall lands in [0.5, 1.1] for the matvec-pure Fixed method.
    nmax = maximum(df.n)
    fixed_big = only(filter(r -> r.method_name == "Fixed" && r.n == nmax, eachrow(df)))
    @info "timing pillar" n=nmax core_frac_Fixed=round(fixed_big.core_frac, sigdigits=3)
    @assert 0.5 <= fixed_big.core_frac <= 1.1 "core/wall for Fixed at n=$nmax = $(fixed_big.core_frac) ∉ [0.5,1.1]"

    @printf("\ncore_time/wall_time by (method, n):\n")
    for name in PLOT_ORDER
        sub = sort(filter(:method_name => ==(name), df), :n)
        @printf("  %-7s " , name)
        for r in eachrow(sub); @printf("n=%-5d %4.0f%%   ", r.n, 100 * r.core_frac); end
        println()
    end
    return nothing
end

function main()
    df = run_ls1()
    validate_ls1(df)
    plot_ls1(df)
    return df
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
