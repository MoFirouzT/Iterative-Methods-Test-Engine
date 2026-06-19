using Test
using LinearAlgebra: norm, opnorm, cholesky, Diagonal, I
using Random: MersenneTwister, Xoshiro, randperm

include(joinpath(@__DIR__, "..", "experiments", "_bootstrap.jl"))
include(joinpath(@__DIR__, "testutils.jl"))
import .TestEngine: Regularizer, value, prox

# A regularizer that tallies prox calls, to assert "exactly one prox per step".
mutable struct CountingReg <: Regularizer
    inner::L1Norm
    n_prox::Int
end
CountingReg(λ) = CountingReg(L1Norm(λ), 0)
value(g::CountingReg, x::Vector{Float64}) = value(g.inner, x)
function prox(g::CountingReg, x::Vector{Float64}, γ::Float64)
    g.n_prox += 1
    return prox(g.inner, x, γ)
end

# Helper: run a ProximalGradient method to a fixed iteration budget.
function _run_pg(method, problem, K; seed = 1)
    run_method(method, problem, MaxIterations(n = K), silent_logger("PG"), Xoshiro(seed))
end

# Least-squares slope of log₁₀(f(xₖ)−f*) vs log₁₀(k) over an iteration window —
# the empirically *measured* convergence-rate exponent (≈ −1 for O(1/k),
# ≈ −2 for O(1/k²)). Iterations with a non-positive gap are skipped.
function loglog_slope(res, fstar, klo, khi)
    xs = Float64[]; ys = Float64[]
    for e in res.iter_logs
        klo <= e.iter <= khi || continue
        g = e.objective - fstar
        g > 0 || continue
        push!(xs, log10(e.iter)); push!(ys, log10(g))
    end
    x̄ = sum(xs) / length(xs); ȳ = sum(ys) / length(ys)
    return sum((xs .- x̄) .* (ys .- ȳ)) / sum((xs .- x̄) .^ 2)
end

@testset "ProximalGradient (ISTA / FISTA)" begin

    # ── ZeroRegularizer ⇒ identity prox ⇒ reduces to gradient descent ────────
    # Build an SPD quadratic ½‖M x − b‖² with M = chol(A).U so f* = 0 at x = xopt.
    rng = MersenneTwister(0)
    n = 15
    B = randn(rng, n, n); A = B'B + I
    xopt = randn(rng, n)
    M = Matrix(cholesky(A).U)
    b = M * xopt
    L = opnorm(M)^2
    prob_smooth = Problem(LeastSquares(LeastSquaresKernel(M, b)), zeros(n); x_opt = xopt)

    ista_smooth = ProximalGradient(step_size = FixedStep(α = 1/L), extrapolation = NoExtrapolation())
    res = _run_pg(ista_smooth, prob_smooth, 5000)
    # With no regularizer the prox step is the identity, so this IS gradient
    # descent on the smooth part and must reach the unique minimizer.
    @test norm(res.final_state.iterate.x .- xopt) < 1e-5
    @test res.final_state.metrics.objective < 1e-9

    # FISTA on the same smooth problem must reach the minimizer at least as fast.
    fista_smooth = ProximalGradient(step_size = FixedStep(α = 1/L), extrapolation = NesterovStep())
    res_f = _run_pg(fista_smooth, prob_smooth, 5000)
    @test norm(res_f.final_state.iterate.x .- xopt) < 1e-5

    # ── Composite lasso: total_objective decomposes, FISTA accelerates ───────
    prob = make_problem(RandomProblem(name = :lasso,
                                      params = (m = 120, n = 256, k = 10, λ = 0.05)),
                        Xoshiro(7))
    Ll = prob.meta[:L]
    @test length(prob.gs) == 1 && prob.gs[1] isa L1Norm

    ista  = ProximalGradient(step_size = FixedStep(α = 1/Ll), extrapolation = NoExtrapolation())
    fista = ProximalGradient(step_size = FixedStep(α = 1/Ll), extrapolation = NesterovStep())

    K = 150
    ri = _run_pg(ista, prob, K)
    rf = _run_pg(fista, prob, K)

    # total_objective(x) == value(f, x) + value(g, x) on the final iterate.
    x_hat = rf.final_state.iterate.x
    decomp = value(prob.f, x_hat) + value(prob.gs[1], x_hat)
    @test isapprox(total_objective(prob, x_hat), decomp; rtol = 1e-10)

    # Reference f* from a long FISTA run; FISTA's gap must beat ISTA's at a mid iter.
    fstar = minimum(e.objective for e in _run_pg(fista, prob, 20_000).iter_logs)
    gap_at(res, it) = res.iter_logs[it + 1].objective - fstar   # +1: iter 0 is index 1
    @test gap_at(rf, 50) < gap_at(ri, 50)
    @test gap_at(rf, 50) < 0.1 * gap_at(ri, 50)   # acceleration is substantial, not marginal

    # ── Convergence RATE is measured, not eyeballed ──────────────────────────
    # The O(1/k) vs O(1/k²) rate *separation* is a smooth, non-strongly-convex
    # phenomenon. We exhibit it on a quadratic ½‖Ax‖² whose Hessian AᵀA = diag(λ)
    # has a dense spectrum log-spaced 1 → 1e-7: with no strong convexity there is
    # a long pre-asymptotic window where the textbook sublinear envelopes hold,
    # so the log-log slope of f(xₖ)−f* is a clean rate exponent. (The composite
    # lasso above instead identifies its support fast and then converges
    # *linearly* — it shows acceleration but not the sublinear-rate separation,
    # which is why the slope is measured here on a dedicated instance.)
    @testset "sublinear rate: GD O(1/k), accelerated O(1/k²)" begin
        nq = 500
        λq = exp10.(range(0, -7; length = nq))           # dense spectrum 1 → 1e-7
        Aq = Matrix(Diagonal(sqrt.(λq)))                 # AᵀA = diag(λ), so L = 1
        prob_q = Problem(LeastSquares(LeastSquaresKernel(Aq, zeros(nq))),
                         ones(nq); x_opt = zeros(nq))    # f* = 0 at x* = 0

        gd  = ProximalGradient(step_size = FixedStep(α = 1.0), extrapolation = NoExtrapolation())
        agd = ProximalGradient(step_size = FixedStep(α = 1.0), extrapolation = NesterovStep())
        rg  = _run_pg(gd,  prob_q, 2000)
        ra  = _run_pg(agd, prob_q, 2000)

        s_gd  = loglog_slope(rg, 0.0, 50, 2000)          # f* = 0 exactly here
        s_agd = loglog_slope(ra, 0.0, 50, 2000)

        @test -1.1 <= s_gd  <= -0.9                      # ≈ −1   (O(1/k))
        @test -2.2 <= s_agd <= -1.75                     # ≈ −2   (O(1/k²))
        @test s_agd < s_gd - 0.5                         # a full order steeper
    end

    # ── gradient_norm is the composite stationarity residual (gradient mapping) ──
    # On the composite lasso, ‖G_γ‖ must drive to ~0 at the solution, so
    # GradientTolerance is a valid stop (it never would be for ‖∇f(y)‖, which does
    # not vanish at a composite minimizer). Smooth-case equivalence to ‖∇f(y)‖ is
    # exercised by test_external_validation.jl's GradientTolerance(1e-10) run.
    @testset "gradient mapping = composite stationarity residual" begin
        lg = silent_logger("gm")
        rconv = run_method(fista, prob,
            stop_when_any(MaxIterations(n = 50_000), GradientTolerance(tol = 1e-8)),
            lg, Xoshiro(1))
        @test rconv.stop_reason == :gradient_converged   # stops on the mapping, not the iter cap
        @test rconv.final_state.metrics.gradient_norm < 1e-8
    end

    # ── Support recovery (flagship win condition, pinned in CI) ──────────────
    # A long FISTA solve recovers the planted k-sparse support exactly: every
    # planted spike clears the 0.1 threshold and no off-support coordinate does.
    x_long    = _run_pg(fista, prob, 5_000).final_state.iterate.x
    rec_supp  = findall(>(0.1), abs.(x_long))
    true_supp = prob.meta[:support]
    @test issubset(true_supp, rec_supp)           # all planted spikes recovered
    @test isempty(setdiff(rec_supp, true_supp))   # no spurious spikes

    # ── Exactly one prox call per step ───────────────────────────────────────
    counting = CountingReg(0.05)
    prob_count = Problem(prob.f, Regularizer[counting], copy(prob.x0))
    rc = _run_pg(ista, prob_count, K)
    @test rc.n_iters == K
    @test counting.n_prox == K   # one prox per iteration, no more, no fewer

    # ── Restriction: more than one regularizer is rejected ───────────────────
    bad = Problem(LeastSquares(LeastSquaresKernel(M, b)),
                  Regularizer[L1Norm(0.1), L1Norm(0.2)], zeros(n))
    @test_throws ArgumentError init_state(ista, bad, Xoshiro(1))
end
