using Test
using LinearAlgebra: norm, opnorm, cholesky, I
using Random: MersenneTwister, Xoshiro, randperm

include(joinpath(@__DIR__, "..", "experiments", "_bootstrap.jl"))
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
# NB: qualify make_logger — runtests.jl defines a zero-arg make_logger() in Main
# that would otherwise shadow the engine's exported constructor.
function _run_pg(method, problem, K; seed = 1)
    lg = TestEngine.make_logger("PG", 1, "", VerbosityConfig(level = SILENT))
    run_method(method, problem, MaxIterations(n = K), lg, Xoshiro(seed))
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

    ista_smooth = ProximalGradient(step_size = FixedStep(α = 1/L), minor_update = NoMinorUpdate())
    res = _run_pg(ista_smooth, prob_smooth, 5000)
    # With no regularizer the prox step is the identity, so this IS gradient
    # descent on the smooth part and must reach the unique minimizer.
    @test norm(res.final_state.iterate.x .- xopt) < 1e-5
    @test res.final_state.metrics.objective < 1e-9

    # FISTA on the same smooth problem must reach the minimizer at least as fast.
    fista_smooth = ProximalGradient(step_size = FixedStep(α = 1/L), minor_update = NesterovStep())
    res_f = _run_pg(fista_smooth, prob_smooth, 5000)
    @test norm(res_f.final_state.iterate.x .- xopt) < 1e-5

    # ── Composite lasso: total_objective decomposes, FISTA accelerates ───────
    prob = make_problem(RandomProblem(name = :lasso,
                                      params = (m = 120, n = 256, k = 10, λ = 0.05)),
                        Xoshiro(7))
    Ll = prob.meta[:L]
    @test length(prob.gs) == 1 && prob.gs[1] isa L1Norm

    ista  = ProximalGradient(step_size = FixedStep(α = 1/Ll), minor_update = NoMinorUpdate())
    fista = ProximalGradient(step_size = FixedStep(α = 1/Ll), minor_update = NesterovStep())

    K = 150
    ri = _run_pg(ista, prob, K)
    rf = _run_pg(fista, prob, K)

    # total_objective(x) == value(f, x) + value(g, x) on the final iterate.
    x_hat = rf.final_state.iterate.x
    decomp = value(prob.f, x_hat) + value(prob.gs[1], x_hat)
    @test isapprox(total_objective(prob, x_hat), decomp; rtol = 1e-12)

    # Reference f* from a long FISTA run; FISTA's gap must beat ISTA's at a mid iter.
    fstar = minimum(e.objective for e in _run_pg(fista, prob, 20_000).iter_logs)
    gap_at(res, it) = res.iter_logs[it + 1].objective - fstar   # +1: iter 0 is index 1
    @test gap_at(rf, 50) < gap_at(ri, 50)
    @test gap_at(rf, 50) < 0.1 * gap_at(ri, 50)   # acceleration is substantial, not marginal

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
