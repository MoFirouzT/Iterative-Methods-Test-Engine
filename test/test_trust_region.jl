using Test
using LinearAlgebra: norm, eigvals
using Random: Xoshiro

include(joinpath(@__DIR__, "..", "experiments", "_bootstrap.jl"))
include(joinpath(@__DIR__, "testutils.jl"))

_trlog() = silent_logger("tr")
_inner_crit(maxit) = stop_when_any(MaxIterations(n = maxit), GradientTolerance(tol = 1e-10),
                                   NegativeCurvature(), TrustRegionBoundary())

@testset "TrustRegion + Steihaug-CG inner solver" begin

    # ── Steihaug-CG in isolation (the crux: unit-test before the outer loop) ──
    @testset "Steihaug-CG branches" begin
        H = [3.0 1.0; 1.0 2.0]; g = [1.0, -2.0]      # H ≻ 0
        pstar = -H \ g
        model = Problem(QuadraticModel(g, MatrixHessian(H)), zeros(2))

        # large Δ ⇒ interior solution = exact Newton step
        r = run_method(SteihaugCG(Δ = 100.0), model, _inner_crit(10), _trlog(), Xoshiro(1))
        @test r.stop_reason == :gradient_converged
        @test norm(r.final_state.iterate.x - pstar) < 1e-8

        # small Δ ⇒ boundary solution with ‖p‖ ≈ Δ
        for Δ in (0.3, 0.7)
            rb = run_method(SteihaugCG(Δ = Δ), model, _inner_crit(10), _trlog(), Xoshiro(1))
            @test rb.stop_reason == :boundary_reached
            @test isapprox(norm(rb.final_state.iterate.x), Δ; rtol = 1e-6)
        end

        # indefinite H ⇒ negative-curvature branch, step to the boundary
        Hi = [1.0 0.0; 0.0 -2.0]
        mi = Problem(QuadraticModel([0.5, 0.5], MatrixHessian(Hi)), zeros(2))
        rn = run_method(SteihaugCG(Δ = 1.0), mi, _inner_crit(10), _trlog(), Xoshiro(1))
        @test rn.stop_reason == :negative_curvature
        @test isapprox(norm(rn.final_state.iterate.x), 1.0; rtol = 1e-6)

        # the model always decreases (m(0) = 0)
        @test value(QuadraticModel(g, MatrixHessian(H)), r.final_state.iterate.x) < 0
    end

    # ── New stopping criteria + the _tr_status accessor ──────────────────────
    @testset "trust-region stopping criteria" begin
        @test _tr_status(nothing) === :none      # default for non-TR states
        @test should_stop(NegativeCurvature(), 1, 1, _trlog()) == (false, :none)   # 1 ⇒ _tr_status :none
        @test should_stop(TrustRegionBoundary(), 1, 1, _trlog()) == (false, :none)
    end

    # ── Outer TrustRegion on Rosenbrock: fast convergence + nesting ──────────
    @testset "TrustRegion on Rosenbrock" begin
        p = make_problem(AnalyticProblem(name = :rosenbrock, params = (rho = 100.0,)), Xoshiro(1))
        lg = _trlog()
        res = run_method(TrustRegion(Δ0 = 1.0), p,
                         stop_when_any(MaxIterations(n = 200), GradientTolerance(tol = 1e-8)),
                         lg, Xoshiro(3))

        # Trust-region-Newton: O(20–30) outer iters to high accuracy.
        @test res.stop_reason == :gradient_converged
        @test res.n_iters <= 35
        @test norm(res.final_state.iterate.x - p.x_opt) < 1e-6

        reals = [e for e in res.iter_logs if e.iter > 0]

        # Nesting subsystem is exercised: sub-logs attached, inner core time > 0.
        @test !isempty(lg.pending_sub_logs)
        @test any(e -> get(e.extras, :inner_core_ns, 0) > 0, reals)
        @test all(haskey(e.extras, :sub_logs) for e in reals)
        @test any(!isempty(e.extras[:sub_logs]) for e in reals)

        # Core-time attribution: the inner solve's core time is folded into the
        # outer step's core_time_ns (so outer ≥ inner, both > 0).
        e1 = reals[1]
        @test e1.extras[:inner_core_ns] > 0
        @test e1.core_time_ns >= e1.extras[:inner_core_ns]

        # The boundary branch fires on this trajectory (early Newton steps exceed Δ).
        @test any(e -> get(e.extras, :inner_stop, :none) == :boundary_reached, reals)
    end

    # ── Negative-curvature fires from an indefinite-Hessian start ─────────────
    @testset "negative curvature on an indefinite start" begin
        # At x0=(0,2) the Rosenbrock Hessian is indefinite (eigvals ≈ [-798, 200]).
        Hx0 = hessian(RosenbrockObjective(RosenbrockKernel(100.0)), [0.0, 2.0])
        @test minimum(eigvals(materialize(Hx0))) < 0      # confirm indefinite

        p = make_problem(AnalyticProblem(name = :rosenbrock, params = (rho = 100.0, x0 = [0.0, 2.0])), Xoshiro(1))
        res = run_method(TrustRegion(Δ0 = 1.0), p,
                         stop_when_any(MaxIterations(n = 200), GradientTolerance(tol = 1e-8)),
                         _trlog(), Xoshiro(3))
        reals = [e for e in res.iter_logs if e.iter > 0]
        @test any(e -> get(e.extras, :inner_stop, :none) == :negative_curvature, reals)
        @test res.stop_reason == :gradient_converged    # still converges
    end
end
