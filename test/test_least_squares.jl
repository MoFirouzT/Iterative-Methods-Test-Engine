using Test
using LinearAlgebra: norm, cond, diag, I
using Random: Xoshiro

include(joinpath(@__DIR__, "..", "experiments", "_bootstrap.jl"))

@testset "LeastSquares Hessian modes + :linear_ls" begin

    # ── Selectable Hessian representation ────────────────────────────────────
    A = randn(Xoshiro(1), 8, 5); b = randn(Xoshiro(2), 8)
    kernel = LeastSquaresKernel(A, b)
    d = randn(Xoshiro(3), 5)
    AtA = A' * A

    # Default is :matrix (preserves prior behavior): materialize + diagonal work.
    f_mat = LeastSquares(kernel)
    @test f_mat.hessian_mode === :matrix
    Hm = hessian(f_mat, zeros(5))
    @test Hm isa MatrixHessian
    @test apply(Hm, d) ≈ AtA * d
    @test materialize(Hm) ≈ AtA
    @test diagonal(Hm) ≈ diag(AtA)

    # :operator never materializes; apply matches AᵀA, materialize is unavailable.
    f_op = LeastSquares(kernel, :operator)
    Ho = hessian(f_op, zeros(5))
    @test Ho isa OperatorHessian
    @test apply(Ho, d) ≈ AtA * d
    @test_throws MethodError materialize(Ho)

    @test_throws ArgumentError hessian(LeastSquares(kernel, :bogus), zeros(5))

    # ── :linear_ls generator ─────────────────────────────────────────────────
    for κ in (1.0e1, 1.0e3, 1.0e5)
        p = make_problem(RandomProblem(name = :linear_ls,
                                       params = (n = 40, condition_number = κ)), Xoshiro(7))
        # κ parametrizes cond(AᵀA), NOT cond(A): the squaring must be right.
        @test isapprox(cond(p.f.kernel.A' * p.f.kernel.A), κ; rtol = 1e-6)
        # Consistent system ⇒ x_opt = x_star is the exact minimizer with f* = 0.
        @test total_objective(p, p.x_opt) < 1e-18
        @test p.f.hessian_mode === :operator
        @test p.meta[:condition_number] == κ
    end

    # ── Regression: Cauchy converges on a consistent LS system ────────────────
    # The absolute curvature guard used to misfire as ‖∇f‖→0, collapsing Cauchy
    # to its tiny fallback step and stalling (~240k iters at κ=100). The
    # scale-relative guard fixes it: exact-line-search steepest descent should
    # reach the optimum in O(κ) iters, well under the cap.
    p = make_problem(RandomProblem(name = :linear_ls,
                                   params = (n = 60, condition_number = 1.0e2)), Xoshiro(11))
    cauchy = GradientDescent(step_size = CauchyStep(α_max = Inf))   # exact LS on a true quadratic
    lg  = TestEngine.make_logger("C", 1, "", VerbosityConfig(level = SILENT))
    res = run_method(cauchy, p,
                     stop_when_any(MaxIterations(n = 50_000), DistanceToOptimal(tol = 1e-6)),
                     lg, Xoshiro(3))
    @test res.stop_reason == :optimal_reached
    @test res.n_iters < 5_000      # would be ~240k under the old absolute guard
end
