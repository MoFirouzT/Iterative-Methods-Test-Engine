using Test
using LinearAlgebra: norm, Diagonal
using Random: Xoshiro

include(joinpath(@__DIR__, "..", "experiments", "_bootstrap.jl"))
include(joinpath(@__DIR__, "testutils.jl"))

_mklog() = silent_logger("PCG")
_run(method, p, K; tol = 1e-8) =
    run_method(method, p, stop_when_any(MaxIterations(n = K), DistanceToOptimal(tol = tol)),
               _mklog(), Xoshiro(3))

@testset "PreconditionedGradient + preconditioners" begin

    p = make_problem(RandomProblem(name = :separable_quadratic,
                                   params = (n = 40, condition_number = 1.0e4)), Xoshiro(1))

    # ── Jacobi on a diagonal Hessian IS Newton: one step to the minimizer ────
    jac_fixed = PreconditionedGradient(preconditioner = JacobiPreconditioner(),
                                       step_size = FixedStep(α = 1.0))
    res = _run(jac_fixed, p, 100)
    @test res.n_iters == 1
    @test res.stop_reason == :optimal_reached
    @test total_objective(p, res.final_state.iterate.x) < 1e-18

    # ── Identity reduces to gradient descent (same trajectory as GradientDescent) ─
    idn = PreconditionedGradient(preconditioner = IdentityPreconditioner(),
                                 step_size = FixedStep(α = 1.0))
    gd  = GradientDescent(direction = SteepestDescent(), step_size = FixedStep(α = 1.0))
    r_idn = _run(idn, p, 500_000); r_gd = _run(gd, p, 500_000)
    @test r_idn.n_iters == r_gd.n_iters
    @test r_idn.final_state.iterate.x ≈ r_gd.final_state.iterate.x

    # ── The preconditioning win: Jacobi orders of magnitude fewer iters ──────
    @test r_idn.n_iters > 1000 * res.n_iters

    # ── diagonal-availability trait ──────────────────────────────────────────
    @test _supports_diagonal(DiagonalHessian([1.0, 2.0]))
    @test _supports_diagonal(MatrixHessian([1.0 0.0; 0.0 2.0]))
    @test !_supports_diagonal(OperatorHessian(d -> d, 2))

    # ── Jacobi works on a MatrixHessian (diag available) ─────────────────────
    # A = diag(√d) ⇒ AᵀA = diag(d), a MatrixHessian whose diagonal Jacobi reads.
    dv = exp10.(range(0, -3, length = 12))                    # span [1e-3, 1]
    A  = Matrix(Diagonal(sqrt.(dv)))
    xstar = collect(1.0:12.0)
    pm = Problem(LeastSquares(LeastSquaresKernel(A, A * xstar), :matrix), zeros(12); x_opt = xstar)
    rm = _run(PreconditionedGradient(preconditioner = JacobiPreconditioner(),
                                     step_size = FixedStep(α = 1.0)), pm, 100)
    @test rm.n_iters == 1
    @test norm(rm.final_state.iterate.x .- xstar) < 1e-8

    # ── Jacobi is correctly INAPPLICABLE on an OperatorHessian (clean error) ──
    # The error surfaces in step! (where precondition reads the diagonal), so it
    # propagates out of run_method as a clean ArgumentError — not a silent fallback.
    p_op = make_problem(RandomProblem(name = :linear_ls,
                                      params = (n = 15, condition_number = 1.0e2)), Xoshiro(5))
    @test_throws ArgumentError run_method(jac_fixed, p_op, MaxIterations(n = 1), _mklog(), Xoshiro(1))

    # ── Dual-bucket routing + 2×3 grid expansion in one experiment ───────────
    grid = VariantGrid(
        base_name = "PreconditionedGradient",
        axes = [
            VariantAxis(:preconditioner,
                IdentityPreconditioner() => "Identity", JacobiPreconditioner() => "Jacobi"),
            VariantAxis(:step_size,
                FixedStep(α = 1.0) => "Fixed", ArmijoLS() => "Armijo", CauchyStep(α_max = Inf) => "Cauchy"),
        ],
        builder = (; preconditioner, step_size) ->
            PreconditionedGradient(preconditioner = preconditioner, step_size = step_size),
    )
    specs = expand(grid)
    @test length(specs) == 6                       # 2 × 3 Cartesian product
    @test all(s.method isa PreconditionedGradient for s in specs)

    config = ExperimentConfig(
        name = "test_precond",
        problem_spec = RandomProblem(name = :separable_quadratic, params = (n = 10, condition_number = 1.0e2)),
        baseline_methods = IterativeMethod[GradientDescent(step_size = ArmijoLS())],
        variant_grids = VariantGrid[grid],
    )
    baseline, experimental = resolve_methods(config)
    @test [n for (n, _) in baseline] == ["GradientDescent"]       # baseline_methods → baseline bucket
    @test length(experimental) == 6                               # grid role :experimental → experimental
    @test all(occursin("PreconditionedGradient", n) for (n, _) in experimental)
end
