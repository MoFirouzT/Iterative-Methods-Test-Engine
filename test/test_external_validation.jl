using Test
using LinearAlgebra: norm, dot
using Random: Xoshiro

import Optim
import ProximalOperators
import ProximalAlgorithms

include(joinpath(@__DIR__, "..", "experiments", "_bootstrap.jl"))

# ---------------------------------------------------------------------------
# These checks match *fixed points*, not iterate-by-iterate paths: our solvers
# and the reference solvers should converge to the same minimizer/objective on
# the same instance, to ~1e-4–1e-6. Optim and ProximalAlgorithms are
# validation-only dependencies (see Project.toml).
# ---------------------------------------------------------------------------

# ProximalAlgorithms 0.7 drives its smooth term through `value_and_gradient`;
# supply ½‖Ax−b‖² with its analytic gradient (no autodiff needed).
struct _LSSmooth
    A::Matrix{Float64}
    b::Vector{Float64}
end
function ProximalAlgorithms.value_and_gradient(f::_LSSmooth, x)
    r = f.A * x - f.b
    return 0.5 * dot(r, r), f.A' * r
end

_extlog() = TestEngine.make_logger("ext", 1, "", VerbosityConfig(level = SILENT))

@testset "External validation (Optim, ProximalAlgorithms)" begin

    # ── Smooth least squares: our accelerated GD vs A\b and Optim ────────────
    @testset "least squares vs normal equations + Optim" begin
        p = make_problem(RandomProblem(name = :linear_ls,
                                       params = (n = 60, condition_number = 1.0e3)), Xoshiro(11))
        A = p.f.kernel.A; b = p.f.kernel.b
        L = p.meta[:L]

        x_normal = A \ b                      # normal-equation solution
        @test norm(x_normal - p.x_opt) < 1e-6 # consistent system ⇒ equals the planted x*

        # Our FISTA (ProximalGradient, no regularizer ⇒ accelerated GD on the smooth f).
        fista = ProximalGradient(step_size = FixedStep(α = 1 / L), extrapolation = NesterovStep())
        res = run_method(fista, p,
                         stop_when_any(MaxIterations(n = 50_000), GradientTolerance(tol = 1e-10)),
                         _extlog(), Xoshiro(3))
        x_mine = res.final_state.iterate.x
        @test norm(x_mine - x_normal) < 1e-4

        # Optim's GradientDescent and LBFGS on the same objective/gradient.
        fobj(x) = value(p.f, x)
        g!(G, x) = grad!(G, p.f, x)
        opts = Optim.Options(g_tol = 1e-12, iterations = 200_000)
        for method in (Optim.LBFGS(), Optim.GradientDescent())
            r = Optim.optimize(fobj, g!, copy(p.x0), method, opts)
            @test norm(Optim.minimizer(r) - x_normal) < 1e-4
        end
        # Objective agreement at the optimum (both ≈ 0).
        r = Optim.optimize(fobj, g!, copy(p.x0), Optim.LBFGS(), opts)
        @test isapprox(value(p.f, x_mine), Optim.minimum(r); atol = 1e-8)
    end

    # ── Composite lasso: our ISTA/FISTA vs ProximalAlgorithms ────────────────
    @testset "lasso vs ProximalAlgorithms ForwardBackward / FastForwardBackward" begin
        p = make_problem(RandomProblem(name = :lasso,
                                       params = (m = 128, n = 256, k = 10, λ = 0.1)), Xoshiro(12))
        A = p.f.kernel.A; b = p.f.kernel.b
        λ = p.gs[1].λ; L = p.meta[:L]
        obj(x) = total_objective(p, x)        # ½‖Ax−b‖² + λ‖x‖₁

        solve_mine(mu) = run_method(
            ProximalGradient(step_size = FixedStep(α = 1 / L), extrapolation = mu),
            p, MaxIterations(n = 50_000), _extlog(), Xoshiro(3)).final_state.iterate.x
        x_ista  = solve_mine(NoExtrapolation())
        x_fista = solve_mine(NesterovStep())

        # Reference solvers on the same instance.
        ff = _LSSmooth(A, b); gg = ProximalOperators.NormL1(λ)
        x_fb  = ProximalAlgorithms.ForwardBackward(tol = 1e-10, maxit = 100_000)(x0 = zeros(p.n), f = ff, g = gg)[1]
        x_ffb = ProximalAlgorithms.FastForwardBackward(tol = 1e-10, maxit = 100_000)(x0 = zeros(p.n), f = ff, g = gg)[1]

        # Fixed points agree: same unique lasso minimizer (objective + iterate).
        @test isapprox(obj(x_fista), obj(x_ffb); rtol = 1e-5)
        @test isapprox(obj(x_ista),  obj(x_fb);  rtol = 1e-5)
        @test norm(x_fista - x_ffb) < 1e-3
        @test norm(x_ista  - x_fb)  < 1e-3
        # Our ISTA and FISTA reach the same point (internal consistency).
        @test norm(x_ista - x_fista) < 1e-3
    end
end
