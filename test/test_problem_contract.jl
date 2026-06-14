# test/test_problem_contract.jl
#
# Content conformance harness for the Problem interface.
#
# `check_problem_contract` is the single reusable contract every concrete problem
# must satisfy; it is applied to one example spec per shipped problem family. A
# completeness guard then fails if a registered problem has NO example here — so
# adding a new problem costs one `CONFORMANCE_SPECS` line, not a bespoke test
# suite. The assertions (gradient vs. finite differences, Hessian-vector products,
# prox well-formedness, optimum stationarity) are written once and reused.
#
# Included from runtests.jl immediately after the bootstrap, BEFORE the other
# test files, so the registries hold only content (no throwaway test problems).

using Random: Xoshiro
import .TestEngine: prox    # prox is part of the contract but not exported

"""
    check_problem_contract(prob; rng, minimizer=false)

Assert the universal contract every `Problem` must satisfy:
- the objective (and total objective) is a finite scalar;
- the analytic gradient agrees with central finite differences at `x0` and nearby
  random points;
- Hessian-vector products agree with a directional finite difference (skipped for
  objectives that do not implement `hessian`);
- every regularizer's `prox` returns a finite vector of the right length;
- when `minimizer` is true and `x_opt` is set, `x_opt` is at least as good as `x0`,
  and — for smooth, unconstrained problems — stationary (‖∇f(x_opt)‖ ≈ 0).
"""
function check_problem_contract(prob::Problem; rng::AbstractRNG, minimizer::Bool=false)
    f, x0, n = prob.f, prob.x0, prob.n
    εfd = 1e-6

    # objective is a finite scalar
    @test isfinite(value(f, x0))
    @test isfinite(total_objective(prob, x0))

    # analytic gradient agrees with central finite differences
    for x in (x0, x0 .+ 0.1 .* randn(rng, n), x0 .- 0.1 .* randn(rng, n))
        @test isapprox(grad(f, x), numerical_gradient(f, x, εfd); rtol=1e-4, atol=1e-6)
    end

    # Hessian-vector product agrees with a directional finite diff of the gradient
    try
        H  = hessian(f, x0)
        d  = randn(rng, n)
        Hd = (grad(f, x0 .+ εfd .* d) .- grad(f, x0 .- εfd .* d)) ./ (2εfd)
        @test isapprox(apply(H, d), Hd; rtol=1e-3, atol=1e-4)
    catch err
        err isa MethodError || rethrow()    # objective simply doesn't define a Hessian
    end

    # every regularizer's prox is a finite vector of the right length
    for g in prob.gs
        u = prox(g, x0, 1.0)
        @test length(u) == n
        @test all(isfinite, u)
        @test isfinite(value(g, x0))
    end

    # a known minimizer is at least as good as x0, and stationary when smooth
    if minimizer && prob.x_opt !== nothing
        @test total_objective(prob, prob.x_opt) ≤ total_objective(prob, x0) + 1e-8
        isempty(prob.gs) && @test norm(grad(f, prob.x_opt)) < 1e-5
    end
    return nothing
end

# One example spec per shipped problem family. `minimizer = true` means `x_opt` is
# the true minimizer (assert stationarity); `false` means `x_opt` is absent or a
# planted reference — e.g. the lasso signal is the PLANTED x_star, NOT the lasso
# minimizer, so its stationarity must not be asserted.
const CONFORMANCE_SPECS = [
    (name = :rosenbrock,
     spec = AnalyticProblem(name = :rosenbrock),
     minimizer = true),
    (name = :quadratic,
     spec = AnalyticProblem(name = :quadratic,
                            params = (; A = Matrix{Float64}(I, 3, 3), b = [1.0, 2.0, 3.0])),
     minimizer = false),    # generator leaves x_opt = nothing
    (name = :linear_ls,
     spec = RandomProblem(name = :linear_ls, params = (; n = 20)),
     minimizer = true),
    (name = :separable_quadratic,
     spec = RandomProblem(name = :separable_quadratic, params = (; n = 20)),
     minimizer = true),
    (name = :lasso,
     spec = RandomProblem(name = :lasso, params = (; m = 32, n = 64, k = 4)),
     minimizer = false),    # x_opt is the planted signal, not the lasso minimizer
]

@testset "Problem contract conformance" begin
    for entry in CONFORMANCE_SPECS
        @testset "$(entry.name)" begin
            rng  = Xoshiro(0x1234)
            prob = make_problem(entry.spec, rng)
            check_problem_contract(prob; rng = rng, minimizer = entry.minimizer)
        end
    end

    # Completeness guard: every registered content problem must have an example
    # spec above. New content fails this test until it adds one CONFORMANCE_SPECS
    # line — the lever that makes coverage mandatory while keeping it cheap.
    covered      = Set(e.name for e in CONFORMANCE_SPECS)
    registered   = union(keys(TestEngine.ANALYTIC_PROBLEMS), keys(TestEngine.RANDOM_GENERATORS))
    uncovered    = setdiff(registered, covered)
    isempty(uncovered) || @info "Registered problems missing a CONFORMANCE_SPECS entry" uncovered
    @test isempty(uncovered)
end
