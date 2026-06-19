using Test
using Random: Xoshiro

include(joinpath(@__DIR__, "..", "experiments", "_bootstrap.jl"))
include(joinpath(@__DIR__, "testutils.jl"))

_oclog() = silent_logger("oc")
_rosen() = make_problem(AnalyticProblem(name = :rosenbrock, params = (rho = 100.0,)), Xoshiro(1))

@testset "Oracle counting (opt-in)" begin

    # ── Default off: no counter, nothing surfaced ────────────────────────────
    @testset "off by default" begin
        p   = _rosen()
        res = run_method(GradientDescent(step_size = ArmijoLS()), p,
                         MaxIterations(n = 5), _oclog(), Xoshiro(1))
        @test isnothing(oracle_counts(p))                     # plain problem, no counter
        @test !haskey(res.iter_logs[end].extras, :n_grad)     # nothing surfaced
    end

    # ── value + grad! counted and surfaced cumulatively ──────────────────────
    @testset "value & grad! counted" begin
        p  = with_oracle_counting(_rosen())
        oc = oracle_counts(p)
        @test oc isa OracleCounts
        res = run_method(GradientDescent(step_size = ArmijoLS()), p,
                         MaxIterations(n = 10), _oclog(), Xoshiro(1))
        @test oc.n_grad  >= 10        # one grad! per step (+ init)
        @test oc.n_value >= 10        # Armijo backtracking issues value() calls
        logs  = filter(e -> e.iter > 0, res.iter_logs)
        grads = [e.extras[:n_grad] for e in logs]
        @test grads == sort(grads)            # cumulative ⇒ nondecreasing
        @test last(grads) == oc.n_grad        # final log == live counter
        @test all(haskey(e.extras, :n_value) && haskey(e.extras, :n_hvp) for e in logs)
    end

    # ── Hessian-vector products counted (curvature step) ─────────────────────
    @testset "Hvp counted" begin
        p  = with_oracle_counting(_rosen())
        oc = oracle_counts(p)
        run_method(GradientDescent(step_size = CauchyStep()), p,
                   MaxIterations(n = 8), _oclog(), Xoshiro(1))
        @test oc.n_hvp >= 8           # CauchyStep does one apply(H, d) per step
    end

    # ── Nested: TrustRegion inner CG Hvps fold into the shared counter ───────
    @testset "nested Hvp folds in" begin
        p  = with_oracle_counting(_rosen())
        oc = oracle_counts(p)
        run_method(TrustRegion(Δ0 = 1.0), p, MaxIterations(n = 5), _oclog(), Xoshiro(1))
        @test oc.n_hvp > 5           # several inner CG applies per outer step
    end

    # ── Transparency: Jacobi still works under counting (trait forwards) ─────
    @testset "Jacobi transparent under counting" begin
        p  = with_oracle_counting(
                 make_problem(RandomProblem(name = :separable_quadratic,
                                            params = (n = 20, condition_number = 1.0e3)),
                              Xoshiro(2)))
        oc = oracle_counts(p)
        run_method(PreconditionedGradient(preconditioner = JacobiPreconditioner(),
                                          step_size = FixedStep(α = 1.0)),
                   p, MaxIterations(n = 5), _oclog(), Xoshiro(2))
        @test oc.n_grad >= 5         # ran to completion without error
        @test oc.n_hvp == 0          # Jacobi reads diagonal(), not apply()
    end

    # ── End-to-end through run_experiment with the config flag ───────────────
    @testset "run_experiment count_oracles flag" begin
        cfg = ExperimentConfig(
            name             = "oracle-count smoke",
            problem_spec     = AnalyticProblem(name = :rosenbrock, params = (rho = 100.0,)),
            baseline_methods = [GradientDescent(step_size = ArmijoLS())],
            stopping_criteria = MaxIterations(n = 5),
            count_oracles    = true,
        )
        result = run_experiment(cfg, mktempdir(); verbosity = VerbosityConfig(level = SILENT))
        mr = first(values(result.run_results[1].method_results))   # the lone baseline
        last_log = mr.iter_logs[end]
        @test haskey(last_log.extras, :n_grad) && last_log.extras[:n_grad] >= 5
    end
end
