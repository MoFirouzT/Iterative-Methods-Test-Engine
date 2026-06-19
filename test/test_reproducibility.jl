using Test
using Random: Xoshiro

include(joinpath(@__DIR__, "..", "experiments", "_bootstrap.jl"))
include(joinpath(@__DIR__, "testutils.jl"))

# Reproducibility is a headline property of the engine (committed Manifest, seeded
# runs). Pin it: a fixed seed must reproduce a run bit-for-bit, and the seed must
# actually drive problem generation (a different seed yields a different instance).
@testset "reproducibility (seed determinism)" begin
    spec = RandomProblem(name = :linear_ls, params = (n = 30, condition_number = 1.0e2))

    solve(seed) = run_method(
        GradientDescent(step_size = ArmijoLS()),
        make_problem(spec, Xoshiro(seed)),
        stop_when_any(MaxIterations(n = 10_000), DistanceToOptimal(tol = 1e-8)),
        silent_logger("repro"), Xoshiro(seed))

    r1 = solve(42)
    r2 = solve(42)

    # Same seed ⇒ identical run: same stop, same iteration count, bit-identical
    # final iterate and objective trajectory.
    @test r1.stop_reason == r2.stop_reason
    @test r1.n_iters == r2.n_iters
    @test r1.final_state.iterate.x == r2.final_state.iterate.x
    @test [e.objective for e in r1.iter_logs] == [e.objective for e in r2.iter_logs]

    # Different seed ⇒ different problem instance ⇒ different solution (the RNG is
    # genuinely wired into make_problem, not ignored).
    r3 = solve(43)
    @test r3.final_state.iterate.x != r1.final_state.iterate.x
end
