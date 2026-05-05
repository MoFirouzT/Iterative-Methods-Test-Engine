# experiments/smoke_test.jl
#
# smoke test: confirm the runner contract works end-to-end and
# that @core_timed measures roughly what wall-clock measures.

using Random, LinearAlgebra, Printf

include("../src/TestEngine.jl")
using .TestEngine

# ── Build problem and method ────────────────────────────────────────────────────
problem = make_problem(
    AnalyticProblem(name=:rosenbrock, params=(rho=100.0, x0=[-1.2, 1.0])),
    Xoshiro(42),
)

method   = GradientDescent(
    direction = SteepestDescent(),
    step_size = FixedStep(α=8e-4),
)

criteria = MaxIterations(n=100)

verbosity = VerbosityConfig(
    level       = SUMMARY,
    print_every = 10,
    fields      = [:iter, :objective, :gradient_norm, :step_norm],
)

logger = make_logger("GD[Fixed]", 1, "", verbosity)

# ── Run, with a wall-clock measurement around it for comparison ────────────────
wall_t0 = time_ns()
result  = run_method(method, problem, criteria, logger, Xoshiro(42))
wall_ns = time_ns() - wall_t0

# ── Report ──────────────────────────────────────────────────────────────────────
core_ns      = sum(e.core_time_ns for e in result.iter_logs)
ratio        = core_ns / wall_ns
final_x      = result.final_state.iterate.x
final_f      = result.iter_logs[end].objective
final_gnorm  = result.iter_logs[end].gradient_norm

# Monotonicity check (FixedStep on Rosenbrock with α=8e-4 should be monotone)
fs           = [e.objective for e in result.iter_logs]
n_increases  = count(diff(fs) .> 0)

println("\n────────────────────────────────────────────────────────────────")
println("Stage 0 — smoke test results")
println("────────────────────────────────────────────────────────────────")
@printf("  iters run            : %d\n",          result.n_iters)
@printf("  stop reason          : %s\n",          result.stop_reason)
@printf("  final x              : [%.6f, %.6f]\n", final_x[1], final_x[2])
@printf("  final f(x)           : %.6e\n",         final_f)
@printf("  final ‖∇f(x)‖        : %.6e\n",         final_gnorm)
@printf("  monotone? (n_inc=0)  : %s   (n_increases = %d)\n",
        n_increases == 0 ? "YES" : "NO",  n_increases)
@printf("  core time            : %.3f ms\n",      core_ns / 1e6)
@printf("  wall time            : %.3f ms\n",      wall_ns / 1e6)
@printf("  core / wall          : %.2f%%\n",       100 * ratio)
println()
@printf("  PASS criteria:\n")
@printf("    no exceptions          : ✓ (we got here)\n")
@printf("    monotone f             : %s\n", n_increases == 0 ? "✓" : "✗")
@printf("    f decreased            : %s   (f₀=%.4e, f_end=%.4e)\n",
        final_f < result.iter_logs[1].objective ? "✓" : "✗",
        result.iter_logs[1].objective, final_f)
@printf("    core ≈ wall (50%%–110%%) : %s\n",
        0.50 ≤ ratio ≤ 1.10 ? "✓" : "✗")
println("────────────────────────────────────────────────────────────────")