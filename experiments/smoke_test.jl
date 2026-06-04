# experiments/smoke_test.jl
#
# smoke test: confirm the runner contract works end-to-end on 2D Rosenbrock.
# The @core_timed-vs-wall-clock ratio check lives in Stage 4, where 20_000
# iters amortize per-iter scaffolding enough for the comparison to be a real
# signal. At Stage 0's 100 iters the kernel is below the noise floor.

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

result  = run_method(method, problem, criteria, logger, Xoshiro(42))

# ── Report ──────────────────────────────────────────────────────────────────────
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
println()
@printf("  PASS criteria:\n")
@printf("    no exceptions          : ✓ (we got here)\n")
@printf("    monotone f             : %s\n", n_increases == 0 ? "✓" : "✗")
@printf("    f decreased            : %s   (f₀=%.4e, f_end=%.4e)\n",
        final_f < result.iter_logs[1].objective ? "✓" : "✗",
        result.iter_logs[1].objective, final_f)
println("────────────────────────────────────────────────────────────────")
