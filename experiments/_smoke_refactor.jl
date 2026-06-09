# Throwaway smoke test for the engine/content split. Verifies the bootstrap
# loads, content extends the engine, and a method can be constructed + stepped.
include(joinpath(@__DIR__, "_bootstrap.jl"))
using Random

rng = Random.Xoshiro(1)

# Problem from the content-registered analytic family
prob = make_problem(AnalyticProblem(name = :rosenbrock), rng)
@assert prob.f isa RosenbrockObjective
@assert value(prob.f, [1.0, 1.0]) == 0.0          # rosenbrock global min

# LeastSquares content
ls = LeastSquares(LeastSquaresKernel([1.0 0.0; 0.0 1.0], [1.0, 2.0]))
@assert value(ls, [1.0, 2.0]) == 0.0

# Regularizer content + prox (engine generic extended by content)
@assert prox(L1Norm(0.5), [1.0, -0.2, 0.0], 1.0) == [0.5, 0.0, 0.0]

# Method construction (composes content components) + engine dispatch
m  = GradientDescent(direction = SteepestDescent(), step_size = FixedStep(α = 1e-3))
st = init_state(m, prob, rng)
@assert st isa GradientDescentState
@assert isfinite(st.metrics.objective)

println("SMOKE OK — engine/content split loads and runs")
