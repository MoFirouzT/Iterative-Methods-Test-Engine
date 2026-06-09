# experiments/_bootstrap.jl
#
# Loads the engine + all content in dependency order. Experiments and tests
# `include` this instead of wiring includes by hand.
#
# The engine (src/TestEngine.jl) ships only abstractions + machinery +
# dependency-free utilities. Every concrete problem, method, regularizer, and
# algorithm component is CONTENT that extends the engine via `import .TestEngine`.
# Load order matters: engine first, then components, then the methods/problems
# that compose them. Idempotent within a process.

if !isdefined(Main, :_TESTENGINE_LOADED)
	_BOOT_ROOT = normpath(joinpath(@__DIR__, ".."))

	# 1. Engine module (abstractions + machinery only)
	include(joinpath(_BOOT_ROOT, "src", "TestEngine.jl"))
	using .TestEngine

	# 2. Algorithm components — shared method-construction vocabulary
	include(joinpath(_BOOT_ROOT, "algorithms", "components", "descent_directions.jl"))
	include(joinpath(_BOOT_ROOT, "algorithms", "components", "step_sizes.jl"))
	include(joinpath(_BOOT_ROOT, "algorithms", "components", "minor_updates.jl"))
	include(joinpath(_BOOT_ROOT, "algorithms", "components", "hessian_approx.jl"))

	# 3. Methods — compose the components above
	include(joinpath(_BOOT_ROOT, "algorithms", "conventional", "gradient_descent.jl"))

	# 4. Regularizers + problem families — register themselves with the engine on load
	include(joinpath(_BOOT_ROOT, "problems", "regularizers", "regularizers.jl"))
	include(joinpath(_BOOT_ROOT, "problems", "least_squares", "least_squares.jl"))
	include(joinpath(_BOOT_ROOT, "problems", "rosenbrock", "rosenbrock.jl"))

	_TESTENGINE_LOADED = true
end
