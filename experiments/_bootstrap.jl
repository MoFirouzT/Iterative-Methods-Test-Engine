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
#
# To add new content (a problem, method, or component): create its file with an
# `import .TestEngine: <names it extends>` header (see e.g. problems/rosenbrock/
# rosenbrock.jl), then add an `include(...)` line in the matching section below.

if !isdefined(Main, :_TESTENGINE_LOADED)
	_BOOT_ROOT = normpath(joinpath(@__DIR__, ".."))

	# 1. Engine module (abstractions + machinery only)
	include(joinpath(_BOOT_ROOT, "src", "TestEngine.jl"))
	using .TestEngine

	# 2. Algorithm components — shared method-construction vocabulary
	include(joinpath(_BOOT_ROOT, "algorithms", "components", "descent_directions.jl"))
	include(joinpath(_BOOT_ROOT, "algorithms", "components", "step_sizes.jl"))
	include(joinpath(_BOOT_ROOT, "algorithms", "components", "minor_updates.jl"))
	include(joinpath(_BOOT_ROOT, "algorithms", "components", "preconditioners.jl"))

	# 3. Methods — compose the components above
	include(joinpath(_BOOT_ROOT, "algorithms", "conventional", "gradient_descent.jl"))
	include(joinpath(_BOOT_ROOT, "algorithms", "conventional", "proximal_gradient", "proximal_gradient.jl"))
	include(joinpath(_BOOT_ROOT, "algorithms", "experimental", "preconditioned_gradient", "preconditioned_gradient.jl"))

	# 4. Regularizers + problem families — register themselves with the engine on load
	include(joinpath(_BOOT_ROOT, "problems", "regularizers", "regularizers.jl"))
	include(joinpath(_BOOT_ROOT, "problems", "least_squares", "least_squares.jl"))
	include(joinpath(_BOOT_ROOT, "problems", "lasso", "lasso.jl"))
	include(joinpath(_BOOT_ROOT, "problems", "separable_quadratic", "separable_quadratic.jl"))
	include(joinpath(_BOOT_ROOT, "problems", "rosenbrock", "rosenbrock.jl"))

	_TESTENGINE_LOADED = true
end
