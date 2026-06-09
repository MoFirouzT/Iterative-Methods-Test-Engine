"""
	Module 9 — Problem Factory

Defines the problem abstraction: data fidelity (loss), regularizers, and the
`Problem` composite. Provides a `ProblemSpec` type hierarchy for declarative,
serializable problem definition. The `make_problem` dispatch creates `Problem`
instances from specs, enabling reproducible random generation, file loading,
and registration of analytic problem families.
"""

using Random: AbstractRNG, randn
using LinearAlgebra: norm, mul!, Diagonal


# ─────────────────────────────────────────────────────────────────────────
# Objective Interface
# ─────────────────────────────────────────────────────────────────────────

"""
	abstract type Objective

Base type for objective functions. Every concrete subtype must implement:
- `value(f::Objective, x::Vector) -> Float64`
- `grad!(g::Vector, f::Objective, x::Vector) -> Vector{Float64}`
- `hessian(f::Objective, x::Vector) -> Hessian` (optional)
"""
abstract type Objective end


"""
	value(f, x) -> Float64

Evaluate `f` at `x`. Implemented for both `Objective` and `Regularizer` subtypes:
- `value(f::Objective, x::Vector) -> Float64` — objective value
- `value(g::Regularizer, x::Vector) -> Float64` — regularizer value
"""
function value end


"""
	grad!(g::Vector, f::Objective, x::Vector) -> Vector{Float64}

Compute gradient of f at x in-place.
"""
function grad!(g::Vector{Float64}, f::Objective, x::Vector{Float64})
	throw(MethodError(grad!, (g, f, x)))
end

# Convenience allocating wrapper used by callers that want a fresh gradient.
function grad(f::Objective, x::Vector{Float64})::Vector{Float64}
	return grad!(similar(x), f, x)
end

"""
	hessian(f::Objective, x::Vector) -> Hessian

Compute Hessian of f at x. Returns a Hessian object (optional; default raises error).
"""
function hessian(f::Objective, x::Vector)
	throw(MethodError(hessian, (f, x)))
end


# Backwards-compatible Hessian-vector product wrapper.
function hessian_vec(f::Objective, x::Vector{Float64}, d::Vector{Float64})::Vector{Float64}
	H = hessian(f, x)
	return apply(H, d)
end


# ─────────────────────────────────────────────────────────────────────────
# Regularizer Interface
# ─────────────────────────────────────────────────────────────────────────

"""
	abstract type Regularizer

Base type for regularization penalties. Every concrete subtype must implement:
- `value(g::Regularizer, x::Vector) -> Float64`
- `prox(g::Regularizer, x::Vector, γ::Float64) -> Vector` (proximal operator)
"""
abstract type Regularizer end


"""
	prox(g::Regularizer, x::Vector, γ::Float64) -> Vector

Compute the proximal operator: argmin_u { g(u) + 1/(2γ)‖u−x‖² }
"""
function prox end


# ─────────────────────────────────────────────────────────────────────────
# Hessian Interface
# ─────────────────────────────────────────────────────────────────────────

"""
	abstract type Hessian

Base type for Hessian representations. A Hessian can be:
- An explicit matrix (MatrixHessian)
- An implicit operator / closure (OperatorHessian)
- A structured form like diagonal (DiagonalHessian)

Required methods for every Hessian:
- `apply(H::Hessian, d::Vector) -> Vector` — Hessian-vector product H·d

Optional methods (implemented when feasible):
- `materialize(H::Hessian) -> Matrix{Float64}` — full matrix
- `diagonal(H::Hessian) -> Vector{Float64}` — diagonal
"""
abstract type Hessian end


"""
	apply(H::Hessian, d::Vector{Float64}) -> Vector{Float64}

Compute Hessian-vector product: H · d
"""
function apply end


"""
	materialize(H::Hessian) -> Matrix{Float64}

Return the full Hessian matrix (raises error if not available).
"""
function materialize(H::Hessian)
	throw(MethodError(materialize, (H,)))
end


"""
	diagonal(H::Hessian) -> Vector{Float64}

Return the diagonal of the Hessian (raises error if not available).
"""
function diagonal(H::Hessian)
	throw(MethodError(diagonal, (H,)))
end


"""
	struct MatrixHessian <: Hessian

Explicit dense matrix representation of the Hessian.

# Fields
- `H::Matrix{Float64}` — the Hessian matrix
"""
struct MatrixHessian <: Hessian
	H::Matrix{Float64}
end

function apply(H::MatrixHessian, d::Vector{Float64})::Vector{Float64}
	H.H * d
end

function materialize(H::MatrixHessian)::Matrix{Float64}
	H.H
end

function diagonal(H::MatrixHessian)::Vector{Float64}
	diag(H.H)
end


"""
	struct OperatorHessian <: Hessian

Closure-based Hessian for large-scale problems where forming the matrix is infeasible.
Only Hessian-vector products are available.

# Fields
- `apply_fn::Function` — function computing H · d
- `n::Int` — dimension of the problem
"""
struct OperatorHessian <: Hessian
	apply_fn::Function
	n::Int
end

function apply(H::OperatorHessian, d::Vector{Float64})::Vector{Float64}
	H.apply_fn(d)
end


"""
	struct DiagonalHessian <: Hessian

Diagonal Hessian representation: H = diag(d).

# Fields
- `d::Vector{Float64}` — diagonal entries
"""
struct DiagonalHessian <: Hessian
	d::Vector{Float64}
end

function apply(H::DiagonalHessian, d::Vector{Float64})::Vector{Float64}
	H.d .* d
end

function materialize(H::DiagonalHessian)::Matrix{Float64}
	Diagonal(H.d) |> Matrix
end

function diagonal(H::DiagonalHessian)::Vector{Float64}
	H.d
end


# ─────────────────────────────────────────────────────────────────────────
# Composite Problem
# ─────────────────────────────────────────────────────────────────────────

"""
	Problem

A composite optimization problem: minimize f(x) + g₁(x) + g₂(x) + …

# Fields
- `f::Objective` — the objective (loss) term
- `gs::Vector{Regularizer}` — vector of regularizers (may be empty)
- `x0::Vector{Float64}` — initial point
- `n::Int` — problem dimension
- `meta::Dict{Symbol,Any}` — optional metadata (condition number, sparsity, etc.)
- `x_opt::Union{Nothing,Vector{Float64}}` — known optimal point (nothing if unavailable)
"""
struct Problem
	f::Objective
	gs::Vector{Regularizer}
	x0::Vector{Float64}
	n::Int
	meta::Dict{Symbol,Any}
	x_opt::Union{Nothing,Vector{Float64}}
end


"""
	Problem(f::Objective, x0::Vector; x_opt = nothing)

Convenience constructor for an unregularized problem (g = ∅).
"""
function Problem(f::Objective, x0::Vector{Float64}; meta::Dict{Symbol,Any}=Dict{Symbol,Any}(), x_opt::Union{Nothing,Vector{Float64}}=nothing)
	Problem(f, Regularizer[], x0, length(x0), meta, x_opt)
end


"""
	Problem(f::Objective, g::Regularizer, x0::Vector; x_opt = nothing)

Convenience constructor for a single-regularizer problem.
"""
function Problem(f::Objective, g::Regularizer, x0::Vector{Float64}; meta::Dict{Symbol,Any}=Dict{Symbol,Any}(), x_opt::Union{Nothing,Vector{Float64}}=nothing)
	Problem(f, Regularizer[g], x0, length(x0), meta, x_opt)
end


"""
	Problem(f::Objective, gs::Vector{Regularizer}, x0::Vector; x_opt = nothing)

Convenience constructor for multiple regularizers.
"""
function Problem(f::Objective, gs::Vector{Regularizer}, x0::Vector{Float64}; meta::Dict{Symbol,Any}=Dict{Symbol,Any}(), x_opt::Union{Nothing,Vector{Float64}}=nothing)
	Problem(f, gs, x0, length(x0), meta, x_opt)
end


"""
	total_objective(p::Problem, x::Vector) -> Float64

Compute total objective: f(x) + Σᵢ gᵢ(x).
"""
function total_objective(p::Problem, x::Vector{Float64})
	val = value(p.f, x)
	for g in p.gs
		val += value(g, x)
	end
	val
end

"""
	objective(p::Problem, x::Vector) -> Float64

Alias for total_objective; provided for backward compatibility.
"""
const objective = total_objective


# ─────────────────────────────────────────────────────────────────────────
# Concrete objectives & regularizers live in the CONTENT layer, not here:
#   problems/least_squares/least_squares.jl  — LeastSquares (+ kernel)
#   problems/regularizers/regularizers.jl    — L1Norm / L2Norm / ZeroRegularizer
# The engine defines only the abstract `Objective` / `Regularizer` contracts
# (above) plus the generic `value` / `grad!` / `hessian` / `prox` functions.
# ─────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────
# Problem Specifications
# ─────────────────────────────────────────────────────────────────────────

"""
	abstract type ProblemSpec

Base type for problem specifications. Subclasses include:
- `AnalyticProblem` — registered families of analytic problems
- `FileProblem` — load from disk
- `RandomProblem` — generate randomly from a seed
"""
abstract type ProblemSpec end


"""
	AnalyticProblem <: ProblemSpec

A pre-defined analytic problem looked up by name.
"""
@kwdef struct AnalyticProblem <: ProblemSpec
	name::Symbol
	params::NamedTuple = (;)
end


"""
	FileProblem <: ProblemSpec

Load a problem from a file using a custom loader function.
"""
@kwdef struct FileProblem <: ProblemSpec
	path::String
	loader_name::Symbol
	description::String = ""
end


"""
	RandomProblem <: ProblemSpec

Generate a random problem using a registered generator.
"""
@kwdef struct RandomProblem <: ProblemSpec
	name::Symbol
	params::NamedTuple = (;)
end


# ─────────────────────────────────────────────────────────────────────────
# Registration and Dispatch
# ─────────────────────────────────────────────────────────────────────────

"""
	ANALYTIC_PROBLEMS

Registry of analytic problem generators. Maps Symbol → (params, rng) -> Problem.
"""
const ANALYTIC_PROBLEMS = Dict{Symbol,Function}()
const FILE_LOADERS = Dict{Symbol,Function}()


"""
	RANDOM_GENERATORS

Registry of random problem generators. Maps Symbol → (rng, params) -> Problem.
"""
const RANDOM_GENERATORS = Dict{Symbol,Function}()


"""
	register_analytic_problem!(name::Symbol, builder::Function)

Register a named analytic problem generator.
Builder signature: (params::NamedTuple) -> Problem.
"""
function register_analytic_problem!(name::Symbol, builder::Function)
	# Builder signature: (params::NamedTuple, rng::AbstractRNG) -> Problem
	ANALYTIC_PROBLEMS[name] = builder
end

register_file_loader!(name::Symbol, f::Function) = (FILE_LOADERS[name] = f)


"""
	register_random_problem!(name::Symbol, generator::Function)

Register a named random problem generator.
Generator signature: (rng::AbstractRNG, params::NamedTuple) -> Problem.
"""
function register_random_problem!(name::Symbol, generator::Function)
	RANDOM_GENERATORS[name] = generator
end


"""
	make_problem(spec::ProblemSpec, rng::AbstractRNG) -> Problem

Dispatch on problem specification type to construct a Problem instance.
"""
function make_problem(spec::ProblemSpec, rng::AbstractRNG)
	throw(MethodError(make_problem, (spec, rng)))
end


"""
	make_problem(spec::AnalyticProblem, rng::AbstractRNG) -> Problem

Create an analytic problem by looking up the registered builder.
"""
function make_problem(spec::AnalyticProblem, rng::AbstractRNG)
	if !haskey(ANALYTIC_PROBLEMS, spec.name)
		throw(KeyError("Analytic problem :$(spec.name) not registered"))
	end
	ANALYTIC_PROBLEMS[spec.name](spec.params, rng)
end


"""
	make_problem(spec::FileProblem, rng::AbstractRNG) -> Problem

Load a problem from a file using the provided loader function.
"""
function make_problem(spec::FileProblem, rng::AbstractRNG)
	if !haskey(FILE_LOADERS, spec.loader_name)
		throw(KeyError("File loader :$(spec.loader_name) not registered"))
	end
	FILE_LOADERS[spec.loader_name](spec.path)
end


"""
	make_problem(spec::RandomProblem, rng::AbstractRNG) -> Problem

Create a random problem using a registered generator.
"""
function make_problem(spec::RandomProblem, rng::AbstractRNG)
	if !haskey(RANDOM_GENERATORS, spec.name)
		throw(KeyError("Random problem :$(spec.name) not registered"))
	end
	RANDOM_GENERATORS[spec.name](rng, spec.params)
end


# ─────────────────────────────────────────────────────────────────────────
# Concrete problem families register themselves from the CONTENT layer at load
# time (e.g. problems/rosenbrock/rosenbrock.jl calls register_analytic_problem!).
# The engine ships the registration machinery above, but no concrete problem.
# ─────────────────────────────────────────────────────────────────────────
