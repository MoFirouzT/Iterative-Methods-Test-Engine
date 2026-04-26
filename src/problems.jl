"""
	Layer 9 — Problem Factory

Defines the problem abstraction: data fidelity (loss), regularizers, and the
`Problem` composite. Provides a `ProblemSpec` type hierarchy for declarative,
serializable problem definition. The `make_problem` dispatch creates `Problem`
instances from specs, enabling reproducible random generation, file loading,
and registration of analytic problem families.
"""

using Random: AbstractRNG, randn
using LinearAlgebra: norm, mul!, Diagonal


# ─────────────────────────────────────────────────────────────────────────
# Data Fidelity Interface
# ─────────────────────────────────────────────────────────────────────────

"""
	abstract type DataFidelity

Base type for objective loss functions. Every concrete subtype must implement:
- `value(f::DataFidelity, x::Vector) -> Float64`
- `grad!(g::Vector, f::DataFidelity, x::Vector) -> Vector`
- `hessian_vec(f::DataFidelity, x::Vector, d::Vector) -> Vector` (optional)
"""
abstract type DataFidelity end


"""
	value(f::DataFidelity, x::Vector) -> Float64

Evaluate the data fidelity (loss) at x.
"""
function value end


"""
	grad!(g::Vector, f::DataFidelity, x::Vector) -> Vector

Compute gradient of f at x in-place into g. Returns g.
"""
function grad! end


"""
	hessian_vec(f::DataFidelity, x::Vector, d::Vector) -> Vector

Compute Hessian-vector product H(x) · d (optional; default raises error).
"""
function hessian_vec(f::DataFidelity, x::Vector, d::Vector)
	throw(MethodError(hessian_vec, (f, x, d)))
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
	value(g::Regularizer, x::Vector) -> Float64

Evaluate the regularizer at x.
"""
function value end


"""
	prox(g::Regularizer, x::Vector, γ::Float64) -> Vector

Compute the proximal operator: argmin_u { g(u) + 1/(2γ)‖u−x‖² }
"""
function prox end


# ─────────────────────────────────────────────────────────────────────────
# Composite Problem
# ─────────────────────────────────────────────────────────────────────────

"""
	Problem

A composite optimization problem: minimize f(x) + g₁(x) + g₂(x) + …

# Fields
- `f::DataFidelity` — the data fidelity (loss) term
- `gs::Vector{Regularizer}` — vector of regularizers (may be empty)
- `x0::Vector{Float64}` — initial point
- `n::Int` — problem dimension
- `meta::Dict{Symbol,Any}` — optional metadata (condition number, sparsity, etc.)
"""
struct Problem
	f::DataFidelity
	gs::Vector{Regularizer}
	x0::Vector{Float64}
	n::Int
	meta::Dict{Symbol,Any}
end


"""
	Problem(f::DataFidelity, x0::Vector)

Convenience constructor for an unregularized problem (g = ∅).
"""
function Problem(f::DataFidelity, x0::Vector{Float64})
	Problem(f, Regularizer[], x0, length(x0), Dict{Symbol,Any}())
end


"""
	Problem(f::DataFidelity, g::Regularizer, x0::Vector)

Convenience constructor for a single-regularizer problem.
"""
function Problem(f::DataFidelity, g::Regularizer, x0::Vector{Float64})
	Problem(f, Regularizer[g], x0, length(x0), Dict{Symbol,Any}())
end


"""
	objective(p::Problem, x::Vector) -> Float64

Compute total objective: f(x) + Σᵢ gᵢ(x).
"""
function objective(p::Problem, x::Vector{Float64})
	val = value(p.f, x)
	for g in p.gs
		val += value(g, x)
	end
	val
end


# ─────────────────────────────────────────────────────────────────────────
# Concrete Data Fidelity: Least Squares
# ─────────────────────────────────────────────────────────────────────────

"""
	LeastSquaresKernel

Encapsulates the data matrix A and vector b for least-squares data fidelity.
"""
struct LeastSquaresKernel
	A::Matrix{Float64}
	b::Vector{Float64}
end


"""
	LeastSquares <: DataFidelity

Least-squares data fidelity: f(x) = 0.5 ‖Ax − b‖²
"""
struct LeastSquares <: DataFidelity
	kernel::LeastSquaresKernel
end


function value(f::LeastSquares, x::Vector{Float64})
	residual = f.kernel.A * x - f.kernel.b
	0.5 * norm(residual)^2
end


function grad!(g::Vector{Float64}, f::LeastSquares, x::Vector{Float64})
	residual = f.kernel.A * x - f.kernel.b
	mul!(g, f.kernel.A', residual)
	g
end


function hessian_vec(f::LeastSquares, x::Vector{Float64}, d::Vector{Float64})
	f.kernel.A' * (f.kernel.A * d)
end


# ─────────────────────────────────────────────────────────────────────────
# Concrete Regularizers
# ─────────────────────────────────────────────────────────────────────────

"""
	L1Norm <: Regularizer

ℓ₁ regularization: g(x) = λ ‖x‖₁
"""
@kwdef struct L1Norm <: Regularizer
	λ::Float64 = 0.01
end


function value(g::L1Norm, x::Vector{Float64})
	g.λ * norm(x, 1)
end


function prox(g::L1Norm, x::Vector{Float64}, γ::Float64)
	sign.(x) .* max.(abs.(x) .- γ * g.λ, 0.0)
end


"""
	L2Norm <: Regularizer

ℓ₂ (ridge) regularization: g(x) = λ ‖x‖²
"""
@kwdef struct L2Norm <: Regularizer
	λ::Float64 = 0.01
end


function value(g::L2Norm, x::Vector{Float64})
	g.λ * norm(x)^2
end


function prox(g::L2Norm, x::Vector{Float64}, γ::Float64)
	x ./ (1.0 + 2.0 * γ * g.λ)
end


"""
	ZeroRegularizer <: Regularizer

No-op regularizer (always zero).
"""
struct ZeroRegularizer <: Regularizer end


function value(g::ZeroRegularizer, x::Vector{Float64})
	0.0
end


function prox(g::ZeroRegularizer, x::Vector{Float64}, γ::Float64)
	copy(x)
end


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
	dim::Int = 2
end


"""
	FileProblem <: ProblemSpec

Load a problem from a file using a custom loader function.
"""
@kwdef struct FileProblem <: ProblemSpec
	path::String
	loader::Function
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
	ANALYTIC_PROBLEMS[name] = builder
end


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
	ANALYTIC_PROBLEMS[spec.name](spec.params)
end


"""
	make_problem(spec::FileProblem, rng::AbstractRNG) -> Problem

Load a problem from a file using the provided loader function.
"""
function make_problem(spec::FileProblem, rng::AbstractRNG)
	spec.loader(spec.path)
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
# Convenience Registration: Built-in Analytic Problems
# ─────────────────────────────────────────────────────────────────────────

# Quadratic
register_analytic_problem!(:quadratic, (params) -> begin
	A = get(params, :A, Matrix{Float64}(I, 2, 2))
	b = get(params, :b, zeros(2))
	x0 = get(params, :x0, zeros(length(b)))
	Problem(LeastSquares(LeastSquaresKernel(A, b)), x0)
end)

# Rosenbrock-like: f(x) = sum((1 - x[i])^2 + 100(x[i+1] - x[i]^2)^2)
# For now, a simpler variant
register_analytic_problem!(:rosenbrock, (params) -> begin
	n = get(params, :dim, 2)
	x0 = ones(n) .* 0.5
	# Use a quadratic approximation or a generic dense problem
	# For now, return a simple quadratic
	A = Diagonal(ones(n))
	b = zeros(n)
	Problem(LeastSquares(LeastSquaresKernel(A, b)), x0)
end)
