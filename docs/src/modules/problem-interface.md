# Problem Interface

Every problem has an **objective** `f(x)`, optionally augmented by one or more
**regularizers** `gᵢ(x)`. The total objective is `f(x) + Σ gᵢ(x)`.
Algorithms interact with the problem exclusively through this interface.

## Objective

```julia
abstract type Objective end

# Required dispatch for every concrete subtype:
#   grad!(g, f, x)  → write gradient into preallocated buffer g
#   value(f, x)    → scalar objective value of f at x
#   hessian(f, x)  → a Hessian object representing ∇²f(x); see below
```

The bang form `grad!` is the required method because the framework prefers
in-place work buffers for core numerical kernels. A convenience wrapper
`grad(f, x) = grad!(similar(x), f, x)` may be defined once, in `problems.jl`,
for call sites that want an allocating helper.

`Objective` was previously called `DataFidelity`. The rename reflects that not every
problem has data — the abstraction is about which mathematical operations are
available, not about inverse-problems vocabulary.

## Hessian

The Hessian at a point is itself an object, not a single function. This unifies exact
Hessians and structured forms (diagonal, operator/matvec, ...) under one dispatch
surface, and is designed to accommodate quasi-Newton approximations (BFGS, SR1, L-BFGS)
the same way (a stretch goal — not currently shipped; see the README).

```julia
abstract type Hessian end

# Required of every concrete Hessian:
apply(H::Hessian, d::Vector{Float64}) :: Vector{Float64}        # H · d

# Optional, defined when feasible:
materialize(H::Hessian) :: Matrix{Float64}                      # full matrix
diagonal(H::Hessian)    :: Vector{Float64}                      # diag(H)
```

Built-in concrete types:

```julia
# Explicit matrix; cheap apply and materialize. Use when the matrix is small or
# already computed.
struct MatrixHessian <: Hessian
    H :: Matrix{Float64}
end
apply(H::MatrixHessian, d)    = H.H * d
materialize(H::MatrixHessian) = H.H

# Closure-based; only Hv products are available. Use for very large problems
# where forming the matrix is infeasible.
struct OperatorHessian <: Hessian
    apply_fn :: Function       # d -> H·d
    n        :: Int
end
apply(H::OperatorHessian, d)  = H.apply_fn(d)
# materialize intentionally NOT defined — calling it raises a clear MethodError.

# Pure-diagonal Hessian (or diagonal preconditioner). Both apply and materialize
# are cheap.
struct DiagonalHessian <: Hessian
    d :: Vector{Float64}
end
apply(H::DiagonalHessian, v)  = H.d .* v
materialize(H::DiagonalHessian) = Diagonal(H.d)
diagonal(H::DiagonalHessian)  = H.d
```

A method that only needs Hessian-vector products calls `apply`; a method that needs
the full matrix calls `materialize`; a method that needs the diagonal calls
`diagonal`. Each concrete `Hessian` subtype declares which of these are available.

> **Forward-compatibility with quasi-Newton methods.** Stateful Hessian
> approximations (BFGS, SR1, L-BFGS) plug into the same hierarchy: each is a
> `Hessian` subtype with an internal `update!(H, s, y)` method called by the
> algorithm after each step. L-BFGS, in particular, never defines `materialize` —
> it stores `(s, y)` history and computes `apply` via the two-loop recursion.
> Methods consuming the Hessian are unaware of the internal mechanism.

## Regularizer

```julia
abstract type Regularizer end

# Required dispatch:
#   value(g, x)    → scalar regularizer value
#   prox(g, x, γ)  → argmin_u { g(u) + 1/(2γ)‖u−x‖² }
```

Concrete regularizers (`L1Norm`, `L2Norm`, `ZeroRegularizer`) live in the
content layer (`problems/regularizers/regularizers.jl`), not the engine. Each
is a thin wrapper that keeps a tidy public face — the `.λ` field, the
`value`/`prox` contract returning a plain `Vector{Float64}` — and delegates the
actual proximal math to [`ProximalOperators.jl`](https://github.com/JuliaFirstOrder/ProximalOperators.jl)
(`L1Norm`→`NormL1`, `L2Norm`→`SqrNormL2(2λ)`, `ZeroRegularizer`→`Zero`). An
inner constructor builds the backend operator from `λ`, so the two cannot
drift. Because regularizers are content, swapping the backend touches only that
one file; consumers (`ProximalGradient`, experiments, tests) never see it.

## Problem

```julia
struct Problem
    f     :: Objective
    gs    :: Vector{Regularizer}              # may be empty; total = f + Σ gᵢ
    x0    :: Vector{Float64}                  # initial point
    n     :: Int                              # problem dimension
    meta  :: Dict{Symbol, Any}                # optional metadata
    x_opt :: Union{Nothing, Vector{Float64}}  # known optimal point; nothing if unavailable
end
```

This is the canonical, complete representation. Everything else should be an
outer constructor that forwards into this shape.

Convenience constructors keep simple problems simple:

```julia
# Pure (non-composite) problem — no regularizer, default metadata:
Problem(f, x0)

# Pure problem with metadata and a known optimum:
Problem(f, x0; meta = Dict{Symbol,Any}(), x_opt = nothing)

# Composite with one regularizer:
Problem(f, g::Regularizer, x0; meta = Dict{Symbol,Any}(), x_opt = nothing)

# Composite with multiple regularizers:
Problem(f, gs::Vector{Regularizer}, x0; meta = Dict{Symbol,Any}(), x_opt = nothing)

# Full canonical constructor:
Problem(f, gs::Vector{Regularizer}, x0, n, meta, x_opt)
```

The total objective and gradient are framework-provided helpers — algorithms use
them rather than reaching into `gs` directly:

```julia
total_objective(p::Problem, x) = value(p.f, x) + sum(value(g, x) for g in p.gs; init=0.0)
```

`x_opt` is set by the problem generator when the true minimizer is known
(famous test problems, synthetic data with planted solution, quadratics with closed
form). Algorithms never access `x_opt` — the runner computes
`state.metrics.dist_to_opt` after each step.

## ProblemSpec — Declarative Construction

`ProblemSpec` is the serializable description used inside an `ExperimentConfig`.
`make_problem(spec, rng)` dispatches on the spec type to construct a concrete
`Problem`.

```julia
abstract type ProblemSpec end
make_problem(spec::ProblemSpec, rng::AbstractRNG) :: Problem
```

### Analytic problems

```julia
@kwdef struct AnalyticProblem <: ProblemSpec
    name   :: Symbol
    params :: NamedTuple = (;)
end

const ANALYTIC_PROBLEMS = Dict{Symbol, Function}()

register_problem!(:rosenbrock, (params, rng) -> ...)
register_problem!(:quadratic,  (params, rng) -> ...)

make_problem(s::AnalyticProblem, rng::AbstractRNG) =
    ANALYTIC_PROBLEMS[s.name](s.params, rng)
```

### Data-driven problems

Loaders are referenced by symbol so `FileProblem` stays JLD2-serializable:

```julia
const FILE_LOADERS = Dict{Symbol, Function}()
register_file_loader!(name::Symbol, f::Function) = (FILE_LOADERS[name] = f)

@kwdef struct FileProblem <: ProblemSpec
    path        :: String
    loader_name :: Symbol      # key into FILE_LOADERS
    description :: String = ""
end

make_problem(s::FileProblem, rng::AbstractRNG) = FILE_LOADERS[s.loader_name](s.path)
```

### Random problems

```julia
@kwdef struct RandomProblem <: ProblemSpec
    name   :: Symbol
    params :: NamedTuple = (;)
end

const RANDOM_GENERATORS = Dict{Symbol, Function}()
register_random_problem!(name::Symbol, gen::Function) =
    (RANDOM_GENERATORS[name] = gen)

make_problem(s::RandomProblem, rng::AbstractRNG) =
    RANDOM_GENERATORS[s.name](rng, s.params)
```

---

