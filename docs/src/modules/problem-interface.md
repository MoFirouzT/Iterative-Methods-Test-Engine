# Problem Interface

Every problem has an **objective** `f(x)`, optionally augmented by one or more **regularizers** `gᵢ(x)`.
The total objective is `f(x) + Σ gᵢ(x)`.
Algorithms interact with the problem exclusively through this interface.
(For the problem class, the definition of an iterative method, and the framework's
scope and assumptions, see [Mathematical model & scope](../index.md).)

## Problem

```julia
struct Problem
    f     :: Objective
    gs    :: Vector{Regularizer}              # may be empty
    x0    :: Vector{Float64}                  # initial point
    n     :: Int                              # problem dimension
    meta  :: Dict{Symbol, Any}                # optional metadata
    x_opt :: Union{Nothing, Vector{Float64}}  # known optimal point; nothing if unavailable
end
```

This is the canonical, complete representation.
Everything else should be an outer constructor that forwards into this shape.

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

`x_opt` is set by the problem generator when the true minimizer is known.
Algorithms never access `x_opt` — the runner computes `state.metrics.dist_to_opt` after each step.

## Objective

```julia
abstract type Objective end

# Required dispatch for every concrete subtype:
#   grad!(g, f, x)  → write gradient into preallocated buffer g
#   value(f, x)     → scalar objective value of f at x

# Optional dispatch (only for objectives used by curvature-based methods):
#   hessian(f, x)   → a Hessian object representing ∇²f(x); see below.
#                     The engine's default raises a MethodError, so a first-order-only
#                     objective may omit it.
```

The bang form `grad!` is the required method because the framework prefers in-place work buffers for core numerical kernels.
A convenience wrapper `grad(f, x) = grad!(similar(x), f, x)` may be defined once, in `problems.jl`, for call sites that want an allocating helper.

`hessian` is **optional**: the engine ships a default that raises a `MethodError`, so an
objective consumed only by first-order methods need not define it. This mirrors the
`Hessian` interface itself, where `materialize` / `diagonal` are optional — each level of
the contract declares only the operations it can supply. Curvature-based consumers
(`CauchyStep`, `PreconditionedGradient`'s Jacobi preconditioner, `TrustRegion`) require it,
and the [problem-conformance harness](https://github.com/MoFirouzT/Iterative-Methods-Test-Engine/blob/main/test/test_problem_contract.jl)
skips its Hessian-vector check for objectives that omit it.

## Regularizer

```julia
abstract type Regularizer end

# Required dispatch:
#   value(g, x)    → scalar regularizer value
#   prox(g, x, γ)  → argmin_u { g(u) + 1/(2γ)‖u−x‖² }
```

Concrete regularizers (`L1Norm`, `L2Norm`, `ZeroRegularizer`) live in the content layer (`problems/regularizers/regularizers.jl`), not the engine.
Each is a thin wrapper that keeps a tidy public face — the `.λ` field, the `value`/`prox` contract returning a plain `Vector{Float64}` —
and delegates the actual proximal math to [`ProximalOperators.jl`](https://github.com/JuliaFirstOrder/ProximalOperators.jl).
An inner constructor builds the backend operator from `λ`, so the two cannot drift.
Because regularizers are content, swapping the backend touches only that one file; consumers (`ProximalGradient`, experiments, tests) never see it.

## Hessian

The Hessian at a point is itself an object, not a single function.
This unifies exact Hessians and structured forms (diagonal, operator/matvec, ...) under one dispatch surface, and is designed to accommodate quasi-Newton approximations (BFGS, SR1, L-BFGS) the same way (not currently shipped — see [Stretch Goals](../stretch-goals.md)).

```julia
abstract type Hessian end

# Required of every concrete Hessian:
apply(H::Hessian, d::Vector{Float64}) :: Vector{Float64}        # H · d

# Optional, defined when feasible:
materialize(H::Hessian) :: Matrix{Float64}                      # full matrix
diagonal(H::Hessian)    :: Vector{Float64}                      # diag(H), when cheaply available
```

Built-in concrete types:

```julia
# Explicit matrix; cheap apply and materialize.
# Use when the matrix is small or already computed.
struct MatrixHessian <: Hessian
    H :: Matrix{Float64}
end
apply(H::MatrixHessian, d)    = H.H * d
materialize(H::MatrixHessian) = H.H
diagonal(H::MatrixHessian)    = diag(H.H)

# Closure-based; only Hv products are available.
# Use for very large problems where forming the matrix is infeasible.
struct OperatorHessian <: Hessian
    apply_fn :: Function       # d -> H·d
    n        :: Int
end
apply(H::OperatorHessian, d)  = H.apply_fn(d)
# materialize intentionally NOT defined — calling it raises a clear MethodError.

# Pure-diagonal Hessian (or diagonal preconditioner).
# Both apply and materialize are cheap.
struct DiagonalHessian <: Hessian
    d :: Vector{Float64}
end
apply(H::DiagonalHessian, v)  = H.d .* v
materialize(H::DiagonalHessian) = Diagonal(H.d)
diagonal(H::DiagonalHessian)  = H.d
```

A method that only needs Hessian-vector products calls `apply`;
one that needs the full matrix calls `materialize`;
one that needs the diagonal (e.g. Jacobi preconditioning) calls `diagonal`.
Each concrete `Hessian` subtype declares which of these it can supply — `OperatorHessian`, for instance, offers only `apply`, and consumers detect the absence rather than forcing a materialization.

The same hierarchy is designed to admit stateful quasi-Newton approximations (BFGS, SR1,L-BFGS) — see [Stretch Goals](../stretch-goals.md).

## Oracle counting (opt-in instrumentation)

The optimization-native cost unit is the **oracle call** — one `value`, one `grad!`, or
one Hessian-vector product `apply`. Counting them lets a comparison stay fair across
methods with different per-step oracle cost (an Armijo backtrack issues several `value`
calls; a Barzilai–Borwein step issues one `grad!`). The engine provides this as a
transparent, **opt-in** wrapper — off by default, so the core-time measurement path is
byte-identical when unused.

```julia
@kwdef mutable struct OracleCounts
    n_value :: Int = 0    # value(f, ·) calls
    n_grad  :: Int = 0    # grad!(·, f, ·) calls
    n_hvp   :: Int = 0    # apply(H, ·) Hessian-vector products
end

# Parametric over the inner type, so the forwarded call stays statically dispatched —
# the only added cost is the counter increment.
struct CountingObjective{O<:Objective} <: Objective
    inner  :: O
    counts :: OracleCounts
end

struct CountingHessian{H<:Hessian} <: Hessian
    inner  :: H
    counts :: OracleCounts
end
```

`value` / `grad!` on a `CountingObjective` bump the shared `OracleCounts` and forward to
`inner`; its `hessian` returns a `CountingHessian` wrapping the inner Hessian, so each
`apply` (Hvp) is counted too. `materialize` / `diagonal` forward unchanged — and stay
absent exactly when the inner Hessian lacks them. Because the *same* `OracleCounts` is
shared by the objective and every Hessian it hands out, the tally includes work issued
inside line searches **and nested sub-solvers** — a `TrustRegion` inner CG that calls
`apply` on the (counting) outer Hessian accumulates into the same counts for free.

Activation is via `ExperimentConfig.count_oracles` (see
[Experiment Orchestration](orchestration.md)); when on, the runner wraps `problem.f`
per method and surfaces the cumulative counts in each `IterationLog`'s `extras`
(`:n_value`, `:n_grad`, `:n_hvp`). [Convergence & Cost](../convergence-and-cost.md)
covers using them as a fair-comparison axis.

> **Transparency caveat.** A *content-defined* trait on `Hessian` must forward through
> `CountingHessian` to remain transparent under counting. The one shipped example is the
> preconditioner's `_supports_diagonal`, which adds
> `_supports_diagonal(H::CountingHessian) = _supports_diagonal(H.inner)`. Engine-level
> optional ops (`materialize`, `diagonal`) already forward.

## ProblemSpec — Declarative Construction

Declarative construction is one of the framework's core principles.
A problem is not built by calling a function and capturing the result — it is *described* by a plain, serializable value, a `ProblemSpec`, carried inside the `ExperimentConfig`.
Because the description is data, **defining, running, saving, and reloading an experiment are independent operations**, and the whole run is reproducible:
the single `ExperimentConfig.seed` derives the `rng`, and `make_problem(spec, rng)` turns the spec into a concrete `Problem` deterministically.

This is also why every registry keys on a `Symbol`, never a raw `Function`:
a symbol survives a JLD2 round-trip, a closure does not.
A spec records *which* registered builder to call, not the builder itself.

```julia
abstract type ProblemSpec end
make_problem(spec::ProblemSpec, rng::AbstractRNG) :: Problem
```

The three spec families share one shape — **struct (what you declare) → registry +
`register_*!` (how content plugs in) → `make_problem` (how it is built):**

### Analytic problems

```julia
@kwdef struct AnalyticProblem <: ProblemSpec
    name   :: Symbol
    params :: NamedTuple = (;)
end

const ANALYTIC_PROBLEMS = Dict{Symbol, Function}()
register_analytic_problem!(name, builder) = (ANALYTIC_PROBLEMS[name] = builder)

make_problem(s::AnalyticProblem, rng::AbstractRNG) =
    ANALYTIC_PROBLEMS[s.name](s.params, rng)
```

Content self-registers at load time, e.g. `register_analytic_problem!(:rosenbrock, (params, rng) -> ...)`.

### Data-driven problems

```julia
@kwdef struct FileProblem <: ProblemSpec
    path        :: String
    loader_name :: Symbol      # key into FILE_LOADERS — a Symbol, so the spec stays serializable
    description :: String = ""
end

const FILE_LOADERS = Dict{Symbol, Function}()
register_file_loader!(name::Symbol, f::Function) = (FILE_LOADERS[name] = f)

make_problem(s::FileProblem, rng::AbstractRNG) = FILE_LOADERS[s.loader_name](s.path)
```

### Random problems

```julia
@kwdef struct RandomProblem <: ProblemSpec
    name   :: Symbol
    params :: NamedTuple = (;)
end

const RANDOM_GENERATORS = Dict{Symbol, Function}()
register_random_problem!(name::Symbol, gen::Function) = (RANDOM_GENERATORS[name] = gen)

make_problem(s::RandomProblem, rng::AbstractRNG) =
    RANDOM_GENERATORS[s.name](rng, s.params)
```

---
