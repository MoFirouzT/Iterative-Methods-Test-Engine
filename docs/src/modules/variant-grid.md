# Variant Grid Engine

This module models each **dimension of variation** as a typed component,
then constructs all valid combinations automatically. Grids work uniformly over
any `IterativeMethod` — conventional or experimental — and the routing into the
right bucket happens later in `resolve_methods`.

## Component Abstractions

Each variation axis is an abstract type with concrete implementations. The
framework ships several pluggable component hierarchies: **descent direction**
and **step-size rule** (both consumed by `GradientDescent`), and a
**minor-update** slot (consumed by `ProximalGradient` — see below):

```julia
# Descent direction (see descent_directions.md)
abstract type DescentDirection end
struct SteepestDescent <: DescentDirection end

# Step-size rule (see step_sizes.md)
abstract type StepSize end
abstract type LineSearch <: StepSize end       # subset that performs an actual 1D search

# Direct step-size rules (closed-form α from current state):
@kwdef struct FixedStep       <: StepSize ; α :: Float64 = 1e-3 end
@kwdef struct CauchyStep      <: StepSize ; fallback_α :: Float64 = 1e-3
                                            ε_denom    :: Float64 = 1e-14 end
@kwdef struct BarzilaiBorwein <: StepSize ; variant    :: Symbol  = :BB1
                                            fallback_α :: Float64 = 1e-3
                                            ε_denom    :: Float64 = 1e-14 end

# Genuine line searches (test trial points until a sufficient-decrease condition holds):
@kwdef struct ArmijoLS <: LineSearch
    α₀       :: Float64 = 1.0
    β        :: Float64 = 0.5
    c₁       :: Float64 = 1e-4
    max_iter :: Int     = 50
end
```

```julia
# Minor update — post-step correction / extrapolation (see minor_updates.jl)
abstract type MinorUpdate end
struct NoMinorUpdate <: MinorUpdate end           # plain method (no extrapolation)
@kwdef struct NesterovStep <: MinorUpdate end     # FISTA momentum
@kwdef struct MomentumStep <: MinorUpdate; α::Float64 = 0.1 end   # heavy-ball

# Consumed through two functions a method calls per step:
#   extrapolate(mu, x, x_prev, t)  → gradient-evaluation point y
#   advance_momentum(mu, t)        → next momentum parameter t
```

`ProximalGradient` (the composite-objective method) crosses a **step-size** axis
with this **minor-update** axis: `NoMinorUpdate` ⇒ ISTA, `NesterovStep` ⇒ FISTA.
With a zero/absent regularizer it reduces to (accelerated) gradient descent, so
the same method tells the smooth-acceleration story for free. See
`algorithms/conventional/proximal_gradient/proximal_gradient.md`.

```julia
# Preconditioner — supplies M⁻¹ for d = −M⁻¹∇f (see preconditioners.jl)
abstract type Preconditioner end
struct IdentityPreconditioner <: Preconditioner end   # M⁻¹ = I  ⇒ plain GD
struct JacobiPreconditioner   <: Preconditioner end   # M⁻¹ = diag(∇²f)⁻¹

precondition(M, g, problem, x) -> M⁻¹·g
```

`PreconditionedGradient` (the *experimental* method) crosses this
**preconditioner** axis with a step-size axis. `JacobiPreconditioner` reads
`diagonal(hessian(f, x))` — so it is exact Newton where the Hessian is diagonal
(`DiagonalHessian`), works on any `diagonal`-capable Hessian (`MatrixHessian`),
and is *correctly inapplicable* on an `OperatorHessian` (clean `ArgumentError`,
no silent fallback). This is the "each `Hessian` declares which operations it
supports" contract made operational. See `components/preconditioners.md`.

Other component hierarchies (Hessian approximations, ...) can be added later.
They follow the same pattern: an abstract type, concrete subtypes, a single
dispatched function on the abstract.

## VariantAxis and VariantGrid

```julia
# One axis of variation: a parameter name, a list of values, and a label per value
struct VariantAxis
    param  :: Symbol
    values :: Vector{Any}
    labels :: Vector{String}
end

# Convenience constructor using Pair syntax: value => "label"
function VariantAxis(param::Symbol, labeled_values::Pair...)
    VariantAxis(param,
        [p.first  for p in labeled_values],
        [p.second for p in labeled_values])
end

# The full grid: axes + a builder function + optional exclusion filters
@kwdef struct VariantGrid
    base_name     :: String
    axes          :: Vector{VariantAxis}
    builder       :: Function          # (;param=value, ...) -> IterativeMethod
    filters       :: Vector{Function}  = []  # [(combo::NamedTuple) -> Bool]
    shared_params :: NamedTuple        = (;)
end
```

> **Builder return type is `IterativeMethod`, not `ExperimentalMethod`.** Grids
> can produce conventional methods (e.g. exploring step-size variants of
> `GradientDescent`) just as readily as experimental ones. `resolve_methods` (Module 7)
> sorts each produced method into the conventional or experimental bucket based on
> its concrete type.

## Grid Expansion

`expand(grid)` takes the Cartesian product of all axes, applies filters, builds each
method instance, and attaches auto-generated names. This function operates on a
**single** `VariantGrid` and is independently callable and unit-testable.

```julia
struct VariantSpec
    name       :: String              # full human-readable name
    short_name :: String              # compact legend label
    params     :: NamedTuple          # the exact parameter combination
    method     :: IterativeMethod     # ready-to-run instance
end

function expand(grid::VariantGrid)::Vector{VariantSpec}
    # Cartesian product → filter → build → name
end
```

## Naming Convention

| Format | Example |
|--------|---------|
| Full (logging, filenames) | `GradientDescent[step_size=Armijo]` |
| Short (plot legends) | `GD/Arm` |

```julia
const ABBREVIATIONS = Dict(
    "GradientDescent" => "GD",
    "SteepestDescent" => "SD",
    "FixedStep"       => "Fix",
    "CauchyStep"      => "Cau",
    "BarzilaiBorwein" => "BB",
    "ArmijoLS"        => "Arm",
)

# Register abbreviations for user-defined components
register_abbreviation!(long::String, short::String) = (ABBREVIATIONS[long] = short)
```

`register_abbreviation!` must be called for any user-defined component name before
`expand` is first called. It is documented in the Extension Guide.

## Defining a Grid (Usage Example)

```julia
step_size_axis = VariantAxis(:step_size,
    FixedStep(α=1e-3)              => "Fixed",
    ArmijoLS(α₀=1.0, β=0.5, c₁=1e-4) => "Armijo",
    BarzilaiBorwein(variant=:BB1)  => "BB1",
    BarzilaiBorwein(variant=:BB2)  => "BB2",
    CauchyStep()                   => "Cauchy",
)

grid = VariantGrid(
    base_name = "GradientDescent",
    axes      = [step_size_axis],
    builder   = (; step_size, kwargs...) ->
                    GradientDescent(direction=SteepestDescent(), step_size=step_size),
)
# 5 combinations, all conventional GradientDescent variants.
```

---

