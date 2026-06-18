# Variant Grid Engine

This module models each **dimension of variation** as a typed component, then constructs all valid combinations automatically.
Grids work uniformly over any `IterativeMethod`; each grid declares a `role` (`:baseline` or `:experimental`) that `resolve_methods` later uses to route its variants into the right comparison bucket.

## Component Abstractions (defined in content)

Each variation axis is an abstract type with concrete subtypes plus a single function dispatched on the abstract.
**These hierarchies live in content, not the engine** — the grid operates over them generically; only the example below is inlined, each axis is documented with its own component. The step-size axis consumed by `GradientDescent`:

```julia
abstract type StepSize end
abstract type LineSearch <: StepSize end          # subset that performs an actual 1D search
@kwdef struct FixedStep <: StepSize ; α::Float64 = 1e-3 end
@kwdef struct ArmijoLS  <: LineSearch ; α₀::Float64 = 1.0; β::Float64 = 0.5
                                        c₁::Float64 = 1e-4; max_iter::Int = 50 end
# apply(step, ...) — dispatched on the abstract type — returns α
```

The other shipped axes follow the same recipe:

- **Descent direction** (`DescentDirection`, e.g. `SteepestDescent`) — also consumed by `GradientDescent`.
- **Extrapolation slot** (`Extrapolation`) — the post-step extrapolation a method applies between steps; a first-class axis.
`ProximalGradient` crosses it with the step-size axis: `NoExtrapolation` ⇒ ISTA, `NesterovStep` ⇒ FISTA.
With a zero/absent regularizer the method reduces to (accelerated) gradient descent, telling the smooth-acceleration story for free.
See [proximal_gradient.md](https://github.com/MoFirouzT/Iterative-Methods-Test-Engine/blob/main/algorithms/proximal_gradient/proximal_gradient.md).
- **Preconditioner** (`Preconditioner`) — supplies `M⁻¹` for `d = −M⁻¹∇f`, crossed with the step-size axis by the *experimental* `PreconditionedGradient`.
`JacobiPreconditioner` is exact Newton on a diagonal Hessian, works on any `diagonal`-capable Hessian, and is *correctly inapplicable* on an operator-only one (clean `ArgumentError`, no silent fallback) — the "each `Hessian` declares which operations it supports" contract made operational.
See [preconditioners.md](https://github.com/MoFirouzT/Iterative-Methods-Test-Engine/blob/main/algorithms/components/preconditioners.md).

New hierarchies (Hessian approximations, ...) plug in the same way: an abstract type, concrete subtypes, one dispatched function.

## VariantAxis and VariantGrid

These two types are the engine's actual contribution.
A **`VariantAxis`** is one dimension of variation:
a builder parameter, the values it ranges over, and a short label per value (the `value => "label"` constructor pairs them).
A **`VariantGrid`** assembles axes into a sweep — a `builder` that turns one parameter combination into an `IterativeMethod`, optional `filters` that drop invalid combinations before they are built, and `shared_params` held fixed across the whole grid.

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
    role          :: Symbol            = :experimental  # :baseline or :experimental
end
```

> **Builder return type is `IterativeMethod`.**
> A grid can serve as a family of baselines or of experimental variants equally well.
> `resolve_methods` (see [Experiment Orchestration](@ref)) routes a grid's variants into the baseline or experimental bucket by the grid's `role` — never by method type.

## Grid Expansion

`expand(grid)` takes the Cartesian product of all axes, applies filters, builds each method instance, and attaches auto-generated names.
This function operates on a **single** `VariantGrid` and is independently callable and unit-testable.

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
| -------- | --------- |
| Full (logging, filenames) | `GradientDescent[step_size=Armijo]` |
| Short (plot legends) | `GD/Arm` |

```julia
# The engine ships only the registry mechanism — it holds no concrete
# method/component vocabulary (generic entries only):
const ABBREVIATIONS = Dict{String,String}("None" => "∅")
abbreviate(value)   = get(ABBREVIATIONS, string(value), string(value))
register_abbreviation!(long, short) = (ABBREVIATIONS[long] = short)
```

Concrete abbreviations are **content**:
each component registers its own on load — `step_sizes.jl` registers `"ArmijoLS" => "Arm"`, `extrapolation.jl` registers `"NesterovStep" => "Nest"` and `"MomentumStep" => "Mom"` — and a method or experiment registers its own method name (e.g. `register_abbreviation!("GradientDescent", "GD")`).
`register_abbreviation!` must be called for any name before `expand` is first invoked;
it is documented in the Extension Guide.

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
    role      = :baseline,
)
# 5 baseline GradientDescent variants (routed to the baseline bucket by role).
```

---
