# Extension Guide

## Adding a new method

Create `algorithms/<group>/<name>/<name>.jl`, where `<group>` is `conventional` or
`experimental` — that directory split is purely organizational; a method's
comparison role is **not** encoded in its type. Define the struct as a subtype of
`IterativeMethod`, implement `init_state` (using `IterateGroup`, `MetricsGroup`,
`TimingGroup`), `step!` with the full signature
`step!(method, state, problem, iter, logger, rng)` (use
`@core_timed state begin ... end` around the kernel), and `extract_log_entry`.
Then declare its **role** when you assemble the experiment: list it under
`baseline_methods` or `experimental_methods` in an `ExperimentConfig`, or build a
`VariantGrid` of it with the matching `role` (`:baseline` / `:experimental`). The
runner, logger, stopping criteria, and plots all pick it up automatically.

Before writing any code, create `algorithms/<group>/<name>/<name>.md`
following the structure of `algorithms/conventional/gradient_descent/gradient_descent.md`:
problem statement, iteration formula, Julia structs, `init_state` / `step!` /
`extract_log_entry` contracts, and a full variable mapping table. If the method
has pluggable components (directions, step-size rules, ...), give each a dedicated
`<component>.md` file in the components directory.

## Adding an algorithm that uses a sub-algorithm

Embed a `SubRunConfig{M}` field in the outer algorithm struct (typed concretely for
type stability). Call `run_sub_method` inside `step!`, forwarding the `logger` and
`rng` that the runner injected. See [Nested Algorithm Infrastructure](@ref) for the schematic.

## Adding a new stopping criterion

Add a struct subtyping `StoppingCriterion` and a `should_stop` method to `stopping.jl`.
It can immediately be used standalone or composed inside `CompositeCriterion`. Access
state quantities via `state.metrics.*`.

```julia
@kwdef struct RelativeObjectiveDecrease <: StoppingCriterion
    tol :: Float64 = 1e-6
end

function should_stop(c::RelativeObjectiveDecrease, state, iter, logger)
    iter < 2 && return (false, :none)
    prev = logger.iter_logs[end-1].objective
    curr = logger.iter_logs[end].objective
    rel  = abs(prev - curr) / max(abs(prev), 1.0)
    rel <= c.tol ? (true, :relative_obj_converged) : (false, :none)
end
```

## Adding a warm-up

**Iterative warm-up** (run a cheap method to find a better x0):

```julia
config = ExperimentConfig(
    ...,
    warmup = IterativeWarmup(
        method   = GradientDescent(step_size=FixedStep(α=0.1)),
        criteria = MaxIterations(n=100),
    ),
)
```

**Function warm-up** (closed-form initialization):

```julia
register_warmup!(:custom_init, (problem, rng) -> begin
    # ... return Vector{Float64}
end)

config = ExperimentConfig(..., warmup = FunctionWarmup(:custom_init))
```

## Adding a known optimal point to a problem

When registering a problem generator, embed `x_opt` in the returned `Problem`:

```julia
register_random_problem!(:quadratic, (rng, p) -> begin
    A     = randn(rng, p.n, p.n); A = A'A + p.μ * I    # positive definite
    b     = randn(rng, p.n)
    x_opt = A \ b                                       # known minimizer
    x0    = zeros(p.n)
    Problem(QuadraticObjective(A, b), x0; x_opt = x_opt)
end)
```

`DistanceToOptimal` and the `dist_to_opt` log column then activate automatically.

## Adding a new debug check

Add a struct subtyping `DebugCheck` and a `debug_check!` method to `debug.jl`.

```julia
@kwdef struct CheckHessianPositiveDefiniteness <: DebugCheck
    sample_directions :: Int = 5
end

function debug_check!(c::CheckHessianPositiveDefiniteness, cfg, state, problem,
                      entry, prev, iter)
    H = hessian(problem.f, state.iterate.x)
    for _ in 1:c.sample_directions
        d = randn(length(state.iterate.x))
        curvature = dot(d, apply(H, d))
        if curvature < 0
            trigger_debug!(cfg, iter,
                "Hessian is not positive definite: d'Hd = $(curvature)")
            return
        end
    end
end
```

## Adding a new problem

Create `problems/<name>/<name>.md` following `problems/rosenbrock/rosenbrock.md`:

1. State the optimization problem in standard form.
2. Derive and document $\nabla f$ and the Hessian (full matrix or H·d).
3. Provide the known minimizer `x_opt` if it exists analytically; document why it
   is `nothing` if it does not.
4. Include the variable mapping table and the `register_analytic_problem!` /
   `register_random_problem!` call.

Then implement `problems/<name>/<name>.jl` with `value`, `grad!`, `hessian`
(returning a `Hessian` object), and the registration call. The `<name>.md` is the
contract; the `.jl` is the implementation.

Finally, register the problem with the conformance harness: add one entry to
`CONFORMANCE_SPECS` in `test/test_problem_contract.jl` — an example spec for your
family, with `minimizer = true` only when `x_opt` is the true minimizer (use
`false` if `x_opt` is absent or a planted reference, as with the lasso signal).
The shared `check_problem_contract` then validates your gradient against finite
differences, your Hessian-vector products, and each regularizer's `prox`
automatically. A completeness guard fails the suite if a registered problem has no
such entry, so this step is mandatory — but it is a single line, not a bespoke
test.

## Adding a new logged field

Add the field to `IterationLog` or to `extras` in `extract_log_entry`. The CSV
sidecar picks up all `extras` keys automatically via `to_dataframe()`.

## Adding a new step-size rule (or line search)

See `step_sizes.md` for the full template. In summary:

1. Decide whether the rule is a closed-form `StepSize` or a genuine `LineSearch`
   (subtyping `LineSearch` enables future code that targets searches specifically).
2. Add the concrete struct and `compute_step_size` method to
   `components/step_sizes.jl`.
3. Wrap only the mathematically core operations in `@core_timed state`.
4. Add the abbreviation via `register_abbreviation!` if you want a custom short name.
5. Pass the new rule as the `step_size` field in `GradientDescent(...)`.

No changes to `step!`, the runner, logging, or stopping criteria are required.

## Adding a fixed color for a method across all plots

```julia
register_method_color!("GradientDescent[step_size=Armijo]", "#0072B2")
```

Call once at session startup. Per-plot `method_styles` in `PlotSpec` take precedence
over the registry when both are present.

## Plotting across multiple experiments

```julia
df1 = to_dataframe(load_experiment("logs/20260417/001/")) |> d -> @transform(d, :exp = "exp1")
df2 = to_dataframe(load_experiment("logs/20260417/002/")) |> d -> @transform(d, :exp = "exp2")
df  = vcat(df1, df2)
PlotSpec(data=df, group_by=:exp, ...)
```
