# Architecture reference

A map of **every document in the project** — personal table of contents.
The engine architecture reference is a **Documenter site** (one page per concern) under
[`docs/src/`](src/), published to GitHub Pages; the method / problem / component specs are
**co-located** with their source.

- **Hosted site:** <https://MoFirouzT.github.io/Iterative-Methods-Test-Engine>
- Build locally: `julia --project=docs docs/make.jl` (output in `docs/build/`).

## Architecture reference (`docs/src/`)

- [Home / reference index](src/index.md) — mathematical model & scope, the module
  catalogue, where to start
- [Design Philosophy](src/design.md) — the six guiding principles + design rationale

**Modules** — grouped into the four phases of a run (dependency order within each phase):

- *Foundations* — [Problem Interface](src/modules/problem-interface.md) ·
  [Algorithm, Core Timing & the Runner](src/modules/algorithm-core.md)
- *Run control* — [Nested Algorithms](src/modules/nested-algorithms.md) ·
  [Stopping Criteria](src/modules/stopping-criteria.md)
- *Experiment construction* — [Variant Grid Engine](src/modules/variant-grid.md) ·
  [Experiment Orchestration](src/modules/orchestration.md)
- *Observability & output* — [Logging & Verbosity](src/modules/logging.md) ·
  [Debug Mode](src/modules/debug-mode.md) ·
  [Persistence](src/modules/persistence.md) ·
  [Analysis & Plotting](src/modules/analysis-plotting.md)

**Cross-cutting & guides**

- [Convergence & Cost](src/convergence-and-cost.md) — what "converged" means per problem
  class, criterion validity, reading empirical rates, the cost model
- [Repository Internals](src/internals.md) — directory layout, data-flow diagram, key decisions
- [Extension Guide](src/extending.md) — how to add a method, problem, component, …
- [Stretch Goals](src/stretch-goals.md) — designed-for extensions not yet shipped

## Co-located content specs

**Methods**

- [Gradient Descent](../algorithms/gradient_descent/gradient_descent.md)
- [Proximal Gradient (ISTA / FISTA)](../algorithms/proximal_gradient/proximal_gradient.md)
- [Trust Region + Steihaug-CG](../algorithms/trust_region/trust_region.md)
- [PreconditionedGradient (experimental)](../algorithms/preconditioned_gradient/preconditioned_gradient.md)

**Shared components** (the method-construction vocabulary)

- [Descent Directions](../algorithms/components/descent_directions.md)
- [Step Sizes & Line Searches](../algorithms/components/step_sizes.md)
- [Extrapolation](../algorithms/components/extrapolation.md)
- [Preconditioners](../algorithms/components/preconditioners.md)

**Problems**

- [Rosenbrock](../problems/rosenbrock/rosenbrock.md)
- [Linear Least Squares](../problems/least_squares/least_squares.md)
- [Lasso (sparse recovery)](../problems/lasso/lasso.md)
- [Separable Quadratic](../problems/separable_quadratic/separable_quadratic.md)

## Experiments & build log

- [Experiments overview](../experiments/README.md) — the portfolio demonstrators + planned work
- [Development stages](../experiments/stages/README.md) — the staged Rosenbrock build log

## Project-level

- [README](../README.md) — one-command demo, layout, how to run
- [DESIGN](../DESIGN.md) — the five-minute tour
- [Copilot agent instructions](../.github/copilot-instructions.md)
