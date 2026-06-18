# Iterative-Methods Test Engine

[![CI](https://github.com/MoFirouzT/Iterative-Methods-Test-Engine/actions/workflows/ci.yml/badge.svg)](https://github.com/MoFirouzT/Iterative-Methods-Test-Engine/actions/workflows/ci.yml)
[![Docs](https://github.com/MoFirouzT/Iterative-Methods-Test-Engine/actions/workflows/docs.yml/badge.svg)](https://MoFirouzT.github.io/Iterative-Methods-Test-Engine)

A Julia framework for **side-by-side comparison of iterative optimization methods** —
define a method once, sweep its variants, run them against conventional baselines on
shared problems, and measure convergence and *core* compute time under one honest,
reproducible harness.

## Design philosophy

- **Dispatch for extension, components for variation**:
  a new method, problem, or stopping criterion is a new type + a method on the relevant function — never an edit to existing code.
  Within a method, the swappable pieces (step size, descent direction,
  extrapolation, preconditioner, ...) are the components you sweep as variants.
- **Engine / content separation**:
  `src/` (the `TestEngine` module) ships only abstractions and machinery; every concrete method, problem, and component is *content* that extends the engine via `import .TestEngine`.
  Each module talks to the next through plain data structures, so any one can be read, tested, or replaced in isolation.
- **Honest timing**:
  `@core_timed` measures only the core math inside each step — logging, stopping checks, and bookkeeping are excluded — so "is it actually faster?" gets a fair answer.
- **Declarative, reproducible experiments**:
  an experiment is a serializable `ExperimentConfig`; all randomness derives from a single seed.

See **[DESIGN.md](DESIGN.md)** for a five-minute tour, or the **[architecture reference](https://MoFirouzT.github.io/Iterative-Methods-Test-Engine)** (one page per module) for the full details.

## Portfolio experiments

```bash
git clone <this repo> && cd Iterative-Methods-Test-Engine
julia --project -e 'using Pkg; Pkg.instantiate()'    # one-time: fetch deps
julia --project reproduce.jl                         # writes all figures to figures/
```

`reproduce.jl` runs each portfolio experiment in its own process and writes the five
figures into `figures/` (flagship figure first).
For a catalog of portfolio experiments — proximal-gradient on the lasso, least-squares
dimension and conditioning sweeps, preconditioning, and a nested trust-region solve —
with what each one demonstrates and all five figures, see
**[experiments/README.md](experiments/README.md)**.

![ISTA vs FISTA on the lasso: FISTA's O(1/k²) acceleration over ISTA's O(1/k), and exact support recovery](figures/lasso_ista_fista.png)

> **The flagship figure (above).** Proximal gradient on the sparse-recovery lasso
> `min ½‖Ax−b‖² + λ‖x‖₁`. *Left:* FISTA's acceleration visibly beats ISTA — a ~1000×
> smaller objective gap by iteration 50 (the `O(1/k)`-vs-`O(1/k²)` sublinear-rate
> *slope* separation is measured directly in `test/test_proximal_gradient.jl`). *Right:* the recovered
> solution lands exactly on the planted ±1 spikes, with a clean zero baseline.

## Going deeper

The engine itself developed and tested capability-by-capability across nine
stages on Rosenbrock — first figure, then trajectories, persistence, stopping
criteria, the orchestrator, multi-run, observability, cross-cutting validation.
That staged build log lives in **[experiments/stages](experiments/stages/)**.

[![GradientDescent step-size variants tracing Rosenbrock's valley toward the optimum, on log-spaced contours](experiments/stages/figures/stage2_trajectories.png)](experiments/stages/)

*Stage 2: five step-size rules navigating the banana valley. ↑ click through to the build log.*

## Run the tests

```bash
julia --project test/runtests.jl     # full unit + cross-validation suite
```

## Layout

```text
src/                 TestEngine module — abstractions + machinery only
algorithms/          content: components/ (step sizes, preconditioners, …) and one
                     directory per method (gradient_descent, proximal_gradient,
                     trust_region, preconditioned_gradient). Role (baseline vs
                     experimental) is set per experiment, not by directory.
problems/            content: rosenbrock, least_squares, lasso,
                     separable_quadratic, regularizers
experiments/         exp_*.jl portfolio scripts (+ _bootstrap.jl loader, _shared.jl helpers)
experiments/stages/  the staged Rosenbrock build log (dev scaffold, not figures)
reproduce.jl         one command → all figures
test/                runtests.jl + per-area test files
```

## Documentation

[**walkthrough.ipynb**](walkthrough.ipynb) is a complete, runnable walkthrough of *using*
the framework: define and register your own problem (ridge regression), author a custom
method (heavy-ball), set up an experiment with a variant grid and baseline/experimental
roles, run it, persist & reload, and plot the comparison. Best starting point if you want
to learn how to drive the tool.

[**docs/ARCHITECTURE.md**](docs/ARCHITECTURE.md) is the map of every document in the
project — the hosted [Documenter site](https://MoFirouzT.github.io/Iterative-Methods-Test-Engine)
for the engine reference, plus the method/problem/component design notes co-located with their source.
