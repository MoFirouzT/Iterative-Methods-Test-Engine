# Iterative-Methods Test Engine

[![CI](https://github.com/MoFirouzT/Iterative-Methods-Test-Engine/actions/workflows/ci.yml/badge.svg)](https://github.com/MoFirouzT/Iterative-Methods-Test-Engine/actions/workflows/ci.yml)
[![Docs](https://github.com/MoFirouzT/Iterative-Methods-Test-Engine/actions/workflows/docs.yml/badge.svg)](https://MoFirouzT.github.io/Iterative-Methods-Test-Engine)

**A Julia harness for side-by-side comparison of iterative optimization methods.**
Define a method once; the harness sweeps its variants, runs them against conventional
baselines on shared problems, and reports convergence and *core* compute time the same
way for every one. It is a method-comparison harness — not a unit-test framework.

![One ProximalGradient on the lasso with its extrapolation component swept — ISTA, heavy-ball, FISTA — and exact support recovery](figures/lasso_ista_fista.png)

> **Flagship — one `ProximalGradient` on the sparse-recovery lasso**, with only its
> **extrapolation component swept**: the project's core idea in one figure.
> `min ½‖Ax−b‖² + λ‖x‖₁`. No momentum (ISTA) → fixed momentum (heavy-ball) → Nesterov's
> schedule (FISTA) orders the convergence (left); the recovered solution lands exactly on
> the planted ±1 spikes (right). Full annotation in **[DESIGN.md](DESIGN.md)**.

## See how it works

[**walkthrough.ipynb**](walkthrough.ipynb) is a complete, runnable walkthrough of *using* the framework:
define and register your own problem (ridge regression), author a custom method (heavy-ball), set up an experiment with a variant grid and baseline/experimental roles, run it, persist & reload, and plot the comparison —
the best place to start if you want to see the tool driven end to end.

## Design philosophy

The engine rests on six principles; these four do most of the visible work
(the full set is in [DESIGN.md](DESIGN.md)):

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
figures into `figures/` (the flagship, shown above, first).
For a catalog of portfolio experiments — proximal-gradient on the lasso, least-squares
dimension and conditioning sweeps, preconditioning, and a nested trust-region solve —
with what each one demonstrates and all five figures, see
**[experiments/README.md](experiments/README.md)**.

## Going deeper

The engine was built and validated capability-by-capability across nine stages on
Rosenbrock — first a convergence figure, then trajectories, persistence, stopping
criteria, the orchestrator, multi-run aggregation, observability, and cross-cutting
validation. That staged build log — with its own trajectory figures — lives in
**[experiments/stages](experiments/stages/)**.

## Tests & validation

```bash
julia --project test/runtests.jl     # full unit + cross-validation suite
```

Converged solutions are **externally cross-checked** against `A\b`,
[Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) (GradientDescent / LBFGS) and
[ProximalAlgorithms.jl](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl)
(ForwardBackward / FastForwardBackward) — see
[`test/test_external_validation.jl`](test/test_external_validation.jl).

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

[**docs/ARCHITECTURE.md**](docs/ARCHITECTURE.md) is the map of every document in the
project — the hosted [Documenter site](https://MoFirouzT.github.io/Iterative-Methods-Test-Engine)
for the engine reference, plus the method/problem/component design notes co-located with their source.
