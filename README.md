# Iterative-Methods Test Engine

[![CI](https://github.com/MoFirouzT/Iterative-Methods-Test-Engine/actions/workflows/ci.yml/badge.svg)](https://github.com/MoFirouzT/Iterative-Methods-Test-Engine/actions/workflows/ci.yml)

A Julia framework for **side-by-side comparison of iterative optimization methods** —
define a method once, sweep its variants, run them against conventional baselines on
shared problems, and measure convergence and *core* compute time under one honest,
reproducible harness.

![ISTA vs FISTA on the lasso: FISTA's O(1/k²) acceleration over ISTA's O(1/k), and exact support recovery](figures/lasso_ista_fista.png)

> **The money figure (above).** Proximal gradient on the sparse-recovery lasso
> `min ½‖Ax−b‖² + λ‖x‖₁`. *Left:* FISTA's `O(1/k²)` convergence visibly beats ISTA's
> `O(1/k)` — a ~1000× smaller objective gap by iteration 50. *Right:* the recovered
> solution lands exactly on the planted ±1 spikes, with a clean zero baseline.
> Reproduce it (and four more figures) with **one command** below.

## Reproduce everything in one command

```bash
git clone <this repo> && cd Iterative-Methods-Test-Engine
julia --project -e 'using Pkg; Pkg.instantiate()'   # one-time: fetch deps
julia --project scripts/reproduce.jl                 # writes all figures to figures/
```

`reproduce.jl` runs each portfolio experiment in its own process and writes the five
figures into `figures/` (money figure first).

## What it demonstrates

Each capability has exactly one clean, working demonstrator you can watch run — no
method zoo. The five shipped experiments:

| Capability | Demonstrator | Figure |
|---|---|---|
| Composite `f + g`, `prox` dispatch, Nesterov acceleration | `ProximalGradient` (ISTA → FISTA) on the lasso | `lasso_ista_fista.png` |
| A scalable second problem family; matrix-free `OperatorHessian`; dimension scaling; the timing pillar as a real signal | Linear least squares + dimension sweep | `ls1_dimension.png` |
| Conditioning controls the rate: `O(κ)` vs `O(√κ)` | Least squares + conditioning sweep | `ls2_conditioning.png` |
| The signature workflow: define one experimental method, sweep it in a `VariantGrid`, route experimental vs conventional buckets; `DiagonalHessian` + Jacobi preconditioning ≈ Newton | `PreconditionedGradient` swept vs a baseline | `precond1_grid.png` |
| Nested optimization: an inner solver minimizing a local model inside each outer step (`run_sub_method` / sub-log attachment / inner-vs-outer core time) | `TrustRegion` with a Steihaug-CG inner solve | `tr1_trust_region.png` |

Correctness is **externally cross-checked**: converged solutions are matched against
`A\b`, [`Optim.jl`](https://github.com/JuliaNLSolvers/Optim.jl) (GradientDescent / LBFGS),
and [`ProximalAlgorithms.jl`](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl)
(ForwardBackward / FastForwardBackward) — see `test/test_external_validation.jl`.

## Design in one breath

- **Multiple dispatch over hierarchies** — every method, component (step size, descent
  direction, minor update, preconditioner), stopping criterion, and problem is a dispatch
  point; adding a variant never edits existing code.
- **Engine / content separation** — `src/` (the `TestEngine` module) ships only
  abstractions and machinery; every concrete method, problem, and component is *content*
  that extends the engine via `import .TestEngine`.
- **Scientific measurement discipline** — `@core_timed` measures only the core math inside
  each step; logging, stopping checks, and bookkeeping are excluded. At `n = 1000` the
  least-squares matvec dominates and `core_time/wall_time` lands at ~98%; at `n = 10` it is
  ~2% (the kernel is below the timing floor). That honest ratio is itself a result.
- **Declarative, reproducible experiments** — an experiment is a serializable
  `ExperimentConfig`; all randomness derives from a single seed.
- **Spec-driven** — every problem and method ships a co-located `.md` spec.

See **[DESIGN.md](DESIGN.md)** for a five-minute tour, or
**[docs/architecture.md](docs/architecture.md)** for the full maintainer reference.

## Run the tests

```bash
julia --project test/runtests.jl     # 200 tests
```

## Layout

```
src/                 TestEngine module — abstractions + machinery only
algorithms/          content: components/ (step sizes, preconditioners, …),
                     conventional/ (GradientDescent, ProximalGradient, TrustRegion),
                     experimental/ (PreconditionedGradient)
problems/            content: rosenbrock, least_squares, lasso,
                     separable_quadratic, regularizers
experiments/         exp_*.jl scripts (+ _bootstrap.jl loader, _shared.jl helpers)
scripts/reproduce.jl one command → all figures
test/                runtests.jl + per-area test files
```

## Stretch goals (scaffolding, not shipped)

The architecture is built to extend; these are deliberately *not* implemented yet, and
are called out here rather than left as empty exported stubs:

- **Quasi-Newton** (BFGS / L-BFGS / SR1) — would plug into the `Hessian` hierarchy with an
  internal secant `update!`; the representation contract is already exercised by
  `OperatorHessian` and `DiagonalHessian`.

CI runs the full suite on every push via [GitHub Actions](.github/workflows/ci.yml)
(`julia --project test/runtests.jl`, 200 tests). `Manifest.toml` is committed, so CI and
`scripts/reproduce.jl` install the exact pinned dependency versions — the build is
reproducible, not "latest that happens to resolve."
