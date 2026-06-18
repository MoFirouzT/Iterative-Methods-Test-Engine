# Iterative-Methods Test Engine

A Julia framework for **side-by-side comparison of iterative optimization methods** —
define your own experimental method once, sweep its variants, and run them against
conventional baselines on shared problems under one honest, reproducible harness that
measures *core* compute time, not bookkeeping.

This is the **architecture reference** (the contributor / maintainer view). If you only
want to run it, start with the
[README](https://github.com/MoFirouzT/Iterative-Methods-Test-Engine#readme) for the
one-command demo, or the
[five-minute tour](https://github.com/MoFirouzT/Iterative-Methods-Test-Engine/blob/main/DESIGN.md).

![One ProximalGradient on the lasso with its extrapolation component swept — ISTA, heavy-ball, FISTA — and exact support recovery](https://raw.githubusercontent.com/MoFirouzT/Iterative-Methods-Test-Engine/main/figures/lasso_ista_fista.png)

## Mathematical model & scope

The engine solves finite-dimensional optimization problems in **composite form**:

    min_x  f(x) + Σᵢ gᵢ(x),    x ∈ ℝⁿ

`f` is the smooth **objective** — its gradient is always available, its Hessian available
to methods that use curvature — and each `gᵢ` is a **simple regularizer** reached through
its proximal operator `prox`. With no regularizer this is the plain smooth problem
`min_x f(x)`.

An **iterative method** generates, from a starting point `x₀`, a sequence
`x₀, x₁, x₂, …` with `x_{k+1} = step(x_k)`, run until a stopping criterion fires. The
engine runs such methods — and their swept variants, against baselines — under one
harness, reporting convergence and *core* compute time identically for each.

**Assumptions.**

| Assumption | Detail |
| --- | --- |
| Smooth `f` | first-order oracle (`value`, `grad!`) always available; second-order (`hessian`) used by curvature-based methods |
| Simple `gᵢ` | each nonsmooth term is reached only through `prox`, never its gradient |
| Finite-dimensional | `x ∈ ℝⁿ`, dense vectors |
| Deterministic by default | randomness enters only via problem generation, warm-up, or explicitly stochastic steps — all seed-derived |

**In scope today:** unconstrained smooth minimization; composite problems with **at most
one** prox-able regularizer (a sum of several nonsmooth terms needs operator splitting);
deterministic first- and second-order methods, including a nested trust-region method
with an inner solver.

**Out of scope (by design, for now):** general constrained optimization beyond what an
indicator-function `prox` expresses; stochastic / mini-batch methods; multiple
simultaneous nonsmooth terms — each reachable without breaking an abstraction (see
[Stretch Goals](stretch-goals.md) and the experiment backlog).

## Design in one breath

Six principles shape the engine — **multiple dispatch over hierarchies**, **engine /
content separation**, **declarative experiment definition**, **scientific measurement
discipline**, **claims demonstrated, not asserted**, and **separation of concerns
across modules**. The [Design Philosophy](design.md) page is the canonical reference
for what each means and the invariants they imply.

## How this reference is organized

The engine is a hub-and-spoke of cohesive **modules** (the orchestrator is the hub).

**Start with orientation:** [Repository Internals](internals.md) carries the directory
layout and the end-to-end **data-flow diagram** — definition → execution → persistence →
analysis on a single page. Skim that first; then read the module contracts below.

- **[Design Philosophy](design.md)** — the six guiding principles and the design rationale.

**Modules** — one page per concern, grouped into the phases of a run. Within each phase
the pages follow dependency order, so they read top-down with no forward references.

**1 · Foundations** — the abstractions for defining and running a single method.

| Module | What it owns |
| --- | --- |
| [Problem Interface](modules/problem-interface.md) | `Objective` / `Regularizer` / `Hessian` / `Problem`, the problem factory |
| [Algorithm, Core Timing & the Runner](modules/algorithm-core.md) | method type hierarchy, composable state groups, `@core_timed`, the run loop |

**2 · Run control** — what governs a single method's execution to termination.

| Module | What it owns |
| --- | --- |
| [Nested Algorithms](modules/nested-algorithms.md) | `SubRunConfig` / `run_sub_method` — a sub-solver inside a step; extends the runner |
| [Stopping Criteria](modules/stopping-criteria.md) | composable termination via `should_stop` dispatch |

**3 · Experiment construction** — turning methods into a reproducible comparison.

| Module | What it owns |
| --- | --- |
| [Variant Grid Engine](modules/variant-grid.md) | typed components, Cartesian grid expansion, auto-naming |
| [Experiment Orchestration](modules/orchestration.md) | config, warm-up, multi-run management, method routing |

**4 · Observability & output** — capturing, persisting, and analyzing results.

| Module | What it owns |
| --- | --- |
| [Logging & Verbosity](modules/logging.md) | per-iteration capture, core-time accumulation, range-gated output |
| [Debug Mode](modules/debug-mode.md) | threshold-triggered diagnostic checks, orthogonal to verbosity |
| [Persistence](modules/persistence.md) | JLD2 + CSV + JSON manifest; date/counter naming |
| [Analysis & Plotting](modules/analysis-plotting.md) | DataFrame pipeline, color registry, multi-figure layout |

- **[Convergence & Cost](convergence-and-cost.md)** — what "converged" means per problem
  class, which stopping criterion certifies it, reading empirical rates, and the cost model.
- **[Repository Internals](internals.md)** — directory layout, the data-flow diagram, and
  the key architectural decisions (with their rationale).
- **[Extension Guide](extending.md)** — step-by-step recipes for adding a method, problem,
  component, stopping criterion, warm-up, or debug check.
