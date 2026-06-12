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

![ISTA vs FISTA on the lasso: FISTA's O(1/k²) acceleration over ISTA's O(1/k), and exact support recovery](https://raw.githubusercontent.com/MoFirouzT/Iterative-Methods-Test-Engine/main/figures/lasso_ista_fista.png)

## Design in one breath

- **Multiple dispatch over hierarchies** — every method, component (step size, descent
  direction, minor update, preconditioner), stopping criterion, and problem is a dispatch
  point; adding a variant never edits existing code.
- **Engine / content separation** — `src/` (the `TestEngine` module) ships only
  abstractions and machinery; every concrete method, problem, and component is *content*
  that extends the engine via `import .TestEngine`.
- **Scientific measurement discipline** — `@core_timed` measures only the core math inside
  each step; logging, stopping checks, and bookkeeping are excluded. That honest
  `core_time / wall_time` ratio is itself a reported result.
- **Declarative, reproducible experiments** — an experiment is a serializable
  `ExperimentConfig`; every source of randomness derives from a single seed.

## How this reference is organized

The engine is a hub-and-spoke of cohesive **modules** (the orchestrator is the hub).
Module numbering follows dependency order — read top-down, no forward references.

- **[Design Philosophy](design.md)** — the guiding principles and the high-level module map.

**Modules** — one page per concern, each the contract for one part of the engine:

| Module | What it owns |
| --- | --- |
| [Problem Interface](modules/problem-interface.md) | `Objective` / `Regularizer` / `Hessian` / `Problem`, the problem factory |
| [Algorithm & Core Timing](modules/algorithm-core.md) | method type hierarchy, composable state groups, `@core_timed`, the run loop |
| [Stopping Criteria](modules/stopping-criteria.md) | composable termination via `should_stop` dispatch |
| [Variant Grid Engine](modules/variant-grid.md) | typed components, Cartesian grid expansion, auto-naming |
| [Nested Algorithms](modules/nested-algorithms.md) | `SubRunConfig` / `run_sub_method` for sub-solvers inside a step |
| [Logging & Verbosity](modules/logging.md) | per-iteration capture, core-time accumulation, range-gated output |
| [Experiment Orchestration](modules/orchestration.md) | config, warm-up, multi-run management, method routing |
| [Persistence](modules/persistence.md) | JLD2 + CSV + JSON manifest; date/counter naming |
| [Debug Mode](modules/debug-mode.md) | threshold-triggered diagnostic checks, orthogonal to verbosity |
| [Analysis & Plotting](modules/analysis-plotting.md) | DataFrame pipeline, color registry, multi-figure layout |

- **[Repository Internals](internals.md)** — directory layout, the end-to-end data-flow
  diagram, and the key architectural decisions (with their rationale).
- **[Extension Guide](extending.md)** — step-by-step recipes for adding a method, problem,
  component, stopping criterion, warm-up, or debug check.
