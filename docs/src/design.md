# Design Philosophy


**Audience.** This document is the contributor / maintainer reference. If you only want
to use the framework, start with the user guide and `gradient_descent.md`.

**Terminology.** The framework is organized into a number of cohesive *modules*.
Earlier drafts used the word "layer," but the structure is hub-and-spoke (the
orchestrator is the hub; everything else is a peer module) rather than strictly
stacked. Module numbering below follows dependency order — read top-down without
forward references.

The framework is built on four Julia-native principles:

- **Multiple dispatch over class hierarchies.** Every algorithm, component, stopping
  criterion, and problem is a dispatch point. Adding a new variant never requires
  touching existing code.
- **Separation of concerns across modules.** Algorithms know nothing about logging.
  Loggers know nothing about plotting. Stopping criteria know nothing about algorithms.
  Each module communicates through well-defined data structures.
- **Declarative experiment definition.** An experiment is a plain, serializable data
  structure (`ExperimentConfig`). Running it, saving it, and reloading it are separate,
  independent operations.
- **Scientific measurement discipline.** Timing records only the core mathematical
  computation inside each step, accumulated per iteration and summed across iterations.
  All bookkeeping (logging, stopping criterion checks, verbosity output) is
  deliberately excluded from measured time.
- **Specification-driven implementation.** Every problem and every algorithm is
  accompanied by a `.md` file co-located with its source. Each spec is the
  single source of truth for the mathematical formulation, variable mapping, and
  implementation contracts (`init_state`, `step!`, `extract_log_entry`). Pluggable
  components (descent directions, step-size rules, ...) get their own dedicated spec
  file when they are shared across algorithms.

**Additional invariants enforced by the framework:**

- **Logger purity.** Algorithms never hold a reference to the logger. The logger is
  passed as an explicit parameter to `step!` and `run_sub_method` by the runner.
  Algorithm code is free of logging concerns.
- **Reproducibility.** Every source of randomness — data generation, warm-up,
  initial point `x0`, stochastic algorithmic steps, and nested sub-solver calls — is
  derived from a single `ExperimentConfig.seed` via deterministic, session-stable
  hashing. Sub-solver RNGs are child streams derived from the outer method's RNG.
- **Debug orthogonality.** The debug mode is an optional layer activated at experiment
  level. It adds diagnostic computations after each step (e.g. gradient checks,
  objective monotonicity) without touching algorithm or logging code.

---

## High-Level Module Map

| # | File | Responsibility |
|---|------|----------------|
| 1 | `problems.jl` | `Objective`, `Regularizer`, `Hessian`, `Problem`, `ProblemSpec` hierarchy, problem factory |
| 2 | `core.jl` | Type hierarchy, state groups, algorithm interface, `@core_timed`, run loop, nested infrastructure |
| 3 | `stopping.jl` | Stopping criteria abstraction, composites, `should_stop` dispatch |
| 4 | `variants.jl` | Component abstractions, Cartesian grid expansion, auto-naming |
| 5 | `core.jl` | Nested algorithm infrastructure (`SubRunConfig`, `run_sub_method`) |
| 6 | `logging.jl` | Per-iteration capture, core-time accumulation, event logging, sub-logs, verbosity |
| 7 | `experiment.jl` | Experiment config, result types, warm-up, orchestration, multi-run management |
| 8 | `persistence.jl` | JLD2 binary + CSV sidecar + JSON manifest; date/counter naming |
| 9 | `debug.jl` | `DebugConfig`, `DebugCheck` hierarchy, `run_debug_checks!`, diagnostic helpers |
| 10 | `analysis.jl` | DataFrame pipeline, color registry, flexible multi-figure layout |

Modules 2 and 5 are co-located in `core.jl` to avoid circular includes: both depend on
the same base types and the nested infrastructure (`run_sub_method`) calls `init_state`
and `step!` defined in Module 2.

---

