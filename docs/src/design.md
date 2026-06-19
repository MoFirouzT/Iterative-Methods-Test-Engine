# Design Philosophy

**Audience.** This document is the contributor / maintainer reference.
If you only want to use the framework, start with the [README](https://github.com/MoFirouzT/Iterative-Methods-Test-Engine#readme) and the [walkthrough notebook](https://github.com/MoFirouzT/Iterative-Methods-Test-Engine/blob/main/walkthrough.ipynb); for the design rationale read [DESIGN.md](https://github.com/MoFirouzT/Iterative-Methods-Test-Engine/blob/main/DESIGN.md), then the co-located method and problem design notes (e.g. `gradient_descent.md`).

**Terminology.** The framework is organized into a number of cohesive *modules*.
Earlier drafts used the word "layer," but the structure is hub-and-spoke (the
orchestrator is the hub; everything else is a peer module) rather than strictly
stacked. Module numbering below follows dependency order — read top-down without
forward references.

The framework is built on six Julia-native principles:

- **Dispatch for extension, components for variation.** Every algorithm, stopping
  criterion, and problem is an extension point — a new one is a new type + a method,
  never an edit to existing code. Variation rides on *components* specifically: the
  swappable pieces inside a method (step size, descent direction, extrapolation,
  preconditioner) are what the variant grid sweeps.
- **Engine / content separation.** `src/` (the `TestEngine` module) ships only
  abstractions and machinery; every concrete method, problem, and component is *content*
  that extends the engine via `import .TestEngine`. The engine never names a concrete
  method, so it stays small and dependency-lean.
- **Declarative experiment definition.** An experiment is a plain, serializable data
  structure (`ExperimentConfig`). Running it, saving it, and reloading it are separate,
  independent operations.
- **Scientific measurement discipline.** Timing records only the core mathematical
  computation inside each step, accumulated per iteration and summed across iterations.
  All bookkeeping (logging, stopping criterion checks, verbosity output) is
  deliberately excluded from measured time.
- **Claims are demonstrated, not asserted.** A capability is only real once you can
  watch it run. Every problem and every algorithm is accompanied by a co-located `.md`
  design note that records the mathematical formulation, the implementation contracts
  (`init_state`, `step!`, `extract_log_entry`), and the win conditions its demonstrating
  experiment must exhibit; that experiment lives in `experiments/`, and the load-bearing
  claims are pinned by the test suite (a symbol→code variable-mapping table is optional,
  used only where the mapping isn't obvious from the code). Pluggable components (descent
  directions, step-size rules, ...) get their own dedicated design note when shared across
  algorithms. The design note is the contract; the experiment and the tests are the proof.
- **Separation of concerns across modules.** Algorithms know nothing about logging.
  Loggers know nothing about plotting. Stopping criteria know nothing about algorithms.
  Each module communicates through well-defined data structures, so any one module can be
  read, tested, and replaced in isolation.

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

The module pages — one per concern, each owning one part of the engine — are catalogued
on the [reference index](index.md), grouped into the four phases of a run (dependency
order within each phase, so they read top-down with no forward references). The per-file
breakdown of `src/` lives in [Repository Internals](internals.md).

One structural note that the catalogue does not show: the **algorithm-core** and
**nested-algorithm** concerns are co-located in `core.jl` to avoid circular includes —
both depend on the same base types, and the nested infrastructure (`run_sub_method`)
calls the `init_state` / `step!` defined alongside it.

---
