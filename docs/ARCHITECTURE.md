# Architecture reference

The architecture reference is now a **Documenter site**, split into one page per module for easier reading.
The content of this single file now lives under [`docs/src/`](src/) and is published to GitHub Pages.

- **Hosted site:** <https://MoFirouzT.github.io/Iterative-Methods-Test-Engine>
- **Browse the sources on GitHub:**
  - [Design Philosophy](src/design.md) — principles + high-level module map
  - Modules:
    - [Problem Interface](src/modules/problem-interface.md)
    - [Algorithm & Core Timing](src/modules/algorithm-core.md)
    - [Stopping Criteria](src/modules/stopping-criteria.md)
    - [Variant Grid Engine](src/modules/variant-grid.md)
    - [Nested Algorithms](src/modules/nested-algorithms.md)
    - [Logging & Verbosity](src/modules/logging.md)
    - [Experiment Orchestration](src/modules/orchestration.md)
    - [Persistence](src/modules/persistence.md)
    - [Debug Mode](src/modules/debug-mode.md)
    - [Analysis & Plotting](src/modules/analysis-plotting.md)
  - [Repository Internals](src/internals.md) — directory layout, data flow, key decisions
  - [Extension Guide](src/extending.md) — how to add a method, problem, component, …

Build the site locally with `julia --project=docs docs/make.jl` (output in `docs/build/`).
