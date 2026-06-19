# DESIGN — a five-minute tour

This is the short read.
For the full maintainer reference see the [architecture docs site](https://MoFirouzT.github.io/Iterative-Methods-Test-Engine) (one page per module);
for the one-command demo see [README.md](README.md).

## The problem it solves

When you are developing an iterative optimization method, the questions that actually matter are comparative:
*Does my variant beat the conventional methods?
How does it scale with dimension, conditioning or any other variation?
Is it actually faster, or just doing less bookkeeping?*
Answering these by hand means re-plumbing a run loop, a logger, a timing harness, and a plotting pipeline for every experiment — and the comparisons drift apart.

This engine makes the comparison the unit of work.
You define a method once; the harness runs it (and its swept variants, and the conventional baselines) on a shared problem under identical stopping criteria, records convergence metrics and *core* compute time the same way for every method, and hands back a serializable result you can plot or reload.

## Six design principles

1. **Dispatch for extension, components for variation.**
   Methods, stopping criteria, problems, and Hessian representations are all extension points:
   a new one is a new type + a method on the relevant function, never an edit to existing code.
   Variation rides on *components* specifically — the swappable pieces inside a method (step size, descent direction, extrapolation, preconditioner, ...) are what the variant-grid engine sweeps.

2. **Engine / content separation.**
   `src/` (the `TestEngine` module) ships only abstractions and machinery:
   the `Problem`/`Objective`/`Hessian` interfaces, the `run_method` loop, stopping criteria, the variant-grid engine, logging, persistence, plotting.
   Every *concrete* method, problem, and component is **content** under `algorithms/` and `problems/` that extends the engine via `import .TestEngine`.
   The engine never names a concrete method — so it stays lean.

3. **Scientific measurement discipline.**
   A step wraps its real mathematics in `@core_timed`;
   logging, stopping-criterion checks, and verbosity are deliberately excluded from measured time.
   This makes "is it faster?" answerable honestly.

4. **Declarative, reproducible experiments.**
   An experiment is a plain `ExperimentConfig` value.
   Running it, saving it, and reloading it are independent operations, and every source of randomness (data, warm-up, `x0`, stochastic steps, sub-solvers) derives from a single seed by deterministic hashing.

5. **Claims are demonstrated, not asserted.**
   A capability is only real once you can watch it run. Each method and problem ships a co-located `.md` design note that records the math, the implementation contract (`init_state`, `step!`, `extract_log_entry`), and the *win conditions* its demonstrating experiment must exhibit; that experiment lives in `experiments/`, and the load-bearing claims are pinned by the test suite. The design note states the contract; the experiment and the tests are the proof. (A symbol→code variable-mapping table is optional, used only where the mapping isn't obvious from the code.)

6. **Separation of concerns across modules.** Algorithms know nothing about logging,
   loggers nothing about plotting, stopping criteria nothing about algorithms. Each module
   talks to the next through plain data structures, so any one can be read, tested, and
   replaced in isolation.

## One experiment, annotated: sweeping one component on the lasso

The flagship experiment ([experiments/exp_lasso_ista_fista.jl](experiments/exp_lasso_ista_fista.jl))
exercises the whole composite-objective path with a single method — and is the project's
core idea in one figure: *one* method, with only a single swappable component varied.

**Problem.** Sparse recovery: `min_x ½‖Ax − b‖² + λ‖x‖₁`, underdetermined (`m < n`) with a
planted `k`-sparse signal — registered as the `:lasso` family.

**Method.** `ProximalGradient` is one method with two plug-in slots: a `StepSize` and an
`Extrapolation`. Each step extrapolates, takes a gradient step on the smooth `½‖Ax−b‖²`, then
applies the `prox` of `λ‖x‖₁` (soft-thresholding). The figure holds the `StepSize` fixed
(`γ = 1/L`) and sweeps **only the `Extrapolation` component**:

- `Extrapolation = NoExtrapolation()` ⇒ **ISTA** (plain proximal gradient, `O(1/k)`).
- `Extrapolation = MomentumStep(α)` ⇒ **heavy-ball** (fixed-momentum extrapolation).
- `Extrapolation = NesterovStep()` ⇒ **FISTA** (`O(1/k²)`).
- a *zero* regularizer ⇒ the same method is (accelerated) gradient descent on a smooth
  problem — so one method also tells the smooth-acceleration story.

**Result (the flagship figure).**

![One ProximalGradient on the lasso with its extrapolation component swept: ISTA, heavy-ball, FISTA, plus support recovery](figures/lasso_ista_fista.png)

*Left:* `f(xₖ) − f*` on a log axis. The single swept component orders the convergence —
**ISTA > heavy-ball > FISTA** — FISTA plunging lowest with its characteristic non-monotone
ripple, heavy-ball's fixed momentum a clean monotone middle tier. *Right:* the recovered
iterate's nonzeros coincide with the planted ±1 support, against a flat zero baseline.

What the figure shows is the *acceleration ordering*, not the asymptotic rates: on this
benign, well-conditioned instance all three ultimately converge linearly once the support
is identified (and a larger heavy-ball `α` would match or beat FISTA — it is optimal for
nice problems). The clean `O(1/k)`-vs-`O(1/k²)` slope separation is measured separately, on
a dedicated non-strongly-convex instance — see
[Reading an empirical convergence rate](docs/src/convergence-and-cost.md#reading-an-empirical-convergence-rate)
and `test/test_proximal_gradient.jl`.

**Why it's trustworthy.** The regularizer's `prox` is provided by
[ProximalOperators.jl](https://github.com/JuliaFirstOrder/ProximalOperators.jl) behind the
engine's `prox` contract, and the converged solution is cross-checked against
[ProximalAlgorithms.jl](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl)'s
ForwardBackward/FastForwardBackward (`test/test_external_validation.jl`). The same harness
also surfaced — and fixed — two latent bugs that 2-D Rosenbrock never triggered: a Cauchy
step-size curvature guard that misfired as `‖∇f‖→0` (the scale-relative-guard fix is pinned
by the regression test in [`test/test_least_squares.jl`](test/test_least_squares.jl) —
`< 5000` iters where the old absolute guard stalled at ~240k), and a missing-import break in
`diagonal(::MatrixHessian)` (now exercised by the Jacobi-on-`MatrixHessian` tests in
[`test/test_preconditioned_gradient.jl`](test/test_preconditioned_gradient.jl)).
A capability is only demonstrated by watching it run.

## Where to go next

- Run it: `julia --project reproduce.jl` (see [README.md](README.md)).
- The other four figures (dimension scaling + timing pillar, conditioning sweep, Jacobi
  preconditioning ≈ Newton, and trust-region + Steihaug-CG nested optimization) are in
  `figures/` and `experiments/exp_*.jl`.
- Full internals: the [architecture docs site](https://MoFirouzT.github.io/Iterative-Methods-Test-Engine).
