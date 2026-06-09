# Experiment TODOs — What's Left After the Rosenbrock Stages

Companion to `basic_experiments.md`. Stages 0–7 validate every architectural
block that *can* be validated on a single, smooth, well-conditioned 2D problem.
This file enumerates what they cannot reach and why.

Each gap is tagged with one of:

- **(framework)** — the surface area is exported by `TestEngine` but no
  consumer exists in `src/` or `algorithms/`. A new experiment can't validate
  what isn't implemented. Listed here so the dead exports don't get forgotten.
- **(problem-family)** — the consumer exists; what's missing is a problem
  that exercises it. A new file in `problems/` plus a stage in `experiments/`
  closes the gap.
- **(both)** — needs framework work *and* a problem family before the
  validation is possible.

The stage numbers below continue Rosenbrock's 0–7 sequence. Whether to keep
that numbering or restart per problem family is a presentation decision; the
gaps themselves are real either way.

---

## Immediate blockers

### Persistence — JLD2 size on multi-run experiments

**Tag:** (framework). **Symptom:** A single 5-method × 20 000-iter
Rosenbrock run produces a 47 MB `result.jld2`. Stage 6's `n_runs = 20`
projection puts that near 1 GB per experiment directory.

**Status:** `save_experiment(...; compress = …)` is now exposed in
[src/persistence.jl](../src/persistence.jl) and forwards the kwarg to
`JLD2.save`. *But:* compression with the default codec is empirically a
net loss on this payload (104.8% of the uncompressed size — see
[docs/architecture.md §10 JLD2 compression](../docs/architecture.md#jld2-compression)
for measurements). The dominant cost is `IterationLog.extras::Dict{Symbol,Any}`
typing/dispatch overhead, which doesn't compress and adds codec block
headers. So the default is `compress = false`; the kwarg exists for
opt-in when a different problem family or codec wins.

**Remaining work — schema migration.** Realistic JLD2 shrinkage requires
changing the on-disk layout, not the codec:

- Current layout (array-of-structs): one `IterationLog` per iter, each
  carrying its own `Dict{Symbol,Any}` extras. JLD2 stores the dict's
  type machinery per row.
- Proposed (struct-of-arrays per method): one column-major struct per
  method holding `iter::Vector{Int}`, `objective::Vector{Float64}`,
  `gradient_norm::Vector{Float64}`, ... plus a `extras::Dict{Symbol,
  Vector{Any}}` keyed by extras name with one cell per iter (missing
  where absent).

Estimated payoff: 5–10× on Rosenbrock-style payloads where the columns
are uniformly typed and densely populated. Cost: a persistence-schema
migration with a versioned manifest, and `to_dataframe` / `iter_logs`
rewrites. Not Stage-4 blocking; queue under "framework gaps."

---

## Safeguards needing real implementation

### GLL nonmonotone line search for Barzilai-Borwein

**Tag:** (framework). **Currently:** `BarzilaiBorwein` has `α_min`/`α_max`
fields in [step_sizes.jl](../algorithms/components/step_sizes.jl)
but they're set to a permissive `[0, 1e6]` by default — effectively a
no-op safety net for true numerical overflow.

The honest finding from an α_max sweep on Rosenbrock from x₀=(-1.2, 1):

| α_max | n_iters | max f | f_end | comment |
|---|---|---|---|---|
| 10 | 2000 (DNF) | 1.3e7 | 39.7 | clamp is active, **breaks convergence** |
| 30, 50, 100, 1e6, ∞ | 76 | 9.06e8 | 7e-26 | clamp transparent, BB1 converges |

BB1 takes a step at iter 6 with α ≈ 28 that sends f to ~10⁹, then
contracts and recovers over ~40 iters. That excursion is **load-bearing** —
clipping it interrupts the recovery dynamics and BB1 gets stuck. So the
clamp has no useful problem-independent default: tight enough to catch the
spike → breaks BB; loose enough to let BB work → catches nothing.

The textbook safeguard that distinguishes "safe long step that will
recover" from "unsafe long step that will diverge" is the Grippo-
Lampariello-Lucidi (1986) nonmonotone line search, popularized for BB by
Raydan (1997). It allows non-monotonic f but requires

    f_k ≤ max(f_{k-M}, ..., f_{k-1}) − γ · α · ‖∇f_k‖²

for a window of size M (typically 5–10) and small γ (typically 1e-4). If
the BB step violates this, backtrack. BB recovers naturally; only
diverging-not-recovering trajectories get caught.

**Concrete sketch:** add a `nonmonotone_window` field to
`GradientDescentNumerics` (a circular buffer of recent f values), let
`compute_step_size` for BB peek at the window and the candidate `α`, and
backtrack inside the rule if the GLL condition fails. Or wrap BB in a
higher-order safeguard rule `NonmonotoneSafeguard(BarzilaiBorwein(...))`.
Either is ~30 lines. The win is: BB1 keeps its dramatic recovery
behavior on Rosenbrock-class problems where it *does* recover, while
being protected on problems where it would otherwise diverge.

**Unblocked by:** nothing — this is framework-internal work. **Validated
by:** any problem family where unsafeguarded BB diverges (good candidate:
high-`ρ` Rosenbrock from a x₀ far outside the basin, or pathological
quadratics).

---

## Framework gaps — defined, no consumer

These are the largest pieces of dead surface area. Since the engine/content split they
live as **content** under `algorithms/components/` (no longer exported by the engine), but
still have no method that consumes them, so no experiment can validate them as wired.

### Quasi-Newton hierarchy

**Tag:** (framework). **Types:** `HessianApprox`, `FullHessian`, `BFGS`,
`SR1`, `LBFGS`, `DiagBFGS` ([algorithms/components/hessian_approx.jl](../algorithms/components/hessian_approx.jl)).

These are declared as empty structs. There is no `step!` method that takes a
`HessianApprox` and computes a Newton/quasi-Newton direction. To make this
testable:

1. Add a concrete method type (e.g. `QuasiNewton(approx::HessianApprox,
   step_size::StepSize)`) with `init_state` and `step!`.
2. Implement the rank-one and rank-two updates for `BFGS`, `SR1`, `LBFGS`,
   `DiagBFGS` against their secant pairs.
3. Add a stage that runs at least `BFGS` and `LBFGS` on Rosenbrock and
   compares to Stage 1's GD curves on the iter and eval axes. BFGS on
   Rosenbrock should hit `:gradient_converged` in O(20–30) iters at ρ=100.

**Validates (once implemented):** secant-pair bookkeeping, the
`HessianApprox` dispatch surface, comparison curves where the dominant cost
shifts from line search to the update.

### Minor-update hierarchy

**Tag:** (framework). **Types:** `MinorUpdate`, `NoMinorUpdate`,
`MomentumStep`, `NesterovStep`, `CorrectionStep`
([algorithms/components/minor_updates.jl](../algorithms/components/minor_updates.jl)).

**✅ Mostly resolved (portfolio Item 2).** `MinorUpdate` now carries behavior
(`extrapolate` / `advance_momentum`), and `ProximalGradient` composes it:
`NoMinorUpdate` ⇒ ISTA, `NesterovStep` ⇒ FISTA, `MomentumStep` ⇒ heavy-ball.
`NoMinorUpdate` + `NesterovStep` are exercised by `exp_lasso1_ista_fista.jl`
and `test/test_proximal_gradient.jl` (FISTA's O(1/k²) acceleration asserted).
**Still dark:** `CorrectionStep` (no consumer — Tier-3 prune candidate); and
`MomentumStep`, while wired, has no shipped experiment yet (cheap swept
variant for a future stage).

### Experimental methods

**Tag:** (framework). **Exports:** `ExperimentalMethod` (abstract).
**Directory:** [algorithms/experimental/](../algorithms/experimental/) is empty.

`resolve_methods` already routes by concrete type so adding a method here
should not require runner changes. But until at least one experimental method
exists, the `experimental_methods` field of `ExperimentConfig` is a vestigial
slot. Even a single illustrative implementation (e.g. an adaptive-step
heuristic that doesn't fit `ConventionalMethod`) would unlock validation of
the dual-bucket routing.

### Sub-method machinery in experiments

**Tag:** (framework — soft). **Exports:** `run_sub_method`, `SubRunConfig`,
`SubResult`, `attach_sub_logs!`.

These are exercised by [test/runtests.jl](../test/runtests.jl) but no stage in
`experiments/` uses them. The mechanism exists for nested optimization
(inner solves inside outer iterations); validating it in flight requires a
method that actually calls `run_sub_method`. Options:

- Implement a `TrustRegion` method whose subproblem is a bounded quadratic
  minimization — natural fit, and gives the sub-method log-attachment a real
  use case.
- Or implement `IterativeWarmup`'s warm-up as a sub-method run rather than a
  bare `run_method` call, so Stage 6 exercises the sub-method path too.

---

## Problem-family gaps

These need new problem implementations under `problems/`. Each unlocks a
specific class of validations that 2D Rosenbrock cannot reach.

### Linear least squares — dimension and conditioning sweeps

**Tag:** (problem-family). **Unlocks:** `LeastSquares`, `LeastSquaresKernel`,
`OperatorHessian`, dimension scaling, condition-number sweeps, the
`@core_timed`-vs-wall measurement as a real ordering signal.

**✅ Resolved (portfolio Item 1).** Done end to end:
- `LeastSquares` now has a **selectable Hessian mode** (`:matrix` default /
  `:operator`); `:operator` returns `OperatorHessian(d → Aᵀ(A d), n)` — lit up
  by the `:linear_ls` family and applied by `CauchyStep`.
- `:linear_ls` generator registered ([least_squares.{md,jl}](../problems/least_squares/))
  — parametrized by `κ = cond(AᵀA)` (singular values span `1 → κ^(−1/2)`, so the
  squaring is correct), consistent `b = A·x_star` ⇒ `x_opt = x_star`, `f* = 0`.
- **Stage LS-1** ([exp_ls1_dimension.jl](exp_ls1_dimension.jl)) — iters flat in n,
  wall ∝ O(mn), **core_time/wall_time climbs 2% → 89% → 98%** across n∈{10,100,1000},
  landing in [50%,110%] at n=1000 (the timing pillar, finally validated).
- **Stage LS-2** ([exp_ls2_conditioning.jl](exp_ls2_conditioning.jl)) — log-log
  slopes: Fixed/Armijo/Cauchy ≈ **1.0** (O(κ)), BB1/BB2 ≈ **0.5** (O(√κ)).
- Regression tests in [test_least_squares.jl](../test/test_least_squares.jl).
- **Surfaced & fixed two latent bugs** the small-gradient / large-n regimes
  exposed: `CauchyStep`'s absolute curvature guard (→ scale-relative, see
  `step_sizes.md`) and a missing `diag` import breaking `diagonal(::MatrixHessian)`.

κ range was capped at 1e4 (not the sketched 1e7): Fixed/Armijo are O(κ), so 1e7
would need ~1e8 iters — infeasible. Three decades reads the slope cleanly.

<details><summary>Original sketch (for reference)</summary>

The kernel and a basic `:quadratic` registration exist; the conditioning `:linear_ls`
generator and the scaling/conditioning stage are missing. Sketch:

```julia
register_random_problem!(:linear_ls, (rng, params) -> begin
    n = get(params, :n, 100)
    m = get(params, :m, 200)
    κ = get(params, :condition_number, 1e3)
    A = generate_matrix_with_condition(rng, m, n, κ)  # SVD synthesis
    x_star = randn(rng, n)
    b = A * x_star
    x0 = zeros(n)
    Problem(LeastSquares(LeastSquaresKernel(A, b)), x0;
            meta = Dict(:condition_number => κ),
            x_opt = x_star)
end)
```

**Suggested experiments:**

- **Stage 8 — dimension scaling.** Same 5 GD variants, sweep
  `n ∈ {10, 100, 1000}` at fixed κ. Plot iters-to-tolerance and wall time vs.
  n. Validates that no method has accidental O(n²) bookkeeping outside
  `@core_timed`.
- **Stage 9 — conditioning sweep.** Sweep `κ ∈ {1e1, 1e3, 1e5, 1e7}` at
  fixed n=100. Plot iters-to-tolerance vs. κ on log-log. Cauchy and Armijo
  should degrade roughly as O(κ); BB should be much flatter. The slope of
  each curve is the validation.
- **Stage 8/9 timing.** With n ≥ 100 the GD kernel actually dominates per-iter
  scaffolding, so the `core_time / wall_time ∈ [50%, 110%]` band from
  Stage 4 has a real chance of being met. If it still isn't, the diagnostic
  in `print_timing_table` points at the right fix.

</details>

### Lasso — composite f + g, exercises `prox`

**Tag:** (both). **Unlocks:** `L1Norm`, `L2Norm`, `ZeroRegularizer`, `prox`
operator dispatch, the regularizer-sum branch of `total_objective`. **Needs:**
a proximal method.

**✅ Resolved (portfolio Item 2 — the flagship).** All three steps are done:

1. `ProximalGradient` ([algorithms/conventional/proximal_gradient/](../algorithms/conventional/proximal_gradient/))
   interleaves a gradient step on `f` with a `prox` call on `g`.
2. `:lasso` problem registered ([problems/lasso/](../problems/lasso/)):
   `LeastSquares` + `L1Norm`, controllable `m, n, k, λ`, `L = ‖A‖²` in meta.
3. **Stage LASSO-1** ([exp_lasso1_ista_fista.jl](exp_lasso1_ista_fista.jl)) —
   ISTA vs FISTA money figure (FISTA's O(1/k²) vs ISTA's O(1/k); support
   recovery). `test/test_proximal_gradient.jl` asserts `prox` is called once
   per step, `total_objective` sums `f + g`, and FISTA's gap is < 0.1× ISTA's
   at a mid iteration. The planted support is recovered with zero spurious
   coordinates in the shipped instance.

**Remaining (deferred):** the *sparsity-recovery-vs-λ sweep* (Stage LASSO-1
shows a single λ); `L2Norm` (a ridge demo would light it up); and `ProxGrad`
with a `ZeroRegularizer` as an explicit smooth-acceleration stage (the
reduction is unit-tested but has no shipped figure).

**Concrete invariant:** at sufficiently large `λ`, the final iterate's support
is a subset of the true support; "extra" coordinates ProxGD touches should have
`|x_i| ≤ γ λ` — one soft-threshold call away from zero.

### Logistic regression with mini-batches — stochastic `step!`

**Tag:** (both). **Unlocks:** the rng path through `step!` (currently
exercised structurally but never functionally — Stage 5's caveat acknowledges
this), per-run rng derivation as actually affecting iter logs. **Needs:** an
SGD-flavored method.

Rosenbrock has no data, so no `step!` draws from rng. A logistic regression
problem with a random subset of rows per iter would make the rng path real.
This is the only way to validate that:

- the per-`(seed, run_id, name)` rng key actually selects different rows
  across methods that share a method type;
- `aggregate_runs` quantile bands actually widen with seed variance, not just
  the x₀ randomization that Stage 6 already exercises.

**Suggested:** **Stage 11 — SGD on logistic regression.** Sweep batch size
`b ∈ {1, 32, full}`, `n_runs = 20`, plot median + IQR. Confirm that full-batch
collapses the IQR to zero (deterministic) and small-batch fans out.

### File-loaded problem

**Tag:** (problem-family). **Unlocks:** `FileProblem`,
`register_file_loader!`.

Exercised by [test/test_module9.jl](../test/test_module9.jl) but no
experiment touches it. Low priority but cheap: stash a small matrix on disk,
write a loader, run any existing method on it. Mostly a smoke test that the
loader registration works in the live experiment path, not just the unit
test path.

### Constrained / projection-based problems

**Tag:** (both). **Unlocks:** indicator-function regularizers, projected
gradient methods.

No indicator regularizer exists yet (would slot in next to `L1Norm`,
`L2Norm`, `ZeroRegularizer` with a `prox` that projects). Once it does, a
projected-gradient method on a box-constrained quadratic would validate the
non-smooth-prox path with a different `prox` shape from the lasso.

---

## Hessian-representation gaps

**Tag:** (problem-family, mostly).

[src/problems.jl](../src/problems.jl) exports `MatrixHessian`,
`OperatorHessian`, `DiagonalHessian`.

- `MatrixHessian` — constructed by Rosenbrock and by `LeastSquares` in its
  `:matrix` mode. `diagonal(::MatrixHessian)` was latently broken (missing
  `diag` import); **fixed** in Item 1, so the Jacobi-preconditioner path is ready.
- ✅ `OperatorHessian` — **now constructed** by `LeastSquares` in `:operator`
  mode (`d → Aᵀ(A d)`), exercised by the `:linear_ls` family + `CauchyStep`
  (Item 1). No longer dark.
- `DiagonalHessian` — **still dark.** Natural for separable problems (a
  per-coordinate quadratic, `f(x) = ½ Σ dᵢ xᵢ²`); slated for Item 3
  (separable quadratic + Jacobi preconditioner).

These are essentially free once the underlying problems exist; what they
validate is the `apply`/`materialize`/`diagonal` dispatch surface on
non-`MatrixHessian` representations.

---

## Cross-cutting validations not yet covered

These don't fit a single stage but are worth listing so they don't fall
through the cracks.

- **Persistence of `logger.events`.** Stage 7 exercises `on_trigger = :log`
  recording events into memory, but Stage 3's roundtrip check only validates
  `iter_logs`. Whether `events` survive `save_experiment` / `load_experiment`
  is not asserted anywhere.
- **`aggregate_runs(df, :all)` mode.** Stage 6 specifies `:median`; the `:all`
  branch (if it exists) is not exercised. Confirm whether there are
  user-facing aggregation modes beyond `:median` that need coverage.
- **`method_color` registry round-trip.** Stages 1, 5, 6 use the color
  registry. Whether `register_method_color!` mutations persist correctly
  across `save_experiment` and replay is not asserted.
- **`numerical_gradient` correctness on a problem with a non-trivial gradient
  shape.** Stage 7 calls `CheckNumericalGradient` on Rosenbrock, which is
  smooth. A test on a non-quadratic, anisotropic problem would catch
  step-size selection bugs in the central-difference computation that
  Rosenbrock would not surface.
- **`elapsed_core_s` / `elapsed_wall_s` semantics under sub-methods.** Whether
  inner-loop core time is attributed to the outer step's `core_time_ns` or
  kept separate is undocumented. Once `run_sub_method` is used in an
  experiment, an explicit assertion on the attribution would be useful.

---

## Suggested prioritization

If the goal is "maximum coverage gain per unit work," roughly in this order:

1. **Linear least squares + Stage 8/9.** Single problem family, two stages,
   unlocks dimension scaling, conditioning sweeps, and gives the timing
   ratio a fair test. The kernel exists; only registration + experiment
   scripts are missing.
2. **Lasso + ProximalGradient + Stage 10.** Activates the entire composite-
   problem branch (regularizers, `prox`, `total_objective` sum). Highest-
   leverage *new code* item — it lights up an entire module.
3. **Quasi-Newton (`BFGS` at minimum) + a stage.** Standard reference method
   on Rosenbrock; once implemented, becomes a baseline every later stage can
   compare against.
4. **Stochastic / mini-batch logistic regression + Stage 11.** Validates the
   rng path through `step!` functionally rather than structurally.
5. The remaining items (operator/diagonal Hessian, file loader,
   constraints, minor updates) are smaller, follow naturally from the above,
   and can be tackled opportunistically.
