# Basic Experiments — Rosenbrock Validation Plan

Each stage validates a specific block of the framework against a single 2D Rosenbrock problem (ρ = 100, x₀ = (−1.2, 1) unless otherwise noted).
After Stage 7, every architectural block that *can* be validated on Rosenbrock has been;
the remaining blocks require other problem types (see `Experiment_TODOs.md`).

The build-up is intentional.
Stages 0–4 hand-roll the per-method rng derivation and the run loop, rehearsing
the orchestrator's contract before depending on the orchestrator.
Stage 5 is the orchestrator's debut;
Stages 6–7 build on it.

---

## Stage 0 — Smoke test

**Status:** done.
**File:** `smoke_test.jl`.

A 100-iter `FixedStep` `GradientDescent` run that proves the runner contract works
end-to-end.

**Exercises:**
`run_method`, `make_problem`, `make_logger`, `VerbosityConfig`,
`MethodResult`, `@core_timed` accumulation into `state.timing.core_time_ns`.

**Validates:**

- no exceptions;
- objective monotone (`FixedStep` at α = 8e-4 on Rosenbrock is monotone);
- f decreases.

---

## Stage 1 — Convergence panels

**Status:** done.
**File:** `exp_stage1.jl`.

Five `GradientDescent` variants (Fixed, Armijo, Cauchy, BB1, BB2) on Rosenbrock,
each stopped on `stop_when_any(MaxIterations(2000), GradientTolerance(1e-9))`.
Renders a 2×2 panel: `f(x)`, `‖∇f(x)‖`, `‖x − x*‖`, `αₖ` — all on log y-axis.

**Exercises:**

- the full hand-rolled run loop with manual per-method rng derivation
  (`Xoshiro(hash((seed, run_id, name)))`) — mirrors what `run_experiment` will use
  at Stage 5, so iter logs stay byte-comparable across stages for any
  randomness-free method;
- `extract_log_entry` populating fixed fields plus `:step_size` extras;
- `CompositeCriteria(:any)` with `MaxIterations` + `GradientTolerance`;
- `MethodResult` field access (`iter_logs`, `n_iters`, `stop_reason`, `final_state`);
- `Logger` at `MILESTONE` verbosity.

**Validates:**

- all five methods complete;
- BB1/BB2 converge fastest (expected ordering on Rosenbrock);
- Armijo's step-size panel shows discrete `β^j` values;
- Fixed's step-size panel is constant.

**Note on the rng key.** Uses short names (`"Armijo"`) for rng derivation.
Stage 5 will use the long form (`"GradientDescent[step_size=Armijo]"`) — this changes the
per-method rng stream but does not change deterministic-`step!` results.

---

## Stage 2 — Trajectories on the contour map

**Status:** done.
**File:** `exp_stage2.jl`.

Same five methods, visualized as (x₁, x₂) trajectories overlaid on log-spaced contours of f.
The figure that makes the Rosenbrock geometry visible.

**Exercises:**
vector-valued extras plumbing — specifically `extras[:x_iter]` —
from `step!` through `extract_log_entry` to the DataFrame; custom Makie code
outside `FigureLayout` (a deliberate decision: trajectory plots are different
enough from convergence curves that forcing them through the layout DSL hurts
more than it helps).

**Validates:**

- every row in the DataFrame carries a non-missing `:x_iter` (explicit assertion in the script);
- trajectories visually reach (1, 1);
- BB methods take characteristically jagged paths along the valley — visible
  evidence of intrinsic non-monotonicity.

---

## Stage 3 — Persistence roundtrip

**Status:** done.
**File:** `exp_stage3.jl`.

Same five methods, but now everything goes through `save_experiment` → `load_experiment` → `to_dataframe` → plot.
Plotting from the in-memory result is forbidden — all figures are produced from the loaded copy.

**Exercises:**

- `next_experiment_path` (atomic `mkdir` against the EEXIST race);
- `save_experiment` writing JLD2 + per-method CSV sidecars + `manifest.json`;
- `load_experiment`;
- the CSV-vs-JLD2 decision for vector-valued extras (`:x_iter`) — CSV scalars
  only, vectors live in JLD2, omitted keys recorded in `manifest.json`;
- cold-restart `replot(path)` entry point.

**Validates:**

- byte-identical DataFrames between in-memory and disk
  (`assert_roundtrip(df_mem, df_disk)`);
- vector-valued extras survive the JLD2 round-trip;
- cold restart produces visually identical figures and line-for-line identical
  CSVs.

**Bug surface to watch.**
If `assert_roundtrip` fails on `:x_iter` specifically, the JLD2 writer is dropping or corrupting vector extras
— the most common persistence bug at this stage.

---

## Stage 4 — Stopping criteria coverage

**Status:** in progress.
**File:** `exp_stage4.jl`.
Blocker: `DistanceToOptimal` is referenced here but not yet defined in
`src/stopping.jl` (only `MaxIterations`, `TimeLimit`, `GradientTolerance`,
`ObjectiveStagnation`, `StepTolerance`, `CompositeCriterion` exist). The
runner-side `dist_to_opt` update also still needs wiring. Once those land,
the script runs end-to-end and the new `print_timing_table` block (added
in preparation for the assertion move from Stage 0) becomes executable.

Same five methods, now stopping on
`stop_when_any(MaxIterations(20_000), DistanceToOptimal(1e-8), GradientTolerance(1e-10))`.
Produces a bar chart of *iterations to milestone* (first time `‖x − x*‖ ≤ 1e-6`)
with DNF handling for methods that never reach the milestone.

**Exercises:**

- `DistanceToOptimal` stopping criterion;
- runner-side `dist_to_opt` update from `problem.x_opt`;
- `stop_reason` propagation through `MethodResult`;
- multi-criterion `:any` composite;
- the distinction between *when the run stopped* (from `MethodResult.n_iters`)
  and *how long to first reach a looser milestone* (from `findfirst` on the iter
  trace).

**Validates:**

- BB1/BB2 stop with `:optimal_reached`;
- Fixed stops with `:max_iterations` (budget is the binding criterion);
- `dist_final ≤ DIST_TOL` for any method whose `stop_reason` is
  `:optimal_reached` or `:gradient_converged`;
- **`@core_timed` scope correctness:** aggregate `core_time / wall_time` lands
  in `[50%, 110%]`. 20_000 iters / method (post-warm-up) amortize per-iter
  scaffolding (`extract_log_entry`, `should_stop`, dispatch) enough for this
  to be a real signal — unlike Stage 0, where the kernel is below the noise
  floor. If the ratio is too low, either widen `@core_timed`'s scope in
  `step!` to include the norm/copy bookkeeping currently outside it, or the
  problem is too small to make the kernel dominate. If too high, `@core_timed`
  is sweeping in non-kernel work (logger, stopping check).

**Bug surface to watch.** If `dist_final` is large for a method that stopped with
`:optimal_reached`, the runner is not updating `state.metrics.dist_to_opt` before
the stopping check.

---

## Stage 5 — Orchestrator debut: `run_experiment` + `VariantGrid` + fair-comparison plots

**Status:** new.

First experiment that drives the framework through its actual user-facing entry
point — `run_experiment` — rather than a hand-rolled loop. Defines the same five
methods via a `VariantAxis(:step_size, ...)` grid, runs through the orchestrator,
then layers fair-comparison plots on top.

**Exercises:**

- `VariantAxis`, `VariantGrid`, `expand`, `VariantSpec` (Cartesian expansion and
  auto-naming);
- `ABBREVIATIONS` registry and `register_abbreviation!`;
- `resolve_methods` — routes each produced variant into the conventional bucket
  based on its concrete type;
- `run_experiment` orchestration loop, including its own deterministic
  per-`(seed, run_id, name)` rng derivation;
- `n_linesearch_evals` accounting inside Armijo's backtracking loop;
- per-iteration `core_time_ns` accumulation in `Logger.total_core_ns`.

**Plots.** Stage 1's four panels reproduced *three times*:

1. x-axis = iter (baseline, identical to Stage 1's figure);
2. x-axis = cumulative function evaluations (per-iter base cost plus
   `n_linesearch_evals` where applicable);
3. x-axis = cumulative `core_time_ns`, annotated with a noise-floor warning —
   the kernel is tens of nanoseconds on 2D Rosenbrock, OS jitter shifts
   cumulative time by 5–10%, so this plot is sanity-check only, not for
   ordering decisions.

**Validates:**

- **Byte-identical iter logs vs Stage 1.** Programmatic check: load both
  experiments, build their DataFrames, assert equality on `:iter`, `:objective`,
  `:gradient_norm`, `:dist_to_opt` for every method. If they drift, something in
  the orchestrator's setup path differs from the hand-rolled version — and that
  bug needs to surface here, not in Stage 7 when debug mode is also being
  exercised.
- Armijo's curve stretches ~3–6× horizontally on the eval-count axis (visible
  per-iter cost difference);
- BB and Fixed barely move between iter and eval axes (1 eval per iter);
- the BB vs Armijo gap *widens* on the eval-count axis — the whole point of the
  fair-comparison plot.

**Caveat on the byte-identical assertion.** Stage 1 used short names (`"Armijo"`)
for the per-method rng key; Stage 5 uses the long form
(`"GradientDescent[step_size=Armijo]"`). The hashes differ, so the rng streams
differ. The assertion passes only because no `step!` in this grid draws from rng.
Add a comment in the file: introduce any stochastic step-size component and the
assertion silently becomes vacuous.

**Watch out for:**

- if Armijo's eval count is suspiciously close to its iter count,
  `n_linesearch_evals` is not being incremented inside the backtracking loop;
- if cumulative `core_time_ns` is roughly equal across all methods, `@core_timed`
  is wrapping too much (likely the whole `step!` instead of just the kernel);
- a method name with no registered abbreviation falls back to the long form in
  legends — correct but ugly. Populate `ABBREVIATIONS` deliberately.

---

## Stage 6 — Multi-run with randomized x₀ + warm-up

**Status:** new.

Sample x₀ uniformly in [−2, 2]² (registered as a new
`RandomProblem(:rosenbrock_random_x0)`). Set `n_runs = 20`. Plot median curves
with shaded 25–75% IQR via `aggregate_runs(df, :median)`. Add one configuration
with `IterativeWarmup(GradientDescent(FixedStep(α=1e-3)), MaxIterations(50))`.

**Exercises:**

- `RandomProblem` and `register_random_problem!`;
- per-run rng derivation inside `run_experiment`
  (`Xoshiro(hash((seed, run_id, :data)))`);
- `aggregate_runs(df, :median)` and `:all`;
- `NoWarmup` and `IterativeWarmup` dispatch in `run_warmup`;
- the universal `result.final_state.iterate.x` convention used by
  `IterativeWarmup` to read off the warm-started x₀.

**Validates:**

- same seed → byte-identical DataFrames across two invocations of
  `run_experiment` (assert by running twice and diffing);
- different seed → distribution that visibly tightens for stable methods
  (Armijo) and is wider for sensitive ones (BB at small starting distances);
- **warm-up x₀ is shared.** Concrete invariant: `extras[:x_iter]` at `iter = 0`
  is identical across all five methods within any single `run_id` when warm-up
  is active. This is the only test that actually proves the warm-up output is
  shared rather than each method running its own.

**Watch out for:**

- the IQR shading should *contain* the median curve at every iter — if it
  doesn't, the quantile computation is off-by-one;
- `aggregate_runs` on the `:step_size` column produces nonsense for Armijo
  (median of discrete `β^j` values) and meaningless smoothing for Fixed
  (constant). Drop the step-size panel from the multi-run figure, or replace it
  with an unaggregated overlay of the 20 individual curves;
- Rosenbrock from x₁ < 0 wanders for a long time before finding the valley;
  the IQR will be wide for BB methods specifically. This is correct behavior,
  not a bug. Worth flagging in the figure caption.

---

## Stage 7 — Debug mode + extended stopping criteria + range-gated verbosity

**Status:** new.

The "research tooling" stage. Three orthogonal but related observability blocks
bundled into one experiment, since they're all auxiliary verification machinery.
Together they validate everything in `debug.jl`, the remaining `StoppingCriteria`
subtypes, the `:all` composite mode, and the `iter_range` verbosity gate.

### 7.a — Debug mode

Run with all four `DebugCheck`s active at `on_trigger = :warn`:

- `CheckObjectiveMonotonicity(tolerance=0.0)` — fires on BB (~30–60% of iters).
  This is intrinsic BB non-monotonicity, not a bug, and is the validation
  evidence itself.
- `CheckGradientNormBound(max_norm=1e8)` — exercised by adding one configuration
  with `x₀ = (10, 10)` or `ρ = 1e6` so the check actually has something to
  bound.
- `CheckStepDecay(window=20)` — exercised by adding one configuration with
  `FixedStep(α=1e-6)`, where the step norm is constant and tiny.
- `CheckNumericalGradient(epsilon=1e-7, max_error=1e-5)` — passes on Rosenbrock
  at every iteration by default.

Then a separate run with `CheckNumericalGradient` and `on_trigger = :error`,
with `grad!(::RosenbrockObjective, x)` intentionally broken (drop the
`−4ρ x₁ (x₂ − x₁²)` term). Confirm the run halts loudly. Restore the gradient.

One more run at `on_trigger = :log` to confirm silent recording into
`logger.events` works without console output.

### 7.b — Extended stopping criteria

- Add `TimeLimit`, `ObjectiveStagnation`, `StepTolerance` to the `:any`
  composite alongside `MaxIterations`/`GradientTolerance`/`DistanceToOptimal`.
  Verify each `stop_reason` symbol surfaces correctly for at least one method
  (in particular: Fixed will stagnate before max_iter; BB methods will hit
  either gradient or distance criteria first).
- `method_criteria` — give Fixed its own smaller budget
  (`MaxIterations(5_000)`) distinct from the others' larger budget
  (`MaxIterations(50_000)`); verify each method uses its own.
- `CompositeCriteria(:all)` — one configuration with
  `stop_when_all(GradientTolerance(1e-6), StepTolerance(1e-8))`. Both criteria
  eventually hold on a converging method; verify `stop_reason ==
  :all_criteria_met` fires (and not a single component's symbol).

### 7.c — Range-gated verbosity

One configuration with `VerbosityConfig(level=MILESTONE, iter_range=200:300)`.
Visually confirm that DETAILED-level output appears for iters 200–300 only,
with the rest of the run at the MILESTONE fallback.

### Validates

Every part of `debug.jl`; every remaining stopping criterion in `stopping.jl`;
both `:any` and `:all` composite modes; `method_criteria` dispatch; the
`iter_range` branch of `maybe_print`.

### Watch out for

- if `CheckNumericalGradient` *passes* with the intentionally broken gradient,
  the central-difference computation or comparison threshold is wrong —
  restore and investigate before continuing;
- if `:all_criteria_met` never fires, the criteria chosen don't both eventually
  hold; pick a method that converges enough for both `GradientTolerance` and
  `StepTolerance` to be satisfied within the budget;
- if `iter_range` output appears outside the configured range, the gating
  logic in `maybe_print` is broken — and your earlier MILESTONE-level runs
  were noisier than they should have been.

---

After Stage 7, every Rosenbrock-meaningful architectural block has been
validated. Remaining untested blocks need different problem types — see
`Experiment_TODOs.md`.
