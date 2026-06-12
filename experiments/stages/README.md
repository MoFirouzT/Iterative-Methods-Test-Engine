# Development Stages — the Rosenbrock build log

![Five GradientDescent step-size variants tracing Rosenbrock's curved valley toward x*, overlaid on log-spaced contours, with a zoom inset near the optimum](figures/stage2_trajectories.png)

*Stage 2: five step-size rules (Fixed, Armijo, Cauchy, BB1, BB2) navigating the
banana valley from x₀ = (−1.2, 1). Regenerate with `julia --project=. experiments/stages/stage2.jl`.*

The engine wasn't built all at once. It grew capability-by-capability across nine
stages on a single 2D Rosenbrock problem (ρ = 100, x₀ = (−1.2, 1) unless noted) —
each stage validating one architectural block before the next depends on it. This
directory is that build log; it is **development scaffold, not a portfolio result**.

The curated, problem-named experiments that produce the figures in the top-level
README live one level up (`experiments/exp_<problem>N.jl`) and are the project's
headline deliverables. These stages are the rehearsal behind them.

> The build-up is intentional. Stages 0–4 hand-roll the per-method RNG derivation
> and run loop, rehearsing the orchestrator's contract before depending on it.
> Stage 5 is the orchestrator's debut; Stages 6–8 build on it. After Stage 7,
> every block that *can* be validated on Rosenbrock has been; the rest require
> other problem types (see `../Experiment_TODOs.md`).

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
**File:** `stage1.jl`.

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
**File:** `stage2.jl`.

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
**File:** `stage3.jl`.

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

**Status:** done.
**File:** `stage4.jl`.

Same five methods, now stopping on `stop_when_any(MaxIterations(20_000), DistanceToOptimal(1e-8), GradientTolerance(1e-10))`.
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
  to be a real signal — unlike Stage 0, where the kernel is below the noise floor.
  If the ratio is too low, either widen `@core_timed`'s scope in
  `step!` to include the norm/copy bookkeeping currently outside it, or the
  problem is too small to make the kernel dominate.
  If too high, `@core_timed` is sweeping in non-kernel work (logger, stopping check).

**Bug surface to watch.** If `dist_final` is large for a method that stopped with
`:optimal_reached`, the runner is not updating `state.metrics.dist_to_opt` before
the stopping check.

---

## Stage 5 — Orchestrator debut: `run_experiment` + `VariantGrid` + fair-comparison plots

**Status:** done.
**File:** `stage5.jl`.

First experiment that drives the framework through its actual user-facing entry
point — `run_experiment` — rather than a hand-rolled loop.
Defines the same five methods via a `VariantAxis(:step_size, ...)` grid, runs through the orchestrator,
then layers fair-comparison plots on top.

**Framework gaps filled during this stage (landed):**

- `register_abbreviation!(full, short)` added to [src/variants.jl](../src/variants.jl)
  \+ exported. Stage 5 needs it to register `"GradientDescent" => "GD"`,
  `"BarzilaiBorwein" => "BB"`, etc., before `expand(grid)` runs, so that the
  generated `short_name` of each `VariantSpec` uses the friendly form rather
  than the long type name.
- `VariantSpec.method` field widened from `ExperimentalMethod` to `IterativeMethod`.
  The original type was too narrow — the docstring/Stage 5 intent says a
  `VariantGrid` can produce either `Conventional` or `Experimental` methods,
  and Stage 5's `GradientDescent` step-size grid produces the former. With
  the old type, `VariantSpec(...; method = GradientDescent(...))` raised a
  `MethodError: Cannot convert ... to ExperimentalMethod` at expand time.
- `ExperimentConfig.conventional_methods` now defaults to `ConventionalMethod[]`.
  Stage 5 drives the orchestrator entirely through `variant_grids`; the
  previous required-kwarg signature forced every experiment to pass an empty
  `conventional_methods=[]` boilerplate.
- `resolve_methods` now routes each expanded `VariantSpec` into the
  conventional vs experimental bucket based on `spec.method`'s concrete
  type, instead of unconditionally appending to `experimental` (which would
  have wrongly classified Stage 5's `GradientDescent` variants).

**Exercises:**

- `VariantAxis`, `VariantGrid`, `expand`, `VariantSpec` (Cartesian expansion and auto-naming);
- `ABBREVIATIONS` registry and `register_abbreviation!`;
- `resolve_methods` — routes each produced variant into the conventional bucket based on its concrete type;
- `run_experiment` orchestration loop, including its own deterministic per-`(seed, run_id, name)` rng derivation;
- `n_linesearch_evals` accounting inside Armijo's backtracking loop;
- per-iteration `core_time_ns` accumulation in `Logger.total_core_ns`.

**Plots.** Stage 1's four panels reproduced *three times*:

1. x-axis = iter (baseline, identical to Stage 1's figure);
2. x-axis = cumulative function evaluations (per-iter base cost plus `n_linesearch_evals` where applicable);
3. x-axis = cumulative `core_time_ns`, annotated with a noise-floor warning —
   the kernel is tens of nanoseconds on 2D Rosenbrock, OS jitter shifts
   cumulative time by 5–10%, so this plot is sanity-check only, not for
   ordering decisions.

**Validates:**

- **Byte-identical iter logs vs Stage 1.** Programmatic check:
  load both experiments, build their DataFrames, assert equality on `:iter`, `:objective`,
  `:gradient_norm`, `:dist_to_opt` for every method.
  If they drift, something in the orchestrator's setup path differs from the hand-rolled version — and that
  bug needs to surface here, not in Stage 7 when debug mode is also being exercised.
- Armijo's curve stretches ~3–6× horizontally on the eval-count axis (visible per-iter cost difference);
- BB and Fixed barely move between iter and eval axes (1 eval per iter);
- the BB vs Armijo gap *widens* on the eval-count axis — the whole point of the fair-comparison plot.

**Caveat on the byte-identical assertion.** Stage 1 used short names (`"Armijo"`)
for the per-method rng key; Stage 5 uses the long form (`"GradientDescent[step_size=Armijo]"`).
The hashes differ, so the rng streams differ.
The assertion passes only because no `step!` in this grid draws from rng.
Add a comment in the file: introduce any stochastic step-size component and the
assertion silently becomes vacuous.

**Watch out for (stage5):**

- if Armijo's eval count is suspiciously close to its iter count,
  `n_linesearch_evals` is not being incremented inside the backtracking loop;
- if cumulative `core_time_ns` is roughly equal across all methods, `@core_timed`
  is wrapping too much (likely the whole `step!` instead of just the kernel);
- a method name with no registered abbreviation falls back to the long form in
  legends — correct but ugly. Populate `ABBREVIATIONS` deliberately.

---

## Stage 6 — Multi-run with randomized x₀ + warm-up

**Status:** done.
**File:** `stage6.jl`.

Sample x₀ uniformly in [−2, 2]² (registered as a new `RandomProblem(:rosenbrock_random_x0)`). Set `n_runs = 20`.
Plot median curves with shaded 25–75% IQR via `aggregate_runs(df, :median)`.
Add one configuration with `IterativeWarmup(GradientDescent(FixedStep(α=1e-3)), MaxIterations(50))`.

**Framework gaps filled during this stage (landed):**

- `WarmupStrategy` hierarchy — `NoWarmup`, `IterativeWarmup`, `FunctionWarmup`
  — plus the `run_warmup` dispatch added to [src/experiment.jl](../src/experiment.jl).
  `ExperimentConfig` gains a `warmup::WarmupStrategy` field defaulting to
  `NoWarmup()`. The orchestrator derives a per-run warm-up rng
  `Xoshiro(hash((seed, run_id, :warmup)))`, calls `run_warmup` once, and
  rebuilds the `Problem` with the returned x₀ so every method in that run
  sees the same starting point.
- `register_warmup!(name, gen)` + `WARMUP_FUNCTIONS` registry exported
  alongside, so `FunctionWarmup(:name)` strategies are serialisation-safe
  (the strategy carries only the registered symbol).
- `log_init!` now emits an `iter == 0` entry via `extract_log_entry`,
  capturing the initial state (`x_iter`, objective, gradient_norm, …) in
  `iter_logs[1]`. Without this, the warm-up-x₀-shared invariant could not
  be checked at the data level — the smallest available iter was 1, which
  is already post-first-step where methods diverge by step rule. Stages 1
  and 3 already filter `iter == 0` from the step-size panel because α₀ = 0,
  and the Stage-5 byte-identity check still passes since both experiments
  now carry the extra row identically.

**Exercises:**

- `RandomProblem` and `register_random_problem!`;
- per-run rng derivation inside `run_experiment` (`Xoshiro(hash((seed, run_id, :data)))`);
- `aggregate_runs(df, :median)` and `:all`;
- `NoWarmup` and `IterativeWarmup` dispatch in `run_warmup`;
- the universal `result.final_state.iterate.x` convention used by `IterativeWarmup` to read off the warm-started x₀.

**Validates:**

- same seed → byte-identical DataFrames across two invocations of `run_experiment` (assert by running twice and diffing);
- different seed → distribution that visibly tightens for stable methods
  (Armijo) and is wider for sensitive ones (BB at small starting distances);
- **warm-up x₀ is shared.** Concrete invariant: `extras[:x_iter]` at `iter = 0`
  is identical across all five methods within any single `run_id` when warm-up is active.
  This is the only test that actually proves the warm-up output is shared rather than each method running its own.

**Watch out for (stage6):**

- the IQR shading should *contain* the median curve at every iter — if it
  doesn't, the quantile computation is off-by-one;
- `aggregate_runs` on the `:step_size` column produces nonsense for Armijo
  (median of discrete `β^j` values) and meaningless smoothing for Fixed (constant).
  Drop the step-size panel from the multi-run figure, or replace it with an unaggregated overlay of the 20 individual curves;
- Rosenbrock from x₁ < 0 wanders for a long time before finding the valley;
  the IQR will be wide for BB methods specifically.
  This is correct behavior, not a bug. Worth flagging in the figure caption.

---

## Stage 7 — Debug mode + extended stopping criteria + range-gated verbosity

**Status:** done.
**File:** `stage7.jl`.

The "research tooling" stage.
Three orthogonal but related observability blocks bundled into one experiment, since they're all auxiliary verification machinery.
Together they validate everything in `debug.jl`, the remaining `StoppingCriteria`
subtypes, the `:all` composite mode, and the `iter_range` verbosity gate.

**Framework gaps filled during this stage (landed):**

- `DebugConfig.checks` default in [src/debug.jl](../src/debug.jl) was
  `Any[nothing]`, which made `run_debug_checks!` iterate a single `nothing`
  and `MethodError` on the first call. Now defaults to `DebugCheck[]`,
  and the abstract `DebugCheck` type is declared before `DebugConfig` so
  the type annotation resolves.
- `ExperimentConfig` gains a `debug::DebugConfig` field (default
  `DebugConfig()` — disabled). The orchestrator forwards it to
  `run_method` as a keyword.
- `run_method` ([src/core.jl](../src/core.jl)) accepts a `debug` keyword
  (untyped to avoid a hard dep on `src/debug.jl`). After each `log_iter!`,
  when `debug !== nothing && debug.enabled`, it calls
  `run_debug_checks!(debug, logger, state, problem, entry, prev_entry, iter)`
  with the previous `IterationLog` so monotonicity and step-decay checks
  have a window to compare against. Stages 1–6's positional callers stay
  byte-compatible (the keyword defaults to `nothing`).
- TestEngine include order: `debug.jl` now precedes `experiment.jl` so
  `ExperimentConfig`'s `debug::DebugConfig` field resolves at parse time.

**A note on Stage 7.a.2's gradient-norm bound.** The plan suggested a bound
of 1e8 with ρ = 1e6 (or x₀ = (10, 10)). At x₀ = (−1.2, 1) with ρ = 1e6 the
empirical ‖∇f‖ peaks around 2.3e6 — below 1e8, so the check never fires
under that configuration. The experiment uses `max_norm = 1e6` with the
same ρ to make the bound a binding signal. An equivalent fix would be to
move x₀ further from the valley (e.g. (10, 10) with ρ = 1e6 gives
‖∇f‖ ≈ 3.6e9, which clears 1e8 comfortably).

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

**Watch out for (stage7):**

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

## Stage 8 — Cross-cutting validations

**Status:** done.
**File:** `stage8.jl`.

The four "doesn't fit a single earlier stage" checks called out in
`Experiment_TODOs.md` § "Cross-cutting validations not yet covered". All
Rosenbrock-only; no new problem family is needed. Bundled into one script
because each is small.

**Framework gaps filled during this stage (landed):**

- `MethodResult` gains an `events::Vector{NamedTuple}` field
  ([src/experiment.jl](../src/experiment.jl)) populated by `finalize!` from
  `logger.events` ([src/logging.jl](../src/logging.jl)). Before this, every
  event recorded during a run — the stopping event from `log_event!`, debug
  events emitted by `:log`-mode checks — was dropped at `finalize!` and
  never reached the persistence layer. A backward-compatible 5-arg
  constructor preserves the previous call sites. `ExperimentResult`
  roundtrips events through JLD2 automatically.
- `debug_check!` for `CheckObjectiveMonotonicity`, `CheckGradientNormBound`,
  and `CheckNumericalGradient` ([src/debug.jl](../src/debug.jl)) now
  forwards the `logger` keyword to `trigger_debug!`. Previously these
  invoked `trigger_debug!(cfg, iter, msg)` positionally, so on
  `on_trigger = :log` they fell through `trigger_debug!`'s "no logger
  available" branch and printed to console anyway — i.e. Stage 7.a.5 was
  *not* actually silent before this fix, and Stage 8.a's events would have
  been empty. Only `CheckStepDecay` had been forwarding the logger.

### 8.a — `logger.events` roundtrip

`save_experiment` → `load_experiment` on an experiment whose
`DebugConfig.on_trigger = :log`. Asserts that the loaded `MethodResult`
carries a non-empty `events` vector, that the stopping event is present,
and that at least one `:debug` event survived the trip on BB1 (where
`CheckObjectiveMonotonicity` is guaranteed to fire).

### 8.b — `aggregate_runs(df, :all)`

Run 3 runs of the five-method grid, build the DataFrame, call
`aggregate_runs(df, :all)`, and assert shape + content equality with the
input. Sanity-checks `:median` against `:all` (the median frame collapses
the `run_id` dimension; the row-count drop is part of the assertion).

### 8.c — `method_color` registry in-session contract

`METHOD_COLOR_REGISTRY` is a process-global `Dict` — intentionally not
persisted into the experiment manifest. The validation asserts the
in-session contract: after `register_method_color!`, `get_method_color`
returns the registered value, and that value is unaffected by an
intervening `load_experiment` of an unrelated run. Tidies up the test
fixture (`delete!(METHOD_COLOR_REGISTRY, name)`) so re-runs see a clean
slate.

### 8.d — `run_sub_method` invocation shape

The sub-method machinery (`run_sub_method` / `SubResult` /
`attach_sub_logs!`) is exported and covered by `test/runtests.jl` but no
stage was using it. Invokes `run_sub_method` once against a manually
constructed outer logger and asserts:

- `SubResult.n_iters == 1`, `stop_reason == :max_iterations`
  (`MaxIterations(n=1)` was the criterion);
- `SubResult.iter_logs` has length 2 (`iter == 0` init + `iter == 1` step;
  matches the Stage-6 log-init invariant);
- the outer logger's `pending_sub_logs` is populated (`attach_sub_logs!`
  fired because `log_sub_iters = true`);
- the final iterate has actually moved off `x₀`.

Does not rework `IterativeWarmup` to ride on top of `run_sub_method` — the
TODOs file lists that as an option, but the warm-up runs *before* any
outer method exists, so the natural "outer logger" for `attach_sub_logs!`
would have to be invented. Out of scope for Stage 8.

**Watch out for (stage8):**

- if 8.a's BB1 debug count is 0, a `debug_check!` method is invoking
  `trigger_debug!` without the `logger` keyword again — fix in
  [src/debug.jl](../src/debug.jl);
- if 8.c says the registry was wiped after `load_experiment`, persistence
  is silently mutating global state — the registry should remain entirely
  in-memory;
- if 8.d's `pending_sub_logs` is empty, `attach_sub_logs!` is not being
  called from inside `run_sub_method`, or `log_sub_iters = true` isn't
  flowing through.

---

After Stage 8, every Rosenbrock-meaningful architectural block has been validated.
Remaining untested blocks need different problem types — see `Experiment_TODOs.md`.
