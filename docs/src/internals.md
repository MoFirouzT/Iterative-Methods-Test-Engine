# Repository Internals
## Directory & File Structure

The **engine** (`src/`) contains only abstractions, machinery, and dependency-free
utilities — no concrete problem, method, or regularizer. Concrete **content** lives under
`algorithms/` and `problems/`, extends the engine via `import .TestEngine`, and is
assembled together with the engine by `experiments/_bootstrap.jl` (which experiments and
tests `include`). This keeps the engine standalone and dependency-lean.

```
TestEngine.jl/
├── src/
│   ├── TestEngine.jl     # Module entry; includes the src/ engine files only (NO content);
│   │                     #   exports the public API
│   │
│   ├── problems.jl       # Objective / Regularizer / Hessian abstractions (no concrete
│   │                     #   problems/regularizers — those are content); Hessian reps
│   │                     #   (MatrixHessian, OperatorHessian, DiagonalHessian);
│   │                     #   Regularizer; Problem (with optional gs and x_opt) +
│   │                     #   convenience constructors; total_objective; ProblemSpec
│   │                     #   hierarchy (AnalyticProblem, FileProblem with FILE_LOADERS,
│   │                     #   RandomProblem); make_problem dispatch; register_problem!,
│   │                     #   register_file_loader!, register_random_problem!
│   │
│   ├── core.jl           # Abstract types & type hierarchy; state groups
│   │                     #   (IterateGroup, MetricsGroup, TimingGroup); algorithm
│   │                     #   interface (init_state, step!, extract_log_entry);
│   │                     #   @core_timed macro; generic runner (run_method);
│   │                     #   nested infrastructure (SubRunConfig{M}, SubResult{S},
│   │                     #   run_sub_method)
│   │
│   ├── stopping.jl       # StoppingCriteria hierarchy; should_stop dispatch;
│   │                     #   CompositeCriteria; stop_when_any / stop_when_all;
│   │                     #   DistanceToOptimal
│   │
│   ├── variants.jl       # VariantAxis, VariantGrid, VariantSpec; expand(); ABBREVIATIONS;
│   │                     #   register_abbreviation!  (the method/component *vocabulary* —
│   │                     #   StepSize, DescentDirection, MinorUpdate, HessianApprox — is
│   │                     #   content under algorithms/components/, not here)
│   │
│   ├── logging.jl        # IterationLog (incl. dist_to_opt); Logger; log_init!,
│   │                     #   log_iter!, log_event!, attach_sub_logs!, finalize!;
│   │                     #   elapsed_core_s, elapsed_wall_s; VerbosityLevel,
│   │                     #   VerbosityConfig, maybe_print()
│   │
│   ├── experiment.jl     # ExperimentConfig; ExperimentResult / RunResult /
│   │                     #   MethodResult{S}; WarmupStrategy (NoWarmup,
│   │                     #   IterativeWarmup, FunctionWarmup); run_warmup();
│   │                     #   resolve_methods(); run_experiment();
│   │                     #   next_experiment_path()
│   │
│   ├── persistence.jl    # save_experiment(); load_experiment(); load_manifest();
│   │                     #   list_experiments(); CSV sidecar writer
│   │
│   ├── debug.jl          # DebugConfig; DebugCheck hierarchy
│   │                     #   (CheckObjectiveMonotonicity, CheckGradientNormBound,
│   │                     #   CheckStepDecay, CheckNumericalGradient);
│   │                     #   run_debug_checks!; debug_check! dispatch;
│   │                     #   trigger_debug!; numerical_gradient()
│   │
│   └── analysis.jl       # to_dataframe(); filter_methods(); aggregate_runs();
│                         #   MethodStyle; METHOD_PALETTE; METHOD_COLOR_REGISTRY;
│                         #   get_method_color(); register_method_color!;
│                         #   PlotSpec; FigureLayout; render_figure(); save_figure()
│
├── algorithms/                   # CONTENT — concrete methods + shared components
│   ├── components/               #   shared method-construction vocabulary (extend engine)
│   │   ├── descent_directions.{md,jl}   # DescentDirection, SteepestDescent, compute_direction
│   │   ├── step_sizes.{md,jl}           # StepSize/LineSearch; Fixed/Armijo/Cauchy/BB
│   │   ├── minor_updates.jl             # MinorUpdate + NoMinorUpdate/Momentum/Nesterov
│   │   │                                #   + extrapolate / advance_momentum behavior (FISTA)
│   │   └── preconditioners.{md,jl}      # Preconditioner + Identity/Jacobi; precondition()
│   ├── conventional/
│   │   ├── gradient_descent.jl
│   │   ├── proximal_gradient/    # proximal_gradient.{md,jl} — ProximalGradient (ISTA/FISTA)
│   │   └── trust_region/         # trust_region.{md,jl} — QuadraticModel + SteihaugCG + TrustRegion
│   └── experimental/
│       └── preconditioned_gradient/ # preconditioned_gradient.{md,jl} — PreconditionedGradient
│
├── problems/                     # CONTENT — concrete problem families (self-register on load)
│   ├── rosenbrock/               # rosenbrock.{md,jl} — RosenbrockObjective; :rosenbrock
│   ├── least_squares/            # least_squares.{md,jl} — LeastSquares (selectable
│   │                             #   :matrix/:operator Hessian); :quadratic + :linear_ls
│   ├── lasso/                    # lasso.{md,jl}     — :lasso sparse-recovery generator
│   ├── separable_quadratic/      # separable_quadratic.{md,jl} — :separable_quadratic (DiagonalHessian)
│   └── regularizers/             # regularizers.jl   — L1/L2/Zero, prox via ProximalOperators.jl
│
├── experiments/                  # load engine + content via _bootstrap.jl
│   ├── README.md                # folder guide: portfolio track, stages, planned work
│   ├── _bootstrap.jl             # assembles engine (TestEngine) + all content, in order
│   ├── _shared.jl                # shared plotting helpers (Rosenbrock trajectory figure)
│   ├── exp_lasso_ista_fista.jl            # portfolio experiment: lasso (flagship)
│   ├── exp_ls1_dimension.jl               #   ls1: dimension scaling + timing pillar
│   ├── exp_ls2_conditioning.jl            #   ls2: GD rate vs κ (slope 1 vs √κ)
│   ├── exp_precond_grid.jl                #   precond: VariantGrid + dual routing; Jacobi≈Newton
│   ├── exp_tr_steihaug_cg.jl              #   tr: TrustRegion + Steihaug-CG (nested optimization)
│   └── stages/                            # engine build log: capability-by-capability validation
│       ├── stage1.jl … stage8.jl          #   per-stage validation of one architectural block each
│       ├── smoke_test.jl                  #   Stage 0: end-to-end runner-contract smoke check (in CI)
│       ├── figures/                       #   stage2_trajectories.png (surfaced in the README)
│       └── README.md                      #   the build-log writeup
│
├── logs/                         # Git-ignored; written at runtime
│   └── <date>/<NNN>/             #   manifest.json, result.jld2, run{N}_{method}.csv
│
└── test/                         # load engine + content via ../experiments/_bootstrap.jl
    ├── runtests.jl               # step sizes, core runner, nesting, variant grid; includes ↓
    ├── test_module5.jl           # experiment orchestration (resolve_methods, run_experiment)
    ├── test_module7.jl           # verbosity system
    ├── test_module8.jl           # persistence (save/load, manifest, CSV)
    ├── test_module9.jl           # problem factory: LeastSquares / regularizer content
    ├── test_proximal_gradient.jl # ProximalGradient: ISTA↔GD reduction, FISTA acceleration
    ├── test_least_squares.jl     # Hessian modes, :linear_ls conditioning, Cauchy-guard regression
    ├── test_preconditioned_gradient.jl # Jacobi=Newton, dual-bucket routing, diagonal contract
    ├── test_external_validation.jl # cross-check solutions vs Optim.jl + ProximalAlgorithms.jl
    └── test_trust_region.jl      # Steihaug-CG branches, TR convergence, nesting + core-time attribution
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│  DEFINITION PHASE                                                       │
│                                                                         │
│  problems.jl        Objective / Hessian / Regularizer / Problem        │
│       │             AnalyticProblem / FileProblem / RandomProblem       │
│       │                                                                 │
│  variants.jl        VariantAxis(:step_size, FixedStep=>"Fix", ...)     │
│       │                   │                                             │
│       └──────────► VariantGrid → expand() → [VariantSpec, ...]         │
│                                     │                                   │
│  stopping.jl        StoppingCriteria (per-experiment or per-method)    │
│                                     │                                   │
│                         ExperimentConfig                                │
│                    (+ warmup :: WarmupStrategy)                         │
│                    (+ debug  :: DebugConfig)                            │
└─────────────────────────────────────┬───────────────────────────────────┘
                                      │
┌─────────────────────────────────────▼───────────────────────────────────┐
│  EXECUTION PHASE                                                        │
│                                                                         │
│         run_experiment(config, log_root)                                │
│                  │                                                      │
│         next_experiment_path()  →  logs/YYYYMMDD/NNN/ (atomic mkdir)   │
│         resolve_methods()       →  routes by ConventionalMethod /      │
│                                     ExperimentalMethod                  │
│                  │                                                      │
│   ┌──── WARM-UP (once per run, if configured) ──────────────────────┐   │
│   │  rng_warmup = Xoshiro(hash((seed, run_id, :warmup)))            │   │
│   │  x0_warm   = run_warmup(config.warmup, problem, rng_warmup, …)  │   │
│   │  problem   = Problem(…, x0=x0_warm, …)   ← shared by all       │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                  │                                                      │
│        for each run × method:                                           │
│         method_rng = Xoshiro(hash((seed, run_id, method_name)))         │
│                  │                                                      │
│      ┌───────────▼────────────┐                                         │
│      │    run_method()        │  ◄── Logger + Criteria + DebugConfig    │
│      │  init_state(…, rng)    │                                         │
│      │  while true:           │                                         │
│      │    core_time_ns = 0    │                                         │
│      │    step!(…, logger,    │  ◄── logger & rng explicit params       │
│      │          rng)          │  ◄── @core_timed inside step!           │
│      │    dist_to_opt         │  ──► runner computes from x_opt         │
│      │    extract_log_entry() │  ──► entry.dist_to_opt copied           │
│      │    log_iter!()         │  ──► logger.total_core_ns accumulated   │
│      │    run_debug_checks!() │  ──► checks fire → warn/error/log       │
│      │    should_stop()       │  ◄── DistanceToOptimal reads dist_to_opt│
│      └───────────┬────────────┘                                         │
│                  │  (if nested algorithm used)                          │
│      ┌───────────▼────────────┐                                         │
│      │  run_sub_method()      │  ◄── SubRunConfig{M}                    │
│      │  sub_rng = Xoshiro(    │  ──► child RNG derived from outer_rng   │
│      │    rand(outer_rng,…))  │  ──► independent core time tracking     │
│      │  step!(…, sub_logger,  │  ──► sub logs → outer IterationLog      │
│      │         sub_rng)       │                                         │
│      └───────────┬────────────┘                                         │
│                  │                                                      │
│           finalize!() → MethodResult{S}                                 │
│           collected into RunResult → ExperimentResult                   │
└─────────────────────────────────────┬───────────────────────────────────┘
                                      │
┌─────────────────────────────────────▼───────────────────────────────────┐
│  PERSISTENCE PHASE                                                      │
│                                                                         │
│   save_experiment()                                                     │
│       ├── result.jld2              (full binary, fast reload)           │
│       ├── run{N}_{method}.csv      (per-method, human-readable,        │
│       │                            includes dist_to_opt column)        │
│       └── manifest.json           (name, metadata, no binary needed)   │
└─────────────────────────────────────┬───────────────────────────────────┘
                                      │
┌─────────────────────────────────────▼───────────────────────────────────┐
│  ANALYSIS PHASE                                                         │
│                                                                         │
│   load_experiment()  ──►  to_dataframe()  (incl. :dist_to_opt col)     │
│                                │                                        │
│                      filter_methods() / aggregate_runs()                │
│                                │                                        │
│                      user transforms   (DataFrame -> DataFrame)         │
│                                │                                        │
│                       METHOD_COLOR_REGISTRY + MethodStyle               │
│                       PlotSpec / FigureLayout                           │
│                                │                                        │
│                      render_figure()  ──►  save_figure(.pdf / .png)    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Key Architectural Decisions

| Decision | Rationale |
|----------|-----------|
| `Objective` abstract type (renamed from `DataFidelity`) | Honest about what the abstraction is — the (typically smooth) main objective term — without inverse-problems jargon that misleads on problems like Rosenbrock |
| `gs :: Vector{Regularizer}` with empty default + convenience constructors | Pure (non-composite) problems remain expressible as `Problem(f, x0)`; composite problems carry explicit `gs`; no `Union{Nothing,Vector}` indirection |
| `Hessian` abstract type with `apply` / `materialize` / `diagonal` interface | Unifies exact Hessians, full quasi-Newton (BFGS, SR1), L-BFGS, and structured forms under one dispatch surface; each concrete type declares which operations are available |
| `StepSize` umbrella with `LineSearch <: StepSize` subset | Type-honest: closed-form rules (Fixed, BB, Cauchy) and genuine 1D searches (Armijo) are both valid step-size strategies; the subset gives a real dispatch handle for code that needs to specifically target line searches |
| `StoppingCriteria` hierarchy replaces `converged` + `for` loop | Full control over termination: count, time, tolerance, composites, all independently testable |
| Stopping criteria separated from algorithm struct | Same algorithm, different run budgets across experiments; no code changes required |
| `@core_timed` in algorithm code, exception-safe, accumulates into `state.timing.core_time_ns` | Scientific discipline: only the kernel is measured; bookkeeping invisible to the clock; error recovery does not corrupt timing |
| `log_iter!` accumulates `entry.core_time_ns` into `logger.total_core_ns` | `TimeLimit` reads `elapsed_core_s(logger)` — per-iteration core time is logged, then summed; wall-clock never used as a stopping criterion |
| Logger passed as explicit `step!` parameter — not stored in state | Algorithm code is pure: no logging infrastructure in state structs; logger strategy controlled entirely by the runner |
| `step!(method, state, problem, iter, logger, rng)` extended signature | Logger and rng are injected by the runner and forwarded by algorithms to `run_sub_method` — clean, testable, no hidden state |
| Four canonical state groups (`IterateGroup`, `MetricsGroup`, `TimingGroup`, method-specific `Numerics`) | Clean separation of concerns; sub-routines can receive independent groups; `extract_log_entry` default is trivial; no field duplication permitted |
| `VariantGrid.builder` returns `IterativeMethod` (not `ExperimentalMethod`) | Grids work uniformly for conventional and experimental methods; `resolve_methods` routes each produced method to the right bucket based on its concrete type |
| `SubRunConfig{M}` parametric over method type | Type-stable `init_state` → `SubResult{S}` with concrete `S` → type-stable `final_state` access in `step!` |
| Child RNG `Xoshiro(rand(outer_rng, UInt64))` in `run_sub_method` | Deterministic, reproducible sub-runs; independent of outer rng's future draws |
| Per-method RNG `Xoshiro(hash((seed, run_id, name)))` | Adding or removing a method does not alter any other method's RNG stream; full between-run and between-method independence |
| `MethodResult{S}` parametric over state type | Concrete `final_state` type preserved through the result hierarchy; `finalize!` is type-stable; warm-up can access `result.final_state.iterate.x` without Any-dispatch |
| `Problem.x_opt` set by generator; `dist_to_opt` computed by runner | Algorithms are unaware of optimality tracking; `DistanceToOptimal` criterion and `dist_to_opt` logging activate automatically when `x_opt` is non-nothing |
| `DistanceToOptimal` returns `(false, :none)` when `x_opt` is nothing | Criterion is safe to include in any stopping config regardless of problem type; never fires spuriously |
| `WarmupStrategy` hierarchy with `run_warmup` dispatch | Warm-up is optional, declarative, and serializable; warm-up result (x0) is shared across all methods in a run |
| `IterativeWarmup` calls `run_method` and extracts `final_state.iterate.x` | Warm-up reuses the full runner machinery (logging, stopping, debug); relies on universal `iterate :: IterateGroup` convention |
| `FunctionWarmup` uses `name :: Symbol` + `WARMUP_FUNCTIONS` registry | Pure-function warm-ups remain JLD2-serializable |
| `FileProblem.loader_name :: Symbol` + `FILE_LOADERS` registry | Raw functions are not JLD2-serializable; symbol reference is |
| `DebugConfig` in `ExperimentConfig`; checks run by runner after `log_iter!` | Debug mode is orthogonal to algorithms, verbosity, and logging; disabled by default; zero cost when `enabled = false` |
| `DebugCheck` dispatch with `prev_entry` parameter | Checks that require two consecutive entries (e.g. `CheckObjectiveMonotonicity`) receive both; first-iteration check is a no-op via `isnothing(prev)` guard |
| `ObjectiveStagnation` uses direct index access instead of slice | Eliminates per-check array allocation |
| `next_experiment_path` uses atomic `mkdir` | Eliminates TOCTOU race when two processes write to the same log root simultaneously |
| Verbosity colocated with logging in `logging.jl` | Both share the `Logger` struct; separating into its own file would create artificial coupling |
| `aggregate_runs` modes `:all`, `:mean`, `:median` | `:all` preserves every run for full distribution; `:mean`/`:median` reduce to a representative curve; `:best` omitted — cherry-picking runs has no sound benchmarking interpretation |
| `FigureLayout` as `Matrix{Union{PlotSpec,Nothing}}` | Any grid formation expressible as a Julia matrix literal; blank cells are `nothing`; arbitrary sizes |
| Transforms as `DataFrame -> DataFrame` | No DSL to learn; composable with DataFramesMeta; independently unit-testable |
| CSV sidecar holds scalar extras only; vector / composite extras are JLD2-only | CSV is for grep-able tabular data; vectors and composites have no stable text encoding. JLD2 captures everything; the manifest records which keys were dropped so the omission is auditable without loading the binary. |

---

