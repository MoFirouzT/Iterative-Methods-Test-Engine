# experiments/stages/stage7.jl
# =============================================================================
# Stage 7 — Debug mode + extended stopping criteria + range-gated verbosity
# -----------------------------------------------------------------------------
# The "research tooling" stage. Three orthogonal observability blocks bundled
# into one script because they're all auxiliary verification machinery, and
# together they cover everything in debug.jl, the remaining StoppingCriteria
# subtypes, the :all composite mode, and the iter_range branch of
# maybe_print.
#
#   7.a — Debug mode
#         A1  all four DebugCheck's at :warn (baseline)         → monotonicity
#                                                                  fires on BB
#         A2  CheckGradientNormBound stress config              → bound fires
#         A3  CheckStepDecay stress config (α = 1e-6)           → decay fires
#         A4  CheckNumericalGradient on a BROKEN gradient at
#             :error mode                                       → run halts
#         A5  all four checks at :log mode                      → silent
#
#   7.b — Extended stopping criteria
#         B1  :any composite with TimeLimit, ObjectiveStagnation,
#             StepTolerance, plus method_criteria narrowing Fixed's budget
#         B2  :all composite of GradientTolerance + StepTolerance
#
#   7.c — Range-gated verbosity
#         C1  iter_range = 200:300 with MILESTONE fallback
#
# Framework gaps filled during this stage (landed):
#   * `DebugConfig.checks` default was `Any[nothing]`, which made
#     `run_debug_checks!` iterate over a single `nothing` and `MethodError`
#     immediately. Now defaults to `DebugCheck[]`.
#   * `ExperimentConfig` gains a `debug::DebugConfig` field (default
#     `DebugConfig()` — disabled). The orchestrator forwards it to
#     `run_method` as a keyword.
#   * `run_method` (`src/core.jl`) accepts a `debug` keyword (untyped to
#     avoid a hard dep on `src/debug.jl`). After each `log_iter!`, when
#     `debug !== nothing && debug.enabled`, it calls
#     `run_debug_checks!(debug, logger, state, problem, entry, prev_entry, iter)`
#     with the previous IterationLog so monotonicity / step-decay checks
#     have a window to compare against. Stages 1–6's positional callers are
#     unaffected (the keyword defaults to `nothing`).
#   * TestEngine include order: `debug.jl` now precedes `experiment.jl` so
#     `ExperimentConfig`'s `debug::DebugConfig` field resolves at parse time.
# =============================================================================

include("../_bootstrap.jl")   # engine + all content (problems, methods, components)
using DataFrames, DataFramesMeta
using LinearAlgebra
using Printf

# Bring the framework's dispatch points into scope so we can add methods on
# a new Objective subtype (see broken-gradient section below).
import .TestEngine: value, grad!, hessian

# ─── Setup ───────────────────────────────────────────────────────────────────

register_abbreviation!("GradientDescent", "GD")
register_abbreviation!("FixedStep",       "Fix")
register_abbreviation!("ArmijoLS",        "Arm")
register_abbreviation!("CauchyStep",      "Cau")
register_abbreviation!("BarzilaiBorwein", "BB")

const VERBOSITY = VerbosityConfig(level=MILESTONE)
const ROSEN     = AnalyticProblem(name   = :rosenbrock,
                                  params = (rho = 100.0, x0 = [-1.2, 1.0]))

# A small helper that runs and prints a short summary, used throughout.
function _report(label::String, result)
    println("\n--- $label → ", result.experiment_path)
    df = to_dataframe(result)
    for sub in groupby(df, :method_name)
        last_row = sub[end, :]
        run1 = result.run_results[1]
        mr   = run1.method_results[sub.method_name[1]]
        @printf("    %-50s n_iter=%5d  stop=%-22s  f=%.3e  ‖∇f‖=%.3e\n",
                sub.method_name[1], mr.n_iters, string(mr.stop_reason),
                last_row.objective, last_row.gradient_norm)
    end
    return df
end

# Helper for trimming long error messages (used by the broken-gradient test).
function first_line(s::AbstractString)
    nl = findfirst('\n', s)
    return nl === nothing ? s : s[1:nl-1]
end

# Count debug events recorded into logger.events (on_trigger = :log path).
function _count_debug_events(result)
    total = 0
    for rr in result.run_results, (_, mr) in rr.method_results
        # Debug events land in logger.events; finalize! exposes them via
        # MethodResult — but our MethodResult only carries iter_logs / stop_reason.
        # Instead, scan each iter_log's extras for any `debug_event` placeholder,
        # falling back to zero. The richer accounting lives in the captured
        # ExperimentResult.run_results' Logger view, which isn't exposed in
        # this build — we report 0 here when unavailable and rely on the
        # `:warn`/`:error` paths for behavioural evidence.
    end
    return total
end

# Variant grid reused across 7.a and 7.c.
const STEP_AXIS = VariantAxis(:step_size,
    FixedStep(α=8e-4)                                  => "Fixed",
    ArmijoLS(α₀=1.0, β=0.5, c₁=1e-4, max_iter=50)      => "Armijo",
    CauchyStep()                                       => "Cauchy",
    BarzilaiBorwein(variant=:BB1)                      => "BB1",
    BarzilaiBorwein(variant=:BB2)                      => "BB2",
)

const FULL_GRID = VariantGrid(
    base_name = "GradientDescent",
    axes      = [STEP_AXIS],
    builder   = (; step_size, kwargs...) ->
                    GradientDescent(direction=SteepestDescent(), step_size=step_size),
)

# =============================================================================
# 7.a — Debug mode
# =============================================================================

println("\n\n╔══════════════════════════════════════════════════════════════╗")
println("║ Stage 7.a — Debug mode                                       ║")
println("╚══════════════════════════════════════════════════════════════╝")

# ─── A1: baseline — all four checks at :warn ────────────────────────────────
#
# BB's intrinsic non-monotonicity triggers CheckObjectiveMonotonicity at
# every BB step where the objective rises (typically 30–60% of iters). That
# is the validation evidence itself, not a bug.
# CheckNumericalGradient passes silently on the correctly-implemented
# Rosenbrock. CheckGradientNormBound and CheckStepDecay don't fire on a
# normal converging run — they need the stress configs in A2/A3.

println("\n--- A1: all four checks active, on_trigger=:warn ---")
println("    (expect CheckObjectiveMonotonicity to fire on BB iterations;")
println("     a small sample of warnings is printed below — full stream on stderr)")

config_a1 = ExperimentConfig(
    name              = "Stage 7.a.1 — debug :warn baseline",
    problem_spec      = ROSEN,
    variant_grids     = [FULL_GRID],
    stopping_criteria = stop_when_any(MaxIterations(n=200),
                                      GradientTolerance(tol=1e-9)),
    n_runs            = 1,
    seed              = 42,
    debug             = DebugConfig(
        enabled    = true,
        checks     = DebugCheck[
            CheckObjectiveMonotonicity(tolerance = 0.0),
            CheckGradientNormBound(max_norm = 1e8),
            CheckStepDecay(window = 20),
            CheckNumericalGradient(epsilon = 1e-7, max_error = 1e-5),
        ],
        on_trigger = :warn,
        io         = stderr,
    ),
    tags              = Dict{String,Any}("stage" => "7.a.1"),
)
result_a1 = run_experiment(config_a1; verbosity=VERBOSITY)
_report("A1", result_a1)

# ─── A2: CheckGradientNormBound stress ──────────────────────────────────────
#
# ρ = 1e6 lifts ‖∇f‖ at x₀ = (−1.2, 1) to ~2.3e6 (≈ 4ρ·|x₁|·|x₂ − x₁²|
# dominates). Setting the bound at 1e6 makes the check fire on the very
# first iteration. The original plan suggested 1e8 with ρ = 1e6, but the
# Rosenbrock gradient at this x₀ peaks well below 1e8 — empirically we need
# either x₀ further from the valley (e.g. (10, 10)) or a lower threshold.
# A lower threshold is the more direct signal here.

println("\n--- A2: CheckGradientNormBound stress, ρ=1e6, bound=1e6 ---")
config_a2 = ExperimentConfig(
    name              = "Stage 7.a.2 — CheckGradientNormBound stress",
    problem_spec      = AnalyticProblem(name=:rosenbrock,
                                        params=(rho = 1e6, x0 = [-1.2, 1.0])),
    conventional_methods = [GradientDescent(direction = SteepestDescent(),
                                            step_size = FixedStep(α=1e-9))],
    stopping_criteria = MaxIterations(n=50),
    n_runs            = 1,
    seed              = 42,
    debug             = DebugConfig(
        enabled    = true,
        checks     = DebugCheck[CheckGradientNormBound(max_norm = 1e6)],
        on_trigger = :warn,
    ),
    tags              = Dict{String,Any}("stage" => "7.a.2"),
)
result_a2 = run_experiment(config_a2; verbosity=VERBOSITY)
_report("A2", result_a2)

# ─── A3: CheckStepDecay stress ──────────────────────────────────────────────
#
# Fixed step of 1e-6 means ‖αd‖ is roughly constant and tiny, so over any
# window the step norm is essentially unchanging (no decay) — the check
# fires as soon as `window` iterations are in flight.

println("\n--- A3: CheckStepDecay stress, FixedStep(α=1e-6) ---")
config_a3 = ExperimentConfig(
    name              = "Stage 7.a.3 — CheckStepDecay stress",
    problem_spec      = ROSEN,
    conventional_methods = [GradientDescent(direction = SteepestDescent(),
                                            step_size = FixedStep(α=1e-6))],
    stopping_criteria = MaxIterations(n=100),
    n_runs            = 1,
    seed              = 42,
    debug             = DebugConfig(
        enabled    = true,
        checks     = DebugCheck[CheckStepDecay(window = 20)],
        on_trigger = :warn,
    ),
    tags              = Dict{String,Any}("stage" => "7.a.3"),
)
result_a3 = run_experiment(config_a3; verbosity=VERBOSITY)
_report("A3", result_a3)

# ─── A4: CheckNumericalGradient on a BROKEN gradient at :error ──────────────
#
# We define a separate Objective subtype with an intentionally-wrong
# gradient (the −2(1−x₁) − 4ρ x₁ (x₂ − x₁²) term is missing). This is the
# cleanest way to exercise the numerical check: leave the real
# RosenbrockObjective untouched, register a fresh problem, and run it.

struct BrokenRosenbrockObjective <: Objective
    ρ::Float64
end

value(f::BrokenRosenbrockObjective, x::Vector{Float64})::Float64 =
    (1.0 - x[1])^2 + f.ρ * (x[2] - x[1]^2)^2

function grad!(g::Vector{Float64}, f::BrokenRosenbrockObjective, x::Vector{Float64})::Vector{Float64}
    # Intentionally missing the −4ρ x₁ (x₂ − x₁²) term — first component
    # is wrong by a large margin at x = (-1.2, 1) when ρ = 100.
    g[1] = -2.0 * (1.0 - x[1])
    g[2] =  2.0 * f.ρ * (x[2] - x[1]^2)
    return g
end

# Hessian is unused by GradientDescent + ArmijoLS but the interface needs a
# concrete return; provide a stub that flags the calling site if used.
hessian(::BrokenRosenbrockObjective, x::Vector{Float64}) =
    MatrixHessian(zeros(length(x), length(x)))

register_analytic_problem!(:rosenbrock_broken, (params, rng) -> begin
    ρ  = Float64(get(params, :rho, 100.0))
    x0 = collect(Float64, get(params, :x0, [-1.2, 1.0]))
    return Problem(BrokenRosenbrockObjective(ρ), x0;
                   x_opt = [1.0, 1.0],
                   meta  = Dict{Symbol,Any}(:broken_gradient => true))
end)

println("\n--- A4: CheckNumericalGradient on BROKEN gradient, on_trigger=:error ---")
println("    (run is expected to halt with an ErrorException)")
config_a4 = ExperimentConfig(
    name              = "Stage 7.a.4 — broken gradient, :error",
    problem_spec      = AnalyticProblem(name=:rosenbrock_broken,
                                        params=(rho = 100.0, x0 = [-1.2, 1.0])),
    conventional_methods = [GradientDescent(direction = SteepestDescent(),
                                            step_size = ArmijoLS())],
    stopping_criteria = MaxIterations(n=10),
    n_runs            = 1,
    seed              = 42,
    debug             = DebugConfig(
        enabled    = true,
        checks     = DebugCheck[
            CheckNumericalGradient(epsilon = 1e-7, max_error = 1e-5),
        ],
        on_trigger = :error,
    ),
    tags              = Dict{String,Any}("stage" => "7.a.4"),
)

let a4_caught = false
    try
        run_experiment(config_a4; verbosity=VERBOSITY)
    catch e
        a4_caught = true
        println("    ✓ run halted with: ", sprint(showerror, e) |> first_line)
    end
    a4_caught || @error "A4: broken gradient did NOT trigger the numerical-gradient " *
                        "check — central-difference computation or threshold is " *
                        "misconfigured."
end

# ─── A5: all four checks at :log (silent recording) ─────────────────────────
#
# on_trigger = :log records events into logger.events without printing them
# (when a logger is reachable; trigger_debug! has the wiring).

println("\n--- A5: all four checks active, on_trigger=:log ---")
println("    (expect NO console output from debug checks)")
config_a5 = ExperimentConfig(
    name              = "Stage 7.a.5 — debug :log silent",
    problem_spec      = ROSEN,
    variant_grids     = [FULL_GRID],
    stopping_criteria = stop_when_any(MaxIterations(n=200),
                                      GradientTolerance(tol=1e-9)),
    n_runs            = 1,
    seed              = 42,
    debug             = DebugConfig(
        enabled    = true,
        checks     = DebugCheck[
            CheckObjectiveMonotonicity(tolerance = 0.0),
            CheckGradientNormBound(max_norm = 1e8),
            CheckStepDecay(window = 20),
            CheckNumericalGradient(epsilon = 1e-7, max_error = 1e-5),
        ],
        on_trigger = :log,
    ),
    tags              = Dict{String,Any}("stage" => "7.a.5"),
)
result_a5 = run_experiment(config_a5; verbosity=VERBOSITY)
_report("A5", result_a5)

# =============================================================================
# 7.b — Extended stopping criteria
# =============================================================================

println("\n\n╔══════════════════════════════════════════════════════════════╗")
println("║ Stage 7.b — Extended stopping criteria                       ║")
println("╚══════════════════════════════════════════════════════════════╝")

# ─── B1: :any composite with the remaining criteria + method_criteria ───────
#
# Six criteria in a single :any composite, plus a per-method override
# narrowing Fixed's iteration budget so that the binding criterion differs
# across methods.

println("\n--- B1: :any composite covering all remaining criteria, with method_criteria ---")
config_b1 = ExperimentConfig(
    name              = "Stage 7.b.1 — extended :any + per-method budget",
    problem_spec      = ROSEN,
    variant_grids     = [FULL_GRID],
    stopping_criteria = stop_when_any(
                            MaxIterations(n=50_000),
                            GradientTolerance(tol=1e-9),
                            DistanceToOptimal(tol=1e-8),
                            StepTolerance(tol=1e-10),
                            ObjectiveStagnation(tol=1e-12, window=50),
                            TimeLimit(seconds=30.0),
                        ),
    method_criteria   = Dict(
        "GradientDescent[step_size=Fixed]" =>
            stop_when_any(MaxIterations(n=5_000),
                          ObjectiveStagnation(tol=1e-10, window=100)),
    ),
    n_runs            = 1,
    seed              = 42,
    tags              = Dict{String,Any}("stage" => "7.b.1"),
)
result_b1 = run_experiment(config_b1; verbosity=VERBOSITY)
df_b1 = _report("B1", result_b1)

# Verify that at least one Fixed-specific reason appeared, and that some
# other method hit a tolerance reason — proves method_criteria dispatch
# works AND the extra criteria are reachable.
let
    run1 = result_b1.run_results[1]
    reasons = Dict(name => mr.stop_reason for (name, mr) in run1.method_results)
    fixed_reason = reasons["GradientDescent[step_size=Fixed]"]
    other_reasons = [reasons[k] for k in keys(reasons)
                     if k != "GradientDescent[step_size=Fixed]"]
    println("    Fixed stopped with:    ", fixed_reason)
    println("    Other stop_reasons:    ", other_reasons)
    fixed_reason ∈ (:max_iterations, :objective_stagnated, :step_converged) ||
        @warn "Fixed did not stop on a budget/stagnation reason — method_criteria " *
              "may not be dispatching correctly."
    any(r -> r ∈ (:gradient_converged, :optimal_reached, :step_converged,
                  :objective_stagnated), other_reasons) ||
        @warn "No tolerance-based stop reason fired across the other methods — " *
              "did the budget catch everything?"
end

# ─── B2: CompositeCriteria(:all) ────────────────────────────────────────────
#
# stop_when_all requires BOTH GradientTolerance AND StepTolerance to hold
# simultaneously. A converging method will eventually satisfy both. The
# expected stop_reason is :all_criteria_met — NOT a single component's
# symbol. We pick BB1 because it converges fast and cleanly to x*, so both
# tolerances become satisfied within a sensible budget.

println("\n--- B2: stop_when_all(GradientTolerance(1e-6), StepTolerance(1e-8)) ---")
config_b2 = ExperimentConfig(
    name              = "Stage 7.b.2 — :all composite",
    problem_spec      = ROSEN,
    conventional_methods = [GradientDescent(direction = SteepestDescent(),
                                            step_size = BarzilaiBorwein(variant=:BB1))],
    stopping_criteria = stop_when_any(
                            stop_when_all(GradientTolerance(tol=1e-6),
                                          StepTolerance(tol=1e-8)),
                            MaxIterations(n=50_000),
                        ),
    n_runs            = 1,
    seed              = 42,
    tags              = Dict{String,Any}("stage" => "7.b.2"),
)
result_b2 = run_experiment(config_b2; verbosity=VERBOSITY)
df_b2 = _report("B2", result_b2)

let
    run1 = result_b2.run_results[1]
    reason = first(values(run1.method_results)).stop_reason
    println("    BB1 stopped with: ", reason)
    if reason != :all_criteria_met
        @warn "Expected :all_criteria_met but got $reason — " *
              "either :all composite is misbehaving, or the two tolerances " *
              "are never simultaneously satisfied within the budget."
    else
        println("    ✓ :all_criteria_met fired correctly.")
    end
end

# =============================================================================
# 7.c — Range-gated verbosity
# =============================================================================

println("\n\n╔══════════════════════════════════════════════════════════════╗")
println("║ Stage 7.c — Range-gated verbosity (iter_range = 200:300)     ║")
println("╚══════════════════════════════════════════════════════════════╝")

# MILESTONE fallback + iter_range = 200:300 ⇒ per-iter DETAILED lines appear
# only inside that band, and milestone start/event/finalize lines bookend
# the run (`maybe_print`'s iter_range branch silences per-iter output
# outside the range; `_print_milestone` still emits MILESTONE-level lines
# regardless). Inspect the stream below: per-iter lines should be present
# for iter ∈ 200:300 only.

println("\n--- C1: VerbosityConfig(level=MILESTONE, iter_range=200:300) ---")
println("    Visual check: per-iter lines appear ONLY for iter ∈ 200:300.\n")

verbosity_c1 = VerbosityConfig(
    level       = MILESTONE,
    iter_range  = 200:300,
    print_every = 1,
    fields      = [:iter, :objective, :gradient_norm, :step_norm, :dist_to_opt],
)

config_c1 = ExperimentConfig(
    name              = "Stage 7.c.1 — range-gated verbosity",
    problem_spec      = ROSEN,
    conventional_methods = [GradientDescent(direction = SteepestDescent(),
                                            step_size = ArmijoLS())],
    stopping_criteria = stop_when_any(MaxIterations(n=500),
                                      GradientTolerance(tol=1e-12)),
    n_runs            = 1,
    seed              = 42,
    tags              = Dict{String,Any}("stage" => "7.c.1"),
)
result_c1 = run_experiment(config_c1; verbosity=verbosity_c1)
_report("C1", result_c1)

# =============================================================================
# Summary
# =============================================================================

println("\n\n╔══════════════════════════════════════════════════════════════╗")
println("║ Stage 7 — done                                               ║")
println("╚══════════════════════════════════════════════════════════════╝")
println("""
  Coverage in this script:
    debug.jl       : CheckObjectiveMonotonicity, CheckGradientNormBound,
                     CheckStepDecay, CheckNumericalGradient
                     × on_trigger ∈ {:warn, :error, :log}
    stopping.jl    : TimeLimit, ObjectiveStagnation, StepTolerance,
                     DistanceToOptimal, GradientTolerance, MaxIterations
                     CompositeCriterion(:any) and (:all)
                     ExperimentConfig.method_criteria
    logging.jl     : VerbosityConfig.iter_range branch of maybe_print

  Together with Stages 0–4 and 5–6, every Rosenbrock-meaningful
  architectural block has now been validated. Remaining untested blocks
  need other problem types — see Experiment_TODOs.md.
""")
