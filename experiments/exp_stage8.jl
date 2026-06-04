# experiments/exp_stage8.jl
# =============================================================================
# Stage 8 — Cross-cutting validations on Rosenbrock
# -----------------------------------------------------------------------------
# After Stage 7, the planned Rosenbrock-meaningful list is closed. This file
# bundles the four "doesn't fit a single earlier stage" items called out in
# experiments/Experiment_TODOs.md ("Cross-cutting validations not yet
# covered"). All four are pure Rosenbrock checks — no new problem family is
# needed.
#
#   8.a — logger.events roundtrip through save_experiment / load_experiment.
#         (Previously dropped: MethodResult didn't carry events; Stage 8
#         lands an events field on MethodResult and verifies the trip.)
#   8.b — aggregate_runs(df, :all) — the "passthrough copy" mode that Stage 6
#         specifies but never exercises directly.
#   8.c — method_color registry roundtrip across save_experiment + a fresh
#         load_experiment call in the same session. The registry is
#         intentionally process-global (not persisted to disk); the
#         validation here is the in-session contract: register_method_color!
#         mutations are visible to get_method_color regardless of save/load.
#   8.d — run_sub_method invocation shape. The sub-method machinery
#         (run_sub_method / SubResult / attach_sub_logs!) is exported and
#         covered by unit tests, but no experiment exercises it. We invoke
#         it once against an outer logger and assert sub_logs are attached
#         and the SubResult fields are well-formed.
#
# Framework gaps filled during this stage (landed):
#   * MethodResult gains an `events::Vector{NamedTuple}` field, populated
#     by finalize! from logger.events. Without this, every event recorded
#     during a run (the stopping event, debug `:log` events, etc.) was
#     dropped at finalize and never reached the persistence layer. A
#     backward-compatible constructor preserves the previous 5-arg call
#     sites in src/experiment.jl#_to_method_result. ExperimentResult round-
#     trips events through JLD2 automatically.
# =============================================================================

include("../src/TestEngine.jl")
using .TestEngine
using DataFrames, DataFramesMeta
using Random
using LinearAlgebra
using Printf

include("_shared.jl")

register_abbreviation!("GradientDescent", "GD")
register_abbreviation!("FixedStep",       "Fix")
register_abbreviation!("ArmijoLS",        "Arm")
register_abbreviation!("CauchyStep",      "Cau")
register_abbreviation!("BarzilaiBorwein", "BB")

const VERBOSITY = VerbosityConfig(level=SILENT)

# Common base config — five-method grid on Rosenbrock, short budget.
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

const ROSEN = AnalyticProblem(name=:rosenbrock,
                              params=(rho=100.0, x0=[-1.2, 1.0]))

# =============================================================================
# 8.a — logger.events roundtrip
# =============================================================================
#
# We turn on Stage-7-style debug at `:log` mode so the run records debug
# events into logger.events alongside the regular stopping event. After
# finalize!, MethodResult carries those events; save_experiment writes the
# ExperimentResult to JLD2; load_experiment reads it back; we assert
# (a) the loaded MethodResult has a non-empty `events` vector, (b) the
# stopping-reason event is present, and (c) at least one `:debug` event
# survives the roundtrip when triggered.

println("\n\n╔══════════════════════════════════════════════════════════════╗")
println("║ Stage 8.a — logger.events roundtrip                          ║")
println("╚══════════════════════════════════════════════════════════════╝")

config_8a = ExperimentConfig(
    name              = "Stage 8.a — events roundtrip",
    problem_spec      = ROSEN,
    variant_grids     = [FULL_GRID],
    stopping_criteria = stop_when_any(MaxIterations(n=100),
                                      GradientTolerance(tol=1e-9)),
    n_runs            = 1,
    seed              = 42,
    debug             = DebugConfig(
        enabled    = true,
        checks     = DebugCheck[CheckObjectiveMonotonicity(tolerance = 0.0)],
        on_trigger = :log,
    ),
    tags              = Dict{String,Any}("stage" => "8.a"),
)
result_8a_mem = run_experiment(config_8a; verbosity=VERBOSITY)
println("    saved to: ", result_8a_mem.experiment_path)

# Reload from disk and assert.
result_8a_disk = load_experiment(result_8a_mem.experiment_path)

let
    run1_mem  = result_8a_mem.run_results[1]
    run1_disk = result_8a_disk.run_results[1]
    @assert keys(run1_mem.method_results) == keys(run1_disk.method_results) (
        "method_results keys differ across save/load")

    for name in keys(run1_mem.method_results)
        mr_mem  = run1_mem.method_results[name]
        mr_disk = run1_disk.method_results[name]
        @assert length(mr_mem.events) == length(mr_disk.events) (
            "events length differs for $name: mem=$(length(mr_mem.events)) " *
            "disk=$(length(mr_disk.events))")
        @assert mr_mem.stop_reason == mr_disk.stop_reason (
            "stop_reason differs for $name")
    end

    bb1 = run1_disk.method_results["GradientDescent[step_size=BB1]"]
    n_debug = count(e -> get(e, :kind, :none) === :debug, bb1.events)
    n_stop  = count(e -> haskey(e, :reason),              bb1.events)
    @printf("    BB1 events on disk: total=%d, debug=%d, stop=%d\n",
            length(bb1.events), n_debug, n_stop)
    @assert n_stop >= 1     "no stopping event roundtripped for BB1"
    @assert n_debug >= 1    "no :debug event roundtripped for BB1 — the " *
                            "objective-monotonicity check should have fired " *
                            "at least once on BB1 within 100 iters"
    println("    ✓ events roundtripped intact for all $(length(run1_disk.method_results)) methods.")
end

# =============================================================================
# 8.b — aggregate_runs(df, :all)
# =============================================================================
#
# :all is the "give me back every row, untouched, but go through the
# aggregation API so my downstream code shape is uniform" mode. We assert
# it's the identity on the input frame's content (no rows dropped, no
# columns dropped). Per src/analysis.jl this is a `copy(df)`; the contract
# is value-equality, not pointer-equality.

println("\n\n╔══════════════════════════════════════════════════════════════╗")
println("║ Stage 8.b — aggregate_runs(df, :all)                         ║")
println("╚══════════════════════════════════════════════════════════════╝")

config_8b = ExperimentConfig(
    name              = "Stage 8.b — aggregate_runs all-mode",
    problem_spec      = ROSEN,
    variant_grids     = [FULL_GRID],
    stopping_criteria = stop_when_any(MaxIterations(n=100),
                                      GradientTolerance(tol=1e-9)),
    n_runs            = 3,
    seed              = 42,
    tags              = Dict{String,Any}("stage" => "8.b"),
)
result_8b = run_experiment(config_8b; verbosity=VERBOSITY)

let
    df_raw = to_dataframe(result_8b)
    df_all = aggregate_runs(df_raw, :all)

    @assert nrow(df_raw)         == nrow(df_all)         "row count drift on :all"
    @assert propertynames(df_raw) == propertynames(df_all) "column drift on :all"
    sort!(df_raw, [:method_name, :run_id, :iter])
    sort!(df_all, [:method_name, :run_id, :iter])
    @assert isequal(df_raw, df_all)  ":all output is not value-equal to input"
    println("    ✓ aggregate_runs(df, :all) preserves shape + content ",
            "($(nrow(df_all)) rows across $(length(unique(df_all.run_id))) runs).")

    # Sanity vs :median: aggregation collapses the run_id dimension, so the
    # median frame has fewer rows and may drop the :run_id column entirely.
    df_med = aggregate_runs(df_raw, :median)
    @assert nrow(df_med) < nrow(df_raw) (
        ":median did not collapse runs — got $(nrow(df_med)) rows")
    println("    ✓ :median collapsed $(nrow(df_raw)) → $(nrow(df_med)) rows ",
            "(by aggregating $(length(unique(df_raw.run_id))) runs per (method, iter)).")
end

# =============================================================================
# 8.c — method_color registry roundtrip (in-session contract)
# =============================================================================
#
# The registry is process-global and intentionally NOT persisted into the
# experiment manifest. The contract worth asserting here is the in-session
# one: after register_method_color!, get_method_color returns the
# registered value, and that value is unaffected by an unrelated
# save_experiment / load_experiment cycle in between (i.e., the registry
# is independent of the result struct).

println("\n\n╔══════════════════════════════════════════════════════════════╗")
println("║ Stage 8.c — method_color registry roundtrip                  ║")
println("╚══════════════════════════════════════════════════════════════╝")

let
    name  = "Stage8Method"
    color = "#123456"
    @assert get_method_color(name) != color (
        "test fixture is dirty — $name already registered before the test")

    register_method_color!(name, color)
    @assert get_method_color(name) == color (
        "register_method_color! did not take effect")
    println("    register_method_color!(\"$name\", \"$color\") → ",
            get_method_color(name))

    # Round-trip the prior experiment between the register and the readback.
    _ = load_experiment(result_8b.experiment_path)
    @assert get_method_color(name) == color (
        "save/load wiped or shadowed the in-session registry entry")
    println("    ✓ registry mutation visible after an intervening load_experiment.")

    # Tidy up so subsequent runs of this script see a clean fixture.
    delete!(METHOD_COLOR_REGISTRY, name)
end

# =============================================================================
# 8.d — run_sub_method invocation shape
# =============================================================================
#
# We construct an outer logger, then invoke run_sub_method with a one-iter
# FixedStep GD as the inner method. Assertions:
#   (i)  SubResult.n_iters == 1 and stop_reason == :max_iterations.
#   (ii) SubResult.iter_logs has length 2 (iter=0 from log_init! + iter=1).
#   (iii) The outer logger's pending_sub_logs is non-empty after the call
#         (attach_sub_logs! was invoked because config.log_sub_iters = true).
#   (iv) SubResult.final_state.iterate.x has actually moved off the
#        starting point (the inner GD really stepped).

println("\n\n╔══════════════════════════════════════════════════════════════╗")
println("║ Stage 8.d — run_sub_method invocation shape                  ║")
println("╚══════════════════════════════════════════════════════════════╝")

let
    rng     = Xoshiro(42)
    problem = make_problem(ROSEN, rng)

    outer_logger = make_logger("OuterDriver", 1, "", VerbosityConfig(level=SILENT))
    log_init!(outer_logger, GradientDescent(direction=SteepestDescent(),
                                            step_size=FixedStep(α=8e-4)),
              init_state(GradientDescent(direction=SteepestDescent(),
                                         step_size=FixedStep(α=8e-4)),
                         problem, rng))

    sub_cfg = SubRunConfig(
        method        = GradientDescent(direction = SteepestDescent(),
                                        step_size = FixedStep(α=8e-4)),
        criteria      = MaxIterations(n=1),
        log_sub_iters = true,
        verbosity     = VerbosityConfig(level=SILENT),
    )

    x0_before = copy(problem.x0)
    sub_result = run_sub_method(sub_cfg, problem, outer_logger, rng)

    @assert sub_result isa SubResult                 "run_sub_method returned wrong type"
    @assert sub_result.n_iters == 1                  "expected n_iters=1, got $(sub_result.n_iters)"
    @assert sub_result.stop_reason == :max_iterations (
        "expected :max_iterations, got $(sub_result.stop_reason)")
    @assert length(sub_result.iter_logs) == 2 (
        "expected 2 iter logs (iter=0 init + iter=1 step), got $(length(sub_result.iter_logs))")
    @assert !isempty(outer_logger.pending_sub_logs) (
        "attach_sub_logs! did not push into outer_logger.pending_sub_logs")
    @assert sub_result.iter_logs === outer_logger.pending_sub_logs || (
            sub_result.iter_logs == outer_logger.pending_sub_logs) (
        "outer's pending_sub_logs do not match the sub-run's iter_logs")
    @assert norm(sub_result.final_state.iterate.x .- x0_before) > 1e-6 (
        "sub-method did not actually move off x0")

    println("    ✓ run_sub_method ran $(sub_result.n_iters) iter ",
            "(stop=$(sub_result.stop_reason)); sub_logs attached ",
            "($(length(outer_logger.pending_sub_logs)) entries); ",
            "‖x_final − x0‖ = ",
            @sprintf("%.3e", norm(sub_result.final_state.iterate.x .- x0_before)),
            ".")
end

# =============================================================================
# Summary
# =============================================================================

println("\n\n╔══════════════════════════════════════════════════════════════╗")
println("║ Stage 8 — done                                               ║")
println("╚══════════════════════════════════════════════════════════════╝")
println("""
  Coverage in this script:
    persistence.jl : ExperimentResult roundtrip of MethodResult.events
                     (new field; see basic_experiments.md Stage 8 gaps).
    analysis.jl    : aggregate_runs :all branch + run-id collapse on :median.
    analysis.jl    : METHOD_COLOR_REGISTRY in-session contract under
                     save_experiment / load_experiment.
    core.jl        : run_sub_method, SubResult, attach_sub_logs!.
""")
