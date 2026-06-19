using Test

# Engine + all content via the shared bootstrap (idempotent), plus shared helpers.
include(joinpath(@__DIR__, "..", "experiments", "_bootstrap.jl"))
include(joinpath(@__DIR__, "testutils.jl"))    # silent_logger and other shared helpers

# ── Test groups ──────────────────────────────────────────────────────────────
# TEST_GROUP selects which slice runs (default "all"):
#   core      — everything except the external cross-validation (fast; no Optim/
#               ProximalAlgorithms solves);
#   external  — only the cross-validation against Optim.jl / ProximalAlgorithms.jl
#               (slow: 50k–200k-iteration reference solves);
#   all       — both.
const TEST_GROUP   = lowercase(get(ENV, "TEST_GROUP", "all"))
const RUN_CORE     = TEST_GROUP in ("all", "core")
const RUN_EXTERNAL = TEST_GROUP in ("all", "external")
TEST_GROUP in ("all", "core", "external") ||
    error("TEST_GROUP must be one of \"all\", \"core\", \"external\"; got \"$TEST_GROUP\"")
@info "Running tests" TEST_GROUP

if RUN_CORE
    # Content conformance harness — runs FIRST so its completeness guard sees only
    # registered content (before test_module9.jl registers throwaway problems).
    include(joinpath(@__DIR__, "test_problem_contract.jl"))

    # Wiring guard: every test_*.jl on disk must be referenced by an include in
    # this file. Catches a new test file that was added but never wired in — it
    # would otherwise pass silently by never running. (Mirrors the
    # CONFORMANCE_SPECS completeness guard, but for test files instead of problems.)
    @testset "all test files are wired into runtests" begin
        runtests_src = read(@__FILE__, String)
        test_files = filter(readdir(@__DIR__)) do file
            startswith(file, "test_") && endswith(file, ".jl")
        end
        for file in test_files
            @test occursin(file, runtests_src)
        end
    end

    include(joinpath(@__DIR__, "test_step_sizes.jl"))          # step-size rules: Fixed/Armijo/Cauchy/Barzilai-Borwein
    include(joinpath(@__DIR__, "test_core.jl"))                # core abstraction, nested runner, variant-grid expansion
    include(joinpath(@__DIR__, "test_reproducibility.jl"))     # seed determinism: same seed ⇒ bit-identical run
    include(joinpath(@__DIR__, "test_module5.jl"))             # experiment orchestration (resolve_methods, run_experiment)
    include(joinpath(@__DIR__, "test_module7.jl"))             # verbosity system
    include(joinpath(@__DIR__, "test_module8.jl"))             # persistence (save/load, manifest, CSV, columnar shim)
    include(joinpath(@__DIR__, "test_module9.jl"))             # problem factory: LeastSquares / regularizer content
    include(joinpath(@__DIR__, "test_proximal_gradient.jl"))   # ProximalGradient: ISTA↔GD reduction, FISTA acceleration, one-prox-per-step
    include(joinpath(@__DIR__, "test_least_squares.jl"))       # LeastSquares Hessian modes, :linear_ls conditioning, Cauchy-guard regression
    include(joinpath(@__DIR__, "test_preconditioned_gradient.jl"))   # PreconditionedGradient: Jacobi=Newton, dual-bucket routing, diagonal contract
    include(joinpath(@__DIR__, "test_trust_region.jl"))        # TrustRegion + Steihaug-CG: branches, nesting, core-time attribution
    include(joinpath(@__DIR__, "test_oracle_counting.jl"))     # opt-in oracle counting: value/grad!/Hvp tally, nesting, Jacobi transparency
end

if RUN_EXTERNAL
    include(joinpath(@__DIR__, "test_external_validation.jl")) # cross-check converged solutions vs Optim.jl + ProximalAlgorithms.jl
end
