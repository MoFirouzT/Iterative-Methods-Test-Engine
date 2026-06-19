"""
    Proximal Gradient (ISTA / FISTA)

A composite-objective method for  min_x f(x) + g(x)  with `f`
smooth and `g` proximable. Composes a `StepSize` rule with a `Extrapolation`
slot: `NoExtrapolation` ⇒ ISTA, `NesterovStep` ⇒ FISTA. With a zero / absent
regularizer it reduces to (accelerated) gradient descent on `f`.

See `proximal_gradient.md` for the full spec.
"""

using Random: AbstractRNG
using LinearAlgebra: norm
using .TestEngine                                          # engine types + functions this method uses
import .TestEngine: init_state, step!, extract_log_entry   # engine dispatch points this method extends


# ─────────────────────────────────────────────────────────────────────────
# Method Definition
# ─────────────────────────────────────────────────────────────────────────

"""
    ProximalGradient <: IterativeMethod

Proximal-gradient method. Fields:
- step_size    :: StepSize    — step-size rule (use `FixedStep(α = 1/L)`).
- extrapolation :: Extrapolation — `NoExtrapolation()` ⇒ ISTA, `NesterovStep()` ⇒ FISTA.
"""
@kwdef struct ProximalGradient <: IterativeMethod
    step_size::StepSize       = FixedStep()
    extrapolation::Extrapolation = NoExtrapolation()
end


# ─────────────────────────────────────────────────────────────────────────
# Method-Specific Numerics
# ─────────────────────────────────────────────────────────────────────────

"""
    ProximalGradientNumerics

Working storage for proximal gradient.

Fields:
- t                  : FISTA momentum parameter t_k (1.0 ⇒ no extrapolation yet).
- α_k                : last step size produced by `compute_step_size`; logged.
- direction          : −∇f(y); handed to the step-size rule (FixedStep ignores it).
- n_linesearch_evals : cumulative line-search trial evaluations (for LineSearch rules).
- grad_prev          : buffer the BB step-size rule expects (unused by FixedStep).
- x_trial            : scratch buffer the line-search rules write into.
"""
@kwdef mutable struct ProximalGradientNumerics
    t::Float64                  = 1.0
    α_k::Float64                = 0.0
    direction::Vector{Float64}  = Float64[]
    n_linesearch_evals::Int     = 0
    grad_prev::Vector{Float64}  = Float64[]
    x_trial::Vector{Float64}    = Float64[]
end


"""
    ProximalGradientState

Composes IterateGroup / MetricsGroup / TimingGroup (shared) with
ProximalGradientNumerics (method-specific).
"""
@kwdef mutable struct ProximalGradientState
    iterate::IterateGroup
    metrics::MetricsGroup
    timing::TimingGroup
    numerics::ProximalGradientNumerics
end


# ─────────────────────────────────────────────────────────────────────────
# Dispatch Point: Initialization
# ─────────────────────────────────────────────────────────────────────────

"""
    init_state(method::ProximalGradient, problem, rng)

Initialize at x₀. ProximalGradient supports at most one regularizer; a sum of
several nonsmooth terms would need operator splitting.
"""
function init_state(method::ProximalGradient, problem, rng::AbstractRNG)
    length(problem.gs) <= 1 || throw(ArgumentError(
        "ProximalGradient supports at most one regularizer (got $(length(problem.gs))); " *
        "a sum of nonsmooth terms needs operator splitting."))

    x0 = copy(problem.x0)
    g0 = similar(x0)
    grad!(g0, problem.f, x0)
    f0 = total_objective(problem, x0)

    return ProximalGradientState(
        iterate = IterateGroup(x = x0, gradient = g0, x_prev = Float64[]),
        metrics = MetricsGroup(
            objective     = f0,
            gradient_norm = norm(g0),
            step_norm     = 0.0,
            dist_to_opt   = Inf,            # runner fills this from problem.x_opt
        ),
        timing  = TimingGroup(core_time_ns = 0),
        numerics = ProximalGradientNumerics(t = 1.0, x_trial = similar(x0)),
    )
end


# ─────────────────────────────────────────────────────────────────────────
# Dispatch Point: Step
# ─────────────────────────────────────────────────────────────────────────

"""
    step!(method::ProximalGradient, state, problem, iter, logger, rng)

One proximal-gradient iteration (see `proximal_gradient.md` §2):
extrapolate → ∇f(y) → step size → prox → shift history → advance momentum.
Exactly one gradient evaluation and one `prox` call per step.
"""
function step!(method::ProximalGradient, state::ProximalGradientState,
               problem::Problem, iter::Int, logger::Logger, rng::AbstractRNG)

    nu = state.numerics

    # ── Core: extrapolation point y and gradient of the smooth part at y ──────
    #    ISTA: y = x. FISTA: y = x + β(x − x_prev). One gradient eval per step.
    local y, g
    @core_timed state begin
        y = extrapolate(method.extrapolation, state.iterate.x, state.iterate.x_prev, nu.t)
        g = grad(problem.f, y)                          # ∇f(y)
        nu.direction = -g                               # for step-size rules; FixedStep ignores
    end

    # ── Step-size selection (FixedStep(α = 1/L) is the supported rule) ────────
    γ = compute_step_size(method.step_size, state, problem, nu.direction)
    nu.α_k = γ

    # ── Core: proximal-gradient step and history shift ────────────────────────
    @core_timed state begin
        v  = y .- γ .* g                                # forward (gradient) step at y
        xⁿ = isempty(problem.gs) ? v : prox(problem.gs[1], v, γ)   # backward (prox) step

        # x_prev ← x_k (allocate once on iter 1, then reuse)
        if isempty(state.iterate.x_prev)
            state.iterate.x_prev = copy(state.iterate.x)
        else
            copyto!(state.iterate.x_prev, state.iterate.x)
        end

        state.metrics.step_norm = norm(xⁿ .- state.iterate.x)   # ‖xⁿ − x_k‖ (gradient-mapping proxy)
        copyto!(state.iterate.x, xⁿ)                            # x ← x_{k+1}

        state.metrics.objective = total_objective(problem, state.iterate.x)   # f + g
    end

    # ── Advance FISTA momentum (no-op for ISTA) ───────────────────────────────
    nu.t = advance_momentum(method.extrapolation, nu.t)

    # ── Bookkeeping (untimed) ─────────────────────────────────────────────────
    #    Report the smooth-part gradient at the evaluation point (no extra eval).
    state.iterate.gradient = g
    #    gradient_norm is the (prox-)gradient-mapping norm ‖G_γ(y)‖ = ‖(y − xⁿ)/γ‖,
    #    the proper composite-stationarity residual (→ 0 iff 0 ∈ ∇f + ∂g). When there
    #    is no regularizer the prox is the identity, so xⁿ = y − γ∇f(y) and this is
    #    exactly ‖∇f(y)‖ — a strict generalization at zero extra cost (reuses y, xⁿ, γ;
    #    no extra grad/prox eval, so the one-prox-per-step invariant holds). This makes
    #    GradientTolerance a valid stopping test on composite problems too.
    state.metrics.gradient_norm = isempty(problem.gs) ?
        norm(g) :                                          # smooth: ‖∇f(y)‖ exactly
        norm(y .- state.iterate.x) / γ                     # composite: ‖G_γ(y)‖
end


# ─────────────────────────────────────────────────────────────────────────
# Dispatch Point: Log Entry Extraction
# ─────────────────────────────────────────────────────────────────────────

"""
    extract_log_entry(method::ProximalGradient, state, iter)

Build IterationLog with the step size and FISTA momentum parameter as extras.
"""
function extract_log_entry(method::ProximalGradient, state::ProximalGradientState, iter::Int)
    entry = IterationLog(
        iter          = iter,
        core_time_ns  = state.timing.core_time_ns,
        objective     = state.metrics.objective,
        gradient_norm = state.metrics.gradient_norm,
        step_norm     = state.metrics.step_norm,
        dist_to_opt   = state.metrics.dist_to_opt,
        extras = Dict{Symbol,Any}(
            :step_size => state.numerics.α_k,
            :t         => state.numerics.t,
        ),
    )
    if length(state.iterate.x) <= 2               # for trajectory plots
        entry.extras[:x_iter] = copy(state.iterate.x)
    end
    return entry
end
