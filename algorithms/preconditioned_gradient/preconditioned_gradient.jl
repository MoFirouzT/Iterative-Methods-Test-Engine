"""
    Preconditioned Gradient (experimental)

An `IterativeMethod` that crosses a **preconditioner** axis with a **step-size**
axis: the descent direction is `d = −M⁻¹ ∇f(x)`. With `IdentityPreconditioner`
it is plain gradient descent; with `JacobiPreconditioner` on a diagonal Hessian
it is exactly Newton. Its purpose is to exercise the framework's signature
workflow — define one method, sweep its variants in a `VariantGrid` with
`role = :experimental`, and have `resolve_methods` route them into the
*experimental* bucket alongside a reference baseline. See `preconditioned_gradient.md`.
"""

using Random: AbstractRNG
using LinearAlgebra: norm
using .TestEngine
import .TestEngine: init_state, step!, extract_log_entry


# ─────────────────────────────────────────────────────────────────────────
# Method Definition
# ─────────────────────────────────────────────────────────────────────────

"""
    PreconditionedGradient <: IterativeMethod

Fields:
- preconditioner :: Preconditioner — `IdentityPreconditioner()` (⇒ GD) or
  `JacobiPreconditioner()` (⇒ Newton on a diagonal Hessian).
- step_size      :: StepSize       — step-size rule along `d = −M⁻¹∇f`.
"""
@kwdef struct PreconditionedGradient <: IterativeMethod
    preconditioner::Preconditioner = IdentityPreconditioner()
    step_size::StepSize            = ArmijoLS()
end


# ─────────────────────────────────────────────────────────────────────────
# Method-Specific Numerics  (same shape GradientDescent uses, so the shared
# StepSize rules — Armijo/Cauchy/BB — work unchanged)
# ─────────────────────────────────────────────────────────────────────────

@kwdef mutable struct PreconditionedGradientNumerics
    direction::Vector{Float64}  = Float64[]
    α_k::Float64                = 0.0
    n_linesearch_evals::Int     = 0
    grad_prev::Vector{Float64}  = Float64[]
    x_trial::Vector{Float64}    = Float64[]
end

@kwdef mutable struct PreconditionedGradientState
    iterate::IterateGroup
    metrics::MetricsGroup
    timing::TimingGroup
    numerics::PreconditionedGradientNumerics
end


# ─────────────────────────────────────────────────────────────────────────
# Dispatch Point: Initialization
# ─────────────────────────────────────────────────────────────────────────

function init_state(method::PreconditionedGradient, problem, rng::AbstractRNG)
    x0 = copy(problem.x0)
    g0 = similar(x0); grad!(g0, problem.f, x0)
    f0 = total_objective(problem, x0)
    return PreconditionedGradientState(
        iterate = IterateGroup(x = x0, gradient = g0, x_prev = Float64[]),
        metrics = MetricsGroup(
            objective     = f0,
            gradient_norm = norm(g0),
            step_norm     = 0.0,
            dist_to_opt   = Inf,            # runner fills this from problem.x_opt
        ),
        timing  = TimingGroup(core_time_ns = 0),
        numerics = PreconditionedGradientNumerics(x_trial = similar(x0)),
    )
end


# ─────────────────────────────────────────────────────────────────────────
# Dispatch Point: Step  (mirrors GradientDescent; only the direction differs)
# ─────────────────────────────────────────────────────────────────────────

"""
    step!(method::PreconditionedGradient, state, problem, iter, logger, rng)

x_{k+1} = x_k + α_k · d_k,  d_k = −M⁻¹ ∇f(x_k). Order mirrors `GradientDescent`
so the BB secant pair (x_prev, grad_prev) is valid at `compute_step_size`.
"""
function step!(method::PreconditionedGradient, state::PreconditionedGradientState,
               problem::Problem, iter::Int, logger::Logger, rng::AbstractRNG)

    # ── Core: preconditioned descent direction at x_k ─────────────────────────
    #    Write into the preallocated direction buffer (sized lazily) rather than
    #    allocating a fresh `-M⁻¹g` each step.
    @core_timed state begin
        Minvg = precondition(method.preconditioner, state.iterate.gradient, problem, state.iterate.x)
        d = state.numerics.direction
        length(d) == length(Minvg) || (d = state.numerics.direction = similar(Minvg))
        d .= .-Minvg
    end

    # ── Step-size selection (BB reads the secant pair below; don't reorder) ───
    α_k = compute_step_size(method.step_size, state, problem, state.numerics.direction)
    state.numerics.α_k = α_k

    # ── Save x_k → x_prev (after the step-size rule, before the update) ───────
    if isempty(state.iterate.x_prev)
        state.iterate.x_prev = copy(state.iterate.x)
    else
        copyto!(state.iterate.x_prev, state.iterate.x)
    end

    # ── Core: iterate update ──────────────────────────────────────────────────
    @core_timed state begin
        state.iterate.x .+= α_k .* state.numerics.direction
    end

    # ── Save ∇f(x_k) → grad_prev before the refresh ───────────────────────────
    if isempty(state.numerics.grad_prev)
        state.numerics.grad_prev = copy(state.iterate.gradient)
    else
        copyto!(state.numerics.grad_prev, state.iterate.gradient)
    end

    # ── Core: refresh objective and gradient at x_{k+1} ───────────────────────
    @core_timed state begin
        state.metrics.objective = total_objective(problem, state.iterate.x)
        grad!(state.iterate.gradient, problem.f, state.iterate.x)
    end

    # ── Bookkeeping (untimed; dist_to_opt is filled by the runner) ────────────
    state.metrics.gradient_norm = norm(state.iterate.gradient)
    state.metrics.step_norm     = abs(α_k) * norm(state.numerics.direction)
end


# ─────────────────────────────────────────────────────────────────────────
# Dispatch Point: Log Entry Extraction
# ─────────────────────────────────────────────────────────────────────────

function extract_log_entry(method::PreconditionedGradient, state::PreconditionedGradientState, iter::Int)
    entry = IterationLog(
        iter          = iter,
        core_time_ns  = state.timing.core_time_ns,
        objective     = state.metrics.objective,
        gradient_norm = state.metrics.gradient_norm,
        step_norm     = state.metrics.step_norm,
        dist_to_opt   = state.metrics.dist_to_opt,
        extras = Dict{Symbol,Any}(
            :n_linesearch_evals => state.numerics.n_linesearch_evals,
            :step_size          => state.numerics.α_k,
        ),
    )
    if length(state.iterate.x) <= 2
        entry.extras[:x_iter] = copy(state.iterate.x)
    end
    return entry
end
