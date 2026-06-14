# TrustRegion + Steihaug truncated-CG

A trust-region Newton method whose inner subproblem is solved by Steihaug
truncated CG run as a **genuine sub-method on a genuine `Problem`**. This is the
consumer that lights up the nested-optimization subsystem (`run_sub_method`,
`SubRunConfig`, `SubResult`, `attach_sub_logs!`).

## The model subproblem

At the current iterate `x`, with `g = ∇f(x)` and `H = ∇²f(x)`, the quadratic
model of the step `p` is

    m(p) = gᵀp + ½ pᵀ H p          (offset so m(0) = 0)

represented by `QuadraticModel <: Objective` — `value(m,p)=gᵀp+½pᵀHp`,
`grad!(m,p)=g+Hp`, `hessian(m,p)=H`. `H` is whatever `hessian(f, x)` returns: a
`MatrixHessian` (Rosenbrock), an `OperatorHessian` (matrix-free least squares), …
— the inner solver only calls `apply(H, ·)`, so no matrix is ever required.

## SteihaugCG — the inner solver

`SteihaugCG(Δ)` minimizes `m(p)` inside the trust region `‖p‖ ≤ Δ`, one CG
iteration per `step!` (so the inner trace is a real iteration log). Starting from
`p₀ = 0`, `r₀ = g`, `d₀ = −g`, each step:

1. `Hd = H d`, `dᵀHd` (the curvature along `d`).
2. **Negative curvature** (`dᵀHd ≤ 0`): the model decreases without bound along
   `d` — step to the boundary `p + τd` (`τ ≥ 0` solving `‖p+τd‖ = Δ`) and stop.
3. `α = rᵀr / dᵀHd`. If `‖p + αd‖ ≥ Δ`, the step would **leave the region** —
   truncate to the boundary `p + τd` and stop.
4. Otherwise take the CG step, update `r ← r + αHd`, `d ← −r + βd`.

Termination is by **composable `StoppingCriterion`**:
`MaxIterations ∨ GradientTolerance(‖r‖) ∨ NegativeCurvature ∨ TrustRegionBoundary`.
The last two read the inner status via `_tr_status(state)` (default `:none`;
`SteihaugCGState` overrides it). `gradient_norm` is set to the residual norm `‖r‖`,
so `GradientTolerance` serves directly as the residual tolerance.

Unit-tested in isolation (`test_trust_region.jl`): large `Δ` ⇒ the exact Newton
step `−H⁻¹g`; small `Δ` ⇒ `‖p‖ = Δ` (boundary); indefinite `H` ⇒ the
negative-curvature branch.

## TrustRegion — the outer method

Each outer `step!`:

1. Build `QuadraticModel(g, H)` at `x`; solve it with `SteihaugCG(Δ)` via
   `run_sub_method` (`log_sub_iters = true`).
2. Compute the **actual / predicted reduction ratio**
   `ρ = (f(x) − f(x+p)) / (m(0) − m(p))`.
3. **Accept** the step (`x ← x + p`) when `ρ > η`; otherwise reject (x unchanged).
4. **Update Δ**: shrink (`×¼`) when `ρ < ¼`; expand (`×2`, capped at `Δmax`) when
   `ρ > ¾` *and* the inner step hit the boundary / negative curvature; else keep.

Fields: `Δ0`, `Δmax`, `η`, `max_inner` (0 ⇒ `n+1`), `inner_tol`.

## Core-time attribution

The inner solve's total core time is **folded into the outer step's
`core_time_ns`** (`state.timing.core_time_ns += sub.core_time_ns`) so
cumulative-core plots reflect *all* work — and **also** exposed per-step as
`extras[:inner_core_ns]` for an inner/outer breakdown. `run_sub_method` is *not*
wrapped in `@core_timed` (that would count inner wall/scaffolding time). Each
outer entry also carries the inner CG trace in `extras[:sub_logs]`. See
`docs/src/modules/nested-algorithms.md`.

## Win conditions (trust-region experiment)

- Trust-region-Newton reaches high accuracy on Rosenbrock in **~24 outer
  iterations** (`gradient_converged`) — far fewer than the first-order baselines.
- `attach_sub_logs!` populates the outer logger; `SubResult.core_time_ns > 0`;
  each outer entry carries its inner trace.
- The **boundary** branch fires on the standard trajectory; the
  **negative-curvature** branch fires from an indefinite-Hessian start
  (`x₀ = (0, 2)`, where the Rosenbrock Hessian has a negative eigenvalue).
