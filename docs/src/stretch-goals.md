# Stretch Goals

Capabilities the architecture is *designed to admit* but does not currently ship. This
page exists so the design intent is not lost: each entry records *how* the capability
plugs in — or, where it does not yet fit, exactly what would have to move.

The organizing axis is the engine's **oracle vocabulary**. A `Problem` exposes a fixed
set of oracles — `value`, `gradient`, the [`Hessian`](modules/problem-interface.md#Hessian)
faces (`apply` / `materialize` / `diagonal`), and a regularizer `prox` — and every method
is assembled from those plus the open dispatch points (`IterativeMethod`,
`StoppingCriterion`, the pluggable components, and the nested-solve infrastructure of the
[Nested Algorithm Infrastructure](@ref)). A capability is cheap to add when it needs
nothing outside that vocabulary, and costly in proportion to how far beyond it the
required oracle reaches. The two bands below are ordered by exactly that cost.

## Reachable as content — no engine change

Everything here is added by writing *content* — a new `IterativeMethod` (its
`init_state` / `step!` / `extract_log_entry`), a component, or a stopping criterion —
against the existing oracle vocabulary; the engine itself is untouched. Full derivations
belong in each method's co-located design note (the "claims are demonstrated, not asserted"
principle); the intent is indexed
here in one place.

### Quasi-Newton Hessian approximations (BFGS, SR1, L-BFGS)

Stateful Hessian approximations plug into the existing [`Hessian`](modules/problem-interface.md#Hessian)
hierarchy without changes to any consumer: each is a `Hessian` subtype carrying an
internal `update!(H, s, y)` method that the algorithm calls after each step. L-BFGS,
in particular, never defines `materialize` — it stores the `(s, y)` history and
computes `apply` via the two-loop recursion. Methods consuming the Hessian are
unaware of the internal mechanism; they only ever call `apply` (and, when available,
`materialize` / `diagonal`).

### Inexact Newton via truncated CG (Newton-CG)

Solve the Newton system `∇²f(xₖ) d = −∇f(xₖ)` *iteratively* with conjugate gradients run
as a genuine sub-method through the [Nested Algorithm Infrastructure](@ref). Truncating on
an Eisenstat–Walker forcing sequence `‖rₖ‖ ≤ ηₖ ‖∇f(xₖ)‖` tightens the inner accuracy as
the outer iterate approaches a minimizer, recovering superlinear convergence at a cost of
only Hessian-vector products. It needs solely the `apply` (Hessian-vector) face of the
[`Hessian`](modules/problem-interface.md#Hessian) interface — never `materialize` — and is
globalized by an existing Armijo step size. It is the *residual-truncated* counterpart of
the shipped Steihaug truncated-CG (see [trust_region.md](https://github.com/MoFirouzT/Iterative-Methods-Test-Engine/blob/main/algorithms/trust_region/trust_region.md)),
which instead truncates on the trust-region boundary and on negative curvature; sharing
the nested infrastructure, the two differ only in their inner stopping rule.

### Gauss–Newton and Levenberg–Marquardt

For least-squares objectives `½‖r(x)‖²`, the Gauss–Newton model replaces the Hessian with
`JᵀJ`, which the existing [`Hessian`](modules/problem-interface.md#Hessian) interface
already expresses as a structured operator — exactly the `:matrix` / `:operator` Hessian
modes the [`least_squares.md`](https://github.com/MoFirouzT/Iterative-Methods-Test-Engine/blob/main/problems/least_squares/least_squares.md)
problem ships. Gauss–Newton is the undamped solve; Levenberg–Marquardt adds a diagonal
damping `JᵀJ + λI` and accepts or rejects each step with the same ρ-ratio test the engine's
trust-region machinery already uses, adjusting `λ` in place of a radius. The ready-made
demonstrator is that `least_squares` problem under an ill-conditioned `:linear_ls` sweep.

### Projected gradient and projected Newton via indicator regularizers

A constraint `x ∈ C` enters with no new machinery as a [`Regularizer`](modules/problem-interface.md#Regularizer)
whose `prox` is the Euclidean projection `Π_C` (the proximal operator of the indicator
`δ_C`). Projected gradient is then *literally* `ProximalGradient` with an indicator `g`
(see [proximal_gradient.md](https://github.com/MoFirouzT/Iterative-Methods-Test-Engine/blob/main/algorithms/proximal_gradient/proximal_gradient.md)),
and projected / proximal Newton is the same construction with a metric prox solved through
the [Nested Algorithm Infrastructure](@ref) — so the composite machinery already shipped
covers *projection-expressible* constrained optimization as a special case. The cleanest
first demonstrator is a box- or simplex-constrained quadratic, whose projection is
closed-form. (Constraints whose projection is *not* cheap are a different matter — see the
Tier 1 frontier below.)

### Nonlinear conjugate gradient (Fletcher–Reeves, Polak–Ribière⁺)

Nonlinear CG is a descent-direction component `dₖ = −∇f(xₖ) + βₖ dₖ₋₁` that carries the
previous direction and gradient in the method's `numerics` group and reuses the existing
line-search step sizes (a Wolfe step is required for the `β` formulas to stay descent
directions). Only `β` varies across the classical variants — Fletcher–Reeves,
Polak–Ribière, and PR⁺ with its `max(β, 0)` restart — so they enter as sibling content
selectable through the variant grid. It complements the *linear* CG already living inside
the Steihaug solver: the same recurrence, but on a general smooth `f` rather than a fixed
quadratic.

### AdaGrad-style adaptive diagonal preconditioning

An adaptive diagonal scale `Dₖ = diag(√(Σ gᵢ²) + ε)` is a *stateful*
[preconditioner](https://github.com/MoFirouzT/Iterative-Methods-Test-Engine/blob/main/algorithms/components/preconditioners.md)
that accumulates the gradient second moment across iterations and slots directly into
`PreconditionedGradient` — the consumer only ever asks the preconditioner to apply itself,
so a fixed Jacobi preconditioner and an evolving AdaGrad / RMSProp scale are
indistinguishable to it. This exercises the one capability the shipped preconditioners do
not yet use: per-iteration internal state, updated in `step!` exactly as a quasi-Newton
`Hessian` updates its `(s, y)` memory.

### Non-method refinements

Two designed-for extensions are not new methods but refinements to existing machinery,
each analysed next to the code it touches:

- **Barzilai–Borwein nonmonotone (GLL) safeguard** — the principled replacement for
  BB's `[α_min, α_max]` clamp, permitting BB's load-bearing excursions while catching
  genuine divergence. Full derivation and a ~30-line implementation sketch in
  [`step_sizes.md` §5.2](https://github.com/MoFirouzT/Iterative-Methods-Test-Engine/blob/main/algorithms/components/step_sizes.md).

## Requires moving a boundary — engine extension

Each family here needs an oracle *outside* the current vocabulary, or a deeper change to
the iterate representation itself. Cataloguing them marks where the engine's boundary
actually lies; they are tiered by how far past that boundary the required change reaches.

### Tier 1 — a new oracle on the problem interface

The runner, logging, and nested-solve machinery stay untouched; the cost is one new oracle
on the `Problem`, plus the method that consumes it.

- **Constrained optimization by sequential unconstrained subproblems** — *penalty*,
  *augmented Lagrangian* (method of multipliers), and *log-barrier / interior-point*
  methods. Each needs a **constraint oracle** (`c(x)`, `h(x)` and their Jacobians) on the
  problem, but the outer-loop / inner-solve structure they share is exactly what the
  [Nested Algorithm Infrastructure](@ref) already provides: the inner solve is a shipped
  unconstrained method, and the outer loop only updates the penalty weight `μ` or the
  multipliers `λ`. This is the highest-leverage extension in the engine — a single new
  oracle unlocks a whole family that reuses the existing solvers as inner engines.
- **Sequential quadratic programming (SQP)** — the same constraint oracle, but the inner
  subproblem is a QP (a Lagrangian-Hessian model under linearized constraints) rather than
  an unconstrained minimization; admitting it also needs a QP sub-solver, itself written as
  a nested method.
- **Frank–Wolfe / conditional gradient** — needs a **linear-minimization oracle**
  `argmin_{s ∈ C} ⟨∇f, s⟩`, a small oracle distinct from `prox`. Being projection-free, it
  earns its place precisely where projection is expensive but the LMO is cheap (the
  simplex, the nuclear-norm ball).
- **Stochastic / finite-sum methods** — SGD, mini-batch, SVRG / SAGA, Adam — need a
  **sampled-gradient oracle** over `f = Σᵢ fᵢ` (the variance-reduced variants additionally
  take periodic full-gradient snapshots). The reproducibility plumbing already supplies
  what the sampling needs: deterministic child RNG streams per run and per sub-call.
- **Operator-splitting with a linear map** — ADMM (`Ax + Bz = c`) and primal–dual /
  Chambolle–Pock (PDHG) — need a **linear operator and its adjoint** as problem data,
  alongside the two proxes the composite interface already exposes.

### Tier 2 — the oracle fits, but the convergence vocabulary does not

- **Derivative-free / zeroth-order** — Nelder–Mead, pattern search, CMA-ES, SPSA — use
  *only* the `value` oracle the engine already ships, so they are content, not an engine
  change. The friction is in the metrics: `gradient_norm` is inert without a gradient, so
  each needs a method-specific stopping criterion — a simplex diameter, mesh size, or
  stagnation test — which the engine already permits as content-defined dispatch (see
  [Stopping Criteria](@ref)). They fit the oracle but not the *default* convergence test.

### Tier 3 — a different iterate abstraction

These reach past the oracle into how an iterate is represented and updated — the deepest
change, touching engine core types rather than adding to them.

- **Riemannian / manifold optimization** — variables constrained to a manifold (the
  sphere, the Stiefel manifold, the SPD cone) replace the Euclidean update `x .-= α·d` with
  a retraction `x ← R_x(−α · grad f)` and a tangent-space gradient. This changes the
  `IterateGroup` update rule itself, not just the oracle.
- **Saddle-point / minimax** — `min_x max_y Φ(x, y)`, solved by extragradient or optimistic
  gradient descent-ascent, needs a two-block iterate and a coupled `(∇_x Φ, ∇_y Φ)` oracle
  — a structural rather than additive change to both the problem and the state.

## Not-yet-built demonstrator experiments

Distinct from the architectural stretch goals above is the backlog of demonstrator
*experiments* — SGD / logistic regression, constrained / projection problems, a live
`FileProblem` path, and smaller swept-variant figures — tracked in
[experiments/README.md](https://github.com/MoFirouzT/Iterative-Methods-Test-Engine/blob/main/experiments/README.md#planned--not-yet-built).
These are blocked on new content (and, where noted above, on the matching engine
extension — e.g. the stochastic oracle behind an SGD demonstrator), not on a gap in the
engine's design.
