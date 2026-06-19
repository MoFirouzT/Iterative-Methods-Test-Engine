# Experiments

Runnable scripts that drive the engine end-to-end. Two tracks live here:

- **Portfolio experiments** (`exp_<topic>.jl`) — the curated demonstrators, named
  after their headline topic: sometimes the problem (`ls1`/`ls2` least-squares),
  sometimes the method or capability (`precond`, `tr_steihaug_cg`, `lasso_ista_fista`).
  They produce the five figures in [`figures/`](../figures/), catalogued with what
  each one demonstrates below. One clean experiment per capability — the headline
  deliverables.
- **[Development stages](stages/)** (`stages/stageN.jl`) — a capability-by-capability
  validation of the engine, built in dependency order on a single 2D Rosenbrock
  problem: each stage drives one architectural block and asserts its contract
  before the next stage depends on it. Stage 0 runs in CI. These are the
  integration checks behind the portfolio figures, not figures themselves — see
  [stages/README.md](stages/README.md).

`_bootstrap.jl` loads the engine + all content in dependency order; `_shared.jl`
holds the shared method builders, palette, and plotting recipes used across the
GD step-size comparisons. Both are includes, not runnable scripts.

## Portfolio experiments

Each capability has exactly one clean, working demonstrator you can watch run — no
method zoo. The five shipped experiments:

| Script | Figure | Demonstrates |
| --- | --- | --- |
| [`exp_lasso_ista_fista.jl`](exp_lasso_ista_fista.jl) | `lasso_ista_fista.png` (flagship) | One `ProximalGradient`, **extrapolation component swept** (ISTA → heavy-ball → FISTA); composite `f + g`, `prox` dispatch, Nesterov acceleration on the lasso |
| [`exp_ls1_dimension.jl`](exp_ls1_dimension.jl) | `ls1_dimension.png` | Least-squares dimension sweep; matrix-free `OperatorHessian`; the core-time/wall-time timing pillar |
| [`exp_ls2_conditioning.jl`](exp_ls2_conditioning.jl) | `ls2_conditioning.png` | Conditioning controls the rate: `O(κ)` vs `O(√κ)` |
| [`exp_precond_grid.jl`](exp_precond_grid.jl) | `precond_grid.png` | `VariantGrid` sweep + role-based baseline/experimental routing; Jacobi preconditioning ≈ Newton |
| [`exp_tr_steihaug_cg.jl`](exp_tr_steihaug_cg.jl) | `tr_steihaug_cg.png` | Nested optimization: `TrustRegion` with a Steihaug-CG inner solve |

Run any one with `julia --project=. experiments/exp_<name>.jl`, or regenerate
every figure at once with [`reproduce.jl`](../reproduce.jl).

Correctness is **externally cross-checked**: converged solutions are matched against
`A\b`, [`Optim.jl`](https://github.com/JuliaNLSolvers/Optim.jl) (GradientDescent / LBFGS),
and [`ProximalAlgorithms.jl`](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl)
(ForwardBackward / FastForwardBackward) — see [`test/test_external_validation.jl`](../test/test_external_validation.jl).

### Figures

<img src="../figures/lasso_ista_fista.png" width="560" alt="One ProximalGradient on the lasso with its extrapolation component swept — ISTA, heavy-ball, FISTA — plus exact support recovery"><br>
*`lasso_ista_fista` (flagship) — one `ProximalGradient` with the extrapolation component swept: ISTA → heavy-ball → FISTA orders the convergence, plus exact support recovery.*

<img src="../figures/ls1_dimension.png" width="560" alt="Least-squares dimension sweep: convergence and the core-time/wall-time timing pillar across problem sizes"><br>
*`ls1_dimension` — dimension sweep; matrix-free `OperatorHessian`; the core-time/wall-time timing pillar.*

<img src="../figures/ls2_conditioning.png" width="560" alt="Least squares under a conditioning sweep: GD's rate degrades as O(κ) vs O(√κ)"><br>
*`ls2_conditioning` — conditioning controls the rate: `O(κ)` vs `O(√κ)`.*

<img src="../figures/precond_grid.png" width="560" alt="VariantGrid sweep of PreconditionedGradient vs a baseline; Jacobi preconditioning matches Newton on a diagonal quadratic"><br>
*`precond_grid` — `VariantGrid` sweep + role-based baseline/experimental routing; Jacobi preconditioning ≈ Newton.*

<img src="../figures/tr_steihaug_cg.png" width="560" alt="TrustRegion with a Steihaug-CG inner solve: nested optimization with inner-vs-outer core time"><br>
*`tr_steihaug_cg` — nested optimization: `TrustRegion` with a Steihaug-CG inner solve.*

## Where the design extends next

These figures cover the deterministic method families end-to-end. The one method
*category* they don't exercise is the **stochastic `step!` / rng path** — an
SGD-flavored method on a mini-batched problem (logistic regression), with
seed-variance bands over per-`(seed, run_id)` sampling. The hooks for it already
exist in the engine; it's the natural next demonstrator.

Adding any new method, problem, or component is the same one-type-plus-one-method
extension shown in [walkthrough.ipynb](../walkthrough.ipynb) and
[extending.md](../docs/src/extending.md) — that's the design working as intended,
not a backlog.
