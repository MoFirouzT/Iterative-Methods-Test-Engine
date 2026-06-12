# Experiments

Runnable scripts that drive the engine end-to-end. Two tracks live here:

- **Portfolio experiments** (`exp_<problem>N.jl`) — the curated, problem-named
  demonstrators that produce the figures in the [top-level README](../README.md).
  One clean experiment per capability; these are the headline deliverables.
- **[Development stages](stages/)** (`stages/stageN.jl`) — the demoted Rosenbrock
  build log: scaffold that rehearsed the engine contract capability-by-capability.
  Not portfolio results — see [stages/README.md](stages/README.md).

`_bootstrap.jl` loads the engine + all content in dependency order; `_shared.jl`
holds shared plotting recipes. Both are includes, not runnable scripts.

## Portfolio experiments

| Script | Figure | Demonstrates |
|---|---|---|
| [`exp_lasso1_ista_fista.jl`](exp_lasso1_ista_fista.jl) | `lasso_ista_fista.png` (flagship) | Composite `f + g`, `prox` dispatch, Nesterov acceleration (ISTA → FISTA) on the lasso |
| [`exp_ls1_dimension.jl`](exp_ls1_dimension.jl) | `ls1_dimension.png` | Least-squares dimension sweep; matrix-free `OperatorHessian`; the core-time/wall-time timing pillar |
| [`exp_ls2_conditioning.jl`](exp_ls2_conditioning.jl) | `ls2_conditioning.png` | Conditioning controls the rate: `O(κ)` vs `O(√κ)` |
| [`exp_precond1_grid.jl`](exp_precond1_grid.jl) | `precond1_grid.png` | `VariantGrid` sweep + experimental/conventional routing; Jacobi preconditioning ≈ Newton |
| [`exp_tr1.jl`](exp_tr1.jl) | `tr1_trust_region.png` | Nested optimization: `TrustRegion` with a Steihaug-CG inner solve |

Run any one with `julia --project=. experiments/exp_<name>.jl`, or regenerate
every figure at once with [`scripts/reproduce.jl`](../scripts/reproduce.jl).

## Planned — not yet built

Deliberate scope boundary, listed so the extension points don't get lost. Each
is blocked on the noted work, not on the engine design.

- **SGD / logistic regression** — exercises the stochastic `step!` rng path
  *functionally* (per-`(seed, run_id)` row sampling, seed-variance IQR bands).
  Needs an SGD-flavored method + a logistic problem with mini-batches.
- **Constrained / projection problems** — an indicator-function regularizer
  (`prox` = projection) + a projected-gradient method on a box-constrained
  quadratic; a different `prox` shape from the lasso.
- **File-loaded problem** — exercise `FileProblem` / `register_file_loader!` in
  the live experiment path (unit-tested in `test/test_module9.jl`, no experiment).
- **`numerical_gradient` on an anisotropic problem** — central-difference
  correctness beyond smooth Rosenbrock, to catch step-selection bugs.
- **BB nonmonotone (GLL) safeguard** — the real fix for BB's clamp limitation
  (analysis in [`step_sizes.md §5.2`](../algorithms/components/step_sizes.md)).
- **JLD2 struct-of-arrays schema migration** — ~5–10× smaller `result.jld2`
  (analysis in [`architecture.md §10`](../docs/architecture.md#10-module-8--persistence--experiment-naming)).
- **Smaller swept variants** — a `MomentumStep` (heavy-ball) figure, a lasso
  sparsity-vs-λ sweep, and an `L2Norm` ridge demo are all cheap one-offs.
