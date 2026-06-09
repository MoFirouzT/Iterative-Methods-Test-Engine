# `:lasso` — sparse recovery problem family

The flagship composite problem for `ProximalGradient` (ISTA / FISTA):

    min_x  ½‖A x − b‖²  +  λ‖x‖₁

It reuses existing content — `LeastSquares` (data fidelity) and `L1Norm` (the
regularizer) — so this file only **registers a random generator**; it defines
no new objective.

## Regime

Underdetermined (`m < n`) with a sparse planted signal `x_star` (`k` nonzeros),
the regime where ℓ₁ recovery is interesting:

| param | default | meaning |
|---|---|---|
| `m` | 128 | rows (measurements) |
| `n` | 256 | columns (unknowns), `n > m` |
| `k` | 10  | nonzeros in the planted `x_star` |
| `λ` | 0.1 | ℓ₁ weight |

`A = randn(m, n) / √m` (benign Gaussian, near-isometric columns), `x_star` has
`±1` on a random support, `b = A x_star + 0.01·noise`.

## Metadata

- `meta[:L] = ‖A‖²` — the Lipschitz constant of `∇(½‖A·−b‖²)`. Feeds
  `FixedStep(α = 1/L)` directly, so the method needs no power iteration.
- `meta[:support]` — the true support indices (for the recovery panel).
- `meta[:λ]` — the ℓ₁ weight in effect.

## ⚠ `x_opt` is the *planted signal*, not the lasso minimizer

`x_opt = x_star` is the **ground-truth signal**, used by the support-recovery
panel and as a distance reference. With noise and shrinkage, the actual lasso
minimizer `x̂` differs from `x_star`, so:

- Do **not** stop on `DistanceToOptimal` expecting it to hit `0` — it won't.
- For convergence in the money figure, estimate `f* = min (f+g)` from a long
  reference run and plot `f − f*`; do not use `f(x_star)` as `f*`.

## Recovery regime caveat

Exact support recovery depends on incoherence-type conditions. The benign
Gaussian `A` and moderate sparsity here recover cleanly, but this is **not**
unconditional — state the regime in any figure caption.
