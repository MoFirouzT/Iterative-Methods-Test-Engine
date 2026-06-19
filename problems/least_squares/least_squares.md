# Linear least squares — `LeastSquares` + the `:linear_ls` / `:quadratic` families

A second, scalable, well-understood problem family. Where 2-D Rosenbrock is
small and fixed-shape, least squares lets us sweep **dimension** and
**conditioning** independently and exercise the parts of the engine Rosenbrock
structurally can't reach: `OperatorHessian`, dimension scaling, and the
core-time/wall-time timing pillar (the 2-D kernel sits below the timing noise
floor; an `n = 1000` matvec does not).

## Objective

    f(x) = ½‖A x − b‖²,   A ∈ ℝ^{m×n}
    ∇f(x)  = Aᵀ(A x − b)
    ∇²f    = AᵀA          (constant)

```julia
mutable struct LeastSquaresKernel
    A::Matrix{Float64}; b::Vector{Float64}; residual::Vector{Float64}
    AtA::Union{Nothing,Matrix{Float64}}     # memoized AᵀA for :matrix-mode Hessian (lazy)
end
LeastSquaresKernel(A, b) = LeastSquaresKernel(A, b, similar(b), nothing)  # all call sites use this 2-arg form
struct LeastSquares <: Objective; kernel::LeastSquaresKernel; hessian_mode::Symbol; end
LeastSquares(kernel) = LeastSquares(kernel, :matrix)   # default preserves prior behavior
```

`value`/`grad!` are **allocation-free**: both write the residual `A x − b` through
`mul!` into the kernel's preallocated length-`m` `residual` buffer rather than
allocating `A*x` and `A*x − b` each call (`grad!` then does `g ← Aᵀ·residual`
in place). This is the per-step hot kernel — it runs every iteration for every
method — so on `n = 1000` the change drops `GradientDescent` from ~10 length-`n`
temporaries/step to ~2 (the residual buffer is single-threaded scratch; the
engine runs methods sequentially). Per-step allocation is `O(n)` bytes while the
dense matvec is `O(n²)` FLOPs, so the allocation share vanishes at the scale
where the timing pillar speaks.

The Hessian path is cached too, since `∇²f = AᵀA` is constant: `:matrix` mode
memoizes `AᵀA` in the kernel and forms the `n×n` product at most once (lazily, on
the first `hessian` call) instead of on every call; `:operator` mode's `apply`
reuses a preallocated length-`m` scratch for `A d` via `mul!`, leaving only the
fresh result `n`-vector that the `Hessian` contract requires callers to receive.

### Selectable Hessian representation (`hessian_mode`)

`∇²f = AᵀA` is constant; *how* it is represented is a deliberate choice, not a
fixed one — this is the "each `Hessian` declares which operations it supports"
contract made concrete:

| mode | returns | `apply` | `materialize` / `diagonal` | use |
|---|---|---|---|---|
| `:matrix` (default) | `MatrixHessian(AᵀA)` | matrix-vector | ✓ | small n; methods that read the diagonal (Jacobi preconditioner) |
| `:operator` | `OperatorHessian(d → Aᵀ(A d), n)` | two O(mn) matvecs | ✗ | large n; the `:linear_ls` conditioning family |

The default is `:matrix` so every existing call site (`:quadratic`, the lasso,
the tests) is unchanged. `CauchyStep` only calls `apply`, so it works in either
mode — which is what lets the `:operator` family light `OperatorHessian` up
without any method change.

## Families

### `:quadratic` (analytic)
A small explicit quadratic `½‖A x − b‖²` from caller-supplied `A`, `b`, `x0`
(defaults to 2×2 identity). Used by the early stage tests.

### `:linear_ls` (random) — conditioning & dimension

Parametrized by the **Hessian** condition number `κ = cond(AᵀA)`, the quantity
that actually governs the gradient-method rate — **not** `cond(A)`. Because
`cond(AᵀA) = cond(A)²`, the generator sets `A`'s singular values to span
`1 → κ^(−1/2)`, giving `cond(A) = √κ` and hence `cond(AᵀA) = κ`. Getting this
squaring wrong would make the conditioning plot lie.

| param | default | meaning |
|---|---|---|
| `n` | 100 | unknowns |
| `m` | `2n` | rows (`m ≥ n`, full column rank) |
| `condition_number` | 1e3 | `κ = cond(AᵀA)` |

Construction: `A = U·Diag(s)·Vᵀ` with `U` (m×n) and `V` (n×n) orthonormal
(materialize the compact `qr(...).Q` with `Matrix(...)`, then slice `U` to `n`
columns), `s = 10^range(0, −½log₁₀κ; length=n)`. The system is **consistent**:
`b = A·x_star`, so the minimizer is unique, `x_opt = x_star`, and `f(x_opt) = 0`.

> **Stop on gradient/distance, never on f-value.** `f* = 0` makes a relative-`f`
> tolerance degenerate. Use `GradientTolerance` or `DistanceToOptimal`.

Metadata: `meta[:condition_number] = κ`, `meta[:L] = σ_max² = 1` (the Lipschitz
constant of `∇f`, ready for `FixedStep(α = 1/L)`), `meta[:m] = m`.

## Experiments (the `exp_ls*` portfolio files)

- **`ls1` — dimension scaling** (`exp_ls1_dimension.jl`). Five GD variants,
  fixed `κ`, sweep `n ∈ {10, 100, 1000}` (`m = 2n`). Iters-to-tolerance is ~flat
  in `n` (the rate depends on `κ`, not `n`); wall time grows ~`O(mn)` from the
  matvec. The **core_time/wall_time** ratio climbs into `[50%, 110%]` at
  `n = 1000`, retroactively validating the timing-discipline design pillar.
- **`ls2` — conditioning sweep** (`exp_ls2_conditioning.jl`). Fixed
  `n = 100`, sweep `κ`. Iters-to-tolerance vs `κ`, log-log: Fixed/Armijo/Cauchy
  scale `O(κ)` (slope ≈ 1); **BB is markedly flatter** (`O(√κ)`) — the slope
  difference is the validation.

## Notes / gotchas

- Cauchy on a *true quadratic*: pass `CauchyStep(α_max = Inf)` — the quadratic
  model is exact, so the trust-radius cap (needed on Rosenbrock) only throttles
  the legitimate exact-line-search step (which routinely exceeds 1).
- This family surfaced a latent bug in `CauchyStep`'s curvature guard (an
  absolute threshold misfiring in the small-gradient regime); fixed to a
  scale-relative guard mirroring `BarzilaiBorwein`. See `step_sizes.md`.
