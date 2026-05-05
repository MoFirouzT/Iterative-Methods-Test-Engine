# Rosenbrock Problem — Specification

## 1. Problem Statement

Find the minimizer of the **Rosenbrock function**
(also known as Rosenbrock's banana function or the valley function):

$$f_\rho(x_1, x_2) = (1 - x_1)^2 + \rho\,(x_2 - x_1^2)^2, \qquad \rho > 0$$

The scalar $\rho$ is the **curvature parameter**.
The default value $\rho = 100$ is used unless stated otherwise.

This is a **non-composite** problem: there is no regularizer, so $f$ alone is the
total objective.

---

## 2. Mathematical Properties

### 2.1 Global Minimizer

$$x^* = (1,\, 1), \qquad f(x^*) = 0$$

The minimizer is unique and lies inside a narrow, curved parabolic valley defined
by $x_2 = x_1^2$.

### 2.2 Gradient

$$\nabla f(x) =
\begin{bmatrix}
-2(1 - x_1) - 4\rho\, x_1 (x_2 - x_1^2) \\[4pt]
2\rho\,(x_2 - x_1^2)
\end{bmatrix}$$

At the minimizer: $\nabla f(x^*) = \mathbf{0}$.

### 2.3 Hessian

$$\nabla^2 f(x) =
\begin{bmatrix}
2 - 4\rho(x_2 - 3x_1^2) & -4\rho\, x_1 \\[4pt]
-4\rho\, x_1              & 2\rho
\end{bmatrix}$$

At the minimizer:

$$\nabla^2 f(x^*) =
\begin{bmatrix}
8\rho + 2 & -4\rho \\
-4\rho     & 2\rho
\end{bmatrix}
\xrightarrow{\rho=100}
\begin{bmatrix}
802 & -400 \\
-400 & 200
\end{bmatrix}$$

**Condition number** at $x^*$:

$$\kappa(\nabla^2 f(x^*)) = \frac{\lambda_{\max}}{\lambda_{\min}} \approx 2508 \quad (\rho = 100)$$

This very large condition number, together with the non-convex curved valley, is
what makes the Rosenbrock function a canonical stress-test for iterative methods.

### 2.4 Convexity

The function is **non-convex** globally. It has no saddle points or local minima
other than $x^*$, but gradient-based methods can stall inside the curved valley
for many iterations.

---

## 3. Julia Implementation

### 3.1 Struct

```julia
# In: problems/rosenbrock/rosenbrock.jl

struct RosenbrockKernel
    ρ :: Float64    # curvature parameter (default 100.0)
end

struct RosenbrockObjective <: Objective
    kernel :: RosenbrockKernel
end
```

`RosenbrockObjective` subtypes the framework's `Objective` abstract type
(see `architecture.md` §3 — Module 1).

### 3.2 Objective Value

$$f(x) = (1 - x_1)^2 + \rho\,(x_2 - x_1^2)^2$$

```julia
function value(f::RosenbrockObjective, x::Vector{Float64})::Float64
    ρ      = f.kernel.ρ
    x₁, x₂ = x[1], x[2]
    return (1 - x₁)^2 + ρ * (x₂ - x₁^2)^2
end
```

### 3.3 Gradient

$$\nabla f(x) =
\begin{bmatrix}
-2(1 - x_1) - 4\rho\, x_1(x_2 - x_1^2) \\
2\rho\,(x_2 - x_1^2)
\end{bmatrix}$$

```julia
function grad(f::RosenbrockObjective, x::Vector{Float64})::Vector{Float64}
    ρ      = f.kernel.ρ
    x₁, x₂ = x[1], x[2]
    v      = x₂ - x₁^2          # shared intermediate: x₂ - x₁²
    return [
        -2(1 - x₁) - 4ρ * x₁ * v,
         2ρ * v
    ]
end
```

### 3.4 Hessian

The framework's `Objective` interface returns a `Hessian` object, not a single
vector or matrix (see `architecture.md` §3 — Module 1). For Rosenbrock the matrix
is 2×2, so the natural choice is `MatrixHessian` — both `apply(H, d)` and
`materialize(H)` are essentially free.

```julia
function hessian(f::RosenbrockObjective, x::Vector{Float64})::Hessian
    ρ      = f.kernel.ρ
    x₁, x₂ = x[1], x[2]
    H = [ 2 - 4ρ*(x₂ - 3x₁^2)   -4ρ*x₁ ;
          -4ρ*x₁                  2ρ    ]
    return MatrixHessian(H)
end
```

Consumers obtain Hessian-vector products and the full matrix uniformly:

```julia
H  = hessian(problem.f, x)     # MatrixHessian(2×2)
Hd = apply(H, d)               # H · d
M  = materialize(H)            # 2×2 Matrix{Float64}
```

> **Why `MatrixHessian` and not `OperatorHessian` or a custom subtype?** For a
> 2-dimensional problem the matrix is two flops away from any closure-based
> alternative, and `materialize(H)` lets debug checks (e.g. positive-definiteness
> sampling) and any future direct-solver step rule access the full matrix
> directly. For higher-dimensional problems where forming the matrix is
> expensive, `OperatorHessian` (closure over $d \mapsto \nabla^2 f(x) \cdot d$) is
> the right choice.

---

## 4. Problem Factory Registration

The Rosenbrock problem has no regularizer, so it should use the
**non-composite convenience constructor** `Problem(f, x0; meta = ..., x_opt = ...)`.
That keeps the registration readable while still allowing custom metadata:

```julia
register_problem!(:rosenbrock, (params, rng) -> begin
    ρ   = get(params, :rho, 100.0)
    x0  = get(params, :x0, [-1.2, 1.0])    # Rosenbrock's classical starting point
    f   = RosenbrockObjective(RosenbrockKernel(ρ))
    Problem(
        f, x0;
        x_opt = [1.0, 1.0],                # known minimizer, always (1, 1)
        meta  = Dict(:rho => ρ, :condition_number => 2508.0),
    )
end)
```

If a future Rosenbrock variant needs one or more regularizers, it should switch
to the composite convenience constructor instead of constructing the struct
positionally.

### Usage in ExperimentConfig

```julia
problem_spec = AnalyticProblem(
    name   = :rosenbrock,
    params = (rho = 100.0, x0 = [-1.2, 1.0]),
    dim    = 2,
)
```

---

## 5. Variable Mapping

| Math symbol            | Julia expression                  | Type             | Notes                          |
|------------------------|-----------------------------------|------------------|--------------------------------|
| $x \in \mathbb{R}^2$  | `state.iterate.x`                 | `Vector{Float64}` | current iterate               |
| $x_1, x_2$            | `state.iterate.x[1]`, `...[2]`   | `Float64`         | components                    |
| $x^* = (1, 1)$        | `problem.x_opt`                   | `Vector{Float64}` | known minimizer               |
| $f(x)$                | `value(problem.f, x)`             | `Float64`         | via `Objective` interface     |
| total $f(x)$ (= $f$)  | `total_objective(problem, x)`     | `Float64`         | reduces to `value(problem.f,x)` since `gs` is empty |
| $\nabla f(x)$         | `grad(problem.f, x)`              | `Vector{Float64}` | analytical gradient           |
| $\nabla^2 f(x)$       | `hessian(problem.f, x)`           | `MatrixHessian`   | 2×2 dense                     |
| $\nabla^2 f(x)\,d$   | `apply(hessian(problem.f, x), d)`| `Vector{Float64}` | Hessian-vector product        |
| full matrix           | `materialize(hessian(problem.f, x))` | `Matrix{Float64}` | full 2×2 matrix         |
| $\rho$                | `problem.f.kernel.ρ`              | `Float64`         | curvature parameter           |
| $v = x_2 - x_1^2$    | local `v` in `grad`/`value`       | `Float64`         | shared intermediate           |

---

## 6. Numerical Notes

- **Classical starting point**: $x^0 = (-1.2,\ 1.0)$ — lies outside the valley;
  standard in the optimization literature (Rosenbrock 1960).
- **Alternative starting point**: $x^0 = (0, 0)$ — inside the valley, easier.

---

## 7. References

- Rosenbrock, H.H. (1960). *An automatic method for finding the greatest or least
  value of a function*. The Computer Journal, 3(3), 175–184.
- Nocedal, J. & Wright, S.J. (2006). *Numerical Optimization* (2nd ed.). Springer.
  Example 2.1 (p. 28); condition number discussion §2.1.
