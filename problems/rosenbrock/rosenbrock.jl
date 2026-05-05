"""
    rosenbrock.jl — Rosenbrock Problem Implementation

Implements the Rosenbrock function as an Objective:

    f_ρ(x₁, x₂) = (1 - x₁)² + ρ·(x₂ - x₁²)²

with analytical gradient and Hessian.
"""

# Objective, MatrixHessian, and Problem are available from src/problems.jl

# ─────────────────────────────────────────────────────────────────────────
# Kernel & Fidelity Types
# ─────────────────────────────────────────────────────────────────────────

"""
    RosenbrockKernel

Encapsulates the curvature parameter of the Rosenbrock function.

# Fields
- `ρ::Float64` — curvature parameter (default 100.0)
"""
struct RosenbrockKernel
    ρ :: Float64
end

RosenbrockKernel() = RosenbrockKernel(100.0)


"""
    RosenbrockObjective <: Objective

The Rosenbrock objective function: f_ρ(x) = (1 - x₁)² + ρ·(x₂ - x₁²)²

# Fields
- `kernel::RosenbrockKernel` — contains the curvature parameter ρ
"""
struct RosenbrockObjective <: Objective
    kernel :: RosenbrockKernel
end

RosenbrockObjective() = RosenbrockObjective(RosenbrockKernel())


# ─────────────────────────────────────────────────────────────────────────
# Objective Interface Implementation
# ─────────────────────────────────────────────────────────────────────────

"""
    value(f::RosenbrockObjective, x::Vector{Float64}) -> Float64

Evaluate the Rosenbrock function at x:

    f(x) = (1 - x₁)² + ρ·(x₂ - x₁²)²
"""
function value(f::RosenbrockObjective, x::Vector{Float64})::Float64
    ρ  = f.kernel.ρ
    x₁ = x[1]
    x₂ = x[2]
    return (1 - x₁)^2 + ρ * (x₂ - x₁^2)^2
end


"""
    grad(f::RosenbrockObjective, x::Vector{Float64}) -> Vector{Float64}

Compute the gradient of the Rosenbrock function:

    ∇f(x) = [
        -2(1 - x₁) - 4ρ·x₁·(x₂ - x₁²)
        2ρ·(x₂ - x₁²)
    ]

Returns a new gradient vector.
"""
function grad(f::RosenbrockObjective, x::Vector{Float64})::Vector{Float64}
    ρ  = f.kernel.ρ
    x₁ = x[1]
    x₂ = x[2]
    v  = x₂ - x₁^2    # shared intermediate: x₂ - x₁²
    
    return [
        -2(1 - x₁) - 4ρ * x₁ * v,
         2ρ * v
    ]
end


"""
    hessian(f::RosenbrockObjective, x::Vector{Float64}) -> MatrixHessian

Compute the Hessian of the Rosenbrock function as a MatrixHessian object:

    ∇²f(x) = [
        2 - 4ρ(x₂ - 3x₁²)    -4ρ·x₁
        -4ρ·x₁                2ρ
    ]

# Arguments
- `f::RosenbrockObjective` — the Rosenbrock objective
- `x::Vector{Float64}` — the point at which to evaluate the Hessian

# Returns
- `MatrixHessian` — the 2×2 Hessian matrix
"""
function hessian(f::RosenbrockObjective, x::Vector{Float64})::Hessian
    ρ  = f.kernel.ρ
    x₁ = x[1]
    x₂ = x[2]
    
    H = [
        2 - 4ρ * (x₂ - 3x₁^2)   -4ρ * x₁ ;
        -4ρ * x₁                  2ρ
    ]
    
    return MatrixHessian(H)
end


# ─────────────────────────────────────────────────────────────────────────
# Problem Factory Registration
# ─────────────────────────────────────────────────────────────────────────

"""
    register_rosenbrock!()

Register the Rosenbrock problem with the problem factory.
Invoked automatically when this module is loaded.
"""
function register_rosenbrock!()
    register_analytic_problem!(:rosenbrock, (params) -> begin
        ρ   = get(params, :rho, 100.0)
        x0  = get(params, :x0, [-1.2, 1.0])
        f   = RosenbrockObjective(RosenbrockKernel(ρ))
        Problem(
            f, x0;
            x_opt = [1.0, 1.0],
            meta  = Dict(:rho => ρ, :condition_number => 2508.0),
        )
    end)
end

# Auto-register on module load
register_rosenbrock!()
