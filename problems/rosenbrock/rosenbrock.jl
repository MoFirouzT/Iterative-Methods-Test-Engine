"""
    rosenbrock.jl — Rosenbrock Problem Implementation

Implements the Rosenbrock function as a DataFidelity loss:

    f_ρ(x₁, x₂) = (1 - x₁)² + ρ·(x₂ - x₁²)²

with analytical gradient and Hessian-vector product.
"""

# Import the DataFidelity abstract type
# (This file is typically included after src/problems.jl is loaded)
# If loading standalone, ensure DataFidelity is available in the calling scope

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
    RosenbrockFidelity <: DataFidelity

The Rosenbrock objective function: f_ρ(x) = (1 - x₁)² + ρ·(x₂ - x₁²)²

# Fields
- `kernel::RosenbrockKernel` — contains the curvature parameter ρ
"""
struct RosenbrockFidelity <: DataFidelity
    kernel :: RosenbrockKernel
end

RosenbrockFidelity() = RosenbrockFidelity(RosenbrockKernel())


# ─────────────────────────────────────────────────────────────────────────
# DataFidelity Interface Implementation
# ─────────────────────────────────────────────────────────────────────────

"""
    value(f::RosenbrockFidelity, x::Vector{Float64}) -> Float64

Evaluate the Rosenbrock function at x:

    f(x) = (1 - x₁)² + ρ·(x₂ - x₁²)²
"""
function value(f::RosenbrockFidelity, x::Vector{Float64})::Float64
    ρ  = f.kernel.ρ
    x₁ = x[1]
    x₂ = x[2]
    return (1 - x₁)^2 + ρ * (x₂ - x₁^2)^2
end


"""
    grad!(g::Vector{Float64}, f::RosenbrockFidelity, x::Vector{Float64}) -> Vector{Float64}

Compute the gradient of the Rosenbrock function in-place into g:

    ∇f(x) = [
        -2(1 - x₁) - 4ρ·x₁·(x₂ - x₁²)
        2ρ·(x₂ - x₁²)
    ]

Returns the gradient vector g.
"""
function grad!(g::Vector{Float64}, f::RosenbrockFidelity, x::Vector{Float64})::Vector{Float64}
    ρ  = f.kernel.ρ
    x₁ = x[1]
    x₂ = x[2]
    v  = x₂ - x₁^2    # shared intermediate: x₂ - x₁²
    
    g[1] = -2(1 - x₁) - 4ρ * x₁ * v
    g[2] = 2ρ * v
    
    return g
end


"""
    hessian_vec(f::RosenbrockFidelity, x::Vector{Float64}, d::Vector{Float64}) -> Vector{Float64}

Compute the Hessian-vector product H(x)·d:

    ∇²f(x)·d = [
        (2 - 4ρ(x₂ - 3x₁²))·d₁ - 4ρ·x₁·d₂
        -4ρ·x₁·d₁ + 2ρ·d₂
    ]

# Arguments
- `f::RosenbrockFidelity` — the Rosenbrock objective
- `x::Vector{Float64}` — the point at which to evaluate the Hessian
- `d::Vector{Float64}` — the direction vector

# Returns
- `Vector{Float64}` — the Hessian-vector product
"""
function hessian_vec(f::RosenbrockFidelity,
                     x::Vector{Float64},
                     d::Vector{Float64})::Vector{Float64}
    ρ  = f.kernel.ρ
    x₁ = x[1]
    x₂ = x[2]
    d₁ = d[1]
    d₂ = d[2]
    
    # Hessian entries
    h₁₁ = 2 - 4ρ * (x₂ - 3x₁^2)
    h₁₂ = -4ρ * x₁
    h₂₂ = 2ρ
    
    return [
        h₁₁ * d₁ + h₁₂ * d₂,
        h₁₂ * d₁ + h₂₂ * d₂
    ]
end
