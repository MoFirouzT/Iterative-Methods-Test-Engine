"""
    descent_directions.jl — Descent direction implementations

Provides the `DescentDirection` abstraction and the `SteepestDescent`
concrete direction (negative gradient).
"""

# Abstraction
abstract type DescentDirection end

# Steepest descent (negative gradient)
struct SteepestDescent <: DescentDirection end

"""
    compute_direction(::SteepestDescent, state, problem) -> Vector{Float64}

Return the steepest descent direction d_k = -∇f(x_k).
The function reads `state.iterate.gradient` and must not mutate `state`.
"""
function compute_direction(::SteepestDescent, state, problem)::Vector{Float64}
    return -state.iterate.gradient
end
