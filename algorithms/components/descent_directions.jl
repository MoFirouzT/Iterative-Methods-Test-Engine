"""
    descent_directions.jl — Descent direction implementations

Provides the `DescentDirection` abstraction and the `SteepestDescent`
concrete direction (negative gradient).
"""

# Abstraction
abstract type DescentDirection end

"""
    compute_direction(dir, state, problem) -> Vector{Float64}

Returns the descent direction d_k at the current iterate.
The returned vector is NOT normalized.
Normalization, if desired, is the responsibility of the step-size rule.

Preconditions (guaranteed by step! on entry):
  - state.iterate.x         holds the current iterate x_k
  - state.iterate.gradient  holds ∇f(x_k) (must be current — compute inside step!)
  - state.metrics.objective holds f(x_k) (must be current — compute inside step!)
"""
function compute_direction(dir::DescentDirection, state, problem)::Vector{Float64} end

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
