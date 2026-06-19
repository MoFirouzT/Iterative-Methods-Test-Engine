"""
	descent_directions.jl — Descent direction implementations

Provides the `DescentDirection` abstraction and the `SteepestDescent`
concrete direction (negative gradient).
"""

using .TestEngine: @core_timed

# Abstraction
abstract type DescentDirection end

"""
	compute_direction(dir, state, problem) -> Vector{Float64}

Returns the descent direction d_k at the current iterate.
The returned vector is NOT normalized.
Normalization, if desired, is the responsibility of the step-size rule.

Timing contract: each direction rule wraps its own core computation in
`@core_timed state` — mirroring the step-size rules — so that a rule with
internal bookkeeping (e.g. a quasi-Newton two-loop recursion or a linear solve)
counts only its mathematical kernel, not its scaffolding. The caller (`step!`)
therefore does **not** wrap `compute_direction` in `@core_timed`.

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
Reads `state.iterate.gradient`; mutates only `state.timing` (via `@core_timed`).
Kept allocation-returning rather than writing in place: the `DescentDirection`
contract (see descent_directions.md §1) is deliberately *pure* — the component
reads `state` and returns a fresh direction; the caller stores it. The per-step
`O(n)` temporary is the price of that purity and is dominated by the objective's
`O(mn)`-`O(n²)` kernel work at any scale where the timing pillar speaks.
"""
function compute_direction(::SteepestDescent, state, problem)::Vector{Float64}
	local d
	@core_timed state begin
		d = -state.iterate.gradient
	end
	return d
end
