"""
    minor_updates.jl — Minor-update (post-step correction) components (content)

Shared `MinorUpdate` vocabulary that methods compose — e.g. `NesterovStep`
turns ProximalGradient (ISTA) into FISTA. The engine machinery never
references these; they belong to the method-construction layer.
"""

using Base: @kwdef


"""
	abstract type MinorUpdate

Base type for post-step correction strategies.
"""
abstract type MinorUpdate end

"""
	struct NoMinorUpdate <: MinorUpdate

No-op correction.
"""
struct NoMinorUpdate <: MinorUpdate end

"""
	struct MomentumStep <: MinorUpdate

Simple momentum correction.
"""
@kwdef struct MomentumStep <: MinorUpdate
	α::Float64 = 0.1
end

"""
	struct NesterovStep <: MinorUpdate

Nesterov-style correction.
"""
@kwdef struct NesterovStep <: MinorUpdate
	α::Float64 = 0.1
end

"""
	struct CorrectionStep <: MinorUpdate

Iterated correction step.
"""
@kwdef struct CorrectionStep <: MinorUpdate
	n_inner::Int = 3
end
