"""
	Module 2 — Variant Grid Engine

Defines the shared component vocabulary for experimental methods and provides
Cartesian expansion plus deterministic auto-naming for variant grids.
"""

using Base: @kwdef


# ─────────────────────────────────────────────────────────────────────────
# Component Abstractions
# ─────────────────────────────────────────────────────────────────────────

"""
	abstract type HessianApprox

Base type for Hessian approximation strategies.
"""
abstract type HessianApprox end

"""
	struct FullHessian <: HessianApprox

Dense exact Hessian.
"""
struct FullHessian <: HessianApprox end

"""
	struct BFGS <: HessianApprox

BFGS rank-2 update.
"""
struct BFGS <: HessianApprox end

"""
	struct SR1 <: HessianApprox

Symmetric rank-1 update.
"""
struct SR1 <: HessianApprox end

"""
	struct LBFGS <: HessianApprox

Limited-memory BFGS update.
"""
@kwdef struct LBFGS <: HessianApprox
	m::Int = 5
end

"""
	struct DiagBFGS <: HessianApprox

Diagonal BFGS approximation.
"""
@kwdef struct DiagBFGS <: HessianApprox
	damped::Bool = false
end

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



# ─────────────────────────────────────────────────────────────────────────
# Grid Types
# ─────────────────────────────────────────────────────────────────────────

"""
	VariantAxis

One named dimension of a variant grid.
"""
struct VariantAxis
	param::Symbol
	values::Vector{Any}
	labels::Vector{String}
end

"""
	VariantAxis(param::Symbol, labeled_values::Pair...)

Convenience constructor using `value => "label"` pairs.
"""
function VariantAxis(param::Symbol, labeled_values::Pair...)
	values = Any[first(pair) for pair in labeled_values]
	labels = String[last(pair) for pair in labeled_values]
	length(values) == length(labels) || throw(ArgumentError("values and labels must have the same length"))
	VariantAxis(param, values, labels)
end

"""
	VariantGrid

Declarative description of a family of method variants.
"""
@kwdef struct VariantGrid
	base_name::String
	axes::Vector{VariantAxis}
	builder::Function
	filters::Vector{Function} = Function[]
	shared_params::NamedTuple = (;)
end

"""
	VariantSpec

Concrete expanded variant ready to run.
"""
struct VariantSpec
	name::String
	short_name::String
	params::NamedTuple
	method::ExperimentalMethod
end


# ─────────────────────────────────────────────────────────────────────────
# Naming Helpers
# ─────────────────────────────────────────────────────────────────────────

const ABBREVIATIONS = Dict{String,String}(
	"MyMethod" => "MM",
	"BFGS" => "BFGS",
	"SR1" => "SR1",
	"LBFGS" => "LBFG",
	"None" => "∅",
	"NoMinorUpdate" => "∅",
	"Momentum" => "Mom",
	"MomentumStep" => "Mom",
	"Nesterov" => "Nest",
	"NesterovStep" => "Nest",
	"CorrectionStep" => "Corr",
	"Armijo" => "Arm",
	"ArmijoLS" => "Arm",
	"Wolfe" => "Wlf",
	"WolfeLS" => "Wlf",
)

abbreviate(value) = get(ABBREVIATIONS, string(value), string(value))

_format_full_piece(param::Symbol, label::AbstractString) = string(param, "=", label)
_format_short_piece(label::AbstractString) = abbreviate(label)

_full_name(base_name::AbstractString, pieces::Vector{String}) = isempty(pieces) ? String(base_name) : string(base_name, "[", join(pieces, ","), "]")
_short_name(base_name::AbstractString, pieces::Vector{String}) = isempty(pieces) ? abbreviate(base_name) : string(abbreviate(base_name), "/", join(pieces, "/"))


# ─────────────────────────────────────────────────────────────────────────
# Expansion
# ─────────────────────────────────────────────────────────────────────────

function _axis_choices(axis::VariantAxis)
	length(axis.values) == length(axis.labels) || throw(ArgumentError("axis $(axis.param) has mismatched values and labels"))
	collect(zip(axis.values, axis.labels))
end

function _combo_namedtuple(params::Vector{Pair{Symbol,Any}})
	isempty(params) && return (;)
	(; params...)
end

function _merge_namedtuples(axis_params::Vector{Pair{Symbol,Any}}, shared_params::NamedTuple)
	merged_params = copy(axis_params)
	for (key, value) in pairs(shared_params)
		push!(merged_params, key => value)
	end
	_combo_namedtuple(merged_params)
end

function expand(grid::VariantGrid)::Vector{VariantSpec}
	axis_choices = [_axis_choices(axis) for axis in grid.axes]
	products = isempty(axis_choices) ? ((),) : Iterators.product(axis_choices...)

	specs = VariantSpec[]
	for selection in products
		axis_params = Pair{Symbol,Any}[]
		full_pieces = String[]
		short_pieces = String[]

		for (axis, choice) in zip(grid.axes, selection)
			value, label = choice
			push!(axis_params, axis.param => value)
			push!(full_pieces, _format_full_piece(axis.param, label))
			push!(short_pieces, _format_short_piece(label))
		end

		combo = _combo_namedtuple(axis_params)

		if !isempty(grid.shared_params)
			for key in intersect(collect(keys(grid.shared_params)), collect(keys(combo)))
				throw(ArgumentError("shared parameter $(key) conflicts with grid axis $(key)"))
			end
		end

		merged = _merge_namedtuples(axis_params, grid.shared_params)

		if !isempty(grid.shared_params)
			for (key, value) in pairs(grid.shared_params)
				shared_label = string(key, "=", value)
				push!(full_pieces, shared_label)
			end
		end

		if any(filter -> !filter(merged), grid.filters)
			continue
		end

		method = grid.builder(; merged...)
		push!(specs, VariantSpec(
			_full_name(grid.base_name, full_pieces),
			_short_name(grid.base_name, short_pieces),
			merged,
			method,
		))
	end

	specs
end
