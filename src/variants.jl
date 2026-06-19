"""
	Variant Grid Engine

Defines the shared component vocabulary for experimental methods and provides
Cartesian expansion plus deterministic auto-naming for variant grids.
"""

using Base: @kwdef


# ─────────────────────────────────────────────────────────────────────────
# Method-construction component vocabulary (Extrapolation / Preconditioner and
# their concretes) lives in the CONTENT layer — the engine grid machinery
# below is method-agnostic and never references it:
#   algorithms/components/extrapolation.jl   — Extrapolation, NesterovStep, …
#   algorithms/components/preconditioners.jl — Preconditioner, Identity/Jacobi
# ─────────────────────────────────────────────────────────────────────────


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

`role` declares the comparison role every variant in the grid plays —
`:baseline` or `:experimental`. Because a method's category is no longer part
of its type, `resolve_methods` routes the whole grid's expanded specs into the
bucket named by `role`. Defaults to `:experimental` (the typical variant-driven
exploration); a grid of reference baselines should set `role = :baseline`.
"""
@kwdef struct VariantGrid
	base_name::String
	axes::Vector{VariantAxis}
	builder::Function
	filters::Vector{Function} = Function[]
	shared_params::NamedTuple = (;)
	role::Symbol = :experimental
end

"""
	VariantSpec

Concrete expanded variant ready to run.

The `method` field is typed `IterativeMethod` — the single method category.
Comparison role (baseline vs experimental) is not carried by the method; it is
declared on the producing `VariantGrid` via its `role`, and `resolve_methods`
routes every spec from a grid into that grid's bucket.
"""
struct VariantSpec
	name::String
	short_name::String
	params::NamedTuple
	method::IterativeMethod
end


# ─────────────────────────────────────────────────────────────────────────
# Naming Helpers
# ─────────────────────────────────────────────────────────────────────────

# Generic, content-agnostic entries only. Concrete method/component abbreviations
# are registered by their own content files via `register_abbreviation!` on load
# (e.g. algorithms/components/extrapolation.jl, step_sizes.jl), keeping the engine
# free of any concrete method/component vocabulary.
const ABBREVIATIONS = Dict{String,String}(
	"None" => "∅",
)

abbreviate(value) = get(ABBREVIATIONS, string(value), string(value))


"""
    register_abbreviation!(full::AbstractString, short::AbstractString) -> short

Register `full` ↦ `short` in the [`ABBREVIATIONS`](@ref) lookup so subsequent
calls to `abbreviate(full)` (used by `VariantSpec` short_name generation,
plot legends, etc.) return `short`. Overwrites any existing entry for `full`.

Call this *before* expanding a `VariantGrid` so the generated `short_name`
of each spec uses the friendly form rather than the long type name.

# Example
```julia
register_abbreviation!("GradientDescent", "GD")
register_abbreviation!("BarzilaiBorwein", "BB")
```
"""
function register_abbreviation!(full::AbstractString, short::AbstractString)
    ABBREVIATIONS[String(full)] = String(short)
    return short
end

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
