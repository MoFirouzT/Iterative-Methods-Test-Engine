using DataFrames
using Statistics: mean, median
using Makie
using CairoMakie


const METHOD_PALETTE = [
	"#0072B2", "#E69F00", "#009E73", "#CC79A7",
	"#56B4E9", "#D55E00", "#F0E442", "#000000",
]

const METHOD_COLOR_REGISTRY = Dict{String,String}()


function method_color(name::String)::String
	METHOD_PALETTE[(hash(name) % length(METHOD_PALETTE)) + 1]
end


register_method_color!(name::String, color::String) = (METHOD_COLOR_REGISTRY[name] = color)


get_method_color(name::String)::String = get(METHOD_COLOR_REGISTRY, name, method_color(name))


@kwdef struct MethodStyle
	color     :: Any
	linestyle :: Symbol = :solid
	linewidth :: Float64 = 2.0
	marker    :: Union{Nothing, Symbol} = nothing
	label     :: Union{Nothing, String} = nothing
end


@kwdef struct PlotSpec
	data          :: DataFrame
	x             :: Symbol = :iter
	y             :: Symbol = :objective
	group_by      :: Symbol = :method_name
	title         :: String = ""
	xlabel        :: String = ""
	ylabel        :: String = ""
	yscale        :: Symbol = :linear
	xscale        :: Symbol = :linear
	xlim          :: Union{Nothing,Tuple} = nothing
	ylim          :: Union{Nothing,Tuple} = nothing
	legend        :: Bool = true
	method_styles :: Dict{String,MethodStyle} = Dict()
	extra_kwargs  :: Dict = Dict()
end


@kwdef struct FigureLayout
	plots       :: Matrix{Union{PlotSpec,Nothing}}
	figure_size :: Tuple{Int,Int} = (1200, 900)
	title       :: String = ""
	padding     :: Int = 20
end


const _ANALYSIS_BASE_COLUMNS = [:run_id, :method_name, :iter, :objective, :gradient_norm,
								:step_norm, :dist_to_opt, :core_time_ns]


function _iterlog_row(run_id::Int, method_name::String, entry::IterationLog)
	row = Dict{Symbol,Any}(
		:run_id => run_id,
		:method_name => method_name,
		:iter => entry.iter,
		:objective => entry.objective,
		:gradient_norm => entry.gradient_norm,
		:step_norm => entry.step_norm,
		:dist_to_opt => entry.dist_to_opt,
		:core_time_ns => entry.core_time_ns,
	)

	for (key, value) in entry.extras
		haskey(row, key) || (row[key] = value)
	end

	return row
end


function _ordered_row_keys(rows::Vector{Dict{Symbol,Any}})
	ordered = Symbol[]
	seen = Set{Symbol}()

	for key in _ANALYSIS_BASE_COLUMNS
		any(row -> haskey(row, key), rows) || continue
		push!(ordered, key)
		push!(seen, key)
	end

	for row in rows
		for key in keys(row)
			key in seen && continue
			push!(ordered, key)
			push!(seen, key)
		end
	end

	return ordered
end


function to_dataframe(result::ExperimentResult)::DataFrame
	rows = Dict{Symbol,Any}[]

	for run_result in sort(result.run_results, by = r -> r.run_id)
		for method_name in sort(collect(keys(run_result.method_results)))
			method_result = run_result.method_results[method_name]
			for entry in method_result.iter_logs
				push!(rows, _iterlog_row(run_result.run_id, method_name, entry))
			end
		end
	end

	isempty(rows) && return DataFrame()

	ordered_keys = _ordered_row_keys(rows)
	columns = Dict(key => Vector{Any}(undef, length(rows)) for key in ordered_keys)

	for (index, row) in enumerate(rows)
		for key in ordered_keys
			columns[key][index] = get(row, key, missing)
		end
	end

	return DataFrame(columns)
end


function filter_methods(df::DataFrame, methods::Vector{String})::DataFrame
    :method_name in propertynames(df) || return copy(df)
    mask = in.(df[!, :method_name], Ref(methods))
    return df[mask, :]
end


function _aggregate_column(values::AbstractVector, mode::Symbol)
    filtered = collect(skipmissing(values))
    isempty(filtered) && return missing

    if mode == :mean
        return mean(filtered)
    elseif mode == :median
        return median(filtered)
    else
        throw(ArgumentError("unsupported aggregation mode $(mode)"))
    end
end


function aggregate_runs(df::DataFrame, mode::Symbol)::DataFrame
    mode == :all && return copy(df)
    mode in (:mean, :median) || throw(ArgumentError("aggregate_runs mode must be :all, :mean, or :median"))
    isempty(df) && return copy(df)

    column_names = collect(propertynames(df))

    group_keys = Symbol[]
    for key in (:method_name, :iter)
        key in column_names && push!(group_keys, key)
    end

    grouped = isempty(group_keys) ? [df] : groupby(df, group_keys; sort = true)
    out_rows = Dict{Symbol,Any}[]

    for subdf in grouped
        row = Dict{Symbol,Any}()

        for key in group_keys
            row[key] = subdf[1, key]
        end

        for name in column_names
            name in group_keys && continue
            name == :run_id && continue

            column = subdf[!, name]
            if all(v -> v === missing || v isa Number, column)
                row[name] = _aggregate_column(column, mode)
            elseif all(ismissing, column)
                row[name] = missing
            else
                row[name] = first(skipmissing(column))
            end
        end

        push!(out_rows, row)
    end

    isempty(out_rows) && return DataFrame()

    ordered_keys = _ordered_row_keys(out_rows)
    columns = Dict(key => Vector{Any}(undef, length(out_rows)) for key in ordered_keys)
    for (index, row) in enumerate(out_rows)
        for key in ordered_keys
            columns[key][index] = get(row, key, missing)
        end
    end

    return DataFrame(columns)
end


function _style_for_group(spec::PlotSpec, group_name::AbstractString)
    if haskey(spec.method_styles, group_name)
        return spec.method_styles[group_name]
    end

    return MethodStyle(color = get_method_color(group_name))
end


function _render_lines!(ax, spec::PlotSpec)
    if isempty(spec.data) || !(spec.group_by in propertynames(spec.data))
        return nothing
    end

    grouped = groupby(spec.data, spec.group_by; sort = true)
    for subdf in grouped
        group_value = subdf[1, spec.group_by]
        group_name = string(group_value)
        style = _style_for_group(spec, group_name)
        sorted = sort(subdf, spec.x)

        line_kwargs = merge(Dict{Symbol,Any}(
            :color => style.color,
            :linestyle => style.linestyle,
            :linewidth => style.linewidth,
            :label => isnothing(style.label) ? group_name : style.label,
        ), spec.extra_kwargs)

        lines!(ax, sorted[!, spec.x], sorted[!, spec.y]; line_kwargs...)

        if !isnothing(style.marker)
            scatter_kwargs = merge(Dict{Symbol,Any}(
                :color => style.color,
                :marker => style.marker,
            ), spec.extra_kwargs)
            scatter!(ax, sorted[!, spec.x], sorted[!, spec.y]; scatter_kwargs...)
        end
    end

    if spec.legend
        axislegend(ax)
    end

    return nothing
end


function render_figure(layout::FigureLayout)::Makie.Figure
	fig = Figure(resolution = layout.figure_size)

	for row in 1:size(layout.plots, 1), col in 1:size(layout.plots, 2)
		spec = layout.plots[row, col]
		isnothing(spec) && continue

		ax = Axis(fig[row, col],
			title = spec.title,
			xlabel = isempty(spec.xlabel) ? string(spec.x) : spec.xlabel,
			ylabel = isempty(spec.ylabel) ? string(spec.y) : spec.ylabel,
			yscale = spec.yscale == :log10 ? log10 : identity,
			xscale = spec.xscale == :log10 ? log10 : identity,
		)

		_render_lines!(ax, spec)
		!isnothing(spec.xlim) && xlims!(ax, spec.xlim...)
		!isnothing(spec.ylim) && ylims!(ax, spec.ylim...)
	end

	isempty(layout.title) || Label(fig[0, :], layout.title, fontsize = 18)
	return fig
end


function save_figure(fig::Makie.Figure, path::String)
	save(path, fig)
end
