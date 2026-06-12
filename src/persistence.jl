"""
	Module 8 — Persistence & Experiment Naming

Writes and restores experiment outputs in three formats:
- Full binary: result.jld2  (everything; all Julia types preserved)
- Per-method per-run CSV sidecars  (scalar extras only — vectors stay in JLD2)
- Lightweight manifest.json metadata, including the list of extras keys that
  were dropped from the CSV so the omission is auditable.
"""

using Dates
using CSV
using DataFrames
using JLD2
using JSON3


# ── JLD2 serialization shim for VariantGrid ────────────────────────────────
# `VariantGrid.builder` is `Function` and is most often supplied as a closure
# (see experiments/stages/stage5.jl: `(; step_size, kwargs...) -> GradientDescent(...)`).
# JLD2 cannot reliably round-trip closures (`#14#15` etc.) and emits a
# warning during `save_experiment`. Round-tripped closures would also be
# unsafe to call — the captured environment is gone.
#
# The fix: serialize a stripped-down record that drops `builder` and
# `filters` (both `Function`-typed). The expanded `VariantSpec`s inside the
# `ExperimentResult` already carry concrete `IterativeMethod` instances, so
# nothing downstream of `expand(grid)` needs the closure. On load, the grid
# is reconstructed with a sentinel builder that errors loudly if anyone
# tries to re-expand it post-mortem.
struct _SerializedVariantGrid
	base_name::String
	axes::Vector{VariantAxis}
	shared_params::NamedTuple
end

JLD2.writeas(::Type{VariantGrid}) = _SerializedVariantGrid

Base.convert(::Type{_SerializedVariantGrid}, g::VariantGrid) =
	_SerializedVariantGrid(g.base_name, g.axes, g.shared_params)

_deserialized_grid_builder(args...; kwargs...) = error(
	"VariantGrid.builder was not serialized — re-construct the grid " *
	"in-process if you need to re-expand it. The expanded VariantSpecs " *
	"inside ExperimentResult already carry concrete methods; consume those " *
	"instead.")

Base.convert(::Type{VariantGrid}, s::_SerializedVariantGrid) = VariantGrid(
	base_name     = s.base_name,
	axes          = s.axes,
	builder       = _deserialized_grid_builder,
	shared_params = s.shared_params,
)


# ── CSV scalar predicate and classifier ────────────────────────────────────
# A value is "CSV-friendly" if CSV.write can render it as a single cell
# without information loss. Composite values (Vector, Tuple, Dict, sub-logs,
# custom structs) have no stable text representation and are dropped from
# the CSV; they remain in result.jld2.
#
# Extension point: define `_is_csv_scalar(::YourType) = true` if you add a
# new extras type that should be CSV-preserved (e.g. `Dates.DateTime`).
_is_csv_scalar(x::Number)         = true   # covers Int, Float, Bool, Rational, ...
_is_csv_scalar(x::AbstractString) = true
_is_csv_scalar(x::Symbol)         = true
_is_csv_scalar(::Missing)         = true
_is_csv_scalar(::Nothing)         = true
_is_csv_scalar(x)                 = false


"""
	_classify_extras(iter_logs) -> (scalar_keys, vector_keys)

Splits the union of `extras` keys across `iter_logs` into those whose values
are CSV-scalar on every entry that has them, and those that are not. The rule
is all-or-nothing per key: a single non-scalar occurrence demotes the whole
column to JLD2-only, so we never write half a CSV column.
"""
function _classify_extras(iter_logs::Vector{IterationLog})
	all_keys = Set{Symbol}()
	for entry in iter_logs
		union!(all_keys, keys(entry.extras))
	end
	scalar_keys = Symbol[]
	vector_keys = Symbol[]
	for k in sort!(collect(all_keys))
		ok = all(iter_logs) do entry
			!haskey(entry.extras, k) || _is_csv_scalar(entry.extras[k])
		end
		push!(ok ? scalar_keys : vector_keys, k)
	end
	return scalar_keys, vector_keys
end


function _iterlog_row(run_id::Int, method_name::String, entry::IterationLog,
                       scalar_keys::Vector{Symbol})
	row = Dict{Symbol,Any}(
		:run_id => run_id,
		:method_name => method_name,
		:iter => entry.iter,
		:core_time_ns => entry.core_time_ns,
		:objective => entry.objective,
		:gradient_norm => entry.gradient_norm,
		:step_norm => entry.step_norm,
		:dist_to_opt => entry.dist_to_opt,
	)

	for k in scalar_keys
		row[k] = get(entry.extras, k, missing)
	end

	return row
end


"""
	_methodresult_dataframe(run_id, method_result) -> (df, vector_keys)

Builds a DataFrame for one method's run, dropping vector-valued extras.
Returns the dropped keys so save_experiment can aggregate them for the
manifest. Column order is stable: fixed fields first, scalar extras after.
"""
function _methodresult_dataframe(run_id::Int, method_result::MethodResult)
	iter_logs = method_result.iter_logs
	scalar_keys, vector_keys = _classify_extras(iter_logs)
	rows = [_iterlog_row(run_id, method_result.method_name, entry, scalar_keys)
	        for entry in iter_logs]

	if isempty(rows)
		return DataFrame(), vector_keys
	end

	df = DataFrame(rows)
	fixed = [:run_id, :method_name, :iter, :core_time_ns, :objective,
	         :gradient_norm, :step_norm, :dist_to_opt]
	df = df[!, vcat(fixed, scalar_keys)]

	return df, vector_keys
end


function _sanitize_filename(name::String)
	replace(name, r"[^A-Za-z0-9._\-\[\]\(\)=]+" => "_")
end


function _write_csv_sidecars(result::ExperimentResult)
	skipped = Set{Symbol}()
	for run_result in result.run_results
		for (_, method_result) in run_result.method_results
			df, vector_keys = _methodresult_dataframe(run_result.run_id, method_result)
			union!(skipped, vector_keys)
			csv_name = string("run", run_result.run_id, "_",
			                   _sanitize_filename(method_result.method_name), ".csv")
			csv_path = joinpath(result.experiment_path, csv_name)
			CSV.write(csv_path, df)
		end
	end
	return sort!(collect(skipped))
end


# Per-method, per-run termination summary derived from MethodResult and the
# final IterationLog entry. Embedded in every manifest so `jq '.method_results.BB1[0]'`
# is enough to know how a method terminated without loading result.jld2.
function _method_results_summary(result::ExperimentResult)
	out = Dict{String, Vector{Dict{String, Any}}}()
	for run_result in result.run_results
		for (name, mres) in run_result.method_results
			isempty(mres.iter_logs) && continue
			last_entry = mres.iter_logs[end]
			entry = Dict{String, Any}(
				"run_id"       => run_result.run_id,
				"n_iters"      => mres.n_iters,
				"stop_reason"  => string(mres.stop_reason),
				"f_final"      => last_entry.objective,
				"grad_final"   => last_entry.gradient_norm,
				"dist_final"   => last_entry.dist_to_opt,
			)
			push!(get!(out, name, Dict{String, Any}[]), entry)
		end
	end
	# Stable ordering inside each method's vector.
	for v in values(out)
		sort!(v, by = e -> e["run_id"])
	end
	return out
end


function _manifest_payload(result::ExperimentResult, skipped_extras::Vector{Symbol})
	methods = String[]
	if !isempty(result.run_results)
		methods = sort(collect(keys(result.run_results[1].method_results)))
	end

	manifest = Dict{String,Any}(
		"name" => result.config.name,
		"timestamp" => string(result.timestamp),
		"host" => result.host,
		"n_runs" => length(result.run_results),
		"n_methods" => length(methods),
		"methods" => methods,
		"tags" => result.config.tags,
		"method_results" => _method_results_summary(result),
	)

	if !isempty(skipped_extras)
		manifest["csv_skipped_extras"] = sort!(string.(skipped_extras))
		manifest["csv_skipped_extras_note"] = (
			"These extras keys are non-scalar (vectors, sub-logs, ...) " *
			"and are stored in result.jld2 only. Reload via load_experiment " *
			"for the full payload."
		)
	end

	return manifest
end


# JSON has no Inf/NaN. Map non-finite reals to `nothing` (→ null) so manifests
# stay valid for problems without a known optimum (where dist_to_opt = Inf).
_json_safe(x::AbstractFloat)  = isfinite(x) ? x : nothing
_json_safe(x::AbstractDict)   = Dict{String,Any}(string(k) => _json_safe(v) for (k, v) in x)
_json_safe(x::AbstractVector) = Any[_json_safe(v) for v in x]
_json_safe(x) = x


"""
	save_experiment(result::ExperimentResult;
	                compress = false,
	                extra_manifest = Dict{String,Any}())

Writes:
- `result.jld2`  (full binary; uncompressed by default — see `compress`)
- `run{N}_{method}.csv`  (scalar extras only)
- `manifest.json`  (metadata, base `method_results`, CSV-skipped extras keys,
   and any keys from `extra_manifest`)

CSVs are written before the manifest so the manifest can record which
extras were dropped.

# Keyword arguments
- `compress` — controls JLD2 compression for `result.jld2`. `false`
  (default) writes uncompressed; `true` uses JLD2's built-in codec; a
  specific `TranscodingStreams` codec value is also accepted and passed
  through verbatim. **Default is `false`** because on the Rosenbrock
  iter-log payload the built-in codec is a net loss (~5% overhead — the
  Dict{Symbol,Any} extras dominate and don't compress); see
  [docs/architecture.md §10 JLD2 compression] for measurements and the
  scenarios where opting in pays off. `load_experiment` reads either form
  transparently — no matching kwarg required.
- `extra_manifest::Dict{String,Any}` — keys merged into `manifest.json`
  after the framework's own fields. Use this to record stage-specific
  results (e.g. iters-to-milestone, tolerances used) so cold-restart
  queries can answer questions without recomputing from CSVs. Keys in
  `extra_manifest` override the base fields if they collide — caller's
  problem if that's not intended.
"""
function save_experiment(result::ExperimentResult;
                         compress = false,
                         extra_manifest::Dict{String,Any} = Dict{String,Any}())
	mkpath(result.experiment_path)

	jld_path = joinpath(result.experiment_path, "result.jld2")
	JLD2.save(jld_path, Dict("result" => result); compress = compress)

	skipped = _write_csv_sidecars(result)

	manifest = _manifest_payload(result, skipped)
	merge!(manifest, extra_manifest)

	manifest_path = joinpath(result.experiment_path, "manifest.json")
	open(manifest_path, "w") do io
		JSON3.pretty(io, _json_safe(manifest))
	end

	return nothing
end


"""
	load_experiment(path::String) -> ExperimentResult

Loads an experiment from `path/result.jld2`.
"""
function load_experiment(path::String)
	jld_path = joinpath(path, "result.jld2")
	data = JLD2.load(jld_path)
	return data["result"]
end


"""
	load_manifest(path::String)

Loads metadata from a manifest path.
"""
function load_manifest(path::String)
	return JSON3.read(read(path, String))
end


function _parse_experiment_id(path::String)
	number = try
		parse(Int, basename(path))
	catch
		0
	end

	date = basename(dirname(path))
	return date, number
end


function _manifest_get(manifest, key::Symbol, default)
	if hasproperty(manifest, key)
		return getproperty(manifest, key)
	end
	return default
end


"""
	list_experiments(log_root::String = "logs")

Returns a vector of metadata entries:
`(path, date, number, name, timestamp, n_methods, n_runs)`
"""
function list_experiments(log_root::String = "logs")
	isdir(log_root) || return NamedTuple[]

	out = NamedTuple[]
	for date_dir in readdir(log_root; join = true)
		isdir(date_dir) || continue
		date_name = basename(date_dir)
		occursin(r"^\d{8}$", date_name) || continue

		for exp_dir in readdir(date_dir; join = true)
			isdir(exp_dir) || continue
			exp_name = basename(exp_dir)
			occursin(r"^\d{3,}$", exp_name) || continue

			manifest_path = joinpath(exp_dir, "manifest.json")
			if isfile(manifest_path)
				manifest = load_manifest(manifest_path)
				push!(out, (
					path = exp_dir,
					date = String(date_name),
					number = parse(Int, exp_name),
					name = String(_manifest_get(manifest, :name, "")),
					timestamp = String(_manifest_get(manifest, :timestamp, "")),
					n_methods = Int(_manifest_get(manifest, :n_methods, 0)),
					n_runs = Int(_manifest_get(manifest, :n_runs, 0)),
				))
			else
				date, number = _parse_experiment_id(exp_dir)
				push!(out, (
					path = exp_dir,
					date = date,
					number = number,
					name = "",
					timestamp = "",
					n_methods = 0,
					n_runs = 0,
				))
			end
		end
	end

	sort!(out, by = x -> (x.date, x.number))
	return out
end
