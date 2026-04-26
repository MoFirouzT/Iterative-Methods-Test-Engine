"""
	Layer 8 — Persistence & Experiment Naming

Writes and restores experiment outputs in three formats:
- Full binary: result.jld2
- Per-method per-run CSV sidecars
- Lightweight manifest.json metadata
"""

using Dates
using CSV
using DataFrames
using JLD2
using JSON3


function _iterlog_row(run_id::Int, method_name::String, entry::IterationLog)
	row = Dict{Symbol,Any}(
		:run_id => run_id,
		:method_name => method_name,
		:iter => entry.iter,
		:core_time_ns => entry.core_time_ns,
		:objective => entry.objective,
		:gradient_norm => entry.gradient_norm,
		:step_norm => entry.step_norm,
	)

	for (key, value) in entry.extras
		row[key] = value
	end

	return row
end


function _methodresult_dataframe(run_id::Int, method_result::MethodResult)
	rows = [_iterlog_row(run_id, method_result.method_name, entry) for entry in method_result.iter_logs]
	return isempty(rows) ? DataFrame() : DataFrame(rows)
end


function _sanitize_filename(name::String)
	replace(name, r"[^A-Za-z0-9._\-\[\]\(\)=]+" => "_")
end


function _write_csv_sidecars(result::ExperimentResult)
	for run_result in result.run_results
		for (_, method_result) in run_result.method_results
			df = _methodresult_dataframe(run_result.run_id, method_result)
			csv_name = string("run", run_result.run_id, "_", _sanitize_filename(method_result.method_name), ".csv")
			csv_path = joinpath(result.experiment_path, csv_name)
			CSV.write(csv_path, df)
		end
	end
end


function _manifest_payload(result::ExperimentResult)
	methods = String[]
	if !isempty(result.run_results)
		methods = sort(collect(keys(result.run_results[1].method_results)))
	end

	return Dict(
		"name" => result.config.name,
		"timestamp" => string(result.timestamp),
		"host" => result.host,
		"n_runs" => length(result.run_results),
		"n_methods" => length(methods),
		"methods" => methods,
		"tags" => result.config.tags,
	)
end


"""
	save_experiment(result::ExperimentResult)

Writes:
- `result.jld2`
- `manifest.json`
- per-method CSV sidecars
"""
function save_experiment(result::ExperimentResult)
	mkpath(result.experiment_path)

	jld_path = joinpath(result.experiment_path, "result.jld2")
	JLD2.save(jld_path, Dict("result" => result))

	manifest = _manifest_payload(result)
	manifest_path = joinpath(result.experiment_path, "manifest.json")
	open(manifest_path, "w") do io
		JSON3.pretty(io, manifest)
	end

	_write_csv_sidecars(result)
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
