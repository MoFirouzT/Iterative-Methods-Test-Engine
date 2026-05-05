using Test

if !@isdefined(IterationLog)
    include(joinpath(@__DIR__, "..", "src", "logging.jl"))
end

function make_entry(iter::Int)
    IterationLog(
        iter = iter,
        core_time_ns = 100,
        objective = 10.0 / iter,
        gradient_norm = 1.0 / iter,
        step_norm = 0.1 * iter,
        extras = Dict{Symbol,Any}(:custom => iter),
    )
end

function make_logger_with(cfg::VerbosityConfig)
    Logger(
        "VerbTest",
        1,
        "",
        cfg,
        IterationLog[],
        NamedTuple[],
        Dict{Symbol,Any}(),
        0.0,
        0,
        IterationLog[],
    )
end

@testset "Module 7 verbosity system" begin
    io_silent = IOBuffer()
    logger_silent = make_logger_with(VerbosityConfig(level = SILENT, io = io_silent))
    maybe_print(logger_silent, make_entry(1))
    @test String(take!(io_silent)) == ""

    io_summary = IOBuffer()
    logger_summary = make_logger_with(VerbosityConfig(
        level = SUMMARY,
        print_every = 2,
        fields = [:iter, :objective],
        io = io_summary,
    ))
    maybe_print(logger_summary, make_entry(1))
    maybe_print(logger_summary, make_entry(2))
    out_summary = String(take!(io_summary))
    @test occursin("iter=2", out_summary)
    @test occursin("objective=5.0", out_summary)
    @test !occursin("iter=1", out_summary)

    io_detailed = IOBuffer()
    logger_detailed = make_logger_with(VerbosityConfig(
        level = DETAILED,
        print_every = 100,
        fields = [:iter],
        io = io_detailed,
    ))
    maybe_print(logger_detailed, make_entry(1))
    maybe_print(logger_detailed, make_entry(2))
    out_detailed = String(take!(io_detailed))
    @test occursin("iter=1", out_detailed)
    @test occursin("iter=2", out_detailed)

    io_range = IOBuffer()
    logger_range = make_logger_with(VerbosityConfig(
        level = MILESTONE,
        iter_range = 3:4,
        fields = [:iter],
        io = io_range,
    ))
    maybe_print(logger_range, make_entry(2))
    maybe_print(logger_range, make_entry(3))
    maybe_print(logger_range, make_entry(4))
    maybe_print(logger_range, make_entry(5))
    out_range = String(take!(io_range))
    @test !occursin("iter=2", out_range)
    @test occursin("iter=3", out_range)
    @test occursin("iter=4", out_range)
    @test !occursin("iter=5", out_range)

    io_debug = IOBuffer()
    logger_debug = make_logger_with(VerbosityConfig(
        level = DEBUG,
        fields = [:iter],
        io = io_debug,
    ))
    maybe_print(logger_debug, make_entry(1))
    out_debug = String(take!(io_debug))
    @test occursin("extras=", out_debug)
    @test occursin("custom", out_debug)

    io_milestone = IOBuffer()
    logger_milestone = make_logger_with(VerbosityConfig(level = MILESTONE, io = io_milestone))
    log_init!(logger_milestone, nothing, nothing)
    log_event!(logger_milestone, :max_iterations, 7)
    finalize!(logger_milestone, nothing, nothing)
    out_milestone = String(take!(io_milestone))
    @test occursin("start", out_milestone)
    @test occursin("event=max_iterations iter=7", out_milestone)
    @test occursin("finalize stop_reason=max_iterations", out_milestone)
end
