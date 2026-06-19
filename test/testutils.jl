# test/testutils.jl
#
# Shared, dependency-light test helpers. Each test file includes this right after
# the bootstrap; re-including only redefines the helpers, so it is idempotent and
# the files remain runnable standalone.

using Test

# A no-output logger for tests that inspect logger/result *state*, not printed
# text. Fully qualified so it is immune to any Main-level `make_logger` shadowing
# (e.g. fixture loggers defined inside a test file).
silent_logger(name::AbstractString = "test") =
    TestEngine.make_logger(String(name), 1, "", VerbosityConfig(level = SILENT))
