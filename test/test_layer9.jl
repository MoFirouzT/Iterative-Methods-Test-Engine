using Test
using LinearAlgebra: norm, Diagonal, I
using Random: MersenneTwister

include(joinpath(@__DIR__, "..", "src", "logging.jl"))
include(joinpath(@__DIR__, "..", "src", "core.jl"))
include(joinpath(@__DIR__, "..", "src", "problems.jl"))

@testset "Layer 9 Problem Factory" begin
    # Test LeastSquares data fidelity
    A = [1.0 2.0; 3.0 4.0]
    b = [5.0; 6.0]
    f_ls = LeastSquares(LeastSquaresKernel(A, b))
    x = [1.0; 1.0]
    
    # Value: 0.5 * ||(A*x - b)||^2
    residual = A * x - b
    expected_val = 0.5 * norm(residual)^2
    @test value(f_ls, x) ≈ expected_val
    
    # Gradient: A' * (A*x - b)
    grad_vec = zeros(2)
    grad!(grad_vec, f_ls, x)
    expected_grad = A' * residual
    @test grad_vec ≈ expected_grad
    
    # Hessian-vector: A' * A * d
    d = [1.0; 0.0]
    hv = hessian_vec(f_ls, x, d)
    expected_hv = A' * (A * d)
    @test hv ≈ expected_hv
    
    # Test L1 regularizer
    g_l1 = L1Norm(λ = 0.1)
    x = [1.0, -0.5, 0.3]
    expected_val_l1 = 0.1 * (1.0 + 0.5 + 0.3)
    @test value(g_l1, x) ≈ expected_val_l1
    
    # Proximal operator of L1 with γ = 1.0
    # prox_L1(x, γ) = sign(x) * max(|x| - γ*λ, 0)
    γ = 1.0
    prox_x = prox(g_l1, x, γ)
    expected_prox = sign.(x) .* max.(abs.(x) .- γ * 0.1, 0.0)
    @test prox_x ≈ expected_prox
    
    # Test L2 regularizer
    g_l2 = L2Norm(λ = 0.2)
    expected_val_l2 = 0.2 * norm(x)^2
    @test value(g_l2, x) ≈ expected_val_l2
    
    # Test Problem composition
    kernel = LeastSquaresKernel(A, b)
    x0 = zeros(2)
    problem = Problem(LeastSquares(kernel), g_l1, x0)
    
    @test problem.f isa LeastSquares
    @test length(problem.gs) == 1
    @test problem.x0 == x0
    @test problem.n == 2
    
    # Test objective computation
    test_x = [1.0; 1.0]
    obj = objective(problem, test_x)
    expected_obj = value(problem.f, test_x) + value(g_l1, test_x)
    @test obj ≈ expected_obj
    
    # Test zero regularizer
    g_zero = ZeroRegularizer()
    @test value(g_zero, x) == 0.0
    @test prox(g_zero, x, 1.0) ≈ x
    
    # Test AnalyticProblem spec
    spec = AnalyticProblem(name = :quadratic)
    rng = MersenneTwister(42)
    prob = make_problem(spec, rng)
    
    @test prob isa Problem
    @test prob.f isa LeastSquares
    @test prob.n >= 1
    
    # Test RandomProblem registration and instantiation
    register_random_problem!(:test_lasso, (rng, p) -> begin
        m = get(p, :m, 10)
        n = get(p, :n, 5)
        λ = get(p, :λ, 0.1)
        A = randn(rng, m, n)
        b = randn(rng, m)
        x0 = zeros(n)
        Problem(LeastSquares(LeastSquaresKernel(A, b)), L1Norm(λ = λ), x0)
    end)
    
    spec_random = RandomProblem(
        name = :test_lasso,
        params = (m = 15, n = 8, λ = 0.2),
    )
    rng2 = MersenneTwister(43)
    prob_random = make_problem(spec_random, rng2)
    
    @test prob_random isa Problem
    @test prob_random.n == 8
    @test prob_random.f isa LeastSquares
    @test length(prob_random.gs) == 1
    @test prob_random.gs[1] isa L1Norm
    
    # Test FileProblem (with a simple in-memory loader)
    spec_file = FileProblem(
        path = "dummy_path.txt",
        loader = (path) -> begin
            A = Matrix{Float64}(I, 3, 3)
            b = ones(3)
            x0 = zeros(3)
            Problem(LeastSquares(LeastSquaresKernel(A, b)), x0)
        end,
        description = "Test file problem",
    )
    rng3 = MersenneTwister(44)
    prob_file = make_problem(spec_file, rng3)
    
    @test prob_file isa Problem
    @test prob_file.n == 3
    
    # Test error handling for unregistered problems
    spec_missing = AnalyticProblem(name = :nonexistent)
    @test_throws KeyError make_problem(spec_missing, MersenneTwister(1))
    
    spec_missing_random = RandomProblem(name = :nonexistent)
    @test_throws KeyError make_problem(spec_missing_random, MersenneTwister(1))
end
