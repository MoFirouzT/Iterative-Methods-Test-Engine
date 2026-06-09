"""
    least_squares.jl — Linear least-squares objective (content, not engine)

    f(x) = ½‖Ax − b‖²

Plugs into the engine through the `value` / `grad!` / `hessian` contract.
"""

import .TestEngine: Objective, Hessian, MatrixHessian, value, grad!, hessian
using LinearAlgebra: norm, mul!, adjoint


"""
	LeastSquaresKernel

Encapsulates the data matrix A and vector b for least-squares data fidelity.
"""
struct LeastSquaresKernel
	A::Matrix{Float64}
	b::Vector{Float64}
end


"""
	LeastSquares <: Objective

Least-squares objective: f(x) = 0.5 ‖Ax − b‖²
"""
struct LeastSquares <: Objective
	kernel::LeastSquaresKernel
end


function value(f::LeastSquares, x::Vector{Float64})
	residual = f.kernel.A * x - f.kernel.b
	0.5 * norm(residual)^2
end


function grad!(g::Vector{Float64}, f::LeastSquares, x::Vector{Float64})::Vector{Float64}
	residual = f.kernel.A * x - f.kernel.b
	mul!(g, adjoint(f.kernel.A), residual)
	return g
end


function hessian(f::LeastSquares, x::Vector{Float64})::Hessian
	H_matrix = f.kernel.A' * f.kernel.A
	return MatrixHessian(H_matrix)
end
