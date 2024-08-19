module StructureUtils

export upper_half_vals
export random_sparse_symbolics_matrix

export structure

export jacobian_structure
export hessian_of_lagrangian_structure

export dynamics_jacobian_structure
export dynamics_hessian_of_lagrangian_structure
export dynamics_structure

export sparsity_structure

export loss_hessian_structure

using NamedTrajectories
using TrajectoryIndexingUtils
using LinearAlgebra
using SparseArrays
using Symbolics
using ForwardDiff
using Einsum

function sparsity_structure(S::SparseMatrixCSC; return_pairs=false, type::Type=Float64)
    Is, Js = findnz(S)
    if return_pairs
        return zip(Is, Js)
    else
        return sparse(Is, Js, ones(type, nnz(S)), size(S)...)
    end
end

function sparsity_structure(A::AbstractMatrix; kwargs...)
    return sparsity_structure(sparse(A); kwargs...)
end



function upper_half_vals(A::AbstractMatrix)
    n = size(A, 1)
    vals = similar(A, n * (n + 1) ÷ 2)
    k = 1
    for j ∈ axes(A, 2)
        for i = 1:j
            vals[k] = A[i, j]
            k += 1
        end
    end
    return vals
end

# create an m x n sparse matrix filled with l symbolics num variables
function random_sparse_symbolics_matrix(m, n, l)
    A = zeros(Symbolics.Num, m * n)
    xs = collect(Symbolics.@variables(x[1:l])...)
    rands = randperm(m * n)[1:l]
    for i ∈ 1:l
        A[rands[i]] = xs[i]
    end
    return sparse(reshape(A, m, n))
end

function structure(A::SparseMatrixCSC; upper_half=false)
    I, J, _ = findnz(A)
    index_pairs = collect(zip(I, J))
    if upper_half
        @assert size(A, 1) == size(A, 2)
        index_pairs = filter(p -> p[1] <= p[2], index_pairs)
    end
    return index_pairs
end

function jacobian_structure(∂f::Function, xdim::Int)
    x = collect(Symbolics.@variables(x[1:xdim])...)
    return structure(sparse(∂f(x)))
end

dynamics_jacobian_structure(∂f::Function, zdim::Int) = jacobian_structure(∂f, 2zdim)

function hessian_of_lagrangian_structure(∂²f::Function, xdim::Int, μdim::Int)
    x = collect(Symbolics.@variables(x[1:xdim])...)
    μ = collect(Symbolics.@variables(μ[1:μdim])...)
    ∂²f_x = ∂²f(x)
    if length(size(∂²f_x)) == 3
        @einsum μ∂²f[j, k] := μ[i] * ∂²f_x[i, j, k]
    elseif length(size(∂²f_x)) == 2
        μ∂²f = μ[1] * ∂²f_x
    else
        error("hessian of lagrangian must be 2 or 3 dimensional")
    end
    return structure(sparse(μ∂²f); upper_half=true)
end

dynamics_hessian_of_lagrangian_structure(∂²f::Function, zdim::Int, μdim::Int) =
    hessian_of_lagrangian_structure(∂²f, 2zdim, μdim)

# function dynamics_structure(∂f̂::Function, traj::NamedTrajectory, dynamics_dim::Int)
#     ∂²f̂(zz) = reshape(
#         ForwardDiff.jacobian(x -> vec(∂f̂(x)), zz),
#         traj.dims.states,
#         2traj.dim,
#         2traj.dim
#     )

#     ∂f_structure = dynamics_jacobian_structure(∂f̂, traj.dim)

#     ∂F_structure = Tuple{Int,Int}[]

#     for t = 1:traj.T-1
#         ∂fₜ_structure = [
#             (
#                 i + index(t, 0, dynamics_dim),
#                 j + index(t, 0, traj.dim)
#             ) for (i, j) ∈ ∂f_structure
#         ]
#         append!(∂F_structure, ∂fₜ_structure)
#     end

#     μ∂²f_structure =
#         dynamics_hessian_of_lagrangian_structure(∂²f̂, traj.dim, dynamics_dim)

#     μ∂²F_structure = Tuple{Int,Int}[]

#     for t = 1:traj.T-1
#         μ∂²fₜ_structure = [ij .+ index(t, 0, traj.dim) for ij ∈ μ∂²f_structure]
#         append!(μ∂²F_structure, μ∂²fₜ_structure)
#     end

#     return ∂f_structure, ∂F_structure, μ∂²f_structure, μ∂²F_structure
# end

function dynamics_structure(
    ∂f::Function,
    μ∂²f::Function,
    traj::NamedTrajectory,
    dynamics_dim::Int;
    verbose=false,
    jacobian=true,
    hessian=true
)
    @assert hessian ? jacobian : true "if hessian is true, jacobian must be true"

    # getting inter knot point structure
    if jacobian
        if verbose
            println("            computing jacobian sparsity...")
        end
        # getting symbolic variables
        z1 = collect(Symbolics.@variables(z[1:traj.dim])...)
        z2 = collect(Symbolics.@variables(z[1:traj.dim])...)
        ∂f_structure = structure(sparse(∂f(z1, z2)))
    else
        if verbose
            println("            computing full jacobian block...")
        end
        ∂f_structure = [(i, j) for i = 1:dynamics_dim, j = 1:2traj.dim]
    end

    if hessian
        if verbose
            println("            computing hessian sparsity...")
        end
        μ = collect(Symbolics.@variables(μ[1:dynamics_dim])...)
        μ∂²f_structure = structure(sparse(μ∂²f(z1, z2, μ)); upper_half=true)
    else
        if verbose
            println("            computing full hessian block...")
        end
        μ∂²f_structure = [(i, j) for i = 1:2traj.dim, j = 1:2traj.dim if j ≥ i]
    end

    if verbose
        println("            creating full trajectory jacobian structure...")
    end
    ∂F_structure = Tuple{Int,Int}[]
    ∂F_structure = Vector{Tuple{Int,Int}}(undef, length(∂f_structure) * (traj.T - 1))

    @views Threads.@threads for t = 1:traj.T-1
        ∂fₜ_structure = [
            (
                i + index(t, 0, dynamics_dim),
                j + index(t, 0, traj.dim)
            ) for (i, j) ∈ ∂f_structure
        ]
        ∂F_structure[slice(t, length(∂f_structure))] .= ∂fₜ_structure
    end

    μ∂²F_structure = Tuple{Int,Int}[]

    if verbose
        println("            creating full trajectory hessian structure...")
    end

    μ∂²F_structure = Vector{Tuple{Int,Int}}(undef, length(μ∂²f_structure) * (traj.T - 1))

    @views Threads.@threads for t = 1:traj.T-1
        μ∂²fₜ_structure = [ij .+ index(t, 0, traj.dim) for ij ∈ μ∂²f_structure]
        μ∂²F_structure[slice(t, length(μ∂²f_structure))] .= μ∂²fₜ_structure
    end

    return ∂f_structure, ∂F_structure, μ∂²f_structure, μ∂²F_structure
end


function dynamics_structure(
    ∂f::Function,
    traj::NamedTrajectory,
    dynamics_dim::Int;
    verbose=false,
    jacobian=true,
)
    # getting inter knot point structure
    if jacobian
        if verbose
            println("            computing jacobian sparsity...")
        end
        # getting symbolic variables
        z1 = collect(Symbolics.@variables(z[1:traj.dim])...)
        z2 = collect(Symbolics.@variables(z[1:traj.dim])...)
        ∂f_structure = structure(sparse(∂f(z1, z2, 1)))
    else
        if verbose
            println("            computing full jacobian block...")
        end
        ∂f_structure = [(i, j) for i = 1:dynamics_dim, j = 1:2traj.dim]
    end

    if verbose
        println("            creating full trajectory jacobian structure...")
    end

    length_∂f_structure = length(∂f_structure)

    ∂F_structure = Vector{Tuple{Int,Int}}(undef, length_∂f_structure * (traj.T - 1))

    Threads.@threads for t = 1:traj.T-1
        ∂fₜ_structure = [
            (
                i + index(t, 0, dynamics_dim),
                j + index(t, 0, traj.dim)
            ) for (i, j) ∈ ∂f_structure
        ]
        ∂F_structure[slice(t, length_∂f_structure)] = ∂fₜ_structure
    end

    return ∂f_structure, ∂F_structure
end

function loss_hessian_structure(∂²l::Function, xdim::Int)
    x = collect(Symbolics.@variables(x[1:xdim])...)
    ∂²l_x = ∂²l(x)
    return structure(sparse(∂²l_x); upper_half=true)
end

end
