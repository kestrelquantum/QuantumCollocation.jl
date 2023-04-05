module Dynamics

export AbstractDynamics
export QuantumDynamics

using ..IndexingUtils
using ..QuantumUtils
using ..QuantumSystems
using ..Integrators

using NamedTrajectories
using Base.Iterators
using LinearAlgebra
using SparseArrays
using Einsum
using ForwardDiff
using Symbolics
using Zygote



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

function jacobian_structure(∂f̂::Function, zdim::Int)
    zz = collect(Symbolics.@variables(zz[1:2zdim])...)
    ∂f = ∂f̂(zz)
    return structure(sparse(∂f))
end

function hessian_of_lagrangian_structure(∂²f̂::Function, zdim::Int, μdim::Int)
    zz = collect(Symbolics.@variables(zz[1:2zdim])...)
    μ = collect(Symbolics.@variables(μ[1:μdim])...)
    ∂²f = ∂²f̂(zz)
    @einsum μ∂²f[j, k] := μ[i] * ∂²f[i, j, k]
    return structure(sparse(μ∂²f), upper_half=true)
end






abstract type AbstractDynamics end

"""
    QuantumDynamics <: AbstractDynamics
"""
struct QuantumDynamics <: AbstractDynamics
    F::Function
    ∂F::Function
    ∂F_structure::Vector{Tuple{Int, Int}}
    μ∂²F::Function
    μ∂²F_structure::Vector{Tuple{Int, Int}}
    dim::Int
end

function QuantumDynamics(
    integrators::Vector{<:AbstractIntegrator},
    traj::NamedTrajectory
)
    @assert all([
        !isnothing(states(integrator)) &&
        !isnothing(controls(integrator)) &&
        !isnothing(timestep(integrator))
            for integrator ∈ integrators
    ])

    function f(zₜ, zₜ₊₁)
        δs = []
        for integrator ∈ integrators
            δ = integrator(zₜ₊₁, zₜ, traj)
            push!(δs, δ)
        end
        return vcat(δs...)
    end

    # TODO: get rid of this redundancy -- will enforce dynamcis dim = states dim
    dynamics_dim = length(f(traj[1].data, traj[2].data))
    dynamics_comps = NamedTuple(
        state(integrator) => traj.components[state(integrator)]
            for integrator ∈ integrators
    )

    @views function F(Z⃗::AbstractVector{<:Real})
        r = zeros(eltype(Z⃗), dynamics_dim * (traj.T - 1))
        Threads.@threads for t = 1:traj.T-1
            zₜ = Z⃗[slice(t, traj.dim)]
            zₜ₊₁ = Z⃗[slice(t + 1, traj.dim)]
            r[slice(t, dynamics_dim)] = f(zₜ, zₜ₊₁)
        end
        return r
    end

    function ∂f(zₜ, zₜ₊₁)
        ∂ = spzeros(dynamics_dim, 2traj.dim)
        for integrator ∈ integrators
            x_comps, u_comps, Δt_comps = comps(integrator, traj)
            ∂x, ∂u, ∂Δt = jacobian(integrator, zₜ, zₜ₊₁, traj)
            ∂[:, x_comps] = ∂x
            if u_comps isa Tuple
                for (uᵢ_comps, ∂uᵢ) ∈ zip(u_comps, ∂u)
                    ∂[:, uᵢ_comps] = ∂uᵢ
                end
            else
                ∂[:, u_comps] = ∂u
            end
            ∂[:, Δt_comps] = ∂Δt
        end
    end

    @views function ∂F(Z⃗::AbstractVector{R}) where R <: Real
        ∂ = zeros(R, (dynamics_dim * 2 * traj.dim) * (traj.T - 1))
        Threads.@threads for t = 1:traj.T-1
            zₜ = Z⃗[slice(t, traj.dim)]
            zₜ₊₁ = Z⃗[slice(t + 1, traj.dim)]
            ∂[slice(t, dynamics_dim * 2 * traj.dim)] = vec(∂f(zₜ, zₜ₊₁))
        end
        return ∂
    end

    function μ∂²f(zₜ, zₜ₊₁)
        μ∂² = spzeros(dynamics_dim, 2traj.dim, 2traj.dim)
        for integrator ∈ integrators
            x_comps, u_comps, Δt_comps = comps(integrator, traj)
            μ∂²x, μ∂²u, μ∂²Δt = hessian_of_the_lagrangian(integrator, zₜ, zₜ₊₁, traj)
            μ∂²[x_comps, x_comps] = μ∂²x
            if u_comps isa Tuple
                for (uᵢ_comps, μ∂²uᵢ) ∈ zip(u_comps, μ∂²u)
                    μ∂²[uᵢ_comps, uᵢ_comps] = μ∂²uᵢ
                end
            else
                μ∂²[u_comps, u_comps] = μ∂²u
            end
            μ∂²[Δt_comps, Δt_comps] = μ∂²Δt
        end
    end
end

function QuantumDynamics(
    f::Function,
    traj::NamedTrajectory
)
    dynamics_dim = length(f(traj[1].data, traj[2].data))

    @views function F(Z⃗::AbstractVector{<:Real})
        r = zeros(eltype(Z⃗), dynamics_dim * (traj.T - 1))
        Threads.@threads for t = 1:traj.T-1
            zₜ = Z⃗[slice(t, traj.dim)]
            zₜ₊₁ = Z⃗[slice(t + 1, traj.dim)]
            r[slice(t, dynamics_dim)] = f(zₜ, zₜ₊₁)
        end
        return r
    end

    # function ∂f(zₜ, zₜ₊₁)
    #     ∂zₜf, ∂zₜ₊₁f = Zygote.jacobian(f, zₜ, zₜ₊₁)
    #     ∂fₜ = hcat(∂zₜf, ∂zₜ₊₁f)
    #     return ∂fₜ
    # end

    @views f̂(zz) = f(zz[1:traj.dim], zz[traj.dim+1:end])
    @views ∂f̂(zz) = ForwardDiff.jacobian(f̂, zz)

    ∂f(zₜ, zₜ₊₁) = ∂f̂([zₜ; zₜ₊₁])

    ∂f_structure = jacobian_structure(∂f̂, traj.dim)
    ∂f_n_nzvals = length(∂f_structure)

    @views function ∂F(Z⃗::AbstractVector{R}) where R <: Real
        ∂ = zeros(R, ∂f_n_nzvals * (traj.T - 1))
        Threads.@threads for t = 1:traj.T-1
            zₜ = Z⃗[slice(t, traj.dim)]
            zₜ₊₁ = Z⃗[slice(t + 1, traj.dim)]
            ∂fₜ = ∂f(zₜ, zₜ₊₁)
            for (k, (i, j)) ∈ enumerate(∂f_structure)
                ∂[index(t, k, ∂f_n_nzvals)] = ∂fₜ[i, j]
            end
        end
        return ∂
    end

    ∂F_structure = Tuple{Int,Int}[]

    for t = 1:traj.T-1
        ∂fₜ_structure = [
            (
                i + index(t, 0, dynamics_dim),
                j + index(t, 0, traj.dim)
            ) for (i, j) ∈ ∂f_structure
        ]
        append!(∂F_structure, ∂fₜ_structure)
    end

    μf̂(zₜzₜ₊₁, μₜ) = dot(μₜ, f̂(zₜzₜ₊₁))

    @views function μ∂²f̂(zₜzₜ₊₁, μₜ)
        return ForwardDiff.hessian(zz -> μf̂(zz, μₜ), zₜzₜ₊₁)
    end

    ∂²f̂(zz) = reshape(
        ForwardDiff.jacobian(x -> vec(∂f̂(x)), zz),
        traj.dims.states,
        2traj.dim,
        2traj.dim
    )

    μ∂²f_structure = hessian_of_lagrangian_structure(∂²f̂, traj.dim, dynamics_dim)
    μ∂²f_n_nzvals = length(μ∂²f_structure)

    @views function μ∂²F(Z⃗::AbstractVector{R}, μ::AbstractVector{R}) where R <: Real
        μ∂² = zeros(R, μ∂²f_n_nzvals * (traj.T - 1))
        Threads.@threads for t = 1:traj.T-1
            zₜzₜ₊₁ = Z⃗[slice(t:t+1, traj.dim)]
            μₜ = μ[slice(t, dynamics_dim)]
            μ∂²fₜ = μ∂²f̂(zₜzₜ₊₁, μₜ)
            for (i, (j, k)) ∈ enumerate(μ∂²f_structure)
                μ∂²[index(t, i, μ∂²f_n_nzvals)] = μ∂²fₜ[j, k]
            end
        end
        return μ∂²
    end

    μ∂²F_structure = Tuple{Int,Int}[]

    for t = 1:traj.T-1
        μ∂²fₜ_structure = [ij .+ index(t, 0, traj.dim) for ij ∈ μ∂²f_structure]
        append!(μ∂²F_structure, μ∂²fₜ_structure)
    end

    return QuantumDynamics(F, ∂F, ∂F_structure, μ∂²F, μ∂²F_structure, dynamics_dim)
end

end
