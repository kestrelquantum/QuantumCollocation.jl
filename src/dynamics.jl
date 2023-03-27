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

    function ∂f(zₜ, zₜ₊₁)
        return ForwardDiff.jacobian(f̂, [zₜ; zₜ₊₁])
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

    ∂F_structure = Tuple{Int,Int}[]

    for t = 1:traj.T-1
        ∂fₜ_structure = Iterators.product(slice(t, dynamics_dim), slice(t:t+1, traj.dim))
        append!(∂F_structure, collect(∂fₜ_structure))
    end

    μf̂(zₜzₜ₊₁, μₜ) = dot(μₜ, f̂(zₜzₜ₊₁))

    @views function μ∂²f(zₜzₜ₊₁, μₜ)
        return ForwardDiff.hessian(zz -> μf̂(zz, μₜ), zₜzₜ₊₁)
    end

    @views function μ∂²F(Z⃗::AbstractVector{R}, μ::AbstractVector{R}) where R <: Real
        block_upper_half_dim = 2traj.dim * (2traj.dim + 1) ÷ 2
        μ∂² = zeros(R, block_upper_half_dim * (traj.T - 1))
        Threads.@threads for t = 1:traj.T-1
            zₜzₜ₊₁ = Z⃗[slice(t:t+1, traj.dim)]
            μₜ = μ[slice(t, dynamics_dim)]
            HoL = μ∂²f(zₜzₜ₊₁, μₜ)
            vals = upper_half_vals(HoL)
            μ∂²[slice(t, block_upper_half_dim)] .= vals
        end
        return μ∂²
    end

    μ∂²F_structure = Tuple{Int,Int}[]

    for t = 1:traj.T-1
        μₜ∂²fₜ_structure = Iterators.product(slice(t:t+1, traj.dim), slice(t:t+1, traj.dim))
        vals = upper_half_vals(collect(μₜ∂²fₜ_structure))
        append!(μ∂²F_structure, vals)
    end

    return QuantumDynamics(F, ∂F, ∂F_structure, μ∂²F, μ∂²F_structure, dynamics_dim)
end

end
