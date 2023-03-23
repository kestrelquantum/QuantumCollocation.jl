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
using Zygote



function append_upper_half!(list::Vector, mat::AbstractMatrix)
    for i = 1:size(mat, 1)
        for j = i:size(mat, 2)
            push!(list, mat[i, j])
        end
    end
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
end

function QuantumDynamics(
    f::Function,
    traj::NamedTrajectory
)
    function F(Z⃗::AbstractVector{<:Real})
        r = zeros(traj.dims.states * (traj.T - 1))
        for t = 1:traj.T-1
            zₜ = Z⃗[slice(t, traj.dim)]
            zₜ₊₁ = Z⃗[slice(t + 1, traj.dim)]
            r[slice(t, traj.dims.states)] = f(zₜ, zₜ₊₁)
        end
        return r
    end

    function ∂f(zₜ, zₜ₊₁)
        ∂zₜf, ∂zₜ₊₁f = Zygote.jacobian(f, zₜ, zₜ₊₁)
        return hcat(∂zₜf, ∂zₜ₊₁f)
    end

    function ∂F(Z⃗::AbstractVector{<:Real})
        ∂ = []
        for t = 1:traj.T-1
            zₜ = Z⃗[slice(t, traj.dim)]
            zₜ₊₁ = Z⃗[slice(t + 1, traj.dim)]
            append!(∂, ∂f(zₜ, zₜ₊₁))
        end
        return ∂
    end

    ∂F_structure = Tuple{Int,Int}[]

    for t = 1:traj.T-1
        pairs = product(slice(t, traj.dims.states), slice(t:t+1, traj.dim))
        append!(∂F_structure, pairs)
    end

    function μ∂²f(zₜzₜ₊₁, μₜ)
        return ForwardDiff.hessian(
            zz -> μₜ' * f(zz[1:traj.dim], zz[traj.dim+1:end]),
            zₜzₜ₊₁
        )
    end

    function μ∂²F(μ::AbstractVector, Z⃗::AbstractVector{<:Real})
        μ∂² = []
        for t = 1:traj.T-1
            zₜzₜ₊₁ = Z⃗[slice(t:t+1, traj.dim)]
            μₜ = μ[slice(t, traj.dims.states)]
            append_upper_half!(μ∂², μ∂²f(zₜzₜ₊₁, μₜ))
        end
        return μ∂²
    end

    μ∂²F_structure = Tuple{Int,Int}[]

    for t = 1:traj.T-1
        pairs = collect(product(slice(t:t+1, traj.dim), slice(t:t+1, traj.dim)))
        append_upper_half!(μ∂²F_structure, pairs)
    end

    return QuantumDynamics(F, ∂F, ∂F_structure, μ∂²F, μ∂²F_structure)
end







# function QuantumStateDynamics(
#     P::QuantumStateIntegrator,
#     sys::QuantumSystem,
#     ψ̃_names::Tuple{Vararg{Symbol}},
#     controls::Tuple{Vararg{Symbol}},
#     augs::Union{Nothing, Tuple{Vararg{Symbol}}}=nothing,
#     timestep_name::Union{Nothing, Symbol}=nothing
# )
#     a = isnothing(augs) ? control : augs[1]
#     function f(zₜ::AbstractVector, zₜ₊₁::AbstractVector, Z::NamedTrajectory)
#         r = zeros(Z.dims.states)
#         ψ̃ₜs = [zₜ[Z.components[ψ̃_name]] for ψ̃_name in ψ̃_names]
#         aₜ = zₜ[Z.components[a]]
#         ψ̃ₜ₊₁s = [zₜ₊₁[Z.components[ψ̃_name]] for ψ̃_name in ψ̃_names]
#         if !Z.dynamical_dts
#             Δt = Z.dt
#         else
#             Δt = zₜ[Z.components[timestep_name]]
#         end
#         for (ψ̃ⁱₜ₊₁, ψ̃ⁱₜ, ψ̃ⁱ_name) in zip(ψ̃ₜ₊₁s, ψ̃ₜs, ψ̃_names)
#             r[Z.components[ψ̃ⁱ_name]] = P(ψ̃ⁱₜ₊₁, ψ̃ⁱₜ, aₜ, Δt)
#         end
#         r = vcat([P(ψ̃ₜ₊₁, ψ̃ₜ, aₜ, Δt) for (ψ̃ₜ₊₁, ψ̃ₜ) in zip(ψ̃ₜ₊₁s, ψ̃ₜs)]...)
#         return r
#     end
# end

end
