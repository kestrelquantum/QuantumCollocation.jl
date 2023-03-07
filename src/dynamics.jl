module Dynamics

export AbstractDynamics
export QuantumDynamics

using ..IndexingUtils
using ..QuantumUtils
using ..QuantumSystems
using ..Integrators

using NamedTrajectories
using LinearAlgebra
using SparseArrays
using IterTools
using ForwardDiff
using Zygote

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
    function F(Z::NamedTrajectory)
        r = zeros(Z.dims.states * Z.T)
        for t = 1:Z.T-1
            r[slice(t, Z.dims.states)] = f(Z[t], Z[t + 1])
        end
    end


    function ∂F(Z::NamedTrajectory)

        ∂ = []

        function ∂f(zₜ, zₜ₊₁)
            ∂zₜf, ∂zₜ₊₁f = Zygote.jacobian(f, zₜ, zₜ₊₁)
            return hcat(∂zₜf, ∂zₜ₊₁f)
        end

        for t = 1:Z.T-1
            zₜ = Z.data[:, t]
            zₜ₊₁ = Z.data[:, t + 1]
            append!(∂, ∂f(zₜ, zₜ₊₁))
        end
       return ∂
    end

    ∂F_structure = []

    for t = 1:traj.T-1
        for pair ∈ product(slice(t, traj.dims.states), slice(t:t+1, traj.dim))
            append!(∂F_structure, pair)
        end
    end



    function μ∂²F(Z::NamedTrajectory, μ::AbstractVector)
        μ∂² = []

        function μ∂²f(zₜzₜ₊₁, μₜ)
            return ForwardDiff.hessian(
                zz -> μₜ' * f(zz[1:Z.dim], zz[Z.dim+1:end]),
                zₜzₜ₊₁
            )
        end

        for t = 1:Z.T-1
            zₜzₜ₊₁ = vec(Z.data[:, t:t+1])
            μₜ = μ[slice(t, Z.dims.states)]
            append!(μ∂², μ∂²f(zₜzₜ₊₁, μₜ))
        end
        return μ∂², structure
    end

    μ∂²F_structure = []

    for t = 1:traj.T-1
        for pair ∈ product(slice(t, traj.dim), slice(t, traj.dim))
            append!(μ∂²F_structure, pair)
        end
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
