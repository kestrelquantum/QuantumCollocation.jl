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



function upper_half_vals(mat::AbstractMatrix)
    n = size(mat, 1)
    vals = similar(mat, n * (n + 1) ÷ 2)
    k = 1
    for col = axes(mat, 2)
        for row = 1:col
            vals[k] = mat[row, col]
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
end

function QuantumDynamics(
    f::Function,
    traj::NamedTrajectory
)
    @views function F(Z⃗::AbstractVector{<:Real})
        r = zeros(eltype(Z⃗), traj.dims.states * (traj.T - 1))
        Threads.@threads for t = 1:traj.T-1
            zₜ = Z⃗[slice(t, traj.dim)]
            zₜ₊₁ = Z⃗[slice(t + 1, traj.dim)]
            r[slice(t, traj.dims.states)] = f(zₜ, zₜ₊₁)
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

    # Symbolics.@variables zz[1:traj.dim*2]
    # zz = collect(zz)
    # ∂f_symbolic = Symbolics.sparsejacobian(f̂(zz), zz; simplify=true)
    # K, J, _ = findnz(∂f_symbolic)
    # ∂f_structure = collect(zip(K, J))
    # ∂f_expression = Symbolics.build_function(∂f_symbolic, zz)
    # ∂f_sparse = eval(∂f_expression[1])

    # function ∂f(zₜ, zₜ₊₁)
    #     ∂ = ∂f_sparse([zₜ; zₜ₊₁])
    #     return [∂[i, j] for (i, j) in ∂f_structure]
    # end

    @views function ∂F(Z⃗::AbstractVector{R}) where R <: Real
        ∂ = zeros(R, (traj.dims.states * 2 * traj.dim) * (traj.T - 1))
        Threads.@threads for t = 1:traj.T-1
            zₜ = Z⃗[slice(t, traj.dim)]
            zₜ₊₁ = Z⃗[slice(t + 1, traj.dim)]
            ∂[slice(t, traj.dims.states * 2 * traj.dim)] = vec(∂f(zₜ, zₜ₊₁))
            # for (k, j) ∈ ∂f_structure
            #     push!(∂, ∂f(zₜ, zₜ₊₁)[k, j])
            # end
        end
        return ∂
    end

    ∂F_structure = Tuple{Int,Int}[]

    for t = 1:traj.T-1
        # ∂fₜ_structure = [
        #     (
        #         k + index(t, 0, traj.dims.states),
        #         j + index(t, 0, traj.dim)
        #     ) for (k, j) ∈ ∂f_structure
        # ]
        # append!(∂F_structure, ∂fₜ_structure)
        ∂fₜ_structure = Iterators.product(slice(t, traj.dims.states), slice(t:t+1, traj.dim))
        append!(∂F_structure, collect(∂fₜ_structure))
    end

    μf̂(zₜzₜ₊₁, μₜ) = dot(μₜ, f̂(zₜzₜ₊₁))

    @views function μ∂²f(zₜzₜ₊₁, μₜ)
        return ForwardDiff.hessian(zz -> μf̂(zz, μₜ), zₜzₜ₊₁)
    end


    # Symbolics.@variables μ[1:traj.dims.states]
    # μ = collect(μ)

    # μ∂²f_symbolic = Symbolics.sparsehessian(μf̂(zz, μ), zz)

    # K, J, _ = findnz(μ∂²f_symbolic)
    # μ∂²f_structure = collect(zip(K, J))
    # filter!(((k, j),) -> k ≤ j, μ∂²f_structure)
    # μ∂²f_expression = Symbolics.build_function(μ∂²f_symbolic, zz, μ)
    # μ∂²f = eval(μ∂²f_expression[1])


    @views function μ∂²F(Z⃗::AbstractVector{R}, μ::AbstractVector{R}) where R <: Real
        block_upper_half_dim = 2traj.dim * (2traj.dim + 1) ÷ 2
        μ∂² = zeros(R, block_upper_half_dim * (traj.T - 1))
        Threads.@threads for t = 1:traj.T-1
            zₜzₜ₊₁ = Z⃗[slice(t:t+1, traj.dim)]
            μₜ = μ[slice(t, traj.dims.states)]
            HoL = μ∂²f(zₜzₜ₊₁, μₜ)
            vals = upper_half_vals(HoL)
            μ∂²[slice(t, block_upper_half_dim)] .= vals
        end
        return μ∂²
    end

    μ∂²F_structure = Tuple{Int,Int}[]

    for t = 1:traj.T-1
        # μₜ∂²fₜ_structure = [index(t, 0, traj.dim) .+ kj for kj in μ∂²f_structure]
        # append!(μ∂²F_structure, μₜ∂²fₜ_structure)
        μₜ∂²fₜ_structure = Iterators.product(slice(t:t+1, traj.dim), slice(t:t+1, traj.dim))
        vals = upper_half_vals(collect(μₜ∂²fₜ_structure))
        append!(μ∂²F_structure, vals)
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
