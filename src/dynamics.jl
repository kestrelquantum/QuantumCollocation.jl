module Dynamics

export AbstractDynamics
export QuantumDynamics

export dynamics
export dynamics_jacobian
export dynamics_hessian_of_lagrangian
export dynamics_components

using ..QuantumSystems
using ..QuantumUtils
using ..StructureUtils
using ..Integrators

using TrajectoryIndexingUtils
using NamedTrajectories
using LinearAlgebra
using SparseArrays
using ForwardDiff
using Zygote


abstract type AbstractDynamics end

"""
    QuantumDynamics <: AbstractDynamics
"""
struct QuantumDynamics <: AbstractDynamics
    integrators::Union{Nothing, Vector{<:AbstractIntegrator}}
    F::Function
    ∂F::Function
    ∂F_structure::Vector{Tuple{Int, Int}}
    μ∂²F::Function
    μ∂²F_structure::Vector{Tuple{Int, Int}}
    dim::Int
end

function dynamics_components(integrators::Vector{<:AbstractIntegrator})
    dynamics_comps = []
    comp_mark = 0
    for integrator ∈ integrators
        integrator_comps = (comp_mark + 1):(comp_mark + dim(integrator))
        push!(dynamics_comps, integrator_comps)
        comp_mark += dim(integrator)
    end
    return dynamics_comps
end

function dynamics(
    integrators::Vector{<:AbstractIntegrator},
    traj::NamedTrajectory
)
    dynamics_comps = dynamics_components(integrators)
    dynamics_dim = dim(integrators)
    function f(zₜ, zₜ₊₁)
        δ = Vector{eltype(zₜ)}(undef, dynamics_dim)
        for (integrator, integrator_comps) ∈ zip(integrators, dynamics_comps)
            δ[integrator_comps] = integrator(zₜ, zₜ₊₁, traj)
        end
        return δ
    end
    return f
end



function dynamics_jacobian(
    integrators::Vector{<:AbstractIntegrator},
    traj::NamedTrajectory
)
    dynamics_comps = dynamics_components(integrators)
    dynamics_dim = dim(integrators)
    free_time = traj.timestep isa Symbol
    function ∂f(zₜ, zₜ₊₁)
        ∂ = zeros(eltype(zₜ), dynamics_dim, 2traj.dim)
        for (integrator, integrator_comps) ∈ zip(integrators, dynamics_comps)
            if integrator isa QuantumIntegrator
                if integrator.autodiff
                    ∂P(z1, z2) = ForwardDiff.jacobian(
                        zz -> integrator(zz[1:traj.dim], zz[traj.dim+1:end], traj),
                        [z1; z2]
                    )
                    ∂[integrator_comps, 1:2traj.dim] = ∂P(zₜ, zₜ₊₁)
                else
                    if free_time
                        x_comps, u_comps, Δt_comps = comps(integrator, traj)
                        ∂xₜf, ∂xₜ₊₁f, ∂uₜf, ∂Δtₜf =
                            Integrators.jacobian(integrator, zₜ, zₜ₊₁, traj)
                        ∂[integrator_comps, Δt_comps] = ∂Δtₜf
                    else
                        x_comps, u_comps = comps(integrator, traj)
                        ∂xₜf, ∂xₜ₊₁f, ∂uₜf =
                            Integrators.jacobian(integrator, zₜ, zₜ₊₁, traj)
                    end
                    ∂[integrator_comps, x_comps] = ∂xₜf
                    ∂[integrator_comps, x_comps .+ traj.dim] = ∂xₜ₊₁f
                    if u_comps isa Tuple
                        for (uᵢ_comps, ∂uₜᵢf) ∈ zip(u_comps, ∂uₜf)
                            ∂[integrator_comps, uᵢ_comps] = ∂uₜᵢf
                        end
                    else
                        ∂[integrator_comps, u_comps] = ∂uₜf
                    end
                end
            elseif integrator isa DerivativeIntegrator
                if free_time
                    x_comps, dx_comps, Δt_comps = comps(integrator, traj)
                    ∂xₜf, ∂xₜ₊₁f, ∂dxₜf, ∂Δtₜf =
                        Integrators.jacobian(integrator, zₜ, zₜ₊₁, traj)
                else
                    x_comps, dx_comps = comps(integrator, traj)
                    ∂xₜf, ∂xₜ₊₁f, ∂dxₜf =
                        Integrators.jacobian(integrator, zₜ, zₜ₊₁, traj)
                end
                ∂[integrator_comps, x_comps] = ∂xₜf
                ∂[integrator_comps, x_comps .+ traj.dim] = ∂xₜ₊₁f
                ∂[integrator_comps, dx_comps] = ∂dxₜf
                if free_time
                    ∂[integrator_comps, Δt_comps] = ∂Δtₜf
                end
            else
                error("integrator type not supported: $(typeof(integrator))")
            end
        end
        return sparse(∂)
    end
    return ∂f
end

function dynamics_hessian_of_lagrangian(
    integrators::Vector{<:AbstractIntegrator},
    traj::NamedTrajectory
)
    dynamics_comps = dynamics_components(integrators)
    free_time = traj.timestep isa Symbol
    function μ∂²f(zₜ, zₜ₊₁, μₜ)
        μ∂² = zeros(eltype(zₜ), 2traj.dim, 2traj.dim)
        for (integrator, integrator_comps) ∈ zip(integrators, dynamics_comps)
            if integrator isa QuantumIntegrator
                if integrator.autodiff
                    μ∂²P(z1, z2, μ) = ForwardDiff.hessian(
                        zz -> μ' * integrator(zz[1:traj.dim], zz[traj.dim+1:end], traj),
                        [z1; z2]
                    )
                    μ∂²[1:2traj.dim, 1:2traj.dim] = sparse(μ∂²P(zₜ, zₜ₊₁, μₜ[integrator_comps]))
                else
                    if free_time
                        x_comps, u_comps, Δt_comps = comps(integrator, traj)
                        μ∂uₜ∂xₜf, μ∂²uₜf, μ∂Δtₜ∂xₜf, μ∂Δtₜ∂uₜf, μ∂²Δtₜf, μ∂xₜ₊₁∂uₜf, μ∂xₜ₊₁∂Δtₜf =
                            hessian_of_the_lagrangian(integrator, zₜ, zₜ₊₁, μₜ[integrator_comps], traj)
                    else
                        x_comps, u_comps = comps(integrator, traj)
                        μ∂uₜ∂xₜf, μ∂²uₜf, μ∂xₜ₊₁∂uₜf =
                            hessian_of_the_lagrangian(integrator, zₜ, zₜ₊₁, μₜ[integrator_comps], traj)
                    end
                    if u_comps isa Tuple
                        for (uᵢ_comps, μ∂uₜᵢ∂xₜf) ∈ zip(u_comps, μ∂uₜ∂xₜf)
                            μ∂²[x_comps, uᵢ_comps] += μ∂uₜᵢ∂xₜf
                        end
                        for (uᵢ_comps, μ∂²uₜᵢf) ∈ zip(u_comps, μ∂²uₜf)
                            μ∂²[uᵢ_comps, uᵢ_comps] += μ∂²uₜᵢf
                        end
                        if free_time
                            for (uᵢ_comps, μ∂Δtₜ∂uₜᵢf) ∈ zip(u_comps, μ∂Δtₜ∂uₜf)
                                μ∂²[uᵢ_comps, Δt_comps] += μ∂Δtₜ∂uₜᵢf
                            end
                        end
                        for (uᵢ_comps, μ∂xₜ₊₁∂uₜᵢf) ∈ zip(u_comps, μ∂xₜ₊₁∂uₜf)
                            μ∂²[uᵢ_comps, x_comps .+ traj.dim] += μ∂xₜ₊₁∂uₜᵢf
                        end
                    else
                        μ∂²[x_comps, u_comps] += μ∂uₜ∂xₜf
                        μ∂²[u_comps, u_comps] += μ∂²uₜf
                        if free_time
                            μ∂²[u_comps, Δt_comps] += μ∂Δtₜ∂uₜf
                        end
                        μ∂²[u_comps, x_comps .+ traj.dim] += μ∂xₜ₊₁∂uₜf
                    end
                    if free_time
                        μ∂²[x_comps, Δt_comps] += μ∂Δtₜ∂xₜf
                        μ∂²[Δt_comps, x_comps .+ traj.dim] += μ∂xₜ₊₁∂Δtₜf
                        μ∂²[Δt_comps, Δt_comps] .+= μ∂²Δtₜf
                    end
                end
            elseif integrator isa DerivativeIntegrator
                if free_time
                    x_comps, dx_comps, Δt_comps = comps(integrator, traj)
                    μ∂dxₜ∂Δtₜf = -μₜ[integrator_comps]
                    μ∂²[dx_comps, Δt_comps] += μ∂dxₜ∂Δtₜf
                end
            end
        end
        return sparse(μ∂²)
    end
    return μ∂²f
end


function QuantumDynamics(
    integrators::Vector{<:AbstractIntegrator},
    traj::NamedTrajectory;
    verbose=false
)
    if verbose
        println("        constructing knot point dynamics functions...")
    end

    free_time = traj.timestep isa Symbol

    if free_time
        @assert all([
            !isnothing(state(integrator)) &&
            !isnothing(controls(integrator)) &&
            !isnothing(timestep(integrator))
                for integrator ∈ integrators
        ])
    else
        @assert all([
            !isnothing(state(integrator)) &&
            !isnothing(controls(integrator))
                for integrator ∈ integrators
        ])
    end

    for integrator ∈ integrators
        if integrator isa QuantumIntegrator && controls(integrator) isa Tuple
            drive_comps = [traj.components[s] for s ∈ integrator.drive_symb]
            number_of_drives = sum(length.(drive_comps))
            @assert number_of_drives == integrator.n_drives "number of drives ($(number_of_drives)) does not match number of drive terms in Hamiltonian ($(integrator.n_drives))"
        end
    end

    f = dynamics(integrators, traj)
    ∂f = dynamics_jacobian(integrators, traj)
    μ∂²f = dynamics_hessian_of_lagrangian(integrators, traj)

    if verbose
        println("        determining dynamics derivative structure...")
    end

    dynamics_dim = dim(integrators)

    ∂f_structure, ∂F_structure, μ∂²f_structure, μ∂²F_structure =
        dynamics_structure(∂f, μ∂²f, traj, dynamics_dim)

    ∂f_nnz = length(∂f_structure)
    μ∂²f_nnz = length(μ∂²f_structure)

    if verbose
        println("        constructing full dynamics derivative functions...")
    end

    @views function F(Z⃗::AbstractVector{R}) where R <: Real
        δ = Vector{R}(undef, dynamics_dim * (traj.T - 1))
        Threads.@threads for t = 1:traj.T-1
            zₜ = Z⃗[slice(t, traj.dim)]
            zₜ₊₁ = Z⃗[slice(t + 1, traj.dim)]
            δ[slice(t, dynamics_dim)] = f(zₜ, zₜ₊₁)
        end
        return δ
    end

    @views function ∂F(Z⃗::AbstractVector{R}) where R <: Real
        ∂s = zeros(R, length(∂F_structure))
        Threads.@threads for t = 1:traj.T-1
            zₜ = Z⃗[slice(t, traj.dim)]
            zₜ₊₁ = Z⃗[slice(t + 1, traj.dim)]
            ∂fₜ = ∂f(zₜ, zₜ₊₁)
            for (k, (i, j)) ∈ enumerate(∂f_structure)
                ∂s[index(t, k, ∂f_nnz)] = ∂fₜ[i, j]
            end
        end
        return ∂s
    end

    @views function μ∂²F(Z⃗::AbstractVector{<:Real}, μ⃗::AbstractVector{<:Real})
        μ∂²s = Vector{eltype(Z⃗)}(undef, length(μ∂²F_structure))
        Threads.@threads for t = 1:traj.T-1
            zₜ = Z⃗[slice(t, traj.dim)]
            zₜ₊₁ = Z⃗[slice(t + 1, traj.dim)]
            μₜ = μ⃗[slice(t, dynamics_dim)]
            μₜ∂²fₜ = μ∂²f(zₜ, zₜ₊₁, μₜ)
            for (k, (i, j)) ∈ enumerate(μ∂²f_structure)
                μ∂²s[index(t, k, μ∂²f_nnz)] = μₜ∂²fₜ[i, j]
            end
        end
        return μ∂²s
    end

    return QuantumDynamics(
        integrators,
        F,
        ∂F,
        ∂F_structure,
        μ∂²F,
        μ∂²F_structure,
        dynamics_dim
    )
end

QuantumDynamics(P::AbstractIntegrator, traj::NamedTrajectory; kwargs...) =
    QuantumDynamics([P], traj; kwargs...)

function QuantumDynamics(
    f::Function,
    traj::NamedTrajectory;
    verbose=false,
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

    # TODO: benchmark Zygote vs ForwardDiff for jacobian
    # function ∂f(zₜ, zₜ₊₁)
    #     ∂zₜf, ∂zₜ₊₁f = Zygote.jacobian(f, zₜ, zₜ₊₁)
    #     ∂fₜ = hcat(∂zₜf, ∂zₜ₊₁f)
    #     return ∂fₜ
    # end

    @views f̂(zz) = f(zz[1:traj.dim], zz[traj.dim+1:end])

    ∂f̂(zz) = ForwardDiff.jacobian(f̂, zz)

    ∂f_structure, ∂F_structure, μ∂²f_structure, μ∂²F_structure =
        dynamics_structure(∂f̂, traj, dynamics_dim)

    ∂f(zₜ, zₜ₊₁) = ∂f̂([zₜ; zₜ₊₁])

    ∂f_nnz = length(∂f_structure)

    @views function ∂F(Z⃗::AbstractVector{R}) where R <: Real
        ∂ = zeros(R, length(∂F_structure))
        Threads.@threads for t = 1:traj.T-1
            zₜ = Z⃗[slice(t, traj.dim)]
            zₜ₊₁ = Z⃗[slice(t + 1, traj.dim)]
            ∂fₜ = ∂f(zₜ, zₜ₊₁)
            for (k, (i, j)) ∈ enumerate(∂f_structure)
                ∂[index(t, k, ∂f_nnz)] = ∂fₜ[i, j]
            end
        end
        return ∂
    end

    μf̂(zz, μ) = dot(μ, f̂(zz))

    @views function μ∂²f̂(zₜzₜ₊₁, μₜ)
        return ForwardDiff.hessian(zz -> μf̂(zz, μₜ), zₜzₜ₊₁)
    end

    μ∂²f_nnz = length(μ∂²f_structure)

    @views function μ∂²F(Z⃗::AbstractVector{R}, μ::AbstractVector{R}) where R <: Real
        μ∂² = zeros(R, length(μ∂²F_structure))
        Threads.@threads for t = 1:traj.T-1
            zₜzₜ₊₁ = Z⃗[slice(t:t+1, traj.dim)]
            μₜ = μ[slice(t, dynamics_dim)]
            μ∂²fₜ = μ∂²f̂(zₜzₜ₊₁, μₜ)
            for (k, (i, j)) ∈ enumerate(μ∂²f_structure)
                μ∂²[index(t, k, μ∂²f_nnz)] = μ∂²fₜ[i, j]
            end
        end
        return μ∂²
    end

    return QuantumDynamics(nothing, F, ∂F, ∂F_structure, μ∂²F, μ∂²F_structure, dynamics_dim)
end

end
