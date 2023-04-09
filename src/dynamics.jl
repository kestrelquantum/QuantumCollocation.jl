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

function dynamics_structure(∂f̂::Function, traj::NamedTrajectory, dynamics_dim::Int)
    ∂²f̂(zz) = reshape(
        ForwardDiff.jacobian(x -> vec(∂f̂(x)), zz),
        traj.dims.states,
        2traj.dim,
        2traj.dim
    )

    ∂f_structure = jacobian_structure(∂f̂, traj.dim)

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

    μ∂²f_structure = hessian_of_lagrangian_structure(∂²f̂, traj.dim, dynamics_dim)

    μ∂²F_structure = Tuple{Int,Int}[]

    for t = 1:traj.T-1
        μ∂²fₜ_structure = [ij .+ index(t, 0, traj.dim) for ij ∈ μ∂²f_structure]
        append!(μ∂²F_structure, μ∂²fₜ_structure)
    end

    return ∂f_structure, ∂F_structure, μ∂²f_structure, μ∂²F_structure
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
        !isnothing(state(integrator)) &&
        !isnothing(controls(integrator)) &&
        !isnothing(timestep(integrator))
            for integrator ∈ integrators
    ])

    for integrator ∈ integrators
        if integrator isa QuantumIntegrator && controls(integrator) isa Tuple
            drive_comps = [traj.components[s] for s ∈ integrator.drive_symb]
            number_of_drives = sum(length.(drive_comps))
            @assert number_of_drives == integrator.n_drives "number of drives ($(number_of_drives)) does not match number of drive terms in Hamiltonian ($(integrator.n_drives))"
        end
    end

    dynamics_comps = []
    let comp_mark = 0
        for integrator ∈ integrators
            integrator_comps = (comp_mark + 1):(comp_mark + dim(integrator))
            push!(dynamics_comps, integrator_comps)
            comp_mark += dim(integrator)
        end
    end

    dynamics_dim = dim(integrators)

    function f(zₜ, zₜ₊₁)
        δ = Vector{eltype(zₜ)}(undef, dynamics_dim)
        for (integrator, integrator_comps) ∈ zip(integrators, dynamics_comps)
            δ[integrator_comps] = integrator(zₜ₊₁, zₜ, traj)
        end
        return δ
    end

    @views function F(Z⃗::AbstractVector{<:Real})
        δ = zeros(dynamics_dim * (traj.T - 1))
        for t = 1:traj.T-1
            zₜ = Z⃗[slice(t, traj.dim)]
            zₜ₊₁ = Z⃗[slice(t+1, traj.dim)]
            δ[slice(t, dynamics_dim)] = f(zₜ, zₜ₊₁)
        end
    end


    ∂f̂(zₜzₜ₊₁) = ForwardDiff.jacobian(zz -> f(zz[1:traj.dim], zz[traj.dim+1:end]), zₜzₜ₊₁)

    ∂f_structure, ∂F_structure, μ∂²f_structure, μ∂²F_structure =
        dynamics_structure(∂f̂, traj, dynamics_dim)

    function ∂f(zₜ, zₜ₊₁)
        ∂ = spzeros(dynamics_dim, 2traj.dim)
        for (integrator, integrator_comps) ∈ zip(integrators, integrator_comps)
            if integrator isa QuantumIntegrator
                x_comps, u_comps, Δt_comps = comps(integrator, traj)
                ∂xₜf, ∂xₜ₊₁f, ∂uₜf, ∂Δtₜf = jacobian(integrator, zₜ, zₜ₊₁, traj)
                ∂[integrator_comps, x_comps] = ∂xₜf
                ∂[integrator_comps, x_comps .+ traj.dim] = ∂xₜ₊₁f
                if u_comps isa Tuple
                    for (uᵢ_comps, ∂uᵢf) ∈ zip(u_comps, ∂u)
                        ∂[integrator_comps, uᵢ_comps] = ∂uᵢf
                    end
                else
                    ∂[integrator_comps, u_comps] = ∂uₜf
                end
                ∂[integrator_comps, Δt_comps] = ∂Δtₜf
            elseif integrator isa DerivativeIntegrator
                x_comps, dx_comps, Δt_comps = comps(integrator, traj)
                ∂xₜf, ∂xₜ₊₁f, ∂dxₜf, ∂Δtₜf = jacobian(integrator, zₜ, zₜ₊₁, traj)
                ∂[integrator_comps, x_comps] = ∂xₜf
                ∂[integrator_comps, x_comps .+ traj.dim] = ∂xₜ₊₁f
                ∂[integrator_comps, dx_comps] = ∂dxₜf
                ∂[integrator_comps, Δt_comps] = ∂Δtₜf
            else
                error("integrator type not supported: $(typeof(integrator))")
            end
        end
    end

    ∂f_nnz = length(∂f_structure)

    @views function ∂F(Z⃗::AbstractVector{<:Real})
        ∂s = zeros(eltype(Z⃗), length(∂F_structure))
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

    function μ∂²f(zₜ, zₜ₊₁, μₜ)
        μ∂² = spzeros(dynamics_dim, 2traj.dim, 2traj.dim)
        for (integrator, integrator_comps) ∈ zip(integrators, dynamics_comps)
            x_comps, u_comps, Δt_comps = comps(integrator, traj)
            μ∂uₜ∂xₜf, μ∂²uₜf, μ∂Δtₜ∂xₜf, μ∂Δtₜ∂uₜf, μ∂²Δtₜf, μ∂uₜ∂xₜ₊₁f, μ∂Δtₜ∂xₜ₊₁f =
                hessian_of_the_lagrangian(integrator, zₜ, zₜ₊₁, μₜ[integrator_comps], traj)
            if u_comps isa Tuple
                for (uᵢ_comps, μ∂uₜᵢ∂xₜf) ∈ zip(u_comps, μ∂uₜ∂xₜf)
                    μ∂²[x_comps, uᵢ_comps] += μ∂uₜᵢ∂xₜf
                end
                for (uᵢ_comps, μ∂²uₜᵢf) ∈ zip(u_comps, μ∂²uₜf)
                    μ∂²[uᵢ_comps, uᵢ_comps] += μ∂²uₜᵢf
                end
                for (uᵢ_comps, μ∂Δtₜ∂uₜᵢf) ∈ zip(u_comps, μ∂Δtₜ∂uₜf)
                    μ∂²[uᵢ_comps, Δt_comps] += μ∂Δtₜ∂uₜᵢf
                end
                for (uᵢ_comps, μ∂uₜᵢ∂xₜ₊₁f) ∈ zip(u_comps, μ∂uₜ∂xₜ₊₁f)
                    μ∂²[x_comps, uᵢ_comps .+ traj.dim] += μ∂uₜᵢ∂xₜ₊₁f
                end
            else
                μ∂²[x_comps, u_comps] += μ∂uₜ∂xₜf
                μ∂²[u_comps, u_comps] += μ∂²uₜf
                μ∂²[u_comps, Δt_comps] += μ∂Δtₜ∂uₜf
                μ∂²[x_comps, u_comps .+ traj.dim] += μ∂uₜ∂xₜ₊₁f
            end
            μ∂²[x_comps, Δt_comps] += μ∂Δtₜ∂xₜf
            μ∂²[x_comps, Δt_comps .+ traj.dim] += μ∂Δtₜ∂xₜ₊₁f
            μ∂²[Δt_comps, Δt_comps] += μ∂²Δtₜf
        end
        return μ∂²
    end

    μ∂²f_nnz = length(μ∂²f_structure)

    @views function μ∂²F(Z⃗::AbstractVector{<:Real}, μ⃗::AbstractVector{<:Real})
        μ∂²s = zeros(eltype(Z⃗), length(μ∂²F_structure))
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

    return QuantumDynamics(F, ∂F, ∂F_structure, μ∂²F, μ∂²F_structure, dyanmics_dim)
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

    return QuantumDynamics(F, ∂F, ∂F_structure, μ∂²F, μ∂²F_structure, dynamics_dim)
end

end
