"""
This file includes expoential integrators for states and unitaries
"""

using ExponentialAction


abstract type QuantumExponentialIntegrator <: QuantumIntegrator end

struct UnitaryExponentialIntegrator <: QuantumExponentialIntegrator
    G_drift::SparseMatrixCSC{Float64, Int}
    G_drives::Vector{SparseMatrixCSC{Float64, Int}}
    unitary_name::Symbol
    drive_names::Union{Symbol, Tuple{Vararg{Symbol}}}
    n_drives::Int
    ketdim::Int
    dim::Int

    function UnitaryExponentialIntegrator(
        sys::AbstractQuantumSystem,
        unitary_name::Symbol,
        drive_names::Union{Symbol, Tuple{Vararg{Symbol}}}
    )
        n_drives = length(sys.H_drives)
        ketdim = size(sys.H_drift, 1)
        dim = 2ketdim^2

        return new(
            sys.G_drift,
            sys.G_drives,
            unitary_name,
            drive_names,
            n_drives,
            ketdim,
            dim
        )
    end
end

state(integrator::UnitaryExponentialIntegrator) = integrator.unitary_name
controls(integrator::UnitaryExponentialIntegrator) = integrator.drive_names

@views function (ℰ::UnitaryExponentialIntegrator)(
    zₜ::AbstractVector,
    zₜ₊₁::AbstractVector,
    traj::NamedTrajectory
)
    Ũ⃗ₜ₊₁ = zₜ₊₁[traj.components[ℰ.unitary_name]]
    Ũ⃗ₜ = zₜ[traj.components[ℰ.unitary_name]]

    if traj.timestep isa Symbol
        Δtₜ = zₜ[traj.components[traj.timestep]][1]
    else
        Δtₜ = traj.timestep
    end

    if ℰ.drive_names isa Tuple
        aₜ = vcat([zₜ[traj.components[name]] for name ∈ ℰ.drive_names]...)
    else
        aₜ = zₜ[traj.components[ℰ.drive_names]]
    end

    Gₜ = G(aₜ, ℰ.G_drift, ℰ.G_drives)

    return Ũ⃗ₜ₊₁ - expv(Δtₜ, I(ℰ.ketdim) ⊗ Gₜ, Ũ⃗ₜ)
end

function hermitian_exp(G::AbstractMatrix)
    Ĥ = Hermitian(Matrix(QuantumSystems.H(G)))
    λ, V = eigen(Ĥ)
    expG = QuantumSystems.iso(sparse(V * Diagonal(exp.(-im * λ)) * V'))
    droptol!(expG, 1e-12)
    return expG
end

@views function jacobian(
    ℰ::UnitaryExponentialIntegrator,
    zₜ::AbstractVector,
    zₜ₊₁::AbstractVector,
    traj::NamedTrajectory
)
    free_time = traj.timestep isa Symbol

    Ũ⃗ₜ = zₜ[traj.components[ℰ.unitary_name]]

    Δtₜ = free_time ? zₜ[traj.components[traj.timestep]][1] : traj.timestep

    if ℰ.drive_names isa Tuple
        inds = [traj.components[s] for s in ℰ.drive_names]
        inds = vcat(collect.(inds)...)
    else
        inds = traj.components[ℰ.drive_names]
    end

    for i = 1:length(inds) - 1
        @assert inds[i] + 1 == inds[i + 1] "Controls must be in order"
    end

    aₜ = zₜ[inds]

    Gₜ = G(aₜ, ℰ.G_drift, ℰ.G_drives)

    Id = I(ℰ.ketdim)

    expĜₜ = Id ⊗ hermitian_exp(Δtₜ * Gₜ)

    ∂Ũ⃗ₜ₊₁ℰ = sparse(I, ℰ.dim, ℰ.dim)
    ∂Ũ⃗ₜℰ = -expĜₜ

    ∂aₜℰ = -ForwardDiff.jacobian(
        a -> expv(Δtₜ, Id ⊗ G(a, ℰ.G_drift, ℰ.G_drives), Ũ⃗ₜ),
        aₜ
    )

    if free_time
        ∂Δtₜℰ = -(Id ⊗ Gₜ) * expĜₜ * Ũ⃗ₜ
        return ∂Ũ⃗ₜℰ, ∂Ũ⃗ₜ₊₁ℰ, ∂aₜℰ, ∂Δtₜℰ
    else
        return ∂Ũ⃗ₜℰ, ∂Ũ⃗ₜ₊₁ℰ, ∂aₜℰ
    end
end

struct QuantumStateExponentialIntegrator <: QuantumExponentialIntegrator
    G_drift::SparseMatrixCSC{Float64, Int}
    G_drives::Vector{SparseMatrixCSC{Float64, Int}}
    state_name::Symbol
    drive_names::Union{Symbol, Tuple{Vararg{Symbol}}}
    n_drives::Int
    ketdim::Int
    dim::Int

    function QuantumStateExponentialIntegrator(
        sys::AbstractQuantumSystem,
        state_name::Symbol,
        drive_names::Union{Symbol, Tuple{Vararg{Symbol}}}
    )
        n_drives = length(sys.H_drives)
        ketdim = size(sys.H_drift, 1)
        dim = 2ketdim

        return new(
            sys.G_drift,
            sys.G_drives,
            state_name,
            drive_names,
            n_drives,
            ketdim,
            dim
        )
    end
end

state(integrator::QuantumStateExponentialIntegrator) = integrator.state_name
controls(integrator::QuantumStateExponentialIntegrator) = integrator.drive_names

@views function (ℰ::QuantumStateExponentialIntegrator)(
    zₜ::AbstractVector,
    zₜ₊₁::AbstractVector,
    traj::NamedTrajectory
)
    ψ̃ₜ₊₁ = zₜ₊₁[traj.components[ℰ.state_name]]
    ψ̃ₜ = zₜ[traj.components[ℰ.state_name]]

    if traj.timestep isa Symbol
        Δtₜ = zₜ[traj.components[traj.timestep]][1]
    else
        Δtₜ = traj.timestep
    end

    if ℰ.drive_names isa Tuple
        aₜ = vcat([zₜ[traj.components[name]] for name ∈ ℰ.drive_names]...)
    else
        aₜ = zₜ[traj.components[ℰ.drive_names]]
    end

    Gₜ = G(aₜ, ℰ.G_drift, ℰ.G_drives)

    return ψ̃ₜ₊₁ - expv(Δtₜ, Gₜ, ψ̃ₜ)
end

@views function jacobian(
    ℰ::QuantumStateExponentialIntegrator,
    zₜ::AbstractVector,
    zₜ₊₁::AbstractVector,
    traj::NamedTrajectory
)
    free_time = traj.timestep isa Symbol

    ψ̃ₜ = zₜ[traj.components[ℰ.state_name]]

    Δtₜ = free_time ? zₜ[traj.components[traj.timestep]][1] : traj.timestep

    if ℰ.drive_names isa Tuple
        inds = [traj.components[s] for s in ℰ.drive_names]
        inds = vcat(collect.(inds)...)
    else
        inds = traj.components[ℰ.drive_names]
    end

    for i = 1:length(inds) - 1
        @assert inds[i] + 1 == inds[i + 1] "Controls must be in order"
    end

    aₜ = zₜ[inds]

    Gₜ = G(aₜ, ℰ.G_drift, ℰ.G_drives)

    expGₜ = hermitian_exp(Δtₜ * Gₜ)

    ∂ψ̃ₜ₊₁ℰ = sparse(I, ℰ.dim, ℰ.dim)
    ∂ψ̃ₜℰ = -expGₜ

    ∂aₜℰ = -ForwardDiff.jacobian(
        a -> expv(Δtₜ, G(a, ℰ.G_drift, ℰ.G_drives), ψ̃ₜ),
        aₜ
    )

    if free_time
        ∂Δtₜℰ = -Gₜ * expGₜ * ψ̃ₜ
        return ∂ψ̃ₜℰ, ∂ψ̃ₜ₊₁ℰ, ∂aₜℰ, ∂Δtₜℰ
    else
        return ∂ψ̃ₜℰ, ∂ψ̃ₜ₊₁ℰ, ∂aₜℰ
    end
end
