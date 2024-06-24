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
    G::Function

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
            dim,
            G_bilinear
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

    Gₜ = ℰ.G(aₜ, ℰ.G_drift, ℰ.G_drives)

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

    Gₜ = ℰ.G(aₜ, ℰ.G_drift, ℰ.G_drives)

    Id = I(ℰ.ketdim)

    expĜₜ = Id ⊗ hermitian_exp(Δtₜ * Gₜ)

    ∂Ũ⃗ₜ₊₁ℰ = sparse(I, ℰ.dim, ℰ.dim)
    ∂Ũ⃗ₜℰ = -expĜₜ

    ∂aₜℰ = -ForwardDiff.jacobian(
        a -> expv(Δtₜ, Id ⊗ ℰ.G(a, ℰ.G_drift, ℰ.G_drives), Ũ⃗ₜ),
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
    G::Function

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
            dim,
            G_bilinear
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

    Gₜ = ℰ.G(aₜ, ℰ.G_drift, ℰ.G_drives)

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

    Gₜ = ℰ.G(aₜ, ℰ.G_drift, ℰ.G_drives)

    expGₜ = hermitian_exp(Δtₜ * Gₜ)

    ∂ψ̃ₜ₊₁ℰ = sparse(I, ℰ.dim, ℰ.dim)
    ∂ψ̃ₜℰ = -expGₜ

    ∂aₜℰ = -ForwardDiff.jacobian(
        a -> expv(Δtₜ, ℰ.G(a, ℰ.G_drift, ℰ.G_drives), ψ̃ₜ),
        aₜ
    )

    if free_time
        ∂Δtₜℰ = -Gₜ * expGₜ * ψ̃ₜ
        return ∂ψ̃ₜℰ, ∂ψ̃ₜ₊₁ℰ, ∂aₜℰ, ∂Δtₜℰ
    else
        return ∂ψ̃ₜℰ, ∂ψ̃ₜ₊₁ℰ, ∂aₜℰ
    end
end

# ******************************************************************************* #

@testitem "testing UnitaryExponentialIntegrator" begin
    using NamedTrajectories
    using ForwardDiff

    T = 100
    H_drift = GATES[:Z]
    H_drives = [GATES[:X], GATES[:Y]]
    n_drives = length(H_drives)

    system = QuantumSystem(H_drift, H_drives)

    U_init = GATES[:I]
    U_goal = GATES[:X]

    Ũ⃗_init = operator_to_iso_vec(U_init)
    Ũ⃗_goal = operator_to_iso_vec(U_goal)

    dt = 0.1

    Z = NamedTrajectory(
        (
            Ũ⃗ = unitary_geodesic(U_goal, T),
            a = randn(n_drives, T),
            da = randn(n_drives, T),
            Δt = fill(dt, 1, T),
        ),
        controls=(:da,),
        timestep=:Δt,
        goal=(Ũ⃗ = Ũ⃗_goal,)
    )

    ℰ = UnitaryExponentialIntegrator(system, :Ũ⃗, :a)


    ∂Ũ⃗ₜℰ, ∂Ũ⃗ₜ₊₁ℰ, ∂aₜℰ, ∂Δtₜℰ = jacobian(ℰ, Z[1].data, Z[2].data, Z)

    ∂ℰ_forwarddiff = ForwardDiff.jacobian(
        zz -> ℰ(zz[1:Z.dim], zz[Z.dim+1:end], Z),
        [Z[1].data; Z[2].data]
    )

    @test ∂Ũ⃗ₜℰ ≈ ∂ℰ_forwarddiff[:, 1:ℰ.dim]
    @test ∂Ũ⃗ₜ₊₁ℰ ≈ ∂ℰ_forwarddiff[:, Z.dim .+ (1:ℰ.dim)]
    @test ∂aₜℰ ≈ ∂ℰ_forwarddiff[:, Z.components.a]
    @test ∂Δtₜℰ ≈ ∂ℰ_forwarddiff[:, Z.components.Δt]
end

@testitem "testing QuantumStateExponentialIntegrator" begin
    using NamedTrajectories
    using ForwardDiff

    T = 100
    H_drift = GATES[:Z]
    H_drives = [GATES[:X], GATES[:Y]]
    n_drives = length(H_drives)

    system = QuantumSystem(H_drift, H_drives)

    U_init = GATES[:I]
    U_goal = GATES[:X]

    ψ̃_init = ket_to_iso(quantum_state("g", [2]))
    ψ̃_goal = ket_to_iso(quantum_state("e", [2]))

    dt = 0.1

    Z = NamedTrajectory(
        (
            ψ̃ = linear_interpolation(ψ̃_init, ψ̃_goal, T),
            a = randn(n_drives, T),
            da = randn(n_drives, T),
            Δt = fill(dt, 1, T),
        ),
        controls=(:da,),
        timestep=:Δt,
        goal=(ψ̃ = ψ̃_goal,)
    )

    ℰ = QuantumStateExponentialIntegrator(system, :ψ̃, :a)

    ∂ψ̃ₜℰ, ∂ψ̃ₜ₊₁ℰ, ∂aₜℰ, ∂Δtₜℰ = jacobian(ℰ, Z[1].data, Z[2].data, Z)

    ∂ℰ_forwarddiff = ForwardDiff.jacobian(
        zz -> ℰ(zz[1:Z.dim], zz[Z.dim+1:end], Z),
        [Z[1].data; Z[2].data]
    )

    @test ∂ψ̃ₜℰ ≈ ∂ℰ_forwarddiff[:, 1:ℰ.dim]
    @test ∂ψ̃ₜ₊₁ℰ ≈ ∂ℰ_forwarddiff[:, Z.dim .+ (1:ℰ.dim)]
    @test ∂aₜℰ ≈ ∂ℰ_forwarddiff[:, Z.components.a]
    @test ∂Δtₜℰ ≈ ∂ℰ_forwarddiff[:, Z.components.Δt]
end
