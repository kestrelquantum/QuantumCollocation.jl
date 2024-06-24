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
