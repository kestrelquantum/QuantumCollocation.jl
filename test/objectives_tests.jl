"""
    Testing objective struct functionality
"""

@testset "Objectives" begin
    # initializing test trajectory
    T = 10
    ψ̃_dim = 4
    Z = NamedTrajectory(
        (ψ̃ = randn(4, T), u = randn(2, T)),
        controls=:u,
        dt=0.1,
        goal=(ψ̃ = [1, 0, 0, 0],)
    )

    H_drift = GATES[:Z]
    H_drives = [GATES[:X], GATES[:Y]]

    system = QuantumSystem(H_drift, H_drives)

    P = FourthOrderPade(system)

    function f(zₜ, zₜ₊₁)
        ψ̃ₜ₊₁ = zₜ₊₁[Z.components.ψ̃]
        ψ̃ₜ = zₜ[Z.components.ψ̃]
        uₜ = zₜ[Z.components.u]
        δψ̃ = P(ψ̃ₜ₊₁, ψ̃ₜ, uₜ, Z.dt)
        return δψ̃
    end

    dynamics = QuantumDynamics(f, Z)
    evaluator = PicoEvaluator(Z, J, dynamics, true)

    @testset "Quantum State Objective" begin

        loss = :InfidelityLoss
        Q = 100.0

        J = QuantumObjective(:ψ̃, Z, loss, Q)

        L = Z⃗ -> J.L(Z⃗, Z)
        ∇L = Z⃗ -> J.∇L(Z⃗, Z)
        ∂²L = Z⃗ -> J.∂²L(Z⃗, Z)
        ∂²L_structure = J.∂²L_structure(Z)

        # test objective function gradient
        @test ForwardDiff.gradient(L, Z.datavec) ≈ ∇L(Z.datavec)

        # test objective function hessian
        shape = (Z.dim * Z.T, Z.dim * Z.T)
        @test ForwardDiff.hessian(L, Z.datavec) ≈ dense(∂²L(Z.datavec), ∂²L_structure, shape)
    end

    @testset "Unitary Objective" begin
        loss = :UnitaryInfidelityLoss
    end
end
