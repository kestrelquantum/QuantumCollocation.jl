"""
    Testing objective struct functionality
"""

 @testset "Objectives" begin

    # initializing test trajectory
    T = 10
    # H_drift = GATES[:Z]
    # H_drives = [GATES[:X], GATES[:Y]]

    # system = QuantumSystem(H_drift, H_drives)

    # P = FourthOrderPade(system)

    # function f(zₜ, zₜ₊₁)
    #     ψ̃ₜ₊₁ = zₜ₊₁[Z.components.ψ̃]
    #     ψ̃ₜ = zₜ[Z.components.ψ̃]
    #     uₜ = zₜ[Z.components.u]
    #     δψ̃ = P(ψ̃ₜ₊₁, ψ̃ₜ, uₜ, Z.timestep)
    #     return δψ̃
    # end

    # dynamics = QuantumDynamics(f, Z)
    # evaluator = PicoEvaluator(Z, J, dynamics, true)

#     @testset "Quantum State Objective" begin

#         Z = NamedTrajectory(
#             (ψ̃ = randn(4, T), u = randn(2, T)),
#             controls=:u,
#             dt=0.1,
#             goal=(ψ̃ = [1, 0, 0, 0],)
#         )



#         loss = :InfidelityLoss
#         Q = 100.0

#         J = QuantumObjective(:ψ̃, Z, loss, Q)

#         L = Z⃗ -> J.L(Z⃗, Z)
#         ∇L = Z⃗ -> J.∇L(Z⃗, Z)
#         ∂²L = Z⃗ -> J.∂²L(Z⃗, Z)
#         ∂²L_structure = J.∂²L_structure(Z)

#         # test objective function gradient
#         @test ForwardDiff.gradient(L, Z.datavec) ≈ ∇L(Z.datavec)

#         # test objective function hessian
#         shape = (Z.dim * Z.T, Z.dim * Z.T)
#         @test ForwardDiff.hessian(L, Z.datavec) ≈ dense(∂²L(Z.datavec), ∂²L_structure, shape)
#     end

    @testset "Quadratic Regularizer Objective" begin

        Z = NamedTrajectory(
            (ψ̃ = randn(4, T), u = randn(2, T)),
            controls=:u,
            timestep=0.1,
            goal=(ψ̃ = [1.0, 0.0, 0.0, 0.0],)
        )


        J = QuadraticRegularizer(:u, Z, [1., 1.])

        L = Z⃗ -> J.L(Z⃗, Z)
        ∇L = Z⃗ -> J.∇L(Z⃗, Z)
        ∂²L = Z⃗ -> J.∂²L(Z⃗, Z)
        ∂²L_structure = J.∂²L_structure(Z)

        # test objective function gradient

        @test all(ForwardDiff.gradient(L, Z.datavec) .≈ ∇L(Z.datavec))

        # test objective function hessian
        shape = (Z.dim * Z.T, Z.dim * Z.T)
        @test all(isapprox(
            ForwardDiff.hessian(L, Z.datavec),
            dense(∂²L(Z.datavec), ∂²L_structure, shape);
            atol=1e-7
        ))
    end

    @testset "Quadratic Smoothness Regularizer Objective" begin

        Z = NamedTrajectory(
            (ψ̃ = randn(4, T), u = randn(2, T)),
            controls=:u,
            timestep=0.1,
            goal=(ψ̃ = [1.0, 0.0, 0.0, 0.0],)
        )


        J = QuadraticSmoothnessRegularizer(:u, Z, [1., 1.])

        L = Z⃗ -> J.L(Z⃗, Z)
        ∇L = Z⃗ -> J.∇L(Z⃗, Z)
        ∂²L = Z⃗ -> J.∂²L(Z⃗, Z)
        ∂²L_structure = J.∂²L_structure(Z)

        # test objective function gradient

        @test all(ForwardDiff.gradient(L, Z.datavec) .≈ ∇L(Z.datavec))

        # test objective function hessian
        shape = (Z.dim * Z.T, Z.dim * Z.T)
        @test all(isapprox(
            ForwardDiff.hessian(L, Z.datavec),
            dense(∂²L(Z.datavec), ∂²L_structure, shape);
            atol=1e-7
        ))
    end

    @testset "Unitary Objective" begin

        U_init = GATES[:I]
        U_goal = GATES[:X]

        Ũ⃗_init = operator_to_iso_vec(U_init)
        Ũ⃗_goal = operator_to_iso_vec(U_goal)

        Z = NamedTrajectory(
            (Ũ⃗ = randn(length(Ũ⃗_init), T), u = randn(2, T)),
            controls=:u,
            timestep=0.1,
            initial=(Ũ⃗ = Ũ⃗_init,),
            goal=(Ũ⃗ = Ũ⃗_goal,)
        )

        loss = :UnitaryInfidelityLoss
        Q = 100.0

        J = QuantumObjective(:Ũ⃗, Z, loss, Q)

        L = Z⃗ -> J.L(Z⃗, Z)
        ∇L = Z⃗ -> J.∇L(Z⃗, Z)
        ∂²L = Z⃗ -> J.∂²L(Z⃗, Z)
        ∂²L_structure = J.∂²L_structure(Z)

        # test objective function gradient
        @test all(ForwardDiff.gradient(L, Z.datavec) ≈ ∇L(Z.datavec))

        # test objective function hessian
        shape = (Z.dim * Z.T, Z.dim * Z.T)
        H = dense(∂²L(Z.datavec), ∂²L_structure, shape)
        H_forwarddiff = ForwardDiff.hessian(L, Z.datavec)
        @test all(H .≈ H_forwarddiff)
    end
end
