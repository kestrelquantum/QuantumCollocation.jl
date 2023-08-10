# Testing dynamics derivatives 

@testset "Dynamics" begin
    # initializing test system
    T = 5
    H_drift = GATES[:Z]
    H_drives = [GATES[:X], GATES[:Y]]
    n_drives = length(H_drives)

    system = QuantumSystem(H_drift, H_drives)

    # @testset "State Dynamics" begin

    #     P = FourthOrderPade(system)

    #     Z = NamedTrajectory(
    #         (
    #             ψ̃ = randn(4, T),
    #             u = randn(2, T),
    #             du = randn(2, T)
    #         ),
    #         controls=:du,
    #         timestep=0.1,
    #         goal=(ψ̃ = [1, 0, 0, 0],)
    #     )

    #     function f(zₜ, zₜ₊₁)
    #         ψ̃ₜ₊₁ = zₜ₊₁[Z.components.ψ̃]
    #         ψ̃ₜ = zₜ[Z.components.ψ̃]
    #         uₜ = zₜ[Z.components.u]
    #         uₜ₊₁ = zₜ₊₁[Z.components.u]
    #         duₜ = zₜ[Z.components.du]
    #         δψ̃ = P(ψ̃ₜ₊₁, ψ̃ₜ, uₜ, Z.timestep)
    #         δu = uₜ₊₁ - uₜ - duₜ * Z.timestep
    #         return vcat(δψ̃, δu)
    #     end

    #     dynamics = QuantumDynamics(f, Z)

    #     # test dynamics jacobian
    #     shape = (Z.dims.states * (Z.T - 1), Z.dim * Z.T)
    #     @test all(ForwardDiff.jacobian(dynamics.F, Z.datavec) .≈
    #         dense(dynamics.∂F(Z.datavec), dynamics.∂F_structure, shape))


    #     # test dynamics hessian of the lagrangian
    #     shape = (Z.dim * Z.T, Z.dim * Z.T)
    #     μ = rand(Z.dims.states * (Z.T - 1))
    #     @test all(ForwardDiff.hessian(Z⃗ -> μ' * dynamics.F(Z⃗), Z.datavec) .≈
    #         dense(dynamics.μ∂²F(Z.datavec, μ), dynamics.μ∂²F_structure, shape))
    # end

    @testset "Unitary dynamics" begin
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

        @testset "UnitaryPadeIntegrator integrator + DerivativeIntegrator on a & da" begin

            P = UnitaryPadeIntegrator(system, :Ũ⃗, :a)
            D = DerivativeIntegrator(:a, :da, Z)

            f = [P, D]

            dynamics = QuantumDynamics(f, Z)

            # test dynamics jacobian
            shape = (Z.dims.states * (Z.T - 1), Z.dim * Z.T)

            J_dynamics = dense(dynamics.∂F(Z.datavec), dynamics.∂F_structure, shape)
            # display(Z.data)
            # println(Z.dim)
            #display(Z.datavec)
            J_forward_diff = ForwardDiff.jacobian(dynamics.F, Z.datavec)
            # display(J_dynamics)
            # display(J_forward_diff)
            @test all(J_forward_diff .≈ J_dynamics)
            show_diffs(J_forward_diff, J_dynamics)

            # test dynamics hessian of the lagrangian
            shape = (Z.dim * Z.T, Z.dim * Z.T)

            μ = ones(Z.dims.states * (Z.T - 1))

            HoL_dynamics = dense(dynamics.μ∂²F(Z.datavec, μ), dynamics.μ∂²F_structure, shape)

            hessian_atol = 1e-15

            HoL_forward_diff = ForwardDiff.hessian(Z⃗ -> dot(μ, dynamics.F(Z⃗)), Z.datavec)
            display(HoL_dynamics)
            display(HoL_forward_diff)
            @test all(isapprox.(HoL_forward_diff, HoL_dynamics; atol=hessian_atol))
            show_diffs(HoL_forward_diff, HoL_dynamics; atol=hessian_atol)
        end

        # @testset "UnitaryPadeIntegrator integrator w/ autodiff + DerivativeIntegrator on a & da" begin

        #     P = UnitaryPadeIntegrator(system, :Ũ⃗, :a, :Δt; order=10, autodiff=true)
        #     D = DerivativeIntegrator(:a, :da, :Δt, n_drives)

        #     f = [P, D]

        #     dynamics = QuantumDynamics(f, Z)

        #     # test dynamics jacobian
        #     shape = (Z.dims.states * (Z.T - 1), Z.dim * Z.T)

        #     J_dynamics = dense(dynamics.∂F(Z.datavec), dynamics.∂F_structure, shape)

        #     J_forward_diff = ForwardDiff.jacobian(dynamics.F, Z.datavec)
        #     @test all(J_forward_diff .≈ J_dynamics)
        #     show_diffs(J_forward_diff, J_dynamics)

        #     # test dynamics hessian of the lagrangian
        #     shape = (Z.dim * Z.T, Z.dim * Z.T)

        #     μ = ones(Z.dims.states * (Z.T - 1))

        #     HoL_dynamics = dense(dynamics.μ∂²F(Z.datavec, μ), dynamics.μ∂²F_structure, shape)

        #     hessian_atol = 1e-15

        #     HoL_forward_diff = ForwardDiff.hessian(Z⃗ -> dot(μ, dynamics.F(Z⃗)), Z.datavec)
        #     @test all(isapprox.(HoL_forward_diff, HoL_dynamics; atol=hessian_atol))
        #     show_diffs(HoL_forward_diff, HoL_dynamics; atol=hessian_atol)
        # end




        # @testset "unitary FourthOrderPade dynamics function" begin
        #     P = FourthOrderPade(system)

        #     function f2(zₜ, zₜ₊₁)
        #         Ũ⃗ₜ₊₁ = zₜ₊₁[Z.components.Ũ⃗]
        #         Ũ⃗ₜ = zₜ[Z.components.Ũ⃗]
        #         aₜ = zₜ[Z.components.a]
        #         Δtₜ = zₜ[Z.components.Δt][1]

        #         δŨ⃗ = P(Ũ⃗ₜ₊₁, Ũ⃗ₜ, aₜ, Δtₜ)
        #         return δŨ⃗
        #     end
        #     dynamics_2 = QuantumDynamics(f2, Z)



        #     # function f()
        #     # end
        # end

    end

end
