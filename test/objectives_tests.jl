"""
    Testing objective struct functionality
"""

@testset "Objectives" begin
    # initializing test trajectory
    T = 10
    ψ̃_dim = 4
    Ψ = NamedTrajectory(
        (ψ̃ = randn(4, T), u = randn(2, T)),
        controls=:u,
        dt=0.1,
        goal=(ψ̃ = [1, 0, 0, 0])
    )
    @testset "Quantum Objective" begin
        J = QuantumObjective(:ψ̃, Ψ)
    end
end
