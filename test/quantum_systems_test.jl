"""
Tests: QuantumSystems submodule
"""

@testset "Quantum Systems" begin
    @test MultiModeSystem(2, 14) isa AbstractSystem
    @test MultiModeSystem(3, 14) isa AbstractSystem
    @test MultiModeSystem(4, 14) isa AbstractSystem
end
