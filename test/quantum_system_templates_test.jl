"""
Test: Quantum System Templates [RydbergChainSystem, TransmonSystem, QuantumOpticsSystem]
"""

@testitem "Rydberg Chain System" begin
    
end

@testitem "Transmon System" begin

end

@testitem "Quantum Optics System" begin
    using QuantumToolbox
    N = rand(1:5);
    a =  QuantumToolbox.create(N);
    H = a + a';
    sys = QuantumOpticsSystem(H, [H, H]);
    @test typeof(sys) == QuantumSystem
end