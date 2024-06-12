"""
Test: Quantum System Templates [RydbergChainSystem, TransmonSystem, QuantumOpticsSystem]
"""

@testitem "Rydberg Chain System" begin
    
end

@testitem "Transmon System" begin

end

@testitem "Quantum Optics System" begin
    using QuantumOpticsBase
    N = rand(1:5);
    T = ComplexF64;
    b = FockBasis(N);
    a =  QuantumOpticsBase.create(T, b);
    H = a + a';
    sys = QuantumOpticsSystem(H, [H, H]);
    @test typeof(sys) == QuantumSystem
end