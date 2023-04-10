using Revise
using QuantumCollocation
using NamedTrajectories

using Test
using LinearAlgebra
using ForwardDiff

include("test_utils.jl")

@testset "QuantumCollocation.jl" begin
    #include("quantum_systems_tests.jl")
    include("objectives_tests.jl")
    #include("dynamics_tests.jl")
end
