using Pico
using NamedTrajectories

using Test
using LinearAlgebra
using ForwardDiff

include("test_utils.jl")

@testset "Pico.jl" begin
    include("quantum_systems_tests.jl")
    include("objectives_tests.jl")
end
