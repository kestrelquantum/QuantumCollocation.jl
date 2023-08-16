using Revise
using QuantumCollocation
using NamedTrajectories

using Test
using LinearAlgebra
using ForwardDiff
#using FiniteDiff
using SparseArrays
using Random

Random.seed!(1234)

include("test_utils.jl")

@testset "QuantumCollocation.jl" begin
    # include("quantum_systems_tests.jl")
    # include("objectives_tests.jl")
    include("dynamics_tests.jl")
    include("objectives_tests.jl")
end
