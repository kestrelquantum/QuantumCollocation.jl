using Test
using TestItemRunner

using LinearAlgebra
using ForwardDiff
using SparseArrays
using Random; Random.seed!(1234)

using QuantumCollocation
using NamedTrajectories




include("test_utils.jl")

@testset "QuantumCollocation.jl" begin
    # include("objectives_test.jl")
    # include("dynamics_test.jl")
    include("problem_templates_test.jl")
end

@run_package_tests
