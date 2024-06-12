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
    include("integrators_test.jl")
    include("quantum_utils_test.jl")
    include("quantum_system_templates_test.jl")
end

@run_package_tests
