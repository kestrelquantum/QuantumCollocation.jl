using Test
using TestItemRunner

using LinearAlgebra
using ForwardDiff
using SparseArrays
using Random; Random.seed!(1234)

using QuantumCollocation
using NamedTrajectories




include("test_utils.jl")

# Run testitem 
@run_package_tests
