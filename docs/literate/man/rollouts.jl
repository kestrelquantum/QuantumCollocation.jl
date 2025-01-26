# ```@meta
# CollapsedDocStrings = true
# ```

# # Rollouts

using QuantumCollocation
using SparseArrays # for visualization

#=

Rollouts are a way to visualize the evolution of a quantum system. The various rollout 
functions provided in this module allow for the validation of the solution to a quantum
optimal control problem. 

=#