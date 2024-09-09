# ```@meta
# CollapsedDocStrings = true
# ```
# # Embedded Operators

# In this manual, we will discuss embedding operators in subspaces of larger quantum systems.

# ## The `embed` and `unembed` functions

# A frequent situation in quantum optimal control is the need to embed a quantum operator in a larger Hilbert space. This is often necessary when the control Hamiltonian acts on a subspace of the full Hilbert space.

# The [`embed`](@ref) function allows to embed a quantum operator in a larger Hilbert space.
# ```@docs
# embed
# ```

# The [`unembed`](@ref) function allows to unembed a quantum operator from a larger Hilbert space.
# ```@docs
# unembed
# ```

# For example, for a single qubit X gate embedded in a multilevel system:

using QuantumCollocation
using SparseArrays # for visualization

## define levels of full system
levels = 3

## get a 2-level X gate
X = GATES[:X]

## define subspace indices as lowest two levels
subspace_indices = 1:2

## embed the X gate in the full system
X_embedded = embed(X, subspace_indices, levels)

# We can retrieve the original operator:

X_unembedded = unembed(X_embedded, subspace_indices)


# ## The `EmbeddedOperator` type

# The `EmbeddedOperator` type is a convenient way to define an operator embedded in a subspace of a larger quantum system.

# The `EmbeddedOperator` type is defined as follows:

# ```@docs
# EmbeddedOperator
# ```

# And can be constructed using the following method:

# ```@docs
# EmbeddedOperator(op::Matrix{<:Number}, subspace_indices::AbstractVector{Int}, subsystem_levels::AbstractVector{Int})
# ```

# For example, for an X gate on the first qubit of two qubit, 3-level system:

## define the target operator X ⊗ I
gate = GATES[:X] ⊗ GATES[:I]

## define the subsystem levels
subsystem_levels = [3, 3]

## define the subspace indices
subspace_indices = get_subspace_indices([1:2, 1:2], subsystem_levels)

## create the embedded operator
op = EmbeddedOperator(gate, subspace_indices, subsystem_levels)

## show the full operator
op.operator .|> abs |> sparse

# We can get the original operator back:

gate_unembeded = unembed(op)

gate_unembeded .|> abs |> sparse

# ## The `get_subspace_indices` function

# The `get_subspace_indices` function is a convenient way to get the indices of a subspace in a larger quantum system.

# ### Simple quantum systems
# For simple (non-composite) quantum systems, such as a single multilevel qubit, we provode the following method:

# ```@docs
# get_subspace_indices(subspace::AbstractVector{Int}, levels::Int)
# ```

## get the indices of the lowest two levels of a 3-level system
subspace_indices = get_subspace_indices(1:2, 3)

# ### Comosite quantum systems
# For composite quantum systems, such as a two qubit system, we provide the following methods.

# Targeting subspaces in a composite quantum system, with general subsystem levels:
# ```@docs
# get_subspace_indices(subspaces::Vector{<:AbstractVector{Int}}, subsystem_levels::AbstractVector{Int})
# ```

## get the subspace indices for a three level qubit coupled to a 9-level cavity
get_subspace_indices([1:2, 1:2], [3, 9])


# Targeting subspaces in a composite quantum system, with all subsystems having the same number of levels:
# ```@docs
# get_subspace_indices(levels::AbstractVector{Int}; subspace=1:2, kwargs...)
# ```

## get the subspace indices for a two qubit system with 3 levels each
get_subspace_indices([3, 3])
