```@meta
EditURL = "../../../literate/man/embedded_operators.jl"
```

```@meta
CollapsedDocStrings = true
```
# Embedded Operators

In this manual, we will discuss embedding operators in subspaces of larger quantum systems.

## The `embed` and `unembed` functions

A frequent situation in quantum optimal control is the need to embed a quantum operator in a larger Hilbert space. This is often necessary when the control Hamiltonian acts on a subspace of the full Hilbert space.

The [`embed`](@ref) function allows to embed a quantum operator in a larger Hilbert space.
```@docs
embed
```

The [`unembed`](@ref) function allows to unembed a quantum operator from a larger Hilbert space.
```@docs
unembed
```

For example, for a single qubit X gate embedded in a multilevel system:

````@example embedded_operators
using QuantumCollocation

# define levels of full system
levels = 3

# get a 2-level X gate
X = GATES[:X]

# define subspace indices as lowest two levels
subspace_indices = 1:2

# embed the X gate in the full system
X_embedded = embed(X, subspace_indices, levels)
````

We can retrieve the original operator:

````@example embedded_operators
X_unembedded = unembed(X_embedded, subspace_indices)
````

## The `EmbeddedOperator` type

The `EmbeddedOperator` type is a convenient way to define an operator embedded in a subspace of a larger quantum system.

The `EmbeddedOperator` type is defined as follows:

```@docs
EmbeddedOperator
```

And can be constructed using the following method:

```@docs
EmbeddedOperator(op::Matrix{<:Number}, subspace_indices::AbstractVector{Int}, subsystem_levels::AbstractVector{Int})
```

For example, for a single qubit X gate embedded in a multilevel system:

````@example embedded_operators
# define the target operator

gate = GATES[:X] ⊗ GATES[:I]

subsystem_levels = [3, 3]

subspace_indices = get_subspace_indices([1:2, 1:2], subsystem_levels)

op = EmbeddedOperator(gate, subspace_indices, subsystem_levels)

op.operator

# show the full operator
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

