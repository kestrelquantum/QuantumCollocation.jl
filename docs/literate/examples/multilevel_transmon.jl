# # Multilevel Transmon

# In this example we will look at a multilevel transmon qubit with a Hamiltonian given by
#
# ```math
# \hat{H}(t) = \frac{\delta}{2} \hat{n}(\hat{n} - 1) + u_1(t) (\hat{a} + \hat{a}^\dagger) + u_2(t) i (\hat{a} - \hat{a}^\dagger)
# ```
# where $\hat{n} = \hat{a}^\dagger \hat{a}$ is the number operator, $\hat{a}$ is the annihilation operator, $\delta$ is the anharmonicity, and $u_1(t)$ and $u_2(t)$ are control fields.
#
# We will use the following parameter values:
#
# ```math
# \begin{aligned}
# \delta &= 0.2 \text{ GHz}\\
# \abs{u_i(t)} &\leq 0.2 \text{ GHz}\\
# T_0 &= 10 \text{ ns}\\
# \end{aligned}
# ```

# ## Setting up the problem

using QuantumCollocation
using NamedTrajectories
using LinearAlgebra

## define the time parameters

T₀ = 10     # total time in ns
T = 50      # number of time steps
Δt = T₀ / T # time step

## define the number of levels to model
levels = 3

## create operators
n̂ = number(levels)
â = annihilate(levels)
â_dag = create(levels)

## define the Hamiltonian
δ = 0.2
H_drift = 2π * δ * n̂ * (n̂ - I(levels)) / 2
H_drives = [
    2π * (â + â_dag),
    2π * im * (â - â_dag),
]

## define the goal unitary in the computational subspace
U_init, U_goal = subspace_unitary([levels], :X, 1)

U_goal

# Let's get the subspace indices as well as we will need them later.

subspace = subspace_indices([levels])

## check that these are the correct indices (trivial in the case of a single transmon, but a useful check for more complicated systems)
U_goal[subspace, subspace]

# WE also can look at U_init, which is not exactly the identity

U_init

# Now will set up the optimization problem using the [`UnitarySmoothPulseProblem`](@ref) type.

## set the bound on the pulse amplitude

a_bound = 2π * 0.2

prob = UnitarySmoothPulseProblem(
    H_drift,
    H_drives,
    U_goal,
    T,
    Δt;
    U_init=U_init,
    subspace=subspace,
    a_bound=a_bound
)

## and we can solve this problem

solve!(prob; max_iter=100)

# and we can look at the fidelity in the subspace

f = unitary_fidelity(prob; subspace=subspace)

println("Fidelity: $f")

# We can also look at the pulse shapes

transformations = OrderedDict(
    :Ũ⃗ => [
        x -> populations(iso_vec_to_operator(x)[:, 1]),
        x -> populations(iso_vec_to_operator(x)[:, 2]),
        x -> populations(iso_vec_to_operator(x)[:, 3]),
    ]
)

transforamtion_labels = OrderedDict(
    :Ũ⃗ => [
        "\\psi^g",
        "\\psi^e",
        "\\psi^f",
    ]
)

transformation_titles = OrderedDict(
    :Ũ⃗ => [
        "Populations of evolution from |0⟩",
        "Populations of evolution from |1⟩",
        "Populations of evolution from |2⟩",
    ]
)

plot(prob.trajectory, [:a];
    res=(1200, 1200),
    transformations=transformations,
    transformation_labels=transforamtion_labels,
    include_transformation_labels=true,
    transformation_titles=transformation_titles
)

# ## Leakage suppression

# As can bee seen in the plot above, although the fidelity is high, the $f$ level of the transmon is highly populated throughout the evolution. This is suboptimal, but we can account for this by penalizing the leakage elements of the unitary, namely those elements of the form $U_{f, i}$ where $i \neq f$.  We utilize an $L_1$ penalty on these elements, which is implemented in the [`UnitarySmoothPulseProblem`](@ref) type as the `leakage_penalty` keyword argument.
