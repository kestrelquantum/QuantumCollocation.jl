```@meta
EditURL = "../../../literate/examples/two_qubit_gates.jl"
```

# Two Qubit Gates

In this example we will solve for a selection of two-qubit gates using a simple two-qubit system. We will use the [`UnitarySmoothPulseProblem`](@ref) template to solve for the optimal control fields.

## Defining our Hamiltonian

In quantum optimal control we work with Hamiltonians of the form

```math
H(t) = H_{\text{drift}} + \sum_{j} u^j(t) H_{\text{drive}}^j,
```

Specifically, for a simple two-qubit system in a rotating frame, we have

```math
H = J_{12} \sigma_1^x \sigma_2^x + \sum_{i \in {1,2}} a_i^R(t) {\sigma^x_i \over 2} + a_i^I(t) {\sigma^y_i \over 2}.
```

where

```math
\begin{align*}
J_{12} &= 0.001 \text{ GHz}, \\
|a_i^R(t)| &\leq 0.1 \text{ GHz} \\
\end{align*}
```

And the duration of the gate will be capped at $400 \ \mu s$.

Let's now set this up using some of the convenience functions available in QuantumCollocation.jl.

````@example two_qubit_gates
using QuantumCollocation
using NamedTrajectories
using LinearAlgebra

# Define our operators
σx = GATES[:X]
σy = GATES[:Y]
Id = GATES[:I]

# Lift the operators to the two-qubit Hilbert space
σx_1 = σx ⊗ Id
σx_2 = Id ⊗ σx

σy_1 = σy ⊗ Id
σy_2 = Id ⊗ σy

# Define the parameters of the Hamiltonian
J_12 = 0.001 # GHz
a_bound = 0.100 # GHz

# Define the drift (coupling) Hamiltonian
H_drift = J_12 * (σx ⊗ σx)

# Define the control Hamiltonians
H_drives = [σx_1 / 2, σy_1 / 2, σx_2 / 2, σy_2 / 2]

# Define control (and higher derivative) bounds
a_bound = 0.1
da_bound = 0.0005
dda_bound = 0.0025

# Scale the Hamiltonians by 2π
H_drift *= 2π
H_drives .*= 2π

# Define the time parameters
T = 100 # timesteps
duration = 100 # μs
Δt = duration / T
Δt_max = 400 / T

# Define the system
sys = QuantumSystem(H_drift, H_drives)

# Look at max eigenvalue of the generator (for deciding if Pade integrators are viable)
maximum(abs.(eigvals(Δt_max * (H_drift + sum(a_bound .* H_drives)))))
````

That this value above is greater than one means that we must use an exponential integrator for these problems. We can set the kwarg `integrator=:exponential` in the [`PiccoloOptions`](@ref) struct as follows.

````@example two_qubit_gates
piccolo_options = PiccoloOptions(
    integrator=:exponential,
)
````

## SWAP gate

````@example two_qubit_gates
# Define the goal operation
U_goal = [
    1 0 0 0;
    0 0 1 0;
    0 1 0 0;
    0 0 0 1
] |> Matrix{ComplexF64}

# Set up and solve the problem

prob = UnitarySmoothPulseProblem(
    sys,
    U_goal,
    T,
    Δt;
    a_bound=a_bound,
    da_bound=da_bound,
    dda_bound=dda_bound,
    R_da=0.01,
    R_dda=0.01,
    Δt_max=Δt_max,
    piccolo_options=piccolo_options
)

solve!(prob; max_iter=100)

# Let's take a look at the final fidelity
unitary_fidelity(prob)
````

Looks good!

Now let's plot the pulse and the population trajectories for the first two columns of the unitary, i.e. initial state of $\ket{00}$ and $\ket{01}$. For this we provide the function [`plot_unitary_populations`](@ref).

````@example two_qubit_gates
plot_unitary_populations(prob)
````

For fun, let's look at a minimum time pulse for this problem

````@example two_qubit_gates
min_time_prob = UnitaryMinimumTimeProblem(prob; final_fidelity=.99)

solve!(min_time_prob; max_iter=300)

unitary_fidelity(min_time_prob)
````

And let's plot this solution

````@example two_qubit_gates
plot_unitary_populations(min_time_prob)
````

It looks like our pulse derivative bounds are holding back the solution, but regardless, the duration has decreased:

````@example two_qubit_gates
get_duration(prob.trajectory) - get_duration(min_time_prob.trajectory)
````

## Mølmer–Sørensen gate

Here we will solve for a [Mølmer–Sørensen gate](https://en.wikipedia.org/wiki/M%C3%B8lmer%E2%80%93S%C3%B8rensen_gate) between two. The gate is generally described, for N qubits, by the unitary matrix

```math
U_{\text{MS}}(\vec\theta) = \exp\left(i\sum_{j=1}^{N-1}\sum_{k=j+1}^{N}\theta_{jk}\sigma_j^x\sigma_k^x\right),
```

where $\sigma_j^x$ is the Pauli-X operator acting on the $j$-th qubit, and $\vec\theta$ is a vector of real parameters. The Mølmer–Sørensen gate is a two-qubit gate that is particularly well-suited for trapped-ion qubits, where the interaction between qubits is mediated.

Here we will focus on the simplest case of a Mølmer–Sørensen gate between two qubits. The gate is described by the unitary matrix

```math
U_{\text{MS}}\left({\pi \over 4}\right) = \exp\left(i\frac{\pi}{4}\sigma_1^x\sigma_2^x\right).
```

Let's set up the problem.

````@example two_qubit_gates
# Define the goal operation
U_goal = exp(im * π/4 * σx_1 * σx_2)

# Set up and solve the problem

prob = UnitarySmoothPulseProblem(
    sys,
    U_goal,
    T,
    Δt;
    a_bound=a_bound,
    da_bound=da_bound,
    dda_bound=dda_bound,
    R_da=0.01,
    R_dda=0.01,
    Δt_max=Δt_max,
    piccolo_options=piccolo_options
)

solve!(prob; max_iter=1_000)

# Let's take a look at the final fidelity
unitary_fidelity(prob)
````

Again, looks good!

Now let's plot the pulse and the population trajectories for the first two columns of the unitary, i.e. initial state of $\ket{00}$ and $\ket{01}$.

````@example two_qubit_gates
plot_unitary_populations(prob)
````

For fun, let's look at a minimum time pulse for this problem

````@example two_qubit_gates
min_time_prob = UnitaryMinimumTimeProblem(prob; final_fidelity=.999)

solve!(min_time_prob; max_iter=300)

unitary_fidelity(min_time_prob)
````

And let's plot this solution

````@example two_qubit_gates
plot_unitary_populations(min_time_prob)
````

It looks like our pulse derivative bounds are holding back the solution, but regardless, the duration has decreased:

````@example two_qubit_gates
get_duration(prob.trajectory) - get_duration(min_time_prob.trajectory)
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

