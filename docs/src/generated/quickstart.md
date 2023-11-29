```@meta
EditURL = "../../literate/quickstart.jl"
```

# Quickstart Guide

To set up and solve a quantum optimal control problems we provide high level problem templates to quickly get started. For unitary gate problems, where we want to realize a gate $U_{\text{goal}}$, with a system Hamiltonian of the form,
```math
H(t) = H_0 + \sum_i a^i(t) H_i
```
there is the `UnitarySmoothPulseProblem` constructor which only requires
- the drift Hamiltonian, $H_0$
- the drive Hamiltonians, $\qty{H_i}$
- the target unitary, $U_{\text{goal}}$
- the number of timesteps, $T$
- the (initial) time step size, $\Delta t$

## Basic Usage

For example, to create a problem for a single qubit $X$ gate (with a bound on the drive of $|a^i| < a_{\text{bound}}$), i.e., with system hamiltonian
```math
H(t) = \frac{\omega}{2} \sigma_z + a^1(t) \sigma_x + a^2(t) \sigma_y
```
we can do the following:

````@example quickstart
using NamedTrajectories
using QuantumCollocation

# set time parameters
T = 100
Δt = 0.1

# use the exported gate dictionary to get the gates we need
σx = gate(:X)
σy = gate(:Y)
σz = gate(:Z)

# define drift and drive Hamiltonians
H_drift = 0.5 * σz
H_drives = [σx, σy]

# define target unitary
U_goal = σx

# set bound on the drive
a_bound = 1.0

# build the problem
prob = UnitarySmoothPulseProblem(
    H_drift,
    H_drives,
    U_goal,
    T,
    Δt;
    a_bound=a_bound,
)

# solve the problem
solve!(prob; max_iter=30)
````

The above output comes from the Ipopt.jl solver. To see the final fidelity we can use the `unitary_fidelity` function exported by QuantumCollocation.jl.

````@example quickstart
println("Final fidelity: ", unitary_fidelity(prob))
````

We can also easily plot the solutions using the `plot` function exported by NamedTrajectories.jl.

````@example quickstart
plot(prob.trajectory, [:Ũ⃗, :a])
````

## Minimum Time Problems

We can also easily set up and solve a minimum time problem, where we enforce a constraint on the final fidelity:
```math
\mathcal{F}(U_T, U_{\text{goal}}) \geq \mathcal{F}_{\text{min}}
```
Using the problem we just solved we can do the following:

````@example quickstart
# final fidelity constraint
final_fidelity = 0.99

# weight on the minimum time objective
D = 10.0

prob_min_time = UnitaryMinimumTimeProblem(
    prob;
    final_fidelity=final_fidelity,
    D=D
)

solve!(prob_min_time; max_iter=30)
````

We can see that the final fidelity is indeed greater than the minimum fidelity we set.

````@example quickstart
println("Final fidelity:    ", unitary_fidelity(prob_min_time))
````

and that the duration of the pulse has decreased.

````@example quickstart
initial_dur = times(prob.trajectory)[end]
min_time_dur = times(prob_min_time.trajectory)[end]

println("Initial duration:  ", initial_dur)
println("Minimum duration:  ", min_time_dur)
println("Duration decrease: ", initial_dur - min_time_dur)
````

We can also plot the solutions for the minimum time problem.

````@example quickstart
plot(prob_min_time.trajectory, [:Ũ⃗, :a])
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

