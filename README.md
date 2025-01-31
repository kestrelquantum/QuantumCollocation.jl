# QuantumCollocation.jl

<div align="center"> <a href="https://github.com/kestrelquantum/Piccolo.jl">
    <img src="docs/src/assets/logo.svg" alt="logo" width="35%"/>
</a> </div>


<div align="center">

| **Documentation** | **Build Status** | **Support** | **Paper** | **License** |
|:-----------------:|:----------------:|:-----------:|:---------:|:-----------:|
| [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://kestrelquantum.github.io/QuantumCollocation.jl/dev/) | [![Build Status](https://github.com/aarontrowbridge/QuantumCollocation.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/kestrelquantum/QuantumCollocation.jl/actions/workflows/CI.yml?query=branch%3Amain) [![Coverage](https://codecov.io/gh/kestrelquantum/QuantumCollocation.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/kestrelquantum/QuantumCollocation.jl)| [![Unitary Fund](https://img.shields.io/badge/Supported%20By-Unitary%20Fund-FFFF00.svg)](https://unitary.fund) | [![arXiv](https://img.shields.io/badge/arXiv-2305.03261-b31b1b.svg)](https://arxiv.org/abs/2305.03261) | [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


</div>


**QuantumCollocation.jl** uses [NamedTrajectories.jl](https://github.com/kestrelquantum/NamedTrajectories.jl) to set up and solve direct collocation problems specific to quantum optimal control, i.e. generating a *pulse* sequence $a_{1:T-1}$ to drive a quantum system and realize a target gate $U_{\text{goal}}$. We formulate this problem as a nonlinear program (NLP) of the form

```math
\begin{aligned}
\underset{U, a, \Delta t}{\text{minimize}} & \quad \ell(U_T, U_{\text{goal}})\\
\text{ subject to } & \quad U_{t+1} = \exp(-i \Delta t H(a_t)) U_t 
\end{aligned}
```

Where the dynamics between *knot points* $(U_t, a_t)$ and $(U_{t+1}, a_{t+1})$ are enforced as constraints on the states which are free variables in the solver; this optimization framework is called *direct collocation*.  For details of our implementation please see our award-winning IEEE QCE 2023 paper, [Direct Collocation for Quantum Optimal Control](https://arxiv.org/abs/2305.03261). If you use QuantumCollocation.jl in your work, please cite :raised_hands:!

QuantumCollocation.jl gives the user the ability to add other constraints and objective functions to this problem and solve it efficiently using [Ipopt.jl](https://github.com/jump-dev/Ipopt.jl) and [MathOptInterface.jl](https://github.com/jump-dev/MathOptInterface.jl) under the hood.

## :warning: Notice :warning:

This package is under active development and issues may arise -- please be patient and report any issues you find!

## Installation

QuantumCollocation.jl is registered! To install:

```julia
using Pkg
Pkg.add(QuantumCollocation)
```

## Example

### Single Qubit X-Gate
```Julia
using QuantumCollocation

T = 50
Δt = 0.2
system = QuantumSystem([PAULIS[:X], PAULIS[:Y]])

# Hadamard Gate
prob = UnitarySmoothPulseProblem(system, GATES[:H], T, Δt)
solve!(prob, max_iter=100)

plot_unitary_populations(prob)
```
![Single Qubit X-Gate](docs/src/assets/x_gate_unitary_populations.svg)
