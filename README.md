<!--```@raw html-->
<div align="center">
  <a href="https://github.com/kestrelquantum/Piccolo.jl">
    <img src="assets/logo.svg" alt="Piccolo.jl" width="25%"/>
  </a>
</div>

<div align="center">
  <table>
    <tr>
      <td align="center">
        <b>Documentation</b>
        <br>
        <a href="https://kestrelquantum.github.io/QuantumCollocation.jl/stable/">
          <img src="https://img.shields.io/badge/docs-stable-blue.svg" alt="Stable"/>
        </a>
        <a href="https://kestrelquantum.github.io/QuantumCollocation.jl/dev/">
          <img src="https://img.shields.io/badge/docs-dev-blue.svg" alt="Dev"/>
        </a>
        <a href="https://arxiv.org/abs/2305.03261">
          <img src="https://img.shields.io/badge/arXiv-2305.03261-b31b1b.svg" alt="arXiv"/>
        </a>
      </td>
      <td align="center">
        <b>Build Status</b>
        <br>
        <a href="https://github.com/kestrelquantum/QuantumCollocation.jl/actions/workflows/CI.yml?query=branch%3Amain">
          <img src="https://github.com/kestrelquantum/QuantumCollocation.jl/actions/workflows/CI.yml/badge.svg?branch=main" alt="Build Status"/>
        </a>
        <a href="https://codecov.io/gh/kestrelquantum/QuantumCollocation.jl">
          <img src="https://codecov.io/gh/kestrelquantum/QuantumCollocation.jl/branch/main/graph/badge.svg" alt="Coverage"/>
        </a>
      </td>
      <td align="center">
        <b>License</b>
        <br>
        <a href="https://opensource.org/licenses/MIT">
          <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License"/>
        </a>
      </td>
      <td align="center">
        <b>Support</b>
        <br>
        <a href="https://unitary.fund">
          <img src="https://img.shields.io/badge/Supported%20By-Unitary%20Fund-FFFF00.svg" alt="Unitary Fund"/>
        </a>
      </td>
    </tr>
  </table>
</div>

<div align="center">
  <i> Quickly set up and solve problem templates for quantum optimal control</i>
  <br>
</div>
<!--```-->

# QuantumCollocation.jl

**QuantumCollocation.jl** sets up and solves *quantum control problems* as nonlinear programs (NLPs). In this context, a generic quantum control problem looks like
```math
\begin{aligned}
    \arg \min_{\mathbf{Z}}\quad & J(\mathbf{Z}) \\
    \nonumber \text{s.t.}\qquad & \mathbf{f}(\mathbf{Z}) = 0 \\
    \nonumber & \mathbf{g}(\mathbf{Z}) \le 0  
\end{aligned}
```
where $\mathbf{Z}$ is a trajectory  containing states and controls, from [NamedTrajectories.jl](https://github.com/kestrelquantum/NamedTrajectories.jl).

### Problem Templates 

*Problem Templates* are reusable design patterns for setting up and solving common quantum control problems. 

For example, a *UnitarySmoothPulseProblem* is tasked with generating a *pulse* sequence $a_{1:T-1}$ in orderd to minimize infidelity, subject to constraints from the Schroedinger equation,
```math
    \begin{aligned}
        \arg \min_{\mathbf{Z}}\quad & |1 - \mathcal{F}(U_T, U_\text{goal})|  \\
        \nonumber \text{s.t.}
        \qquad & U_{t+1} = \exp\{- i H(a_t) \Delta t_t \} U_t, \quad \forall\, t \\
    \end{aligned}
```
while a *UnitaryMinimumTimeProblem* minimizes time and constrains fidelity,
```math
    \begin{aligned}
        \arg \min_{\mathbf{Z}}\quad & \sum_{t=1}^T \Delta t_t \\
        \qquad & U_{t+1} = \exp\{- i H(a_t) \Delta t_t \} U_t, \quad \forall\, t \\
        \nonumber & \mathcal{F}(U_T, U_\text{goal}) \ge 0.9999
    \end{aligned}
```

In each case, the dynamics between *knot points* $(U_t, a_t)$ and $(U_{t+1}, a_{t+1})$ are enforced as constraints on the states, which are free variables in the solver; this optimization framework is called *direct collocation*. For details of our implementation please see our award-winning IEEE QCE 2023 paper, [Direct Collocation for Quantum Optimal Control](https://arxiv.org/abs/2305.03261). If you use QuantumCollocation.jl in your work, please cite :raised_hands:!

Problem templates give the user the ability to add other constraints and objective functions to this problem and solve it efficiently using [Ipopt.jl](https://github.com/jump-dev/Ipopt.jl) and [MathOptInterface.jl](https://github.com/jump-dev/MathOptInterface.jl) under the hood.

## Installation

This package is registered! To install, enter the Julia REPL, type `]` to enter pkg mode, and then run:
```julia
pkg> add QuantumCollocation
```

## Example

### Single Qubit Hadamard Gate
```Julia
using QuantumCollocation

T = 50
Δt = 0.2
system = QuantumSystem([PAULIS[:X], PAULIS[:Y]])
U_goal = GATES.H

# Hadamard Gate
prob = UnitarySmoothPulseProblem(system, U_goal, T, Δt)
solve!(prob, max_iter=100)
```
