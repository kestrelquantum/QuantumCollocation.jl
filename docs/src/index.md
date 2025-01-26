```@meta
CurrentModule = QuantumCollocation
```

# QuantumCollocation.jl

*Quickly set up and solve a zoo of quantum optimal control problems.*


## Motivation 

[QuantumCollocation.jl](https://github.com/kestrelquantum/NamedTrajectories.jl) sets up and solves *quantum control problems*. A generic quantum control problem looks like
```math
\begin{aligned}
    \arg \min_{\mathbf{Z}}\quad & J(\mathbf{Z}) \\
    \nonumber \text{s.t.}\qquad & \mathbf{f}(\mathbf{Z}) = 0 \\
    \nonumber & \mathbf{g}(\mathbf{Z}) \le 0  
\end{aligned}
```
where $\mathbf{Z}$ is a trajectory.

*Problem Templates* provide a reusable design pattern for setting up and solving common quantum control problems. For example, a `UnitarySmoothPulseProblem` minimizes infidelity,
```math
    \begin{aligned}
        \arg \min_{\mathbf{Z}}\quad & |1 - \mathcal{F}(U_T, U_\text{goal})|  \\
        \nonumber \text{s.t.}
        \qquad & U_{t+1} = \exp{- i \Delta t_t H(a_t)} U_t \quad \forall\, t \\
    \end{aligned}
```
while a `UnitaryMinimumTimeProblem` minimizes time and constrains fidelity,
```math
    \begin{aligned}
        \arg \min_{\mathbf{Z}}\quad & \sum_{t=1}^T \Delta t_t \\
        \qquad & U_{t+1} = \exp{- i \Delta t_t H(a_t)} U_t \quad \forall\, t \\
        \nonumber & \mathcal{F}(U_T, U_\text{goal}) \ge 0.9999
    \end{aligned}
```

## TODO

- [ ] From core: Document Saving and Loading
- [ ] Document rollouts.
- [ ] Internal links