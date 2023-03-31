# Pico.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://aarontrowbridge.github.io/Pico.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://aarontrowbridge.github.io/Pico.jl/dev/)
[![Build Status](https://github.com/aarontrowbridge/Pico.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/aarontrowbridge/Pico.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/aarontrowbridge/Pico.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/aarontrowbridge/Pico.jl)

**Pico.jl** gets it's name from "Pade integrator collocation" and uses [NamedTrajectories.jl](https://github.com/aarontrowbridge/NamedTrajectories.jl) to set up and solve direct collocation problems specific to quantum optimal control, i.e. problems of the form:

```math
\begin{aligned}
\underset{U_{1:T}, a_{1:T-1}}{\text{minimize}} & \quad \ell(U_T, U_{\text{goal}})\\
\text{ subject to } & \quad U_{t+1} = \exp(-i H(a_t)) U_t 
\end{aligned}
```

Pico.jl gives the user the ability to add other constraints and objective functions to this problem and solve it efficiently using [Ipopt.jl](https://github.com/jump-dev/Ipopt.jl) and [MathOptInterface.jl](https://github.com/jump-dev/MathOptInterface.jl) under the hood.

## Notice!

This package is under active development and issues may arise -- please be patient and report any issues you find!

## Installation

Pico.jl is not yet registered, so you will need to install it manually:

```julia
using Pkg
Pkg.add(url="https://github.com/aarontrowbridge/Pico.jl", rev="main")
```
