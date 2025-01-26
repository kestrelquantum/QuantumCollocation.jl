# Contribution Guide

## Introduction

We welcome contributiuons to QuantumCollocation.jl! This document outlines the guidelines for contributing to the project. If you know what you want to see, but are unsure of the best way to achieve it, [add an issue](https://github.com/kestrelquantum/QuantumCollocation.jl/issues) and start a discussion with the community! 

## Developing

We recommend creating a fresh environment, and using the `dev` command to install `QuantumCollocation.jl` from source. Adding [`Revise.jl`](https://github.com/timholy/Revise.jl) to this environment allows us to make changes to `QuantumCollocation.jl` during development, without restarting Julia or notebooks in VSCode.

### Documentation

Documentation is built using [Documenter.jl](https://github.com/JuliaDocs/Documenter.jl) and uses [Literate.jl](https://github.com/fredrikekre/Literate.jl) to generate markdown files from scripts stored in *docs/literate*. To build the documentation locally, start Julia with the docs environment:

```bash
julia --project=docs
```

Then (for ease of development) load the following packages:

```julia
using Revise, LiveServer, QuantumCollocation
```

To live-serve the docs, run
```julia
servedocs(literate_dir="docs/literate", skip_dir="docs/src/generated")
```

Changes made to files in the docs directory should be automatically reflected in the live server. To reflect changes in the source code (e.g. doc strings), since we are using Revise, simply kill the live server running in the REPL (with, e.g., Ctrl-C) and restart it with the above command. 

### Writing tests

Tests are implemented using the [`TestItems.jl`](https://www.julia-vscode.org/docs/stable/userguide/testitems/) package. 

```Julia
@testitem "Hadamard gate" begin
    H_drift, H_drives = GATES[:Z], [GATES[:X], GATES[:Y]]
    U_goal = GATES[:H]
    T = 51
    Δt = 0.2

    prob = UnitarySmoothPulseProblem(
        H_drift, H_drives, U_goal, T, Δt,
        ipopt_options=IpoptOptions(print_level=1)
    )

    solve!(prob, max_iter=100)
    @test unitary_rollout_fidelity(prob) > 0.99
end
```

Individual tests will populate in the Testing panel in VSCode. All tests are integrated into the base test system for CI, which occurs at each PR submission.

Tests should be included in the same file as the code they test, so `problem_templates/unitary_smooth_pulse_problem.jl` contains the test items for `UnitarySmoothPulseProblem`.

### Reporting Issues

Issue templates are available on GitHub. We are happy to take feature requests!
