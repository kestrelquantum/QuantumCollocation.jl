# Contribution Guide

## Introduction

We welcome contributiuons to QuantumCollocation.jl! This document outlines the guidelines for contributing to the project. If you know what you want to see, but are unsure of the best way to achieve it, [add an issue](https://github.com/aarontrowbridge/QuantumCollocation.jl/issues) and start a discussion with the community! 

*Let us know how you are using Piccolo!* We enjoy hearing about the problems our users are solving. If you find 

## Developing

We recommend creating a fresh environment, and using the `dev` command to install `QuantumCollocation.jl` from source. Adding [`Revise.jl`](https://github.com/timholy/Revise.jl) to this environment allows us to make changes to `QuantumCollocation.jl` during development, without restarting Julia or notebooks in VSCode.

Here are a few places to think about participating!

**Documentation:**
- [ ] cross-referencing to library
- [ ] adding docstrings, examples, and tests of utilities
- [ ] examples
  - [ ] two-qubit
  - [ ] cat qubit 
  - [ ] three-qubit 
  - [ ] qubit-cavity
  - [ ] qubit-cavity-qubit
- [ ] document type requirements for `Constraints`, `Losses`, and `Objectives` 


**Functionality:**
- [ ] custom `QuantumTrajectory` types (repr. of isomorphic states)
- [ ] better quantum system constructors (e.g. storing composite system info) 
- [ ] refactor `Objectives` and distinguish from `Losses`

### Documentation

Documentation is built using [Documenter.jl](https://github.com/JuliaDocs/Documenter.jl) and uses [Literate.jl](https://github.com/fredrikekre/Literate.jl) to generate markdown files from scripts stored in [docs/literate](docs/literate). To build the documentation locally, start julia with the docs environment:

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

Tests are implemented using the [`TestItemRunner.jl`](https://github.com/julia-vscode/TestItemRunner.jl) package. 

```Julia
@testitem "Hadamard gate" begin
    H_drift, H_drives = GATES[:Z], [GATES[:X], GATES[:Y]]
    U_goal = GATES[:H]
    T = 51
    Δt = 0.2

    prob = UnitarySmoothPulseProblem(
        H_drift, H_drives, U_goal, T, Δt,
        ipopt_options=Options(print_level=1)
    )

    solve!(prob, max_iter=100)
    @test unitary_fidelity(prob) > 0.99
end
```

Individual tests will populate in the Testing panel in VSCode. All tests are integrated into the base test system for CI, which occurs at each PR submission.

We organize our tests in two ways:
1. Modules in single files (e.g. `quantum_utils.jl`, `direct_sums.jl`) should have a single test file in the `test/` directory.
2. Module directories containing templates (e.g. `quantum_system_templates/`, `problem_templates/`) should include tests in the same file that the template is defined, so `problem_templates/unitary_smooth_pulse_problem.jl` includes the test items for `UnitarySmoothPulseProblem`.

### Reporting Issues

Issue templates are available on GitHub. We are happy to take feature requests!