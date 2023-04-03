using QuantumCollocation
using Documenter

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

DocMeta.setdocmeta!(QuantumCollocation, :DocTestSetup, :(using QuantumCollocation); recursive=true)

makedocs(;
    modules=[QuantumCollocation],
    authors="Aaron Trowbridge <aaron.j.trowbridge@gmail.com> and contributors",
    repo="https://github.com/aarontrowbridge/QuantumCollocation.jl/blob/{commit}{path}#{line}",
    sitename="QuantumCollocation.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://aarontrowbridge.github.io/QuantumCollocation.jl",
        edit_link="main",
        assets=String[],
        mathengine = MathJax3(Dict(
            :loader => Dict("load" => ["[tex]/physics"]),
            :tex => Dict(
                "inlineMath" => [["\$","\$"], ["\\(","\\)"]],
                "tags" => "ams",
                "packages" => [
                    "base",
                    "ams",
                    "autoload",
                    "physics"
                ],
            ),
        )),
    ),
    pages=[
        "Introduction" => "index.md",
        "Getting Started" => "getting_started.md",
        "Manual" => [
            "Quantum Systems"   => "quantum_systems.md",
            "Quantum Utilities" => "quantum_utils.md",
            # "Quantum Losses"     => "quantum_losss.md",
            # "Objectives"        => "objectives.md",
            # "Losses"             => "losss.md",
            # "Constraints"       => "constraints.md",
            # "Integrators"       => "integrators.md",
            # "Problems"          => "problems.md",
        ],
        # "Examples" => "examples.md",
    ],
)

deploydocs(;
    repo="github.com/aarontrowbridge/QuantumCollocation.jl",
    devbranch="main",
)
