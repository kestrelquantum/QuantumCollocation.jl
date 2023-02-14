using Pico
using Documenter

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

DocMeta.setdocmeta!(Pico, :DocTestSetup, :(using Pico); recursive=true)

makedocs(;
    modules=[Pico],
    authors="Aaron Trowbridge <aaron.j.trowbridge@gmail.com> and contributors",
    repo="https://github.com/aarontrowbridge/Pico.jl/blob/{commit}{path}#{line}",
    sitename="Pico.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://aarontrowbridge.github.io/Pico.jl",
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
            # "Quantum Costs"     => "quantum_costs.md",
            # "Objectives"        => "objectives.md",
            # "Costs"             => "costs.md",
            # "Constraints"       => "constraints.md",
            # "Integrators"       => "integrators.md",
            # "Problems"          => "problems.md",
        ],
        # "Examples" => "examples.md",
    ],
)

deploydocs(;
    repo="github.com/aarontrowbridge/Pico.jl",
    devbranch="main",
)
