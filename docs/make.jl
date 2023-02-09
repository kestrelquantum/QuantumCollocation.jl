using Pico
using Documenter

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
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/aarontrowbridge/Pico.jl",
    devbranch="main",
)
