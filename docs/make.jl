using QuantumCollocation
using Documenter
using Literate

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

# DocMeta.setdocmeta!(QuantumCollocation, :DocTestSetup, :(using QuantumCollocation); recursive=true)

pages = [
    "Home" => "index.md",
    "Contribution Guide" => "contribution_guide.md",
    "Manual" => [
        "Problem Templates" => "generated/man/problem_templates.md",
        "Rollouts" => "generated/man/rollouts.md",
        "Callbacks" => "generated/man/ipopt_callbacks.md",
    ],
    "Examples" => [
    ],
    "Library" => "lib.md",
]

format = Documenter.HTML(;
    prettyurls=get(ENV, "CI", "false") == "true",
    canonical="https://kestrelquantum.github.io/QuantumCollocation.jl",
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
)

src = joinpath(@__DIR__, "src")
lit = joinpath(@__DIR__, "literate")

lit_output = joinpath(src, "generated")

for (root, _, files) ∈ walkdir(lit), file ∈ files
    splitext(file)[2] == ".jl" || continue
    ipath = joinpath(root, file)
    opath = splitdir(replace(ipath, lit=>lit_output))[1]
    Literate.markdown(ipath, opath)
end

makedocs(;
    modules=[QuantumCollocation],
    authors="Aaron Trowbridge <aaron.j.trowbridge@gmail.com> and contributors",
    sitename="QuantumCollocation.jl",
    format=format,
    pages=pages,
    warnonly=true,
)

deploydocs(;
    repo="github.com/kestrelquantum/QuantumCollocation.jl.git",
    devbranch="main",
)
