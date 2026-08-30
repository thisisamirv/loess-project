using Documenter
using FastLOESS

# Use the top-level README as the homepage instead of a separately maintained page.
cp(joinpath(@__DIR__, "..", "README.md"), joinpath(@__DIR__, "src", "index.md"); force = true)

makedocs(
    sitename = "FastLOESS.jl",
    modules = [FastLOESS],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://thisisamirv.github.io/loess-project/julia/stable/",
        repolink = "https://github.com/thisisamirv/loess-project",
    ),
    pages = [
        "Home" => "index.md",
        "Introduction" => ["installation.md", "quickstart.md", "concepts.md"],
        "User Guide" => [
            "parameters.md",
            "degree.md",
            "dimensions.md",
            "adapter-choice.md",
            "batch.md",
            "streaming.md",
            "online.md",
            "intervals.md",
            "cross-validation.md",
        ],
        "Weight & Robustness" => ["kernels.md", "robustness.md", "scaling.md"],
        "Advanced" => ["boundary.md", "merge.md"],
        "Use Cases" =>
            ["use-case-genomics.md", "use-case-time-series.md", "use-case-real-time.md"],
        "Performance" => ["benchmarks.md"],
        "API Reference" => "api.md",
    ],
    authors = "Amir Valizadeh",
    warnonly = true,
    checkdocs = :none,
)
