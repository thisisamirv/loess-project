using Documenter
using FastLOESS

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
		"API Reference" => "api.md",
	],
	authors = "Amir Valizadeh",
	warnonly = true,
	checkdocs = :none,
)
