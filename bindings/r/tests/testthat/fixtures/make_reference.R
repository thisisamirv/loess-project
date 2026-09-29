# Regenerate the stored golden reference fixtures in this directory.
#
# Usage from the package root:
#   Rscript tests/testthat/fixtures/make_reference.R

args <- commandArgs(trailingOnly = FALSE)
self <- sub("^--file=", "", grep("^--file=", args, value = TRUE))
fixture_dir <- normalizePath(dirname(self))
testthat_dir <- normalizePath(file.path(fixture_dir, ".."))

if (!requireNamespace("rfastloess", quietly = TRUE)) {
    stop(
        "rfastloess is not installed. Build and install it first, e.g.\n",
        "  R CMD INSTALL .  (from bindings/r)"
    )
}
suppressPackageStartupMessages(library("rfastloess", character.only = TRUE))

source(file.path(testthat_dir, "helper-golden.R"))
cases <- golden_cases()

if (!dir.exists(fixture_dir)) {
    dir.create(fixture_dir, recursive = TRUE)
}
for (nm in names(cases)) {
    golden_write_csv(cases[[nm]], file.path(fixture_dir, paste0(nm, ".csv")))
}

files <- paste0(names(cases), ".csv")
sums <- vapply(
    files,
    function(f) unname(tools::md5sum(file.path(fixture_dir, f))),
    character(1)
)
provenance <- c(
    sprintf("generated_by: %s", "tests/testthat/fixtures/make_reference.R"),
    sprintf("generated_at: %s", format(Sys.time(), tz = "UTC", usetz = TRUE)),
    sprintf("r_version: %s", as.character(getRversion())),
    sprintf(
        "rfastloess_version: %s",
        as.character(utils::packageVersion("rfastloess"))
    ),
    sprintf("platform: %s", R.version$platform),
    sprintf("seed: %d", golden_seed()),
    sprintf("rng_kind: %s", paste(RNGkind(), collapse = " / ")),
    sprintf("tolerance: %.1e", 1e-10),
    "files:",
    sprintf("  %s: %s", files, unname(sums))
)
writeLines(provenance, file.path(fixture_dir, "PROVENANCE.txt"))
cat("Wrote", length(cases), "fixtures to", fixture_dir, "\n")
