#!/usr/bin/env Rscript

repos <- "https://cloud.r-project.org"

if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager", repos = repos)
}

BiocManager::install(version = "3.22", ask = FALSE, update = FALSE)
BiocManager::install(c("alakazam", "shazam", "scoper", "ggtree", "ggmsa"), ask = FALSE, update = FALSE)
install.packages(c("jsonlite", "readr", "remotes"), repos = repos)

if (!requireNamespace("statnet", quietly = TRUE)) {
  message("statnet failed to install automatically; fastBCR may require manual setup.")
}

if (requireNamespace("remotes", quietly = TRUE)) {
  remotes::install_github("ZhangLabTJU/fastBCR", upgrade = "never")
}

cat("Installed packages:\n")
for (pkg in c("scoper", "fastBCR")) {
  cat(sprintf("  %s: %s\n", pkg, requireNamespace(pkg, quietly = TRUE)))
}
