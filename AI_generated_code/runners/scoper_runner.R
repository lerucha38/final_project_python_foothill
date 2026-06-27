#!/usr/bin/env Rscript
# Run SCOPer clonal assignment on an AIRR TSV file.

suppressPackageStartupMessages({
  library(scoper)
  library(readr)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Usage: scoper_runner.R <input_airr.tsv> <output.tsv> <method>\n",
       "  method: hierarchical | spectral")
}

input_path <- args[1]
output_path <- args[2]
method <- args[3]

start_time <- Sys.time()
db <- read_tsv(input_path, show_col_types = FALSE, progress = FALSE)

required <- c("sequence_id", "v_call", "j_call", "junction", "junction_length")
missing <- setdiff(required, colnames(db))
if (length(missing) > 0) {
  stop("Missing required columns: ", paste(missing, collapse = ", "))
}

nproc <- as.integer(Sys.getenv("BENCHMARK_NPROC", "1"))
if (is.na(nproc) || nproc < 1) {
  nproc <- 1L
}

threshold <- 0.15

if (method == "hierarchical") {
  result <- hierarchicalClones(
    db,
    threshold = threshold,
    seq_id = "sequence_id",
    v_call = "v_call",
    j_call = "j_call",
    junction = "junction",
    clone = "clone_id",
    nproc = nproc,
    summarize_clones = FALSE
  )
} else if (method == "spectral") {
  result <- spectralClones(
    db,
    method = "novj",
    v_call = "v_call",
    j_call = "j_call",
    junction = "junction",
    clone = "clone_id",
    nproc = nproc,
    summarize_clones = FALSE
  )
} else {
  stop("Unknown method: ", method)
}

elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
write_tsv(result[, c("sequence_id", "clone_id")], output_path)

stats_path <- sub("\\.tsv$", ".stats.json", output_path)
stats <- list(
  method = paste0("scoper_", method),
  runtime_sec = elapsed,
  n_sequences = nrow(result),
  n_clone_groups = length(unique(result$clone_id)),
  nproc = nproc
)
jsonlite::write_json(stats, stats_path, auto_unbox = TRUE, pretty = TRUE)

cat(sprintf("SCOPer %s finished in %.2f sec (%d sequences, %d clones)\n",
            method, elapsed, nrow(result), length(unique(result$clone_id))))
