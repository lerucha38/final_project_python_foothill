#!/usr/bin/env Rscript
# Run fastBCR clonal assignment on an AIRR TSV file.

suppressPackageStartupMessages({
  library(fastBCR)
  library(readr)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: fastbcr_runner.R <input_airr.tsv> <output.tsv>")
}

input_path <- args[1]
output_path <- args[2]

nproc <- as.integer(Sys.getenv("BENCHMARK_NPROC", "1"))
if (is.na(nproc) || nproc < 1) {
  nproc <- 1L
}

start_time <- Sys.time()
db <- read_tsv(input_path, show_col_types = FALSE, progress = FALSE)

required <- c("sequence_id", "v_call", "j_call", "junction_aa")
missing <- setdiff(required, colnames(db))
if (length(missing) > 0) {
  stop("Missing required columns: ", paste(missing, collapse = ", "))
}

if (!"junction" %in% colnames(db)) {
  db$junction <- NA_character_
}

if (!"productive" %in% colnames(db)) {
  db$productive <- TRUE
}

count_col <- if ("duplicate_count" %in% colnames(db)) "duplicate_count" else NA

raw_data_list <- list(sample = db)
pro_data_list <- data.preprocess(
  raw_data_list = raw_data_list,
  productive_only = TRUE,
  count_col_name = count_col
)

cluster_list <- data.BCR.clusters(
  pro_data_list = pro_data_list,
  min_depth_thre = 3,
  min_depth_thre_adjustment = TRUE,
  max_depth_thre = 1000,
  overlap_thre = 0.1,
  consensus_thre = 0.8,
  paired = FALSE,
  singletons_backtrack = TRUE
)

backtrack_list <- data.clusters.backtrack(
  raw_data_list = raw_data_list,
  clusters_list = cluster_list
)

family_list <- backtrack_list[[1]]
if (is.null(family_list) || length(family_list) == 0) {
  stop("fastBCR backtrack returned no clonal families")
}

parts <- lapply(seq_along(family_list), function(i) {
  family <- family_list[[i]]
  if (!"sequence_id" %in% colnames(family)) {
    stop("fastBCR backtrack family missing sequence_id")
  }
  data.frame(
    sequence_id = family$sequence_id,
    clone_id = paste0("fastbcr_", i),
    stringsAsFactors = FALSE
  )
})
out <- do.call(rbind, parts)

assigned_ids <- unique(out$sequence_id)
missing_ids <- setdiff(db$sequence_id, assigned_ids)
if (length(missing_ids) > 0) {
  singletons <- data.frame(
    sequence_id = missing_ids,
    clone_id = paste0("fastbcr_singleton_", missing_ids),
    stringsAsFactors = FALSE
  )
  out <- rbind(out, singletons)
}

out <- out[!duplicated(out$sequence_id), ]
write_tsv(out, output_path)

elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
stats_path <- sub("\\.tsv$", ".stats.json", output_path)
stats <- list(
  method = "fastbcr",
  runtime_sec = elapsed,
  n_sequences = nrow(out),
  n_clone_groups = length(unique(out$clone_id)),
  nproc = nproc
)
jsonlite::write_json(stats, stats_path, auto_unbox = TRUE, pretty = TRUE)

cat(sprintf("fastBCR finished in %.2f sec (%d sequences, %d clones)\n",
            elapsed, nrow(out), length(unique(out$clone_id))))
