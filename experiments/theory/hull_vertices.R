## Appendix E: which log-integral points are vertices of their convex hull?
##
##   Rscript hull_vertices.R POINTS.json
##
## POINTS.json is written by lattice.py: one record per relaxation (Sp(n), or Q_T for one tree
## shape T) with its log-integral points x (x = 2^(n-1) w) and whether each is a tree. cddlib
## (rcdd::redundant, exact rational arithmetic) removes every point that is a convex
## combination of the others; what remains are the vertices of conv(Q and log-integral).
## Prints, for each record, the vertices that are trees and those that are not, and writes
## <POINTS>_hull.json beside the input.
suppressPackageStartupMessages({
  library(rcdd)
  library(jsonlite)
})
say <- function(...) message(format(Sys.time(), "%H:%M:%S"), " ", sprintf(...))

args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 1)
recs <- fromJSON(args[1], simplifyVector = FALSE)
out <- lapply(recs, function(r) {
  X <- do.call(rbind, lapply(r$x, unlist))
  isTree <- unlist(r$is_tree)
  label <- if (is.null(r$shape)) r$relaxation else paste(r$relaxation, r$shape)
  if (nrow(X) == 1) {
    say("%s: 1 point, a vertex (tree: %s)", label, isTree)
    return(list(label = label, points = 1L, vertices = 1L, tree_vertices = sum(isTree),
                non_tree_vertices = sum(!isTree), trees_not_vertices = 0L))
  }
  say("%s: %d points, removing redundant ones", label, nrow(X))
  V <- makeV(points = d2q(X))
  keep <- redundant(V, representation = "V")$new.position
  vert <- which(keep > 0)
  res <- list(label = label, points = nrow(X), vertices = length(vert),
              tree_vertices = sum(isTree[vert]), non_tree_vertices = sum(!isTree[vert]),
              trees_not_vertices = sum(isTree) - sum(isTree[vert]))
  say("%s: %d vertices: %d trees, %d non-trees; %d of %d trees are not vertices", label,
      res$vertices, res$tree_vertices, res$non_tree_vertices, res$trees_not_vertices, sum(isTree))
  res
})
f <- sub("\\.json$", "_hull.json", args[1])
writeLines(toJSON(out, auto_unbox = TRUE, pretty = TRUE), f)
say("wrote %s", f)
