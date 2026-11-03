## Section 5.1 and Appendix E: vertices of the faces Q_T of the splitohedron, Sp(n) with the
## cut of every nontrivial split of a tree T held at 1/2. Only T displays all its splits, so
## w_T is the only tree in Q_T.
##
##   Rscript qt_vertices.R N DIR [floor]
##
## With "floor", every nontrivial bipartition that is not a split of T is also held at
## W[S] >= 3/4, the ladder's floor for non-splits (Theorem 2): every cut is then at least
## its lowest possible rung. The output file gets the suffix _floor.
##
## Reads one tree per shape from DIR/points_n<N>_shapes.json (written by lattice.py),
## builds Sp(n) as in splitohedron_vertices.R plus the split equalities, and enumerates the
## vertices with cddlib (exact rational arithmetic). Prints, per shape, the number of vertices,
## whether x_T is one, and how many vertices are log-integral, to stderr; writes
## DIR/qt_vertices_n<N>[_floor].json.
suppressPackageStartupMessages({
  library(rcdd)
  library(jsonlite)
})
say <- function(...) message(format(Sys.time(), "%H:%M:%S"), " ", sprintf(...))
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) %in% 2:3, length(args) == 2 || args[3] == "floor")
n <- as.integer(args[1])
dir <- args[2]
floor34 <- length(args) == 3
nonLeaf1 <- function(S) sort(if (1L %in% S) setdiff(seq_len(n), S) else S)
recs <- fromJSON(file.path(dir, sprintf("points_n%d_shapes.json", n)), simplifyVector = FALSE)
pairs <- t(combn(n, 2L)); npair <- nrow(pairs)
idx <- function(i, j) which(pairs[, 1] == min(i, j) & pairs[, 2] == max(i, j))
cutRow <- function(S) as.numeric(xor(pairs[, 1] %in% S, pairs[, 2] %in% S))
rows <- list(); rhs <- numeric(0)
for (p in seq_len(npair)) {                                   # 1 <= x <= 2^(n-3)
  a <- numeric(npair); a[p] <- -1; rows[[length(rows) + 1]] <- a; rhs <- c(rhs, -1)
  a <- numeric(npair); a[p] <- 1; rows[[length(rows) + 1]] <- a; rhs <- c(rhs, 2^(n - 3))
}
for (tri in combn(n, 3L, simplify = FALSE)) for (k in tri) {  # intersecting cherry
  ab <- setdiff(tri, k); a <- numeric(npair)
  a[idx(k, ab[1])] <- 1; a[idx(k, ab[2])] <- 1; a[idx(ab[1], ab[2])] <- -1
  rows[[length(rows) + 1]] <- a; rhs <- c(rhs, 2^(n - 3))
}
for (s in 2:(n - 2)) for (S in combn(2:n, s, simplify = FALSE)) {   # every cut, side without leaf 1
  rows[[length(rows) + 1]] <- -cutRow(S); rhs <- c(rhs, -2^(n - 2))
}
A <- do.call(rbind, rows)
K <- t(sapply(seq_len(n), function(i) as.numeric(pairs[, 1] == i | pairs[, 2] == i)))
out <- lapply(recs, function(r) {
  Ss <- lapply(r$tree_splits, function(S) unlist(S) + 1L)       # lattice.py numbers leaves from 0
  E <- rbind(K, t(sapply(Ss, cutRow)))
  e <- c(rep(2^(n - 2), n), rep(2^(n - 2), length(Ss)))
  xT <- unlist(r$tree_x)
  Ar <- A; br <- rhs
  if (floor34) {                                                # non-splits at W >= 3/4
    splitKeys <- sapply(Ss, function(S) paste(nonLeaf1(S), collapse = ","))
    for (s in 2:(n - 2)) for (S in combn(2:n, s, simplify = FALSE)) {
      if (!(paste(S, collapse = ",") %in% splitKeys)) {
        Ar <- rbind(Ar, -cutRow(S)); br <- c(br, -3 * 2^(n - 3))
      }
    }
  }
  stopifnot(all(E %*% xT == e), all(Ar %*% xT <= br))
  H <- rbind(cbind(1, e, -E), cbind(0, br, -Ar))
  t0 <- Sys.time()
  V <- scdd(d2q(H), representation = "H")$output
  stopifnot(all(V[, 1] == "0"), all(V[, 2] == "1"))
  P <- V[, -(1:2), drop = FALSE]
  key <- function(v) paste(v, collapse = ",")
  isT <- apply(P, 1, key) == key(format(xT, scientific = FALSE, trim = TRUE))
  logint <- apply(P, 1, function(v) all(v %in% as.character(2^(0:(n - 3)))))
  res <- list(shape = r$shape, splits = Ss, vertices = nrow(P), tree_is_vertex = any(isT),
              non_tree_vertices = sum(!isT), log_integral_non_tree_vertices = sum(logint & !isT),
              seconds = round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1))
  say("n = %d, shape %s: %d vertices; x_T a vertex: %s; %d non-tree vertices (%d log-integral)",
      n, r$shape, res$vertices, res$tree_is_vertex, res$non_tree_vertices,
      res$log_integral_non_tree_vertices)
  res
})
writeLines(toJSON(out, auto_unbox = TRUE, pretty = TRUE),
           file.path(dir, sprintf("qt_vertices_n%d%s.json", n, if (floor34) "_floor" else "")))
