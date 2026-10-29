## Appendix I: the vertices of the splitohedron Sp(n) of Forcey, Keefe and Sands (2017,
## Definition 5.1), an outer relaxation of the BME polytope P_n. Sp(n) contains P_n, and every tree is a vertex of Sp(n) for n <= 11
## (their Theorem 5.2), so the two are equal exactly when Sp(n) has no other vertex.
##
##   Rscript splitohedron.R N OUTDIR
##
## We work in x_ij = 2^(n-1) w_ij = 2^(n-1-tau_ij), the paper's integer scaling, and write
## Sp(n) as the n Kraft equalities plus four families of inequalities:
##   Kraft              sum_{j != i} x_ij = 2^(n-2)                      for every leaf i
##   caterpillar        x_ij >= 1                                         for every pair
##   cherry clade-face  x_ij <= 2^(n-3)                                   for every pair
##   intersecting-cherry x_ab + x_bc - x_ac <= 2^(n-3)                    for every triple, b in the middle
##   split              sum_{i<j in S} x_ij <= (|S|-1) 2^(n-3)            for every split S|S' with |S|, |S'| >= 3
## A split and its complement give the same inequality on the Kraft space, so each split is
## listed once (|S| <= n/2, and S contains leaf 1 when |S| = n/2).
## cddlib (rcdd::scdd, exact rational arithmetic) enumerates the vertices, which we compare
## with the (2n-5)!! tree vectors.
##
## Writes OUTDIR/vertices_n<N>.rds (the V-representation), OUTDIR/summary_n<N>.json, and
## progress to stderr, then checks the counts against those stated in Appendix I
## (published.json, key splitohedron_vertices).
suppressPackageStartupMessages({
  library(rcdd)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2)
n <- as.integer(args[1])
outdir <- args[2]
stopifnot(n >= 4)
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
say <- function(...) message(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " ", sprintf(...))

## ---- trees (as in p6_facets.R) ---------------------------------------------------------
allTrees <- function(n) {
  trees <- list(rbind(c(1L, n + 1L), c(2L, n + 1L), c(3L, n + 1L)))
  for (k in 4:n) {
    mid <- n + k - 2L
    trees <- do.call(c, lapply(trees, function(E) {
      lapply(seq_len(nrow(E)), function(e) {
        rbind(E[-e, , drop = FALSE], c(E[e, 1], mid), c(mid, E[e, 2]), c(k, mid))
      })
    }))
  }
  trees
}

leafDistances <- function(E, n) {
  nodes <- 2L * n - 2L
  adj <- lapply(seq_len(nodes), function(v) c(E[E[, 1] == v, 2], E[E[, 2] == v, 1]))
  D <- matrix(0L, n, n)
  for (s in seq_len(n)) {
    dist <- rep(NA_integer_, nodes); dist[s] <- 0L; frontier <- s
    while (length(frontier) > 0) {
      nxt <- integer(0)
      for (v in frontier) for (u in adj[[v]]) if (is.na(dist[u])) {
        dist[u] <- dist[v] + 1L; nxt <- c(nxt, u)
      }
      frontier <- nxt
    }
    D[s, ] <- dist[seq_len(n)]
  }
  D
}

pairs <- t(combn(n, 2L))
npair <- nrow(pairs)
dimension <- npair - n
trees <- allTrees(n)
nTrees <- prod(seq(2L * n - 5L, 1L, by = -2L))
stopifnot(length(trees) == nTrees)
X <- t(sapply(trees, function(E) 2^(n - 1 - leafDistances(E, n)[pairs])))   # integers
stopifnot(all(X == round(X)), nrow(unique(X)) == nTrees)
say("n = %d: %d trees, %d coordinates, dimension %d", n, nTrees, npair, dimension)

## ---- H-representation of Sp(n): rows (a, b, kind) for a.x <= b or a.x = b -------------
idx <- function(i, j) which(pairs[, 1] == min(i, j) & pairs[, 2] == max(i, j))
unitVec <- function(entries) {                     # entries: list of c(i, j, coefficient)
  a <- numeric(npair)
  for (e in entries) a[idx(e[1], e[2])] <- a[idx(e[1], e[2])] + e[3]
  a
}
row <- function(a, b, family) list(a = a, b = b, family = family)

kraft <- lapply(seq_len(n), function(i)
  row(as.numeric(pairs[, 1] == i | pairs[, 2] == i), 2^(n - 2), "Kraft"))
caterpillar <- lapply(seq_len(npair), function(p) row(-unitVec(list(c(pairs[p, ], 1))), -1, "caterpillar"))
cherry <- lapply(seq_len(npair), function(p) row(unitVec(list(c(pairs[p, ], 1))), 2^(n - 3), "cherry clade-face"))
intersecting <- do.call(c, lapply(combn(n, 3L, simplify = FALSE), function(tri) lapply(tri, function(b) {
  ac <- setdiff(tri, b)
  row(unitVec(list(c(ac[1], b, 1), c(b, ac[2], 1), c(ac[1], ac[2], -1))), 2^(n - 3), "intersecting-cherry")
})))
splitSides <- do.call(c, lapply(seq_len(max(0L, n %/% 2L - 2L)) + 2L, function(k) {
  Ss <- combn(n, k, simplify = FALSE)
  if (2L * k == n) Ss <- Filter(function(S) 1L %in% S, Ss)
  Ss
}))
split <- lapply(splitSides, function(S)
  row(as.numeric(pairs[, 1] %in% S & pairs[, 2] %in% S), (length(S) - 1) * 2^(n - 3), "split"))
ineqs <- c(caterpillar, cherry, intersecting, split)
families <- sapply(ineqs, function(r) r$family)
A <- t(sapply(ineqs, function(r) r$a)); b <- sapply(ineqs, function(r) r$b)
K <- t(sapply(kraft, function(r) r$a)); kb <- sapply(kraft, function(r) r$b)
say("inequalities: %s", paste(names(table(families)), as.integer(table(families)), sep = " ", collapse = ", "))

## Every tree satisfies Sp(n).
stopifnot(all(K %*% t(X) == kb))
slack <- b - A %*% t(X)                          # rows x trees
stopifnot(all(slack >= 0))
tightPerRow <- rowSums(slack == 0)
stopifnot(all(tightPerRow > 0))                  # every row is a face of P_n containing a tree

## ---- vertex enumeration ------------------------------------------------------------------
## rcdd H-rows: (l, b, -a) meaning b - a.x >= 0, with l = 1 for an equality.
H <- rbind(cbind(1, kb, -K), cbind(0, b, -A))
say("scdd on %d rows (%d equalities) in %d coordinates", nrow(H), n, npair)
t0 <- Sys.time()
V <- scdd(d2q(H), representation = "H")$output
secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
saveRDS(V, file.path(outdir, sprintf("vertices_n%d.rds", n)))
stopifnot(all(V[, 1] == "0"), all(V[, 2] == "1"))   # bounded: points only, no rays or lines
P <- V[, -(1:2), drop = FALSE]
say("scdd done in %.1f s: %d vertices", secs, nrow(P))

treeKeys <- apply(X, 1, function(v) paste(format(v, scientific = FALSE, trim = TRUE), collapse = ","))
vertexKeys <- apply(P, 1, paste, collapse = ",")
isTree <- vertexKeys %in% treeKeys
## Theorem 5.2 of Forcey et al. says every tree is a vertex for n <= 11; we report, not assume.
say("trees among the vertices: %d of %d", sum(treeKeys %in% vertexKeys), nTrees)
extra <- P[!isTree, , drop = FALSE]
extraIntegral <- sum(apply(extra, 1, function(v) !any(grepl("/", v))))
say("vertices: %d trees, %d others (%d of them integral)", sum(isTree), nrow(extra), extraIntegral)

## A few non-tree vertices, in w = x / 2^(n-1), for the record.
example <- if (nrow(extra) > 0) {
  w <- qdq(extra[1, ], rep(as.character(2^(n - 1)), npair))
  setNames(as.list(w), apply(pairs, 1, paste, collapse = "-"))
} else NULL
summary <- list(
  n = n, trees = nTrees, dimension = dimension,
  inequalities = as.list(table(families)), equalities = n,
  trees_among_vertices = sum(treeKeys %in% vertexKeys),
  vertices = nrow(P), tree_vertices = sum(isTree), other_vertices = nrow(extra),
  other_vertices_integral = extraIntegral, equal_to_bme_polytope = nrow(extra) == 0 && all(treeKeys %in% vertexKeys),
  scdd_seconds = secs, example_other_vertex_w = example)
writeLines(toJSON(summary, auto_unbox = TRUE, pretty = TRUE, digits = NA),
           file.path(outdir, sprintf("summary_n%d.json", n)))
say("Sp(%d) %s P_%d", n, if (nrow(extra) == 0) "=" else "!=", n)

## ---- comparison with the numbers stated in the paper ----------------------------------
pub <- fromJSON(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE))),
                          "published.json"))
stated <- pub$splitohedron_vertices[[as.character(n)]]
if (!is.null(stated)) stopifnot(nrow(P) == stated$vertices, sum(isTree) == stated$trees)
