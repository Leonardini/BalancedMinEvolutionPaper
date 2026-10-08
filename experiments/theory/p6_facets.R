## Section 4: the facets of the BME polytope P_6 in w-space.
##
##   Rscript p6_facets.R OUTDIR
##
## P_n is the convex hull of w_T (w_ij = 2^-tau_ij) over the binary trees T on n leaves.
## It lies in the affine space of the n Kraft equalities sum_j w_ij = 1/2. We eliminate
## the n coordinates w_ij with i, j at circular distance 2 on the cycle 1, 2, ..., n
## (their Kraft submatrix is invertible) and keep the other n(n-3)/2 coordinates, in
## which P_n is full-dimensional. cddlib (rcdd::scdd, exact rational arithmetic) turns
## the vertices into the facet list. We then lift F1, F2, F6 and the cut inequalities
## W[S] >= 1/2 into the same coordinates and look them up among the facets, and record
## the number of nonzero coefficients of every facet.
##
## Writes OUTDIR/facets.rds (the H-representation, rcdd format: rows (0, b, -a) for
## a.x <= b), OUTDIR/summary.json, and progress to stderr.
suppressPackageStartupMessages({
  library(rcdd)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 1)
outdir <- args[1]
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
n <- 6L
say <- function(...) message(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " ", sprintf(...))

## ---- trees and vertices ------------------------------------------------------------
## Every binary tree on leaves 1..k arises once by attaching leaf k to an edge of a tree
## on 1..k-1. Internal nodes are numbered from n+1.
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
trees <- allTrees(n)
nTrees <- prod(seq(2L * n - 5L, 1L, by = -2L))
stopifnot(length(trees) == nTrees)
W <- t(sapply(trees, function(E) 2^-leafDistances(E, n)[pairs]))   # dyadic, exact in double
stopifnot(nrow(unique(W)) == nTrees)
say("%d trees on %d leaves", nTrees, n)

## ---- coordinates ---------------------------------------------------------------------
circ <- pmin(abs(pairs[, 1] - pairs[, 2]), n - abs(pairs[, 1] - pairs[, 2]))
dep <- which(circ == 2L)
free <- which(circ != 2L)
K <- t(sapply(seq_len(n), function(i) as.numeric(pairs[, 1] == i | pairs[, 2] == i)))
stopifnot(all(K %*% t(W) == 1 / 2))                                   # vertices lie on Kraft
Adep <- K[, dep]; Afree <- K[, free]
detA <- round(abs(det(Adep)))
stopifnot(detA > 0)
Ainv <- round(solve(Adep) * detA) / detA                             # Cramer: entries in Z/det
stopifnot(all(Adep %*% Ainv == diag(n)))
h <- rep(1 / 2, n)
X <- W[, free, drop = FALSE]                                         # projected vertices

## a.w >= beta on the Kraft space  <=>  cvec.x >= beta - const in the kept coordinates
lift <- function(a, beta) {
  cvec <- a[free] - as.numeric(t(Ainv %*% Afree) %*% a[dep])
  const <- sum(a[dep] * as.numeric(Ainv %*% h))
  c(-(beta - const), cvec)                                            # rcdd row without type
}

## Canonical key of a row (b, -a): divide by the largest absolute entry (positive scaling).
rowKey <- function(q) {
  m <- qmax(qabs(q))
  if (m == "0") return(NA_character_)
  paste(qdq(q, rep(m, length(q))), collapse = ",")
}

## ---- facets ----------------------------------------------------------------------------
say("scdd on %d vertices in %d coordinates", nrow(X), ncol(X))
H <- scdd(makeV(points = d2q(X)))$output
say("scdd done: %d rows", nrow(H))
saveRDS(H, file.path(outdir, "facets.rds"))
linearity <- sum(H[, 1] == "1")
facetRows <- H[H[, 1] == "0", -1, drop = FALSE]
## rcdd's row is (b, -a) for a.x <= b, i.e. (b, c) for c.x + b >= 0; the lifted families
## below use the same (b, c) layout.
facetKeys <- apply(facetRows, 1, rowKey)
stopifnot(!anyNA(facetKeys), !anyDuplicated(facetKeys))
support <- rowSums(facetRows[, -1, drop = FALSE] != "0")          # GMP prints zero as "0"
supportHist <- table(factor(support, levels = seq_len(ncol(X))))
say("facet keys and support sizes computed")

idx <- function(i, j) which(pairs[, 1] == min(i, j) & pairs[, 2] == max(i, j))
unitVec <- function(entries) {                     # entries: list of c(i, j, coefficient)
  a <- numeric(npair)
  for (e in entries) a[idx(e[1], e[2])] <- a[idx(e[1], e[2])] + e[3]
  a
}
matchFamily <- function(ineqs) {                   # ineqs: list of list(a, beta)
  keys <- sapply(ineqs, function(q) rowKey(d2q(lift(q$a, q$beta))))
  list(size = length(ineqs), matched = sum(keys %in% facetKeys),
       distinct_facets = length(unique(keys[keys %in% facetKeys])),
       trivial = sum(is.na(keys)), keys = keys)
}

## F1: w_ij >= 2^-(n-1)
F1 <- lapply(seq_len(npair), function(p) list(a = unitVec(list(c(pairs[p, ], 1))),
                                              beta = 2^-(n - 1)))
## F2: w_ij + w_jk - w_ik <= 1/4, for every triple and every choice of the middle leaf j
F2 <- list()
for (tri in combn(n, 3L, simplify = FALSE)) for (j in tri) {
  ik <- setdiff(tri, j)
  F2[[length(F2) + 1]] <- list(a = unitVec(list(c(ik[1], j, -1), c(j, ik[2], -1),
                                                c(ik[1], ik[2], 1))), beta = -1 / 4)
}
## F6: w_kl <= 2^(n-4) w_ij for disjoint pairs {i,j}, {k,l} (ordered)
F6 <- list()
for (p in seq_len(npair)) for (q in seq_len(npair)) {
  if (p != q && length(intersect(pairs[p, ], pairs[q, ])) == 0)
    F6[[length(F6) + 1]] <- list(a = unitVec(list(c(pairs[p, ], 2^(n - 4)), c(pairs[q, ], -1))),
                                 beta = 0)
}
## Cuts W[S] = sum_{i in S, j not in S} w_ij >= 1/2, one per bipartition (S contains leaf 1
## when |S| = n/2)
cutIneq <- function(S) {
  a <- as.numeric(xor(pairs[, 1] %in% S, pairs[, 2] %in% S))
  list(a = a, beta = 1 / 2)
}
cuts <- list()
for (s in 1:(n %/% 2)) {
  Ss <- combn(n, s, simplify = FALSE)
  if (2 * s == n) Ss <- Filter(function(S) 1 %in% S, Ss)
  cuts[[sprintf("(%d,%d)", s, n - s)]] <- lapply(Ss, cutIneq)
}

fam <- list(F1 = matchFamily(F1), F2 = matchFamily(F2), F6 = matchFamily(F6))
namedKeys <- unique(unlist(lapply(fam, function(f) f$keys[f$keys %in% facetKeys])))
cutRes <- lapply(cuts, matchFamily)
say("families matched")

summary <- list(
  n = n, vertices = nTrees, coordinates_kept = ncol(X), eliminated_pairs = dep,
  eliminated_pairs_labels = apply(pairs[dep, ], 1, paste, collapse = "-"),
  det_eliminated_kraft_block = detA, linearity_rows = linearity,
  facets = nrow(facetRows),
  families = lapply(fam, function(f) f[c("size", "matched", "distinct_facets", "trivial")]),
  named_distinct_facets = length(namedKeys),
  named_fraction = length(namedKeys) / nrow(facetRows),
  support_histogram = as.list(setNames(as.integer(supportHist), names(supportHist))),
  full_support_fraction = as.numeric(supportHist[ncol(X)]) / nrow(facetRows),
  cuts = lapply(cutRes, function(f) f[c("size", "matched", "trivial")]))
writeLines(toJSON(summary, auto_unbox = TRUE, pretty = TRUE), file.path(outdir, "summary.json"))
say("facets %d; F1 %d/%d, F2 %d/%d, F6 %d/%d; named distinct %d (%.2f%%); full support %.2f%%",
    nrow(facetRows), fam$F1$matched, fam$F1$size, fam$F2$matched, fam$F2$size,
    fam$F6$matched, fam$F6$size, length(namedKeys), 100 * summary$named_fraction,
    100 * summary$full_support_fraction)
for (k in names(cutRes)) say("cuts %s: %d of %d are facets, %d vanish on the Kraft space",
                             k, cutRes[[k]]$matched, cutRes[[k]]$size, cutRes[[k]]$trivial)
say("support histogram: %s", paste(names(supportHist), as.integer(supportHist), sep = ":",
                                   collapse = " "))

## ---- comparison with the numbers stated in the paper ----------------------------------
pub <- fromJSON(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE))),
                          "published.json"))
stopifnot(linearity == 0,
          nrow(facetRows) == pub$p6_facets,
          fam$F1$matched == pub$p6_family_matches$F1,
          fam$F2$matched == pub$p6_family_matches$F2,
          fam$F6$matched == pub$p6_family_matches$F6,
          length(namedKeys) == pub$p6_named_facets,
          all(as.integer(supportHist) == unlist(pub$p6_support_histogram[names(supportHist)])),
          cutRes[["(3,3)"]]$matched == cutRes[["(3,3)"]]$size,
          cutRes[["(2,4)"]]$matched == 0,
          cutRes[["(1,5)"]]$trivial == cutRes[["(1,5)"]]$size)
say("all numbers agree with the paper")
