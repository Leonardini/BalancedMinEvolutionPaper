"""Parse pairwise distance matrix files (R-format: n on line 1, then n*n values)."""

import numpy as np


def parse_matrix(filename):
    """Parse a distance matrix from a text file.

    Supported formats:
      Flat:   first line = n; next n*n lines = values row-major.
      Named:  first line = n; next n lines = '<name>\\t<v1>\\t<v2>...'.
    """
    with open(filename) as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    n = int(lines[0])
    rest = lines[1:]

    if len(rest) == n * n:
        mat = np.array([float(x) for x in rest], dtype=float).reshape(n, n)
    elif len(rest) == n:
        rows = []
        for ln in rest:
            fields = ln.split('\t')
            fields = [f.strip() for f in fields if f.strip()]
            rows.append([float(v) for v in fields[1:]])  # drop taxon name
        mat = np.array(rows, dtype=float)
    else:
        raise ValueError(
            f"Unrecognised format in {filename}: "
            f"n={n}, lines after header={len(rest)}"
        )

    assert np.allclose(np.diag(mat), 0), "Diagonal must be zero"
    assert np.allclose(mat, mat.T, atol=1e-10), "Matrix must be symmetric"
    return mat
