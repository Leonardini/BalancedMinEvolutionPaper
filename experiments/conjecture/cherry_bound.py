"""The cherry conjecture for small n from the manifold alone (Discussion and open problems).

    python cherry_bound.py OUT.json [N ...]          (default N = 4..15)

A matrix with no cherry has every entry >= 3, so each of its rows is a no-cherry row type
(rowtypes.py). The manifold (9), doubled, is sum_i m_i = 2n - 3 with
m_i = sum_{j != i} tau_ij 2^-tau_ij, so it needs n * (smallest m over no-cherry row types)
<= 2n - 3. Where the smallest m exceeds 2 - 3/n, no matrix satisfying (4)-(9) lacks a
cherry. All arithmetic is exact (Fractions). Writes OUT.json, one record per n; progress to
stderr.
"""
import json
import sys
import time
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rowtypes  # noqa: E402


def row_manifold(r):
    return sum(Fraction(t, 2 ** t) for t in r)


def main():
    out = Path(sys.argv[1])
    ns = [int(a) for a in sys.argv[2:]] or list(range(4, 16))
    recs = []
    for n in ns:
        rows = rowtypes.row_types(n)
        nc = rowtypes.nocherry_row_types(n)
        threshold = 2 - Fraction(3, n)
        m_min = min((row_manifold(r) for r in nc), default=None)
        forced = m_min is None or m_min > threshold
        recs.append(dict(n=n, row_types=len(rows), nocherry_row_types=len(nc),
                         min_nocherry_row_manifold=None if m_min is None else str(m_min),
                         threshold=str(threshold), cherry_forced=forced))
        print(time.strftime("%H:%M:%S"), f"n={n}: {len(rows)} row types, {len(nc)} without a cherry; "
              f"smallest row manifold {m_min if m_min is None else float(m_min):} vs 2 - 3/n = "
              f"{float(threshold):.5f}: cherry {'forced' if forced else 'not forced'}",
              file=sys.stderr, flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(recs, indent=1) + "\n")


if __name__ == "__main__":
    main()
