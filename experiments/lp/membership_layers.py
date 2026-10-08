"""Section 8.4: root dual gaps of the membership model of Section 5.2, layer by layer.

    BME_THREADS=1 python membership_layers.py INSTANCE SPLITS OUT.json [LAMINARITY]

Cumulative arms: Kraft; + min-cut; + manifold; + split layer and coupling;
+ laminarity (LAMINARITY = quadrant (default) or nested, see lib/membership.py).
SPLITS is the certified optimal tree, which gives L*. Each arm also has the safe value
of its bound and the optimum check at that tree (lib/membership.py, solve_with_cuts).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import membership  # noqa: E402
from lib.common import log, optimal_tree, optimum, parse_matrix, peak_rss_mb, write_json  # noqa: E402

ARMS = [
    ("kraft", dict(mincut=False, manifold=False, split=False, lam=False)),
    ("mincut", dict(mincut=True, manifold=False, split=False, lam=False)),
    ("manifold", dict(mincut=True, manifold=True, split=False, lam=False)),
    ("split_coupling", dict(mincut=True, manifold=True, split=True, lam=False)),
    ("laminarity", dict(mincut=True, manifold=True, split=True, lam=True)),
]


def main():
    inst, splits, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    lam = sys.argv[4] if len(sys.argv) > 4 else "quadrant"
    D = parse_matrix(inst)
    n = D.shape[0]
    L = optimum(D, splits)
    tree = optimal_tree(D, splits_path=splits)
    rec = dict(instance=inst, n=n, optimum=L, laminarity=lam, arms={})
    for name, cfg in ARMS:
        M, w, layers = membership.build(D, cfg["mincut"], cfg["split"],
                                        lam if cfg["lam"] else None)
        r = membership.solve_with_cuts(M, w, n, cfg["manifold"], label=name, tree=tree,
                                       point=membership.tree_point(n, tree, w, layers))
        r["gap"] = (L - r["bound"]) / L
        r["rows"], r["vars"] = M.NumConstrs, M.NumVars
        rec["arms"][name] = r
        rec["peak_rss_mb"] = peak_rss_mb()
        write_json(out, rec)
        log(f"{name}: LB={r['bound']:.8f} gap={100 * r['gap']:.2f}% converged={r['converged']} ({r['stop']})")


if __name__ == "__main__":
    main()
