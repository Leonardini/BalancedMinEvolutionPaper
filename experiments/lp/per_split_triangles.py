"""Sections 5.3 and 8.4: do the per-split cut-polytope triangles add anything to the
level-variable LP beyond the strong triangle inequality they sum to?

    BME_THREADS=1 python per_split_triangles.py INSTANCE SPLITS OUT.json

The root LP of lib/xlevel.py for every combination of the strong triangle (strongtri),
the min-cut family (mincut) and the B variables with per-split triangles (cutpoly).
The comparison the manuscript needs is, for each mincut setting,
    cutpoly alone vs strongtri alone, and strongtri + cutpoly vs strongtri alone.
Each arm also has the safe value of its bound and the optimum check at the SPLITS tree
(lib/safe_bound.py).
"""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import xlevel  # noqa: E402
from lib import safe_bound  # noqa: E402
from lib.common import log, optimal_tree, optimum, parse_matrix, peak_rss_mb, write_json  # noqa: E402


def main():
    inst, splits, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    from gurobipy import GRB
    D = parse_matrix(inst)
    L = optimum(D, splits)
    tree = optimal_tree(D, splits_path=splits)
    rec = dict(instance=inst, n=D.shape[0], optimum=L, arms={})
    for mc, st, cp in itertools.product([False, True], repeat=3):
        name = f"mincut={int(mc)} strongtri={int(st)} cutpoly={int(cp)}"
        M, Bm, x = xlevel.build(D, st, mc, cp)
        M.optimize()
        if M.Status != GRB.OPTIMAL:
            raise RuntimeError(f"{name}: Gurobi status {M.Status}")
        sb = safe_bound.certify(M, name, tree=tree, point=xlevel.tree_point(D.shape[0], tree, x, Bm))
        rec["arms"][name] = dict(mincut=mc, strongtri=st, cutpoly=cp, bound=M.ObjVal,
                                 **safe_bound.fields(sb), gap=(L - M.ObjVal) / L,
                                 rows=M.NumConstrs, safe_bound_detail=sb)
        rec["peak_rss_mb"] = peak_rss_mb()
        write_json(out, rec)
        log(f"{name}: LB={M.ObjVal:.10f} gap={100 * (L - M.ObjVal) / L:.4f}% "
            f"safe - LP {sb['safe_minus_reported']:.3g}, rows cutting off the optimum "
            f"{sb['rows_cut_off_optimum']}")
    a = rec["arms"]
    for mc in (0, 1):
        def b(st, cp):
            return a[f"mincut={mc} strongtri={st} cutpoly={cp}"]["bound"]
        rec[f"mincut={mc}"] = dict(
            cutpoly_minus_strongtri=b(0, 1) - b(1, 0),
            both_minus_strongtri=b(1, 1) - b(1, 0))
        log(f"mincut={mc}: LB(cutpoly) - LB(strongtri) = {b(0, 1) - b(1, 0):.3e}; "
            f"LB(both) - LB(strongtri) = {b(1, 1) - b(1, 0):.3e}")
    write_json(out, rec)


if __name__ == "__main__":
    main()
