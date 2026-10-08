"""Table D4 (Section 8.3) with the manifold imposed exactly: which cut families carry the
compact model's root bound.

    BME_THREADS=1 python cut_family_shares.py INSTANCE OUT.json [CAP]

Arms: base, full, default, mincut, only_F and without_F for every optional family F and
for the manifold (definitions in Section 7.4). An arm with the manifold carries it as the
one convex constraint sum_{i<j} entr(w_ij) >= (n - 3/2) ln 2, and every relaxation is
solved by MOSEK through CVXPY (conic_manifold.py).

Every other family is separated by the solver's own separators, uncapped, and its rows are
built by the solver's own row builders (solver/cut_loop.py), recorded instead of added to
an LP. Each round solves the relaxation and then runs every enabled separator at its
optimum, adding all the cuts found; the arm has converged when no separator finds a
violated cut. (The converged relaxation does not depend on the order of separation.)
The bound of an arm is the rigorous Lagrangian bound of its last solve
(conic_manifold.rigorous_bound); every row of the final relaxation is evaluated exactly at
the certified optimal tree (lib/safe_bound.tree_check), and so is the manifold.

For full and default the record has, per family, the rows of the final relaxation, the
rows tight at its optimum (|slack| < TIGHT) and those with a nonzero multiplier
(> TIGHT). The run exits with status 1, after writing the record, if any arm is
unfinished or the full arm's bound is below another arm's by more than MONOTONE_REL.
"""
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "compact"))
sys.path.insert(0, str(HERE))
from conic_manifold import CONIC_SOLVER, base_rows, rigorous_bound, solve  # noqa: E402
from lib import safe_bound  # noqa: E402
from lib.common import log, optimal_tree, parse_matrix, peak_rss_mb, write_json  # noqa: E402

from solver import cut_loop  # noqa: E402
from solver.f45_separator import separate_f4, separate_f5  # noqa: E402
from solver.f6_separator import separate_f6  # noqa: E402
from solver.f7_separator import f7_rhs, separate_f7  # noqa: E402
from solver.paper_cuts import p50_rhs, separate_p50, separate_p53, separate_p54  # noqa: E402
from solver.pm_separator import separate_pm  # noqa: E402
from solver.w_space import (EPS, build_W, find_half_cuts, first_crossing_pair,  # noqa: E402
                            global_min_cut)

OPTIONAL = ['pm', 'f6', 'f7', 'f4', 'f5', 'p50', 'p53', 'p54', 'crossing']
TIGHT = 1e-7
MONOTONE_REL = 1e-9
ALL = 10 ** 9           # no per-call limit on the facet separators


class RowSink:
    """Stands in for a CPLEX model: cut_loop's row builders call
    c.linear_constraints.add(lin_expr=[SparsePair], senses=[s], rhs=[b])."""

    def __init__(self):
        self.new = []
        self.linear_constraints = self

    def add(self, lin_expr, senses, rhs, names=None):
        for sp, s, b in zip(lin_expr, senses, rhs):
            row = {}
            for i, v in zip(sp.ind, sp.val):
                row[int(i)] = row.get(int(i), 0.0) + float(v)
            self.new.append((row, s, float(b)))


def separate(w, n, pairs, pti, disable):
    """Every violated cut of the enabled families at w, as (family, row, sense, rhs)."""
    W = build_W(list(w), n)
    found = []

    def take(fam, fn, *args):
        sink = RowSink()
        fn(sink, *args)
        found.extend((fam, r, s, b) for r, s, b in sink.new)

    mc_val, mc_set = global_min_cut(W, n)
    if 'mincut' not in disable and mc_val < 0.5 - EPS:
        take("mincut", cut_loop._add_cut_mincut, mc_set, n, pti)
    wl = list(w)
    if 'pm' not in disable:
        _, idx, vals, Cn = separate_pm(wl, pairs, pti, n)
        if idx is not None:
            take("pm", cut_loop._add_cut_pm, idx, vals, Cn)
    if 'f6' not in disable:
        cuts = separate_f6(wl, pairs, n, max_cuts=ALL)
        if cuts:
            take("f6", cut_loop._add_cut_f6, cuts, n)
    if 'f7' not in disable:
        cuts = separate_f7(wl, pairs, pti, n, max_cuts=ALL)
        if cuts:
            take("f7", cut_loop._add_cut_f7, cuts, f7_rhs(n))
    if 'f4' not in disable:
        cuts = separate_f4(wl, pairs, pti, n, max_cuts=ALL)
        if cuts:
            take("f4", cut_loop._add_cut_f4, cuts, 2.0 ** (n - 5))
    if 'f5' not in disable:
        cuts = separate_f5(wl, pairs, pti, n, max_cuts=ALL)
        if cuts:
            take("f5", cut_loop._add_cut_f5, cuts, 2.0 ** (n - 4))
    if 'p50' not in disable:
        cuts = separate_p50(wl, pairs, pti, n, max_cuts=ALL)
        if cuts:
            take("p50", cut_loop._add_cut_p50, cuts, p50_rhs(n))
    if 'p53' not in disable:
        cuts = separate_p53(wl, pairs, pti, n, max_cuts=ALL)
        if cuts:
            take("p53", cut_loop._add_cut_p53, cuts, n)
    if 'p54' not in disable:
        cuts = separate_p54(wl, pairs, pti, n, max_cuts=ALL)
        if cuts:
            take("p54", cut_loop._add_cut_p54, cuts, n)
    if 'crossing' not in disable:
        pair = first_crossing_pair(find_half_cuts(W, n))
        if pair is not None:
            take("crossing", cut_loop._add_cut_crossing, pair[0], pair[1], n, pti)
    return found


def root_solve(D, cap, label, disable, manifold, tree, census=False):
    n = D.shape[0]
    pairs = [(i, j) for i in range(1, n + 1) for j in range(i + 1, n + 1)]
    pti = {p: k for k, p in enumerate(pairs)}
    Dv = np.array([D[i - 1, j - 1] for i, j in pairs], dtype=float)
    lo, hi = 2.0 ** -(n - 1), 0.25
    rows, sen, rhs = base_rows(n, pti)
    tags = ["kraft"] * n + ["double_cherry"] * (len(rows) - n)
    keys = {(tuple(sorted(r.items())), s, b) for r, s, b in zip(rows, sen, rhs)}
    counts = {f: 0 for f in ['mincut'] + OPTIONAL}
    t0 = time.time()
    rec = dict(converged=False, capped=False, inaccurate_solves=0)
    rounds = 0
    while True:
        rounds += 1
        w, val, A, b, y, mu, iters, inaccurate = solve(Dv, n, rows, sen, rhs, lo, hi,
                                                       manifold=manifold)
        rec["inaccurate_solves"] += inaccurate
        new = 0
        for fam, r, s, bb in separate(w, n, pairs, pti, disable):
            key = (tuple(sorted(r.items())), s, bb)
            if key in keys:
                continue
            keys.add(key)
            rows.append(r), sen.append(s), rhs.append(bb), tags.append(fam)
            counts[fam] += 1
            new += 1
        if rounds % 25 == 0:
            log(f"  {label}: round {rounds}, bound {val:.12f}, rows {len(rows)}, "
                f"{time.time() - t0:.0f}s")
        if new == 0:
            rec["converged"] = True
            rec["stop"] = "no violated cut"
            break
        if time.time() - t0 >= cap:
            rec["capped"] = True
            rec["stop"] = f"time cap {cap:.0f} s"
            # The last solve's multipliers cover only the rows it saw: its bound is valid.
            break
    m_solved = len(y)
    bound = float(rigorous_bound(Dv, A, b, y, mu, lo, hi, n, sen[:m_solved]))
    rec.update(bound=bound, primal=val, rounds=rounds, rows=counts, manifold=manifold,
               manifold_multiplier=mu, seconds=time.time() - t0, finished=rec["converged"])
    if census:
        slack = A @ w - b
        per = {}
        for i in range(m_solved):
            r = per.setdefault(tags[i], {"rows": 0, "tight": 0, "nonzero_dual": 0})
            r["rows"] += 1
            r["tight"] += int(abs(slack[i]) < TIGHT)
            r["nonzero_dual"] += int(abs(y[i]) > TIGHT)
        rec["census"] = per
    if tree is not None:
        x = np.array([2.0 ** -int(tree["tau"][i - 1, j - 1]) for i, j in pairs])
        lp = safe_bound.LP(A, sen[:m_solved], b, Dv, 0.0, np.full(len(x), lo), np.full(len(x), hi),
                           np.zeros(m_solved), math.nan, CONIC_SOLVER)
        chk = safe_bound.tree_check(lp, x, tags[:m_solved])
        rec["rows_cut_off_optimum"] = chk["rows_cut_off_optimum"]
        rec["manifold_at_optimum_minus_rhs"] = float(sum(v * math.log2(v) for v in x)) - (1.5 - n)
    log(f"  {label}: bound {bound:.12f} ({rec['stop']}), {rounds} rounds, rows {counts}, "
        f"cut off {rec.get('rows_cut_off_optimum')}, {rec['seconds']:.0f}s")
    return rec


def main():
    inst, out = sys.argv[1], Path(sys.argv[2])
    cap = float(sys.argv[3]) if len(sys.argv) > 3 else 1800.0
    D = parse_matrix(inst)
    n = D.shape[0]
    stem = Path(inst).stem
    tree = optimal_tree(D, instance=inst)
    rec = dict(instance=inst, n=n, cap_per_arm_seconds=cap, manifold=f"exact ({CONIC_SOLVER})",
               separator_threshold=EPS, facet_separators_per_call_limit="none",
               optimal_tree="no certified tree" if tree is None else tree["source"],
               standalone_from_base=dict(
                   possible=False,
                   reason="every arm but base contains min-cut"),
               arms={})

    def arm(name, **kw):
        r = root_solve(D, cap, name, tree=tree, **kw)
        rec["arms"][name] = r
        rec["peak_rss_mb"] = peak_rss_mb()
        write_json(out, rec)
        log(f"{stem} {name}: bound={r['bound']:.12f} "
            f"{'converged' if r['converged'] else 'UNFINISHED (' + r['stop'] + ')'} {r['seconds']:.0f}s")

    every = set(OPTIONAL)
    arm("base", disable=every | {"mincut"}, manifold=False)
    arm("full", disable=set(), manifold=True, census=True)
    arm("default", disable={'f6', 'f7', 'f4', 'f5', 'p50', 'p53', 'p54'}, manifold=True,
        census=True)
    arm("mincut", disable=every, manifold=False)
    for f in ['manifold'] + OPTIONAL:
        if f == 'manifold':
            arm("only_manifold", disable=every, manifold=True)
            arm("without_manifold", disable=set(), manifold=False)
        else:
            arm(f"only_{f}", disable=every - {f}, manifold=False)
            arm(f"without_{f}", disable={f}, manifold=True)

    a = rec["arms"]
    full, base = a["full"]["bound"], a["base"]["bound"]
    closeable = full - base
    rec["cut_closeable_gap"] = closeable

    def share(num, arms):
        return dict(value=num / closeable,
                    valid=all(a[x]["finished"] for x in arms + ["full", "base"]))

    rec["shares"] = {"mincut_from_base": share(a["mincut"]["bound"] - base, ["mincut"])}
    for f in ['manifold'] + OPTIONAL:
        rec["shares"][f] = {
            "over_mincut": share(a[f"only_{f}"]["bound"] - a["mincut"]["bound"],
                                 [f"only_{f}", "mincut"]),
            "marginal": share(full - a[f"without_{f}"]["bound"], [f"without_{f}"])}
    rec["unfinished_arms"] = [k for k, r in a.items() if not r["finished"]]
    rec["arms_above_full"] = [k for k, r in a.items()
                              if r["bound"] > full + MONOTONE_REL * abs(full)]
    rec["rows_cut_off_optimum_total"] = sum(r.get("rows_cut_off_optimum") or 0 for r in a.values())
    write_json(out, rec)
    log(f"min-cut from base {rec['shares']['mincut_from_base']['value']:.1%}; "
        f"unfinished arms: {rec['unfinished_arms']}; arms above full: {rec['arms_above_full']}; "
        f"rows cutting off the optimum: {rec['rows_cut_off_optimum_total']}")
    if rec["unfinished_arms"] or rec["arms_above_full"]:
        log("FAILED: the shares are not computed against a converged highest reference")
        sys.exit(1)


if __name__ == "__main__":
    main()
