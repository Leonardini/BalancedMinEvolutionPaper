"""Write the tables of the paper's Appendices D and E, results/APPENDIX_TABLES.md.

    python experiments/appendix_tables.py [OUT.md]

The same result files as experiments/summarize.py, condensed to the columns the paper
prints; the paper's tables are copied from this file. Timed cells are median [min–max]
over the seeds. A result whose file does not exist yet, or whose task is still running,
shows as "pending".
"""
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize as S  # noqa: E402
from summarize import PENDING, RES, SEEDS, load, table  # noqa: E402


def spread(values, fmt):
    """'median [min–max]' formatted with fmt; a single value when the formatted minimum and
    maximum agree."""
    if not values:
        return "–"
    lo, hi, med = fmt(min(values)), fmt(max(values)), fmt(statistics.median(values))
    return med if lo == hi else f"{med} [{lo}–{hi}]"


def gap(x, d=2):
    return f"{100 * x:.{d}f}%"


def secs(x):
    return f"{x:.1f}" if x < 100 else f"{x:.0f}"


# ------------------------------------------------------------------------- Appendix D

D1_INSTANCES = ("test_n5", "test_n6", "test_n7", "test_n8", "test_n9", "test_n10", "Primates12",
                "M17", "M18", "20_euros2", "20_B-HA", "20_rosids", "21_nucleic", "25_proteic")
# One run each, at seed 1, with the same ten workers and time limit.


def _compact_runs(inst):
    recs = [load(RES / "compact_solver" / inst / f"seed{s}.json") for s in SEEDS]
    return [r for r in recs if r is not None]


def _optimum(opt, inst, recs):
    """L* from the certified optima, the synthetic enumeration, or (20_B-HA) the compact
    solver's own runs when every one of them certified the same value."""
    if inst in opt:
        return opt[inst]["compact"]
    if inst.startswith("test_n"):
        return S.synthetic_optimum(inst)
    vals = {round(r["objective"], 12) for r in recs}
    if recs and all(r["certified"] for r in recs) and len(vals) == 1:
        return recs[0]["objective"]
    return None


def d1_compact(opt):
    rows = []
    for inst in D1_INSTANCES:
        recs = _compact_runs(inst)
        if not recs:
            rows.append([inst, PENDING, "", "", "", ""])
            continue
        cert = sum(r["certified"] for r in recs)
        L = _optimum(opt, inst, recs)
        root = spread([(L - r["root_lb"]) / L for r in recs], lambda x: gap(x)) if L else "–"
        uncert = [r for r in recs if not r["certified"]]
        final = spread([(L - r["lb"]) / L for r in uncert], lambda x: gap(x, 3)) if uncert and L else "–"
        rows.append([inst, f"{cert} of {len(recs)}",
                     spread([r["n_nodes"] for r in recs], lambda x: f"{x:.0f}"),
                     spread([r["seconds"] for r in recs], secs), root, final])
    return ("**Table D1. The compact solver** (Section 8.1): ten workers of one thread each, "
            "time limit 3600 s; seeds 2 and 3 where seed 1 certified. Time in seconds. Gaps relative to $L^*$; the final "
            "gap is given for runs stopped by the time limit.\n\n"
            + table(["instance", "certified", "nodes", "time (s)", "root gap", "final gap"], rows))


D2_INSTANCES = (("Primates12", "distance_solver/Primates12"), ("M17", "distance_solver/M17"), ("M18", "distance_solver/M18"),
                ("20_euros2", "distance_solver/20_euros2"), ("20_B-HA", "distance_solver/20_B-HA"),
                ("20_rosids", "distance_cut_families/full/20_rosids"), ("21_nucleic", "distance_solver/21_nucleic"),
                ("25_proteic", "distance_solver/25_proteic"))


def d2_versus(opt):
    rows = []
    for inst, cdir in D2_INSTANCES:
        recs = _compact_runs(inst)
        c = S.timed_cells(S.seed_runs(RES / cdir))
        if not recs or c is None:
            rows.append([inst, PENDING, "", "", ""])
            continue
        mine = (spread([r["seconds"] for r in recs], secs)
                if all(r["certified"] for r in recs) else
                f"{sum(r['certified'] for r in recs)} of {len(recs)} certified")
        theirs = c["recs"]
        rows.append([inst, mine, spread([r["n_nodes"] for r in recs], lambda x: f"{x:.0f}"),
                     spread([float(r["time"]) for r in theirs], secs)
                     if all(S.certified(r) for r in theirs) else f"{c['cert']} of {c['seeds']} certified",
                     spread([int(r["nodes"]) for r in theirs], lambda x: f"{x:.0f}")])
    return ("**Table D2. The two solvers on the instances of Table D1 with $n\\ge12$** "
            "(Section 8.1): the compact solver as in Table D1, the distance-indexed solver in the "
            "configuration of Section 7.1 (Table D8; 20_rosids from the configuration full of "
            "Table D9, which is the same). Time in seconds to certify, and nodes, for each solver.\n\n"
            + table(["instance", "compact (s)", "nodes", "distance-indexed (s)", "nodes"], rows))


def d3_facets():
    r = load(RES / "lp/named_facets.json")
    if r is None:
        return "**Table D3.** pending"
    rows = [[x["n"], f"{x['opt']:.4f}", f"{x['base']:.4f}", f"{x['f1']:.4f}", f"{x['f1_f6']:.4f}",
             gap(x["closed"], 0)] for x in r["rows"]]
    return ("**Table D3. Root bound with F1 and F6 on the synthetic instances** (Section 8.2). "
            "$L^*$ by enumeration of all trees.\n\n"
            + table(["$n$", "$L^*$", "base", "$+$F1", "$+$F1$+$F6", "gap closed"], rows))


def d4_families(opt):
    rows = []
    others = [f for f in S.T6_FAM if f != "manifold"]
    for inst, kind in S.T6_INST:
        rec = load(RES / "lp/cut_family_shares" / f"{inst}.json")
        if rec is None or "full" not in rec["arms"]:
            rows.append([inst, PENDING, "", "", "", "", ""])
            continue
        a = rec["arms"]
        base, full, mc = a["base"]["bound"], a["full"]["bound"], a["mincut"]["bound"]
        C = full - base
        inc = {f: (a[f"only_{f}"]["bound"] - mc) / C for f in S.T6_FAM if f"only_{f}" in a}
        marg = {f: (full - a[f"without_{f}"]["bound"]) / C for f in S.T6_FAM if f"without_{f}" in a}
        best_other = max(others, key=lambda f: inc.get(f, -1))
        t = a["full"]["census"]["mincut"]
        rows.append([inst, gap((mc - base) / C, 0), f"{t['tight']} of {t['rows']}",
                     gap(inc["manifold"], 0), gap(marg["manifold"], 0),
                     f"{S.T6_NAME[best_other].replace('perfect matching (PM)', 'PM')} "
                     f"{gap(inc[best_other], 0)}",
                     gap(max(abs(marg[f]) for f in others), 1)])
    return ("**Table D4. Shares of the root cut-closeable gap** (Section 8.3; definitions in "
            "Section 7.4), with the manifold imposed exactly (Section 7.2) and every other family separated by the solver's own separators until "
            "none is violated. Min-cut is standalone; the manifold and the other families are "
            "measured with min-cut on: \"incr.\" is the increment over min-cut alone and "
            "\"marg.\" the marginal share in the full set. \"Tight\" counts the min-cut rows "
            "tight at the root optimum. \"Other\" gives the other family with the largest "
            "increment, and the largest marginal share of any other family in absolute value.\n\n"
            + table(["instance", "min-cut", "tight", "manifold incr.", "manifold marg.",
                     "other incr.", "other marg."], rows))


def d5_split_layer():
    names = [("kraft", "Kraft"), ("mincut", "+ min-cut"), ("manifold", "+ manifold"),
             ("split_coupling", "+ split layer and coupling"), ("laminarity", "+ laminarity")]
    recs = {lam: load(RES / "lp/membership_layers" / f"{lam}.json") for lam in ("quadrant", "nested")}
    rows = []
    for key, label in names:
        cells = [label]
        for lam in ("quadrant", "nested"):
            r = recs[lam]
            cells.append(PENDING if r is None or key not in r["arms"] else gap(r["arms"][key]["gap"]))
        rows.append(cells)
    out = ["**Table D5. Root gap on Primates12 of the membership model and the lifted ladder "
           "model** (Section 8.4). Left: membership model, layers added cumulatively, with the "
           "two laminarity encodings. Right: lifted ladder model.", "",
           table(["membership model", "quadrant", "nested/disjoint"], rows), ""]
    rows = []
    for rung in (False, True):
        for cl in (None, "prop4"):
            for man in (False, True):
                name = ("rung" if rung else "ladder") + (f"_{cl}" if cl else "") + ("_manifold" if man else "")
                r = load(RES / "lp/lifted_ladder" / f"{name}.json")
                rows.append(["yes" if rung else "no",
                             {None: "none", "prop4": "Prop. 4"}[cl],
                             "yes" if man else "no", PENDING if r is None else gap(r["ladder"]["gap"])])
    out += [table(["$\\tfrac34$-rung indicator", "cliques", "manifold", "root gap"], rows)]
    return "\n".join(out)


def d6_cut_polytope():
    r = load(RES / "lp/per_split_triangles/Primates12.json")
    if r is None:
        return "**Table D6.** pending"
    rows = []
    for mc in (0, 1):
        cells = ["on" if mc else "off"]
        for st, cp in ((0, 0), (1, 0), (0, 1), (1, 1)):
            cells.append(gap(r["arms"][f"mincut={mc} strongtri={st} cutpoly={cp}"]["gap"], 3))
        diff = r[f"mincut={mc}"]
        cells.append(f"{max(abs(diff['cutpoly_minus_strongtri']), abs(diff['both_minus_strongtri'])):.0e}")
        rows.append(cells)
    return ("**Table D6. Root gap on Primates12 of the level-indicator model with per-split "
            "triangles** (Section 8.4). The last column is the largest difference between the "
            "bound with per-split triangles (alone or with the strong triangle) and the bound "
            "with the strong triangle alone.\n\n"
            + table(["min-cut", "neither", "strong triangle", "per-split", "both", "difference"], rows))


def d7_quartets():
    rows = []
    for inst in ("M17", "M18", "20_euros2", "20_rosids"):
        r = load(RES / "lp/quartet_fixes" / f"{inst}.json")
        if r is None or "summary" not in r:
            rows.append([inst] + [PENDING] + [""] * 6)
            continue
        F, sm = r["fixes"], r["summary"]
        v = len(F)
        no_box = sum(f["box"]["status"] == "infeasible" for f in F)
        no_rows = sum(f["rows"]["status"] == "infeasible" for f in F)
        strong = [f for f in F if 2 - f["separation"] > 0.25]
        no_strong = sum(f["rows"]["status"] == "infeasible" for f in strong)
        rows.append([inst, gap((r["optimum"] - r["root"]["bound"]) / r["optimum"]),
                     f"{v} ({100 * v / r['quartets']:.0f}%)",
                     f"{100 * no_box / v:.0f}%", f"{100 * sm['box']['keeps_manifold'] / v:.0f}%",
                     f"{100 * no_rows / v:.0f}%",
                     f"{100 * no_strong / len(strong):.0f}%" if strong else "–",
                     f"{sm['weights_differing_from_tree']}/{sm['pairs']}"])
    return ("**Table D7. Local fixes of the quartets violated at the compact model's root** "
            "(Section 8.5). At the root optimum $w^*$ (manifold exact): quartets whose "
            "optimal-tree resolution $w^*$ violates; the share of them with no local fix "
            "(a change of the quartet's six weights that keeps Kraft and satisfies the tree's "
            "four-point inequalities) within the box $2^{-(n-1)}\\le w\\le\\tfrac14$; with a "
            "local fix within the box that also keeps the manifold constraint; with no local fix "
            "once every row of the root relaxation is imposed as well (rows), over all violated "
            "quartets and over those violated by more than 12.5% (separation below 1.75 bits); "
            "and the number of weights, out of $\\binom n2$, in which $w^*$ differs from the "
            "optimal tree's point.\n\n"
            + table(["instance", "root gap", "violated", "no fix (box)", "keeps manifold",
                     "no fix (rows)", "> 12.5%", "differ"], rows))


def d8_distance_solver():
    rows = []
    for label in S.TABLE7:
        c = S.timed_cells(S.seed_runs(RES / "distance_solver" / label))
        if c is None:
            rows.append([label, PENDING, "", "", "", ""])
            continue
        recs = c["recs"]
        uncert = [r for r in recs if not S.certified(r)]
        rows.append([label, recs[0]["n"], f"{c['cert']} of {c['seeds']}",
                     spread([float(r["time"]) for r in recs], secs),
                     spread([int(r["nodes"]) for r in recs], lambda x: f"{x:.0f}"),
                     spread([float(r["final_gap_pct"]) for r in uncert], lambda x: f"{x:.3f}%")
                     if uncert else "–"])
    return ("**Table D8. The distance-indexed solver** (Section 8.6), configuration of "
            "Section 7.1, time limit 3600 s. Time in seconds. Last column: final gap of "
            "the runs that did not certify.\n\n"
            + table(["instance", "$n$", "certified", "time (s)", "nodes", "final gap"], rows))


def d9_cut_families(opt):
    labels = {"a": "(a) base", "b": "(b) parity triangles", "gki": "+ GK", "c": "+ 2K, CO",
              "d": "+ 4pt", "full": "full", "cooff": "full − CO"}
    grow, nrow = [], []
    for cfg, _ in S.T8_CFG:
        g, nd = [labels[cfg]], [labels[cfg]]
        for inst in S.T8_INST:
            L = opt.get(inst, {}).get("scaled")
            c = S.timed_cells(S.seed_runs(RES / "distance_cut_families" / cfg / inst))
            if c is None:
                g.append(PENDING), nd.append(PENDING)
                continue
            g.append(spread([(L - float(r["last_root_bound"])) / L for r in c["recs"]],
                            lambda x: f"{100 * x:.3f}"))
            if cfg in ("a", "b"):
                nd.append("–")
                continue
            below = [r for r in c["recs"] if S.certified(r) and float(r["best_sol"]) < L * (1 - S.OPT_REL_TOL)]
            if below:
                nd.append("non-tree")
            elif c["cert"] == c["seeds"]:
                nd.append(spread([int(r["nodes"]) for r in c["recs"]], lambda x: f"{x:.0f}"))
            else:
                nd.append(f"{c['cert']} of {c['seeds']} certified")
        grow.append(g), nrow.append(nd)
    return ("**Table D9. The distance-indexed model by cut configuration** (Section 8.7). "
            "(a) base, no separated cuts; (b) strong triangles strengthened by parity on all "
            "triples; then cumulatively generalized Kraft (GK), 2-K-split and circular order "
            "(2K, CO), and four-point (4pt); full is the configuration of Section 7.1, and "
            "full − CO the same without circular order. (a) and (b) are root LPs; the other rows "
            "are full solves (time limit 3600 s, three seeds where the first certifies), of which we report the converged root bound. "
            "On M18, the + GK run with seed 2 ended early, in each of three attempts, with an internal error of the "
            "solver after it rejected a heuristic solution; we report seeds 1 and 3 for that cell. Top: root gap to $L^*$ "
            "(%). Bottom: nodes to certification; \"non-tree\" marks runs that end at a matrix "
            "below $L^*$ that is not a tree.\n\n"
            + table(["configuration"] + S.T8_INST, grow) + "\n\n"
            + table(["configuration"] + S.T8_INST, nrow))


def d10_lift(opt):
    arms = ["compact", "lift", "lift_manifold", "lift_triangle", "lift_base", "lift_base_2k",
            "lift_2k_K4", "lift_base_buneman"]
    labels = {"compact": "compact", "lift": "+ L", "lift_manifold": "+ L + M",
              "lift_triangle": "+ L + T", "lift_base": "+ L + M + T", "lift_base_2k": "+ L + M + T + 2K",
              "lift_2k_K4": "+ L + 2K$_4$", "lift_base_buneman": "+ L + M + T + 4pt$_1$"}
    insts = ("Primates12", "M17", "M18", "20_euros2", "20_rosids")
    recs = {inst: load(RES / "base_rows" / f"{inst}.json") for inst in insts}
    rows = []
    for k in arms:
        cells = [labels[k]]
        for inst in insts:
            r = recs[inst]
            if r is None or k not in r["arms"]:
                cells.append(PENDING)
                continue
            a_ = r["arms"][k]
            cells.append(f"{a_['root_gap_pct']:.3f}" + ("" if a_.get("finished", True) else "*"))
        rows.append(cells)
    return ("**Table D10. Rows of the distance-indexed base model added to the compact root "
            "relaxation** (Section 8.7), with the compact model's manifold imposed exactly "
            "(Section 7.2). Root gap to $L^*$ (%); a cell marked * is the bound of the arm's last "
            "solve before the conic solver failed, which is valid but may be weaker. L: the level indicators; M: the manifold "
            "equality; T: the triangle inequalities on all triples; 2K: lifted 2-K-splits for "
            "every $K$; 2K$_4$: the distance-indexed solver's root 2-K-split separation, "
            "$K\\le4$; 4pt$_1$: the static four-point rows of every quartet containing leaf 1. "
            "Relaxations with M are linear programs (M implies the compact model's convex "
            "manifold inequality) and are solved by dual simplex with the safe bound; the others "
            "by MOSEK with the rigorous dual bound (Section 7.2).\n\n"
            + table(["relaxation"] + list(insts), rows))


def d11_integrality(opt):
    rows = []
    for inst in ("M17", "M18", "20_euros2", "20_rosids"):
        L = opt.get(inst, {}).get("scaled")
        cells = [inst]
        for v in ("cuts_off", "cuts_on"):
            st, r = S.run_status(RES / "distance_root_cuts" / inst / v)
            cells.append(PENDING if st != "done" or L is None else
                         gap((L - float(r["last_root_bound"])) / L, 3))
        x = load(RES / "base_rows" / f"{inst}.json")
        cells.append(PENDING if x is None else f"{x['arms']['lift_base_2k']['root_gap_pct']:.3f}%")
        cells.append(PENDING if x is None else f"{x['arms']['compact']['root_gap_pct']:.3f}%")
        rows.append(cells)
    return ("**Table D11. Where the distance-indexed model's root bound comes from** (Section "
            "8.7). Root gap to $L^*$ of the distance-indexed model in the configuration with "
            "generalized Kraft only (Table D9), stopped at the root node, with Gurobi's "
            "general-purpose cutting planes off (Cuts 0) and on (default), one thread; and, "
            "for comparison, the compact relaxation with the distance-indexed model's level "
            "indicators, manifold equality, triangles and lifted 2-K-splits added (Table D10), "
            "and the compact relaxation alone, both with the manifold imposed exactly.\n\n"
            + table(["instance", "Gurobi cuts off", "Gurobi cuts on", "compact + base rows",
                     "compact"], rows))


def d12_two_solvers():
    rows = []
    for inst in ("M17", "M18", "20_euros2", "20_rosids", "28_nucleic"):
        r = load(RES / "static_root_two_solvers" / f"{inst}.json")
        if r is None:
            rows.append([inst, PENDING, "", "", "", "", ""])
            continue
        g, c = r["gurobi"], r["cplex"]
        rows.append([inst, f"{g['lp']['root_gap_pct']:.3f}"]
                    + [f"{x[k]['root_gap_pct']:.3f}" for x in (g, c) for k in ("root_on", "root_max")]
                    + [f"{c['root_off']['root_gap_pct']:.3f}"])
    return ("**Table D12. The distance-indexed model's static root in two MIP solvers** (Section "
            "8.7). The model of the generalized-Kraft configuration exported before any callback "
            "cut (so without generalized Kraft), solved on one thread to the end of the root node "
            "with each solver's general-purpose cutting planes at their defaults and at their most "
            "aggressive settings (Gurobi Cuts 3; every CPLEX cut family at its upper limit). Root "
            "gap to $L^*$ (%). Last column: CPLEX with every cut family off; Gurobi with Cuts 0 "
            "stays within 0.12 points of the LP value.\n\n"
            + table(["instance", "LP", "Gurobi", "Gurobi aggr.", "CPLEX", "CPLEX aggr.",
                     "CPLEX off"], rows))


def _short(stem):
    """'21_nucleic_M2839_...' -> '21_nucleic', '20_B-HA-573-...' -> '20_B-HA'."""
    if "_" not in stem:
        return stem
    size, name = stem.split("_")[:2]
    return f"{size}_{'-'.join(name.split('-')[:2])}"


def d13_heuristics():
    r = load(RES / "heuristics_vs_optimum.json")
    if r is None:
        return "**Table D13. NJ and FastME against the certified optimum** (Section 8.8). " + PENDING
    order = ["Primates12", "M17", "M18", "20_euros2", "20_B-HA", "20_rosids"] + S.TABLE7[4:-1]
    recs = sorted(r["instances"], key=lambda x: order.index(_short(x["instance"])))

    def cell(h):
        if h["same_tree"]:
            return "optimal", "0"
        return ("optimal (tie)" if h["optimal"] else gap(h["relative_gap"], 3)), str(h["splits_missing"])

    rows = [[_short(x["instance"]), x["n"], *cell(x["nj"]), *cell(x["fastme"])] for x in recs]
    s = r["summary"]
    return ("**Table D13. NJ and FastME against the certified optimum** (Section 8.8). "
            f"FastME {r['fastme_version'].split()[-1]}: NJ is its method N, without topology search; FastME is "
            "balanced greedy addition followed by balanced NNI and SPR. For each tree: its BME "
            'gap to $L^*$ ("optimal" when it is the optimal tree; "optimal (tie)" when its value is '
            "$L^*$ but the tree differs, the optimum being tied), and the number of splits of the "
            "optimal tree it lacks. NJ is "
            f"optimal on {s['nj']['optimal']} of {s['nj']['instances']} instances, FastME on "
            f"{s['fastme']['optimal']}.\n\n"
            + table(["instance", "$n$", "NJ", "splits", "FastME", "splits"], rows))


# ------------------------------------------------------------------------- Appendix E

E1_GROUPS = [("Table D3", "lp/named_facets.json"),
             ("Table D5", "lp/membership_layers/*.json"), ("Table D5", "lp/lifted_ladder/*.json"),
             ("Table D6", "lp/per_split_triangles/*.json"), ("Section 8.4, odd cycles", "lp/odd_cycles/*.json"),
             ("Table D10", "base_rows/*.json")]


def e1_safe():
    groups = {}
    for name, pattern in E1_GROUPS:
        g = groups.setdefault(name, dict(lps=0, rel=[], cut=0, inf=0, pending=0))
        for p in sorted(RES.glob(pattern)):
            rec = load(p)
            if rec is None:
                g["pending"] += 1
                continue
            for _, rel, cut, inf in S._safe_entries(rec, ""):
                g["lps"] += 1
                g["cut"] += cut or 0
                g["inf"] += bool(inf)
                if rel is not None:
                    g["rel"].append(rel)
    rows = []
    for name, g in groups.items():
        lo = min(g["rel"]) if g["rel"] else None
        rows.append([name, str(g["lps"]) + (f" ({g['pending']} files pending)" if g["pending"] else ""),
                     "–" if lo is None else f"{lo:.0e}", str(g["cut"]), str(g["inf"])])
    return ("**Table E1. Safe bounds and the exact check at the optimal tree.** For each group "
            "of LP bounds: the number of LPs, the most negative relative difference between the "
            "safe bound and the solver's LP value (negative: the floating-point value overstates "
            "the bound), the number of LP rows that cut off the certified optimal tree in exact "
            "arithmetic, and the number of LPs whose safe bound is $-\\infty$.\n\n"
            + table(["result", "LPs", "safe vs LP", "rows cutting off $T^*$", "infinite"], rows))


def e2_enumeration():
    rows = []
    for p in sorted(RES.glob("enumeration/*.json")):
        rec = load(p)
        if rec is None:
            rows.append([p.stem, PENDING, "", "", "", ""])
            continue
        Sz = rec["sizes"]
        gaps = [r["gap_to_runner_up"] for r in Sz.values() if r["gap_to_runner_up"] is not None]
        rows.append([p.stem.replace("_random", " (random)"), str(len(Sz)),
                     f"{sum(bool(r['distance'].get('certified')) for r in Sz.values())}",
                     f"{sum(bool(r['compact']['certified']) for r in Sz.values())}",
                     f"{min(gaps):.0e}" if gaps else "–",
                     str(sum(r[s].get("false_certificate", False) for r in Sz.values()
                             for s in ("distance", "compact")))])
    if not rows:
        return "**Table E2.** pending"
    return ("**Table E2. Certificates against exhaustive enumeration.** Crops of each instance "
            "to $n=6,\\dots,12$ leaves (the first $n$, or three random subsets per $n$). "
            "Columns: crops, crops certified by each solver, smallest relative gap between the "
            "optimum and the best non-optimal tree, and false certificates (a certified tree "
            "that is not optimal). Both solvers had a time limit of one hour per crop.\n\n"
            + table(["instance", "crops", "distance-indexed", "compact", "runner-up gap",
                     "false certificates"], rows))


def e3_conditioning():
    show = (6, 8, 10, 12, 15, 20, 25, 30, 35, 40, 43)
    out = []
    for p in sorted(RES.glob("conditioning/*.json")):
        rec = load(p)
        if rec is None:
            out.append(f"{p.stem}: pending")
            continue
        rows = []
        for n, r in sorted(rec["sizes"].items(), key=lambda kv: int(kv[0])):
            if int(n) not in show:
                continue
            cells = [n]
            for arm in ("compact_without_manifold", "distance"):
                k, rz = r[arm]["kappa_exact"], r[arm]["basis_cond_ruiz"]
                cells.append(f"{k:.0e}" + ("" if rz is None else f" ({rz:.0e})"))
            rows.append(cells)
        out.append(f"*{p.stem}*\n\n" + table(["$n$", "compact", "distance-indexed"], rows))
    if not out:
        return "**Table E3.** pending"
    return ("**Table E3. Condition number of the optimal root basis**, first $n$ leaves: "
            "Gurobi's `KappaExact`, and in parentheses our own 2-norm value after Ruiz "
            "equilibration where the basis is small enough for a dense decomposition. "
            "\"Compact\" is the root LP of the compact model without the manifold constraint; "
            "\"distance-indexed\" the static model of Catanzaro *et al.*\n\n"
            + "\n\n".join(out))


def e4_robustness():
    names = {"no_gki": "without generalized Kraft", "no_2k": "without 2-K-splits",
             "no_co": "without circular order", "no_wb": "without weak four-point",
             "strict": "strict tolerances"}
    rows = []
    for cfg, name in names.items():
        for inst in ("29_B-NS1", "30_B-NS1"):
            d = RES / "robustness" / cfg / inst / "seed1"
            st, r = S.run_status(d)
            if st != "done":
                rows.append([name, inst, st, "", ""])
                continue
            same = (d / "tau.txt").read_bytes() == (RES / "distance_solver" / inst / "seed1" / "tau.txt").read_bytes()
            rows.append([name, inst, "yes" if S.certified(r) else f"no ({float(r['final_gap_pct']):.2f}%)",
                         secs(float(r["time"])), "same" if same else "DIFFERENT"])
    return ("**Table E4. The distance-indexed solver with one cut family switched off, or with "
            "strict tolerances.** Seed 1, time limit 3600 s. Strict tolerances: Gurobi's numerical "
            "emphasis at its maximum (NumericFocus 3) and its feasibility, optimality and "
            "integrality tolerances at $10^{-9}$. Last column: the best tree found, compared "
            "with the tree of the full configuration (Table D8).\n\n"
            + table(["configuration", "instance", "certified", "time (s)", "tree"], rows))


# The threads of each table (Section 7.1): runs that report a time use ten cores, the others one thread.
TIMED, BOUND = "10 threads.", "One thread."
TIMED_BOTH = "Ten cores."
CLASS = {"D1": TIMED_BOTH, "D2": TIMED_BOTH, "D3": BOUND, "D4": BOUND, "D5": BOUND, "D6": BOUND,
         "D7": BOUND, "D8": TIMED,
         "D9": "10 threads, except the two root-LP rows (one thread).",
         "D10": BOUND, "D11": BOUND, "D12": BOUND, "D13": BOUND, "E1": BOUND,
         "E2": "One thread per crop; not comparable with the times of Tables D1 and D8.",
         "E3": BOUND, "E4": TIMED}


def with_threads(block):
    """Insert the table's thread count right after its bold title."""
    import re
    m = re.match(r"\*\*Table ([DE]\d+)\.[^*]*\*\*( \(Section [^)]*\))?[.,:;]?", block)
    if not m:
        return block
    if m.group(1) not in CLASS:
        raise KeyError(f"no thread count for Table {m.group(1)}")
    head = block[:m.end()].rstrip(".,:;")
    rest = block[m.end():]
    sep = " " if head.endswith(".**") else ". "
    if rest.startswith("\n") or not rest.strip():
        return f"{head}{sep}{CLASS[m.group(1)]}{rest}"
    rest = rest.lstrip(" ")
    return f"{head}{sep}{CLASS[m.group(1)]} {rest[0].upper()}{rest[1:]}"


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else RES / "APPENDIX_TABLES.md"
    opt = S.collect_optima()
    parts = ["# Tables of Appendices D and E",
             "Generated by experiments/appendix_tables.py; do not edit by hand.",
             d1_compact(opt), d2_versus(opt), d3_facets(), d4_families(opt), d5_split_layer(),
             d6_cut_polytope(), d7_quartets(), d8_distance_solver(), d9_cut_families(opt), d10_lift(opt), d11_integrality(opt), d12_two_solvers(), d13_heuristics(),
             e1_safe(), e2_enumeration(), e3_conditioning(), e4_robustness()]
    out.write_text("\n\n".join(with_threads(x) for x in parts) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
