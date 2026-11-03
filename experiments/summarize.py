"""Collect the results of the paper that are not in its appendix tables into one Markdown
file, results/SUMMARY.md: the certified optima, the facets of P_6 (Section 7.1), the
membership model (Section 7.2), the clique covers (Section 4.2 and Appendix D), the enumeration behind
Theorem 7 (Appendix H), the spurious points of the faces Q_T (Section 5.1, Appendix E), the
evidence for Conjecture 8 (Section 11), and, last, every number of
experiments/theory/published.json that a result file restates, recomputed from that file.
The tables of Appendices I and J come from appendix_tables.py, which uses the helpers of
this file.

    python experiments/summarize.py [OUT.md]

Reads only: the run directories of the distance-indexed solver (results/distance_solver,
results/distance_cut_families) and the JSON records under results/. Runs nothing. A file that does not exist yet is shown as "pending", so
the script can be re-run while experiments are still going; anything else that is wrong
with a file (unparseable, missing field, two certified optima that disagree) stops it, and so
does a recomputed number that disagrees with published.json, after the whole list of
comparisons has been printed to stderr.
"""
import json
import re
import statistics
import sys
from fractions import Fraction
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"
sys.path.insert(0, str(REPO / "run"))
sys.path.insert(0, str(REPO / "experiments/conjecture"))
sys.path.insert(0, str(REPO / "experiments/theory"))
from parse_report import CERT_GAP_PCT, read_run  # noqa: E402
import rowtypes  # noqa: E402

PENDING = "pending"
SEEDS = (1, 2, 3)
# Relative agreement required between two certified optima of the same instance.
OPT_REL_TOL = 1e-6


# ---------------------------------------------------------------- small helpers

def in_progress_outputs():
    """Result paths named on a task line that has started (its log exists) but not
    finished (no .done marker): run/run_tasks.py writes those files incrementally, so they
    are read as pending rather than as complete records."""
    logs = RES / "counting_logs"
    paths = set()
    for f in sorted((REPO / "run/counting_runs.d").glob("*.tasks")):
        for k, line in enumerate(f.read_text().splitlines(), 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            log = logs / f"{f.stem}_{k}.log"
            if log.exists() and not Path(f"{log}.done").exists():
                for tok in line.split():
                    if tok.startswith("$RESULTS/"):
                        paths.add((RES / tok[len("$RESULTS/"):]).resolve())
                    elif tok.startswith("results/"):
                        paths.add((REPO / tok).resolve())
    return paths


IN_PROGRESS = in_progress_outputs()


def load(path):
    """The JSON record at path, or None if it does not exist yet or its task is still
    running."""
    path = Path(path)
    if not path.exists() or path.resolve() in IN_PROGRESS:
        return None
    return json.loads(path.read_text())


def pct(x, digits=2):
    return PENDING if x is None else f"{100 * x:.{digits}f}%"


def num(x, digits=4):
    return PENDING if x is None else f"{x:.{digits}f}"


def spread(values, fmt):
    """'median [min-max]' of the values, formatted with fmt; one value is shown alone."""
    if not values:
        return "–"
    if len(values) == 1 or min(values) == max(values):
        return fmt(values[0])
    return f"{fmt(statistics.median(values))} [{fmt(min(values))}–{fmt(max(values))}]"


def secs(x):
    return f"{x:.1f} s" if x < 100 else f"{x:.0f} s"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def scale(n):
    """The distance-indexed solver reports objective values times 2^(n+1)."""
    return 2.0 ** (n + 1)


# ---------------------------------------------------------- distance-indexed solver

def run_status(rundir):
    """('done', record) | ('crashed', reason) | ('pending', None) for one run directory."""
    rundir = Path(rundir)
    if (rundir / "report.txt").exists():
        return "done", read_run(rundir)
    tfile = rundir / "time.txt"
    if tfile.exists() and "terminated abnormally" in tfile.read_text():
        log = (rundir / "log.txt").read_text() if (rundir / "log.txt").exists() else ""
        m = re.search(r"terminating due to uncaught exception of type ([^\n]+)", log)
        return "crashed", (m.group(1) if m else "solver aborted, no report")
    return PENDING, None


def seed_runs(base):
    """Runs of one configuration on one instance: {seed: (status, record)}."""
    return {s: run_status(Path(base) / f"seed{s}") for s in SEEDS}


def certified(rec):
    return float(rec["final_gap_pct"]) <= CERT_GAP_PCT


# Table I9 configurations whose model is exact for BME: they separate the four-point
# condition, without which (4)-(9) admit non-tree matrices (Theorem 7), so a "certified"
# run of any other configuration certifies a relaxation, not a tree.
EXACT_T8 = ("d", "full", "cooff")


def collect_optima():
    """Certified optimum per instance label (solver scale), from every certified run of
    Table I8 and of the exact Table I9 configurations; all of them must agree."""
    found = {}
    dirs = list(RES.glob("distance_solver/*/seed*"))
    for cfg in EXACT_T8:
        dirs += list(RES.glob(f"distance_cut_families/{cfg}/*/seed*"))
    for rundir in sorted(dirs):
        status, rec = run_status(rundir)
        if status != "done" or not certified(rec):
            continue
        label = rundir.parent.name
        found.setdefault(label, []).append((float(rec["best_sol"]), int(rec["n"]), str(rundir)))
    opt = {}
    for label, vals in found.items():
        ref = vals[0][0]
        for v, _, where in vals:
            assert abs(v - ref) <= OPT_REL_TOL * ref, \
                f"{label}: certified optima disagree, {v} ({where}) vs {ref} ({vals[0][2]})"
        opt[label] = dict(scaled=ref, n=vals[0][1], compact=ref / scale(vals[0][1]),
                          runs=len(vals))
    return opt


def check_optima_against_records(opt):
    """Every JSON record that quotes an optimum must agree with the certified runs."""
    checks = []
    for path in sorted(RES.glob("lp/**/*.json")) + sorted(RES.glob("base_rows/*.json")):
        data = load(path)
        if data is None:
            continue
        # A record is one JSON object; a few scripts (lp/tree_hull.py) write a list of them,
        # one per configuration, and each is checked in the same way.
        if isinstance(data, list):
            recs = data
        elif isinstance(data, dict):
            recs = [data]
        else:
            raise TypeError(f"{path}: expected a JSON object or a list of them, got {type(data).__name__}")
        for rec in recs:
            if not isinstance(rec, dict):
                raise TypeError(f"{path}: a list entry is a {type(rec).__name__}, not a JSON object")
            L = rec.get("optimum", rec.get("L_star"))
            if L is None:
                continue
            label = Path(rec["instance"]).stem
            label = {"01-Primates12": "Primates12", "02-M17": "M17", "03-M18": "M18"}.get(label, label)
            if label not in opt:
                continue
            ref = opt[label]["compact"]
            assert abs(L - ref) <= OPT_REL_TOL * ref, f"{path}: optimum {L} != certified {ref}"
            checks.append(path.relative_to(RES))
    return checks


# ------------------------------------------------------------------------ sections

def sec_optima(opt, checked):
    rows = [(k, v["n"], f"{v['scaled']:.6f}", f"{v['compact']:.9f}", v["runs"])
            for k, v in sorted(opt.items(), key=lambda kv: (kv[1]["n"], kv[0]))]
    return "\n".join([
        "## Certified optima used throughout",
        "",
        "Every certified run of the distance-indexed solver in Table I8, and in the Table I9 "
        f"configurations that separate the four-point condition ({', '.join(EXACT_T8)}), gives "
        "an optimum; for each instance all of them agree to a relative "
        f"{OPT_REL_TOL:g} (asserted). Solver scale is ours times 2^(n+1). The optimum quoted "
        f"in {len(checked)} JSON records under results/ agrees with these values (asserted).",
        "",
        table(["instance", "n", "optimum (solver scale)", "optimum (our scale)",
               "certified runs agreeing"], rows)])


TABLE7 = ["Primates12", "M17", "M18", "20_euros2", "20_B-HA", "20_rosids", "21_nucleic", "22_euros2", "23_euros2",
          "24_proteic", "25_proteic", "26_proteic", "27_nucleic", "28_nucleic", "29_B-NS1",
          "30_B-NS1", "M43"]


def timed_cells(runs):
    """Summary cells of the seeds of one timed configuration on one instance."""
    done = {s: r for s, (st, r) in runs.items() if st == "done"}
    crashed = {s: r for s, (st, r) in runs.items() if st == "crashed"}
    if not done and not crashed:
        return None
    seed1 = runs[1]
    seed1_cert = seed1[0] == "done" and certified(seed1[1])
    recs = list(done.values())
    cert = [r for r in recs if certified(r)]
    notes = []
    for s, why in crashed.items():
        notes.append(f"seed {s} crashed ({why})")
    for s, (st, _) in runs.items():
        if st == PENDING and (s == 1 or seed1_cert):
            notes.append(f"seed {s} pending")
    uncert = [r for r in recs if not certified(r)]
    result = f"{len(cert)} of {len(recs)} certified"
    if uncert:
        result += "; final gap " + spread([float(r["final_gap_pct"]) for r in uncert],
                                          lambda x: f"{x:.3f}%")
    return dict(
        seeds=len(recs), cert=len(cert), result=result, notes="; ".join(notes),
        time=spread([float(r["time"]) for r in recs], secs),
        time_cert=spread([float(r["time"]) for r in cert], secs),
        nodes=spread([int(r["nodes"]) for r in recs], lambda x: f"{x:.0f}"),
        nodes_cert=spread([int(r["nodes"]) for r in cert], lambda x: f"{x:.0f}"),
        mem=spread([float(r["peak_rss_mb"]) for r in recs], lambda x: f"{x:.0f} MB"),
        fourpt=spread([int(r["4pt_root"]) for r in recs], lambda x: f"{x:.0f}"),
        recs=recs)


# Table I9 configurations, in the order of the cumulative ladder.
T8_INST = ["M17", "M18", "20_euros2", "20_rosids"]
T8_CFG = [
    ("a", "(a) base: no separated cuts; root LP only"),
    ("b", "(b) (a) + strong triangles on all triples (flag --str A); root LP only"),
    ("gki", "(gki) (b) + generalized Kraft (our min-cut family)"),
    ("c", "(c) (gki) + 2-K-split + circular order"),
    ("d", "(d) (c) + four-point"),
    ("full", "(full) the Table I8 configuration: strong triangle and four-point on the "
             "quartets containing leaf 1, generalized Kraft, weak four-point, 2-K-split, "
             "circular order"),
    ("cooff", "(cooff) (full) with circular-order cuts switched off"),
]


def synthetic_optimum(inst):
    """L* of a synthetic instance test_nN, from the enumeration of all trees in the Table I3
    record (results/lp/named_facets.json); None if it is not there."""
    r = load(RES / "lp/named_facets.json")
    if r is None or not inst.startswith("test_n"):
        return None
    vals = [row["opt"] for row in r["rows"] if row["n"] == int(inst[len("test_n"):])]
    return vals[0] if vals else None


T6_INST = [("test_n10", "synthetic"), ("Primates12", "real"), ("M17", "real"),
           ("M18", "real"), ("20_B-HA-573-585", "real")]
T6_FAM = ["manifold", "pm", "f6", "f7", "f4", "f5", "p50", "p53", "p54", "crossing"]
T6_NAME = {"manifold": "manifold", "pm": "perfect matching (PM)", "f6": "F6", "f7": "F7",
           "f4": "F4", "f5": "F5", "p50": "(50)", "p53": "(53)", "p54": "(54)",
           "crossing": "crossing"}


def sec_theorem5():
    out = ["## Theorem 7 and Table H1 (Section 8.3, Appendix H)", ""]
    rows = []
    for n in range(5, 12):
        r = load(RES / "theorem5" / f"canon_n{n}.json")
        f = load(RES / "theorem5" / f"full_n{n}.json") if n <= 8 else None
        if r is None:
            rows.append((n, PENDING, "", "", "", ""))
            continue
        rows.append((n, r["survivors"], r["violating"], r["row_types"],
                     f["survivors"] if f else "–",
                     "all pass" if all(c["ok"] for c in r["checks"]) else "FAILED"))
    out += ["Canonical survivors (leaf 1's row in non-decreasing order) and, for n ≤ 8, "
            "survivors without that ordering, which must equal (2n−5)!!:", "",
            table(["n", "canonical survivors", "violating (10)", "row types",
                   "survivors without ordering", "cross-checks against published numbers"],
                  rows), ""]
    items = []
    for name in ("fpviol_n12_t2", "nocherry_n12", "weak_n6"):
        r = load(RES / "theorem5" / f"{name}.json")
        if r is not None:
            bad = [c["label"] for c in r["checks"] if not c["ok"]]
            items.append(f"- {name}: {len(r['checks'])} cross-checks against published numbers, "
                         + ("all pass." if not bad else f"FAILED: {', '.join(bad)}."))
    fp = load(RES / "theorem5/fpviol_n12_t2.json")
    items.append("- n = 12, cherry fixed at {1,2} and a four-point violation at {1,2,3,4}: "
                 + (PENDING if fp is None else
                    f"{fp['survivors']} witnesses, all violating (10), in {fp['classes']} "
                    f"relabelling class(es); the class is the cherry lift of r: "
                    f"{fp['class_is_lift_of_r']}; row types {fp['row_types']}; "
                    f"{fp['seconds']} s."))
    nc = load(RES / "theorem5/nocherry_n12.json")
    items.append("- n = 12, matrices satisfying (4)–(9) with no cherry: "
                 + (PENDING if nc is None else f"{nc['survivors']}."))
    w = load(RES / "theorem5/weak_n6.json")
    items.append("- 6-node matrices with Kraft rows and the weak triangle inequality: "
                 + (PENDING if w is None else
                    f"{w['survivors']}, of which {w['survivors'] - w['violating']} are trees and "
                    f"{w['violating']} are not; the non-trees form {w['weak_orbits']} S_6-orbit "
                    f"(that of r: {w['orbit_is_r']}), orbit size {w['r_orbit_size']}, "
                    f"stabiliser order {w['r_stabiliser']}."))
    cl = load(RES / "theorem5/cherry_lift.json")
    items.append("- Cherry lift of r: " + (PENDING if cl is None else
                 "; ".join(f"{k} {'holds' if v else 'fails'}" for k, v in cl["conditions"].items())
                 + f"; {cl['violating_quartets']} quartets violate (10); pair sums of quartet "
                   f"{{1,2,3,5}}: {cl['quartet_1235_pair_sums']}."))
    p = load(RES / "theorem5/catanzaro_p492.json")
    items.append("- Matrix of Catanzaro et al. (2026, p. 492): " + (PENDING if p is None else
                 "; ".join(f"{k} {'holds' if v else 'fails'}" for k, v in p["conditions"].items())
                 + f"; {p['violating_quartets']} violating quartets; isomorphic to the cherry "
                   f"lift: {p['isomorphic_to_cherry_lift']}."))
    return "\n".join(out + items)


def sec_splitohedron():
    rs = [r for r in (load(RES / f"splitohedron/summary_n{n}.json") for n in (5, 6, 7)) if r is not None]
    if not rs:
        return "## Vertices of the splitohedron (Appendix E)\n\npending"
    return "\n".join(["## Vertices of the splitohedron (Appendix E)", ""] + [
        f"- n = {r['n']}: {r['vertices']} vertices, of which {r['tree_vertices']} are trees "
        f"(of {r['trees']}) and {r['other_vertices']} are not." for r in rs])


def sec_p6():
    r = load(RES / "p6_facets/summary.json")
    if r is None:
        return "## Facets of P_6 (Section 7.1)\n\npending"
    fam = "; ".join(f"{k}: {v['matched']} of {v['size']}" for k, v in r["families"].items())
    cuts = "; ".join(f"{k}-cuts: {v['size']} inequalities, {v['matched']} are facets, "
                     f"{v['trivial']} are trivial (implied by the Kraft equalities)"
                     for k, v in r["cuts"].items())
    hist = ", ".join(f"{k}: {v}" for k, v in r["support_histogram"].items())
    return "\n".join([
        "## Facets of P_6 (Section 7.1)", "",
        f"- Vertices: {r['vertices']}; facets: {r['facets']} (in the {r['coordinates_kept']} "
        f"coordinates left after eliminating the Kraft equalities).",
        f"- Named families matching a facet: {fam}; distinct facets matched "
        f"{r['named_distinct_facets']} ({100 * r['named_fraction']:.2f}% of all facets).",
        f"- Facets with full support ({r['coordinates_kept']} coordinates): "
        f"{100 * r['full_support_fraction']:.1f}%. Support histogram: {hist}.",
        f"- Cut inequalities W[S] ≥ 1/2: {cuts}."])


def sec_conjecture1():
    rows = []
    for n, lam, tb in [(6, "nested-disjoint", "on"), (6, "quadrant", "on"),
                       (7, "nested-disjoint", "on"), (7, "quadrant", "on"),
                       (8, "nested-disjoint", "on"), (8, "quadrant", "on"),
                       (8, "nested-disjoint", "off"), (8, "quadrant", "off")]:
        r = load(RES / "conjecture1" / f"n{n}_{lam}_tiebreak-{tb}.json")
        if r is None:
            rows.append((n, lam, tb, PENDING, "", "", ""))
            continue
        rows.append((n, lam, tb, r["points"], r["distinct_split_systems"], r["tree_systems"],
                     r["non_tree_systems"]))
    return "\n".join([
        "## Conjecture 6: integral points of the membership model (Section 7.2)", "",
        "Complete enumeration of the integral points of the membership model with "
        "symmetry breaking. \"Leaf-1 tie-break\" places leaf 1 outside every split of size "
        "n/2 (only matters for even n). An integral point is a full assignment of the "
        "membership variables; its split system is the set of splits it encodes.", "",
        table(["n", "laminarity encoding", "leaf-1 tie-break", "integral points",
               "distinct split systems", "of which trees", "non-trees"], rows)])


def sec_clique():
    rows = []
    for n in range(4, 9):
        r = load(RES / "clique_cover" / f"n{n}.json")
        rows.append((n, PENDING, "", "", "", "", "", "") if r is None else
                    (n, r["vertices"], r["edges"], r["maximal_cliques"],
                     r["fractional_edge_clique_cover"], r.get("min_edge_clique_cover", "not computed"),
                     r["pruned_cliques"], r["proposition4_cliques"]))
    out = ["## Crossing graph: edge clique covers (Section 4.2 and Appendix D)", "",
           table(["n", "nontrivial bipartitions", "crossing pairs", "maximal cliques",
                  "fractional minimum", "minimum edge clique cover", "subfamily of Appendix D",
                  "cliques of Proposition 4"], rows), ""]
    r = load(RES / "clique_cover/integrality.json")
    if r is None:
        return "\n".join(out + ["Integrality: pending"])
    fc = r["five_cycle"]
    out += [f"Induced 5-cycle at n = {fc['n']}: consecutive pairs cross: "
            f"{fc['consecutive_cross']}; non-consecutive pairs compatible: "
            f"{fc['non_consecutive_compatible']}.", "",
            f"LPs over the clique inequalities of Proposition 4 with random objectives "
            f"(seed {r['seed']}):", "",
            table(["n", "splits", "cliques", "trials", "fractional optima",
                   "optima with every coordinate 1/2", "distinct fractional values (first few)"],
                  [(x["n"], x["splits"], x["cliques"], x["trials"], x["fractional_optima"],
                    x["optima_with_every_coordinate_half"],
                    ", ".join(x["fractional_values"][:6]) +
                    (f", … ({len(x['fractional_values'])} in all)"
                     if len(x["fractional_values"]) > 6 else ""))
                   for x in r["lp"]])]
    return "\n".join(out)


def _safe_entries(node, where):
    """(where, safe_minus_lp, rows_cut_off_optimum, safe_infinite) for every LP in a record
    that carries a safe bound; the per-LP detail blocks repeat the summary, so they are
    skipped."""
    out = []
    if isinstance(node, dict):
        for k, v in node.items():
            if k.endswith("safe_minus_lp"):
                prefix = k[: -len("safe_minus_lp")]
                sb = node.get(f"{prefix}safe_bound")
                rel = None if v is None or not sb else v / abs(sb)
                out.append((f"{where}{prefix.rstrip('_')}", rel, node.get(f"{prefix}rows_cut_off_optimum"),
                            node.get(f"{prefix}safe_infinite", False)))
        for k, v in node.items():
            if k != "safe_bound_detail" and isinstance(v, (dict, list)):
                out += _safe_entries(v, f"{where}{k}/")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            out += _safe_entries(v, f"{where}{i}/")
    return out


# ------------------------------------------------- cross-checks against published.json

PUBLISHED = json.loads((REPO / "experiments/theory/published.json").read_text())
CHECKS = []      # (label, computed, expected, status), filled by the sections below


def check(label, computed, expected):
    """Record one comparison with a number stated in the paper; None means the result file
    does not exist yet. A disagreement is collected and stops main() after every section
    has run, so that the whole list is reported."""
    status = PENDING if computed is None else "pass" if computed == expected else "FAIL"
    CHECKS.append((label, computed, expected, status))
    return status


def published_checks():
    """Every number of published.json that a result file under results/ restates."""
    P = PUBLISHED
    t5 = RES / "theorem5"
    for n, v in P["canonical_survivors"].items():
        r = load(t5 / f"canon_n{n}.json")
        check(f"Table H1: canonical survivors, n = {n}", r and r["survivors"], v)
    for n, v in P["row_types"].items():
        r = load(t5 / (f"canon_n{n}.json" if int(n) <= 11 else f"fpviol_n{n}_t2.json"))
        check(f"row types (enumerator), n = {n}", r and r["row_types"], v)
        check(f"row types (rowtypes.py), n = {n}", len(rowtypes.row_types(int(n))), v)
    r = load(t5 / "fpviol_n12_t2.json")
    check("Appendix H (ii): witnesses with a cherry at {1,2}", r and r["survivors"], P["cherry_fixed_witnesses"])
    check("Appendix H (ii): relabelling classes", r and r["classes"], P["cherry_fixed_classes"])
    r = load(t5 / "weak_n6.json")
    check("Appendix H (iii): weak-triangle 6-node matrices", r and r["survivors"], P["weak6_survivors"])
    check("Appendix H (iii): of which non-trees", r and r["violating"], P["weak6_non_trees"])
    check("Appendix H (iii): S_6-orbits of the non-trees", r and r["weak_orbits"], P["weak6_orbits"])
    check("Appendix H (iii): stabiliser order of r", r and r["r_stabiliser"], P["weak6_stabiliser"])
    r = load(t5 / "cherry_lift.json")
    check("Section 8.3: quartets of the cherry lift violating (10)", r and r["violating_quartets"],
          P["cherry_lift_violating_quartets"])
    check("Section 8.3: pair sums of the quartet {1,2,3,5}", r and r["quartet_1235_pair_sums"],
          P["cherry_lift_quartet_1235_pair_sums"])
    r = load(RES / "p6_facets/summary.json")
    check("Section 7.1: facets of P_6", r and r["facets"], P["p6_facets"])
    for k, v in P["p6_family_matches"].items():
        check(f"Section 7.1: facets matched by {k}", r and r["families"][k]["distinct_facets"], v)
    check("Section 7.1: facets matched by a named family", r and r["named_distinct_facets"], P["p6_named_facets"])
    check("Section 7.1: support histogram of the facets", r and r["support_histogram"], P["p6_support_histogram"])
    for n, v in P["membership_systems"].items():
        for lam in ("nested-disjoint", "quadrant"):
            r = load(RES / "conjecture1" / f"n{n}_{lam}_tiebreak-on.json")
            check(f"Section 7.2: split systems, n = {n}, {lam}", r and r["distinct_split_systems"], v)
    for lam in ("nested-disjoint", "quadrant"):
        r = load(RES / "conjecture1" / f"n8_{lam}_tiebreak-off.json")
        check(f"Section 7.2: split systems, n = 8, {lam}, no tie-break", r and r["distinct_split_systems"],
              P["membership_n8_no_tiebreak"])
    for key, field, what in (("min_edge_clique_cover", "min_edge_clique_cover", "minimum edge clique cover"),
                             ("fractional_edge_clique_cover", "fractional_edge_clique_cover", "fractional minimum"),
                             ("pruned_clique_cover", "pruned_cliques", "cover of Appendix D")):
        for n, v in P[key].items():
            r = load(RES / "clique_cover" / f"n{n}.json")
            check(f"Appendix D: {what}, n = {n}", r and r[field], v)
    for n, v in P["splitohedron_vertices"].items():
        r = load(RES / "splitohedron" / f"summary_n{n}.json")
        check(f"Appendix E: vertices of Sp({n})", r and r["vertices"], v["vertices"])
        check(f"Appendix E: tree vertices of Sp({n})", r and r["tree_vertices"], v["trees"])


def sec_checks():
    rows = [(label, "–" if c is None else c, e, st) for label, c, e, st in CHECKS]
    n_fail = sum(1 for *_, st in CHECKS if st == "FAIL")
    n_pend = sum(1 for *_, st in CHECKS if st == PENDING)
    return "\n".join([
        "## Cross-checks against the numbers stated in the paper", "",
        f"Every number of experiments/theory/published.json that a result file restates, "
        f"recomputed from that file: {len(CHECKS) - n_fail - n_pend} pass, {n_fail} fail, "
        f"{n_pend} pending.", "",
        table(["check", "computed", "stated in the paper", "status"], rows)])


# ------------------------------------------------------------- the cherry conjecture

def sec_cherry_conjecture():
    import coverage  # experiments/conjecture
    import plm       # experiments/theory
    P = PUBLISHED["cherry_conjecture"]
    base = RES / "conjecture/nocherry"
    out = ["## Conjecture 8: a cherry always exists (Section 11)", ""]
    bound = load(RES / "conjecture/cherry_bound.json")
    if bound is None:
        out += ["Manifold bound: pending", ""]
    else:
        forced = [r["n"] for r in bound if r["cherry_forced"]]
        out += ["Manifold bound (cherry_bound.py): every row of a matrix without a cherry is a "
                "no-cherry row type, so the manifold needs n times the smallest row manifold of "
                f"those types to be at most 2n − 3. This forces a cherry for n = "
                f"{', '.join(map(str, forced))} and not for the other n computed.", ""]
    rows, max_n = [], P["exhaustive_max_n"]
    dirs = sorted((int(d.name[1:]), d) for d in base.glob("n*") if d.is_dir())
    for n, d in dirs:
        rep = coverage.coverage(d, n)
        status = ("covered" if rep["covered"] else
                  "incomplete, missing " + "; ".join("_".join(map(str, m)) for m in rep["missing"]))
        dup = "; ".join(f"{'_'.join(map(str, x['prefix'])) or 'whole search'} ({x['how']})"
                        for x in rep["duplicates"]) or "–"
        rows.append((n, rep["pieces_used"], rep["pieces_unfinished"], rep["pieces_outside"], dup,
                     f"{rep['cpu_seconds'] / 3600:.1f}", rep["survivors"], status))
        if n <= max_n:
            check(f"Conjecture 8: the exhaustive search covers n = {n}", rep["covered"], True)
            check(f"Conjecture 8: matrices without a cherry, n = {n}", rep["survivors"], 0)
        else:
            pend = P["pending"].get(str(n))
            if pend is not None:
                check(f"Conjecture 8: the pieces still missing at n = {n}", rep["missing"], pend)
    covered_n = [n for n, d in dirs if n <= max_n]
    # n = 4 is the smallest number of leaves with a quartet, so the claim starts there.
    check(f"Conjecture 8: the exhaustive search has a result for every n from 4 to {max_n}",
          covered_n, list(range(4, max_n + 1)))
    out += [f"Exhaustive search without cherries (backtrack_wide8, cut into pieces by the first "
            f"values of leaf 0's sorted row; coverage.py). The paper states n ≤ {max_n}. "
            "\"Unfinished\" pieces were stopped and replaced by their split; \"outside\" pieces "
            "fix a value that leaves no completion of row 0; a part covered twice must give the "
            "same count both times (checked).", "",
            table(["n", "pieces used", "unfinished", "outside", "covered twice", "core hours (all pieces)",
                   "survivors", "coverage"],
                  rows), ""]
    r12 = load(RES / "theorem5/nocherry_n12.json")
    if r12 is not None and (base / "n12/bt12.out").exists():
        w8 = coverage.parse_out(base / "n12/bt12.out")
        assert (w8["survivors"], w8["nodes"]) == (r12["survivors"], r12["nodes"]), (w8, r12)
        out += [f"At n = 12 the search of backtrack_wide8 and the one of Appendix H "
                f"(experiments/theory/backtrack.c --nocherry) visit the same {w8['nodes']} nodes "
                "(asserted).", ""]
    return "\n".join(out)


# ------------------------------------------------------ faces Q_T of the splitohedron

FACES = RES / "splitohedron/faces"
# constraint set: (file-name suffix of lattice.py, --floor, double cherry kept)
VARIANTS = {"no_floor": ("", False, True), "floor": ("_floor", True, True),
            "floor_no_dc": ("_floor_nodc", True, False)}


def n_cherries(splits, n):
    """Cherries of a tree given by its nontrivial splits (the side without leaf 1)."""
    return sum(1 for S in splits if len(S) in (2, n - 2))


def sec_faces():
    P = PUBLISHED
    out = ["## Spurious points of the faces Q_T (Section 5.1, Appendix E)", "",
           "Log-integral points of Q_T (lattice.py --shapes), one tree per shape; a shape fails "
           "when Q_T has a log-integral point other than w_T.", ""]
    rows, shapes = [], {}
    for v, (suffix, floor, dc) in VARIANTS.items():
        per_n = {}
        for f in FACES.glob(f"points_n*_shapes{suffix}.json"):
            m = re.fullmatch(rf"points_n(\d+)_shapes{suffix}\.json", f.name)
            if not m:
                continue
            recs = load(f)
            n = int(m.group(1))
            for r in recs:
                assert r["n"] == n and r["relaxation"] == "Q_T", f
                # older records of lattice.py do not name the flags; their file name does
                assert r.get("floor", floor) == floor and r.get("dc", dc) == dc, f
                assert r["trees"] == 1 and r["points"] == r["trees"] + r["non_trees"], f
            per_n[n] = sorted(r["non_trees"] for r in recs if r["non_trees"])
            assert shapes.setdefault(n, len(recs)) == len(recs), f"{f}: shape count differs between constraint sets"
            rows.append((v.replace("_", " "), n, len(recs), len(per_n[n]),
                         ", ".join(map(str, per_n[n])) or "–"))
        ns = sorted(per_n)
        if not ns:
            continue
        assert ns == list(range(ns[0], ns[-1] + 1)), (v, ns)
        first = next((n for n in ns if per_n[n]), None)
        pub = P["qt_log_integral"][v]
        check(f"Section 5.1: first n with a spurious log-integral point ({v.replace('_', ' ')}; "
              f"checked from n = {ns[0]})", first, pub["first_failing_n"])
        for n, counts in pub["failing"].items():
            check(f"Section 5.1: spurious log-integral points per failing shape, {v.replace('_', ' ')}, n = {n}",
                  per_n.get(int(n)), counts)
    for n, k in P["tree_shapes"].items():
        check(f"Section 5.1: tree shapes, n = {n}", shapes.get(int(n)), k)
    out += [table(["constraints", "n", "shapes", "failing shapes", "spurious points per failing shape"],
                  sorted(rows, key=lambda r: (r[0], r[1]))), "",
            "Vertices of Q_T (qt_vertices.R, cddlib in exact arithmetic):", ""]
    rows = []
    for v in ("no_floor", "floor"):
        for n in range(5, 14):
            recs = load(FACES / f"qt_vertices_n{n}{VARIANTS[v][0]}.json")
            if recs is None:
                continue
            not_vertex = sum(1 for r in recs if not r["tree_is_vertex"])
            spurious = [r for r in recs if r["non_tree_vertices"]]
            for r in recs:
                rows.append((v.replace("_", " "), n, r["shape"], n_cherries(r["splits"], n), r["vertices"],
                             "yes" if r["tree_is_vertex"] else "no", r["non_tree_vertices"]))
            pub = P["qt_vertices"][v].get(str(n), {})
            if "tree_not_vertex" in pub:
                check(f"Section 5.1: shapes with w_T not a vertex of Q_T, {v.replace('_', ' ')}, n = {n}",
                      not_vertex, pub["tree_not_vertex"])
            if "with_spurious_vertex" in pub:
                check(f"Section 5.1: shapes with a spurious vertex, {v.replace('_', ' ')}, n = {n}",
                      len(spurious), pub["with_spurious_vertex"])
            if "spurious_shape_cherries" in pub:
                check(f"Section 5.1: the shape with a spurious vertex is the caterpillar (2 cherries), "
                      f"{v.replace('_', ' ')}, n = {n}",
                      [n_cherries(r["splits"], n) for r in spurious], pub["spurious_shape_cherries"])
    out += [table(["constraints", "n", "shape", "cherries", "vertices", "w_T a vertex", "spurious vertices"],
                  rows), "", "Hull of the log-integral points (hull_vertices.R):", ""]
    rows = []
    for f in sorted(FACES.glob("*_hull.json")):
        for r in load(f):
            if r["points"] > 1 or r["label"] == "Sp":
                rows.append((f.name, r["label"], r["points"], r["vertices"], r["tree_vertices"],
                             r["non_tree_vertices"], r["trees_not_vertices"]))
    out += [table(["file", "relaxation", "log-integral points", "hull vertices", "trees",
                   "non-trees", "trees not vertices"], rows), "",
            "Log-integral points of Sp(n) itself (lattice.py without --shapes):", ""]
    rows = []
    for n in range(5, 14):
        recs = load(FACES / f"points_n{n}.json")
        if recs is not None:
            r, = recs
            rows.append((n, r["points"], r["trees"], r["non_trees"]))
    out.append(table(["n", "log-integral points", "trees", "non-trees"], rows))
    return "\n".join(out)


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else RES / "SUMMARY.md"
    opt = collect_optima()
    checked = check_optima_against_records(opt)
    parts = [
        "# Summary of results\n\n"
        "Generated by experiments/summarize.py from the files under results/; do not edit by "
        "hand. \"pending\" marks a result whose file does not exist yet. Timed cells are "
        "median [min–max] over the seeds that finished; a single value means all seeds agree "
        "or only one seed ran.",
        sec_optima(opt, checked),
        sec_p6(),
        sec_splitohedron(),
        sec_conjecture1(),
        sec_clique(),
        sec_theorem5(),
        sec_faces(),
        sec_cherry_conjecture(),
    ]
    published_checks()
    parts.append(sec_checks())
    for label, c, e, st in CHECKS:
        print(f"{st:7s} {label}: computed {c}, paper {e}", file=sys.stderr)
    failed = [label for label, _, _, st in CHECKS if st == "FAIL"]
    if failed:
        raise SystemExit("published.json cross-checks failed:\n  " + "\n  ".join(failed))
    out.write_text("\n\n".join(parts) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
