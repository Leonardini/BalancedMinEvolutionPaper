"""Collect the results of the paper that are not in its appendix tables into one Markdown
file, results/SUMMARY.md: the certified optima, the facets of P_6 (Section 4), the
membership model (Section 5.2), the clique covers (Section 5.1 and Appendix H), the enumeration behind
Theorem 5 (Appendix C). The tables of Appendices D and E
come from appendix_tables.py, which uses the helpers of this file.

    python experiments/summarize.py [OUT.md]

Reads only: the run directories of the distance-indexed solver (results/distance_solver,
results/distance_cut_families) and the JSON records under results/. Runs nothing. A file that does not exist yet is shown as "pending", so
the script can be re-run while experiments are still going; anything else that is wrong
with a file (unparseable, missing field, two certified optima that disagree) stops it.
"""
import json
import re
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"
sys.path.insert(0, str(REPO / "run"))
from parse_report import CERT_GAP_PCT, read_run  # noqa: E402

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


# Table D9 configurations whose model is exact for BME: they separate the four-point
# condition, without which (4)-(9) admit non-tree matrices (Theorem 5), so a "certified"
# run of any other configuration certifies a relaxation, not a tree.
EXACT_T8 = ("d", "full", "cooff")


def collect_optima():
    """Certified optimum per instance label (solver scale), from every certified run of
    Table D8 and of the exact Table D9 configurations; all of them must agree."""
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
        rec = load(path)
        if rec is None:
            continue
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
        "Every certified run of the distance-indexed solver in Table D8, and in the Table D9 "
        f"configurations that separate the four-point condition ({', '.join(EXACT_T8)}), gives "
        "an optimum; for each instance all of them agree to a relative "
        f"{OPT_REL_TOL:g} (asserted). Solver scale is ours times 2^(n+1). The optimum quoted "
        f"in {len(checked)} JSON records under results/ agrees with these values (asserted).",
        "",
        table(["instance", "n", "optimum (solver scale)", "optimum (our scale)",
               "certified runs agreeing"], rows)])


TABLE7 = ["Primates12", "M17", "M18", "20_euros2", "21_nucleic", "22_euros2", "23_euros2",
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


# Table D9 configurations, in the order of the cumulative ladder.
T8_INST = ["M17", "M18", "20_euros2", "20_rosids"]
T8_CFG = [
    ("a", "(a) base: no separated cuts; root LP only"),
    ("b", "(b) (a) + strong triangles on all triples (flag --str A); root LP only"),
    ("gki", "(gki) (b) + generalized Kraft (our min-cut family)"),
    ("c", "(c) (gki) + 2-K-split + circular order"),
    ("d", "(d) (c) + four-point"),
    ("full", "(full) the Table D8 configuration: strong triangle and four-point on the "
             "quartets containing leaf 1, generalized Kraft, weak four-point, 2-K-split, "
             "circular order"),
    ("cooff", "(cooff) (full) with circular-order cuts switched off"),
]


def synthetic_optimum(inst):
    """L* of a synthetic instance test_nN, from the enumeration of all trees in the Table D3
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
    out = ["## Theorem 5 and Table C1 (Section 6.3, Appendix C)", ""]
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


def sec_p6():
    r = load(RES / "p6_facets/summary.json")
    if r is None:
        return "## Facets of P_6 (Section 4)\n\npending"
    fam = "; ".join(f"{k}: {v['matched']} of {v['size']}" for k, v in r["families"].items())
    cuts = "; ".join(f"{k}-cuts: {v['size']} inequalities, {v['matched']} are facets, "
                     f"{v['trivial']} are trivial (implied by the Kraft equalities)"
                     for k, v in r["cuts"].items())
    hist = ", ".join(f"{k}: {v}" for k, v in r["support_histogram"].items())
    return "\n".join([
        "## Facets of P_6 (Section 4)", "",
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
        "## Conjecture 1: integral points of the membership model (Section 5.2)", "",
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
    out = ["## Crossing graph: edge clique covers (Section 5.1 and Appendix H)", "",
           table(["n", "nontrivial bipartitions", "crossing pairs", "maximal cliques",
                  "fractional minimum", "minimum edge clique cover", "subfamily of Appendix H",
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
        sec_conjecture1(),
        sec_clique(),
        sec_theorem5(),
    ]
    out.write_text("\n\n".join(parts) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
