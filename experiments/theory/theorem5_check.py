"""Read one enumeration run (Theorem 5, Appendix C), record its counts as JSON, analyse
the dumped counterexamples where the run has any, and compare with the paper.

    python theorem5_check.py OUTDIR NAME N [backtrack options]

Expects OUTDIR/NAME.out (stdout of backtrack) and, where the run violates (10),
OUTDIR/NAME.dump. Writes OUTDIR/NAME.json.

What is compared, by run type:
  canonical (no option)    survivors = Table C1, none violating (10); row types (n >= 8)
  --full                   survivors = (2n-5)!!, none violating (10)
  --nocherry               no survivors (every matrix satisfying (4)-(9) has a cherry)
  --weak --full (n = 6)    285 survivors, 180 violating (10); the 180 form one orbit of
                           S_6, that of r, whose stabiliser has order 4
  --fpviol --t01 2         84 witnesses, all 6-cherry lifts, all in the class of r
"""
import json
import re
import sys
from math import factorial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plm  # noqa: E402
import published as P  # noqa: E402

SUMMARY = re.compile(r"^n=(\d+): survivors=(\d+) \((canonical|FULL)\)  non-tree=(\d+)  "
                     r"\[(\d+) nodes, (\d+)s, (\d+) row-types\]")


def double_factorial(k):
    p = 1
    while k > 1:
        p *= k
        k -= 2
    return p


def parse_summary(path):
    hits = [m for m in map(SUMMARY.match, path.read_text().splitlines()) if m]
    if len(hits) != 1:
        raise RuntimeError(f"{path}: expected one summary line, found {len(hits)}")
    n, surv, mode, viol, nodes, secs, types = hits[0].groups()
    return dict(n=int(n), survivors=int(surv), ordered=(mode == "canonical"),
                violating=int(viol), nodes=int(nodes), seconds=int(secs), row_types=int(types))


def classes_of_lifts(mats, triangle_slack):
    """Every matrix must be a counterexample whose cherries partition the leaves;
    returns the set of canonical forms of the reduced matrices."""
    forms = set()
    for idx, tau in enumerate(mats):
        if not plm.is_counterexample(tau, triangle_slack):
            raise AssertionError(f"dumped matrix {idx} does not satisfy (4)-(9) and violate (10)")
        r, ch = plm.reduce_cherries(tau)
        forms.add(plm.canon(r)[0])
    return forms


def main():
    outdir, name, n = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
    opts = sys.argv[4:]
    rec = parse_summary(outdir / f"{name}.out")
    rec.update(name=name, options=opts)
    if rec["n"] != n:
        raise AssertionError(f"summary is for n={rec['n']}, expected {n}")
    full, weak = "--full" in opts, "--weak" in opts
    nocherry, fpviol = "--nocherry" in opts, "--fpviol" in opts
    t01 = int(opts[opts.index("--t01") + 1]) if "--t01" in opts else None
    checks = []

    def check(label, computed, expected):
        checks.append(dict(label=label, computed=computed, expected=expected,
                           ok=computed == expected))

    mats = []
    if rec["violating"]:
        mats = plm.read_dump(outdir / f"{name}.dump")
        check("dumped matrices = violating survivors", len(mats), rec["violating"])
        check("dumped matrices pairwise distinct", len({str(M) for M in mats}), len(mats))
    r6_form = plm.canon(plm.R6)[0]

    if weak:
        if not (full and n == 6):
            raise SystemExit("--weak is analysed only with --full at n = 6")
        forms = set()
        for idx, r in enumerate(mats):
            if not plm.is_counterexample(r, triangle_slack=0):
                raise AssertionError(f"dumped 6-node matrix {idx} is not a weak (4)-(9) non-tree")
            forms.add(plm.canon(r)[0])
        stab = plm.stabiliser_order(plm.R6)
        rec.update(weak_orbits=len(forms), r_stabiliser=stab,
                   r_orbit_size=factorial(n) // stab, orbit_is_r=(forms == {r6_form}))
        check("weak survivors", rec["survivors"], P.WEAK6_SURVIVORS)
        check("weak survivors satisfying (10) = (2n-5)!!",
              rec["survivors"] - rec["violating"], double_factorial(2 * n - 5))
        check("weak non-trees", rec["violating"], P.WEAK6_NON_TREES)
        check("S_6 orbits among the non-trees", len(forms), P.WEAK6_ORBITS)
        check("the orbit is that of r", forms == {r6_form}, True)
        check("stabiliser order of r", stab, P.WEAK6_STABILISER)
        check("orbit size n!/|stabiliser| = non-trees", factorial(n) // stab, rec["violating"])
    elif nocherry:
        check("survivors without a cherry", rec["survivors"], 0)
    elif fpviol:
        if t01 != 2:
            raise SystemExit("--fpviol is analysed only with --t01 2 (cherry at {1,2})")
        forms = classes_of_lifts(mats, 2)
        rec.update(classes=len(forms), class_is_lift_of_r=(forms == {r6_form}))
        check("witnesses", rec["violating"], P.CHERRY_FIXED_WITNESSES)
        check("every survivor violates (10)", rec["survivors"], rec["violating"])
        check("relabelling classes", len(forms), P.CHERRY_FIXED_CLASSES)
        check("the class is the cherry lift of r", forms == {r6_form}, True)
    elif full:
        check("labelled survivors = (2n-5)!!", rec["survivors"], double_factorial(2 * n - 5))
        check("violating (10)", rec["violating"], 0)
    else:
        check("canonical survivors (Table C1)", rec["survivors"], P.CANONICAL_SURVIVORS[n])
        check("violating (10)", rec["violating"], 0)
    if n in P.ROW_TYPES and not nocherry:
        check("row types", rec["row_types"], P.ROW_TYPES[n])

    rec["checks"] = checks
    (outdir / f"{name}.json").write_text(json.dumps(rec, indent=1) + "\n")
    for c in checks:
        print(f"{'ok  ' if c['ok'] else 'FAIL'} {c['label']}: computed {c['computed']}, "
              f"paper {c['expected']}")
    bad = [c["label"] for c in checks if not c["ok"]]
    if bad:
        raise AssertionError(f"{name}: disagrees with the paper on: {', '.join(bad)}")


if __name__ == "__main__":
    main()
