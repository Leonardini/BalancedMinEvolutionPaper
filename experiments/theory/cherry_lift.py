"""The counterexample of Section 6.3: the cherry lift of the 6-node matrix r.

    python cherry_lift.py OUT.json

Builds tau from r (leaves 2a-1, 2a form cherry a), checks (4)-(9) exactly, lists the
quartets that violate (10), and checks the quartet {1,2,3,5} named in the text.
Also writes the 12 x 12 matrix next to OUT.json as cherry_lift_tau.txt.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plm  # noqa: E402
import published as P  # noqa: E402

out = Path(sys.argv[1])
out.parent.mkdir(parents=True, exist_ok=True)
tau = plm.cherry_lift(plm.R6)
cond = plm.conditions(tau)
viol = plm.quartet_violations(tau)
named = (0, 1, 2, 4)                       # {1,2,3,5} in the paper's 1-based labels
named_sums = dict(viol).get(named)          # None if (10) holds on this quartet
rec = dict(n=len(tau), conditions=cond, violating_quartets=len(viol),
           quartet_1235_pair_sums=named_sums,
           violating_quartets_list=[[[x + 1 for x in q], s] for q, s in viol])
out.write_text(json.dumps(rec, indent=1) + "\n")
(out.parent / "cherry_lift_tau.txt").write_text(
    "\n".join(" ".join(str(x) for x in row) for row in tau) + "\n")
print(f"cherry lift, n={len(tau)}: " + ", ".join(f"{k} {'holds' if v else 'fails'}"
                                                  for k, v in cond.items()))
print(f"quartets violating (10): {len(viol)}; quartet {{1,2,3,5}} pair sums {named_sums}")
assert all(cond[k] for k in ("(4)", "(5)", "(6)", "(7)", "(8)", "(9)")), cond
assert named_sums == P.CHERRY_LIFT_QUARTET_1235_SUMS, named_sums
assert len(viol) == P.CHERRY_LIFT_VIOLATING_QUARTETS, len(viol)
