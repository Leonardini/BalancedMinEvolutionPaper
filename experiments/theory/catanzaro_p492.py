"""The 12 x 12 matrix of Catanzaro et al. (2026, p. 492) is a relabelling of the cherry
lift of Section 6.3.

    python catanzaro_p492.py MATRIX.txt OUT.json

Checks (4)-(9) and the failure of (10) on the given matrix, reduces it by its cherries,
and constructs an explicit leaf bijection pi with theirs[pi(p)][pi(q)] = ours[p][q],
which is then verified entry by entry.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plm  # noqa: E402

theirs = plm.read_matrix(sys.argv[1])
out = Path(sys.argv[2])
out.parent.mkdir(parents=True, exist_ok=True)
ours = plm.cherry_lift(plm.R6)
n = len(ours)
if len(theirs) != n:
    raise SystemExit(f"expected a {n} x {n} matrix, read {len(theirs)} rows")

cond = plm.conditions(theirs)
viol = plm.quartet_violations(theirs)
r_theirs, ch_theirs = plm.reduce_cherries(theirs)
form_ours, perms_ours = plm.canon(plm.R6)
form_theirs, perms_theirs = plm.canon(r_theirs)
iso = form_ours == form_theirs
pi = None
if iso:
    # R6[p1[a]][p1[b]] = r_theirs[p2[a]][p2[b]], so node x of R6 maps to p2[p1^-1[x]].
    p1, p2 = perms_ours[0], perms_theirs[0]
    inv1 = {x: a for a, x in enumerate(p1)}
    sigma = [p2[inv1[x]] for x in range(len(plm.R6))]
    pi = [None] * n
    for a, b in enumerate(sigma):                  # our cherry a is leaves {2a, 2a+1}
        pi[2 * a], pi[2 * a + 1] = ch_theirs[b]
    assert sorted(pi) == list(range(n))
    assert all(theirs[pi[p]][pi[q]] == ours[p][q] for p in range(n) for q in range(n))
rec = dict(conditions=cond, violating_quartets=len(viol), cherries=[[i + 1, j + 1] for i, j in ch_theirs],
           reduced=r_theirs, isomorphic_to_cherry_lift=iso,
           leaf_map_ours_to_theirs=None if pi is None else {p + 1: pi[p] + 1 for p in range(n)})
out.write_text(json.dumps(rec, indent=1) + "\n")
print(", ".join(f"{k} {'holds' if v else 'fails'}" for k, v in cond.items()))
print(f"violating quartets: {len(viol)}; cherries: {rec['cherries']}")
print(f"isomorphic to the cherry lift of r: {iso}")
if pi is not None:
    print("leaf map (ours -> theirs, 1-based): "
          + " ".join(f"{p}->{q}" for p, q in rec["leaf_map_ours_to_theirs"].items()))
assert all(cond[k] for k in ("(4)", "(5)", "(6)", "(7)", "(8)", "(9)")) and not cond["(10)"], cond
assert iso
