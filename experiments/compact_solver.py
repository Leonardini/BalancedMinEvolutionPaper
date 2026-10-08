"""One run of the compact solver (Section 8.1).

    BME_THREADS=1 BME_WORKERS=10 BME_SEED=1 python compact_solver.py INSTANCE CAP OUT.json [MAX_NODES]

The branch and bound of Section 2.4 and Appendix F: balanced branching, the manifold
constraint imposed exactly at every node by the conic solver, the cluster equalities at
every node. With BME_WORKERS=k it runs the parallel search (solver/bnb_parallel.py) with k
worker processes, each with BME_THREADS threads (the timed solves use k = 10 and one thread
each); without it, the sequential search (solver/bnb_balanced.py), which is the parallel
search with one worker. BME_SEED is the LP solver's random seed. The search stops after CAP
seconds, and with MAX_NODES after that many nodes.

Writes OUT.json: the best value found, the lower bound, whether it certified, the node
count, the time, the root bound, and the search's instrumentation (events, the node log,
the bound over time). Progress lines go to stderr every BME_PROGRESS seconds (default 60;
0 turns them off).
"""
import json
import os
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "compact"))
from solver.conic import CONIC_SOLVER  # noqa: E402
from solver.parse_matrix import parse_matrix  # noqa: E402


def main():
    if len(sys.argv) not in (4, 5):
        sys.exit(__doc__)
    inst, cap, out = sys.argv[1], float(sys.argv[2]), Path(sys.argv[3])
    max_nodes = int(sys.argv[4]) if len(sys.argv) > 4 else None
    D = parse_matrix(inst)
    t0 = time.time()
    workers = int(os.environ.get("BME_WORKERS", "0"))
    if workers:
        from solver.bnb_parallel import solve_bme_bnb_parallel  # noqa: E402
        r = solve_bme_bnb_parallel(D, workers, time_limit=cap, max_nodes=max_nodes)
    else:
        from solver.bnb_balanced import solve_bme_bnb  # noqa: E402
        r = solve_bme_bnb(D, time_limit=cap, max_nodes=max_nodes)
    rec = {k: v for k, v in r.items() if k != "w"}
    rec.update(conic_solver=CONIC_SOLVER, rule="balanced-exact", instance=inst, n=int(D.shape[0]),
               cap=cap, max_nodes=max_nodes, wall_seconds=time.time() - t0,
               peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1, default=float) + "\n")
    print(json.dumps(rec, default=float))


if __name__ == "__main__":
    main()
