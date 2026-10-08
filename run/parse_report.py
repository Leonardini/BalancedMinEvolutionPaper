"""Read one run directory of Catanzaro's solver.

    python parse_report.py RUNDIR [FIELD]

With FIELD, prints that field only (used by the drivers, e.g. `certified`); without it,
prints every field as tab-separated key=value pairs.
"""
import re
import sys
from pathlib import Path

# Columns of the solver's report (src/report.hpp), in order.
COLUMNS = ["instance", "n", "best_sol", "time", "node_best_sol", "first_root_bound",
           "last_root_bound", "best_dual_bound", "final_gap_pct", "nodes", "inf",
           "co_root", "co_search", "gki_root", "gki_search", "ehc_root", "ehc_search",
           "2k_root", "2k_search", "4pt_root", "4pt_search", "total_cuts"]
# Gurobi stops at its default MIPGap; "certified" is the paper's 0.0000% (4 decimals).
CERT_GAP_PCT = 0.5e-4


def read_run(rundir):
    rundir = Path(rundir)
    lines = (rundir / "report.txt").read_text().splitlines()
    if len(lines) != 2:
        raise ValueError(f"{rundir}/report.txt: expected header + 1 row, got {len(lines)} lines")
    vals = [v.strip() for v in lines[1].split("|")]
    if len(vals) != len(COLUMNS):
        raise ValueError(f"{rundir}/report.txt: {len(vals)} columns, expected {len(COLUMNS)}")
    rec = dict(zip(COLUMNS, vals))
    rec["certified"] = str(float(rec["final_gap_pct"]) <= CERT_GAP_PCT)
    m = re.search(r"(\d+)\s+maximum resident set size", (rundir / "time.txt").read_text())
    rec["peak_rss_mb"] = f"{int(m.group(1)) / 2**20:.0f}" if m else "NA"
    rec["gurobi_env"] = (rundir / "gurobi.env").read_text().replace("\n", ";")
    return rec


if __name__ == "__main__":
    rec = read_run(sys.argv[1])
    if len(sys.argv) > 2:
        print(rec[sys.argv[2]])
    else:
        print("\t".join(f"{k}={v}" for k, v in rec.items()))
