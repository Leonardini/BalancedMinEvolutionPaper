"""Progress lines for a branch-and-bound run, written to stderr (no effect on the search).

The cadence is in seconds, so it adapts to the per-node cost of the instance:
    BME_PROGRESS  seconds between lines (default 60); 0 turns the lines off
Each line gives the elapsed time, nodes solved, open nodes, the global lower bound,
the incumbent and the relative gap, so a run's state can be read with `tail` at any time.
"""
import os
import sys
import time


class Progress:
    def __init__(self, t0, label):
        self.t0 = t0
        self.label = label
        self.every = float(os.environ.get("BME_PROGRESS", "60"))
        self.last = t0

    def line(self, n_nodes, n_open, lb, inc, note=""):
        gap = (inc - lb) / abs(inc) if inc else float("nan")
        self.last = time.time()
        print(f"{time.strftime('%F %T')} {self.label} {time.time() - self.t0:8.0f}s "
              f"nodes {n_nodes} open {n_open} lb {lb:.10g} inc {inc:.10g} "
              f"gap {100 * gap:.4f}%{' ' + note if note else ''}",
              file=sys.stderr, flush=True)

    def tick(self, n_nodes, n_open, lb, inc):
        """Print a line if the cadence has elapsed since the last one."""
        if self.every <= 0:
            return
        if time.time() - self.last >= self.every:
            self.line(n_nodes, n_open, lb, inc)
