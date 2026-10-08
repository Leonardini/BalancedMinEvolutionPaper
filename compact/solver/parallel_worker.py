"""The worker side of the parallel search (bnb_parallel.py).

A worker process has its own CPLEX model and MOSEK task and processes the nodes it is given
exactly as the sequential search does (bnb_balanced.branch_node). Besides the children and
any better incumbent, it returns the rows its separators added to its model, which are all
globally valid cuts, so that the coordinator can pass them to the other workers.
"""
import cplex

from .bnb_balanced import CUT_KEYS, EVENT_KEYS, _incumbent_splits, _solve_node, branch_node, new_lp_model
from .exact_node import new_counts as new_exact_counts


class _Node:
    """A worker's state: its own LP model and conic task."""

    def __init__(self, D):
        self.D, self.n = D, D.shape[0]
        self.c, self.pair_to_idx, self.pairs = new_lp_model(D)
        self.cut_counts = {k: 0 for k in CUT_KEYS}
        self.exact_counts = new_exact_counts()
        self.inc_key, self.inc_splits = None, None

    def add_rows(self, rows):
        if rows:
            self.c.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind=list(ind), val=list(val)) for ind, val, _, _ in rows],
                senses=[s for _, _, s, _ in rows], rhs=[r for _, _, _, r in rows])

    def _rows_since(self, k0):
        k1 = self.c.linear_constraints.get_num()
        if k1 == k0:
            return []
        idx = list(range(k0, k1))
        sps = self.c.linear_constraints.get_rows(idx)
        return [(tuple(sp.ind), tuple(sp.val), s, r) for sp, s, r in
                zip(sps, self.c.linear_constraints.get_senses(idx), self.c.linear_constraints.get_rhs(idx))]

    def _counts(self):
        return dict(self.cut_counts), dict(self.exact_counts)

    def root(self, deadline):
        k0 = self.c.linear_constraints.get_num()
        bound, w, _ = _solve_node(self.c, self.D, self.pair_to_idx, self.pairs, len(self.pairs),
                                  [], [], self.cut_counts, self.exact_counts, deadline=deadline)
        return dict(bound=bound, w=w, rows=self._rows_since(k0), counts=self._counts())

    def process(self, node_no, item, from_dive, inc_obj, inc_w, deadline):
        """Process one taken node (bnb_balanced.branch_node), returning what the coordinator
        needs."""
        node_bound, _node_id, forced, forbidden, w_vals = item
        k0 = self.c.linear_constraints.get_num()
        events = {k: 0 for k in EVENT_KEYS}
        heuristic = {'lp_cherries': 0, 'nni': 0}
        key = float(inc_obj)
        if key != self.inc_key:
            self.inc_splits, self.inc_key = _incumbent_splits(inc_w, self.n), key
        res = branch_node(self.c, self.D, self.pair_to_idx, self.pairs, node_bound, forced,
                          forbidden, w_vals, inc_obj, inc_w, self.inc_splits, self.cut_counts,
                          self.exact_counts, events, heuristic, deadline)
        res["record"].update(node=node_no, origin=None, dived=from_dive)
        new_obj, new_w = res["inc"]
        return dict(children=res["children"], record=res["record"],
                    inc=(new_obj, new_w) if new_obj < inc_obj else None,
                    infeasible_prunes=res["infeasible_prunes"],
                    unresolved=[(node_no, node_bound)] if res["unresolved"] else [],
                    incomplete=res["unresolved"], events=events, heuristic=heuristic,
                    rows=self._rows_since(k0), counts=self._counts())


def _worker_main(conn, D):
    node = _Node(D)
    while True:
        msg = conn.recv()
        if msg[0] == "stop":
            conn.close()
            return
        if msg[0] == "root":
            conn.send(node.root(msg[1]))
        elif msg[0] == "process":
            node.add_rows(msg[1])
            conn.send(node.process(*msg[2:]))


def _merge(total, delta, before):
    """Add to total the increase of the worker's cumulative counts since before."""
    for k, v in delta.items():
        total[k] = total.get(k, 0) + v - before.get(k, 0)
