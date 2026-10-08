"""The parallel search: the search of bnb_balanced.solve_bme_bnb with k worker processes
(parallel_worker.py), each processing one node at a time.

As soon as any worker returns a node, the coordinator merges its result (the incumbent, the
cuts it found, which are queued for every other worker's next node, and its children) and
hands that worker a new node. The new node is the worker's own keeping child when it
produced one (the sequential search explores that child next), otherwise the best open node.

The global lower bound is the minimum over the open nodes, the nodes being processed and the
unresolved nodes. The search is certified only once no node is being processed and the best
open bound has reached the incumbent; a child's bound is at least its parent's, so a node
still being processed cannot produce a child that undercuts it.

The order in which workers return depends on timing, so the search is not deterministic for
k > 1. With k = 1 it is the sequential search.
"""
import heapq
import itertools
import multiprocessing as mp
import time
from multiprocessing.connection import wait

from .bnb_balanced import CUT_KEYS, EPS, EVENT_KEYS
from .parallel_worker import _merge, _worker_main
from .exact_node import new_counts as new_exact_counts
from .fastme_incumbent import initial_incumbent
from .nni_incumbent import w_from_tree
from .progress import Progress


def solve_bme_bnb_parallel(D, workers, time_limit=300.0, max_nodes=None):
    """The search of bnb_balanced.solve_bme_bnb with up to `workers` nodes processed at once,
    one per worker process (each should run its LP solver on one thread: BME_THREADS=1), a
    worker getting a new node as soon as it returns one. Returns what
    bnb_balanced.solve_bme_bnb returns, with 'workers' and 'worker_utilisation'."""
    n = D.shape[0]
    t0 = time.time()
    deadline = t0 + time_limit
    ctx = mp.get_context("spawn")
    pipes, procs = [], []
    for _ in range(workers):
        a, b = ctx.Pipe()
        p = ctx.Process(target=_worker_main,
                        args=(b, D),
                        daemon=True)
        p.start()
        pipes.append(a)
        procs.append(p)
    try:
        return _search(D, n, t0, deadline, time_limit, max_nodes, pipes, workers)
    finally:
        for a in pipes:
            a.send(("stop",))
        for p in procs:
            p.join(timeout=30)


def _search(D, n, t0, deadline, time_limit, max_nodes, pipes, workers):
    inc = initial_incumbent(D, n)
    inc_obj = inc['objective']
    inc_w = w_from_tree(inc['tree'], n)
    cut_counts = {k: 0 for k in CUT_KEYS}
    exact_counts = new_exact_counts()
    seen_counts = [({}, {}) for _ in range(workers)]   # each worker's cumulative counts so far

    def take_counts(i, counts):
        cc, ec = counts
        _merge(cut_counts, cc, seen_counts[i][0])
        _merge(exact_counts, ec, seen_counts[i][1])
        seen_counts[i] = (cc, ec)

    # Rows each worker still has to add (the cuts the others found), and the rows each
    # worker's model holds or will hold, so that no worker receives a row twice.
    pending = [[] for _ in range(workers)]
    has = [set() for _ in range(workers)]

    def share(i, rows):
        """rows: the rows worker i added while processing its last node (already in its model)."""
        has[i].update(rows)
        for r in rows:
            for j in range(workers):
                if j != i and r not in has[j]:
                    pending[j].append(r)
                    has[j].add(r)

    pipes[0].send(("root", deadline))
    root = pipes[0].recv()
    take_counts(0, root["counts"])
    share(0, root["rows"])
    bound, w_vals = root["bound"], root["w"]
    if w_vals is None:
        return {'objective': inc_obj, 'lb': float('-inf'), 'gap': float('nan'), 'certified': False,
                'n_nodes': 0, 'seconds': time.time() - t0, 'cut_counts': cut_counts, 'w': inc_w}
    counter = itertools.count()
    frontier = [(bound, next(counter), [], [], w_vals)]
    own_dive = [None] * workers      # a worker's keeping child, to be its next node
    in_flight = {}                   # worker -> (item, from_dive, node number)
    progress = Progress(t0, 'balanced-exact')
    progress.line(0, 1, bound, inc_obj, 'root')
    events = {k: 0 for k in EVENT_KEYS}
    heuristic = {'lp_cherries': 0, 'nni': 0}
    trajectory, infeasible_prunes, unresolved, node_log = [], [], [], []
    trajectory_seconds = []   # (seconds, lower bound counting unresolved nodes) per node
    origin = {}
    n_dispatched = 0
    proven, incomplete = False, False
    busy_seconds = 0.0        # summed over workers: time with a node in hand
    started = {}

    def global_bound():
        return min(([frontier[0][0]] if frontier else [])
                   + [d[0] for d in own_dive if d is not None]
                   + [it[0] for it, _, _ in in_flight.values()], default=inc_obj)

    def next_item(i):
        """The node worker i should process next, or None. Pruned dives are dropped."""
        d, own_dive[i] = own_dive[i], None
        if d is not None:
            events["pops"] += 1
            if d[0] < inc_obj - EPS:
                return d, True
            events["dive_pruned_bound"] += 1
        if not frontier:
            return None
        if frontier[0][0] >= inc_obj - EPS:
            # Best first: every queued node is bounded at the incumbent too.
            return None
        events["pops"] += 1
        return heapq.heappop(frontier), False

    def dispatch(i):
        nonlocal n_dispatched
        if time.time() - t0 > time_limit or (max_nodes and n_dispatched >= max_nodes):
            return False
        nxt = next_item(i)
        if nxt is None:
            return False
        item, from_dive = nxt
        n_dispatched += 1
        gb = min(global_bound(), item[0])
        trajectory.append((n_dispatched - 1, gb, inc_obj))
        trajectory_seconds.append((time.time() - t0, min([gb] + [u[1] for u in unresolved])))
        progress.tick(n_dispatched - 1, len(frontier) + len(in_flight) + 1,
                      min([gb] + [u[1] for u in unresolved]), inc_obj)
        pipes[i].send(("process", pending[i], n_dispatched, item, from_dive, inc_obj, inc_w,
                       deadline))
        pending[i] = []
        in_flight[i] = (item, from_dive, n_dispatched)
        started[i] = time.time()
        return True

    idle = list(range(workers))
    while True:
        # Hand a node to every idle worker that can get one (lowest index first).
        still_idle = []
        for i in idle:
            if not dispatch(i):
                still_idle.append(i)
        idle = still_idle
        if not in_flight:
            break
        ready = wait([pipes[i] for i in in_flight])
        for conn in ready:
            i = pipes.index(conn)
            res = conn.recv()
            item, _, _ = in_flight.pop(i)
            busy_seconds += time.time() - started.pop(i)
            share(i, res["rows"])
            take_counts(i, res["counts"])
            for k, v in res["events"].items():
                events[k] += v
            for k, v in res["heuristic"].items():
                heuristic[k] += v
            if res["inc"] is not None and res["inc"][0] < inc_obj - EPS:
                inc_obj, inc_w = res["inc"]
            rec = res["record"]
            rec["origin"] = origin.pop(item[1], "root")
            node_log.append(rec)
            infeasible_prunes += res["infeasible_prunes"]
            unresolved += res["unresolved"]
            incomplete |= res["incomplete"]
            for cb, cf, cfb, cw, kind, keeps, branch_kind in res["children"]:
                if cb >= inc_obj - EPS:
                    continue
                cid = next(counter)
                origin[cid] = dict(parent=rec["node"], branch=branch_kind, child=kind, keeps=keeps)
                if keeps and own_dive[i] is None:
                    own_dive[i] = (cb, cid, cf, cfb, cw)
                else:
                    heapq.heappush(frontier, (cb, cid, cf, cfb, cw))
            idle.append(i)
        idle.sort()

    # A dive left undispatched (time or node limit) is still open.
    for d in own_dive:
        if d is not None:
            heapq.heappush(frontier, d)
    elapsed = time.time() - t0
    if frontier and frontier[0][0] >= inc_obj - EPS and not in_flight:
        events["stop_bound_reached"] += 1
        proven = True
    open_bounds = [it[0] for it in frontier]
    if (proven or not open_bounds) and not incomplete:
        lb, certified = inc_obj, True
    else:
        lb = min(open_bounds + [u[1] for u in unresolved], default=inc_obj)
        certified = False
    gap = max(0.0, (inc_obj - lb) / inc_obj) if inc_obj > 0 else 0.0
    progress.line(n_dispatched, len(frontier), lb, inc_obj,
                  'certified' if certified else 'stopped')
    return {
        'objective': inc_obj, 'lb': lb, 'gap': gap, 'certified': certified, 'n_nodes': n_dispatched,
        'seconds': elapsed, 'cut_counts': cut_counts, 'w': inc_w, 'root_lb': bound,
        'trajectory': trajectory, 'trajectory_seconds': trajectory_seconds, 'unresolved': unresolved,
        'heuristic_improvements': heuristic, 'exact_counts': exact_counts,
        'infeasible_prunes': infeasible_prunes, 'events': events,
        'node_log': node_log, 'workers': workers,
        'worker_utilisation': busy_seconds / (workers * elapsed) if elapsed > 0 else None,
    }
