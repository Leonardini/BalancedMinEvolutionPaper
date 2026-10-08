"""Explicit best-first branch-and-bound for BME.

CPLEX is used only as the per-node LP solver.  A single persistent model
accumulates all globally-valid cuts (min-cut / PM / F6 / F7) across every node
solve, so each cut is separated at most once and then tightens the whole
search tree.  Only the node-local forced/forbidden split constraints are added
before a node solve and removed afterward.

Node = (forced, forbidden): lists of canonical bipartitions.
  forced S    -> W[S] == 1/2   (S is a split of every tree in this subtree)
  forbidden S -> W[S] >= 3/4    (S is not a split)

Every binary tree has W[S] = 1/2 (S a split) or W[S] >= 3/4 (S not a split),
so this dichotomy partitions the trees while excluding the current fractional
LP point (which has W[S] in (1/2, 3/4)).  Hence branching makes strict
progress and the search is complete.

Branching is on the most balanced fractional split of the current incumbent,
exploring the child that forbids it first; see branch_node. Each node's bound imposes the
manifold constraint exactly (exact_node.py).

This module holds the sequential search and the node routines that the parallel search
(bnb_parallel.py, parallel_worker.py) shares with it; with one worker the parallel search
is this one.
"""

import heapq
import itertools
import time

import cplex
import numpy as np

from .base_model import build_base_model
from .cut_loop import separate_cuts
from .exact_node import exact_refine
from .exact_node import new_counts as new_exact_counts
from .face import _cut, contract_committed, incumbent_search, settle_face
from .fastme_incumbent import initial_incumbent
from .nni_incumbent import w_from_tree
from .progress import Progress
from .w_space import (
    bipartition_key,
    build_W,
    cut_row_sparse,
    extract_tree_cherry,
    find_half_cuts,
    pick_branching_cut_fast,
)

EPS = 1e-6


def _crosses(A, B, full):
    """True if bipartitions A and B (sets of leaves, either side) cross: no tree has both."""
    return bool(A & B) and bool(A - B) and bool(B - A) and bool(full - (A | B))


def _cluster_eq_rows(forced, n, pair_to_idx):
    """The cluster equalities of a node as sparse rows (index list, value list, right-hand
    side). For each cluster Z left by contracting the committed cherries (leaf i lying e_i
    edges below Z's root), every tree of the node has w_ic = 2^-(e_i + d(Z, c)) for c outside
    Z, so 2^e_i w_ic = 2^e_r w_rc for a fixed leaf r of Z: (|Z| - 1)(n - |Z|) rows per
    cluster; and w_ij = 2^-p_ij for the fixed path length p_ij between two leaves of Z:
    |Z|(|Z| - 1)/2 rows. With the second kind a child's rows imply its parent's, so the
    child's relaxation lies inside its parent's."""
    clusters, e, within, _consumed = contract_committed(forced, n)
    rows = []
    for Z in clusters:
        if len(Z) < 2:
            continue
        r, *others = sorted(Z)
        for i in others:
            for c in range(1, n + 1):
                if c not in Z:
                    rows.append(([pair_to_idx[(min(i, c), max(i, c))], pair_to_idx[(min(r, c), max(r, c))]],
                                 [2.0 ** e[i], -2.0 ** e[r]], 0.0))
        for i, j in itertools.combinations(sorted(Z), 2):
            rows.append(([pair_to_idx[(i, j)]], [1.0], 2.0 ** -within[(i, j)]))
    return rows


def _cluster_spread(W, forced, n):
    """How far the point is from the contracted problem. For each cluster Z left by
    contracting the committed cherries (leaf i lying e_i edges below Z's root) and each leaf c
    outside Z, every tree of the node has 2^e_i * w_ic equal over i in Z, and w_ij = 2^-p_ij
    for i, j in Z. Returns the largest deviation from these (the spread of 2^e_i w_ic over i,
    or |w_ij - 2^-p_ij|), the number of (Z, c) and pairs within Z whose deviation exceeds
    1e-7, and the number of them."""
    clusters, e, within, _consumed = contract_committed(forced, n)
    worst, violated, total = 0.0, 0, 0
    for Z in clusters:
        if len(Z) < 2:
            continue
        for c in range(1, n + 1):
            if c in Z:
                continue
            v = [2.0 ** e[i] * W[i - 1, c - 1] for i in Z]
            spread = max(v) - min(v)
            worst = max(worst, spread)
            violated += spread > 1e-7
            total += 1
        for i, j in itertools.combinations(sorted(Z), 2):
            dev = abs(W[i - 1, j - 1] - 2.0 ** -within[(i, j)])
            worst = max(worst, dev)
            violated += dev > 1e-7
            total += 1
    return dict(max=worst, violated=violated, rows=total)


def _incumbent_splits(inc_w, n):
    """The n-3 internal splits of the incumbent tree (its 1/2-cuts).  ``inc_w`` is
    already the n×n W matrix (w_from_tree / extract_tree_cherry output)."""
    return find_half_cuts(inc_w, n)


def pick_balanced_incumbent_split(W, n, inc_splits, forced, forbidden, tol=EPS):
    """Certification-driven branching among the incumbent's splits whose current
    LP cut value is FRACTIONAL (in (1/2, 3/4)) — restricting to fractional values
    guarantees both children exclude the current LP point (progress).  Returns the
    most BALANCED such split (min ||S|-|Sc||), tie-broken by most-fractional, or
    None (caller falls back to the generic rule).  A balanced cut spans ~n^2/4
    cross-pairs (vs ~2n for a cherry), the biggest cut-weight move.
    """
    full = frozenset(range(1, n + 1))
    excluded = {bipartition_key(s, n) for s in forced} | \
               {bipartition_key(s, n) for s in forbidden}
    best = None  # (key, S);  key minimised
    for S in inc_splits:
        if bipartition_key(S, n) in excluded:
            continue
        Sc = full - S
        cutval = sum(W[i - 1, j - 1] for i in S for j in Sc)
        if 0.5 + tol < cutval < 0.75 - tol:
            key = (abs(2 * len(S) - n), abs(cutval - 0.625))
            if best is None or key < best[0]:
                best = (key, S)
    return best[1] if best is not None else None


def _splits_from_edges_fast(edges, m):
    """Leaf-bipartitions induced by each edge of a tree on leaves 1..m (internal
    nodes negative), by depth-first search."""
    adj = {}
    for u, v in edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    all_leaves = frozenset(range(1, m + 1))
    splits = []
    for u, v in edges:
        seen = {u}
        stack = [u]
        side = []
        while stack:
            x = stack.pop()
            if x > 0:
                side.append(x)
            for y in adj[x]:
                if (x == u and y == v) or (x == v and y == u):
                    continue  # do not cross the cut edge
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        lu = frozenset(side)
        lv = all_leaves - lu
        if len(lu) >= 2 and len(lv) >= 2:
            if len(lu) < len(lv) or (len(lu) == len(lv) and 1 in lu):
                splits.append(lu)
            else:
                splits.append(lv)
    return splits


# The most manifold tangent cuts a root cut loop adds. The branch and bound imposes the
# manifold exactly instead and adds none; some bound experiments use tangents with this cap.
MANIFOLD_ROOT_CAP = 300

def _solve_node(c, D, pair_to_idx, pairs, np_pairs, forced, forbidden,
                cut_counts, exact_counts, deadline=None):
    """Solve one B&B node's LP on the persistent model.

    Adds the node's forced (W[S]=1/2) and forbidden (W[S]>=3/4) split
    constraints, runs cut separation (separated cuts are RETAINED in c — they
    are globally valid), imposes the manifold exactly (exact_refine), reads the bound and
    w-vector, then removes ONLY the node-local constraints: the forced and forbidden splits
    and, at a node other than the root, its cluster equalities (_cluster_eq_rows).

    Returns (bound, w_vals, decided).  w_vals is None when no optimal LP bound
    was obtained (either the node is infeasible or a solve aborted on the time
    cap); ``decided`` is False ONLY in the latter (aborted, status-unknown)
    case, signalling that the node must not be treated as pruned/infeasible.
    """
    n = D.shape[0]
    base_rows = c.linear_constraints.get_num()

    for S in forced:
        idx, vals = cut_row_sparse(S, n, pair_to_idx)
        c.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=idx, val=vals)], senses=["E"], rhs=[0.5])
    for S in forbidden:
        idx, vals = cut_row_sparse(S, n, pair_to_idx)
        c.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=idx, val=vals)], senses=["G"], rhs=[0.75])
    eq_rows = _cluster_eq_rows(forced, n, pair_to_idx)
    if eq_rows:
        c.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=idx, val=vals) for idx, vals, _ in eq_rows],
            senses=["E"] * len(eq_rows), rhs=[rhs for _, _, rhs in eq_rows])
    n_node_rows = c.linear_constraints.get_num() - base_rows

    bound, w_vals, _, decided = separate_cuts(
        c, D, pair_to_idx, pairs, np_pairs, cut_counts, deadline=deadline,
        manifold_cap=0, lean=True)
    if w_vals is not None:
        # The node's bound with the manifold imposed exactly.
        bound, w_vals = exact_refine(c, D, pair_to_idx, pairs, np_pairs, cut_counts, bound,
                                     w_vals, deadline, exact_counts,
                                     node_block=(base_rows, n_node_rows))

    # Remove the node-local block [base_rows, base_rows + n_node_rows).  Cuts
    # added by separate_cuts sit at higher indices and are
    # left in place (CPLEX shifts them down to fill the gap, but they remain).
    if n_node_rows > 0:
        c.linear_constraints.delete(base_rows, base_rows + n_node_rows - 1)

    return bound, w_vals, decided


def new_lp_model(D):
    """The node LP: the base model of build_base_model with its variables continuous,
    declared an LP so that CPLEX solves each node by its simplex, with the settings every
    node solve uses. Returns (c, pair_to_idx, pairs)."""
    c, pair_to_idx, pairs = build_base_model(D, verbose=False)
    c.variables.set_types(len(pairs), c.variables.type.continuous)
    c.set_problem_type(c.problem_type.LP)
    # The cut families introduce coefficients up to 2^{n-4}, giving the LP a
    # very large dynamic range; numerical emphasis keeps deep-node solves stable.
    c.parameters.emphasis.numerical.set(1)
    # Dual simplex: warm-starts well after adding a cut and honours the (soft)
    # time limit far better than barrier, which can overshoot badly at n>=21.
    c.parameters.lpmethod.set(c.parameters.lpmethod.values.dual)
    # The tightest feasibility and optimality tolerances CPLEX allows, so that a cut the
    # separators find violated (by more than EPS = 1e-7) is enforced by the next solve.
    c.parameters.simplex.tolerances.feasibility.set(1e-9)
    c.parameters.simplex.tolerances.optimality.set(1e-9)
    c.set_log_stream(None)
    c.set_results_stream(None)
    c.set_warning_stream(None)
    return c, pair_to_idx, pairs


CUT_KEYS = ('mincut', 'pm', 'f6', 'f7', 'f4', 'f5', 'p50', 'p53', 'p54', 'manifold',
            'crossing', 'bal_branch', 'fb_prune', 'fb_open')
EVENT_KEYS = (
    "pops", "stop_bound_reached", "tree_at_node", "branch_balanced", "branch_fallback",
    "face_tree", "face_empty", "face_branch", "face_cherries", "S_crosses_committed",
    "commit_infeasible", "commit_infeasible_S_crosses", "commit_pruned_bound",
    "commit_pushed", "forbid_infeasible", "forbid_pruned_bound", "forbid_pushed",
    "child_undecided", "child_inherited", "dive_pruned_bound")


def branch_node(c, D, pair_to_idx, pairs, node_bound, forced, forbidden, w_vals, inc_obj,
                inc_w, inc_splits, cut_counts, exact_counts, events, heuristic, deadline):
    """Process one node taken from the queue: recognise a tree, or choose the branching and
    solve the children. Shared by the sequential and the parallel search.

    Returns a dict: inc (the incumbent (objective, w) after the node), record (the node's
    log entry, without its origin), children (those bounded below the incumbent, as
    (bound, forced, forbidden, w, kind, keeps, branch_kind)), infeasible_prunes, unresolved
    (True if a child's solve was aborted: its status is unknown).

    A keeping child, one whose new split constraint the parent's point already satisfies, is
    not solved when the point also satisfies its cluster equalities: the point is optimal
    for the parent's (larger) problem, so it is optimal for the child, which inherits the
    parent's bound and point. The caller explores it next (a dive), so the face is contracted
    until it can be enumerated."""
    n = D.shape[0]
    np_pairs = len(pairs)
    W = build_W(w_vals, n)
    record = dict(bound=node_bound, branch=None, children=[],
                  cluster_spread=_cluster_spread(W, forced, n))
    out = dict(record=record, children=[], infeasible_prunes=[], unresolved=False)

    # If the LP optimum is itself a tree at this bound, the node is resolved.
    # Cherry contraction recognises this in O(n^3) — no exponential half-cut
    # enumeration, and no n<=20 ceiling.
    tree = extract_tree_cherry(W, n, D)
    if tree is not None and abs(tree['objective'] - node_bound) < EPS:
        events["tree_at_node"] += 1
        if tree['objective'] < inc_obj - EPS:
            inc_obj, inc_w = tree['objective'], tree['w']
        out["inc"] = (inc_obj, inc_w)
        return out

    # Not a tree -> branch.  CERTIFICATION-DRIVEN selection: prefer the most
    # balanced FRACTIONAL split of the incumbent (max leverage; aims to prune
    # the 's-absent' alternative fast).
    S = pick_balanced_incumbent_split(W, n, inc_splits, forced, forbidden)
    balanced_branch = S is not None
    if S is None:
        # No fractional incumbent split: fall back to the O(n^2) fractional-
        # cherry rule (then settling the face, face.py, if it, too,
        # finds nothing).
        b = pick_branching_cut_fast(W, n, forced, forbidden)
        if b['mode'] == 'none':
            # No split has its cut value in (1/2, 3/4).  Settle the face exactly when
            # at most ENUM_MAX_CLUSTERS clusters remain once the committed cherries
            # are contracted; otherwise branch on a half-cut or on cherries (face.py).
            # Either way the node is resolved soundly.
            status, res = settle_face(W, n, D, forced, forbidden)
            events["face_" + status] += 1
            if status == "tree" and res['objective'] < inc_obj - EPS:
                inc_obj, inc_w = res['objective'], res['w']
            if status not in ("branch", "cherries"):
                out["inc"] = (inc_obj, inc_w)
                return out
            S = res if status == "branch" else None
            branch_kind = "face" if status == "branch" else "face-cherry"
            cand, how = incumbent_search(W, n, D, forced, forbidden)
            if cand is not None and cand['objective'] < inc_obj - EPS:
                inc_obj, inc_w = cand['objective'], cand['w']
                heuristic[how] += 1
        else:
            S = b['S']
            branch_kind = "fractional"
    else:
        branch_kind = "balanced"

    # Branch on S: forbid (W[S]>=3/4, the 's-absent' alternative) FIRST so its
    # prune lands early, then force (W[S]=1/2).  Both children are explored
    # for completeness; the bound prunes the dead one.
    events["branch_balanced" if balanced_branch else "branch_fallback"] += 1
    if balanced_branch:
        cut_counts['bal_branch'] += 1
    # Children as (forced, forbidden, kind, keeps): keeps says whether the parent's point
    # satisfies the child's new constraints (a commit child if the committed split has cut
    # value 1/2 there, a forbid child if the forbidden one has cut value >= 3/4).
    full_set = frozenset(range(1, n + 1))
    S_crosses = False
    if S is None:
        cuts = [_cut(W, C) for C, _ in res]
        children = [(forced + [C], forbidden + list(F), "commit", abs(cut - 0.5) <= EPS)
                    for (C, F), cut in zip(res, cuts)]
        record["branch"] = dict(kind=branch_kind, children=len(children), cuts=cuts)
    else:
        S_crosses = any(_crosses(S, F, full_set) for F in forced)
        events["S_crosses_committed"] += S_crosses
        cut_S = _cut(W, S)
        # A split that crosses a committed one is in no tree of this face: forbid it
        # instead of branching, since its commit child would hold no tree.
        children = [(forced, forbidden + [S], "forbid", cut_S >= 0.75 - EPS)]
        if not S_crosses:
            children.append((forced + [S], forbidden, "commit", abs(cut_S - 0.5) <= EPS))
        record["branch"] = dict(kind=branch_kind, cut=cut_S, crosses=bool(S_crosses))
    for cf, cfb, kind, keeps in children:
        is_forbid = kind == "forbid"
        t_child = time.perf_counter()
        conic_before = exact_counts["conic_seconds"]
        lp_before = cut_counts.get('lp_seconds', 0.0)
        if keeps and _cluster_spread(W, cf, n)["max"] > 1e-9:
            # The parent's point violates the child's cluster equalities: solve the child.
            keeps = False
        if keeps:
            cb, cw, c_decided = node_bound, w_vals, True
            events["child_inherited"] += 1
        else:
            cb, cw, c_decided = _solve_node(c, D, pair_to_idx, pairs, np_pairs, cf, cfb,
                                            cut_counts, exact_counts, deadline=deadline)
        record["children"].append(dict(
            kind=kind, keeps=keeps, bound=cb if cw is not None else None,
            seconds=time.perf_counter() - t_child,
            conic_seconds=exact_counts["conic_seconds"] - conic_before,
            lp_seconds=cut_counts.get('lp_seconds', 0.0) - lp_before,
            # How far the child's point moved from the parent's: the share of weights that
            # changed, and the largest change.
            moved=(float(np.mean(np.abs(np.subtract(cw, w_vals)) > 1e-9)) if cw is not None else None),
            max_move=(float(np.max(np.abs(np.subtract(cw, w_vals)))) if cw is not None else None)))
        if not c_decided:
            # Solve aborted before any bound: the child's status is unknown,
            # so we cannot prune it — do not certify.
            events["child_undecided"] += 1
            out["unresolved"] = True
            continue
        if cw is None:
            events[f"{kind}_infeasible"] += 1
            if kind == "commit" and S is not None and S_crosses:
                events["commit_infeasible_S_crosses"] += 1
        elif cb >= inc_obj - EPS:
            events[f"{kind}_pruned_bound"] += 1
        else:
            events[f"{kind}_pushed"] += 1
        pruned = (cw is None) or (cb >= inc_obj - EPS)
        if balanced_branch and is_forbid:
            # Did forbidding the balanced split prune the s-absent alternative?
            cut_counts['fb_prune' if pruned else 'fb_open'] += 1
        if cw is None:
            out["infeasible_prunes"].append(dict(
                forced=[sorted(x) for x in cf], forbidden=[sorted(x) for x in cfb],
                status=c.solution.get_status()))
            continue  # proven-infeasible subtree
        if cb < inc_obj - EPS:
            out["children"].append((cb, cf, cfb, list(cw), kind, keeps, branch_kind))
    out["inc"] = (inc_obj, inc_w)
    return out


def solve_bme_bnb(D, time_limit=300.0, max_nodes=None):
    """Solve BME to certified optimality via explicit best-first B&B, one node at a time.

    Returns a dict with keys: objective, lb, gap, certified, n_nodes, seconds,
    cut_counts, w, and the search's instrumentation (no effect on the search).

    Every node imposes the manifold constraint exactly (exact_node.py) and, other than the
    root, its cluster equalities (_cluster_eq_rows): for each cluster left by contracting its
    committed cherries, leaf i lying e_i edges below the cluster's root, 2^e_i w_ic is the
    same for all i in the cluster, for each leaf c outside it, and w_ij is fixed at 2^-p_ij
    for two leaves of one cluster at path length p_ij. Every tree of the node satisfies them,
    so they only tighten the node's relaxation, and a child's relaxation lies inside its
    parent's. A face too large to enumerate is branched as described in face.py.

    The result's node_log records, for every node taken, how it arose (root, or the kind
    of branching, the child, whether it kept the parent's point, the parent's cut value of
    the split) and the seconds spent solving its children and in the conic solver. It has
    no effect on the search.
    """
    n = D.shape[0]
    t0 = time.time()
    c, pair_to_idx, pairs = new_lp_model(D)
    np_pairs = len(pairs)
    cut_counts = {k: 0 for k in CUT_KEYS}

    # Initial incumbent: the better of NJ + NNI and FastME's tree.
    inc = initial_incumbent(D, n)
    inc_obj = inc['objective']
    inc_w = w_from_tree(inc['tree'], n)
    # Splits of the current incumbent (the targets of certification branching),
    # recomputed lazily whenever the incumbent improves (identity check below).
    inc_splits = _incumbent_splits(inc_w, n)
    inc_w_seen = inc_w

    counter = itertools.count()  # tiebreaker so heap never compares node payloads
    deadline = t0 + time_limit   # bounds each node solve, so no node overruns
    incomplete = False           # set if any node could not be solved (aborted)

    exact_counts = new_exact_counts()
    bound, w_vals, _decided = _solve_node(c, D, pair_to_idx, pairs, np_pairs, [], [],
                                          cut_counts, exact_counts, deadline=deadline)
    # Guard: if the root LP could not even be solved within the budget, report
    # the initial incumbent with no lower bound rather than crashing.
    if w_vals is None:
        return {
            'objective': inc_obj, 'lb': float('-inf'), 'gap': float('nan'),
            'certified': False, 'n_nodes': 0, 'seconds': time.time() - t0,
            'cut_counts': cut_counts, 'w': inc_w,
        }
    frontier = [(bound, next(counter), [], [], w_vals)]
    progress = Progress(t0, 'balanced-exact')
    progress.line(0, 1, bound, inc_obj, 'root')

    n_nodes = 0
    proven = False  # True once optimality is established (not a timeout exit)
    # Instrumentation (no effect on the search): the global lower bound when each node is
    # popped (with the seconds elapsed and the lower bound counting unresolved nodes in
    # trajectory_seconds), and every child pruned because its LP was reported infeasible.
    trajectory = []
    trajectory_seconds = []
    infeasible_prunes = []
    unresolved = []
    # Incumbent improvements found by face.incumbent_search, by source.
    heuristic = {'lp_cherries': 0, 'nni': 0}
    # Counts of search events (no effect on the search). "S crosses a committed split":
    # the split chosen for branching crosses one already committed at that node, so its
    # commit child holds no tree.
    events = {k: 0 for k in EVENT_KEYS}
    # How each queued node arose, keyed by its heap tiebreaker (instrumentation).
    origin = {}
    node_log = []
    dive = None  # a keeping child to explore next

    while frontier or dive is not None:
        if time.time() - t0 > time_limit or (max_nodes and n_nodes >= max_nodes):
            if dive is not None:
                heapq.heappush(frontier, dive)
            break

        from_dive = dive is not None
        if from_dive:
            item, dive = dive, None
        else:
            item = heapq.heappop(frontier)
        node_bound, node_id, forced, forbidden, w_vals = item
        # A dived node was not taken from the heap: the lowest open bound may be elsewhere.
        global_bound = min(node_bound, frontier[0][0]) if (from_dive and frontier) else node_bound
        progress.tick(n_nodes, len(frontier) + 1,
                      min([global_bound] + [u[1] for u in unresolved]), inc_obj)
        trajectory.append((n_nodes, global_bound, inc_obj))
        trajectory_seconds.append((time.time() - t0, min([global_bound] + [u[1] for u in unresolved])))
        events["pops"] += 1

        if node_bound >= inc_obj - EPS:
            if from_dive:
                events["dive_pruned_bound"] += 1
                continue
            # Best-first: the popped bound is the global lower bound.  If it already
            # meets the incumbent, every remaining node does too -> optimal.
            events["stop_bound_reached"] += 1
            proven = True
            break

        n_nodes += 1
        # Recompute the incumbent's splits only when the incumbent has actually improved
        # (identity check).
        if inc_w is not inc_w_seen:
            inc_splits = _incumbent_splits(inc_w, n)
            inc_w_seen = inc_w
        res = branch_node(c, D, pair_to_idx, pairs, node_bound, forced, forbidden, w_vals,
                          inc_obj, inc_w, inc_splits, cut_counts, exact_counts, events,
                          heuristic, deadline)
        if res["inc"][0] < inc_obj - EPS:
            inc_obj, inc_w = res["inc"]
        record = res["record"]
        record.update(node=n_nodes, origin=origin.pop(node_id, "root"), dived=from_dive)
        node_log.append(record)
        infeasible_prunes += res["infeasible_prunes"]
        if res["unresolved"]:
            incomplete = True
            unresolved.append((n_nodes, node_bound))
        for cb, cf, cfb, cw, kind, keeps, branch_kind in res["children"]:
            if cb >= inc_obj - EPS:
                continue
            cid = next(counter)
            origin[cid] = dict(parent=n_nodes, branch=branch_kind, child=kind, keeps=keeps)
            if keeps and dive is None:
                dive = (cb, cid, cf, cfb, cw)
            else:
                heapq.heappush(frontier, (cb, cid, cf, cfb, cw))

    elapsed = time.time() - t0

    # Optimality is proven if the frontier emptied (all nodes explored) or we
    # popped a node whose bound already met the incumbent.  In both cases every
    # remaining bound is >= inc_obj, so inc_obj is the certified optimum and the
    # valid global lower bound is inc_obj itself.  On a timeout exit the global
    # lower bound is the minimum bound still open in the frontier.  ``incomplete``
    # (a child's solve was aborted at the time limit) blocks
    # certification: the search tree was not fully explored.
    if (proven or not frontier) and not incomplete:
        lb = inc_obj
        certified = True
    else:
        # A node left unresolved (a child whose solve was aborted) stays open: its bound
        # (the parent's) is part of the lower bound.
        lb = min([item[0] for item in frontier] + [u[1] for u in unresolved], default=inc_obj)
        certified = False
    gap = max(0.0, (inc_obj - lb) / inc_obj) if inc_obj > 0 else 0.0
    progress.line(n_nodes, len(frontier), lb, inc_obj,
                  'certified' if certified else 'stopped')

    return {
        'objective': inc_obj,
        'lb': lb,
        'gap': gap,
        'certified': certified,
        'n_nodes': n_nodes,
        'seconds': elapsed,
        'cut_counts': cut_counts,
        'w': inc_w,
        'root_lb': bound,
        'trajectory': trajectory,
        'trajectory_seconds': trajectory_seconds,
        'unresolved': unresolved,
        'heuristic_improvements': heuristic,
        'exact_counts': exact_counts,
        'infeasible_prunes': infeasible_prunes,
        'events': events,
        'node_log': node_log,
    }
