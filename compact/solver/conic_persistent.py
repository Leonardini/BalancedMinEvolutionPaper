"""The conic model of a node, kept in one MOSEK task for the whole search.

conic.solve rebuilds the model through CVXPY at every solve, from every row of the CPLEX model
read back into Python. This module keeps a single MOSEK task per CPLEX model instead: the
manifold constraint (one exponential cone per weight, t_j <= -w_j ln w_j, and the row
sum_j t_j >= (n - 3/2) ln 2) is built once; the CPLEX model's global rows (base rows and
cuts, which are only ever appended) are added to the task as they appear; and a node's own
rows (committed and forbidden splits, cluster equalities: the block CPLEX holds at
[start, start + count) during the node's solve) are swapped in at each solve. Rows are scaled
by exact powers of two as in conic.solve. The interior-point solve itself is the same; only
the rebuilding is avoided.

solve returns what conic.solve returns, with the row senses: (w, objective, A, b, y, mu,
iterations, inaccurate, senses), A and b the scaled rows in the task's order and y their
multipliers sign-corrected to the cone of each sense, mu >= 0 the manifold row's; or
("infeasible", status) / ("failed", None).
"""
import math

import mosek
import numpy as np

from .conic import RETRY_TOLS, TOL, log

INF = 0.0  # MOSEK ignores the bound value on an infinite side


class PersistentConic:
    def __init__(self, Dv, n, lo, hi):
        self.n, self.N = n, len(Dv)
        self.env = mosek.Env()
        self.task = self.env.Task()
        t = self.task
        t.putintparam(mosek.iparam.num_threads, 1)
        N = self.N
        t.appendvars(2 * N)                      # w_0..w_{N-1}, then t_0..t_{N-1}
        t.putclist(list(range(N)), [float(d) for d in Dv])
        t.putvarboundsliceconst(0, N, mosek.boundkey.ra, lo, hi)
        t.putvarboundsliceconst(N, 2 * N, mosek.boundkey.fr, -math.inf, math.inf)
        t.putobjsense(mosek.objsense.minimize)
        # Row 0: the manifold, sum_j t_j >= (n - 3/2) ln 2.
        t.appendcons(1)
        t.putarow(0, list(range(N, 2 * N)), [1.0] * N)
        t.putconbound(0, mosek.boundkey.lo, (n - 1.5) * math.log(2.0), INF)
        # (1, w_j, t_j) in the exponential cone: 1 >= w_j exp(t_j / w_j), i.e. t_j <= -w_j ln w_j.
        t.appendafes(3 * N)
        t.putafefentrylist([3 * j + 1 for j in range(N)] + [3 * j + 2 for j in range(N)],
                           list(range(N)) + list(range(N, 2 * N)), [1.0] * (2 * N))
        t.putafeglist([3 * j for j in range(N)], [1.0] * N)
        dom = t.appendprimalexpconedomain()
        for j in range(N):
            t.appendacc(dom, [3 * j, 3 * j + 1, 3 * j + 2], None)
        self.n_global = 0          # global CPLEX rows already in the task (rows 1..n_global)
        self.n_node = 0            # node rows at the end of the task
        self.A = np.zeros((0, N))  # scaled global rows, in task order
        self.b = np.zeros(0)
        self.sen = []

    def _append(self, rows, sens, rhs):
        """Append rows (lists of (index, value)) to the task, scaled; return their dense form."""
        t = self.task
        k = len(rows)
        A = np.zeros((k, self.N))
        for r, (ind, val) in enumerate(rows):
            if any(i >= self.N for i in ind):
                raise AssertionError("a row of the LP involves the dummy variable")
            A[r, ind] = val
        big = np.abs(A).max(axis=1)
        scale = np.where(big > 0, 2.0 ** -np.round(np.log2(np.where(big > 0, big, 1.0))), 1.0)
        A *= scale[:, None]
        b = np.array(rhs, dtype=float) * scale
        first = t.getnumcon()
        t.appendcons(k)
        for r in range(k):
            nz = np.flatnonzero(A[r])
            t.putarow(first + r, nz.tolist(), A[r, nz].tolist())
            s = sens[r]
            bk = (mosek.boundkey.lo if s == "G" else mosek.boundkey.up if s == "L"
                  else mosek.boundkey.fx)
            t.putconbound(first + r, bk, float(b[r]), float(b[r]))
        return A, b

    def sync(self, c, start, count):
        """Bring the task up to date with the CPLEX model c, whose node block is
        [start, start + count)."""
        t = self.task
        if self.n_node:
            t.removecons(list(range(1 + self.n_global, 1 + self.n_global + self.n_node)))
            self.n_node = 0
        total = c.linear_constraints.get_num() - count
        if total < self.n_global:
            raise AssertionError(f"the CPLEX model lost global rows ({total} < {self.n_global})")
        if total > self.n_global:
            idx = [k if k < start else k + count for k in range(self.n_global, total)]
            sps = c.linear_constraints.get_rows(idx)
            A, b = self._append([(sp.ind, sp.val) for sp in sps],
                                c.linear_constraints.get_senses(idx),
                                c.linear_constraints.get_rhs(idx))
            self.A, self.b = np.vstack([self.A, A]), np.concatenate([self.b, b])
            self.sen += list(c.linear_constraints.get_senses(idx))
            self.n_global = total
        if count:
            idx = list(range(start, start + count))
            sps = c.linear_constraints.get_rows(idx)
            sens = c.linear_constraints.get_senses(idx)
            self.An, self.bn = self._append([(sp.ind, sp.val) for sp in sps], sens,
                                            c.linear_constraints.get_rhs(idx))
            self.sen_node = list(sens)
            self.n_node = count
        else:
            self.An, self.bn, self.sen_node = np.zeros((0, self.N)), np.zeros(0), []

    def solve(self, c, start, count, allow_infeasible=True):
        self.sync(c, start, count)
        t = self.task
        N = self.N
        for tol in (TOL,) + RETRY_TOLS:
            for p in (mosek.dparam.intpnt_co_tol_pfeas, mosek.dparam.intpnt_co_tol_dfeas,
                      mosek.dparam.intpnt_co_tol_rel_gap):
                t.putdouparam(p, tol)
            t.optimize()
            solsta = t.getsolsta(mosek.soltype.itr)
            prosta = t.getprosta(mosek.soltype.itr)
            if solsta == mosek.solsta.optimal:
                inaccurate = False
                break
            if solsta == mosek.solsta.prim_infeas_cer:
                if allow_infeasible:
                    return "infeasible", str(prosta)
                raise RuntimeError(f"MOSEK: infeasible ({prosta})")
            if solsta == mosek.solsta.unknown and prosta in (mosek.prosta.prim_and_dual_feas,
                                                              mosek.prosta.unknown) and tol == RETRY_TOLS[-1]:
                inaccurate = True       # the Lagrangian bound stays valid for any multipliers
                break
            if tol == RETRY_TOLS[-1]:
                if allow_infeasible:
                    return "failed", None
                raise RuntimeError(f"MOSEK: solution status {solsta}, problem status {prosta}")
            log(f"MOSEK: solution status {solsta} at tolerance {tol:g}; retrying looser")
        w = np.array(t.getxxslice(mosek.soltype.itr, 0, N))
        yall = np.array(t.gety(mosek.soltype.itr))
        mu = max(float(yall[0]), 0.0)
        sen = self.sen + self.sen_node
        y = yall[1:]
        # Multipliers sign-corrected to the cone of each sense: >= 0 for G, <= 0 for L.
        y = np.array([max(v, 0.0) if s == "G" else min(v, 0.0) if s == "L" else v
                      for v, s in zip(y, sen)])
        A = np.vstack([self.A, self.An])
        b = np.concatenate([self.b, self.bn])
        iters = t.getintinf(mosek.iinfitem.intpnt_iter)
        return w, float(t.getprimalobj(mosek.soltype.itr)), A, b, y, mu, iters, inaccurate, sen
