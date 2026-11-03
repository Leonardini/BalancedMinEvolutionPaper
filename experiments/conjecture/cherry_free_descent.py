"""Annealing with manifold-neutral moves for a matrix satisfying (4)-(9) with no cherry, a
counterexample to the cherry conjecture (Section 11); the second stage after
cherry_free_sa2.py.

    python cherry_free_descent.py START OUTDIR [ITERS SEED_BASE W RESTARTS]
    (defaults 8000000, 90000, 2^24, 150)

START is a square matrix (lines starting with # are skipped): the doubled n = 12
counterexample (cherry_free_sa2.py, function doubled_template) or a matrix that
cherry_free_sa2.py wrote. Same objective V as cherry_free_sa2.py (manifold weight 1, triangle
weight W), V = 0 exactly at a counterexample. Besides the +/-1, transfer and split-and-join
moves, it applies whole multiset moves A -> B within a row (|A| = |B| in {2, 3}) that keep the
manifold contribution and change Kraft, such as {3, 8} -> {4, 5}. Each restart r > 0 starts
from START with 3r random increments (seed SEED_BASE + 7r) and anneals with seed SEED_BASE + r.

Writes OUTDIR/cherry_free_running_n<N>_<SEED_BASE>.txt (the best matrix so far, its V in the
first line) on every improvement, and OUTDIR/cherry_free_n<N>.txt if V = 0 is reached. One
line per restart to stderr.
"""
import sys, random, time, itertools, math
from pathlib import Path
from collections import Counter

def load(p):
    return [[int(x) for x in l.split()] for l in open(p).read().strip().split('\n')
            if l.strip() and not l.startswith('#')]

def tripdef(x, y, z):
    d = 0
    if x+z-y < 2: d += 2-(x+z-y)
    if x+y-z < 2: d += 2-(x+y-z)
    if y+z-x < 2: d += 2-(y+z-x)
    return d

def build_manifold_neutral_moves(n):
    # multisets A->B (|A|=|B| in {2,3}), values in [3,n-1], with sum t*2^{-t} preserved (manifold-neutral)
    # and sum 2^{-t} changed (Kraft-changing).  Store as (sorted A tuple, sorted B tuple).
    V = list(range(3, n)); Mnum = lambda t: t*(1 << (n-1-t)); moves = []
    for sz in (2, 3):
        byM = {}
        for A in itertools.combinations_with_replacement(V, sz):
            byM.setdefault(sum(Mnum(t) for t in A), []).append(A)
        for lst in byM.values():
            for A, B in itertools.permutations(lst, 2):
                if sum(1 << (n-1-t) for t in A) != sum(1 << (n-1-t) for t in B):
                    moves.append((A, B))
    return moves

def eval_changes(changes, T, K, Mani, KT, MT, n, pw):
    cp = {}; rows = set()
    for (i, j, b) in changes: cp[(min(i, j), max(i, j))] = b; rows.add(i); rows.add(j)
    dK = {r: 0 for r in rows}
    for (i, j, b) in changes:
        d = pw[b]-pw[T[i][j]]; dK[i] += d; dK[j] += d
    dkv = sum(abs(K[r]+dK[r]-KT)-abs(K[r]-KT) for r in rows)
    dMani = sum(b*pw[b]-T[i][j]*pw[T[i][j]] for (i, j, b) in changes)
    dmv = abs(Mani+dMani-MT)-abs(Mani-MT)
    def nv(x, y): return cp.get((min(x, y), max(x, y)), T[x][y])
    seen = set(); dtd = 0
    for (i, j, _) in changes:
        for m in range(n):
            if m == i or m == j: continue
            key = tuple(sorted((i, j, m)))
            if key in seen: continue
            seen.add(key); a, b, c = key
            dtd += tripdef(nv(a, b), nv(a, c), nv(b, c)) - tripdef(T[a][b], T[a][c], T[b][c])
    return dkv, dmv, dtd, dK, dMani

def try_matched(T, n, mm_moves):
    i = random.randrange(n); byl = {}
    for j in range(n):
        if j != i: byl.setdefault(T[i][j], []).append(j)
    for _ in range(30):                                        # sample a few random moves (fast)
        A, B = random.choice(mm_moves)
        need = Counter(A)
        if all(len(byl.get(v, [])) >= c for v, c in need.items()):
            cells = []; used = {}
            for v in A:
                used.setdefault(v, 0); cells.append(byl[v][used[v]]); used[v] += 1
            Bl = list(B); random.shuffle(Bl)
            return [(i, cells[k], Bl[k]) for k in range(len(A))]
    return None

def descent(T, n, KT, MT, W, wm, iters, seed, sbase, outdir):
    random.seed(seed)
    pw = [1 << (n-1-t) for t in range(n)]
    mm = build_manifold_neutral_moves(n)
    K = [sum(pw[T[i][j]] for j in range(n) if j != i) for i in range(n)]
    Mani = sum(T[i][j]*pw[T[i][j]] for i in range(n) for j in range(i+1, n))
    tdef = sum(tripdef(T[i][j], T[i][k], T[j][k])
               for i in range(n) for j in range(i+1, n) for k in range(j+1, n))
    kv = sum(abs(K[i]-KT) for i in range(n)); mv = abs(Mani-MT)
    V = kv + wm*mv + W*tdef; best = V; bestT = [r[:] for r in T]
    t0 = time.time(); logT0, logTend = math.log(1 << 23), math.log(1 << 12)   # geometric schedule (like the SA)
    for it in range(iters):
        temp = math.exp(logT0 + (logTend-logT0)*it/iters)
        u = random.random()
        if u < 0.30:                                            # matched-manifold (Kraft, hold manifold)
            changes = try_matched(T, n, mm)
        elif u < 0.45:                                          # split+join (manifold, hold Kraft)
            i = random.randrange(n); byl = {}
            for j in range(n):
                if j != i: byl.setdefault(T[i][j], []).append(j)
            bvals = [b for b in byl if len(byl[b]) >= 2 and b >= 4]
            if not bvals: continue
            b = random.choice(bvals); jb1, jb2 = random.sample(byl[b], 2)
            avals = [a for a in byl if a+1 <= n-1 and (a != b or len(byl[a]) >= 3)]
            if not avals: continue
            a = random.choice(avals)
            if a == b:
                others = [j for j in byl[a] if j != jb1 and j != jb2]
                if not others: continue
                ja = random.choice(others)
            else:
                ja = random.choice(byl[a])
            changes = [(i, ja, a+1), (i, jb1, a+1), (i, jb2, b-1)]
        elif u < 0.70:                                          # transfer (Kraft redistribute)
            i = random.randrange(n); js = [j for j in range(n) if j != i]
            j = random.choice(js); p = T[i][j]
            ks = [k for k in js if k != j and T[i][k] == p+1]
            if not ks or p+1 > n-1 or p < 3: continue
            k = random.choice(ks); changes = [(i, j, p+1), (i, k, p)]
        else:                                                  # single +/-1
            i = random.randrange(n); j = random.randrange(n)
            if i == j: continue
            b = T[i][j] + (1 if random.random() < 0.5 else -1)
            if b < 3 or b > n-1: continue
            changes = [(i, j, b)]
        if not changes: continue
        dkv, dmv, dtd, dK, dMani = eval_changes(changes, T, K, Mani, KT, MT, n, pw)
        dV = dkv + wm*dmv + W*dtd
        if dV <= 0 or random.random() < math.exp(-min(dV/temp, 700)):
            for (a, b, nvv) in changes: T[a][b] = T[b][a] = nvv
            for r, d in dK.items(): K[r] += d
            Mani += dMani; tdef += dtd; kv += dkv; mv += dmv; V += dV
            if V < best:
                best = V; bestT = [r[:] for r in T]
                open(outdir / ('cherry_free_running_n%d_%d.txt' % (n, sbase)), 'w').write(
                    '# V=%d\n' % best + '\n'.join(' '.join(map(str, r)) for r in T))
                if V == 0:
                    open(outdir / ('cherry_free_n%d.txt' % n), 'w').write('\n'.join(' '.join(map(str, r)) for r in T))
                    print("*** V=0 -> cherry_free_n%d.txt (a counterexample to the cherry conjecture) ***" % n, file=sys.stderr, flush=True)
                    return 0, T
    return best, bestT

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    start = load(sys.argv[1]); n = len(start)
    outdir = Path(sys.argv[2]); outdir.mkdir(parents=True, exist_ok=True)
    iters = int(sys.argv[3]) if len(sys.argv) > 3 else 8_000_000
    sbase = int(sys.argv[4]) if len(sys.argv) > 4 else 90000
    W = int(sys.argv[5]) if len(sys.argv) > 5 else (1 << 24)    # hard-ish triangles (like the SA)
    restarts = int(sys.argv[6]) if len(sys.argv) > 6 else 150
    KT = 1 << (n-2); MT = (2*n-3)*(1 << (n-2))
    print(f"SA+matched-manifold from {sys.argv[1]} (n={n}), {restarts} restarts x {iters} iters, W={W}", file=sys.stderr, flush=True)
    gbest = None
    for r in range(restarts):
        T = [row[:] for row in start]
        if r > 0:                                              # perturb to diversify restarts (like the SA)
            random.seed(sbase + r*7)
            for _ in range(r*3):
                i = random.randrange(n); j = random.randrange(n)
                if i != j and T[i][j] < n-1: T[i][j] = T[j][i] = T[i][j]+1
        best, T = descent(T, n, KT, MT, W, wm=1.0, iters=iters, seed=sbase+r, sbase=sbase, outdir=outdir)
        print(time.strftime("%H:%M:%S"), f"restart {r:3d}: best V = {best}", file=sys.stderr, flush=True)
        if gbest is None or best < gbest: gbest = best
        if best == 0:
            print("*** V=0 FOUND -> cherry_free_n%d.txt ***" % n, file=sys.stderr); break
    print(f"global best V = {gbest}", file=sys.stderr, flush=True)
