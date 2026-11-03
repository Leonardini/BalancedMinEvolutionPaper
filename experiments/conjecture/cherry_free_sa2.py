"""Simulated annealing for a matrix satisfying (4)-(9) with no cherry, a counterexample to the
cherry conjecture (Section 11), at n = 24.

    python cherry_free_sa2.py CEX12 OUTDIR [RESTARTS ITERS SEED_BASE MANIFOLD_WEIGHT TRANSFER_PROB]
    (defaults 20, 4000000, 1000, 1.0, 0.4)

Start: the first matrix of CEX12 (an n = 12 counterexample to (10), a relabelling of the
cherry lift of Section 8.3) doubled: every leaf becomes two leaves at path length 3, and every
other entry grows by 1, which gives a 24 x 24 matrix that has no cherry and satisfies the
strong triangle inequality but not Kraft or the manifold. The annealing moves integer entries
in [3, n-1] symmetrically and minimises, in units of 2^-(n-1),
    V = sum_i |Kraft_i - 1/2| + MANIFOLD_WEIGHT * |manifold - (n - 3/2)| + 2^24 * strong-triangle deficit,
so V = 0 exactly when the matrix satisfies (4)-(9) with every entry >= 3. Moves: one entry
+/-1; a Kraft-neutral transfer between two entries of a row; and a row-Kraft-neutral split and
join {a, b, b} -> {a+1, a+1, b-1} that shifts the manifold. Each restart r > 0 starts from the
doubled matrix with 3r random increments of one entry and anneals with seed SEED_BASE + r.

Writes OUTDIR/cherry_free_running_n24_<SEED_BASE>.txt (the best matrix so far, its V in the
first line) whenever a restart improves on it, OUTDIR/cherry_free_best_n24_<SEED_BASE>.txt at
the end, and OUTDIR/cherry_free_n24.txt if V = 0 is reached. One line per restart to stderr.
"""
import sys, random, math
from pathlib import Path


def log(*a):
    print(*a, file=sys.stderr, flush=True)

def load_cex(path):
    blk=open(path).read().split('--')[0].strip().split('\n')
    return [[int(x) for x in l.split()] for l in blk]

def doubled_template(M):
    m=len(M); n=2*m
    T=[[0]*n for _ in range(n)]
    for i in range(m):
        for j in range(m):
            for a in (0,1):
                for b in (0,1):
                    ii,jj=2*i+a,2*j+b
                    if ii!=jj: T[ii][jj]= 3 if i==j else M[i][j]+1
    return T

def tripdef(x,y,z):                       # deficit of strong triangle on edges (ij=x, ik=y, jk=z)
    d=0
    if x+z-y<2: d+=2-(x+z-y)
    if x+y-z<2: d+=2-(x+y-z)
    if y+z-x<2: d+=2-(y+z-x)
    return d

def eval_changes(changes, T, K, Mani, KT, MT, n, pw):
    # changes: list of (i,j,newval); returns (dkv,dmv,dtd,dK-dict,dMani). Handles 1- or 2-entry moves.
    cp={}; rows=set()
    for (i,j,b) in changes: cp[(min(i,j),max(i,j))]=b; rows.add(i); rows.add(j)
    dK={r:0 for r in rows}
    for (i,j,b) in changes:
        d=pw[b]-pw[T[i][j]]; dK[i]+=d; dK[j]+=d
    dkv=sum(abs(K[r]+dK[r]-KT)-abs(K[r]-KT) for r in rows)
    dMani=sum(b*pw[b]-T[i][j]*pw[T[i][j]] for (i,j,b) in changes)
    dmv=abs(Mani+dMani-MT)-abs(Mani-MT)
    def nv(x,y): return cp.get((min(x,y),max(x,y)), T[x][y])
    seen=set(); dtd=0
    for (i,j,_) in changes:
        for m in range(n):
            if m==i or m==j: continue
            key=(min(i,j,m),sorted((i,j,m))[1],max(i,j,m))
            if key in seen: continue
            seen.add(key); a,b,c=key
            dtd+=tripdef(nv(a,b),nv(a,c),nv(b,c))-tripdef(T[a][b],T[a][c],T[b][c])
    return dkv,dmv,dtd,dK,dMani

def anneal(T, n, KT, MT, W, seed, iters, T0, Tend, wm=1, trp=0.4):
    random.seed(seed)
    pw=[1<<(n-1-t) for t in range(n)]                      # 2^{n-1-tau}, tau in 0..n-1
    K=[sum(pw[T[i][j]] for j in range(n) if j!=i) for i in range(n)]
    Mani=sum(T[i][j]*pw[T[i][j]] for i in range(n) for j in range(i+1,n))
    tdef=sum(tripdef(T[i][j],T[i][k],T[j][k])
             for i in range(n) for j in range(i+1,n) for k in range(j+1,n))
    kv=sum(abs(K[i]-KT) for i in range(n)); mv=abs(Mani-MT)
    V=kv+wm*mv+W*tdef                                      # wm = manifold weight (>1 => land on manifold first)
    best=V; bestT=[row[:] for row in T]
    logT0,logTend=math.log(T0),math.log(Tend)
    for it in range(iters):
        temp=math.exp(logT0+(logTend-logT0)*it/iters)
        u=random.random()
        if u<trp:                                          # TRANSFER (manifold-neutral Kraft shift)
            i=random.randrange(n)
            js=[j for j in range(n) if j!=i]
            j=random.choice(js); p=T[i][j]
            ks=[k for k in js if k!=j and T[i][k]==p+1]
            if not ks or p+1>n-1 or p<3: continue
            k=random.choice(ks); changes=[(i,j,p+1),(i,k,p)]
        elif u<trp+0.15:                                   # general split and join: {a,b,b}->{a+1,a+1,b-1}
            i=random.randrange(n); byl={}                  # row-Kraft-neutral; manifold += 2^-a - 2^-(b-1); a==b => old SPREAD
            for j in range(n):
                if j!=i: byl.setdefault(T[i][j],[]).append(j)
            bvals=[bb for bb in byl if len(byl[bb])>=2 and bb>=4]        # join two b's -> one b-1 (needs b-1>=3)
            if not bvals: continue
            b=random.choice(bvals); jb1,jb2=random.sample(byl[b],2)
            avals=[aa for aa in byl if aa+1<=n-1 and (aa!=b or len(byl[aa])>=3)]  # split a -> two a+1 (a==b needs a 3rd cell)
            if not avals: continue
            a=random.choice(avals)
            if a==b:
                others=[j for j in byl[a] if j!=jb1 and j!=jb2]
                if not others: continue
                ja=random.choice(others)
            else:
                ja=random.choice(byl[a])
            changes=[(i,ja,a+1),(i,jb1,a+1),(i,jb2,b-1)]
        else:                                              # single +/-1 move (reduces Kraft+manifold in balance)
            i=random.randrange(n); j=random.randrange(n)
            if i==j: continue
            b=T[i][j]+(1 if random.random()<0.5 else -1)
            if b<3 or b>n-1: continue
            changes=[(i,j,b)]
        dkv,dmv,dtd,dK,dMani=eval_changes(changes,T,K,Mani,KT,MT,n,pw)
        dV=dkv+wm*dmv+W*dtd
        if dV<=0 or random.random()<math.exp(-min(dV/temp,700)):
            for (a,b,nvv) in changes: T[a][b]=T[b][a]=nvv
            for r,d in dK.items(): K[r]+=d
            Mani+=dMani; tdef+=dtd; kv+=dkv; mv+=dmv; V+=dV
            if V<best:
                best=V; bestT=[row[:] for row in T]
                if V==0: return 0,bestT
    return best,bestT

if __name__=="__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    M=load_cex(sys.argv[1]); n=2*len(M); KT=1<<(n-2); MT=(2*n-3)*(1<<(n-2)); W=1<<24
    outdir=Path(sys.argv[2]); outdir.mkdir(parents=True, exist_ok=True)
    restarts=int(sys.argv[3]) if len(sys.argv)>3 else 20
    iters=int(sys.argv[4]) if len(sys.argv)>4 else 4_000_000
    sbase=int(sys.argv[5]) if len(sys.argv)>5 else 1000    # distinct seed stream per parallel instance
    wm=float(sys.argv[6]) if len(sys.argv)>6 else 1.0      # manifold weight (>1: land on the manifold first)
    trp=float(sys.argv[7]) if len(sys.argv)>7 else 0.4     # transfer-move probability
    globalbest=None
    for r in range(restarts):
        T=doubled_template(M)
        if r>0:                                            # perturb: random increments to diversify
            for _ in range(r*3):
                i=random.randrange(n); j=random.randrange(n)
                if i!=j and T[i][j]<n-1: T[i][j]=T[j][i]=T[i][j]+1
        best,bestT=anneal(T,n,KT,MT,W,seed=sbase+r,iters=iters,T0=float(1<<23),Tend=float(1<<14),wm=wm,trp=trp)
        # decompose best
        K=[sum(1<<(n-1-bestT[i][j]) for j in range(n) if j!=i) for i in range(n)]
        Mani=sum(bestT[i][j]*(1<<(n-1-bestT[i][j])) for i in range(n) for j in range(i+1,n))
        kv=sum(abs(x-KT) for x in K); mv=abs(Mani-MT)
        td=sum(tripdef(bestT[i][j],bestT[i][k],bestT[j][k])
               for i in range(n) for j in range(i+1,n) for k in range(j+1,n))
        cherries=sum(1 for i in range(n) for j in range(i+1,n) if bestT[i][j]==2)
        log(f"restart {r:2d}: bestV={best:>12}  Kraft-dev={kv/KT:.3f}rows  manifold-dev={mv/MT:.4f}  tri-deficit={td}  cherries={cherries}")
        if globalbest is None or best<globalbest[0]:
            globalbest=(best,[row[:] for row in bestT])
            open(outdir / ('cherry_free_running_n%d_%d.txt'%(n,sbase)),'w').write(   # save-on-improve (V in first-line comment)
                '# V=%d\n'%best + '\n'.join(' '.join(map(str,r)) for r in bestT))
        if best==0:
            log("  *** FOUND cherry-free (4)-(9) matrix (a counterexample to the cherry conjecture)! ***")
            for row in bestT: log("   "," ".join(f"{x:2d}" for x in row))
            open(outdir / ('cherry_free_n%d.txt'%n),'w').write('\n'.join(' '.join(map(str,r)) for r in bestT))
            break
    log(f"global best V = {globalbest[0]}")
    G=globalbest[1]
    Kb=[sum(1<<(n-1-G[i][j]) for j in range(n) if j!=i) for i in range(n)]
    dev=[k-KT for k in Kb]
    log("per-row Kraft residual (K_i - KT):", dev)
    log(f"  #rows off: {sum(1 for d in dev if d)}   total sum(K_i)-{n}*KT = {sum(Kb)-n*KT}"
          f"  ({'pure imbalance -> transfer-fixable' if sum(Kb)==n*KT else 'TOTAL off -> needs manifold-touching move'})")
    open(outdir / ('cherry_free_best_n%d_%d.txt'%(n,sbase)),'w').write('\n'.join(' '.join(map(str,r)) for r in G))  # a start for cherry_free_descent.py
