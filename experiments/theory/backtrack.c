/* Exhaustive enumeration of integer matrices tau satisfying conditions (4)-(9) of
 * Catanzaro, Pesenti and Wolsey (2020), each then tested against the strong four-point
 * condition (10). Used for Theorem 5 and Appendix C.
 *
 * The upper triangle is filled row by row. A branch is cut only when it violates a
 * condition:
 *   (1) the strong triangle (6) is checked as soon as a triple is complete;
 *   (2) every partially filled row must stay a sub-multiset of at least one row type
 *       (a multiset of path lengths that can form a complete Kraft row), tracked as a
 *       bitmask over the types, which makes (7) and (8) exact at completion;
 *   (3) the manifold (9) is bounded using the minimum and maximum manifold contribution
 *       over each incomplete row's remaining compatible types.
 * By default leaf 0's row is put in non-decreasing order (valid for an existence
 * question, since any solution can be relabelled to satisfy it). A survivor is a
 * complete matrix satisfying (4)-(9).
 *
 * Usage: backtrack N [options]
 *   --full          no ordering on leaf 0's row: every labelled survivor is counted
 *   --nocherry      only row types with every tau >= 3 (no pair at distance 2)
 *   --weak          weak triangle (slack >= 0) in place of the strong one (slack >= 2)
 *   --fpviol        require (10) to fail on the quartet {0,1,2,3}; leaf 0's row is then
 *                   ordered only on columns 4..N-1
 *   --t01 V         fix tau[0][1] = V
 *   --dump PATH     write every survivor that violates (10) to PATH (N rows, then "--")
 *   --progress S    report progress to stderr every S seconds (default 60)
 * The last stdout line is the summary: survivor count, violating count, nodes, seconds,
 * number of row types. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <limits.h>
typedef unsigned __int128 u128;
#define MAXN 20
#define MAXT 1024
static int n, canon, nocherry, weak, fpviol, t01;
static int VAL[MAXT][MAXN];      /* VAL[x][t] = multiplicity of path length t in row type x */
static long ROWM[MAXT];          /* manifold contribution of row type x (scaled to integers) */
static int NT;
static u128 maskGE[MAXN][MAXN];  /* row types x with VAL[x][t] >= c */
static long Mc[MAXN], TTGT, MINM, MAXM;
static int tau[MAXN][MAXN], cnt[MAXN][MAXN], rcnt[MAXN];
static long rowM[MAXN];
static u128 compat[MAXN];
static long compatMin[MAXN], compatMax[MAXN];   /* min/max ROWM over compat[r] */
static int pi[MAXN*MAXN], pj[MAXN*MAXN], NP;
static long long surv=0, nontree=0, nodes=0;
static int ex[MAXN][MAXN], have_ex=0;
static FILE *fx;
static time_t t0, tlast;
static long progress_every = 60;

static void stamp(FILE *f) {
    char buf[32]; time_t now=time(NULL);
    strftime(buf, sizeof buf, "%Y-%m-%d %H:%M:%S", localtime(&now));
    fputs(buf, f);
}

static int tmpcnt[MAXN];
/* Row types: multisets of n-1 powers of two summing to 2^{n-2}, the part 2^e standing
 * for path length n-1-e (Kraft scaled by 2^{n-1}). */
static void gen(long rem, int cleft, long maxp) {
    if (cleft==0) {
        if (rem==0) {
            if (NT>=MAXT) { fprintf(stderr,"error: more than %d row types\n",MAXT); exit(2); }
            for (int t=2;t<n;t++) VAL[NT][t]=tmpcnt[t];
            long m=0; for(int t=2;t<n;t++) m += (long)tmpcnt[t]*Mc[t];
            ROWM[NT]=m; NT++;
        }
        return;
    }
    if (rem < cleft || rem > maxp*cleft) return;
    for (long p=maxp; p>=1; p>>=1) {
        if (p>rem) continue;
        int e=__builtin_ctzl(p); int t=n-1-e;
        tmpcnt[t]++; gen(rem-p, cleft-1, p); tmpcnt[t]--;
    }
}

static void minmax_rowm(u128 mask, long*mn, long*mx){
    long lo=LONG_MAX, hi=LONG_MIN;
    unsigned long long w=(unsigned long long)mask;
    while(w){ int b=__builtin_ctzll(w); long r=ROWM[b]; if(r<lo)lo=r; if(r>hi)hi=r; w&=w-1; }
    w=(unsigned long long)(mask>>64);
    while(w){ int b=__builtin_ctzll(w); long r=ROWM[64+b]; if(r<lo)lo=r; if(r>hi)hi=r; w&=w-1; }
    *mn=lo; *mx=hi;
}

/* (10) fails on a quartet if its two largest pair sums differ, or the largest exceeds
 * the smallest by less than 2. */
static int four_point_fail(void) {
    for (int a=0;a<n;a++) for (int b=a+1;b<n;b++) for (int c=b+1;c<n;c++) for (int d=c+1;d<n;d++){
        int s0=tau[a][b]+tau[c][d], s1=tau[a][c]+tau[b][d], s2=tau[a][d]+tau[b][c];
        int lo=s0,mid=s1,hi=s2,tmp;
        if(lo>mid){tmp=lo;lo=mid;mid=tmp;} if(mid>hi){tmp=mid;mid=hi;hi=tmp;} if(lo>mid){tmp=lo;lo=mid;mid=tmp;}
        if (mid!=hi || hi-lo<2) return 1;
    }
    return 0;
}

static void rec(int p, long T, int ncomp) {
    if (((++nodes) & ((1LL<<24)-1))==0) {
        time_t now=time(NULL);
        if (now-tlast >= progress_every) {
            tlast=now; stamp(stderr);
            fprintf(stderr," nodes=%lldM survivors=%lld violating=%lld elapsed=%lds\n",
                    nodes/1000000, surv, nontree, (long)(now-t0));
        }
    }
    if (p==NP) {
        surv++;
        if (four_point_fail()) {
            nontree++;
            if (!have_ex) { memcpy(ex,tau,sizeof(tau)); have_ex=1; }
            if (fx) {
                for(int a=0;a<n;a++){ for(int b=0;b<n;b++) fprintf(fx,"%d ",tau[a][b]); fprintf(fx,"\n"); }
                fprintf(fx,"--\n"); fflush(fx);
            }
        }
        return;
    }
    int i=pi[p], j=pj[p];
    int prev = 0;
    if (canon && i==0 && j>1) prev = tau[0][j-1];
    if (fpviol && i==0 && j>4) prev = tau[0][j-1];    /* {0,1,2,3} is pinned; order the rest */
    for (int v=(nocherry?3:2); v<n; v++) {
        if (v<prev) continue;
        if (t01 && i==0 && j==1 && v!=t01) continue;
        int bad=0, slk=weak?0:2;                       /* triangles {i,j,k}, k<i */
        for (int k=0;k<i;k++){ int tik=tau[i][k],tjk=tau[j][k];
            if (v+tik<tjk+slk || v+tjk<tik+slk || tik+tjk<v+slk){bad=1;break;} }
        if (bad) continue;
        if (fpviol && i==2 && j==3) {                  /* keep only fills where (10) fails on {0,1,2,3} */
            int s0=tau[0][1]+v, s1=tau[0][2]+tau[1][3], s2=tau[0][3]+tau[1][2];
            int lo=s0,md=s1,hi=s2,t;
            if(lo>md){t=lo;lo=md;md=t;} if(md>hi){t=md;md=hi;hi=t;} if(lo>md){t=lo;lo=md;md=t;}
            if (md==hi && hi-lo>=2) continue;
        }
        cnt[i][v]++; cnt[j][v]++;                      /* sub-multiset prune */
        u128 ci=compat[i], cj=compat[j];
        long si_mn=compatMin[i], si_mx=compatMax[i], sj_mn=compatMin[j], sj_mx=compatMax[j];
        compat[i] = ci & maskGE[v][cnt[i][v]];
        compat[j] = cj & maskGE[v][cnt[j][v]];
        if (compat[i] && compat[j]) {
            long mc=Mc[v]; rowM[i]+=mc; rowM[j]+=mc; rcnt[i]++; rcnt[j]++;
            long dT=0; int dnc=0;
            if(rcnt[i]==n-1){dT+=rowM[i];dnc++;} else minmax_rowm(compat[i],&compatMin[i],&compatMax[i]);
            if(rcnt[j]==n-1){dT+=rowM[j];dnc++;} else minmax_rowm(compat[j],&compatMin[j],&compatMax[j]);
            long Tn=T+dT; int ncn=ncomp+dnc; long need=TTGT-Tn;
            long lo=0, hi=0;                           /* per-row manifold bound */
            for(int r=0;r<n;r++) if(rcnt[r]<n-1){ lo+=compatMin[r]; hi+=compatMax[r]; }
            if (need >= lo && need <= hi) {
                tau[i][j]=tau[j][i]=v;
                rec(p+1, Tn, ncn);
                tau[i][j]=tau[j][i]=0;
            }
            rowM[i]-=mc; rowM[j]-=mc; rcnt[i]--; rcnt[j]--;
        }
        compat[i]=ci; compat[j]=cj; cnt[i][v]--; cnt[j][v]--;
        compatMin[i]=si_mn; compatMax[i]=si_mx; compatMin[j]=sj_mn; compatMax[j]=sj_mx;
    }
}

static void usage(void) {
    fprintf(stderr,"usage: backtrack N [--full] [--nocherry] [--weak] [--fpviol] [--t01 V] "
                   "[--dump PATH] [--progress S]\n");
    exit(1);
}

int main(int argc, char**argv){
    if (argc<2) usage();
    n = atoi(argv[1]); canon = 1;
    const char *dump=NULL;
    if (n<5 || n>=MAXN) { fprintf(stderr,"error: N must be in 5..%d\n",MAXN-1); return 1; }
    for(int a=2;a<argc;a++){
        if(!strcmp(argv[a],"--full")) canon=0;
        else if(!strcmp(argv[a],"--nocherry")) nocherry=1;
        else if(!strcmp(argv[a],"--weak")) weak=1;
        else if(!strcmp(argv[a],"--fpviol")){ fpviol=1; canon=0; }
        else if(!strcmp(argv[a],"--t01") && a+1<argc) t01=atoi(argv[++a]);
        else if(!strcmp(argv[a],"--dump") && a+1<argc) dump=argv[++a];
        else if(!strcmp(argv[a],"--progress") && a+1<argc) progress_every=atol(argv[++a]);
        else { fprintf(stderr,"error: unknown or incomplete option %s\n",argv[a]); usage(); }
    }
    for(int t=2;t<n;t++) Mc[t]=(long)t*(1L<<(n-1-t));
    TTGT=(long)(2*n-3)*(1L<<(n-1));                   /* (9): sum tau w = n - 3/2, scaled */
    for(int t=2;t<n;t++) tmpcnt[t]=0;
    gen(1L<<(n-2), n-1, 1L<<(n-3));
    if(nocherry){ int m=0;
        for(int x=0;x<NT;x++) if(VAL[x][2]==0){ if(m!=x){ for(int t=2;t<n;t++) VAL[m][t]=VAL[x][t]; ROWM[m]=ROWM[x]; } m++; }
        NT=m; }
    if(NT>128){ fprintf(stderr,"error: %d row types exceed the 128-bit mask\n",NT); return 2; }
    if(NT==0){ fprintf(stderr,"error: no row types\n"); return 2; }
    MINM=ROWM[0]; MAXM=ROWM[0];
    for(int x=1;x<NT;x++){ if(ROWM[x]<MINM)MINM=ROWM[x]; if(ROWM[x]>MAXM)MAXM=ROWM[x]; }
    for(int t=2;t<n;t++) for(int c=1;c<n;c++){ u128 m=0;
        for(int x=0;x<NT;x++) if(VAL[x][t]>=c) m |= ((u128)1)<<x; maskGE[t][c]=m; }
    NP=0; for(int i=0;i<n;i++) for(int j=i+1;j<n;j++){ pi[NP]=i; pj[NP]=j; NP++; }
    u128 all = (NT>=128) ? ~(u128)0 : (((u128)1<<NT)-1);
    for(int r=0;r<n;r++){ compat[r]=all; compatMin[r]=MINM; compatMax[r]=MAXM; }
    if (dump) { fx=fopen(dump,"w"); if(!fx){ perror(dump); return 1; } }
    t0=time(NULL); tlast=t0;
    stamp(stderr); fprintf(stderr," start n=%d, %d row types\n", n, NT);
    rec(0,0,0);
    if (fx && fclose(fx)!=0) { perror(dump); return 1; }
    double dt=difftime(time(NULL),t0);
    stamp(stderr); fprintf(stderr," done\n");
    printf("n=%d: survivors=%lld%s  non-tree=%lld  [%lld nodes, %.0fs, %d row-types]  ->  %s\n",
        n, surv, canon?" (canonical)":" (FULL)", nontree, nodes, dt, NT,
        nontree? "COUNTEREXAMPLE":"(4)-(9) => (10): SUFFICIENT");
    if(have_ex){ printf("  first non-tree:\n"); for(int i=0;i<n;i++){ printf("   ");
        for(int j=0;j<n;j++) printf("%d ",ex[i][j]); printf("\n"); } }
    return 0;
}
