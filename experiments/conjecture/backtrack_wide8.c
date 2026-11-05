/* Exhaustive search for integer matrices tau satisfying conditions (4)-(9) with no cherry
 * (no entry tau_ij = 2), for the cherry conjecture (Discussion and open problems). Same search as
 * experiments/theory/backtrack.c (see its header for the pruning rules), with two changes:
 *   - the set of row types still compatible with a partial row is an array of WORDS 64-bit
 *     words instead of one 128-bit integer, so --nocherry runs at n = 15 (225 row types);
 *   - the search can be cut into independent pieces by fixing a prefix of leaf 0's row,
 *     --t01 V1 ... --tNN VNN fixing tau[0][1] = V1, ..., tau[0][NN] = VNN (0, the default,
 *     leaves an entry free). Leaf 0's row is in non-decreasing order, so the pieces whose
 *     prefixes are the non-decreasing sequences partition the search (nocherry_search.py).
 *
 * Usage: backtrack_wide8 N [--nocherry] [--weak] [--full] [--fpviol] [--t01 V .. --t<N-1> V]
 *   --nocherry  only row types with every tau >= 3
 *   --weak      weak triangle (slack >= 0) in place of the strong one
 *   --full      no ordering on leaf 0's row (do not combine with --tXX)
 *   --fpviol    require (10) to fail on the quartet {0,1,2,3}
 * stdout: one summary line "n=N: survivors=S (canonical|FULL)  non-tree=K  [M nodes, Ts,
 * R row-types]  -> ...", then the first non-tree survivor if there is one. stderr: a progress
 * line every 2^29 nodes. Every survivor that violates (10) is also written to
 * cex_n<N>[_nc|_wk|_fp][_t<prefix>].txt in the working directory (created, empty, at start).
 * The search itself is the one that produced the n = 15 results; only this comment and the
 * rejection of unknown or incomplete options were added for the public version. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <limits.h>
#define MAXN 22
#define MAXT 4096
#define WORDS 4                                          /* 256 bits: covers n<=15 (225 no-cherry types) */
typedef struct { unsigned long long w[WORDS]; } Mask;
static inline Mask m_and(Mask a, Mask b){ Mask r; for(int i=0;i<WORDS;i++) r.w[i]=a.w[i]&b.w[i]; return r; }
static inline int  m_nz (Mask a){ for(int i=0;i<WORDS;i++) if(a.w[i]) return 1; return 0; }

static int n, canon, nocherry, weak, fpviol;
static int tpre[MAXN];  /* tpre[j] != 0 fixes tau[0][j] = tpre[j] (the row-0 prefix of a piece) */
static int VAL[MAXT][MAXN];
static long ROWM[MAXT];
static int NT;
static Mask maskGE[MAXN][MAXN];
static long Mc[MAXN], TTGT, MINM, MAXM;
static int tau[MAXN][MAXN], cnt[MAXN][MAXN], rcnt[MAXN];
static long rowM[MAXN];
static Mask compat[MAXN];
static long compatMin[MAXN], compatMax[MAXN];
static int pi[MAXN*MAXN], pj[MAXN*MAXN], NP;
static long long surv=0, nontree=0, nodes=0;
static int ex[MAXN][MAXN], have_ex=0;
static FILE *fx;
static time_t t0;

static int tmpcnt[MAXN];
static void gen(long rem, int cleft, long maxp) {
    if (cleft==0) {
        if (rem==0) {
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

static void minmax_rowm(Mask mask, long*mn, long*mx){
    long lo=LONG_MAX, hi=LONG_MIN;
    for(int wi=0; wi<WORDS; wi++){ unsigned long long w=mask.w[wi];
        while(w){ int b=__builtin_ctzll(w); long r=ROWM[wi*64+b]; if(r<lo)lo=r; if(r>hi)hi=r; w&=w-1; } }
    *mn=lo; *mx=hi;
}

static int four_point_fail(void) {
    for (int a=0;a<n;a++) for (int b=a+1;b<n;b++) for (int c=b+1;c<n;c++) for (int d=c+1;d<n;d++){
        int s0=tau[a][b]+tau[c][d], s1=tau[a][c]+tau[b][d], s2=tau[a][d]+tau[b][c];
        int lo=s0,mid=s1,hi=s2,tmp;
        if(lo>mid){tmp=lo;lo=mid;mid=tmp;} if(mid>hi){tmp=mid;mid=hi;hi=tmp;} if(lo>mid){tmp=lo;lo=mid;mid=tmp;}
        if (mid!=hi || hi-lo<2) return 1;
    }
    return 0;
}

/* Progress: row 0 is assigned first and, in canonical mode, is non-decreasing, so the search
 * visits the completions of row 0 in lexicographic order. ways[j][v] counts the non-decreasing
 * completions of columns j..n-1 (fixed columns forced) whose column j-1 holds v; the rank of
 * the current row 0 among them says how far the search has gone and what remains. */
static long long ways[MAXN+1][MAXN];
static void count_ways(void){
    int lo = nocherry ? 3 : 2;
    for(int v=0;v<n;v++) ways[n][v]=1;
    for(int j=n-1;j>=1;j--) for(int prev=0;prev<n;prev++){
        long long w=0;
        for(int v=(prev>lo?prev:lo); v<n; v++) if(!tpre[j] || v==tpre[j]) w+=ways[j+1][v];
        ways[j][prev]=w; }
}
static void report_progress(int p){
    char row[3*MAXN+1]=""; int k=0;
    long long rank=0; int lo = nocherry ? 3 : 2, prev=0, full = (p >= n-1);
    for(int j=1;j<n;j++){
        if(j<=p){ int v=tau[0][j];
            k+=snprintf(row+k,sizeof row-k,j>1?",%d":"%d",v);
            if(canon) for(int u=(prev>lo?prev:lo); u<v; u++) if(!tpre[j] || u==tpre[j]) rank+=ways[j+1][u];
            prev=v; } }
    if(canon && full)
        fprintf(stderr,"   ... nodes=%lldM surv=%lld nontree=%lld t=%lds  row0=%s: %lld of %lld row-0 completions done (%.1f%%)\n",
                nodes/1000000, surv, nontree, (long)(time(NULL)-t0), row, rank, ways[1][0], 100.0*rank/(double)ways[1][0]);
    else
        fprintf(stderr,"   ... nodes=%lldM surv=%lld nontree=%lld t=%lds\n",
                nodes/1000000, surv, nontree, (long)(time(NULL)-t0));
}

static void rec(int p, long T, int ncomp) {
    if (((++nodes) & ((1LL<<29)-1))==0) report_progress(p);
    if (p==NP) { surv++; if(four_point_fail()){ nontree++; if(!have_ex){ memcpy(ex,tau,sizeof(tau)); have_ex=1; }
        if(fx){ for(int a=0;a<n;a++){ for(int b=0;b<n;b++) fprintf(fx,"%d ",tau[a][b]); fprintf(fx,"\n"); } fprintf(fx,"--\n"); fflush(fx); } } return; }
    int i=pi[p], j=pj[p];
    int prev = 0;
    if (canon && i==0 && j>1) prev = tau[0][j-1];
    if (fpviol && i==0 && j>4) prev = tau[0][j-1];
    for (int v=(nocherry?3:2); v<n; v++) {
        if (v<prev) continue;
        if (i==0 && tpre[j] && v!=tpre[j]) continue;      /* parallel partition on the sorted row-0 prefix */
        int bad=0, slk=weak?0:2;
        for (int k=0;k<i;k++){ int tik=tau[i][k],tjk=tau[j][k];
            if (v+tik<tjk+slk || v+tjk<tik+slk || tik+tjk<v+slk){bad=1;break;} }
        if (bad) continue;
        if (fpviol && i==2 && j==3) {
            int s0=tau[0][1]+v, s1=tau[0][2]+tau[1][3], s2=tau[0][3]+tau[1][2];
            int lo=s0,md=s1,hi=s2,t;
            if(lo>md){t=lo;lo=md;md=t;} if(md>hi){t=md;md=hi;hi=t;} if(lo>md){t=lo;lo=md;md=t;}
            if (md==hi && hi-lo>=2) continue;
        }
        cnt[i][v]++; cnt[j][v]++;
        Mask ci=compat[i], cj=compat[j];
        long si_mn=compatMin[i], si_mx=compatMax[i], sj_mn=compatMin[j], sj_mx=compatMax[j];
        compat[i] = m_and(ci, maskGE[v][cnt[i][v]]);
        compat[j] = m_and(cj, maskGE[v][cnt[j][v]]);
        if (m_nz(compat[i]) && m_nz(compat[j])) {
            long mc=Mc[v]; rowM[i]+=mc; rowM[j]+=mc; rcnt[i]++; rcnt[j]++;
            long dT=0; int dnc=0;
            if(rcnt[i]==n-1){dT+=rowM[i];dnc++;} else minmax_rowm(compat[i],&compatMin[i],&compatMax[i]);
            if(rcnt[j]==n-1){dT+=rowM[j];dnc++;} else minmax_rowm(compat[j],&compatMin[j],&compatMax[j]);
            long Tn=T+dT; int ncn=ncomp+dnc; long need=TTGT-Tn;
            long lo=0, hi=0;
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

int main(int argc, char**argv){
    if (argc < 2) { fprintf(stderr,"usage: %s N [--nocherry] [--weak] [--full] [--fpviol] [--t01 V .. --t<N-1> V]\n", argv[0]); return 2; }
    n = atoi(argv[1]); canon = 1;
    if (n < 4 || n >= MAXN) { fprintf(stderr,"error: N must be in [4, %d]\n", MAXN-1); return 2; }
    for(int a=2;a<argc;a++){
        if(!strcmp(argv[a],"--full")) canon=0;
        else if(!strcmp(argv[a],"--nocherry")) nocherry=1;
        else if(!strcmp(argv[a],"--weak")) weak=1;
        else if(!strcmp(argv[a],"--fpviol")){ fpviol=1; canon=0; }
        else if(strlen(argv[a])==5 && !strncmp(argv[a],"--t",3) && isdigit((unsigned char)argv[a][3]) && isdigit((unsigned char)argv[a][4])){
            int j = atoi(argv[a]+3);
            if (j < 1 || j >= n) { fprintf(stderr,"error: %s: the column must be in [1, N-1]\n", argv[a]); return 2; }
            if (a+1 >= argc) { fprintf(stderr,"error: %s needs a value\n", argv[a]); return 2; }
            int v = atoi(argv[a+1]);
            if (v < 2 || v >= n) { fprintf(stderr,"error: %s %s is outside [2, N-1]\n", argv[a], argv[a+1]); return 2; }
            tpre[j] = v; a++;
        }
        else { fprintf(stderr,"error: unknown option %s\n", argv[a]); return 2; }
    }
    for(int t=2;t<n;t++) Mc[t]=(long)t*(1L<<(n-1-t));
    TTGT=(long)(2*n-3)*(1L<<(n-1));
    for(int t=2;t<n;t++) tmpcnt[t]=0;
    gen(1L<<(n-2), n-1, 1L<<(n-3));
    if(nocherry){ int m=0;
        for(int x=0;x<NT;x++) if(VAL[x][2]==0){ if(m!=x){ for(int t=2;t<n;t++) VAL[m][t]=VAL[x][t]; ROWM[m]=ROWM[x]; } m++; }
        NT=m; }
    if(NT > WORDS*64){ fprintf(stderr,"ERROR: %d row-types exceed WORDS*64=%d bits (raise WORDS)\n",NT,WORDS*64); return 2; }
    MINM=ROWM[0]; MAXM=ROWM[0];
    for(int x=1;x<NT;x++){ if(ROWM[x]<MINM)MINM=ROWM[x]; if(ROWM[x]>MAXM)MAXM=ROWM[x]; }
    for(int t=2;t<n;t++) for(int c=1;c<n;c++){ Mask m; for(int wi=0;wi<WORDS;wi++)m.w[wi]=0;
        for(int x=0;x<NT;x++) if(VAL[x][t]>=c) m.w[x>>6]|=1ULL<<(x&63); maskGE[t][c]=m; }
    NP=0; for(int i=0;i<n;i++) for(int j=i+1;j<n;j++){ pi[NP]=i; pj[NP]=j; NP++; }
    Mask all; for(int wi=0;wi<WORDS;wi++)all.w[wi]=0;
    for(int x=0;x<NT;x++) all.w[x>>6]|=1ULL<<(x&63);
    for(int r=0;r<n;r++){ compat[r]=all; compatMin[r]=MINM; compatMax[r]=MAXM; }
    { char fn[160], suf[128]="";
      const char* mode = nocherry?"_nc" : weak?"_wk" : fpviol?"_fp" : "";
      int last = 0; for(int j=1;j<n;j++) if(tpre[j]) last = j;
      if(last){  /* the nine-value name of the t01..t09 pieces, extended when a later column is fixed */
          int k = snprintf(suf,sizeof suf,"%s_t",mode);
          for(int j=1;j<=(last>9?last:9);j++) k += snprintf(suf+k,sizeof suf-k,j>1?"_%d":"%d",tpre[j]);
      }
      else snprintf(suf,sizeof suf,"%s",mode);
      snprintf(fn,sizeof fn,"cex_n%d%s.txt",n,suf); fx=fopen(fn,"w"); }
    count_ways();
    t0=time(NULL);
    rec(0,0,0);
    double dt=difftime(time(NULL),t0);
    printf("n=%d: survivors=%lld%s  non-tree=%lld  [%lld nodes, %.0fs, %d row-types]  ->  %s\n",
        n, surv, canon?" (canonical)":" (FULL)", nontree, nodes, dt, NT,
        nontree? "COUNTEREXAMPLE":"(4)-(9) => (10): SUFFICIENT");
    if(have_ex){ printf("  first non-tree:\n"); for(int i=0;i<n;i++){ printf("   ");
        for(int j=0;j<n;j++) printf("%d ",ex[i][j]); printf("\n"); } }
    return 0;
}
