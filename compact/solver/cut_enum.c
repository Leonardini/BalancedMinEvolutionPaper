/* Minimum cut separating two vertex sets of a small undirected graph (m <= 62 vertices),
 * by shortest augmenting paths on the dense residual matrix, in double precision; a residual
 * capacity below tol counts as saturated.
 *
 *   C:       m x m symmetric capacities, row-major
 *   src:     bitmask of the vertices forced to the source side (nonempty)
 *   snk:     bitmask of the vertices forced to the sink side (nonempty, disjoint from src)
 *   side:    on return, the bitmask of the vertices on the source side
 *
 * Returns the cut value, recomputed from C over the returned side.
 */
#include <string.h>

#define MAXV 64

double min_cut(int m, const double *C, unsigned long long src, unsigned long long snk,
               double tol, unsigned long long *side)
{
    static double R[MAXV][MAXV];
    int s = m, t = m + 1, n = m + 2;
    double big = 1.0;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++) {
            R[i][j] = C[i * m + j];
            big += C[i * m + j];
        }
    for (int i = 0; i < n; i++) { R[i][s] = R[i][t] = 0.0; R[s][i] = R[t][i] = 0.0; }
    for (int a = 0; a < m; a++) {
        if (src >> a & 1ULL) R[s][a] = big;
        if (snk >> a & 1ULL) R[a][t] = big;
    }
    int parent[MAXV], queue[MAXV];
    for (;;) {
        for (int i = 0; i < n; i++) parent[i] = -1;
        parent[s] = s;
        int head = 0, tail = 0;
        queue[tail++] = s;
        while (head < tail && parent[t] < 0) {
            int u = queue[head++];
            for (int v = 0; v < n; v++)
                if (parent[v] < 0 && R[u][v] > tol) { parent[v] = u; queue[tail++] = v; }
        }
        if (parent[t] < 0) break;
        double push = 1e300;
        for (int v = t; v != s; v = parent[v])
            if (R[parent[v]][v] < push) push = R[parent[v]][v];
        for (int v = t; v != s; v = parent[v]) {
            R[parent[v]][v] -= push;
            R[v][parent[v]] += push;
        }
    }
    unsigned long long S = 0;
    for (int v = 0; v < m; v++)
        if (parent[v] >= 0) S |= 1ULL << v;
    *side = S;
    double value = 0.0;
    for (int i = 0; i < m; i++)
        if (S >> i & 1ULL)
            for (int j = 0; j < m; j++)
                if (!(S >> j & 1ULL)) value += C[i * m + j];
    return value;
}
