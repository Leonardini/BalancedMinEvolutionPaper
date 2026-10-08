"""Initial incumbent from FastME: balanced-ME greedy tree, then balanced NNI and SPR.

Requires the `fastme` binary (FastME 2.1.6) on PATH. The tree is returned in the
representation of nni_incumbent (networkx.Graph, leaves 1..n, internal n+1..2n-2).
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

import networkx as nx

from .nni_incumbent import bme_obj_from_tree, nj_plus_nni, w_from_tree


def _parse_newick(text, n):
    """Unrooted binary Newick with leaf labels t1..tn -> networkx.Graph."""
    s = text.strip().rstrip(";")
    G = nx.Graph()
    next_internal = [n + 1]
    pos = [0]

    def label():
        start = pos[0]
        while pos[0] < len(s) and s[pos[0]] not in ",():":
            pos[0] += 1
        name = s[start:pos[0]]
        if pos[0] < len(s) and s[pos[0]] == ":":  # skip the branch length
            pos[0] += 1
            while pos[0] < len(s) and s[pos[0]] not in ",()":
                pos[0] += 1
        return name

    def subtree():
        if s[pos[0]] == "(":
            node = next_internal[0]
            next_internal[0] += 1
            pos[0] += 1
            while True:
                G.add_edge(node, subtree())
                if s[pos[0]] == ",":
                    pos[0] += 1
                    continue
                pos[0] += 1  # ')'
                break
            label()
            return node
        name = label()
        if not name.startswith("t"):
            raise ValueError(f"unexpected leaf label {name!r} in FastME output")
        return int(name[1:])

    root = subtree()
    if G.degree(root) == 2:  # a rooted output: suppress the degree-2 root
        a, b = list(G.neighbors(root))
        G.remove_node(root)
        G.add_edge(a, b)
    if sorted(v for v in G if G.degree(v) == 1) != list(range(1, n + 1)):
        raise ValueError("FastME tree does not have leaves 1..n")
    if any(G.degree(v) != 3 for v in G if v > n):
        raise ValueError("FastME tree is not binary")
    return nx.relabel_nodes(G, {v: i for i, v in enumerate(sorted(u for u in G if u > n), n + 1)})


def fastme_tree(D):
    n = D.shape[0]
    exe = shutil.which("fastme")
    if exe is None:
        raise RuntimeError("fastme not found on PATH")
    with tempfile.TemporaryDirectory() as tmp:
        inp, out = Path(tmp) / "d.phy", Path(tmp) / "t.nwk"
        rows = [f"{n}"] + [f"t{i + 1} " + " ".join(repr(float(x)) for x in D[i]) for i in range(n)]
        inp.write_text("\n".join(rows) + "\n")
        subprocess.run([exe, "-i", str(inp), "-o", str(out), "-m", "B", "-n", "B", "-s",
                        "-T", "1"], check=True, capture_output=True,
                       cwd=tmp)
        tree = _parse_newick(out.read_text(), n)
    return {"objective": bme_obj_from_tree(tree, D, n), "tree": tree, "w": w_from_tree(tree, n)}


def initial_incumbent(D, n):
    """The better of NJ + NNI and FastME (balanced ME + NNI + SPR)."""
    a = nj_plus_nni(D, n)
    b = fastme_tree(D)
    return b if b["objective"] < a["objective"] else a
