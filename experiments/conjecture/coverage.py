"""Check that the pieces of a no-cherry search together cover the whole search for one n.

    python coverage.py DIR N [--with-cherries]

DIR holds the outputs of backtrack_wide8 written by nocherry_search.py (or by the original
n = 15 drivers): bt<N>.out for a whole search, bt<N>_<v1>_..._<vk>.out for the piece whose
leaf-0 row starts tau[0][1..k] = v1..vk, and bt<N>_full.out for a whole search without the
ordering of leaf 0's row. An "s" before a value (bt15_3_3_3_4_5_6_9_s9_10.out, the naming of
the original drivers) is ignored. An empty .out is a piece that was stopped before it
finished.

Leaf 0's row is sorted, so a prefix (v1..vk) is covered when a finished piece has exactly
that prefix, or when every one-longer prefix (v1..vk, v) with vk <= v <= N-1 is covered.
The empty prefix needs only the values v1 that are the smallest entry of some row type
(rowtypes.first_entries); any other v1 leaves no way to complete row 0. Exits 0 when the
search is covered and has no survivor, 1 otherwise, after printing the report: the
prefixes still missing, the prefixes covered more than once, and finished pieces that lie
outside the required ranges (empty by construction, but listed).
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rowtypes  # noqa: E402

SUMMARY = re.compile(r"^n=(\d+): survivors=(\d+) \((canonical|FULL)\)\s+non-tree=(\d+)\s+"
                     r"\[(\d+) nodes, (\d+)s, (\d+) row-types\]")
RC = re.compile(r"^rc=(\d+)$")
MAX_PREFIX = 9          # backtrack_wide8 fixes at most tau[0][1..9]


def piece_name(n, prefix, full=False):
    return f"bt{n}" + "".join(f"_{v}" for v in prefix) + ("_full" if full else "") + ".out"


def parse_name(name, n):
    """(prefix, full) of a piece file name, or None if the name is not a piece of n."""
    m = re.fullmatch(rf"bt{n}((?:_s?\d+)*)(_full)?\.out", name)
    if not m:
        return None
    prefix = tuple(int(t.lstrip("s")) for t in m.group(1).split("_")[1:])
    if m.group(2) and prefix:
        raise ValueError(f"{name}: a search without row ordering cannot have a prefix")
    if len(prefix) > MAX_PREFIX:
        raise ValueError(f"{name}: prefix longer than {MAX_PREFIX}")
    return prefix, bool(m.group(2))


def parse_out(path):
    """The summary of one piece, or None if the piece did not finish (empty file)."""
    text = Path(path).read_text()
    if not text.strip():
        return None
    lines = text.splitlines()
    hits = [m for m in map(SUMMARY.match, lines) if m]
    if len(hits) != 1:
        raise ValueError(f"{path}: expected one summary line, found {len(hits)}")
    for m in filter(None, map(RC.match, lines)):
        if int(m.group(1)) != 0:
            raise ValueError(f"{path}: the enumerator exited with {m.group(0)}")
    n, surv, mode, nontree, nodes, secs, types = hits[0].groups()
    return dict(n=int(n), survivors=int(surv), ordered=(mode == "canonical"),
                nontree=int(nontree), nodes=int(nodes), seconds=int(secs), row_types=int(types))


def read_pieces(d, n, nocherry=True):
    """{prefix: [record, ...]} of every piece file of n in d; unfinished pieces have
    record['done'] False. Every finished piece is checked against n, the row ordering and
    the number of row types."""
    want_types = len(rowtypes.nocherry_row_types(n) if nocherry else rowtypes.row_types(n))
    pieces = {}
    for f in sorted(Path(d).glob(f"bt{n}*.out")):
        parsed = parse_name(f.name, n)
        if parsed is None:
            continue
        prefix, full = parsed
        rec = parse_out(f)
        if rec is None:
            pieces.setdefault(prefix, []).append(dict(file=f.name, done=False))
            continue
        if rec["n"] != n:
            raise ValueError(f"{f}: summary is for n={rec['n']}")
        if rec["ordered"] == full:
            raise ValueError(f"{f}: name and summary disagree on the ordering of row 0")
        if rec["row_types"] != want_types:
            raise ValueError(f"{f}: {rec['row_types']} row types, expected {want_types}")
        if rec["nontree"] > rec["survivors"]:
            raise ValueError(f"{f}: more non-trees than survivors")
        rec.update(file=f.name, done=True)
        pieces.setdefault(prefix, []).append(rec)
    return pieces


def required_children(n, prefix, nocherry=True):
    if not prefix:
        return rowtypes.first_entries(n, nocherry)
    return list(range(prefix[-1], n))


def coverage(d, n, nocherry=True):
    """Coverage report for n. Survivors are counted once per part of the search: where a
    part is covered twice (a finished piece and its complete split, or two files), both
    counts must agree, which is checked."""
    pieces = read_pieces(d, n, nocherry)
    done = {p: [r for r in rs if r["done"]] for p, rs in pieces.items()}
    below = {p[:k] for p in pieces for k in range(len(p))}      # prefixes with pieces under them
    duplicates, visited = [], set()

    def walk(prefix):
        """(covered, missing prefixes, (survivors, non-trees)) under prefix."""
        visited.add(prefix)
        recs = done.get(prefix, [])
        counts = {(r["survivors"], r["nontree"]) for r in recs}
        if len(counts) > 1:
            raise ValueError(f"prefix {prefix}: finished pieces disagree: {counts}")
        if len(recs) > 1:
            duplicates.append(dict(prefix=list(prefix), how="several finished files",
                                   files=[r["file"] for r in recs]))
        if prefix in below:
            sub = [walk(prefix + (v,)) for v in required_children(n, prefix, nocherry)]
            split_ok = all(c for c, _, _ in sub)
            split_counts = (sum(s for _, _, (s, _) in sub), sum(t for _, _, (_, t) in sub))
            if recs and split_ok:
                duplicates.append(dict(prefix=list(prefix), how="finished piece and a complete split"))
                if counts != {split_counts}:
                    raise ValueError(f"prefix {prefix}: piece {counts} and its split {split_counts} disagree")
            elif recs and any(c for c, _, _ in sub):
                duplicates.append(dict(prefix=list(prefix), how="finished piece and a partial split"))
            if recs:
                return True, [], counts.pop()
            if split_ok:
                return True, [], split_counts
            return False, [m for _, ms, _ in sub for m in ms], split_counts
        if recs:
            return True, [], counts.pop()
        return False, [list(prefix)], (0, 0)

    covered, missing, (surv, nontree) = walk(())
    # Finished pieces that the walk never reached have a value outside its required range:
    # empty by construction (their leaf-0 row cannot be completed), so they must be empty.
    outside = sorted(p for p in done if done[p] and p not in visited)
    for p in outside:
        if any(r["survivors"] for r in done[p]):
            raise ValueError(f"piece {p} lies outside the required range but has survivors")
    finished = [r for rs in done.values() for r in rs]
    return dict(n=n, nocherry=nocherry, covered=covered, survivors=surv, nontree=nontree,
                pieces_done=len(finished),
                pieces_unfinished=sum(1 for rs in pieces.values() for r in rs if not r["done"]),
                pieces_used=sum(1 for p in visited if done.get(p)), pieces_outside=len(outside),
                nodes=sum(r["nodes"] for r in finished), cpu_seconds=sum(r["seconds"] for r in finished),
                missing=missing, duplicates=duplicates, outside=[list(p) for p in outside])


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 2:
        sys.exit(__doc__)
    nocherry = "--with-cherries" not in sys.argv
    rep = coverage(Path(args[0]), int(args[1]), nocherry)
    print(json.dumps(rep, indent=1))
    ok = rep["covered"] and rep["survivors"] == 0
    print(f"n={rep['n']}: {'COVERED, no survivor' if ok else 'NOT COVERED' if not rep['covered'] else 'SURVIVORS FOUND'}"
          f" ({rep['pieces_used']} pieces used, {len(rep['missing'])} prefixes missing)", file=sys.stderr)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
