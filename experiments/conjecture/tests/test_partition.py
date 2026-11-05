"""Known-answer tests of the piecewise no-cherry search (a few seconds on one core).

    python experiments/conjecture/tests/test_partition.py

1. With cherries allowed, the survivors of the canonical search are the trees counted in
   table "Canonical survivors of the enumeration" (published.json, canonical_survivors). Cut into the pieces of depth 1 to 3 that
   nocherry_search.py runs, the pieces must add up to the same count for n = 8, 9, 10, and
   coverage.py must report the pieces as covering the search.
2. Negative control: with one piece that has survivors deleted, coverage.py must report that
   prefix as missing (and the survivor total must drop).
3. Split and restart: splitting one piece one level deeper, the pieces still cover, and
   coverage.py lists the prefix as covered twice.
4. No cherry, n = 5..13: no survivor, whole search and pieces of depth 2 alike.
5. The n = 15 pieces: coverage.py reports exactly the slice 3_3_3_5_5_5_6 / t08 = 8 as
   missing, and reports the search as covered once its seven pieces t09 = 8..14 are added
   (here as stand-ins with zero survivors, in a temporary copy).
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import coverage  # noqa: E402

PUBLISHED = json.loads((REPO / "experiments/theory/published.json").read_text())
CANON = {int(k): v for k, v in PUBLISHED["canonical_survivors"].items()}


def search(binary, n, outdir, depth=None, split=None, cherries=False):
    cmd = [sys.executable, str(HERE / "nocherry_search.py")]
    cmd += ["top", str(n), str(binary), str(outdir), "--depth", str(depth)] if split is None else \
           ["split", str(n), str(binary), str(outdir), *split]
    if cherries:
        cmd.append("--with-cherries")
    subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)


def main():
    tmp = Path(tempfile.mkdtemp())
    binary = tmp / "backtrack_wide8"
    subprocess.run(["cc", "-O3", "-o", str(binary), str(HERE / "backtrack_wide8.c")], check=True)
    for n in (8, 9, 10):
        for depth in (1, 2, 3):
            d = tmp / f"cherries_n{n}_d{depth}"
            search(binary, n, d, depth=depth, cherries=True)
            rep = coverage.coverage(d, n, nocherry=False)
            assert rep["covered"] and not rep["missing"], rep
            assert rep["survivors"] == CANON[n], (n, depth, rep["survivors"], CANON[n])
            assert rep["nontree"] == 0, rep
        print(f"n={n}: pieces of depth 1-3 add up to {CANON[n]} survivors (table \"Canonical survivors of the enumeration\")")
        # negative control: remove the piece with the most survivors
        d = tmp / f"cherries_n{n}_d3"
        recs = [(coverage.parse_out(f)["survivors"], f) for f in d.glob(f"bt{n}_*.out")]
        k, f = max(recs)
        prefix, _ = coverage.parse_name(f.name, n)
        f.unlink()
        rep = coverage.coverage(d, n, nocherry=False)
        assert not rep["covered"] and rep["missing"] == [list(prefix)], rep
        assert rep["survivors"] == CANON[n] - k, rep
        # split the removed piece instead: covered again
        search(binary, n, d, split=["_".join(map(str, prefix))], cherries=True)
        rep = coverage.coverage(d, n, nocherry=False)
        assert rep["covered"] and rep["survivors"] == CANON[n], rep
        # and re-run the piece itself: covered twice, reported as such
        search(binary, n, d, split=["_".join(map(str, prefix[:-1]))], cherries=True)
        rep = coverage.coverage(d, n, nocherry=False)
        assert rep["covered"] and rep["survivors"] == CANON[n], rep
        assert {"prefix": list(prefix), "how": "finished piece and a complete split"} in rep["duplicates"], rep
        print(f"n={n}: a deleted piece is reported missing; its split restores the coverage")
    for n in range(5, 14):
        for depth in (0, 2):
            d = tmp / f"nocherry_n{n}_d{depth}"
            search(binary, n, d, depth=depth)
            rep = coverage.coverage(d, n)
            assert rep["covered"] and rep["survivors"] == 0, rep
    print("no cherry, n = 5..13: covered, no survivor (whole search and depth 2)")
    n15 = REPO / "results/conjecture/nocherry/n15"
    if n15.exists():
        rep = coverage.coverage(n15, 15)
        pending = [3, 3, 3, 5, 5, 5, 6, 8]
        if rep["missing"]:
            assert rep["missing"] == [pending], rep["missing"]
            d = tmp / "n15"
            shutil.copytree(n15, d)
            for t09 in range(8, 15):
                (d / coverage.piece_name(15, pending + [t09])).write_text(
                    "n=15: survivors=0 (canonical)  non-tree=0  [1 nodes, 0s, 225 row-types]  ->  stand-in\n")
            assert coverage.coverage(d, 15)["covered"]
            print("n = 15: only 3_3_3_5_5_5_6 / t08 = 8 is missing; with its seven pieces it is covered")
        else:
            assert rep["covered"], rep
            print("n = 15: covered")
    shutil.rmtree(tmp)
    print("all tests passed")


if __name__ == "__main__":
    main()
