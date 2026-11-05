"""Run the no-cherry search of the cherry conjecture (Discussion and open problems) in independent pieces.

    python nocherry_search.py top   N BIN OUTDIR [--depth K] [--workers W] [--with-cherries]
    python nocherry_search.py split N BIN OUTDIR PREFIX [PREFIX ...] [--workers W] [--with-cherries]

BIN is backtrack_wide8 (built from backtrack_wide8.c). A piece is one run of
`BIN N --nocherry --t01 v1 ... --tk vk`, the part of the search in which leaf 0's row
starts with v1 <= ... <= vk (the row is sorted, so v1 <= v2 <= ...).

  top    runs every piece of depth K: v1 is the smallest entry of some row type
         (rowtypes.first_entries; for any other v1 row 0 cannot be completed) and
         v1 <= v2 <= ... <= vK <= N-1. K = 0 runs the whole search as one piece.
  split  replaces each PREFIX (values joined by "_", e.g. 3_3_3_5_5_5_6_8) by its pieces one
         level deeper, PREFIX + (v,) for every v from the last value of PREFIX to N-1. A
         piece that is too slow is split in this way; coverage.py checks that the pieces
         finished in OUTDIR still cover the search.

Each piece writes OUTDIR/bt<N>_<v1>_..._<vk>.out (the enumerator's summary line; written
under .part and renamed when the run ends with exit code 0) and .err (its progress, a line
every 2^29 nodes). The enumerator's file of non-tree survivors is kept in OUTDIR/cex/ when
it holds a matrix, and removed when the piece has none.
Pieces with a finished .out are skipped, so the command can be stopped and restarted.
One timestamped line per start and end of a piece, and a line every --heartbeat seconds
(default 600) naming the pieces still running, go to stderr. --with-cherries drops
--nocherry, for the known-answer tests (tests/test_partition.py), where the survivors are
the trees of table "Canonical survivors of the enumeration".

The n = 15 search of the paper was run as `top 15 BIN DIR --depth 5`, then, for the pieces
too slow to finish, `split` on 3_3_3_4_5, 3_3_3_4_6, 3_3_3_4_7 and 3_3_3_5_5 twice (to
t06 and t07), on 3_3_3_4_5_6_9, 3_3_3_4_5_7_7, 3_3_3_4_5_7_8, 3_3_3_4_5_8_8 and
3_3_3_5_5_5_6 (to t08), and on the seven slowest of the resulting pieces (to t09); see
README.md for the exact list and the pieces still missing.
"""
import itertools
import re
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import coverage  # noqa: E402
import rowtypes  # noqa: E402


def say(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, file=sys.stderr, flush=True)


def top_prefixes(n, depth, nocherry=True):
    """Every non-decreasing prefix of length depth whose first value is the smallest entry of
    some row type and whose values are at most n - 1."""
    if depth == 0:
        return [()]
    if depth > coverage.MAX_PREFIX:
        raise ValueError(f"the enumerator fixes at most {coverage.MAX_PREFIX} entries")
    out = []
    for v1 in rowtypes.first_entries(n, nocherry):
        for rest in itertools.combinations_with_replacement(range(v1, n), depth - 1):
            out.append((v1,) + rest)
    return out


def split_prefixes(n, prefix, nocherry=True):
    if len(prefix) >= coverage.MAX_PREFIX:
        raise ValueError(f"{prefix}: the enumerator fixes at most {coverage.MAX_PREFIX} entries")
    return [prefix + (v,) for v in coverage.required_children(n, prefix, nocherry)]


def run_piece(n, binary, outdir, prefix, nocherry):
    out = outdir / coverage.piece_name(n, prefix)
    args = [str(binary), str(n)] + (["--nocherry"] if nocherry else [])
    for k, v in enumerate(prefix, 1):
        args += [f"--t0{k}", str(v)]
    part = out.with_name(out.name + ".part")
    t0 = time.time()
    with open(part, "w") as fo, open(out.with_suffix(".err"), "w") as fe:
        rc = subprocess.run(args, stdout=fo, stderr=fe, cwd=outdir / "cex").returncode
    if rc != 0:
        raise RuntimeError(f"{' '.join(args)} exited with {rc}; see {out.with_suffix('.err')}")
    part.rename(out)
    rec = coverage.parse_out(out)
    if rec is None:
        raise RuntimeError(f"{out}: no output")
    # The enumerator writes its non-tree survivors to a file named after the mode and the nine
    # prefix values (0 for a free entry); kept only when it holds a matrix.
    vals = list(prefix) + [0] * (coverage.MAX_PREFIX - len(prefix))
    cex = outdir / "cex" / (f"cex_n{n}{'_nc' if nocherry else ''}"
                            + (f"_t{'_'.join(map(str, vals))}" if prefix else "") + ".txt")
    if not cex.exists():
        raise RuntimeError(f"{cex}: the enumerator did not create its survivor file")
    if rec["nontree"] == 0:
        if cex.stat().st_size:
            raise RuntimeError(f"{cex}: not empty, but the summary reports no non-tree survivor")
        cex.unlink()
    rec.update(prefix=prefix, wall=time.time() - t0)
    return rec


def main():
    argv = sys.argv[1:]
    opts = {}
    for name in ("--depth", "--workers", "--heartbeat"):
        if name in argv:
            i = argv.index(name)
            opts[name] = int(argv[i + 1])
            del argv[i:i + 2]
    nocherry = "--with-cherries" not in argv
    argv = [a for a in argv if a != "--with-cherries"]
    if len(argv) < 4 or argv[0] not in ("top", "split") or any(a.startswith("--") for a in argv):
        sys.exit(__doc__)
    cmd, n, binary, outdir = argv[0], int(argv[1]), Path(argv[2]).resolve(), Path(argv[3])
    if cmd == "top":
        if len(argv) != 4:
            sys.exit(__doc__)
        jobs = top_prefixes(n, opts.get("--depth", 0), nocherry)
    else:
        if len(argv) < 5:
            sys.exit(__doc__)
        jobs = []
        for s in argv[4:]:
            if not re.fullmatch(r"\d+(_\d+)*", s):
                sys.exit(f"bad prefix {s!r}: values joined by _")
            jobs += split_prefixes(n, tuple(int(v) for v in s.split("_")), nocherry)
    workers, beat = opts.get("--workers", 1), opts.get("--heartbeat", 600)
    (outdir / "cex").mkdir(parents=True, exist_ok=True)
    todo = [p for p in jobs if not (outdir / coverage.piece_name(n, p)).exists()]
    say(f"n={n}{'' if nocherry else ' (with cherries)'}: {len(jobs)} pieces, {len(jobs) - len(todo)} "
        f"already finished, {len(todo)} to run on {workers} worker(s) -> {outdir}")
    surv = nontree = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        running = {ex.submit(run_piece, n, binary, outdir, p, nocherry): (p, time.time()) for p in todo}
        finished = 0
        while running:
            done, _ = wait(running, timeout=beat, return_when=FIRST_COMPLETED)
            if not done:
                now = time.time()
                say("still running: " + ", ".join(f"{'_'.join(map(str, p)) or 'whole'} ({now - t:.0f} s)"
                                                  for p, t in running.values()))
                continue
            for fut in done:
                p, _ = running.pop(fut)
                rec = fut.result()
                finished += 1
                surv += rec["survivors"]
                nontree += rec["nontree"]
                flag = "  *** SURVIVORS: a matrix satisfying (4)-(9) with no cherry ***" \
                    if nocherry and rec["survivors"] else ""
                say(f"[{finished}/{len(todo)}] {'_'.join(map(str, p)) or 'whole search'}: "
                    f"survivors={rec['survivors']} non-tree={rec['nontree']} nodes={rec['nodes']} "
                    f"({rec['wall']:.0f} s){flag}")
    if not any((outdir / "cex").iterdir()):
        (outdir / "cex").rmdir()
    say(f"n={n}: {len(todo)} pieces run, survivors={surv}, non-tree={nontree}; "
        f"check the coverage with coverage.py {outdir} {n}"
        f"{'' if nocherry else ' --with-cherries'}")


if __name__ == "__main__":
    main()
