"""Run the task lines of the counting runs, PARALLEL at a time.

    python run_tasks.py PARALLEL TASKFILE...

Each non-blank line not starting with # is one shell command, run with bash from the
repo root in the current environment. A task's stdout and stderr go to
results/counting_logs/<file>_<line>.log; a finished task leaves <...>.log.done and is
skipped on a rerun. One line per start and end goes to stdout.
"""
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

repo = Path(os.environ["REPO"])
logs = Path(os.environ["RESULTS"]) / "counting_logs"
logs.mkdir(parents=True, exist_ok=True)


def stamp():
    return time.strftime("%F %T")


tasks = []
for f in sys.argv[2:]:
    for k, line in enumerate(Path(f).read_text().splitlines(), 1):
        if line.strip() and not line.lstrip().startswith("#"):
            tasks.append((f"{Path(f).stem}_{k}", line))


def run(task):
    tid, cmd = task
    log, done = logs / f"{tid}.log", logs / f"{tid}.log.done"
    if done.exists():
        print(f"{stamp()} skip (done) {tid}", flush=True)
        return
    print(f"{stamp()} start {tid}: {cmd}", flush=True)
    with open(log, "w") as fh:
        rc = subprocess.run(["bash", "-c", cmd], cwd=repo, stdout=fh, stderr=subprocess.STDOUT).returncode
    if rc == 0:
        done.touch()
    print(f"{stamp()} end   {tid} {'ok' if rc == 0 else f'FAILED (exit {rc})'}", flush=True)


with ThreadPoolExecutor(max_workers=int(sys.argv[1])) as pool:
    list(pool.map(run, tasks))
