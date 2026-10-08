"""Thread count and random seed for every LP the solver builds.

Both are read from the environment so that one launcher sets them for a whole run:
    BME_THREADS  required; the solver refuses to start without it, so no run can fall
                 back to the LP solver's default of one thread per core
    BME_SEED     optional; the LP solver's random seed
"""
import os


def apply_solver_config(c):
    if "BME_THREADS" not in os.environ:
        raise RuntimeError("set BME_THREADS (the LP solver's thread count) before running")
    c.parameters.threads.set(int(os.environ["BME_THREADS"]))
    if "BME_SEED" in os.environ:
        c.parameters.randomseed.set(int(os.environ["BME_SEED"]))
