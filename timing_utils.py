"""
timing_utils.py
Prints timestamped "what's loading right now" lines to the terminal — the
same pattern from the main project, kept here since it's genuinely useful
for a POC: you want to see, in order, what's happening and how long each
step took (model load vs. chunking vs. embedding vs. ChromaDB vs. Gemini).
"""
import time
from contextlib import contextmanager


@contextmanager
def stage(label: str):
    """
    with stage("Loading embedding model"):
        ...code...
    Prints '>> Loading embedding model ...' then '   done (X.XXs)'.
    """
    print(f">> {label} ...", flush=True)
    t0 = time.perf_counter()
    yield
    print(f"   done ({time.perf_counter() - t0:.2f}s)", flush=True)
