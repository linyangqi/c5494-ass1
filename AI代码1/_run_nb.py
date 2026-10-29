"""Execute AI写的.ipynb in place (outputs written back), optionally a cell range.

Usage:
  python _run_nb.py 0 14      # run cells 0..13 (setup + main task 1.1 results)
  python _run_nb.py 14 18     # run cells 14..17 (ablation + timing summary)
  python _run_nb.py           # run everything
"""
import os
import sys
import time

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplcache"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import nbformat
from nbclient import NotebookClient

SRC = "AI写的.ipynb"
nb = nbformat.read(SRC, as_version=4)

lo = int(sys.argv[1]) if len(sys.argv) > 1 else 0
hi = int(sys.argv[2]) if len(sys.argv) > 2 else len(nb.cells)

# keep only the requested code cells, in order
kept = [c for c in nb.cells[lo:hi] if c.cell_type == "code"]
print(f"executing {len(kept)} code cells from index range [{lo}, {hi})", flush=True)

def make_client(notebook):
    # nbclient >= 0.11 takes the notebook positionally in the constructor
    return NotebookClient(notebook, timeout=3600, kernel_name="python3",
                          resources={"metadata": {"path": os.getcwd()}},
                          allow_errors=False)


t0 = time.perf_counter()
# Execute the whole notebook but stop after the requested range by marking the
# remaining cells as raw temporarily.
if lo != 0 or hi != len(nb.cells):
    saved = [c.cell_type for c in nb.cells]
    for i, c in enumerate(nb.cells):
        if not (lo <= i < hi):
            c.cell_type = "raw"
    make_client(nb).execute()
    for c, t in zip(nb.cells, saved):
        c.cell_type = t
else:
    make_client(nb).execute()

print(f"executed in {time.perf_counter() - t0:.1f}s")
nbformat.write(nb, SRC)
print(f"wrote outputs back to {SRC}")
