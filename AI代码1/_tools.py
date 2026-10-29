"""Assembly / execution helpers for AI写的.ipynb.

Two entry points:

  python _tools.py preserve_sections
      Rebuild AI写的.ipynb so that the Task 1.1 and Task 1.2 sections keep
      their executed outputs while the Task 1.3 section is regenerated from
      the original untouched backup. Use this after adding new cells so the
      expensive results are never lost.

  python _tools.py run [--all | --from N]
      Execute the notebook headlessly and write the outputs back.
"""
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "AI写的.ipynb")
T3_SOURCE = os.path.join(HERE, "_backup_AI写的.ipynb.bak")   # original file

os.environ.setdefault("MPLCONFIGDIR", os.path.join(HERE, ".mplcache"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)


def section_bounds(cells):
    """Return {name: (start, end)} for the three task sections."""
    marks = {}
    for i, c in enumerate(cells):
        if c.cell_type != "markdown":
            continue
        text = c.source
        if "Task 1.1" in text:
            marks["t1"] = i
        elif "Task 1.2" in text:
            marks["t2"] = i
        elif "Task 1.3" in text or "任务3" in text:
            marks["t3"] = i
    return marks


def preserve_sections():
    """Keep Task 1.1 and 1.2 outputs; restore Task 1.3 from the original file."""
    import nbformat as nbf

    nb = nbf.read(SRC, as_version=4)
    marks = section_bounds(nb.cells)
    if "t1" not in marks or "t2" not in marks:
        raise SystemExit(f"could not locate task sections: {marks}")
    t1_start = marks["t1"]
    t2_start = marks["t2"]
    t3_start = marks.get("t3", len(nb.cells))

    task12 = nb.cells[t1_start:t3_start]          # task 1.1 + task 1.2, with outputs
    head = nb.cells[:t1_start]                    # setup cells

    orig = nbf.read(T3_SOURCE, as_version=4)
    o_marks = section_bounds(orig.cells)
    t3 = orig.cells[o_marks.get("t3", len(orig.cells) - 2):]

    cells = head + task12 + list(t3)
    new = nbf.v4.new_notebook(cells=cells)
    new.metadata = nb.metadata
    nbf.write(new, SRC)
    nbf.validate(new)
    print(f"preserved task 1.1+1.2 ({len(task12)} cells), restored task 1.3 "
          f"({len(t3)} cells); total {len(cells)} cells, validated OK")
    for i, c in enumerate(new.cells):
        h = c.source.strip().split("\n")[0][:56] if c.source.strip() else "(EMPTY)"
        print(f"  [{i:2d}] {c.cell_type:8s} out={len(c.get('outputs', [])):2d} | {h}")


def run_all(start_at=0):
    """Execute cells [start_at:] headlessly and write outputs back."""
    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(SRC, as_version=4)
    saved = [c.cell_type for c in nb.cells]
    if start_at:
        for i, c in enumerate(nb.cells):
            if i < start_at:
                c.cell_type = "raw"

    print(f"executing from cell {start_at} ({len(nb.cells)} cells total)", flush=True)
    t0 = time.perf_counter()
    NotebookClient(nb, timeout=7200, kernel_name="python3",
                   resources={"metadata": {"path": HERE}},
                   allow_errors=False).execute()
    dt = time.perf_counter() - t0

    for c, t in zip(nb.cells, saved):
        c.cell_type = t
    nbformat.write(nb, SRC)
    print(f"executed in {dt:.1f}s ({dt/60:.1f} min), outputs written back")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd == "preserve_sections":
        preserve_sections()
    elif cmd == "run":
        start = 0
        if "--from" in sys.argv:
            start = int(sys.argv[sys.argv.index("--from") + 1])
        elif "--all" in sys.argv:
            start = 0
        else:
            start = 0
        run_all(start)
    else:
        print(__doc__)
