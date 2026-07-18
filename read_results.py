"""
read_results.py

Reads the optimization output files in save_results/ (or a sub-folder of it)
and reports the maximal value in each file. Can also print a LaTeX table and
produce a plot in the style of the Ising_benchmark.ipynb "Plot" section.

Each result file is a pickled numpy array (shape = (sample,)) of the per-sample
losses saved by `File_access.save_data` in multilayer_optimizarion.py. Files use
a `.txt` extension but are binary pickles, not text.

Because `loss_save = np.zeros(sample)` is saved incrementally after each sample,
an unfinished run leaves trailing 0.0 entries. Those are treated as
"not yet computed" and excluded from the max.

Filenames encode the run parameters, e.g.:
    codeX_p05obcN20lr0.002num_steps10000sample20staircasedepth2.txt
      p=05  bc=obc  N=20  num_steps=10000  sample=20  depth=2

The optimizer minimizes a loss = -(fidelity), so the plotted quantity is
-max (the best fidelity per run) unless --raw is given.

Usage:
    python read_results.py                          # scan ./save_results
    python read_results.py --folder codeXp05        # scan ./save_results/codeXp05
    python read_results.py <dir>                    # scan an arbitrary folder
    python read_results.py <file.txt>               # inspect a single file
    python read_results.py --folder codeXp05 --latex          # LaTeX table
    python read_results.py --folder codeXp05 --plot           # plot (linear axes)
    python read_results.py --folder codeXp05 --plot --loglog  # plot (log-log)
    python read_results.py --folder codeXp05 --plot --raw     # plot raw max
Optional filters (combine with any of the above):
    --prefix codeX_p05      only files whose name starts with this
    --since 2026-06-17      only files modified on/after this date
    --out figure.png        where to save the plot (default: figures/<folder>.png)
"""

import os
import re
import sys
import glob
import pickle
from datetime import datetime

import numpy as np

# --- numpy version-compat shim -------------------------------------------
# Data files were pickled with numpy 2.x (module layout `numpy._core.*`).
# If this environment has numpy 1.x (`numpy.core.*`), alias the module names so
# the old numpy can still unpickle them. Harmless on numpy 2.x.
try:
    import numpy._core  # noqa: F401  (exists on numpy>=2)
except ImportError:
    import numpy.core as _np_core
    sys.modules["numpy._core"] = _np_core
    for _sub in ("multiarray", "umath", "numeric", "_multiarray_umath"):
        try:
            _m = __import__(f"numpy.core.{_sub}", fromlist=[_sub])
            sys.modules[f"numpy._core.{_sub}"] = _m
        except ImportError:
            pass
# -------------------------------------------------------------------------

# color palette taken from Ising_benchmark.ipynb (ColorBrewer Dark2)
COLOR_SET = ["#1B9E77", "#D95F02", "#7570B3", "#E7298A",
             "#66A61E", "#E6AB02", "#A6761D", "#666666"]
MARKERS = ["o", "^", "s", "v", "D", "P"]


def load_array(path):
    """Load one saved result file (a pickled numpy array)."""
    with open(path, "rb") as f:
        return np.asarray(pickle.load(f))


def parse_params(filename):
    """Pull the run parameters out of a filename. Missing keys -> None."""
    name = os.path.basename(filename)
    patterns = {
        "p": r"p(\d+)",
        "bc": r"(obc|pbc)",
        "N": r"N(\d+)",
        "num_steps": r"num_steps(\d+)",
        "sample": r"sample(\d+)",
        "depth": r"depth(\d+)",
    }
    out = {}
    for key, pat in patterns.items():
        m = re.search(pat, name)
        out[key] = m.group(1) if m else None
    m = re.search(r"sample\d+([a-zA-Z]+)depth", name)
    out["framework"] = m.group(1) if m else None
    return out


def collect_rows(folder, prefix=None, since=None):
    """Return a sorted list of per-file summary dicts."""
    pattern = (prefix + "*.txt") if prefix else "*.txt"
    files = sorted(glob.glob(os.path.join(folder, pattern)))
    if since is not None:
        files = [f for f in files if os.path.getmtime(f) >= since]

    rows = []
    for path in files:
        try:
            arr = load_array(path)
        except Exception as e:
            print(f"  [skip] {os.path.basename(path)}: {e}")
            continue
        done = arr[arr != 0.0]          # exclude not-yet-computed samples
        p = parse_params(path)
        rows.append({
            "file": os.path.basename(path),
            "N": int(p["N"]) if p["N"] else -1,
            "depth": int(p["depth"]) if p["depth"] else -1,
            "p": p["p"],
            "n_done": done.size,
            "n_total": arr.size,
            "max": float(np.max(done)) if done.size else float("nan"),
            "min": float(np.min(done)) if done.size else float("nan"),
            "mean": float(np.mean(done)) if done.size else float("nan"),
        })
    rows.sort(key=lambda r: (str(r["p"]), r["depth"], r["N"]))
    return rows


def print_table(rows):
    if not rows:
        print("No matching result files found.")
        return
    header = (f"{'file':<62} {'N':>3} {'depth':>5} {'p':>4} "
              f"{'done/all':>9} {'max':>14} {'min':>14} {'mean':>14}")
    print(header)
    print("-" * len(header))
    for r in rows:
        flag = "  <-- incomplete" if r["n_done"] < r["n_total"] else ""
        done_all = f"{r['n_done']}/{r['n_total']}"
        print(f"{r['file']:<62} {r['N']:>3} {r['depth']:>5} {str(r['p']):>4} "
              f"{done_all:>9} {r['max']:>14.8f} {r['min']:>14.8f} "
              f"{r['mean']:>14.8f}{flag}")
    valid = [r for r in rows if not np.isnan(r["max"])]
    if valid:
        best = max(valid, key=lambda r: r["max"])
        print("-" * len(header))
        print(f"Overall maximal value: {best['max']:.8f}  "
              f"(N={best['N']}, depth={best['depth']}, p={best['p']}, "
              f"from {best['file']})")


def print_latex_table(rows):
    """Emit a LaTeX table pivoted by N: one column per depth."""
    depths = sorted({r["depth"] for r in rows})
    by_N = {}
    for r in rows:
        by_N.setdefault(r["N"], {})[r["depth"]] = r["max"]

    def fmt(v):
        return "--" if v is None or np.isnan(v) else f"{v:.8f}"

    col_spec = "c" * (1 + len(depths))
    depth_headers = " & ".join(f"max (depth={d})" for d in depths)
    print(r"\begin{table}[htbp]")
    print(r"  \centering")
    print(rf"  \begin{{tabular}}{{{col_spec}}}")
    print(r"    \hline")
    print(rf"    $N$ & {depth_headers} \\")
    print(r"    \hline")
    for N in sorted(by_N):
        cells = " & ".join(fmt(by_N[N].get(d)) for d in depths)
        print(f"    {N} & {cells} \\\\")
    print(r"    \hline")
    print(r"  \end{tabular}")
    print(r"  \caption{Maximal value per run.}")
    print(r"  \label{tab:max_results}")
    print(r"\end{table}")


def plot_results(rows, out_path, loglog=False, raw=False, ylabel=None):
    """Plot best value vs N, one curve per depth, in the notebook's style."""
    import matplotlib
    matplotlib.use("Agg")               # no display needed; save to file
    import matplotlib.pyplot as plt

    depths = sorted({r["depth"] for r in rows})
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=400)

    for i, d in enumerate(depths):
        pts = sorted((r["N"], r["max"]) for r in rows
                     if r["depth"] == d and not np.isnan(r["max"]))
        if not pts:
            continue
        Ns = [n for n, _ in pts]
        ys = [(m if raw else -m) for _, m in pts]   # loss -> fidelity
        ax.plot(Ns, ys,
                marker=MARKERS[i % len(MARKERS)],
                linestyle="--" if i else "-",
                markersize=6, markeredgewidth=0, linewidth=1.5,
                color=COLOR_SET[i % len(COLOR_SET)],
                label=f"depth {d}")

    if loglog:
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks([8, 16, 32]); ax.set_xticklabels([8, 16, 32])

    if ylabel is None:
        ylabel = r"$\max\ \mathrm{loss}$" if raw else r"$F$"
    ax.set_xlabel(r"$N$", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.tick_params(labelsize=12)
    for spine in ax.spines.values():     # framestyle=:box
        spine.set_visible(True)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved plot to: {out_path}")


def main():
    args = sys.argv[1:]
    prefix = since = folder = out_path = None
    latex = plot = loglog = raw = False

    rest = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--prefix" and i + 1 < len(args):
            prefix = args[i + 1]; i += 2
        elif a == "--since" and i + 1 < len(args):
            since = datetime.strptime(args[i + 1], "%Y-%m-%d").timestamp(); i += 2
        elif a == "--folder" and i + 1 < len(args):
            folder = args[i + 1]; i += 2
        elif a == "--out" and i + 1 < len(args):
            out_path = args[i + 1]; i += 2
        elif a == "--latex":
            latex = True; i += 1
        elif a == "--plot":
            plot = True; i += 1
        elif a == "--loglog":
            loglog = True; i += 1
        elif a == "--raw":
            raw = True; i += 1
        else:
            rest.append(a); i += 1

    # resolve where to read from
    base = os.path.join(os.getcwd(), "save_results")
    if rest:
        target = rest[0]
    elif folder:
        target = os.path.join(base, folder)
    else:
        target = base

    # single-file inspection
    if os.path.isfile(target):
        arr = load_array(target)
        done = arr[arr != 0.0]
        print(f"File: {target}")
        print(f"  shape       : {arr.shape}")
        print(f"  values      : {arr}")
        print(f"  done/total  : {done.size}/{arr.size}")
        if done.size:
            print(f"  max         : {np.max(done):.8f}")
            print(f"  min         : {np.min(done):.8f}")
            print(f"  mean        : {np.mean(done):.8f}")
        else:
            print("  (no completed samples yet)")
        return

    if not os.path.isdir(target):
        print(f"Not found: {target}")
        return

    rows = collect_rows(target, prefix=prefix, since=since)
    if not rows:
        print(f"No matching .txt result files found in: {target}")
        return

    if latex:
        print_latex_table(rows)
    else:
        print_table(rows)

    if plot:
        if out_path is None:
            tag = folder or os.path.basename(os.path.normpath(target))
            out_path = os.path.join(os.getcwd(), "figures", f"{tag}.png")
        plot_results(rows, out_path, loglog=loglog, raw=raw)


if __name__ == "__main__":
    main()
