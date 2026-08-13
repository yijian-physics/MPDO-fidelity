"""Extract fidelity lower bounds from multilayer_optimizarion.py output.

Each run saves a length-`sample` array of losses; entries for restarts that had
not finished yet are still 0, so a job killed by the queue time limit leaves a
partially filled array. loss = -|<M1|(1 x U)|M2>|, so F_lower = max(-loss) over
the completed restarts (Uhlmann: F is the max over ALL ancilla unitaries, and the
circuit is a restriction, hence a lower bound).

    python read_opt_results.py [N] [folder]     # default N=16
writes save_results/xxz/optfid_p0.2N<N>.json in the same format as the other
scans, so xxz_plot_renyi2.jl / xxz_plot_crossing.jl can read it.
"""
import glob
import json
import os
import re
import sys

import numpy as np

from file_io import File_access


def collect(N=16, folder="save_results/xxz", sample=20):
    D = File_access()
    pat = os.path.join(folder, f"p0.2N{N}", f"obcN{N}Delta*.txt")
    rows = []
    for f in glob.glob(pat):
        m = re.search(r"Delta(-?\d+\.\d+)", os.path.basename(f))
        if m is None:
            continue
        name = os.path.relpath(f, "save_results").replace("\\", "/")[:-4]
        v = np.asarray(D.get_back(name), dtype=float)
        done = v[v != 0]
        rows.append((float(m.group(1)), -done))  # -loss = fidelity bound
    rows.sort(key=lambda r: r[0])
    return rows, sample


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    folder = sys.argv[2] if len(sys.argv) > 2 else "save_results/xxz"
    rows, sample = collect(N, folder)
    if not rows:
        print(f"no results found for N={N} under {folder}")
        return

    print(f"N = {N},  p = 0.2 (p_paper = 0.1),  floor = 1 (depth 2)")
    print("Delta   done   F_lower(best)     mean       spread      rel.spread")
    best = []
    for d, F in rows:
        if len(F) == 0:
            print(f"{d:+5.1f}    0/{sample}   -- no restart completed --")
            best.append(float("nan"))
            continue
        sp = F.max() - F.min() if len(F) > 1 else 0.0
        print(f"{d:+5.1f}   {len(F):2d}/{sample}   {F.max():.6f}    {F.mean():.6f}   "
              f"{sp:.2e}    {sp/F.max():.1e}")
        best.append(F.max())

    ndone = [len(F) for _, F in rows]
    print(f"\nrestarts completed: min {min(ndone)}, max {max(ndone)}, "
          f"total {sum(ndone)}/{sample*len(rows)}")

    ## The number of completed restarts varies with Delta (the time limit bit
    ## hardest where the optimization is slowest), and max-of-n grows with n. That
    ## puts a bias on the estimate that is monotonic in Delta - i.e. along the very
    ## axis the scan is measuring. Truncating every Delta to the same number of
    ## restarts costs a little tightness but makes the Delta-dependence unbiased.
    k = min(ndone)
    best_k = [float(F[:k].max()) if len(F) >= k and k > 0 else float("nan")
              for _, F in rows]
    print(f"\nequal-restart estimate (best of the first {k} at every Delta):")
    print("Delta   best-of-all   best-of-%-2d    difference" % k)
    for (d, _), ba, bk in zip(rows, best, best_k):
        print(f"{d:+5.1f}   {ba:.6f}      {bk:.6f}     {ba-bk:+.2e}")

    for tag, vals in (("", best), (f"_first{k}", best_k)):
        out = os.path.join(folder, f"optfid_p0.2N{N}{tag}.json")
        with open(out, "w") as fh:
            json.dump({"p": 0.2,
                       "N_tot": [N],
                       "Delta_tot": [d for d, _ in rows],
                       "data": [vals]}, fh, indent=2)
        print(f"saved: {out}")


if __name__ == "__main__":
    main()
