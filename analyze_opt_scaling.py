"""Scaling analysis of the variational fidelity scan at N = 8, 12, 16.

The optimizer returns a LOWER BOUND on F (depth-2 circuit vs. the unrestricted
Uhlmann maximization). Writing F_var = c(N, Delta) * F_exact, the effective
exponent picks up a spurious term,

    eta_var = -dlog F_var/dlog L = eta_exact - dlog c/dlog L,

so the analysis is only meaningful if c is close to Delta-independent. We have
exact F at N = 8 and 12 for every Delta, so c can be measured rather than assumed.
"""
import json
import os

import numpy as np


def load(path):
    with open(path) as f:
        d = json.load(f)
    return np.array(d["data"], dtype=float), d["N_tot"], np.array(d["Delta_tot"], dtype=float)


def main():
    folder = "save_results/xxz"
    var = {}
    for N in (8, 12, 16):
        A, _, D = load(os.path.join(folder, f"optfid_p0.2N{N}.json"))
        var[N] = dict(zip(np.round(D, 4), A[0]))
    Aex, Nex, Dex = load(os.path.join(folder, "fidelity_p0.2.json"))
    exact = {N: dict(zip(np.round(Dex, 4), Aex[i])) for i, N in enumerate(Nex)}

    Ds = sorted(var[8].keys())

    print("(1) variational bias  c = F_var / F_exact   (exact available for N=8,12)")
    print("Delta     c(N=8)    c(N=12)   c8 - c12")
    for d in Ds:
        c8 = var[8][d] / exact[8][d]
        c12 = var[12][d] / exact[12][d]
        print(f"{d:+5.1f}    {c8:.4f}    {c12:.4f}    {c8-c12:+.4f}")

    c8 = np.array([var[8][d] / exact[8][d] for d in Ds])
    c12 = np.array([var[12][d] / exact[12][d] for d in Ds])
    print(f"\n  c(N=8) : range {c8.min():.4f} - {c8.max():.4f}  (variation {c8.max()-c8.min():.4f})")
    print(f"  c(N=12): range {c12.min():.4f} - {c12.max():.4f}  (variation {c12.max()-c12.min():.4f})")

    print("\n(2) effective exponent  eta(L1,L2) = log[Q(L1)/Q(L2)] / log(L2/L1)")
    print("            ---- variational ----      ---- exact ----")
    print("Delta    (8,12)  (12,16)  (8,16)      (8,12)   var-exact(8,12)")
    ev812, ev1216, ee812 = [], [], []
    for d in Ds:
        e812 = np.log(var[8][d] / var[12][d]) / np.log(12 / 8)
        e1216 = np.log(var[12][d] / var[16][d]) / np.log(16 / 12)
        e816 = np.log(var[8][d] / var[16][d]) / np.log(16 / 8)
        x812 = np.log(exact[8][d] / exact[12][d]) / np.log(12 / 8)
        ev812.append(e812); ev1216.append(e1216); ee812.append(x812)
        print(f"{d:+5.1f}   {e812:6.3f}  {e1216:6.3f}  {e816:6.3f}      {x812:6.3f}   {e812-x812:+.3f}")

    ev812, ev1216, ee812 = map(np.array, (ev812, ev1216, ee812))

    def crossing(x, y1, y2):
        dy = y1 - y2
        for k in range(len(dy) - 1):
            if dy[k] * dy[k + 1] < 0:
                t = dy[k] / (dy[k] - dy[k + 1])
                return x[k] + t * (x[k + 1] - x[k])
        return float("nan")

    Da = np.array(Ds)
    print(f"\n(3) phenomenological-RG crossing  eta(8,12) = eta(12,16)")
    print(f"    variational : Delta_c = {crossing(Da, ev812, ev1216):.3f}")
    print(f"    (for reference, exact-vs-variational eta(8,12) differ by "
          f"{np.abs(ev812-ee812).min():.3f} to {np.abs(ev812-ee812).max():.3f})")


if __name__ == "__main__":
    main()
