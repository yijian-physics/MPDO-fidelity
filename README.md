This project is to compute fidelity of matrix product density operators using variational sequential circuit. 

Pipeline: Ising_init_data.jl   →   multilayer_optimizarion.py   →   read_results.py
            (generate data)          (optimize, save losses)       (read/table/plot)


Collaborators: Yuhan Liu and Yijian Zou


## Renyi-2 correlator of the decohered XXZ chain (resubmit/[Guo].pdf, Fig. 3)

    julia xxz_init_data.jl 0.2 renyi2       # scan -> save_results/xxz/renyi2_p0.2.json
    julia xxz_plot_renyi2.jl                # -> figures/renyi2_p0.2_fig3a.{png,pdf}
    julia xxz_plot_renyi2.jl renyi2_p0.2_obc   # the OBC scan (see below)

Computes C_X^II(i,j) = Tr[rho X_i X_j rho X_i X_j] / Tr[rho^2] for the ground state
of H = -sum_bonds (SxSx + SySy + Delta SzSz) on a periodic chain, decohered by the
two-site channel (1-p/2) rho + (p/2) XX rho XX on every bond, with the operators a
half-chain apart. The command-line p is twice the paper's: `0.2` here is p = 0.1
there, the value used for their Fig. 3(a)-(b).

The plot rescales by L^0.66, the paper's y-axis: at p = 0.1 the correlator decays as
L^-0.66 exactly on the critical line, so curves for different L cross at Delta_c = 0.
`renyi2_exponents()` fits the effective exponent from the data; over L = 18..48 it
gives 0.6612 at Delta = 0, against the paper's 0.66.

**Boundary conditions.** Pass `pbc=false` to `xxz_run_direct_renyi2` /
`xxz_renyi2_exact` for an open chain, with the operators at the literal sites L/4 and
3L/4 rather than at 1 and L/2+1. The paper's Fig. 3 is OBC: PBC reproduces the same
Delta_c and the same exponent but sits a factor ~2 higher (two paths connect two
points a half-chain apart on a ring), whereas OBC lands on their numbers - at
Delta = 0 we get L^0.66 C = 0.741 / 0.737 / 0.733 for L = 24 / 32 / 48 against their
crossing at ~0.73, and 0.369 at (Delta, L) = (-0.3, 32) against their ~0.37.
OBC is also ~4x cheaper, because the periodic bond's virtual leg no longer has to be
routed through the whole bulk.

Accuracy knobs, in decreasing order of importance (defaults in `xxz_renyi2_exact`):

| knob | default | effect |
| --- | --- | --- |
| `max_bd` | 384 | bond of the compressed rho-MPO; **this is what limits the accuracy**. At N=48 the answer moves 1.3% from 256 to 384 and 0.04% from 384 to 512. |
| `lpdo_max_err` | 1e-8 | discarded weight when re-canonicalizing the purification. Sets the runtime (cost ~ `max_bd * D^3`), not the accuracy: 1e-8 vs 1e-10 changes C by ~1e-6. |
| `dmrg_max_bd` | 192 | DMRG bond. Not the limiting factor: C moves by 1.5e-6 between 128 and 192. |

Note the ordering: `add_noise_double` inflates the purification bond to 4x the DMRG
bond, so a large DMRG bond is expensive *twice over* and buys nothing here. DMRG runs
with total-Sz conservation (`conserve_qns`), which is exact for |Delta| < 1 and about
6x faster than the dense version at N = 48.
