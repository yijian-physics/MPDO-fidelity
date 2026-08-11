using LinearAlgebra,TensorOperations,LinearMaps,Arpack,ITensors, ITensorMPS

using GLM
using DataFrames
using LaTeXStrings
using HDF5
using Printf

include("evolMPDO.jl")
include("IsingED.jl")  # provides MPS_to_array


function xxz_hamiltonian(N::Int, Delta::Float64, pbc::Bool, sites; conserve_qns=true)
    ## H = -sum_bonds (Sx Sx + Sy Sy + Delta Sz Sz).
    ## With conserve_qns we must spell SxSx + SySy as (S+S- + S-S+)/2: it is the
    ## same operator, but written in a form that manifestly commutes with total Sz,
    ## which is what lets ITensors use block-sparse (quantum-number) tensors.
    ampo = OpSum()
    bonds = pbc ? vcat([(j, j+1) for j in 1:N-1], [(N, 1)]) : [(j, j+1) for j in 1:N-1]
    for (a, b) in bonds
        if conserve_qns
            ampo += -0.5,"S+",a,"S-",b
            ampo += -0.5,"S-",a,"S+",b
        else
            ampo += -1.0,"Sx",a,"Sx",b
            ampo += -1.0,"Sy",a,"Sy",b
        end
        ampo += -1.0*Delta,"Sz",a,"Sz",b
    end
    return MPO(ampo, sites)
end

function XXZ_GS_DMRG(N, Delta=1.0, pbc=true; max_bd=256, nsweeps=24,
                     conserve_qns=true, cutoff=1E-11, energy_tol=1E-10, verbose=false)
    ## Ground state of the XXZ chain.
    ##
    ## conserve_qns = true keeps total Sz as a quantum number. The ground state of
    ## the |Delta| < 1 chain lives in the Sz = 0 sector, so this is exact, and the
    ## block-sparse tensors make DMRG several times faster (and cut its memory).
    ## The returned MPS is densified, so everything downstream is unchanged.
    ##
    ## The bond dimension is ramped up gradually and the sweeps stop early once the
    ## energy is converged to energy_tol - a fixed, large sweep count spends most of
    ## its time re-converging an already converged state.
    sites = siteinds("S=1/2", N; conserve_qns=conserve_qns)
    H = xxz_hamiltonian(N, Delta, pbc, sites; conserve_qns=conserve_qns)

    ## ramp: [16,32,64,...] up to max_bd, then hold
    maxdim = Int[]
    d = 16
    while d < max_bd
        push!(maxdim, d)
        d *= 2
    end
    push!(maxdim, max_bd)
    noise = [1E-5,1E-6,1E-7,1E-8,1E-8,1E-9,0.0]

    psi0 = conserve_qns ?
        random_mps(sites, [isodd(n) ? "Up" : "Dn" for n in 1:N]; linkdims=10) :
        random_mps(sites; linkdims=10)

    obs = DMRGObserver(; energy_tol=energy_tol, minsweeps=length(maxdim) + 4)
    E0, psi = dmrg(H, psi0; nsweeps=nsweeps, maxdim=maxdim, cutoff=[cutoff],
                   noise=noise, observer=obs, outputlevel=(verbose ? 1 : 0))

    return dense(psi), E0
end


function xxz_get_lpdo(N::Int, Delta::Float64=1.0; p1=1.0, divide=2, output=1,
                      dmrg_max_bd=256, dmrg_nsweeps=24, with_M2=true, pbc=true)
    # output is purification tensor (half of LPDO)

    psiMPS, E0 = XXZ_GS_DMRG(N, Delta, pbc; nsweeps=dmrg_nsweeps, max_bd=dmrg_max_bd)
    A = myMPS(MPS_to_array(psiMPS));

    Sx = [0 1; 1 0]
    i, j = 1, Int(N/divide)+1  # fix the ratio to be 1/divide

    M1 = add_noise_double(A, p1; pbc=pbc)
    ## M2 is only used by the fidelity route; building it copies the whole LPDO,
    ## which is the largest object around, so skip it when only M1 is wanted.
    M2 = with_M2 ? add_CP(add_CP(M1, Sx, i), Sx, j) : nothing

    M1_tot, M2_tot = [], []
    if output == 1
        # for output data to python
        M1 = myMPDO_to_array(M1)
        M2 = myMPDO_to_array(M2)
    end

    push!(M1_tot, M1)
    push!(M2_tot, M2)

    return M1_tot, M2_tot
end

function output_data(data, data_name; string="arr")

    N = length(data)

    path = "gapless_data/xxz/" * data_name * ".h5"
    mkpath(dirname(path))  # h5open("w") does not create directories

    h5open(path, "w") do f
        for i in 1:N
            f[string * "_$i"] = data[i]
        end
    end
end

function save_array_json(A::Matrix, N_tot, Delta_tot, p; name="fidelity_array", folder="save_results/xxz")
    ## Save a result array (rows: N, columns: Delta) plus metadata to a JSON file.
    ## Dependency-free serialization (no JSON package required).
    mkpath(folder)
    path = joinpath(folder, name * ".json")
    numvec(v) = "[" * join((string(x) for x in v), ", ") * "]"
    open(path, "w") do f
        println(f, "{")
        println(f, "  \"p\": $(p),")
        println(f, "  \"N_tot\": $(numvec(collect(N_tot))),")
        println(f, "  \"Delta_tot\": $(numvec(collect(Delta_tot))),")
        println(f, "  \"data\": [")
        nrow = size(A, 1)
        for i in 1:nrow
            sep = i < nrow ? "," : ""
            println(f, "    $(numvec(A[i, :]))$sep")
        end
        println(f, "  ]")
        println(f, "}")
    end
    println("saved: $path")
    return path
end


######### Benchmark: exact dense channel vs MPDO construction #########

function apply_XX_channel_dense(rho::Matrix, p::Float64, i::Int, j::Int, N::Int)
    ## Apply (1-p/2)*rho + (p/2)*XX rho XX on sites (i,j) of a dense rho.
    ## Convention: site 1 is the fastest index (matches MPS_to_dense/MPDO_to_dense).
    X = [0.0 1.0; 1.0 0.0]
    Id = [1.0 0.0; 0.0 1.0]
    O = ones(1, 1)
    for k in 1:N
        op = (k == i || k == j) ? X : Id
        O = kron(op, O)  # new site on the left => site 1 stays fastest
    end
    return (1 - p/2)*rho + (p/2)*(O*rho*O)
end

function add_noise_double_dense(psi::Vector, p::Float64)
    ## Exact dense version of add_noise_double: channel on every bond of the
    ## periodic chain (even layer, odd layer, boundary bond (N,1)).
    N = Int(log2(length(psi)))
    rho = psi*psi'
    for i in 1:2:N-1
        rho = apply_XX_channel_dense(rho, p, i, i+1, N)
    end
    for i in 2:2:N-1
        rho = apply_XX_channel_dense(rho, p, i, i+1, N)
    end
    rho = apply_XX_channel_dense(rho, p, N, 1, N)
    return rho
end

function benchmark_add_noise_double(N::Int=6, Delta::Float64=1.0; p::Float64=0.2)
    ## Small-N check (N=6,8) that the MPDO construction and the exact dense
    ## calculation give the same density matrix. Returns the relative error.
    psiMPS, E0 = XXZ_GS_DMRG(N, Delta, true, nsweeps=30, max_bd=100)
    A = myMPS(MPS_to_array(psiMPS))

    # (1) MPDO/LPDO construction
    M1 = add_noise_double(A, p)
    half = MPDO_to_dense(M1)
    rho_mpdo = half*half'

    # (2) exact dense calculation
    psi = MPS_to_dense(A)
    rho_exact = add_noise_double_dense(psi, p)

    err = norm(rho_mpdo - rho_exact)/norm(rho_exact)
    println("N=$N, Delta=$Delta, p=$p: tr(rho_mpdo)=$(tr(rho_mpdo)), tr(rho_exact)=$(tr(rho_exact)), rel. error=$err")
    return err
end

########## direct fidelity calculation for small size #############
function xxz_fidelity_exact(N::Int, Delta::Float64; p1=1.0)

    # M2 = Sx_i Sx_j applied to M1 at i=1, j=N/2+1 (set inside xxz_get_lpdo)
    M1, M2 = xxz_get_lpdo(N, Delta; p1=p1, divide=2, output=0)
    F0 = fidelity_exact(M1[1], M2[1])

    return F0
end

function xxz_renyi2_exact(N::Int, Delta::Float64; p1=1.0, max_bd=384, max_err=1E-11,
                          lpdo_max_err=1E-8, dmrg_max_bd=192, dmrg_nsweeps=20,
                          verbose=true, pbc=true)
    ## Renyi-2 correlator C = tr(X_i X_j rho X_i X_j rho)/tr(rho^2), the quantity
    ## used in 2503.14221 (their C_X^II) with O = X at i=1, j=N/2+1.
    ## NOTE: their channel is (1-p)rho + p XX rho XX, ours is (1-p/2)rho + (p/2)...,
    ## so p1_here = 2 * p_paper (e.g. p1=0.2 <-> paper p=0.1, used for their Fig. 3).
    ## Evaluated with the efficient (on-the-fly compressed) MPO contraction.
    ##
    ## The two knobs that actually set the runtime:
    ##  * lpdo_max_err - discarded squared Schmidt weight when the purification is
    ##    re-canonicalized. add_noise_double inflates the purification bond D to 4x
    ##    the DMRG bond, and the MPO compression costs ~ b*D^3, so shrinking D back
    ##    down is the single biggest win. 1e-8 changes C by ~1e-6 (checked at N=32
    ##    against 1e-10) and roughly halves the runtime.
    ##  * max_bd - the rho-MPO bond b. This is what actually limits the accuracy of
    ##    C: at N=32 the spread over b = 128..384 is ~0.5%, everything else is at
    ##    the 1e-6 level.
    ## dmrg_max_bd = 192 is likewise not the limiting factor - at N=32, C moves by
    ## 1.5e-6 between DMRG bond 128 and 192.
    t0 = time()
    M1, _ = xxz_get_lpdo(N, Delta; p1=p1, divide=2, output=0,
                         dmrg_max_bd=dmrg_max_bd, dmrg_nsweeps=dmrg_nsweeps,
                         with_M2=false, pbc=pbc)
    verbose && println(" --- lpdo done ($(round(time()-t0,digits=1)) s, bond $(max_bond_dim(M1[1]))) ---")
    Sx = [0 1; 1 0]
    ## PBC is translation invariant, so sites 1 and N/2+1 are the paper's L/4 and
    ## 3L/4. Without translation invariance we have to use those sites literally.
    i, j = pbc ? (1, Int(N/2) + 1) : (div(N, 4), 3*div(N, 4))
    C = renyi2_correlator(M1[1], Sx, Sx, i, j; max_bd=max_bd, max_err=max_err,
                          lpdo_max_err=lpdo_max_err, lpdo_max_bd=4096, method=:mpo)
    verbose && println(" --- C = $C  (total $(round(time()-t0,digits=1)) s) ---")
    return C
end

function fidelity_to_latex(A::Matrix, N_tot, Delta_tot; digits=6)
    ## Print fidelity_array (size length(N_tot) x length(Delta_tot))
    ## as a LaTeX table with rows: Delta, columns: N
    fmt(x) = Printf.format(Printf.Format("%.$(digits)f"), x)
    ncol = length(N_tot)
    println("\\begin{tabular}{c|" * "c"^ncol * "}")
    println("\\hline")
    println("\$\\Delta\$ & " * join(["\$N = $(N)\$" for N in N_tot], " & ") * " \\\\")
    println("\\hline")
    for (j, D) in enumerate(Delta_tot)
        println("\$$(D)\$ & " * join([fmt(A[i, j]) for i in 1:ncol], " & ") * " \\\\")
    end
    println("\\hline")
    println("\\end{tabular}")
end

function xxz_run_direct_fidelity(p::Float64, Delta_tot=-1.0:0.1:1.0; N_tot=6:2:10, digits=6)
    ## Direct (dense) fidelity calculation, small N only; prints a LaTeX table
    fidelity_array = zeros(Float64, length(N_tot), length(Delta_tot))
    for (ii, N) in enumerate(N_tot), (jj, Delta) in enumerate(Delta_tot)
        # println("------ N=$N, Delta=$Delta -------")
        fidelity_array[ii, jj] = xxz_fidelity_exact(N, Float64(Delta); p1=p)
    end
    fidelity_to_latex(fidelity_array, collect(N_tot), collect(Delta_tot); digits=digits)
    save_array_json(fidelity_array, N_tot, Delta_tot, p; name="fidelity_p$(p)")
    return fidelity_array
end

function xxz_run_direct_renyi2(p::Float64, Delta_tot=-0.3:0.075:0.3; N_tot=6:2:10, digits=6,
                               max_bd=384, lpdo_max_err=1E-8, dmrg_max_bd=192,
                               dmrg_nsweeps=20, name="renyi2_p$(p)", pbc=true)
    ## Renyi-2 correlator scan (MPO contraction, not restricted to small N);
    ## prints a LaTeX table. Benchmarks C_X^II of 2503.14221 (their Fig. 3(a),
    ## where p_paper = p/2 and the operators sit at distance N/2 apart).
    ## The JSON file is rewritten after every point, so a long scan can be
    ## interrupted (or plotted) without losing what has already been computed.
    renyi2_array = fill(NaN, length(N_tot), length(Delta_tot))
    t_start = time()
    for (ii, N) in enumerate(N_tot), (jj, Delta) in enumerate(Delta_tot)
        println("------ N=$N, Delta=$Delta  (elapsed $(round(time()-t_start,digits=1)) s) -------")
        flush(stdout)
        renyi2_array[ii, jj] = xxz_renyi2_exact(N, Float64(Delta); p1=p, max_bd=max_bd,
                                                lpdo_max_err=lpdo_max_err,
                                                dmrg_max_bd=dmrg_max_bd,
                                                dmrg_nsweeps=dmrg_nsweeps, pbc=pbc)
        save_array_json(renyi2_array, N_tot, Delta_tot, p; name=name)
        flush(stdout)
    end
    fidelity_to_latex(renyi2_array, collect(N_tot), collect(Delta_tot); digits=digits)
    println("total wall time: $(round(time()-t_start,digits=1)) s")
    return renyi2_array
end

####################################
# julia xxz_init_data.jl 0.2 exact (for direct fidelity)
# julia xxz_init_data.jl 0.5 0.5 (for generating data files) 
# julia xxz_init_data.jl 0.2 renyi2 (for Renyi-2 correlator)


if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 2
        println("Usage: julia xxz_init_data.jl <p> <Delta>   # generate data files")
        println("       julia xxz_init_data.jl <p> exact     # direct (dense) fidelity, Delta scan")
        println("       julia xxz_init_data.jl <p> renyi2    # Renyi-2 correlator, Delta scan")
        exit(1)
    end
    p = parse(Float64, ARGS[1])

    if ARGS[2] == "exact"
        xxz_run_direct_fidelity(p)  # Delta_tot = -1.0:0.1:1.0, N_tot = 6:2:10
    elseif ARGS[2] == "renyi2"
        BLAS.set_num_threads(Sys.CPU_THREADS)
        xxz_run_direct_renyi2(p, -0.3:0.075:0.3; N_tot=[18, 24, 32, 48])
    else
        Delta = parse(Float64, ARGS[2])
        ptag = "p$(p)"         # 0.3 -> "p0.3", 1.0 -> "p1.0"
        Deltag = "del$(Delta)" # 0.3 -> "del0.3", 1.0 -> "del1.0"

        for N in 6:2:24
            println("------ N=$N -------")
            M1_save, M2_save = xxz_get_lpdo(N, Delta; p1=p, output=1);
            output_data(M1_save[1], ptag * Deltag * "/M1_a0_XXnoise_" * ptag * Deltag * "_N" * "$N")
            output_data(M2_save[1], ptag * Deltag * "/M2_a0_XXnoise_" * ptag * Deltag * "_N" * "$N")
        end
    end

end
