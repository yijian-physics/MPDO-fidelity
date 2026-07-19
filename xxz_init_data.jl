using LinearAlgebra,TensorOperations,LinearMaps,Arpack,ITensors, ITensorMPS

using GLM
using DataFrames
using LaTeXStrings
using HDF5
using Printf

include("evolMPDO.jl")
include("IsingED.jl")  # provides MPS_to_array


function XXZ_GS_DMRG(N, Delta=1.0, pbc=true;max_bd=200,nsweeps = 40)
    sites = siteinds("S=1/2",N)

    os = OpSum()
    for j in 1:N-1
        os += -1.0,"Sx",j,"Sx",j+1
        os += -1.0,"Sy",j,"Sy",j+1
        os += -1.0*Delta,"Sz",j,"Sz",j+1
    end
    if (pbc)
        os += -1.0,"Sx",N,"Sx",1
        os += -1.0,"Sy",N,"Sy",1
        os += -1.0*Delta,"Sz",N,"Sz",1
    end

    H = MPO(os,sites)

    maxdim = [20,32,64,max_bd] # gradually increase states kept
    cutoff = [1E-10] # desired truncation error
    noise = [1E-6,1E-7,1E-8,1E-8,1E-8,0.0]

    psi0 = randomMPS(sites,10)

    E0,psi0 = dmrg(H,psi0; nsweeps, maxdim, cutoff,noise,outputlevel=0)

    return psi0, E0
end


function xxz_get_lpdo(N::Int, Delta::Float64=1.0; p1=1.0, divide=2, output=1)
    # output is purification tensor (half of LPDO)

    psiMPS, E0= XXZ_GS_DMRG(N,Delta,true,nsweeps=60, max_bd=300);
    A = myMPS(MPS_to_array(psiMPS));

    Sx = [0 1; 1 0]
    i, j = 1, Int(N/divide)+1  # fix the ratio to be 1/divide

    M1 = add_noise_double(A, p1)
    M2 = add_CP(add_CP(M1, Sx, i), Sx, j)

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
    return fidelity_array
end

####################################
# julia xxz_init_data.jl 0.5 exact (for direct fidelity)
# julia xxz_init_data.jl 0.5 0.5 (for generating data files) 


if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 2
        println("Usage: julia xxz_init_data.jl <p> <Delta>   # generate data files")
        println("       julia xxz_init_data.jl <p> exact     # direct (dense) fidelity, Delta scan")
        exit(1)
    end
    p = parse(Float64, ARGS[1])

    if ARGS[2] == "exact"
        xxz_run_direct_fidelity(p)  # Delta_tot = -1.0:0.1:1.0, N_tot = 6:2:10
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
