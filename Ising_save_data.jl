using LinearAlgebra,TensorOperations,LinearMaps,Arpack,ITensors, ITensorMPS

using GLM
using DataFrames
using LaTeXStrings
using HDF5

include("evolMPDO.jl")

# Print a diagnostic only when its absolute value exceeds `cutoff`.
function report_if_large(label, val; cutoff=1e-8)
    if abs(val) > cutoff
        println(label, " = ", val)
    end
end

function check_energies(N)      # Number of sites
    J = 0.5     # Ising coupling
    h = 0.5   # Transverse field
    E_ground_even, E_excited_even, ε_even = tfim_majorana_energies(N, J, h, "even")
    E_ground_odd, E_excited_odd, ε_odd = tfim_majorana_energies(N, J, h, "odd")

    E0exact = E_ground_even
    E1exact = E_ground_odd
    E2exact = E_excited_even;

    return E0exact, E1exact, E2exact
end

function tfim_majorana_energies(N::Int, J::Float64, h::Float64, parity::String)
    # Construct the 2N × 2N antisymmetric matrix A
    # Majorana Hamiltonian: H = (i/2) cᵀ A c

    A = zeros(Float64, 2*N, 2*N)

    # Fill the matrix based on TFIM
    # After Jordan-Wigner + Majorana:
    # σᵢˣ → i c₂ᵢ₋₁ c₂ᵢ
    # σᵢᶻσᵢ₊₁ᶻ → -i c₂ᵢ c₂ᵢ₊₁

    for j in 1:N
        # Transverse field term: -h σⱼˣ → -h(i c₂ⱼ₋₁ c₂ⱼ)
        # Contributes: A[2j-1, 2j] = 2h
        A[2*j-1, 2*j] = 2*h
        A[2*j, 2*j-1] = -2*h

        # Ising term: -J σⱼᶻσⱼ₊₁ᶻ → J(i c₂ⱼ c₂ⱼ₊₁)
        # Contributes: A[2j, 2j+1] = -2J
        j_next = mod1(j+1, N)

        # Apply boundary condition based on parity
        if j == N
            # Last coupling term
            if parity == "even"
                # APBC: change sign
                A[2*j, 2*j_next-1] = 2*J
                A[2*j_next-1, 2*j] = -2*J
            else  # odd parity
                # PBC: normal sign
                A[2*j, 2*j_next-1] = -2*J
                A[2*j_next-1, 2*j] = 2*J
            end
        else
            A[2*j, 2*j_next-1] = -2*J
            A[2*j_next-1, 2*j] = 2*J
        end
    end

    # Schur decomposition for antisymmetric matrix
    # For real antisymmetric matrix, schur gives real eigenvalues in 2x2 blocks
    S = schur(A)

    # Extract single-particle energies from Schur form
    # For antisymmetric matrices, eigenvalues come in pairs ±iε
    # The Schur form has 2x2 blocks on diagonal with eigenvalues ±iω
    ε = Float64[]

    i = 1
    while i <= 2*N
        if i < 2*N && abs(S.T[i+1, i]) > 1e-10
            # 2x2 block found
            # Extract eigenvalue from the block
            a = S.T[i, i]
            b = S.T[i, i+1]
            c = S.T[i+1, i]
            d = S.T[i+1, i+1]

            # For antisymmetric 2x2 block: [[0, b], [-b, 0]]
            # Eigenvalue is ±ib, so ω = b
            ω = sqrt(abs(b * c))
            push!(ε, ω)
            i += 2
        else
            # 1x1 block (should be zero for antisymmetric matrix)
            i += 1
        end
    end

    ε = sort(ε)

    # Ground state: all negative energy modes filled (nₖ = 0 for all k)
    # E = Σₖ εₖ(2*0 - 1) = -Σₖ εₖ
    E_ground = -sum(ε)

    # First excited state: flip the two lowest energy modes
    # Sort by absolute value to find the two modes closest to zero
    ε_sorted_by_abs = sort(ε, by=abs)

    # Change filling: these two modes go from nₖ=0 to nₖ=1
    # ΔE = 2ε₁ + 2ε₂ where ε₁, ε₂ are the two smallest |ε|
    E_excited = E_ground + 2*ε_sorted_by_abs[1] + 2*ε_sorted_by_abs[2]

    return E_ground, E_excited, ε
end

function dephasing_Z(p::Float64)
    Tid = zeros(2,2,2,2)
    Tzz = zeros(2,2,2,2)
    sigmaZ = [[1.0 0.0];[0.0 -1.0]]
    for i in 1:2
        for j in 1:2
            Tid[i,j,i,j] = 1.0
        end
    end
    @tensor Tzz[k1,b1,k0,b0] = sigmaZ[k1,k0] * sigmaZ[b1,b0]
    T = (1-p/2)*Tid + (p/2)*Tzz
    return T
end

function dephasing_X(p::Float64)
    Tid = zeros(2,2,2,2)
    Tzz = zeros(2,2,2,2)
    sigmaX = [[0.0 1.0];[1.0 0.0]]
    for i in 1:2
        for j in 1:2
            Tid[i,j,i,j] = 1.0
        end
    end
    @tensor Tzz[k1,b1,k0,b0] = sigmaX[k1,k0] * sigmaX[b1,b0]
    T = (1-p/2)*Tid + (p/2)*Tzz
    return T
end

function Ising_GS_DMRG(N,h=1.0,pbc=true;max_bd=200,nsweeps = 40,weight_fac=50)
    sites = siteinds("S=1/2",N)
    weight = weight_fac*h

    os = OpSum()
    for j in 1:N-1
        os += -4.0,"Sx",j,"Sx",j+1
    end
    if (pbc)
        os += -4.0,"Sx",N,"Sx",1
    end
    for j in 1:N
        os += -2.0*h,"Sz",j
    end
    H = MPO(os,sites)


    maxdim = [20,32,64,max_bd] # gradually increase states kept
    cutoff = [1E-10] # desired truncation error
    noise = [1E-6,1E-7,1E-8,1E-8,1E-8,0.0]

    psi0 = randomMPS(sites,10)

    E0,psi0 = dmrg(H,psi0; nsweeps, maxdim, cutoff,noise,outputlevel=0)
    E1, psi1 = dmrg(H, [psi0], randomMPS(sites;linkdims=2); nsweeps, maxdim, cutoff,noise,weight,outputlevel=0)
    E2, psi2 = dmrg(H, [psi0, psi1], randomMPS(sites;linkdims=2); nsweeps, maxdim, cutoff,noise,weight,outputlevel=0)

    report_if_large("inner(psi2,psi0)", inner(psi2,psi0))
    report_if_large("inner(psi2,psi1)", inner(psi2,psi1))


    # M_GS = myMPS(MPS_to_array(psi0));
    # M_exc1 = myMPS(MPS_to_array(psi1));
    # M_exc2 = myMPS(MPS_to_array(psi2));

    # println(abs(only(left_environments(M_GS, M_exc2)[end])))

    return psi0, psi1, psi2, E0, E1, E2
end

function M1_M2(N::Int, divide::Int;p1=1.0, output=0, noise='Z')
    # output is purification tensor (half of LPDO)

    h = 1.0;
    psiMPS, psiMPS_exc1, psiMPS_exc2, E0, E1, E2 = Ising_GS_DMRG(N,h,true,nsweeps=60, max_bd=300, weight_fac=50);

    ## Check energies!!!
    E0exact, E1exact,E2exact = check_energies(N)
    report_if_large("abs(E0-E0exact)", abs(E0-E0exact))
    report_if_large("abs(E1-E1exact)", abs(E1-E1exact))
    report_if_large("abs(E2-E2exact)", abs(E2-E2exact))

    A_GS = myMPS(MPS_to_array(psiMPS));
    A_exc1 = myMPS(MPS_to_array(psiMPS_exc1));
    A_exc2 = myMPS(MPS_to_array(psiMPS_exc2));

    A_tot = [A_GS, A_exc1, A_exc2]


    Sx = [0 1; 1 0]
    Sz = [1 0; 0 -1]

    i = 1
    j = Int(N/divide)+1  # fix the ratio to be 1/divide

    if output == 1
        M1_tot = Vector{Vector{Array{eltype(A_GS.TensorList[1])}}}()
        M2_tot = Vector{Vector{Array{eltype(A_GS.TensorList[1])}}}()
    elseif output == 0
        M1_tot = []
        M2_tot = []
    end

    for ii in 1:3
        A = A_tot[ii]
        if noise == 'Z'
            Ws1 = Array{Float64,3}[purified_dephasing_channel(p1,[0,0,1]) for _ in 1:N]
        elseif noise == 'X'
            Ws1 = Array{Float64,3}[purified_dephasing_channel(p1,[1,0,0]) for _ in 1:N]
        end
        M1 = add_noise_MPS(A, Ws1)
        M2 = add_CP(add_CP(M1, Sx, i), Sx, j)

        if output == 1
            # for output data to python
            M1 = myMPDO_to_array(M1)
            M2 = myMPDO_to_array(M2)
        end

        push!(M1_tot, M1)
        push!(M2_tot, M2)
    end

    return M1_tot, M2_tot
end

function output_data(data, data_name; string="arr")

    N = length(data)

    h5open("gapless_data/" * data_name * ".h5", "w") do f
        for i in 1:N
            f[string * "_$i"] = data[i]
        end
    end
end

function MPS_to_array(psi::MPS)
    N=length(psi)
    As=[];
    for i=1:N
        if(i<N)
            rightind=intersect(inds(psi[i]),inds(psi[i+1]))
        else
            rightind=[]
        end
        if(i>1)
            leftind=intersect(inds(psi[i]),inds(psi[i-1]))
        else
            leftind=[]
        end
        physind=setdiff(inds(psi[i]),union(rightind,leftind))
        MPSinds=vcat(leftind,physind,rightind)
        A=Array(psi[i],MPSinds...)                   # Array(T::ITensor, inds::Index...), returns a plain Julia Array with raw numerical values
        push!(As,A)
    end
    As[1]=reshape(As[1],(1,2,2))
    As[end]=reshape(As[end],(2,2,1));
    As = [convert(Array{eltype(A),3},A) for A in As];
    return As
end

# -------------------- main --------------------
# Usage:  julia Ising_save_data.jl <p>
# e.g.    julia Ising_save_data.jl 0.3
p = parse(Float64, ARGS[1])
ptag = "p" * lpad(round(Int, p*10), 2, '0')   # 0.3 -> "p03", 0.5 -> "p05"

### CFT code data (Z, X)

for N in 6:2:32
    println("------ N=$N -------")
    M1_save, M2_save = M1_M2(N, 2, p1=p, output=1,noise='Z');
    output_data(M1_save[1], ptag * "/M1_a0_Znoise_" * ptag * "_N" * "$N")
    output_data(M1_save[3], ptag * "/M1_a2_Znoise_" * ptag * "_N" * "$N")
    output_data(M2_save[3], ptag * "/M2_a2_Znoise_" * ptag * "_N" * "$N")
end

for N in 6:2:32
    println("------ N=$N -------")
    M1_save, M2_save = M1_M2(N, 2, p1=p, output=1,noise='X');
    output_data(M1_save[1], ptag * "/M1_a0_Xnoise_" * ptag * "_N" * "$N")
    output_data(M1_save[3], ptag * "/M1_a2_Xnoise_" * ptag * "_N" * "$N")
end

### SWSSB data (p=1.0, Z)

# for N in 6:2:32
#     println("------ N=$N -------")
#     M1_save, M2_save = M1_M2(N, 2, p1=1.0, output=1,noise='Z');
#     output_data(M1_save[3], "M1_a2_N" * "$N")
#     output_data(M2_save[3], "M2_a2_N" * "$N")
# end
