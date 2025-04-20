using LinearAlgebra,TensorOperations,LinearMaps,Arpack,ITensors

function apply_Ising_H(v::AbstractVector,H_local::Matrix)
    D = length(v)
    N = Int(round(log2(D)))
    Hv = zero(v)
    for i in 1:N
        v = reshape(v,(4,div(D,4)))
        Hv = reshape(Hv,(4,div(D,4)))
        Hv += H_local*v
        v = reshape(v,(2,div(D,2)))
        v = transpose(v)
        Hv = reshape(Hv,(2,div(D,2)))
        Hv = transpose(Hv)
    end
    Hv = reshape(Hv,D)
    return Hv
end

function get_Ising_eigenstates(N::Int,h::Float64 = 1.0)
    ## Get the first three eigenstates of critical Ising model with PBC on N spins 
    sigmaX = [[0.0 1.0];[1.0 0.0]]
    sigmaZ = [[1.0 0.0];[0.0 -1.0]]
    Id = [[1.0 0.0];[0.0 1.0]]
    @tensor H_ising[a,b,c,d] := -sigmaX[a,c] * sigmaX[b,d] - h * sigmaZ[a,c] * Id[b,d]
    H_ising = reshape(H_ising,(4,4));
    f = LinearMap{Float64}(x->apply_Ising_H(x,H_ising)-2*N*x, 2^N, issymmetric = true);
    Es, psis = Arpack.eigs(f,nev=3,which=:LM);
    Es = (2*N).+Es
    return Es,psis
end

function apply_channel!(rho0, Nc, nsite::Int)
    ## Apply channel Nc to the first nsite of rho0
    D = size(rho0,1)
    N = Int(round(log2(D)))
    for i in 1:nsite
        rho0 = reshape(rho0,(2,div(D,2),2,div(D,2)))
        @tensor rho1[k1,k2,b1,b2] := rho0[k0,k2,b0,b2] * Nc[k1,b1,k0,b0]
        rho0 = permutedims(rho1,[2,1,4,3])
    end
    rho0 = reshape(rho0, (2^(N-nsite),2^nsite,2^(N-nsite),2^nsite))
    rho0 = permutedims(rho0,[2,1,4,3])
    rho0 = reshape(rho0, (D,D))
    return rho0
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

function Ising_GS_DMRG(N,h=1.0,max_bd=100,nsweeps = 20)
    sites = siteinds("S=1/2",N)

    os = OpSum()
    for j in 1:N
        os += -4.0,"Sx",j,"Sx",j%N+1
    end
    for j in 1:N
        os += -2.0*h,"Sz",j
    end
    H = MPO(os,sites)

    
    maxdim = [20,32,64,max_bd] # gradually increase states kept
    cutoff = [1E-10] # desired truncation error
    noise = [1E-6,1E-7,1E-8,1E-8,1E-8,0.0]
    
    psi0 = randomMPS(sites,10)

    E0,psi = dmrg(H,psi0; nsweeps, maxdim, cutoff,noise)
    return psi
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
        A=Array(psi[i],MPSinds...)
        push!(As,A)
    end
    As[1]=reshape(As[1],(1,2,2))
    As[end]=reshape(As[end],(2,2,1));
    As = [convert(Array{Float64,3},A) for A in As];
    return As
end

function sqrt_mat(A)
    ## A = Hermitian
    D,U = eigen((A+A')/2)
    sqrtA = U * diagm(sqrt.(abs.(D))) * U'
    return sqrtA
end

function compute_fidelity(rho1,rho2)
    S = svdvals(sqrt_mat(rho1)*sqrt_mat(rho2))
    f = sum(S)
    return f
end