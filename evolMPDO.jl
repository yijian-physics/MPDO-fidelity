using LinearAlgebra, TensorOperations, Statistics

struct myMPS{T<:Number}  # T is a subtype of Number
    TensorList::Array{Array{T,3},1} #List of myMPS tensors that represent the purification 
    #Tensor indices - left bond, system spin, right bond
end

Base.length(M::myMPS) = length(M.TensorList)    #Extend the existing Base.length function to work on new type myMPS
phys_dim(M::myMPS) = size(M.TensorList[1],2)
testnan(M::myMPS) = sum([sum(isnan.(ten)) for ten in M.TensorList])
testnorm(M::myMPS) = findmin([norm(ten) for ten in M.TensorList])[1]
max_bond_dim(M::myMPS) = findmax([size(ten,1) for ten in M.TensorList])[1]
Base.copy(M::myMPS) = myMPS(copy(M.TensorList))
Base.complex(M::myMPS) = myMPS(complex.(M.TensorList))

struct myMPDO{T<:Number}
    TensorList::Array{Array{T,4},1} #List of myMPDO tensors that represent the purification 
    #Tensor indices - left bond, system spin, environment spin, right bond
end

Base.length(M::myMPDO) = length(M.TensorList)
phys_dim(M::myMPDO) = size(M.TensorList[1],2)
ancilla_dim(M::myMPDO) = size(M.TensorList[1],3)
testnan(M::myMPDO) = sum([sum(isnan.(ten)) for ten in M.TensorList])
testnorm(M::myMPDO) = findmin([norm(ten) for ten in M.TensorList])[1]
max_bond_dim(M::myMPDO) = findmax([size(ten,1) for ten in M.TensorList])[1]
Base.copy(M::myMPDO) = myMPDO(copy(M.TensorList))
Base.complex(M::myMPDO) = myMPDO(complex.(M.TensorList))

function product_state_init(T::Type, d::Int, N::Int) 
    ## Initialize a product state |000000>
    ## T - data type
    ## d - local dimension
    ## N - number of sites
    Ten_even = zeros(T,1,d,d,1)
    Ten_even[1,1,1,1] = 1.0
    for i in 1:N
        push!(myMPSTensors, Ten_even)
    end
    return myMPDO(myMPSTensors)
end

function mytruncate(S::Vector{<:Real}, max_bd::Int, max_err::Float64)
    ## Given an array S (descending), determine the truncation 
    ## based on which of max bond dimesion or max err is reached
    err = 0.0
    set_bd = max_bd
    for i in length(S):-1:1
        err = err + S[i]^2
        if(err>max_err)
            if(i<max_bd)
                set_bd = i
            end
            break
        end
    end
    return set_bd
end

function canonicalize_left_one_site(M::myMPDO, site::Int;truncation = false, max_bd = 1024, max_err = 1E-10)
    ## A1 := M[site], A2 := M[site+1]
    ## A1 = USV' => A1=U, A2 = SVt (update)
    ## Truncate S if truncation = true
    ## return S and the updated myMPS M
    A1 = M.TensorList[site]
    DL,d,dA,DR = size(A1)
    A1_mat = reshape(A1, (DL*d*dA,DR))
    U = nothing; S=nothing; V=nothing;
    try
        U,S,V = svd(A1_mat,alg=LinearAlgebra.DivideAndConquer())
    catch
        U,S,V = svd(A1_mat,alg=LinearAlgebra.QRIteration())
    end
    if(norm(S)<eps(Float64))
         throw("zero norm")
    end
    S = S./norm(S)
    if(truncation == true)
        set_bd = mytruncate(S,max_bd,max_err)
        trunc_err = norm(S[set_bd+1:end])^2
        if(trunc_err>1E-6)
            println("truncation error:",trunc_err)
        end
        S = S[1:set_bd]
        U = U[:,1:set_bd]
        V = V[:,1:set_bd]
    end
    M.TensorList[site] = reshape(U, (DL,d,dA,length(S)))
    if(site<length(M))
        SVt = diagm(0=>S)*V'
        M.TensorList[site+1] = ncon([SVt, M.TensorList[site+1]],[[-1,1],[1,-2,-3,-4]])
    end
    return S, M
end

function canonicalize_right_one_site(M::myMPDO, site::Int;truncation = false, max_bd = 1024, max_err = 1E-10)
    ## A1 := M[site], A2 :=M[site-1]
    ## A1 = USV' => A1=V', A2 = US
    ## Truncate S if truncation = true
    A1 = M.TensorList[site]
    DL,d,dA,DR = size(A1)
    A1_mat = reshape(A1, (DL,DR*d*dA))
    U = nothing; S=nothing; V=nothing;
    try
        U,S,V = svd(A1_mat,alg=LinearAlgebra.DivideAndConquer())
    catch
        U,S,V = svd(A1_mat,alg=LinearAlgebra.QRIteration())
    end
    if(norm(S)<eps(Float64))
         throw("zero norm")
    end
    S = S./norm(S)
    if(truncation == true)
        set_bd = mytruncate(S,max_bd,max_err)
        trunc_err = norm(S[set_bd+1:end])^2
        if(trunc_err>1E-6)
            println("truncation error:",trunc_err)
        end
        S = S[1:set_bd]
        U = U[:,1:set_bd]
        V = V[:,1:set_bd]
    end
    M.TensorList[site] = reshape(V', (length(S),d,dA,DR))
    if(site>1)
        US = U*diagm(0=>S)
        M.TensorList[site-1] = ncon([US, M.TensorList[site-1]],[[1,-4],[-1,-2,-3,1]])
    end
    return S, M
end

function canonicalize_left(M::myMPDO;truncation = false, max_bd = 1024, max_err = 1E-10)
    ## Return a left canonical form of the purification (normalized automatically)
    N = length(M)
    for i in 1:N
        ~, M = canonicalize_left_one_site(M, i, truncation=truncation,max_bd=max_bd,max_err=max_err)
    end
    return M
end

function canonicalize_right(M::myMPDO;truncation = false, max_bd = 1024, max_err = 1E-10)
    ## Return a right canonical form of the purification (normalized automatically)
    N = length(M)
    for i in N:-1:1
        ~, M = canonicalize_right_one_site(M, i, truncation=truncation,max_bd=max_bd,max_err=max_err)
    end
    return M
end

function unitary_evol_two_site_system(M::myMPDO, U::Matrix, site::Int, dir = "l"; truncation = false, max_bd = 1024, max_err=1E-10)
    ## Evolve the two system qubit by U 
    ## Assuming U is two-site
    ## further assuming the two sites are the center of canonical form if truncation = true (important!)
    ## site is the first site # of the unitary (1 to N-1)
    ## the [site] of the two sites is put into left canonical form (if dir = "l")
    ## the [site+1] of the two sites is put into right canonical form (if dir = "r")
    ## return S and M
    myMPSTensors = M.TensorList
    @assert site<length(M)
    A1 = myMPSTensors[site]
    A2 = myMPSTensors[site+1]
    d1 = size(A1,2)
    d2 = size(A2,2)
    d1A = size(A1,3)
    d2A = size(A2,3)
    DL = size(A1,1)
    DR = size(A2,4)
    U = reshape(U,(d1,d2,d1,d2)) # site 1 ket, site 2 ket, site 1 bra, site 2 bra
    A_evol = ncon([A1,A2,U],[[-1,1,-3,2],[2,3,-6,-4],[-2,-3,1,3]])
    A_evol_mat = reshape(A_evol,(DL*d1*d1A,DR*d2*d2A))
    U2 = nothing; S2=nothing; V2=nothing;
    try
        U2,S2,V2 = svd(A_evol_mat,alg=LinearAlgebra.DivideAndConquer())
    catch
        U2,S2,V2 = svd(A_evol_mat,alg=LinearAlgebra.QRIteration())
    end
    if(truncation == true)
        set_bd = mytruncate(S2, max_bd, max_err)
        trunc_err = norm(S2[set_bd+1:end])^2
        if(trunc_err>1E-6)
            println("truncation error:",trunc_err)
        end
        S2 = S2[1:set_bd]
        U2 = U2[:,1:set_bd]
        V2 = V2[:,1:set_bd]
    end
    if(dir == "l")
        AL = reshape(U2, (DL,d1,d1A,length(S2)))
        AR = reshape(diagm(0=>S2)*V2',(length(S2),d2,d2A,DR))
    else
        AL = reshape(U2*diagm(0=>S2), (DL,d1,d1A,length(S2)))
        AR = reshape(V2',(length(S2),d2,d2A,DR))
    end
    M.TensorList[site] = AL
    M.TensorList[site+1] = AR    
    return S2, M
end

function unitary_evol_two_site_ancilla(M::myMPDO, U::Matrix, site::Int, dir = "l"; truncation = false, max_bd = 1024, max_err=1E-10)
    ## Evolve the two ancilla **qudit** by U 
    ## Assuming U is two-site
    ## further assuming the two sites are the center of canonical form if truncation = true (important!)
    ## site is the first site # of the unitary (1 to N-1)
    ## the [site] of the two sites is put into left canonical form (if dir = "l")
    ## the [site+1] of the two sites is put into right canonical form (if dir = "r")
    ## return S and M
    myMPSTensors = M.TensorList
    @assert site<length(M)
    A1 = myMPSTensors[site]
    A2 = myMPSTensors[site+1]
    d1 = size(A1,2)
    d2 = size(A2,2)
    d1A = size(A1,3)
    d2A = size(A2,3)
    DL = size(A1,1)
    DR = size(A2,4)
    U = reshape(U,(d1A,d2A,d1A,d2A)) # site 1 ket, site 2 ket, site 1 bra, site 2 bra
    A_evol = ncon([A1,A2,U],[[-1,-2,1,2],[2,-4,3,-6],[-3,-5,1,3]])
    A_evol_mat = reshape(A_evol,(DL*d1*d1A,DR*d2*d2A))
    U2 = nothing; S2=nothing; V2=nothing;
    try
        U2,S2,V2 = svd(A_evol_mat,alg=LinearAlgebra.DivideAndConquer())
    catch
        U2,S2,V2 = svd(A_evol_mat,alg=LinearAlgebra.QRIteration())
    end
    if(truncation == true)
        set_bd = mytruncate(S2, max_bd, max_err)
        trunc_err = norm(S2[set_bd+1:end])^2
        if(trunc_err>1E-6)
            println("truncation error:",trunc_err)
        end
        S2 = S2[1:set_bd]
        U2 = U2[:,1:set_bd]
        V2 = V2[:,1:set_bd]
    end
    if(dir == "l")
        AL = reshape(U2, (DL,d1,d1A,length(S2)))
        AR = reshape(diagm(0=>S2)*V2',(length(S2),d2,d2A,DR))
    else
        AL = reshape(U2*diagm(0=>S2), (DL,d1,d1A,length(S2)))
        AR = reshape(V2',(length(S2),d2,d2A,DR))
    end
    M.TensorList[site] = AL
    M.TensorList[site+1] = AR    
    return S2, M
end

apply_TM_l(A::Array{<:Number,4},B::Array{<:Number,4},l::Array{<:Number,2})=ncon([A,conj.(B),l],[[4,2,3,-2],[1,2,3,-1],[1,4]])
apply_TM_r(A::Array{<:Number,4},B::Array{<:Number,4},r::Array{<:Number,2})=ncon([A,conj.(B),r],[[-1,2,3,1],[-2,2,3,4],[1,4]])

function right_environments(M1::myMPDO,M2::myMPDO)
    ## starting from the right, compute the overlap of <M2|M1> by applying transfer matrices
    N = length(M1)
    r = diagm(ones(1))
    rs = Matrix[]
    push!(rs,r)
    for i in N:-1:1
        A = M1.TensorList[i]
        B = M2.TensorList[i]
        r = apply_TM_r(A,B,r)
        push!(rs,r)
    end
    return rs
end
    
function left_environments(M1::myMPDO,M2::myMPDO)
    ## starting from the left, compute the overlap of <M2|M1> by applying transfer matrices
    N = length(M1)
    l = diagm(ones(1))
    ls = Matrix[]
    push!(ls,l)
    for i in 1:N
        A = M1.TensorList[i]
        B = M2.TensorList[i]
        l = apply_TM_l(A,B,l)
        push!(ls,l)
    end
    return ls
end

function optimize_overlap_onelayer(M1::myMPDO,M2::myMPDO,dir = "l";truncation = true, max_bd = 1024, max_err=1E-10)
    ## Act a sequential circuit on M1 and maximize |<M2|M1>|
    ## dir="l" - add unitary from left to right - this assumes that M1 is right-canonical ** important **
    ## dir="r" - add unitary from right to left - this assumes that M1 is left-canonical **important**
    ## note - unitaries are acted on M1 (ancilla leg) - and we will return the modified M1 in left/right canonical form
    M1cp = copy(M1)
    ov_opts = Float64[] #optimized fidelity after applying each unitary
    rs = right_environments(M1,M2)
    push!(ov_opts,abs(tr(rs[end])))
    N = length(M1)
    l_env = diagm(ones(1))
    for i in 1:N-1
        r_env = rs[N-i] ## right enviroment
        
        A1 = M1cp.TensorList[i]
        A2 = M1cp.TensorList[i+1]
        
        B1 = M2.TensorList[i]
        B2 = M2.TensorList[i+1]
        
        @tensor U_env[a1,a2,a1p,a2p] := l_env[b1,t1] * A1[t1,s,a1,t2] * conj(B1)[b1,s,a1p,b2] * A2[t2,ss,a2,t3] * r_env[t3,b3] *conj(B2)[b2,ss,a2p,b3]
        d1,d2,d1p,d2p = size(U_env)
        U_env = reshape(U_env,(d1*d2,d1p*d2p))
        U = nothing; S=nothing; V=nothing;
        try
            U,S,V = svd(U_env,alg=LinearAlgebra.DivideAndConquer())
        catch
            U,S,V = svd(U_env,alg=LinearAlgebra.QRIteration())
        end
        U_opt = V*U'
        push!(ov_opts,sum(S)) ## This is the optimized overlap
        ~, M1cp = unitary_evol_two_site_ancilla(M1cp, U_opt, i, dir, truncation = truncation, max_bd = max_bd, max_err=max_err)
        l_env = apply_TM_l(M1cp.TensorList[i],M2.TensorList[i],l_env)
    end
    return M1cp, ov_opts
end

function compute_overlap(M1::myMPDO,M2::myMPDO;dir="l")
    ## contract <M2|M1> from the left/right 
    if(dir=="l")
        ls = left_environments(M1,M2)
        ov = abs(tr(ls[end]))
    else
        rs = right_environments(M1,M2)
        ov = abs(tr(rs[end]))
    end
    return ov
end

function optimize_overlap(M1::myMPDO,M2::myMPDO,iters = 10;truncation = true, max_bd = 1024, max_err=1E-10)
    M1cp = copy(M1)
    M2cp = copy(M2)
    ov = nothing
    for j in 1:iters
        println("----iteration $(j) -----")
        M1cp = canonicalize_right(M1cp)
        M2cp = canonicalize_right(M2cp);
        M1cp, ovs = optimize_overlap_onelayer(M1cp,M2cp);
        M2cp, ovs = optimize_overlap_onelayer(M2cp,M1cp);
        ov = ovs[end]
        chi1 = max_bond_dim(M1cp)
        chi2 = max_bond_dim(M2cp)
        println("Bond dimensions: $chi1,$chi2")
        println("Overlap: $ov")
    end
    return M1cp,M2cp,ov
end

function MPS_to_MPDO(M::myMPS{T},d::Int = phys_dim(M)) where T
    ## for a MPS, construct a purification MPDO = |M>|0>
    Ts = Array{T,4}[]
    for i in 1:length(M)
        A = M.TensorList[i]
        DL,dS,DR = size(A)
        A_ext = zeros(T,DL,dS,d,DR)
        A_ext[:,:,1,:] = A
        push!(Ts, A_ext)
    end
    return myMPDO(Ts)
end

function add_noise_MPS(M::myMPS{T}, Ws::Vector{Array{T,3}}) where T
    ## W: System -> System * ancilla, act W to the MPS to create MPDO
    ## W is the dilation of a noisy channel
    Ts = Array{T,4}[]
    for i in 1:length(M)
        A = M.TensorList[i]
        W = Ws[i]
        @tensor A_ext[l,s,a,r] := A[l,s0,r] * W[s,a,s0]
        push!(Ts, A_ext)
    end
    return myMPDO(Ts)
end

function purified_dephasing_channel(p::Float64, dir::Vector)
    ## isometry: |alpha> -> sqrt(1-p/2) |alpha>|0> + sqrt(p/2)(sigma_dir|alpha>)|1>
    ## dir: (X,Y,Z) of dephasing
    sigmaX = [[0.0 1.0];[1.0 0.0]]
    sigmaZ = [[1.0 0.0];[0.0 -1.0]]
    Id = [[1.0 0.0];[0.0 1.0]]
    sigmaY = 1im.*sigmaZ*sigmaX
    dir = dir./norm(dir)
    if(abs(dir[2])<1E-7)
        sigma = dir[1].*sigmaX+dir[3].*sigmaZ
        W = zeros(2,2,2) # outputQ, outputE, inputQ
        W[:,1,:] = sqrt(1-p/2)*Id 
        W[:,2,:] = sqrt(p/2)*sigma
    else
        sigma = dir[1].*sigmaX+dir[2].*sigmaY+dir[3].*sigmaZ
        W = zeros(Complex{Float64},2,2,2) # outputQ, outputE, inputQ
        W[:,1,:] = sqrt(1-p/2)*Id 
        W[:,2,:] = sqrt(p/2)*sigma   
    end
    return W
end

function Haar_random_unitary(T,d::Int)
    ## return a d*d Haar random unitary matrix (complex or real)
    H = randn(T,d,d)
    Q,R = qr(H)
    fac = diagm(0=>[R[i,i]/abs(R[i,i]) for i in 1:d])
    Q = Q*fac
    return Q
end

function random_unitary_layer_ancilla_onsite!(M0::myMPDO{T}) where T
    ## apply onsite random unitary to the ancilla leg
    for i in 1:length(M0)
        A = M0.TensorList[i];
        U = Haar_random_unitary(T,size(A,3))
        @tensor Anew[l,s,a,r] := A[l,s,a0,r] * U[a,a0]
        M0.TensorList[i] = Anew
    end
    return M0
end

function MPS_to_dense(M::myMPS{T}) where T
    ## Only run this for small systems!!
    d = phys_dim(M)
    L = length(M)
    tmp = M.TensorList[1]
    for i in 2:L
        A = M.TensorList[i]
        dR = size(A,3)
        @tensor tmp2[l,s1,s2,r] := tmp[l,s1,r1]*A[r1,s2,r]
        tmp = reshape(tmp2,(1,d^i,dR))
    end
    psi = reshape(tmp,d^L)
    return psi
end

function MPDO_to_MPS(M::myMPDO{T}) where T
    ## MPS in doubled space
    Ts = Array{T,3}[]
    for i in 1:length(M)
        DL,dS,dA,DR = size(M.TensorList[i])
        push!(Ts,reshape(M.TensorList[i],(DL,dS*dA,DR)))
    end
    return myMPS(Ts)
end

function MPDO_to_dense(M::myMPDO{T}) where T
    dS = phys_dim(M)
    dA = ancilla_dim(M)
    L = length(M)
    tmp = M.TensorList[1]
    for i in 2:L
        A = M.TensorList[i]
        dR = size(A,4)
        @tensor tmp2[l,s1,s2,a1,a2,r] := tmp[l,s1,a1,r1]*A[r1,s2,a2,r]
        tmp = reshape(tmp2,(1,dS^i,dA^i,dR))
    end
    rho = reshape(tmp,(dS^L,dA^L))
    return rho
end

function optimize_overlap_onefloor(M1::myMPDO,M2::myMPDO,Us::Vector{<:Matrix};truncation = true, max_bd = 1024, max_err=1E-10)
    ## Act a sequential circuit on M1 and maximize |<M2|M1>|
    ## Us = initial guess of the unitary network. 
    ## Us order: U_{12}, U_{23}, ... U{n-1,n} U_{n-2,n-1} ... U_{12} 
    ## one floor: length(Us) = 2N-3 (k-floor length(Us) = k*(2N-4) + 1) 
    ## Assumes that M1 and M2 are right-canonical ** important **
    ## note - unitaries are acted on M2 (ancilla leg) - and we will return the modified M2 in right canonical form
    N = length(M1)
    @assert length(Us) == 2*N-3  ## This is one-floor constraint
    M1 = canonicalize_right(M1)
    M2 = canonicalize_right(M2);
    
    ov_opts = Float64[] #optimized fidelity after applying each optimization
    
    ## 1. precontract the tensor network from M1 and save all intermidates
    M1_interms = myMPDO[]
    push!(M1_interms, M1);
    M1cp = copy(M1)
    for i in 1:N-1
        U = Us[i]
        site = i;
        ~, M1_out = unitary_evol_two_site_ancilla(M1cp, U, site, "l"; truncation = truncation, max_bd = max_bd, max_err = max_err)
        push!(M1_interms, M1_out)
        M1cp = copy(M1_out)
    end
    for i in N:2*N-3
        site = 2*N-i-2
        U = Us[i]
        ~, M1_out = unitary_evol_two_site_ancilla(M1cp, U, site, "r"; truncation = truncation, max_bd = max_bd, max_err = max_err)
        push!(M1_interms, M1_out)
        M1cp = copy(M1_out) 
    end
    
    ## 2. Left to right sweep - from U12 to U_{n-1,n} (reverse order of Us)
    M1cp = M1_interms[end-1]
    M2_interms = myMPDO[]
    M2cp = copy(M2)
    push!(M2_interms, M2cp)
    l_env = diagm(ones(1))
    r_envs = right_environments(M1cp,M2cp) ## Right canonical form assumed for M1  ## This will NOT change during the sweep 
    push!(ov_opts,abs(tr(r_envs[end]))) ## Initial Us overlap
    for i in 1:N-1
        r_env = r_envs[N-i] ## right enviroment
        
        A1 = M1cp.TensorList[i]
        A2 = M1cp.TensorList[i+1]
        
        B1 = M2cp.TensorList[i]
        B2 = M2cp.TensorList[i+1]
        
        @tensor U_env[a1,a2,a1p,a2p] := l_env[b1,t1] * A1[t1,s,a1,t2] * conj(B1)[b1,s,a1p,b2] * A2[t2,ss,a2,t3] * r_env[t3,b3] *conj(B2)[b2,ss,a2p,b3]
        d1,d2,d1p,d2p = size(U_env)
        U_env = reshape(U_env,(d1*d2,d1p*d2p))
        U = nothing; S=nothing; V=nothing;
        try
            U,S,V = svd(U_env,alg=LinearAlgebra.DivideAndConquer())
        catch
            U,S,V = svd(U_env,alg=LinearAlgebra.QRIteration())
        end
        U_opt = V*U'
        ov = sum(S)
        push!(ov_opts,ov) ## This is the optimized overlap
        
        ~, M2cp = unitary_evol_two_site_ancilla(M2cp, Matrix(U_opt'), i, "l"; truncation = truncation, max_bd = max_bd, max_err = max_err)
        #ov = abs(compute_overlap(M1_interms[end-i],M2cp))
        #@show ov
        push!(M2_interms, M2cp)
        if(i<N-1)
            M1cp = M1_interms[end-i-1]
            l_env = apply_TM_l(M1cp.TensorList[i],M2cp.TensorList[i],l_env)
        end
    end
    
    #println("Switch!!")
    
    ## 3. Right to left sweep from U_{n-2,n-1} to U_{1,2} (reverse order of Us)
    ~,M2cp = canonicalize_right_one_site(M2cp, N)
    M1cp = M1_interms[N-2]
    l_envs = left_environments(M1cp,M2cp)
    r_env = diagm(ones(1))
    r_env = apply_TM_r(M1cp.TensorList[N],M2cp.TensorList[N],r_env)
    for i in N-2:-1:1
        l_env = l_envs[i]
        A1 = M1cp.TensorList[i]
        A2 = M1cp.TensorList[i+1]
        
        B1 = M2cp.TensorList[i]
        B2 = M2cp.TensorList[i+1]  
        @tensor U_env[a1,a2,a1p,a2p] := l_env[b1,t1] * A1[t1,s,a1,t2] * conj(B1)[b1,s,a1p,b2] * A2[t2,ss,a2,t3] * r_env[t3,b3] *conj(B2)[b2,ss,a2p,b3]
        d1,d2,d1p,d2p = size(U_env)
        U_env = reshape(U_env,(d1*d2,d1p*d2p))
        U = nothing; S=nothing; V=nothing;
        try
            U,S,V = svd(U_env,alg=LinearAlgebra.DivideAndConquer())
        catch
            U,S,V = svd(U_env,alg=LinearAlgebra.QRIteration())
        end
        U_opt = V*U'
        ov = sum(S)
        push!(ov_opts,ov) ## This is the optimized overlap
        
        ~, M2cp = unitary_evol_two_site_ancilla(M2cp, Matrix(U_opt'), i, "r"; truncation = truncation, max_bd = max_bd, max_err = max_err)
        
        #ov = abs(compute_overlap(M1_interms[i],M2cp))
       # @show ov
        push!(M2_interms, M2cp)
        if(i>1)
            M1cp = M1_interms[i-1]
            r_env = apply_TM_r(M1cp.TensorList[i+1],M2cp.TensorList[i+1],r_env)
        end
    end
    return M2cp, ov_opts
end