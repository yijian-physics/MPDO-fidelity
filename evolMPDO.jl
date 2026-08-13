using MKL
using LinearAlgebra, TensorOperations, Statistics

struct myMPS{T<:Number}
    TensorList::Array{Array{T,3},1} #List of myMPS tensors that represent the purification 
    #Tensor indices - left bond, system spin, right bond
end

Base.length(M::myMPS) = length(M.TensorList)
phys_dim(M::myMPS) = size(M.TensorList[1],2)
testnan(M::myMPS) = sum([sum(isnan.(ten)) for ten in M.TensorList])
testnorm(M::myMPS) = findmin([norm(ten) for ten in M.TensorList])[1]
max_bond_dim(M::myMPS) = findmax([size(ten,1) for ten in M.TensorList])[1]
Base.copy(M::myMPS) = myMPS([deepcopy(tensor) for tensor in M.TensorList])
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
Base.copy(M::myMPDO) = myMPDO([deepcopy(tensor) for tensor in M.TensorList])
Base.complex(M::myMPDO) = myMPDO(complex.(M.TensorList))
Base.conj(M::myMPDO) = myMPDO(conj.(M.TensorList))

struct myMPO{T<:Number}
    TensorList::Array{Array{T,4},1} #List of myMPO tensors that represent the MPO
    #Tensor indices - left bond, system spin (ket), system spin (bra), right bond
end

Base.length(M::myMPO) = length(M.TensorList)
phys_dim(M::myMPO) = size(M.TensorList[1],2)
testnan(M::myMPO) = sum([sum(isnan.(ten)) for ten in M.TensorList])
testnorm(M::myMPO) = findmin([norm(ten) for ten in M.TensorList])[1]
max_bond_dim(M::myMPO) = findmax([size(ten,1) for ten in M.TensorList])[1]
Base.copy(M::myMPO) = myMPO([deepcopy(tensor) for tensor in M.TensorList])
Base.complex(M::myMPO) = myMPO(complex.(M.TensorList))

######### Basic functions #########

function myMPDO_to_array(M::myMPDO)

    # bring MPDO to array. 
    # note the indices are: left bond, system spin, environment spin, right bond

    L = length(M)
    Ms = Vector{Array{eltype(M.TensorList[1])}}()
    
    for i in 1:L
        push!(Ms, M.TensorList[i])
    end

    return Ms

end

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

######### Canonicalization #########

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

######### MPO canonicalization #########

function canonicalize_left_one_site(M::myMPO, site::Int;truncation = false, max_bd = 1024, max_err = 1E-10)
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

function canonicalize_right_one_site(M::myMPO, site::Int;truncation = false, max_bd = 1024, max_err = 1E-10)
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

function canonicalize_left(M::myMPO;truncation = false, max_bd = 1024, max_err = 1E-10)
    ## Return a left canonical form of the purification (normalized automatically)
    N = length(M)
    for i in 1:N
        ~, M = canonicalize_left_one_site(M, i, truncation=truncation,max_bd=max_bd,max_err=max_err)
    end
    return M
end

function canonicalize_right(M::myMPO;truncation = false, max_bd = 1024, max_err = 1E-10)
    ## Return a right canonical form of the purification (normalized automatically)
    N = length(M)
    for i in N:-1:1
        ~, M = canonicalize_right_one_site(M, i, truncation=truncation,max_bd=max_bd,max_err=max_err)
    end
    return M
end

######### Unitary evolution #########

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
        S2 = S2./norm(S2)  # 2025/5/25
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
        S2 = S2./norm(S2)  # 2025/5/25
    end
    if(dir == "l")
        AL = reshape(U2, (DL,d1,d1A,length(S2)))
        AR = reshape(diagm(0=>S2)*V2',(length(S2),d2,d2A,DR))
    else
        AL = reshape(U2*diagm(0=>S2), (DL,d1,d1A,length(S2)))
        AR = reshape(V2',(length(S2),d2,d2A,DR))
    end

    ## BUG?? S2 is not normalized after truncation??

    M.TensorList[site] = AL
    M.TensorList[site+1] = AR    
    return S2, M
end

######### Transfer matrix and overlap #########

apply_TM_l(A::Array{<:Number,4},B::Array{<:Number,4},l::Array{<:Number,2})=ncon([A,conj.(B),l],[[4,2,3,-2],[1,2,3,-1],[1,4]])
apply_TM_r(A::Array{<:Number,4},B::Array{<:Number,4},r::Array{<:Number,2})=ncon([A,conj.(B),r],[[-1,2,3,1],[-2,2,3,4],[1,4]])

apply_TM_l(A::Array{<:Number,3},B::Array{<:Number,3},l::Array{<:Number,2})=ncon([A,conj.(B),l],[[3,2,-2],[1,2,-1],[1,3]])
apply_TM_r(A::Array{<:Number,3},B::Array{<:Number,3},r::Array{<:Number,2})=ncon([A,conj.(B),l],[[-1,2,1],[-2,2,3],[1,3]])

function right_environments(M1::myMPDO,M2::myMPDO)
    ## starting from the right, compute the overlap of <M2|M1> by applying transfer matrices
    # Output length is N+1
    #  -2 ---M2^*--
    #        |    |
    #  -1 ---M1----
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
    #     ---M2^*-- -1
    #    |   |    
    #     ---M1---- -2
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


function left_environments(A1::myMPS,A2::myMPS)
    ## starting from the left, compute the overlap of <A2|A1> by applying transfer matrices
    #     ---A2^*-- -1
    #    |   |    
    #     ---A1---- -2
    N = length(A1)
    l = diagm(ones(1))
    ls = Matrix[]
    push!(ls,l)
    for i in 1:N
        A = A1.TensorList[i]
        B = A2.TensorList[i]
        l = apply_TM_l(A,B,l)
        push!(ls,l)
    end
    return ls
end


function read_component(A::myMPS, bs::Array)

    # Take component of MPS A. 
    
    res = diagm(ones(1))
    N = length(A)

    for i in 1:N
        tmp = (A.TensorList[i])[:,bs[i],:]
        res = res * tmp
    end

    return res

end


function random_U(d::Int, dt::Float64, N::Int)
    # without ancilla d=2, with ancilla d=2*da where da is the dimension of ancilla
    Us = Matrix[]
    for i in 1:2*N-3
        H = randn(d*d,d*d) + 1im*randn(d*d,d*d)
        H = (H+H')/2
        push!(Us, exp(im*dt*H))
    end
    return Us
end


function MPDO_norm(M1::myMPDO)

    ls = left_environments(M1, M1)

    return ls[end]
end

function MPS_overlap(M1::myMPS, M2::myMPS)

    ls = left_environments(M1, M2)

    return ls[end]
end


function optimize_overlap_onelayer(M1::myMPDO,M2::myMPDO,dir = "l";truncation = true, max_bd = 1024, max_err=1E-10)
    ## Act a sequential circuit on M1 and maximize |<M2|M1>|
    ## dir="l" - add unitary from left to right - this assumes that M1 is right-canonical ** important **
    ## dir="r" - add unitary from right to left - this assumes that M1 is left-canonical **important**
    ## note - unitaries are acted on M1 (ancilla leg) - and we will return the modified M1 in left/right canonical form
    M1cp = copy(M1)
    # println("M1 pointer: ", objectid(M1))
    # println("M1cp pointer: ", objectid(M1cp)) # examine the memory address is changed
    ov_opts = Float64[] #optimized fidelity after applying each unitary

    rs = right_environments(M1,M2)
    push!(ov_opts,abs(tr(rs[end])))
    N = length(M1)
    l_env = diagm(ones(1))
    for i in 1:N-1
        r_env = rs[N-i] ## right enviroment
        
        U_env = getU_env(M1cp, M2, i, l_env, r_env)
        ov, U_opt = optimal_U_from_env(U_env)
        push!(ov_opts, ov)
        
        ~, M1cp = unitary_evol_two_site_ancilla(M1cp, U_opt, i, dir, truncation = truncation, max_bd = max_bd, max_err=max_err)
        l_env = apply_TM_l(M1cp.TensorList[i],M2.TensorList[i],l_env)
    end
    # println("M1cp pointer: ", objectid(M1cp)) 
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
    ov_all = zeros(0)
    for j in 1:iters
        println("----iteration $(j) -----")
        M1cp = canonicalize_right(M1cp) # should be M1cp = canonicalize_right(M1cp)
        println("debug")
        M2cp = canonicalize_right(M2cp); # should be M2cp = canonicalize_right(M2cp)
        M1cp, ovs = optimize_overlap_onelayer(M1cp,M2cp);
        append!(ov_all, ovs)
        M2cp, ovs = optimize_overlap_onelayer(M2cp,M1cp);
        append!(ov_all, ovs)
        ov = ovs[end]
        chi1 = max_bond_dim(M1cp)
        chi2 = max_bond_dim(M2cp)
        println("Bond dimensions: $chi1,$chi2")
        println("Overlap: $ov")
    end
    return M1cp,M2cp,ov_all
end

######### MPS to MPDO through adding noise and adding ancillas ########

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

########## Two-site channel implementation ##########

function xx_channel_dilation(p::Float64)
    ## Bond-factorized dilation of the two-site channel
    ##     N(rho) = (1-p/2) rho + (p/2) (X⊗X) rho (X⊗X)
    ## Returns (WL, WR), each with index order (output, ancilla, input, virtual).
    ## Contracting the virtual legs gives the Kraus isometry
    ##     sqrt(1-p/2) I⊗I |1> + sqrt(p/2) X⊗X |2>,
    ## with the Kraus label on WL's ancilla leg (WR's ancilla is trivial).
    Id = [1.0 0.0; 0.0 1.0]
    X  = [0.0 1.0; 1.0 0.0]

    WL = zeros(2, 2, 2, 2) # (output, ancilla, input, virtual-R)
    WL[:, 1, :, 1] = sqrt(1 - p/2) * Id
    WL[:, 2, :, 2] = sqrt(p/2) * X

    WR = zeros(2, 1, 2, 2) # (output, ancilla, input, virtual-L)
    WR[:, 1, :, 1] = Id
    WR[:, 1, :, 2] = X

    return WL, WR
end

function absorb_W_right(A::Array{<:Number,4}, W::Array{<:Number,4})
    ## Apply Kraus tensor W (output, ancilla, input, virtual) to the physical
    ## leg of LPDO tensor A (l, s, a, r); merge W's ancilla leg into A's ancilla
    ## leg and route W's virtual leg into A's right bond.
    @tensor B[l, s, a, a2, r, v] := A[l, s0, a, r] * W[s, a2, s0, v]
    dl, ds, da, da2, dr, dv = size(B)
    return reshape(B, dl, ds, da*da2, dr*dv)
end

function absorb_W_left(A::Array{<:Number,4}, W::Array{<:Number,4})
    ## Same as absorb_W_right, but route W's virtual leg into A's left bond.
    @tensor B[l, v, s, a, a2, r] := A[l, s0, a, r] * W[s, a2, s0, v]
    dl, dv, ds, da, da2, dr = size(B)
    return reshape(B, dl*dv, ds, da*da2, dr)
end

function insert_bond_wire(A::Array{T,4}, dw::Int) where T<:Number
    ## Tensor a dw-dimensional identity wire onto both bonds of A, used to
    ## route the periodic-boundary virtual leg through the bulk.
    dl, ds, da, dr = size(A)
    B = zeros(T, dl, dw, ds, da, dr, dw)
    for k in 1:dw
        B[:, k, :, :, :, k] = A
    end
    return reshape(B, dl*dw, ds, da, dr*dw)
end

function add_noise_double(M::myMPS{T}, p::Float64; pbc::Bool = true) where T
    ## Apply the two-site channel N(rho) = (1-p/2) rho + (p/2) XX rho XX on
    ## every bond of the chain: even bonds (1,2),(3,4),..., odd bonds
    ## (2,3),(4,5),..., and - for pbc - the boundary bond (N,1).
    ## Input: MPS; output: purification tensor (half of LPDO) as myMPDO.
    ##
    ## The boundary bond is the expensive one: its virtual leg has to be routed
    ## through the whole bulk, which doubles every bond on top of the doubling the
    ## bulk channels already cause (so D -> 4 D_mps for pbc, 2 D_mps for obc).
    N = length(M)
    @assert iseven(N) "add_noise_double assumes an even number of sites"

    WL, WR = xx_channel_dilation(p)

    # promote MPS tensors (l,s,r) to LPDO tensors (l,s,a,r) with trivial ancilla
    Ts = [reshape(A, size(A,1), size(A,2), 1, size(A,3)) for A in M.TensorList]

    # bulk bonds: even layer, then odd layer
    for start in (1, 2), i in start:2:N-1
        Ts[i]   = absorb_W_right(Ts[i], WL)
        Ts[i+1] = absorb_W_left(Ts[i+1], WR)
    end

    if pbc
        # boundary bond (N,1): virtual leg routed through the bulk
        Ts[1] = absorb_W_right(Ts[1], WR)
        for i in 2:N-1
            Ts[i] = insert_bond_wire(Ts[i], 2)
        end
        Ts[N] = absorb_W_left(Ts[N], WL)
    end

    return myMPDO(Ts)
end

function compress_lpdo(M::myMPDO; max_bd::Int = 4096, max_err::Float64 = 1E-8,
                       verbose::Bool = false)
    ## Variationally truncate the purification bond of an LPDO.
    ##
    ## add_noise_double is exact and leaves the bond at 4x the input MPS bond (the
    ## channel dilation doubles it, and routing the periodic bond's virtual leg
    ## through the bulk doubles it again). Most of that is redundant: a proper
    ## canonicalize-and-truncate typically removes a factor of 3-4 while discarding
    ## squared Schmidt weight below max_err.
    ##
    ## Sweep left first without truncating, so that everything to the left of a bond
    ## is left-canonical and the right sweep sees the true Schmidt values - only then
    ## is the truncation variationally optimal.
    ##
    ## Note this is a truncation of the purification, so the discarded weight bounds
    ## the error in |M> and hence (to first order) the error in rho = M M^dagger.
    ## The canonicalizers also normalize, i.e. the result has tr(rho) = 1; the input
    ## from add_noise_double already does, so that is a no-op up to the truncation.
    D0 = max_bond_dim(M)
    ## Work on a fresh tensor list: canonicalize_* assign into the myMPDO they are
    ## given, so without this the caller's object would come back truncated.
    M = myMPDO(copy(M.TensorList))
    M = canonicalize_left(M)
    M = canonicalize_right(M; truncation = true, max_bd = max_bd, max_err = max_err)
    verbose && println(" --- LPDO compressed: bond $(D0) -> $(max_bond_dim(M)) ---")
    return M
end

##########################################################

function add_CP(M::myMPDO, Ks::Array,i::Int)

    M_new = copy(M)
    tmp = copy(M.TensorList[i])
    @tensor tmp2[l,s,a,r] := Ks[s,sp] * tmp[l,sp,a,r]

    M_new.TensorList[i] = tmp2

    return M_new

end


function add_operator(A::myMPS, Ks::Array,i::Int)

    A_new = copy(A)
    tmp = copy(A.TensorList[i])
    @tensor tmp2[l,s,r] := Ks[s,sp] * tmp[l,sp,r]

    A_new.TensorList[i] = tmp2

    return A_new

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


function compute_trace_distance(rho1,rho2)
    S = svdvals(rho1-rho2)
    f = sum(S)/2
    return f
end


function fidelity(rho1::myMPDO, rho2::myMPDO; is_td=0)
    
    rho1n = MPDO_to_dense(rho1);
    rho2n = MPDO_to_dense(rho2);

    rho1_dense = rho1n*rho1n'
    rho2_dense = rho2n*rho2n'

    if is_td == 0
        F0 = compute_fidelity(rho1_dense, rho2_dense)
    else
        F0 = compute_trace_distance(rho1_dense, rho2_dense)
    end

    return F0

end


function fidelity_op(rho::myMPDO,op1::Array,op2::Array,i::Int,j::Int)

    # compute fidelity correlator (rho, O1^i O2^j rho O2'^j O1'^i). Here the input rho is LPDO (half of MPDO)

    rho_op = add_CP(rho, op1, i)
    rho_op = add_CP(rho_op, op2, j)

    rho1 = MPDO_to_dense(rho);
    rho2 = MPDO_to_dense(rho_op);

    rho_dense = rho1*rho1'
    rho_op_dense = rho2*rho2'
    F0 = compute_fidelity(rho_dense, rho_op_dense)

    return F0

end


function trace_distance_op(rho::myMPDO,op1::Array,op2::Array,i::Int,j::Int)
    # Here the input rho is LPDO (half of MPDO)

    # compute (rho, O1^i O2^j rho O2'^j O1'^i). Here the input rho is LPDO (half of MPDO)

    rho_op = add_CP(rho, op1, i)
    rho_op = add_CP(rho_op, op2, j)

    rho1 = MPDO_to_dense(rho);
    rho2 = MPDO_to_dense(rho_op);

    rho_dense = rho1*rho1'
    rho_op_dense = rho2*rho2'
    td = compute_trace_distance(rho_dense, rho_op_dense)

    return td
end

function fidelity_exact(rho0::myMPDO, rho1::myMPDO)

    # Here the input rho is LPDO (half of MPDO)

    rho0d = MPDO_to_dense(rho0);
    rho1d = MPDO_to_dense(rho1);

    rho0_dense = rho0d*rho0d'
    rho1_dense = rho1d*rho1d'
    F0 = compute_fidelity(rho0_dense, rho1_dense)

    return F0
end





function fidelity_op_mps(A::myMPS,op1::Array,op2::Array,i::Int,j::Int)

    # compute |<psi|O1^i O2^j|psi>|

    AS = add_operator(add_operator(A, op1,i), op2, j)
    F0 = abs(only(left_environments(A, AS)[end]))

    return F0

end


function add_ancillas(M::myMPDO{T}; da=2) where T
    ## add ancillas |0> to MPDO
    ## da is dimension of ancilla
    
    Mcp = copy(M)

    for i in 1:length(M)
        Mi = Mcp.TensorList[i]
        D1,d,dp,D2 = size(Mi)
        Mi_acl = zeros(eltype(Mi), D1, d, dp, da, D2)
        
        for l in 1:D1, s in 1:d, a in 1:dp, r in 1:D2
            Mi_acl[l, s, a, 1, r] = Mi[l, s, a, r]
        end

        Mi_acl_reshaped = reshape(Mi_acl, D1, d, dp * da, D2)
        Mcp.TensorList[i] = Mi_acl_reshaped
    end

    return Mcp
end

######### Channel with purification #########

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

######### Random unitary reshuffling for ancilla #########

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

######### MPDO to MPS and dense density matrix #########

function MPS_to_dense(M::myMPS{T}) where T
    ## Only run this for small systems!!
    ## output a wavefunction of MPS state

    d = phys_dim(M)
    L = length(M)
    tmp = M.TensorList[1]
    for i in 2:L
        A = M.TensorList[i]
        dR = size(A,3)
        @tensor tmp2[l,s1,s2,r] := tmp[l,s1,rp] * A[rp,s2,r]
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

######### Other fidelity measures #########

function truncrank(S::Vector{<:Real}, max_bd::Int, max_err::Float64)
    ## Number of singular values to keep: drop the smallest tail whose relative
    ## squared weight is below max_err, capped at max_bd. Same rule as mytruncate
    ## but for an *un-normalized* S (uses the relative threshold max_err*sum(S^2)).
    tot = sum(abs2, S)
    err = 0.0
    set_bd = min(length(S), max_bd)
    for i in length(S):-1:1
        err += S[i]^2
        if err > max_err*tot
            if i < max_bd
                set_bd = i
            end
            break
        end
    end
    return max(set_bd, 1)
end

function MPDO_to_MPO(M::myMPDO{T}; max_bd::Int = 1024, max_err::Float64 = 1E-12, gram::Bool = true) where T
    ## Convert the LPDO M to the MPO of rho = M M^dagger, compressing on the fly.
    ## Zip-up: sweep left to right carrying an already-truncated bond, and build
    ## each site by contracting that bond with A and conj(A) and immediately
    ## SVD-truncating. The full DL^2 x DR^2 site tensor is never materialized;
    ## the largest object is Theta ~ (b*d^2) x (DL_purif^2). Returns a
    ## right-canonical, truncated MPO (as the previous implementation did).
    ##
    ## gram = true: get the left singular vectors from the (b*d^2)^2 Gram matrix
    ## instead of running a full SVD on the short-and-very-wide matrix. Same
    ## decomposition, two GEMMs instead of one LAPACK SVD - see below.
    N = length(M)
    Ws = Array{T,4}[]
    C = ones(T, 1, 1, 1)   # [mpo bond b, ket-purif bond, bra-purif bond]
    for i in 1:N
        A = M.TensorList[i]   # [l, s, a, r]
        @tensor Theta[b, s1, s2, rk, rb] := C[b, l, m] * A[l, s1, a, rk] * conj(A)[m, s2, a, rb]
        b, d1, d2, DRk, DRb = size(Theta)
        mat = reshape(Theta, (b*d1*d2, DRk*DRb))
        if gram && size(mat, 1) < size(mat, 2)
            ## mat is short (m = b*d^2) and very wide (n = DRk*DRb, the *pair* of
            ## purification bonds). A full SVD of it costs O(m^2 n) and also builds
            ## the n x m matrix V - by far the most expensive step of the whole
            ## correlator. All we need is U (-> the MPO tensor) and S*V' = U'*mat
            ## (-> the bond carried to the next site), so take U from the m x m
            ## Gram matrix instead:  mat*mat' = U S^2 U'.  Two GEMMs, no big V.
            ## Note: eigenvalues are S^2, so directions kept at relative weight
            ## max_err are resolved to ~sqrt(eps/max_err) - fine at max_err 1e-12,
            ## set gram = false if you ever push max_err below ~1e-14.
            F = eigen(Hermitian(mat * mat'))          # ascending eigenvalues
            S = sqrt.(max.(reverse(F.values), 0.0))   # = singular values, descending
            k = truncrank(S, max_bd, max_err)
            U = F.vectors[:, end:-1:end-k+1]           # top-k, descending
            push!(Ws, reshape(U, (b, d1, d2, k)))
            C = reshape(U' * mat, (k, DRk, DRb))      # = S_k V_k'
        else
            U = nothing; S = nothing; V = nothing
            try
                U, S, V = svd(mat, alg = LinearAlgebra.DivideAndConquer())
            catch
                U, S, V = svd(mat, alg = LinearAlgebra.QRIteration())
            end
            k = truncrank(S, max_bd, max_err)
            U = U[:, 1:k]; S = S[1:k]; V = V[:, 1:k]
            push!(Ws, reshape(U, (b, d1, d2, k)))
            C = reshape(diagm(0 => S) * V', (k, DRk, DRb))
        end
    end
    # residual right bond is 1; fold the leftover scalar into the last tensor
    Ws[end] = Ws[end] .* C[1, 1, 1]
    MPO = myMPO(Ws)
    MPO = canonicalize_right(MPO, truncation = true, max_bd = max_bd, max_err = max_err)
    return MPO
end

function MPO_overlap(M1::myMPO{T}, M2::myMPO{T}) where T
    ## Compute the overlap of M1 and M2: that is tr(M1*M2')
    ## return the overlap
    N = length(M1)
    @assert length(M1) == length(M2)
    l_env = diagm(ones(T,1))
    for i in 1:N
        A1 = M1.TensorList[i]
        A2 = M2.TensorList[i]
        l_env = apply_TM_l(A1,A2,l_env)
    end
    return LinearAlgebra.tr(l_env)
end

function MPO_product_sqaure_trace(M1::myMPO{T}, M2::myMPO{T}) where T
    ## tr(rho1*rho2*rho1*rho2)
    l_env = ones(T,(1,1,1,1))
    for i in 1:length(M1)
        A1 = M1.TensorList[i]
        A2 = M2.TensorList[i]
        @tensor l_env_new[r1,r2,r3,r4] := l_env[l1,l2,l3,l4] * A1[l1,s1,s2,r1] * A2[l2,s2,s3,r2] * A1[l3,s3,s4,r3] * A2[l4,s4,s1,r4]
        l_env = l_env_new
    end
    return l_env[1,1,1,1]
end

function operator_MPO(ops::Dict, N::Int, T::Type)
    ## Bond-dimension-1 MPO of a product operator: the given on-site operator at
    ## each site key in `ops`, identity elsewhere.
    ## Index order matches myMPO: (left bond, ket, bra, right bond).
    Idm = Matrix{T}(I, 2, 2)
    Ts = Array{T,4}[]
    for i in 1:N
        o = haskey(ops, i) ? T.(ops[i]) : Idm
        push!(Ts, reshape(o, (1, 2, 2, 1)))
    end
    return myMPO(Ts)
end

function MPDO_op_rho_op_rho_trace(M::myMPDO, ops::Dict; renorm::Bool = true)
    ## tr(rho O rho O) for rho = M M^dagger (M the LPDO / purification half) and
    ## O = prod_k ops[k] a product operator (identity on every site not in `ops`).
    ## The four layers  A, conj(A), A, conj(A)  are contracted directly, site by
    ## site: rho is never built as an MPO, so there is no compression and no SVD
    ## anywhere - the result is exact. `ops` empty  ->  tr(rho^2).
    ##
    ## One site (p = left bonds of the four layers, q = right bonds):
    ##   E'[q1,q2,q3,q4] = E[p1,p2,p3,p4] * A[p1,s1,a,q1] * B[p2,s3,a,q2]
    ##                                    * A[p3,s3,b,q3] * B[p4,s1,b,q4]
    ## with B[p,s',a,q] = conj(A)[p,s,a,q]*O[s,s'] the bra layers carrying O, and
    ## a / b the ancilla index shared by layers 1-2 / 3-4. Layers 1,2 make up the
    ## first rho and layers 3,4 the second; the two operator insertions sit on the
    ## (2,3) and (4,1) physical bonds. With O = 1 on every site this reduces to
    ## tr(rho^2) = tr(rho rho^dagger), i.e. exactly MPO_overlap(rho_mpo, rho_mpo).
    ##
    ## Cost / peak memory per site: O(D^5 d^2 dA) / O(D^4 d dA), with D the
    ## *purification* bond of M. Exact, but only cheap while D stays small: the
    ## four-layer environment is D^4, whereas the compressed-MPO route only ever
    ## carries a b^2 environment with b the truncated rho-MPO bond. Prefer this
    ## when b is not much smaller than D^2 (small systems, exact reference
    ## values), and MPDO_to_MPO + MPO_product_sqaure_trace when D is large.
    ##
    ## Returns (val, logscale): the trace is val*exp(logscale). The environment is
    ## renormalized at every site so that long chains cannot over/underflow.
    N = length(M)
    T = eltype(M.TensorList[1])
    for (_, O) in ops
        T = promote_type(T, eltype(O))
    end
    E = ones(T, 1, 1, 1, 1)     # [p1, p2, p3, p4] - one left bond per layer
    logscale = 0.0
    for n in 1:N
        A = convert(Array{T,4}, M.TensorList[n])   # [l, s, a, r]
        if haskey(ops, n)
            P = convert(Matrix{T}, ops[n])
            @tensor B[p, so, a, q] := conj(A)[p, si, a, q] * P[si, so]
        else
            B = conj(A)
        end
        ## absorb one layer at a time, so nothing bigger than ~D^4*d*dA shows up
        @tensor X[p2, p3, p4, s1, a, q1] := E[p1, p2, p3, p4] * A[p1, s1, a, q1]
        @tensor Y[p3, p4, s1, q1, s3, q2] := X[p2, p3, p4, s1, a, q1] * B[p2, s3, a, q2]
        @tensor Z[p4, s1, q1, q2, b, q3] := Y[p3, p4, s1, q1, s3, q2] * A[p3, s3, b, q3]
        @tensor Enew[q1, q2, q3, q4] := Z[p4, s1, q1, q2, b, q3] * B[p4, s1, b, q4]
        E = Enew
        if renorm
            nrm = norm(E)
            if nrm > 0
                E ./= nrm
                logscale += log(nrm)
            end
        end
    end
    return E[1, 1, 1, 1], logscale
end

function renyi2_correlator(M::myMPDO, O1::Array, O2::Array, i::Int, j::Int;
                           max_bd::Int = 1024, max_err::Float64 = 1E-12,
                           canonicalize::Bool = true,
                           lpdo_max_bd::Int = max(max_bond_dim(M), 1),
                           lpdo_max_err::Float64 = max_err,
                           method::Symbol = :auto, direct_max_elems::Real = 2^23)
    ## Renyi-2 correlator  C_{ij} = tr(O_i O_j rho O_i O_j rho) / tr(rho^2),
    ## with rho = M M^dagger the physical density matrix (M is the LPDO / half).
    ## Both traces are quadratic in rho, so any overall normalization cancels
    ## (canonicalize_* normalizes the purification, which is therefore harmless).
    ##
    ## canonicalize = true first brings the purification to right-canonical form
    ## and drops Schmidt weight below lpdo_max_err. Two reasons:
    ##  (a) the zip-up in MPDO_to_MPO truncates the rho-MPO bond as if the tails to
    ##      its right were orthonormal - that only holds for right-canonical M, so
    ##      without this the compression is not variationally controlled;
    ##  (b) add_noise_double inflates the purification bond by ~4x (channel
    ##      dilation + boundary wire) and every downstream cost goes like D^2
    ##      (MPO route) or D^4 (direct route), so this is the cheapest speedup
    ##      available - it is a 2-layer sweep, not a 4-layer one.
    ## lpdo_max_err is the knob that does the work (discarded squared Schmidt
    ## weight per bond; the error in C grows like its square root). lpdo_max_bd is
    ## only a hard ceiling and defaults to the bond M already has, i.e. no extra
    ## hard truncation - a bond cap throws away whatever weight sits above it
    ## regardless of size, so set it only if you need to bound memory. Note it is
    ## unrelated to max_bd, which caps the *rho-MPO* bond b, a different (and
    ## normally much smaller) object than the purification bond D.
    ##
    ## method = :direct - contract the four layers of M directly
    ##                    (MPDO_op_rho_op_rho_trace): exact, no rho-MPO and no
    ##                    SVD, but the environment is D^4 in the purification
    ##                    bond D, so this is only for small D.
    ##        = :mpo    - compress rho into an MPO first (MPDO_to_MPO) and
    ##                    contract MPOs: approximate (max_bd / max_err) but the
    ##                    environment is only b^2 in the truncated rho-MPO bond b,
    ##                    which is the affordable route once D is large.
    ##        = :auto   - :direct while its peak working set stays below
    ##                    direct_max_elems tensor entries, else :mpo.
    if canonicalize
        D0 = max_bond_dim(M)
        ## Work on a fresh tensor *list* before canonicalizing. canonicalize_* both
        ## mutate and return the myMPDO they are handed (they assign into
        ## M.TensorList), so without this the caller's object - e.g. M1[1] from
        ## xxz_get_lpdo - would come back canonicalized and truncated. Rebinding M
        ## below cannot prevent that: it only renames this function's local, the
        ## caller still holds the original object.
        ## A shallow copy is enough (and avoids duplicating all the data): the
        ## canonicalizers *replace* list entries with freshly allocated tensors and
        ## never write into an existing one, so the shared arrays are never touched.
        M = myMPDO(copy(M.TensorList))
        ## Sweep left first (no truncation) so that everything to the left of the
        ## bond is left-canonical; only then does the right sweep see the true
        ## Schmidt values and truncate optimally.
        M = canonicalize_left(M)
        M = canonicalize_right(M; truncation = true, max_bd = lpdo_max_bd,
                                  max_err = lpdo_max_err)
        println(" --- LPDO right-canonicalized: bond $(D0) -> $(max_bond_dim(M)) ---")
    end
    if method == :auto
        peak = float(max_bond_dim(M))^4 * phys_dim(M) * ancilla_dim(M)
        method = peak <= direct_max_elems ? :direct : :mpo
    end
    if method == :direct
        ops = Dict{Int,Array}(i => O1, j => O2)
        num, lnum = MPDO_op_rho_op_rho_trace(M, ops)                 # tr(rho A rho A)
        println(" --- num done (direct four-layer) ---")
        den, lden = MPDO_op_rho_op_rho_trace(M, Dict{Int,Array}())   # tr(rho^2)
        println(" --- den done (direct four-layer) ---")
        return real((num / den) * exp(lnum - lden))
    elseif method == :mpo
        rho_mpo = MPDO_to_MPO(M; max_bd = max_bd, max_err = max_err)
        println(" --- rho_mpo done (bond $(max_bond_dim(rho_mpo))) ---")
        T = eltype(rho_mpo.TensorList[1])
        A_mpo = operator_MPO(Dict(i => O1, j => O2), length(M), T)
        num = MPO_product_sqaure_trace(rho_mpo, A_mpo)  # tr(rho A rho A)
        println(" --- num done ---")
        den = MPO_overlap(rho_mpo, rho_mpo)             # tr(rho^2)
        println(" --- den done ---")
        return real(num / den)
    else
        error("renyi2_correlator: unknown method $(method) - use :auto, :direct or :mpo")
    end
end

function trMPO(M::myMPO{T}) where T
    tmp = I
    for i in 1:length(M)
        A = M.TensorList[i]
        @tensor Amat[l,r] := A[l,s,s,r]
        tmp = tmp*Amat 
    end
    return tr(tmp)
end

function MPO_overlaps_normalized(M1::myMPO{T}, M2::myMPO{T}) where T
    ## Compute the normalized overlap of M1 and M1, M2 and M2, M1 and M2
    ## return the normalized overlap
    tr1 = trMPO(M1)
    tr2 = trMPO(M2)

    ov11 = MPO_overlap(M1,M1)
    ov22 = MPO_overlap(M2,M2)
    ov12 = MPO_overlap(M1,M2)

    pur1 = ov11/(tr1^2)
    pur2 = ov22/(tr2^2)
    tr12 = ov12/(tr1*tr2) 

    ov1212 = MPO_product_sqaure_trace(M1, M2)

    tr1212 = ov1212/(tr1^2*tr2^2)
    return pur1,pur2,tr12,tr1212
end

function sub_and_super_fidelity(M1::myMPDO{T}, M2::myMPDO{T}) where T
    MPO1 = MPDO_to_MPO(M1)
    MPO2 = MPDO_to_MPO(M2)
    @show max_bond_dim(MPO1), max_bond_dim(MPO2)
    pur1, pur2, tr12, tr1212 = MPO_overlaps_normalized(MPO1, MPO2)
    sub_fidelity = tr12 + sqrt(2)*sqrt(tr12^2 - tr1212)
    super_fidelity = tr12 + sqrt((1-pur1)*(1-pur2))
    return sub_fidelity, super_fidelity
end

######### Optimization of sequential unitary #########

function unitary_evolution_two_floor_ancilla(M::myMPDO, Us::Vector{<:Matrix};truncation = true, max_bd = 1024, max_err=1E-10)
    ## Apply a two-floor-unitary Us (length 2N-3) on M
    ## Us order: U_{12}, U_{23}, ... U{n-1,n} U_{n-2,n-1} ... U_{12} 
    ## Assume Right canonical form as input
    ## return all intermidiate MPDOs and the last one canonical center is at site 2.
    # Output length is 2(N-1)
    
    N = length(M)
    @assert length(Us) == 2*N-3  ## This is one-floor constraint
    
    M1_interms = myMPDO[]
    push!(M1_interms, M);
    M1cp = copy(M)
    for i in 1:N-2  ### V20250511 - change grouping 1 - N-2
        U = Us[i]
        site = i;
        ~, M1_out = unitary_evol_two_site_ancilla(M1cp, U, site, "l"; truncation = truncation, max_bd = max_bd, max_err = max_err)
        push!(M1_interms, M1_out)
        M1cp = copy(M1_out)
    end
    for i in N-1:2*N-3 
        site = 2*N-i-2
        U = Us[i]
        ~, M1_out = unitary_evol_two_site_ancilla(M1cp, U, site, "r"; truncation = truncation, max_bd = max_bd, max_err = max_err)
        push!(M1_interms, M1_out)
        M1cp = copy(M1_out) 
    end
    return M1_interms
end

    
function optimal_U_from_env(U_env::Array{T,4}) where T
    ### maximize |tr(env' * U)|
    ### return U (as a rank 4 tensor) and maximum
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
    return ov, U_opt
end

function getU_env(M1::myMPDO, M2::myMPDO, site::Int, lenv::Matrix, renv::Matrix)
    ### Compute the following
    ###        |       |
    ### ------ A1 ---- A2----
    ### |      |       |    |
    ### lenv   |       |   renv
    ### |      |       |    |
    ### ------ B1*---- B2----
    ###        |       |

    ## same as?
    ###        |       |
    ### ------ B1*---- B2*---
    ### |      |       |    |
    ### lenv   |       |   renv
    ### |      |       |    |
    ### ------ A1----- A2----
    ###        |       |
    
    ### This tensor network gives U_env
    
    A1 = M1.TensorList[site]
    A2 = M1.TensorList[site+1]
        
    B1 = M2.TensorList[site]
    B2 = M2.TensorList[site+1]
        
    @tensor U_env[a1,a2,a1p,a2p] := lenv[b1,t1] * A1[t1,s,a1,t2] * conj(B1)[b1,s,a1p,b2] * A2[t2,ss,a2,t3] * renv[t3,b3] *conj(B2)[b2,ss,a2p,b3]
    return U_env
end


function check_environment(M1::myMPDO, M2::myMPDO, site::Int)
    ## the sites where unitary act is site, site+1
    l_env = diagm(ones(1)) 
    r_env = diagm(ones(1)) 
    for i in 1:site-1
        A = M1.TensorList[i]
        B = M2.TensorList[i]
        l_env = apply_TM_l(A,B,l_env)
    end

    for i in length(M1):-1:site+2
        A = M1.TensorList[i]
        B = M2.TensorList[i]
        r_env = apply_TM_r(A,B,r_env)
    end

    return l_env, r_env
end

    
function optimize_overlap_sweep_reverse_order(M1_interms::Vector{<:myMPDO}, M2::myMPDO;truncation = true, max_bd = 1024, max_err=1E-10,verbose=1, debug=0)
    ### Optimize Us in the reverse order
    ### Act the optimized Us on M2
    ### return the M2 intermidates and optimized overlap
    ### Assume M2 is right canonical
    
    N = length(M1)
    ov_opts = Float64[] #optimized fidelity after applying each optimization
    M2_interms = myMPDO[]
    
    ## Left to right sweep U_{12} .... U_{n-1, n}
    M1cp = M1_interms[end] ## M1 after acting on all Us ## equally well to choose end-1, does not matter except the first ov
    M2cp = copy(M2)
    push!(M2_interms, copy(M2cp))
    l_env = diagm(ones(1))
    r_envs = right_environments(M1cp,M2cp) ## This is very tricky - but correct to choose M1_interms[end] in the first argument
    ov = abs(tr(r_envs[end]))
    if(verbose>2)
        println("Overlap before optimization: ",ov)
    end
    push!(ov_opts,ov) ## Initial overlap with Us
    for i in 1:N-2  ## V20250511 change grouping to 1 - N-2 and N-1 - 1
        if(verbose>1)
            println("Optimizing unitary acting on $(i), $(i+1)")
        end
        M1cp = M1_interms[end-i]
        r_env = r_envs[N-i] ## right enviroment, after contracting the **3rd**(each iteration +1 until identity(1)) site

        if debug == 1
            ## see whether the environment tensor is as expected
            l_env_db, r_env_db = check_environment(M1cp, M2cp,i)
            println("Error for l_env: ", norm(l_env-l_env_db))
            println("Error for r_env: ", norm(r_env-r_env_db))
            # l_env, r_env = check_environment(M1cp, M2cp,i)
        end
        
        U_env = getU_env(M1cp, M2cp, i, l_env, r_env)
        ov, U_opt = optimal_U_from_env(U_env)  

        if debug == 2
            println(U_opt)
        end
        
        push!(ov_opts,ov) ## This is the optimized overlap
        if(verbose>1)
            println("Optimized overlap through svd: ",ov)
        end
        ~, M2cp = unitary_evol_two_site_ancilla(M2cp, Matrix(U_opt'), i, "l"; truncation = truncation, max_bd = max_bd, max_err = max_err)
        if(verbose>2)
            ov_debug = compute_overlap(M1_interms[end-i], M2cp)
            println("Debug mode: Directly compute the overlap: ", ov_debug)
        end
        push!(M2_interms, copy(M2cp))
        l_env = apply_TM_l(M1cp.TensorList[i],M2cp.TensorList[i],l_env)
        
    end
    
    ## Right to left sweep U_{n-2,n-1} to U_{1,2} 
    #~,M2cp = canonicalize_right_one_site(M2cp, N) ## This is no longer needed after regrouping: V20250511
    M1cp = M1_interms[N-1]  ## M1 acted with U_{12} .... U_{N-2, N-1}
    l_envs = left_environments(M1cp,M2cp) ## This is again very tricky - but correct to choose M1_interms[N-2] in the first argument
    r_env = diagm(ones(1))
    #r_env = apply_TM_r(M1cp.TensorList[N],M2cp.TensorList[N],r_env) # V20250511 commented out
    for i in N-1:-1:1
        if(verbose>1)
            println("Optimizing unitary acting on $(i), $(i+1)")
        end
        M1cp = M1_interms[i]  # 0516
        l_env = l_envs[i]

        if debug == 1
            ## see whether the environment tensor is as expected
            l_env_db, r_env_db = check_environment(M1cp, M2cp,i)
            println("Error for l_env: ", norm(l_env-l_env_db))
            println("Error for r_env: ", norm(r_env-r_env_db))
            # l_env, r_env = check_environment(M1cp, M2cp,i)
        end
        
        U_env = getU_env(M1cp, M2cp, i, l_env, r_env)
        ov, U_opt = optimal_U_from_env(U_env)

        if debug == 2
            println(U_opt)
        end

        push!(ov_opts,ov) ## This is the optimized overlap
        if(verbose>1)
            println("Optimized overlap through svd: ",ov)
        end
        ~, M2cp = unitary_evol_two_site_ancilla(M2cp, Matrix(U_opt'), i, "r"; truncation = truncation, max_bd = max_bd, max_err = max_err)  
        if(verbose>2)
            ov_debug = compute_overlap(M1_interms[i], M2cp)
            println("Debug mode: Directly compute the overlap: ", ov_debug)
        end
        push!(M2_interms, copy(M2cp))
        if(i>1)
            # M1cp = M1_interms[i-1]  # 0516
            r_env = apply_TM_r(M1cp.TensorList[i+1],M2cp.TensorList[i+1],r_env)
        end
        if(verbose == 1 && i == 1)
            println("New overlap: ", ov_opts[end])
        end
    end
    ## Note that the M2_interms[end] is now with canonical center at site 2
    return M2_interms, ov_opts
end

function optimize_overlap_onefloor(M1::myMPDO,M2::myMPDO,Us::Vector{<:Matrix};truncation = true, max_bd = 1024, max_err=1E-10, debug=0)
    ## Act a sequential circuit on M1 and maximize |<M2|M1>|
    ## Us = initial guess of the unitary network. 
    ## Us order: U_{12}, U_{23}, ... U{n-1,n} U_{n-2,n-1} ... U_{12} 
    ## one floor: length(Us) = 2N-3 (k-floor length(Us) = k*(2N-4) + 1) 
    ## Assumes that M1 and M2 are right-canonical ** important **
    ## note - unitaries are acted on M2 (ancilla leg) - and we will return the modified M2 in right canonical form
    ## This only sweep once

    M1 = canonicalize_right(M1)
    M2 = canonicalize_right(M2)
    M1_interms = unitary_evolution_two_floor_ancilla(M1, Us; truncation = truncation, max_bd = max_bd, max_err = max_err)
    
    M2_interms, ov_opts = optimize_overlap_sweep_reverse_order(M1_interms, M2; truncation = truncation, max_bd = max_bd, max_err = max_err, debug=debug)
    
    return M2_interms, ov_opts
end

function optimize_overlap_sweep_forward_order(M1::myMPDO, M2_interms::Vector{<:myMPDO}; truncation = true, max_bd = 1024, max_err = 1E-10,verbose=1, debug=0)
    ### M2 interms = [M2, U'_{12}*M2, U'_{23}*M2 ....], U_{12} is the LAST of Us
    ### Direct apply previous method to optimize U_{12}, U_{23} .... (forward order)
    M1_interms, ov_opts = optimize_overlap_sweep_reverse_order(M2_interms, M1; truncation = truncation, max_bd = max_bd, max_err = max_err,verbose=verbose, debug=debug)
    return M1_interms, ov_opts
end

function optimize_overlap_onefloor_sweep(M1::myMPDO,M2::myMPDO,Us::Vector{<:Matrix},nsweep::Int = 1;truncation = true, max_bd = 1024, max_err=1E-10,verbose=1)
    ### Sweep for optimizing Us
    ### Assuming both M1 and M2 are right canonical
        
    M1 = canonicalize_right(M1)
    M2 = canonicalize_right(M2)
    println("Initial overlap: ", compute_overlap(M1, M2))

    
    M1_interms = unitary_evolution_two_floor_ancilla(M1, Us; truncation = truncation, max_bd = max_bd, max_err = max_err)
    
    ov_opts = Float64[]
    for k in 1:nsweep
        println("Sweep: $k")
        if(verbose>1)
            println("Reverse Sweep")
        end
        M2_interms, ovs = optimize_overlap_sweep_reverse_order(M1_interms, M2; truncation = truncation, max_bd = max_bd, max_err = max_err,verbose=verbose)
        println("Max bond dim: ", max_bond_dim(M2_interms[end]))
        
        push!(ov_opts, ovs...)

        if(verbose>1)
            println("Forward Sweep")
        end
        M1_interms, ovs = optimize_overlap_sweep_forward_order(M1, M2_interms; truncation = truncation, max_bd = max_bd, max_err = max_err,verbose=verbose)
        println("Max bond dim: ", max_bond_dim(M1_interms[end]))
        
        push!(ov_opts, ovs...)

        if k==nsweep
            println("Final norm: ", MPDO_norm(M1_interms[end]))
        end
    end
    return ov_opts
end  
    

function optimize_overlap_real_nfloor_sweep(M1::myMPDO,M2::myMPDO,Us::Vector{Vector{Matrix}},nsweep::Int = 1;truncation = true, max_bd = 1024, max_err=1E-10,verbose=1)

    # M2 is at the top, and M1 is at the bottum
    M1 = canonicalize_right(M1)
    M2 = canonicalize_right(M2)

    ov_opts = Float64[]
    nfloor = size(Us)[1]

    M1_interms = Vector{Vector{myMPDO}}(undef, nfloor)
    M2_interms = Vector{Vector{myMPDO}}(undef, nfloor)
    M1_mid = Vector{myMPDO}(undef, nfloor+1)
    M2_mid = Vector{myMPDO}(undef, nfloor+1)
    
    M1_mid[nfloor+1] = M1
    M2_mid[1] = M2

    for iflr in 1:nfloor
        ind = nfloor + 1 - iflr
        M1_interms_this = unitary_evolution_two_floor_ancilla(M1_mid[ind+1], Us[ind]; truncation = truncation, max_bd = max_bd, max_err = max_err)
        M1_interms[ind] = M1_interms_this
        M1_mid[ind] = copy(M1_interms_this[end])
    end
    

    for k in 1:nsweep
        println("Sweep: $k")

        for iflr in 1:nfloor
            M2_interms_this, ovs = optimize_overlap_sweep_reverse_order(M1_interms[iflr], M2_mid[iflr]; truncation = truncation, max_bd = max_bd, max_err = max_err,verbose=verbose)
            M2_interms[iflr] = M2_interms_this
            push!(ov_opts, ovs...)
            M2_mid[iflr+1] = copy(M2_interms_this[end])

            if iflr == nfloor
                println("Max bond dim: ", max_bond_dim(M2_interms_this[end]))
            end
        end

        for iflr in 1:nfloor
            ind = nfloor + 1 - iflr
            M1_interms_this, ovs = optimize_overlap_sweep_forward_order(M1_mid[ind+1], M2_interms[ind]; truncation = truncation, max_bd = max_bd, max_err = max_err,verbose=verbose)
            M1_interms[ind] = M1_interms_this
            push!(ov_opts, ovs...)
            M1_mid[ind] = copy(M1_interms_this[end])

            if iflr == nfloor
                println("Max bond dim: ", max_bond_dim(M1_interms_this[end]))
            end
        end

        if k==nsweep
            println("Final norm: ", MPDO_norm(M1_interms[1][end]))
        end
    end
    return ov_opts
end

