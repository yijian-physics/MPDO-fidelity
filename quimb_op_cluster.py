import h5py
import quimb as qu
import quimb.tensor as qtn
import numpy as np
import copy

import torch
from torch import optim
import tqdm
import cotengra as ctg

opti = ctg.ReusableHyperOptimizer(
    progbar=True,
    methods=['greedy'],
    reconf_opts={},
    max_repeats=32, 
    optlib='random',
    # directory=  # set this for persistent cache
)

### read data

def read_data(data_name):
    with h5py.File("../save_results/" + data_name + ".h5", "r") as f:
        keys = sorted(f.keys(), key=lambda x: int(x.split("_")[1]))
        # print(keys)
        data = [np.transpose(f[key][:], (3,2,1,0)) for key in keys]

    return data

def array_to_lpdo(M1, tags):
    # convert input list of arrays to LPDO

    L = len(M1)

    inds = ('s0','e0','l0')
    first_tensor = M1[0][0,:,:,:]
    last_tensor = M1[-1][:,:,:,0]
    lpdo_1 = qtn.Tensor(data=first_tensor, inds=inds, tags=tags)

    for i in range(1, L):
        if i == L-1:
            inds = (f'l{i-1}', f's{i}', f'e{i}')
            current_tensor = qtn.Tensor(data=last_tensor, inds=inds, tags=tags)
        else:
            # bond, system, environment, bond
            inds = (f'l{i-1}', f's{i}', f'e{i}', f'l{i}')
            current_tensor = qtn.Tensor(data=M1[i], inds=inds, tags=tags)

        lpdo_1 = lpdo_1 & current_tensor

    return lpdo_1


def add_ancilla(lpdo, label):
    
    lpdo_acl = lpdo.copy()

    for i in range(len(lpdo.tensors)):
        prod = qtn.Tensor(np.array([1,0]), inds = (label+f'{i}',),tags = 'A')
        # prod.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.complex128))
        
        # like direct product (outer product)
        lpdo_acl = lpdo_acl & prod

    return lpdo_acl


### build unitary circuit

def brickwall_unitary(psi, n_apply, list_u3, depth, n_Qbit, val_iden = 0,rand = False,start_layer=0,is_acl=0, pbc=True):

    if n_Qbit==0 or n_Qbit==1: depth=1

    for r in range(depth):

        if (r+start_layer)%2==0:
            if is_acl==0:
                for i in range(0, n_Qbit-1, 2):
                    # print("U_e", i, i + 1, n_apply)

                    if rand == True:
                        G = qu.rand_uni(4, dtype=complex)
                        #G = qu.fsimg(1,1,1,1,1, dtype=complex)
                    else:
                        G = qu.identity(4,dtype=complex)+qu.rand_uni(4, dtype=complex)*val_iden
                
                    psi.gate_(G, (i, i + 1), tags={'U',f'G{n_apply}', f'L{i}D{r}'})
                    list_u3.append(f'G{n_apply}')
                    n_apply+=1
            
                # PBC wraparound gate if n_Qbit is even and last qubit not covered
                if pbc and n_Qbit % 2 == 0 and n_Qbit >= 2:
                    G = qu.rand_uni(4, dtype=complex) if rand else qu.identity(4,dtype=complex)+qu.rand_uni(4, dtype=complex)*val_iden
                    psi.gate_(G, (n_Qbit-1, 0), tags={'U',f'G{n_apply}', f'PBC_D{r}'})
                    list_u3.append(f'G{n_apply}'); n_apply+=1

            elif is_acl==1:
                for i in range(0, n_Qbit, 4):

                    if rand == True:
                        G = qu.rand_uni(16, dtype=complex)
                    else:
                        G = qu.identity(16,dtype=complex)+qu.rand_uni(16, dtype=complex)*val_iden
                
                    psi.gate_(G, (i, i + 1,i+2,i+3), tags={'U',f'G{n_apply}', f'L{i}D{r}'})
                    list_u3.append(f'G{n_apply}')
                    n_apply+=1

                if pbc and n_Qbit % 4 == 0 and n_Qbit >= 4:
                    G = qu.rand_uni(16, dtype=complex) if rand else qu.identity(16,dtype=complex)+qu.rand_uni(16, dtype=complex)*val_iden
                    psi.gate_(G, (n_Qbit-3, n_Qbit-2, n_Qbit-1, 0), tags={'U',f'G{n_apply}', f'PBC_D{r}'})
                    list_u3.append(f'G{n_apply}'); n_apply+=1

        else:
            if is_acl==0:
                for i in range(0, n_Qbit-2, 2):
                    # print("U_o", i+1, i + 2, n_apply)
            
                    if rand == True:
                        G = qu.rand_uni(4, dtype=complex)
                        #G = qu.fsimg(1,1,1,1,1, dtype=complex)
                    else:
                        G = qu.identity(4,dtype=complex)+qu.rand_uni(4, dtype=complex)*val_iden

                    psi.gate_(G, (i+1, i + 2), tags={'U',f'G{n_apply}', f'L{i}D{r}'})
                    list_u3.append(f'G{n_apply}')
                    n_apply+=1

            elif is_acl == 1:
                for i in range(2, n_Qbit-2, 4):
                    # print("U_o", i, i + 1,i+2,i+3, n_apply)
            
                    if rand == True:
                        G = qu.rand_uni(16, dtype=complex)
                        #G = qu.fsimg(1,1,1,1,1, dtype=complex)
                    else:
                        G = qu.identity(16,dtype=complex)+qu.rand_uni(16, dtype=complex)*val_iden

                    psi.gate_(G, (i, i + 1,i+2,i+3), tags={'U',f'G{n_apply}', f'L{i}D{r}'})
                    list_u3.append(f'G{n_apply}')
                    n_apply+=1

    return n_apply, list_u3


def staircase_unitary(psi, n_apply, list_u3, depth, n_Qbit, val_iden = 0,rand = False,start_layer=0, is_acl=0):

    if n_Qbit==0: depth=1
    if n_Qbit==1: depth=1

    for r in range(depth):

        if (r+start_layer)%2==0:
            if is_acl == 0:
                for i in range(0, n_Qbit-1, 1):
                    # print("U_e", i, i + 1, n_apply)

                    if rand == True:
                        G = qu.rand_uni(4, dtype=complex)
                        #G = qu.fsimg(1,1,1,1,1, dtype=complex)
                    else:
                        G = qu.identity(4,dtype=complex)+qu.rand_uni(4, dtype=complex)*val_iden
                
                    psi.gate_(G, (i, i + 1), tags={'U',f'G{n_apply}',f'L{i}D{r}'})
                    list_u3.append(f'G{n_apply}')
                    n_apply+=1

            elif is_acl == 1:
                # acts on four sites, two system, two ancilla
                for i in range(0, n_Qbit-3, 2):
                    # print("U_e", i, i + 1, n_apply)

                    if rand == True:
                        G = qu.rand_uni(16, dtype=complex)
            
                    else:
                        G = qu.identity(16,dtype=complex)+qu.rand_uni(16, dtype=complex)*val_iden
                
                    psi.gate_(G, (i, i + 1, i+2, i+3), tags={'U',f'G{n_apply}',f'L{i}D{r}'})
                    list_u3.append(f'G{n_apply}')
                    n_apply+=1

        else:
            if is_acl == 0:
                for i in range(n_Qbit-1, 0, -1):
                    # print("U_o", i-1, i, n_apply)
            
                    if rand == True:
                        G = qu.rand_uni(4, dtype=complex)
                        #G = qu.fsimg(1,1,1,1,1, dtype=complex)
                    else:
                        G = qu.identity(4,dtype=complex)+qu.rand_uni(4, dtype=complex)*val_iden

                    psi.gate_(G, (i-1, i), tags={'U',f'G{n_apply}',f'L{i}D{r}'})
                    list_u3.append(f'G{n_apply}')
                    n_apply+=1

            elif is_acl == 1:
                for i in range(n_Qbit-1, 2, -2):
                    # print("U_o", i-1, i, n_apply)
            
                    if rand == True:
                        G = qu.rand_uni(16, dtype=complex)
                        #G = qu.fsimg(1,1,1,1,1, dtype=complex)
                    else:
                        G = qu.identity(16,dtype=complex)+qu.rand_uni(16, dtype=complex)*val_iden

                    psi.gate_(G, (i-3, i-2, i-1, i), tags={'U',f'G{n_apply}',f'L{i}D{r}'})
                    list_u3.append(f'G{n_apply}')
                    n_apply+=1

    return n_apply, list_u3


def qmps_f(L=16, in_depth=2, val_iden = 0, rand = True,start_layer = 0, framework='brickwall', is_acl=0):

    list_u3=[]
    n_apply=0
    psi = qtn.MPS_computational_state('0' * (L))
    for i in range(L):
        t = psi[i]
        indx = 'k'+str(i)
        t.modify(left_inds=[indx])

    for t in range(L):
        psi[t].modify(tags=[f"I{t}", "MPS"])

    if framework == 'brickwall':
        n_apply, list_u3=brickwall_unitary(psi, n_apply, list_u3, in_depth, L, val_iden = val_iden, rand =rand,start_layer=start_layer,is_acl=is_acl)
    elif framework == 'staircase':
        n_apply, list_u3=staircase_unitary(psi, n_apply, list_u3, in_depth, L, val_iden = val_iden, rand =rand,start_layer=start_layer,is_acl=is_acl)
    elif framework == 'mixed':
        n_apply, list_u3=brickwall_unitary(psi, n_apply, list_u3, in_depth, L, val_iden = val_iden, rand =rand,start_layer=start_layer,is_acl=is_acl)
        n_apply, list_u3=staircase_unitary(psi, n_apply, list_u3, in_depth, L, val_iden = val_iden, rand =rand,start_layer=start_layer,is_acl=is_acl)


    return psi.astype_('complex128')#, list_u3


def extract_unitary_circuit(psi_pqc, num_qubits, is_td=0):
    # only system qubits
    # is_td: is trace distance

    pqc = psi_pqc.tensors[num_qubits]
    for i in range (num_qubits+1,len(psi_pqc.tensors)):
        pqc = pqc&psi_pqc.tensors[i] #extrating the circuit part

    if is_td == 0:
        for i in range (num_qubits):
            pqc = pqc.reindex({f'k{i}':f'e{i}'})
            pqc = pqc.reindex({psi_pqc.tensors[i].inds[-1]:f'ep{i}'})
    
    elif is_td == 1:
        for i in range (num_qubits):
            pqc = pqc.reindex({f'k{i}':f's{i}'})
            pqc = pqc.reindex({psi_pqc.tensors[i].inds[-1]:f'sp{i}'})

    return pqc


def extract_unitary_circuit_acl(psi_pqc, num_qubits, is_td=0):
    # for ancilla. num_qubits = 2*n
    # is_td: is trace distance

    pqc = psi_pqc.tensors[num_qubits]
    for i in range (num_qubits+1,len(psi_pqc.tensors)):
        pqc = pqc&psi_pqc.tensors[i] #extrating the circuit part

    for i in range (num_qubits):
        if (i%2):
            if is_td == 0:
                pqc = pqc.reindex({f'k{i}':f'e{i//2}'})
                pqc = pqc.reindex({psi_pqc.tensors[i].inds[-1]:f'ep{i//2}'})
            elif is_td == 1:
                pqc = pqc.reindex({f'k{i}':f's{i//2}'})
                pqc = pqc.reindex({psi_pqc.tensors[i].inds[-1]:f'sp{i//2}'})
        else:
            pqc = pqc.reindex({f'k{i}':f'a{i//2}'})
            pqc = pqc.reindex({psi_pqc.tensors[i].inds[-1]:f'ap{i//2}'})

    return pqc


def full_contraction(pqc, lpdo_1, lpdo_2, is_show=0):
    if is_show == 1:
        (lpdo_1 & lpdo_2 & pqc).draw(['U','M2','M1'])

    output = abs((lpdo_1 & lpdo_2 & pqc).contract(optimize=opti))
    
    # return torch.abs(output).real 
    return -output



def full_contraction_td(pqc, lpdo_1, lpdo_2, lpdo_1_conj, lpdo_2_conj, is_show=0):
    # for trace distance

    if is_show == 1:
        (lpdo_1_conj & lpdo_1 & pqc).draw(['U','M2','M1'])

    ov1 = (lpdo_1_conj & lpdo_1 & pqc).contract(optimize=opti)
    ov2 = (lpdo_2_conj & lpdo_2 & pqc).contract(optimize=opti)
    
    dist = (1/2)*(torch.abs(ov1-ov2))
    return -dist


class TNModel(torch.nn.Module):
    # this class is inheritance of torch.nn.Module

    def __init__(self, pqc, lpdo_1, lpdo_2):
        super().__init__()

        # extract the raw arrays and a skeleton of the TN
        params, self.skeleton = qtn.pack(pqc)
        # n.b. you might want to do extra processing here to e.g. store each
        # parameter as a reshaped matrix (from left_inds -> right_inds), for
        # some optimizers, and for some torch parametrizations

        self.torch_params = torch.nn.ParameterDict({
            # "str(i)" is key conversion: torch requires strings as keys
            str(i): torch.nn.Parameter(initial)
            for i, initial in params.items()
        })
        self._loss_fn = lambda x: full_contraction(x, lpdo_1, lpdo_2)
        
    def forward(self):
        # convert back to original int key format
        params = {int(i): p for i, p in self.torch_params.items()}
        # reconstruct the TN with the new parameters
        pqc = qtn.unpack(params, self.skeleton)
        
        # isometrize and then return the energy
        return self._loss_fn(pqc.isometrize(method='exp'))
    
    def forward_debug(self):
        # convert back to original int key format
        params = {int(i): p for i, p in self.torch_params.items()}
        # reconstruct the TN with the new parameters
        pqc = qtn.unpack(params, self.skeleton)
        
        # isometrize and then return the energy
        return self._loss_fn(pqc)
    
    def get_unitary_circuit(self):
        """Extract the optimized unitary quantum circuit."""
        # Convert parameters back to int keys
        params = {int(i): p for i, p in self.torch_params.items()}
        # Reconstruct the circuit
        pqc = qtn.unpack(params, self.skeleton)
        # Apply isometrization to get unitary gates
        return pqc.isometrize(method='exp')
    

class TNModel_td(torch.nn.Module):
    # this class is inheritance of torch.nn.Module

    def __init__(self, pqc, lpdo_1, lpdo_2, is_acl,n):
        super().__init__()

        pqc_torch = pqc.copy()
        pqc_torch.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.complex128))
        # extract the raw arrays and a skeleton of the TN
        params, self.skeleton = qtn.pack(pqc_torch)
        # n.b. you might want to do extra processing here to e.g. store each
        # parameter as a reshaped matrix (from left_inds -> right_inds), for
        # some optimizers, and for some torch parametrizations

        lpdo_1_conj = lpdo_1.H
        lpdo_2_conj = lpdo_2.H

        if is_acl == 0:
            for i in range(n):
                lpdo_1_conj = lpdo_1_conj.reindex({f's{i}':f'sp{i}'})
                lpdo_2_conj = lpdo_2_conj.reindex({f's{i}':f'sp{i}'})

        elif is_acl == 1:
            for i in range(n):
                lpdo_1_conj = lpdo_1_conj.reindex({f's{i}':f'sp{i}'})
                lpdo_2_conj = lpdo_2_conj.reindex({f's{i}':f'sp{i}'})
                lpdo_1_conj = lpdo_1_conj.reindex({f'a{i}':f'ap{i}'})
                lpdo_2_conj = lpdo_2_conj.reindex({f'a{i}':f'ap{i}'})
                    

        lpdo_1_torch = lpdo_1.copy()
        lpdo_2_torch = lpdo_2.copy()
        lpdo_1_torch.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.complex128))
        lpdo_2_torch.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.complex128))
        lpdo_1_conj.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.complex128))
        lpdo_2_conj.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.complex128))

        self.torch_params = torch.nn.ParameterDict({
            # "str(i)" is key conversion: torch requires strings as keys
            str(i): torch.nn.Parameter(initial)
            for i, initial in params.items()
        })
        self._loss_fn = lambda x: full_contraction_td(x, lpdo_1_torch, lpdo_2_torch, lpdo_1_conj, lpdo_2_conj)
        
    def forward(self):
        # convert back to original int key format
        params = {int(i): p for i, p in self.torch_params.items()}
        # reconstruct the TN with the new parameters
        pqc = qtn.unpack(params, self.skeleton)
        
        # isometrize and then return the energy
        return self._loss_fn(pqc.isometrize(method='exp'))
    
    def get_unitary_circuit(self):
        """Extract the optimized unitary quantum circuit."""
        # Convert parameters back to int keys
        params = {int(i): p for i, p in self.torch_params.items()}
        # Reconstruct the circuit
        pqc = qtn.unpack(params, self.skeleton)
        # Apply isometrization to get unitary gates
        return pqc.isometrize(method='exp')


def get_pqc_torch(n, depth, framework, is_acl, rand=True, val_iden=0.0, is_td=0):

    if is_acl == 0:
        num_qubits = n # physical 
        psi_pqc = qmps_f(num_qubits, in_depth= depth, val_iden = val_iden,rand = rand, framework=framework)
        pqc = extract_unitary_circuit(psi_pqc, num_qubits, is_td=is_td)

    elif is_acl == 1:
        num_qubits = 2 * n # physical + ancilla 
        psi_pqc = qmps_f(num_qubits, in_depth= depth, val_iden = val_iden, rand = rand, framework=framework,is_acl=1)
        pqc = extract_unitary_circuit_acl(psi_pqc, num_qubits, is_td=is_td)

    return pqc


def get_lpdo_torch(M1, M2, is_acl):

    n = len(M1)
    lpdo_1 = array_to_lpdo(M1, ('M1',))

    lpdo_2 = array_to_lpdo(M2, ('M2',))
    lpdo_2 = lpdo_2.H
    for i in range (n):
        lpdo_2 = lpdo_2.reindex({f'e{i}':f'ep{i}'})

    lpdo_1_acl = add_ancilla(lpdo_1,"a")
    lpdo_2_acl = add_ancilla(lpdo_2,"ap")

    if is_acl == 0:
        lpdo_1_torch = lpdo_1.copy()
        lpdo_2_torch = lpdo_2.copy()
    elif is_acl == 1:
        lpdo_1_torch = lpdo_1_acl.copy()
        lpdo_2_torch = lpdo_2_acl.copy()

    return lpdo_1_torch, lpdo_2_torch


def get_lpdo_torch_td(M1, M2, is_acl):

    n = len(M1)
    lpdo_1 = array_to_lpdo(M1, ('M1',))
    lpdo_2 = array_to_lpdo(M2, ('M2',))

    lpdo_1_acl = add_ancilla(lpdo_1,"a")
    lpdo_2_acl = add_ancilla(lpdo_2,"a")

    if is_acl == 0:
        lpdo_1_torch = lpdo_1.copy()
        lpdo_2_torch = lpdo_2.copy()
    elif is_acl == 1:
        lpdo_1_torch = lpdo_1_acl.copy()
        lpdo_2_torch = lpdo_2_acl.copy()

    return lpdo_1_torch, lpdo_2_torch