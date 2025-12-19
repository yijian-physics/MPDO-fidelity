import h5py
import quimb as qu
import quimb.tensor as qtn
import numpy as np
import copy
import torch
from torch import optim
import tqdm
import cotengra as ctg

from file_io import *

# 1. Setup the optimizer
opti = ctg.ReusableHyperOptimizer(
    progbar=True,
    methods=['greedy'],
    reconf_opts={},
    max_repeats=32, 
    optlib='random',
)

# 2. Helper Functions
def read_data(data_name):
    with h5py.File("save_results/" + data_name + ".h5", "r") as f:
        keys = sorted(f.keys(), key=lambda x: int(x.split("_")[1]))
        data = [np.transpose(f[key][:], (3,2,1,0)) for key in keys]
    return data

def array_to_lpdo(M1, tags):
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
            inds = (f'l{i-1}', f's{i}', f'e{i}', f'l{i}')
            current_tensor = qtn.Tensor(data=M1[i], inds=inds, tags=tags)
        lpdo_1 = lpdo_1 & current_tensor
    return lpdo_1

def add_ancilla(lpdo, label):
    lpdo_acl = lpdo.copy()
    for i in range(len(lpdo.tensors)):
        prod = qtn.Tensor(np.array([1,0]), inds=(label+f'{i}',), tags='A')
        lpdo_acl = lpdo_acl & prod
    return lpdo_acl

def brickwall_unitary(psi, n_apply, list_u3, depth, n_Qbit, val_iden=0, rand=False, start_layer=0, is_acl=0, pbc=True):
    """
    Brickwall unitary with optional periodic boundary conditions.
    
    Args:
        pbc: If True, add wraparound gates between first and last qubits
    """
    if n_Qbit==0 or n_Qbit==1: depth=1
    for r in range(depth):
        if (r+start_layer)%2==0:
            if is_acl==0:
                # Standard even layer gates
                for i in range(0, n_Qbit-1, 2):
                    G = qu.rand_uni(4, dtype=complex) if rand else qu.identity(4,dtype=complex)+qu.rand_uni(4, dtype=complex)*val_iden
                    psi.gate_(G, (i, i + 1), tags={'U',f'G{n_apply}', f'L{i}D{r}'})
                    list_u3.append(f'G{n_apply}'); n_apply+=1
                
                # PBC wraparound gate if n_Qbit is even and last qubit not covered
                if pbc and n_Qbit % 2 == 0 and n_Qbit >= 2:
                    G = qu.rand_uni(4, dtype=complex) if rand else qu.identity(4,dtype=complex)+qu.rand_uni(4, dtype=complex)*val_iden
                    psi.gate_(G, (n_Qbit-1, 0), tags={'U',f'G{n_apply}', f'PBC_D{r}'})
                    list_u3.append(f'G{n_apply}'); n_apply+=1
                    
            elif is_acl==1:
                # Ancilla case: 4-qubit gates on even layer
                for i in range(0, n_Qbit-3, 4):
                    G = qu.rand_uni(16, dtype=complex) if rand else qu.identity(16,dtype=complex)+qu.rand_uni(16, dtype=complex)*val_iden
                    psi.gate_(G, (i, i+1, i+2, i+3), tags={'U',f'G{n_apply}', f'L{i}D{r}'})
                    list_u3.append(f'G{n_apply}'); n_apply+=1
                
                # PBC wraparound gate for ancilla if needed
                if pbc and n_Qbit % 4 == 0 and n_Qbit >= 4:
                    G = qu.rand_uni(16, dtype=complex) if rand else qu.identity(16,dtype=complex)+qu.rand_uni(16, dtype=complex)*val_iden
                    psi.gate_(G, (n_Qbit-3, n_Qbit-2, n_Qbit-1, 0), tags={'U',f'G{n_apply}', f'PBC_D{r}'})
                    list_u3.append(f'G{n_apply}'); n_apply+=1
        else:
            if is_acl==0:
                # Standard odd layer gates
                for i in range(0, n_Qbit-2, 2):
                    G = qu.rand_uni(4, dtype=complex) if rand else qu.identity(4,dtype=complex)+qu.rand_uni(4, dtype=complex)*val_iden
                    psi.gate_(G, (i+1, i+2), tags={'U',f'G{n_apply}', f'L{i}D{r}'})
                    list_u3.append(f'G{n_apply}'); n_apply+=1
                
                # PBC wraparound gate if n_Qbit is odd or to complete coverage
                if pbc and n_Qbit >= 2:
                    G = qu.rand_uni(4, dtype=complex) if rand else qu.identity(4,dtype=complex)+qu.rand_uni(4, dtype=complex)*val_iden
                    psi.gate_(G, (n_Qbit-1, 0), tags={'U',f'G{n_apply}', f'PBC_D{r}'})
                    list_u3.append(f'G{n_apply}'); n_apply+=1
                    
            elif is_acl==1:
                # Ancilla case: 4-qubit gates on odd layer
                for i in range(2, n_Qbit-2, 4):
                    G = qu.rand_uni(16, dtype=complex) if rand else qu.identity(16,dtype=complex)+qu.rand_uni(16, dtype=complex)*val_iden
                    psi.gate_(G, (i, i+1, i+2, i+3), tags={'U',f'G{n_apply}', f'L{i}D{r}'})
                    list_u3.append(f'G{n_apply}'); n_apply+=1
                
                # PBC wraparound for ancilla odd layer
                if pbc and n_Qbit >= 4:
                    G = qu.rand_uni(16, dtype=complex) if rand else qu.identity(16,dtype=complex)+qu.rand_uni(16, dtype=complex)*val_iden
                    psi.gate_(G, (n_Qbit-2, n_Qbit-1, 0, 1), tags={'U',f'G{n_apply}', f'PBC_D{r}'})
                    list_u3.append(f'G{n_apply}'); n_apply+=1
    return n_apply, list_u3

def staircase_unitary(psi, n_apply, list_u3, depth, n_Qbit, val_iden=0, rand=False, start_layer=0, is_acl=0, pbc=True, icrm=2, icrm_bdy=2):
    """
    Staircase unitary with optional periodic boundary conditions.
    
    Args:
        pbc: If True, add wraparound gates between first and last qubits
        icrm: increment. For is_acl=1, can be 1 or 2
    """
    if n_Qbit==0 or n_Qbit==1: depth=1
    for r in range(depth):
        if (r+start_layer)%2==0:
            if is_acl == 0:
                # Forward sweep
                for i in range(0, n_Qbit-1, 1):
                    G = qu.rand_uni(4, dtype=complex) if rand else qu.identity(4,dtype=complex)+qu.rand_uni(4, dtype=complex)*val_iden
                    psi.gate_(G, (i, i+1), tags={'U',f'G{n_apply}',f'L{i}D{r}'})
                    list_u3.append(f'G{n_apply}'); n_apply+=1
                
                # PBC wraparound gate
                if pbc and n_Qbit >= 2:
                    G = qu.rand_uni(4, dtype=complex) if rand else qu.identity(4,dtype=complex)+qu.rand_uni(4, dtype=complex)*val_iden
                    psi.gate_(G, (n_Qbit-1, 0), tags={'U',f'G{n_apply}',f'PBC_D{r}'})
                    list_u3.append(f'G{n_apply}'); n_apply+=1
                    
            elif is_acl == 1:
                # Forward sweep with 4-qubit gates
                for i in range(0, n_Qbit-3, icrm): 
                    G = qu.rand_uni(16, dtype=complex) if rand else qu.identity(16,dtype=complex)+qu.rand_uni(16, dtype=complex)*val_iden
                    psi.gate_(G, (i, i+1, i+2, i+3), tags={'U',f'G{n_apply}',f'L{i}D{r}'})
                    list_u3.append(f'G{n_apply}'); n_apply+=1
                
                # PBC wraparound for ancilla
                if pbc and n_Qbit >= 4:
                    # Wraparound connecting last few qubits to first few
                    if icrm_bdy == 1:
                        G = qu.rand_uni(16, dtype=complex) if rand else qu.identity(16,dtype=complex)+qu.rand_uni(16, dtype=complex)*val_iden
                        psi.gate_(G, (n_Qbit-3, n_Qbit-2, n_Qbit-1, 0), tags={'U',f'G{n_apply}',f'PBC_D{r}'})
                        list_u3.append(f'G{n_apply}'); n_apply+=1

                    G = qu.rand_uni(16, dtype=complex) if rand else qu.identity(16,dtype=complex)+qu.rand_uni(16, dtype=complex)*val_iden
                    psi.gate_(G, (n_Qbit-2, n_Qbit-1, 0, 1), tags={'U',f'G{n_apply}',f'PBC_D{r}'})
                    list_u3.append(f'G{n_apply}'); n_apply+=1

                    if icrm_bdy == 1:
                        G = qu.rand_uni(16, dtype=complex) if rand else qu.identity(16,dtype=complex)+qu.rand_uni(16, dtype=complex)*val_iden
                        psi.gate_(G, (n_Qbit-1, 0, 1, 2), tags={'U',f'G{n_apply}',f'PBC_D{r}'})
                        list_u3.append(f'G{n_apply}'); n_apply+=1

                        G = qu.rand_uni(16, dtype=complex) if rand else qu.identity(16,dtype=complex)+qu.rand_uni(16, dtype=complex)*val_iden
                        psi.gate_(G, (n_Qbit-2, n_Qbit-1, 0, 1), tags={'U',f'G{n_apply}',f'PBC_D{r}'})
                        list_u3.append(f'G{n_apply}'); n_apply+=1

                        G = qu.rand_uni(16, dtype=complex) if rand else qu.identity(16,dtype=complex)+qu.rand_uni(16, dtype=complex)*val_iden
                        psi.gate_(G, (n_Qbit-3, n_Qbit-2, n_Qbit-1, 0), tags={'U',f'G{n_apply}',f'PBC_D{r}'})
                        list_u3.append(f'G{n_apply}'); n_apply+=1

        else:
            if is_acl == 0:
                # Backward sweep
                for i in range(n_Qbit-1, 0, -1):
                    G = qu.rand_uni(4, dtype=complex) if rand else qu.identity(4,dtype=complex)+qu.rand_uni(4, dtype=complex)*val_iden
                    psi.gate_(G, (i-1, i), tags={'U',f'G{n_apply}',f'L{i}D{r}'})
                    list_u3.append(f'G{n_apply}'); n_apply+=1
                
                # PBC wraparound gate
                if pbc and n_Qbit >= 2:
                    G = qu.rand_uni(4, dtype=complex) if rand else qu.identity(4,dtype=complex)+qu.rand_uni(4, dtype=complex)*val_iden
                    psi.gate_(G, (n_Qbit-1, 0), tags={'U',f'G{n_apply}',f'PBC_D{r}'})
                    list_u3.append(f'G{n_apply}'); n_apply+=1
                    
            elif is_acl == 1:
                # Backward sweep with 4-qubit gates
                for i in range(n_Qbit-1, 2, -icrm): 
                    G = qu.rand_uni(16, dtype=complex) if rand else qu.identity(16,dtype=complex)+qu.rand_uni(16, dtype=complex)*val_iden
                    psi.gate_(G, (i-3, i-2, i-1, i), tags={'U',f'G{n_apply}',f'L{i}D{r}'})
                    list_u3.append(f'G{n_apply}'); n_apply+=1
                
                # PBC wraparound for ancilla backward
                if pbc and n_Qbit >= 4:

                    if icrm_bdy == 1:
                        G = qu.rand_uni(16, dtype=complex) if rand else qu.identity(16,dtype=complex)+qu.rand_uni(16, dtype=complex)*val_iden
                        psi.gate_(G, (n_Qbit-1, 0, 1, 2), tags={'U',f'G{n_apply}',f'PBC_D{r}'})
                        list_u3.append(f'G{n_apply}'); n_apply+=1

                    G = qu.rand_uni(16, dtype=complex) if rand else qu.identity(16,dtype=complex)+qu.rand_uni(16, dtype=complex)*val_iden
                    psi.gate_(G, (n_Qbit-2, n_Qbit-1, 0, 1), tags={'U',f'G{n_apply}',f'PBC_D{r}'})
                    list_u3.append(f'G{n_apply}'); n_apply+=1

                    if icrm_bdy == 1:
                        G = qu.rand_uni(16, dtype=complex) if rand else qu.identity(16,dtype=complex)+qu.rand_uni(16, dtype=complex)*val_iden
                        psi.gate_(G, (n_Qbit-3, n_Qbit-2, n_Qbit-1, 0), tags={'U',f'G{n_apply}',f'PBC_D{r}'})
                        list_u3.append(f'G{n_apply}'); n_apply+=1

                        G = qu.rand_uni(16, dtype=complex) if rand else qu.identity(16,dtype=complex)+qu.rand_uni(16, dtype=complex)*val_iden
                        psi.gate_(G, (n_Qbit-2, n_Qbit-1, 0, 1), tags={'U',f'G{n_apply}',f'PBC_D{r}'})
                        list_u3.append(f'G{n_apply}'); n_apply+=1

                        G = qu.rand_uni(16, dtype=complex) if rand else qu.identity(16,dtype=complex)+qu.rand_uni(16, dtype=complex)*val_iden
                        psi.gate_(G, (n_Qbit-1, 0, 1, 2), tags={'U',f'G{n_apply}',f'PBC_D{r}'})
                        list_u3.append(f'G{n_apply}'); n_apply+=1

    return n_apply, list_u3

def qmps_f(L=16, in_depth=2, val_iden=0, rand=True, start_layer=0, framework='staircase', is_acl=0, pbc=True, icrm=2, icrm_bdy=2):
    """
    Create a parameterized quantum circuit.
    
    Args:
        L: Number of qubits
        in_depth: Circuit depth
        val_iden: Perturbation from identity
        rand: Use random unitaries
        start_layer: Starting layer offset
        framework: 'brickwall' or 'staircase'
        is_acl: 0 for no ancilla (2-qubit gates), 1 for ancilla (4-qubit gates)
        pbc: If True, add periodic boundary condition wraparound gates
    """
    list_u3, n_apply = [], 0
    psi = qtn.MPS_computational_state('0' * L)
    for i in range(L):
        psi[i].modify(left_inds=['k'+str(i)], tags=[f"I{i}", "MPS"])
    if framework == 'brickwall':
        n_apply, list_u3 = brickwall_unitary(psi, n_apply, list_u3, in_depth, L, 
                                             val_iden=val_iden, rand=rand, 
                                             start_layer=start_layer, is_acl=is_acl, pbc=pbc)
    elif framework == 'staircase':
        n_apply, list_u3 = staircase_unitary(psi, n_apply, list_u3, in_depth, L, 
                                             val_iden=val_iden, rand=rand, 
                                             start_layer=start_layer, is_acl=is_acl, pbc=pbc, icrm=icrm, icrm_bdy=icrm_bdy)
    return psi.astype_('complex128')

def extract_unitary_circuit_acl(psi_pqc, num_qubits):
    pqc = psi_pqc.tensors[num_qubits]
    for i in range(num_qubits+1, len(psi_pqc.tensors)):
        pqc = pqc & psi_pqc.tensors[i]
    for i in range(num_qubits):
        suffix = 'e' if i%2 else 'a'
        pqc = pqc.reindex({f'k{i}': f'{suffix}{i//2}', psi_pqc.tensors[i].inds[-1]: f'{suffix}p{i//2}'})
    return pqc

def full_contraction(pqc, lpdo_1, lpdo_2):
    return -abs((lpdo_1 & lpdo_2 & pqc).contract(optimize=opti))

# 3. Non-Redundant and Picklable Model
class TNModel(torch.nn.Module):
    def __init__(self, pqc, lpdo_1, lpdo_2):
        super().__init__()
        self.lpdo_1, self.lpdo_2 = lpdo_1, lpdo_2
        params, self.skeleton = qtn.pack(pqc)
        self.torch_params, self.param_metadata = torch.nn.ParameterDict(), {}
        for i, initial in params.items():
            d = int(np.sqrt(initial.numel()))
            self.torch_params[str(i)] = torch.nn.Parameter(torch.randn(d**2, dtype=torch.float64)*0.05)
            self.param_metadata[str(i)] = (d, initial.shape)

    def _get_complex_generators(self):
        complex_params = {}
        for i, p in self.torch_params.items():
            d, shape = self.param_metadata[i]
            M = torch.zeros((d, d), dtype=torch.complex128, device=p.device)
            M.diagonal().copy_(1j * p[:d].to(torch.complex128))
            if d > 1:
                idx_u = torch.triu_indices(d, d, offset=1)
                num_off = d * (d - 1) // 2
                M[idx_u[0], idx_u[1]] = (p[d : d + num_off] + 1j * p[d + num_off :]).to(torch.complex128)
            A = M - M.conj().transpose(0, 1)
            complex_params[int(i)] = A.reshape(shape)
        return complex_params

    def forward(self):
        pqc = qtn.unpack(self._get_complex_generators(), self.skeleton)
        return full_contraction(pqc.isometrize(method='exp'), self.lpdo_1, self.lpdo_2)

# 4. Optimization Functions
def run_single_optimization(model, num_steps=1000, lr=0.01):
    """Original single-start optimization logic."""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    pbar = tqdm.tqdm(range(num_steps))
    losses = []
    for step in pbar:
        optimizer.zero_grad()
        loss = model.forward()
        losses.append(loss.detach().numpy())
        loss.backward()
        optimizer.step()
        pbar.set_description(f"Loss={loss.item():.8f}")

    return min(losses)


# 5. PROTECTED MAIN BLOCK
if __name__ == "__main__":

    # Parameters
    n = 12
    sample = 2
    lr = 0.01
    num_steps = 2000
    depth = 2
    framework = 'staircase'
    icrm = 2
    icrm_bdy = 2

    Dir = File_access()
    loss_save = np.zeros(sample)
    save_name = "pbcN"+str(n)+"lr"+str(lr)+"num_steps"+str(num_steps)+"sample"+str(sample)+framework+"depth"+str(depth)+"icrm"+str(icrm)+"icrm_bdy"+str(icrm_bdy)

    for i_sample in range(sample):
        # Load Data
        file1 = "M1_a2_N"+str(n)
        file2 = "M2_a2_N"+str(n)
        M1, M2 = read_data(file1), read_data(file2)
        
        # Pre-calculate LPDDs (once)
        l1, l2 = array_to_lpdo(M1, ('M1',)), array_to_lpdo(M2, ('M2',)).H
        for i in range(n): l2 = l2.reindex({f'e{i}': f'ep{i}'})
        l1_acl, l2_acl = add_ancilla(l1, "a"), add_ancilla(l2, "ap")
        for obj in [l1_acl, l2_acl]: 
            obj.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.complex128))

        # Obtain Unitary Circuit
        
        psi_pqc = qmps_f(2*n, depth, framework=framework, is_acl=1, pbc=True, icrm=icrm, icrm_bdy=icrm_bdy)
        pqc_init = extract_unitary_circuit_acl(psi_pqc, 2*n)
        pqc_init.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.complex128))
        model = TNModel(pqc_init, l1_acl, l2_acl)
        single_loss = run_single_optimization(model,num_steps=num_steps,lr=lr)
        loss_save[i_sample] = single_loss
        Dir.save_data(loss_save, save_name)

    test = Dir.get_back(save_name)
    print(test)
