import h5py
import h5py
import quimb as qu
import quimb.tensor as qtn
import numpy as np
import copy
import torch
from torch import optim
import tqdm
import cotengra as ctg
import sys

from file_io import *


class ModelPara:
    
    def __init__(self,framework, depth, pbc, is_acl, is_td):
        self.framework = framework
        self.depth = depth
        self.pbc = pbc
        self.is_acl = is_acl
        self.is_td = is_td


class OptimizePara:
    
    def __init__(self,lr, num_steps, sample):
        self.lr = lr
        self.num_steps = num_steps
        self.sample = sample


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

def read_data_gapped(data_name):
    with h5py.File("gap_data/" + data_name + ".h5", "r") as f:
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


    


def get_lpdo_torch(M1, M2, model_para, is_td=0):

    n = len(M1)

    if is_td == 0:
        l1, l2 = array_to_lpdo(M1, ('M1',)), array_to_lpdo(M2, ('M2',)).H
        for i in range(n): l2 = l2.reindex({f'e{i}': f'ep{i}'})
        if model_para.is_acl == 1:
            l1_in, l2_in = add_ancilla(l1, "a"), add_ancilla(l2, "ap")
        else:
            l1_in, l2_in = l1, l2

    elif is_td == 1:
        l1, l2 = array_to_lpdo(M1, ('M1',)), array_to_lpdo(M2, ('M2',))
        if model_para.is_acl == 1:
            l1_in, l2_in = add_ancilla(l1, "a"), add_ancilla(l2, "a")
        else:
            l1_in, l2_in = l1, l2

    for obj in [l1_in, l2_in]: 
        obj.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.complex128))

    return l1_in, l2_in

##

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


def full_contraction(pqc, lpdo_1, lpdo_2):
    return -abs((lpdo_1 & lpdo_2 & pqc).contract(optimize=opti))



def full_contraction_td(pqc, lpdo_1, lpdo_2, is_acl, is_show=0):
    # for trace distance

    if is_show == 1:
        (lpdo_1_conj & lpdo_1 & pqc).draw(['U','M2','M1'])

    lpdo_1_conj = lpdo_1.H
    lpdo_2_conj = lpdo_2.H

    for i in range(n):
        lpdo_1_conj = lpdo_1_conj.reindex({f's{i}':f'sp{i}'})
        lpdo_2_conj = lpdo_2_conj.reindex({f's{i}':f'sp{i}'})
        if is_acl == 1:
            lpdo_1_conj = lpdo_1_conj.reindex({f'a{i}':f'ap{i}'})
            lpdo_2_conj = lpdo_2_conj.reindex({f'a{i}':f'ap{i}'})
                
    for obj in [lpdo_1, lpdo_2, lpdo_1_conj, lpdo_2_conj]: 
        obj.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.complex128))

    ov1 = (lpdo_1_conj & lpdo_1 & pqc).contract(optimize=opti)
    ov2 = (lpdo_2_conj & lpdo_2 & pqc).contract(optimize=opti)
    
    dist = (1/2)*(torch.abs(ov1-ov2))
    return -dist


# 3. Non-Redundant and Picklable Model
class TNModel(torch.nn.Module):
    def __init__(self, pqc, lpdo_1, lpdo_2,model_para):
        super().__init__()
        self.lpdo_1, self.lpdo_2 = lpdo_1, lpdo_2
        self.is_acl, self.is_td = model_para.is_acl, model_para.is_td
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
        if self.is_td == 0:
            return full_contraction(pqc.isometrize(method='exp'), self.lpdo_1, self.lpdo_2)
        else:
            return full_contraction_td(pqc.isometrize(method='exp'), self.lpdo_1, self.lpdo_2, self.is_acl)
    


# 4. Optimization Functions
def run_single_optimization(model, num_steps=1000, lr=0.01, lr_schedule='custom_step'):
    """
    Single-start optimization with learning rate scheduling.
    
    Args:
        model: The model to optimize
        num_steps: Total number of optimization steps
        lr: Initial learning rate
        lr_schedule: Type of schedule ('custom_step', 'step_decay', 'cosine_annealing', or None)
    """
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    
    # Setup learning rate scheduler
    if lr_schedule == 'custom_step':
        # First 1/2 epochs: lr
        # Next 1/4 epochs: lr/2
        # Last 1/4 epochs: lr/4
        milestones = [num_steps // 2, num_steps * 3 // 4]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
    elif lr_schedule == 'step_decay':
        # Decay lr by 0.5 every 2000 steps
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)
    elif lr_schedule == 'cosine_annealing':
        # Smoothly decay from lr to lr*0.01 following cosine curve
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=lr*0.01)
    else:
        scheduler = None
    
    pbar = tqdm.tqdm(range(num_steps))
    losses = []
    
    for step in pbar:
        optimizer.zero_grad()
        loss = model.forward()
        losses.append(loss.detach().numpy())
        loss.backward()
        optimizer.step()
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_description(f"Loss={loss.item():.8f}, LR={current_lr:.6f}")
        else:
            pbar.set_description(f"Loss={loss.item():.8f}")

    return min(losses)


def optimization(file1, file2, model_para, optimize_para, save_name="test"):
    Dir = File_access()
    loss_save = np.zeros(optimize_para.sample)
    
    for i_sample in range(optimize_para.sample):
        # Load Data
        M1, M2 = read_data(file1), read_data(file2)
        
        # Pre-calculate LPDDs (once)
        l1_in, l2_in = get_lpdo_torch(M1, M2, model_para, is_td = model_para.is_td)

        # Obtain Unitary Circuit
        N_qb = 2*n if model_para.is_acl == 1 else n
        
        psi_pqc = qmps_f(N_qb, model_para.depth, framework=model_para.framework, is_acl=model_para.is_acl, 
                         pbc=model_para.pbc)
        if model_para.is_acl == 1:
            pqc_init = extract_unitary_circuit_acl(psi_pqc, N_qb,is_td=model_para.is_td)
        else:
            pqc_init = extract_unitary_circuit(psi_pqc, N_qb,is_td=model_para.is_td)

        pqc_init.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.complex128))
        model = TNModel(pqc_init, l1_in, l2_in, model_para)
        single_loss = run_single_optimization(model,num_steps=optimize_para.num_steps,lr=optimize_para.lr)
        loss_save[i_sample] = single_loss
        Dir.save_data(loss_save, save_name)

    return 0


# 5. PROTECTED MAIN BLOCK
if __name__ == "__main__":

    # -------- Parameters ------------------#
    # n = int(sys.argv[1])  # argument
    n = 14
    sample = 1
    lr = 0.002
    num_steps =2000
    # depth = 2*int(sys.argv[2])  # argument
    depth = 2
    framework = 'staircase'
    pbc = False
    is_acl = 1
    is_td = 1  # trace distance
    info = "test"  # for example, codeX, p03, td, etc


    file1 = "M1_a2_N"+str(n)
    file2 = "M2_a2_N"+str(n)
    #file1 = "M1_a0_Xnoise_p03_N"+str(n)
    #file2 = "M1_a2_Xnoise_p03_N"+str(n)
    # file1 = "M1_a2_Znoise_p03_N"+str(n)
    # file2 = "M2_a2_Znoise_p03_N"+str(n)
    #-----------------------------------------#
    if pbc == True:
        print("pbc_optimization, N: ", n)
        save_name = info+"pbcN"+str(n)+"lr"+str(lr)+"num_steps"+str(num_steps)+"sample"+str(sample)+framework+"depth"+str(depth)
    elif pbc == False:
        print("obc_optimization, N: ", n)
        save_name = info+"obcN"+str(n)+"lr"+str(lr)+"num_steps"+str(num_steps)+"sample"+str(sample)+framework+"depth"+str(depth)

    model_para = ModelPara(framework, depth, pbc, is_acl, is_td)
    optimize_para = OptimizePara(lr, num_steps, sample)
    optimization(file1, file2, model_para, optimize_para, save_name=save_name)
    
    Dir = File_access()
    test = Dir.get_back(save_name)
    print(test)