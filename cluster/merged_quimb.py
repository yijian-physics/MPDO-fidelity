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

def read_data(data_name, use_save_results=True):
    """
    Read data from HDF5 file.
    
    Args:
        data_name: Name of the data file (without .h5 extension)
        use_save_results: If True, looks in "save_results/" directory, otherwise in current directory
    """
    path = "save_results/" + data_name + ".h5" if use_save_results else data_name + ".h5"
    with h5py.File(path, "r") as f:
        keys = sorted(f.keys(), key=lambda x: int(x.split("_")[1]))
        data = [np.transpose(f[key][:], (3,2,1,0)) for key in keys]
    return data


def array_to_lpdo(M1, tags):
    """Convert input list of arrays to LPDO."""
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
    """Add ancilla qubits to LPDO."""
    lpdo_acl = lpdo.copy()
    for i in range(len(lpdo.tensors)):
        prod = qtn.Tensor(np.array([1,0]), inds=(label+f'{i}',), tags='A')
        lpdo_acl = lpdo_acl & prod
    return lpdo_acl


### build unitary circuit

def brickwall_unitary(psi, n_apply, list_u3, depth, n_Qbit, val_iden=0, rand=False, start_layer=0, is_acl=0, pbc=True):
    """
    Brickwall unitary circuit with optional periodic boundary conditions.
    
    Args:
        psi: MPS state to apply gates to
        n_apply: Current gate counter
        list_u3: List to track gate names
        depth: Circuit depth
        n_Qbit: Number of qubits
        val_iden: Perturbation from identity
        rand: Use random unitaries if True
        start_layer: Starting layer offset
        is_acl: 0 for no ancilla (2-qubit gates), 1 for ancilla (4-qubit gates)
        pbc: If True, add wraparound gates between first and last qubits
    """
    if n_Qbit==0 or n_Qbit==1: 
        depth=1

    for r in range(depth):
        if (r+start_layer)%2==0:
            if is_acl==0:
                # Standard even layer gates
                for i in range(0, n_Qbit-1, 2):
                    G = qu.rand_uni(4, dtype=complex) if rand else qu.identity(4,dtype=complex)+qu.rand_uni(4, dtype=complex)*val_iden
                    psi.gate_(G, (i, i + 1), tags={'U',f'G{n_apply}', f'L{i}D{r}'})
                    list_u3.append(f'G{n_apply}')
                    n_apply+=1
                
                # PBC wraparound gate if n_Qbit is even and last qubit not covered
                if pbc and n_Qbit % 2 == 0 and n_Qbit >= 2:
                    G = qu.rand_uni(4, dtype=complex) if rand else qu.identity(4,dtype=complex)+qu.rand_uni(4, dtype=complex)*val_iden
                    psi.gate_(G, (n_Qbit-1, 0), tags={'U',f'G{n_apply}', f'PBC_D{r}'})
                    list_u3.append(f'G{n_apply}')
                    n_apply+=1

            elif is_acl==1:
                # Ancilla case: 4-qubit gates on even layer
                for i in range(0, n_Qbit-3, 4):
                    G = qu.rand_uni(16, dtype=complex) if rand else qu.identity(16,dtype=complex)+qu.rand_uni(16, dtype=complex)*val_iden
                    psi.gate_(G, (i, i+1, i+2, i+3), tags={'U',f'G{n_apply}', f'L{i}D{r}'})
                    list_u3.append(f'G{n_apply}')
                    n_apply+=1

                # PBC wraparound gate for ancilla if needed
                if pbc and n_Qbit % 4 == 0 and n_Qbit >= 4:
                    G = qu.rand_uni(16, dtype=complex) if rand else qu.identity(16,dtype=complex)+qu.rand_uni(16, dtype=complex)*val_iden
                    psi.gate_(G, (n_Qbit-3, n_Qbit-2, n_Qbit-1, 0), tags={'U',f'G{n_apply}', f'PBC_D{r}'})
                    list_u3.append(f'G{n_apply}')
                    n_apply+=1

        else:
            if is_acl==0:
                # Standard odd layer gates
                for i in range(0, n_Qbit-2, 2):
                    G = qu.rand_uni(4, dtype=complex) if rand else qu.identity(4,dtype=complex)+qu.rand_uni(4, dtype=complex)*val_iden
                    psi.gate_(G, (i+1, i+2), tags={'U',f'G{n_apply}', f'L{i}D{r}'})
                    list_u3.append(f'G{n_apply}')
                    n_apply+=1

                # PBC wraparound gate for odd layers
                if pbc and n_Qbit >= 2:
                    G = qu.rand_uni(4, dtype=complex) if rand else qu.identity(4,dtype=complex)+qu.rand_uni(4, dtype=complex)*val_iden
                    psi.gate_(G, (n_Qbit-1, 0), tags={'U',f'G{n_apply}', f'PBC_D{r}'})
                    list_u3.append(f'G{n_apply}')
                    n_apply+=1

            elif is_acl==1:
                # Ancilla case: 4-qubit gates on odd layer
                for i in range(2, n_Qbit-2, 4):
                    G = qu.rand_uni(16, dtype=complex) if rand else qu.identity(16,dtype=complex)+qu.rand_uni(16, dtype=complex)*val_iden
                    psi.gate_(G, (i, i+1, i+2, i+3), tags={'U',f'G{n_apply}', f'L{i}D{r}'})
                    list_u3.append(f'G{n_apply}')
                    n_apply+=1

                # PBC wraparound for ancilla odd layer
                if pbc and n_Qbit >= 4:
                    G = qu.rand_uni(16, dtype=complex) if rand else qu.identity(16,dtype=complex)+qu.rand_uni(16, dtype=complex)*val_iden
                    psi.gate_(G, (n_Qbit-2, n_Qbit-1, 0, 1), tags={'U',f'G{n_apply}', f'PBC_D{r}'})
                    list_u3.append(f'G{n_apply}')
                    n_apply+=1

    return n_apply, list_u3


def staircase_unitary(psi, n_apply, list_u3, depth, n_Qbit, val_iden=0, rand=False, start_layer=0, is_acl=0, pbc=True):
    """
    Staircase unitary circuit with optional periodic boundary conditions.
    
    Args:
        psi: MPS state to apply gates to
        n_apply: Current gate counter
        list_u3: List to track gate names
        depth: Circuit depth
        n_Qbit: Number of qubits
        val_iden: Perturbation from identity
        rand: Use random unitaries if True
        start_layer: Starting layer offset
        is_acl: 0 for no ancilla (2-qubit gates), 1 for ancilla (4-qubit gates)
        pbc: If True, add wraparound gates between first and last qubits
    """
    if n_Qbit==0 or n_Qbit==1: 
        depth=1

    for r in range(depth):
        if (r+start_layer)%2==0:
            if is_acl == 0:
                # Forward sweep
                for i in range(0, n_Qbit-1, 1):
                    G = qu.rand_uni(4, dtype=complex) if rand else qu.identity(4,dtype=complex)+qu.rand_uni(4, dtype=complex)*val_iden
                    psi.gate_(G, (i, i + 1), tags={'U',f'G{n_apply}',f'L{i}D{r}'})
                    list_u3.append(f'G{n_apply}')
                    n_apply+=1

                # PBC wraparound gate
                if pbc and n_Qbit >= 2:
                    G = qu.rand_uni(4, dtype=complex) if rand else qu.identity(4,dtype=complex)+qu.rand_uni(4, dtype=complex)*val_iden
                    psi.gate_(G, (n_Qbit-1, 0), tags={'U',f'G{n_apply}',f'PBC_D{r}'})
                    list_u3.append(f'G{n_apply}')
                    n_apply+=1

            elif is_acl == 1:
                # Forward sweep with 4-qubit gates
                for i in range(0, n_Qbit-3, 2):
                    G = qu.rand_uni(16, dtype=complex) if rand else qu.identity(16,dtype=complex)+qu.rand_uni(16, dtype=complex)*val_iden
                    psi.gate_(G, (i, i + 1, i+2, i+3), tags={'U',f'G{n_apply}',f'L{i}D{r}'})
                    list_u3.append(f'G{n_apply}')
                    n_apply+=1

                # PBC wraparound for ancilla
                if pbc and n_Qbit >= 4:
                    G = qu.rand_uni(16, dtype=complex) if rand else qu.identity(16,dtype=complex)+qu.rand_uni(16, dtype=complex)*val_iden
                    psi.gate_(G, (n_Qbit-2, n_Qbit-1, 0, 1), tags={'U',f'G{n_apply}',f'PBC_D{r}'})
                    list_u3.append(f'G{n_apply}')
                    n_apply+=1

        else:
            if is_acl == 0:
                # Backward sweep
                for i in range(n_Qbit-1, 0, -1):
                    G = qu.rand_uni(4, dtype=complex) if rand else qu.identity(4,dtype=complex)+qu.rand_uni(4, dtype=complex)*val_iden
                    psi.gate_(G, (i-1, i), tags={'U',f'G{n_apply}',f'L{i}D{r}'})
                    list_u3.append(f'G{n_apply}')
                    n_apply+=1

                # PBC wraparound gate
                if pbc and n_Qbit >= 2:
                    G = qu.rand_uni(4, dtype=complex) if rand else qu.identity(4,dtype=complex)+qu.rand_uni(4, dtype=complex)*val_iden
                    psi.gate_(G, (n_Qbit-1, 0), tags={'U',f'G{n_apply}',f'PBC_D{r}'})
                    list_u3.append(f'G{n_apply}')
                    n_apply+=1

            elif is_acl == 1:
                # Backward sweep with 4-qubit gates
                for i in range(n_Qbit-1, 2, -2):
                    G = qu.rand_uni(16, dtype=complex) if rand else qu.identity(16,dtype=complex)+qu.rand_uni(16, dtype=complex)*val_iden
                    psi.gate_(G, (i-3, i-2, i-1, i), tags={'U',f'G{n_apply}',f'L{i}D{r}'})
                    list_u3.append(f'G{n_apply}')
                    n_apply+=1

                # PBC wraparound for ancilla backward
                if pbc and n_Qbit >= 4:
                    G = qu.rand_uni(16, dtype=complex) if rand else qu.identity(16,dtype=complex)+qu.rand_uni(16, dtype=complex)*val_iden
                    psi.gate_(G, (n_Qbit-3, n_Qbit-2, n_Qbit-1, 0), tags={'U',f'G{n_apply}',f'PBC_D{r}'})
                    list_u3.append(f'G{n_apply}')
                    n_apply+=1

    return n_apply, list_u3


def qmps_f(L=16, in_depth=2, val_iden=0, rand=True, start_layer=0, framework='brickwall', is_acl=0, pbc=True):
    """
    Create a parameterized quantum circuit.
    
    Args:
        L: Number of qubits
        in_depth: Circuit depth
        val_iden: Perturbation from identity
        rand: Use random unitaries
        start_layer: Starting layer offset
        framework: 'brickwall', 'staircase', or 'mixed'
        is_acl: 0 for no ancilla (2-qubit gates), 1 for ancilla (4-qubit gates)
        pbc: If True, add periodic boundary condition wraparound gates
    """
    list_u3 = []
    n_apply = 0
    psi = qtn.MPS_computational_state('0' * L)
    
    for i in range(L):
        t = psi[i]
        indx = 'k'+str(i)
        t.modify(left_inds=[indx])

    for t in range(L):
        psi[t].modify(tags=[f"I{t}", "MPS"])

    if framework == 'brickwall':
        n_apply, list_u3 = brickwall_unitary(psi, n_apply, list_u3, in_depth, L, 
                                             val_iden=val_iden, rand=rand, 
                                             start_layer=start_layer, is_acl=is_acl, pbc=pbc)
    elif framework == 'staircase':
        n_apply, list_u3 = staircase_unitary(psi, n_apply, list_u3, in_depth, L, 
                                             val_iden=val_iden, rand=rand, 
                                             start_layer=start_layer, is_acl=is_acl, pbc=pbc)
    elif framework == 'mixed':
        n_apply, list_u3 = brickwall_unitary(psi, n_apply, list_u3, in_depth, L, 
                                             val_iden=val_iden, rand=rand, 
                                             start_layer=start_layer, is_acl=is_acl, pbc=pbc)
        n_apply, list_u3 = staircase_unitary(psi, n_apply, list_u3, in_depth, L, 
                                             val_iden=val_iden, rand=rand, 
                                             start_layer=start_layer, is_acl=is_acl, pbc=pbc)

    return psi.astype_('complex128')


def extract_unitary_circuit(psi_pqc, num_qubits, is_td=0):
    """
    Extract unitary circuit from quantum circuit (no ancilla).
    
    Args:
        psi_pqc: Full quantum circuit
        num_qubits: Number of system qubits
        is_td: 0 for overlap calculation, 1 for trace distance
    """
    pqc = psi_pqc.tensors[num_qubits]
    for i in range(num_qubits+1, len(psi_pqc.tensors)):
        pqc = pqc & psi_pqc.tensors[i]

    if is_td == 0:
        for i in range(num_qubits):
            pqc = pqc.reindex({f'k{i}': f'e{i}'})
            pqc = pqc.reindex({psi_pqc.tensors[i].inds[-1]: f'ep{i}'})
    elif is_td == 1:
        for i in range(num_qubits):
            pqc = pqc.reindex({f'k{i}': f's{i}'})
            pqc = pqc.reindex({psi_pqc.tensors[i].inds[-1]: f'sp{i}'})

    return pqc


def extract_unitary_circuit_acl(psi_pqc, num_qubits, is_td=0):
    """
    Extract unitary circuit with ancilla qubits.
    
    Args:
        psi_pqc: Full quantum circuit
        num_qubits: Total number of qubits (2*n for n system + n ancilla)
        is_td: 0 for overlap calculation, 1 for trace distance
    """
    pqc = psi_pqc.tensors[num_qubits]
    for i in range(num_qubits+1, len(psi_pqc.tensors)):
        pqc = pqc & psi_pqc.tensors[i]

    for i in range(num_qubits):
        if i % 2:  # odd indices are system qubits
            if is_td == 0:
                pqc = pqc.reindex({f'k{i}': f'e{i//2}'})
                pqc = pqc.reindex({psi_pqc.tensors[i].inds[-1]: f'ep{i//2}'})
            elif is_td == 1:
                pqc = pqc.reindex({f'k{i}': f's{i//2}'})
                pqc = pqc.reindex({psi_pqc.tensors[i].inds[-1]: f'sp{i//2}'})
        else:  # even indices are ancilla qubits
            pqc = pqc.reindex({f'k{i}': f'a{i//2}'})
            pqc = pqc.reindex({psi_pqc.tensors[i].inds[-1]: f'ap{i//2}'})

    return pqc


def full_contraction(pqc, lpdo_1, lpdo_2, is_show=0):
    """
    Contract the full tensor network for overlap calculation.
    
    Args:
        pqc: Parameterized quantum circuit
        lpdo_1: First LPDO
        lpdo_2: Second LPDO
        is_show: If 1, draw the tensor network
    """
    if is_show == 1:
        (lpdo_1 & lpdo_2 & pqc).draw(['U','M2','M1'])

    output = abs((lpdo_1 & lpdo_2 & pqc).contract(optimize=opti))
    return -output


def full_contraction_td(pqc, lpdo_1, lpdo_2, lpdo_1_conj, lpdo_2_conj, is_show=0):
    """
    Contract the full tensor network for trace distance calculation.
    
    Args:
        pqc: Parameterized quantum circuit
        lpdo_1: First LPDO
        lpdo_2: Second LPDO
        lpdo_1_conj: Conjugate of first LPDO
        lpdo_2_conj: Conjugate of second LPDO
        is_show: If 1, draw the tensor network
    """
    if is_show == 1:
        (lpdo_1_conj & lpdo_1 & pqc).draw(['U','M2','M1'])

    ov1 = (lpdo_1_conj & lpdo_1 & pqc).contract(optimize=opti)
    ov2 = (lpdo_2_conj & lpdo_2 & pqc).contract(optimize=opti)
    
    dist = (1/2)*(torch.abs(ov1-ov2))
    return -dist


class TNModel(torch.nn.Module):
    """
    Tensor Network Model for overlap optimization.
    Uses parameterization from second file (Lie algebra generators).
    """
    def __init__(self, pqc, lpdo_1, lpdo_2):
        super().__init__()
        self.lpdo_1, self.lpdo_2 = lpdo_1, lpdo_2
        
        # Convert to torch if needed
        pqc_torch = pqc.copy()
        pqc_torch.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.complex128))
        
        params, self.skeleton = qtn.pack(pqc_torch)
        self.torch_params, self.param_metadata = torch.nn.ParameterDict(), {}
        
        for i, initial in params.items():
            d = int(np.sqrt(initial.numel()))
            self.torch_params[str(i)] = torch.nn.Parameter(torch.randn(d**2, dtype=torch.float64)*0.05)
            self.param_metadata[str(i)] = (d, initial.shape)

    def _get_complex_generators(self):
        """Generate anti-Hermitian matrices from real parameters."""
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
    
    def forward_debug(self):
        """Forward pass without isometrization (for debugging)."""
        pqc = qtn.unpack(self._get_complex_generators(), self.skeleton)
        return full_contraction(pqc, self.lpdo_1, self.lpdo_2)
    
    def get_unitary_circuit(self):
        """Extract the optimized unitary quantum circuit."""
        params = self._get_complex_generators()
        pqc = qtn.unpack(params, self.skeleton)
        return pqc.isometrize(method='exp')


class TNModel_td(torch.nn.Module):
    """
    Tensor Network Model for trace distance optimization.
    """
    def __init__(self, pqc, lpdo_1, lpdo_2, is_acl, n):
        super().__init__()

        pqc_torch = pqc.copy()
        pqc_torch.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.complex128))
        params, self.skeleton = qtn.pack(pqc_torch)

        lpdo_1_conj = lpdo_1.H
        lpdo_2_conj = lpdo_2.H

        if is_acl == 0:
            for i in range(n):
                lpdo_1_conj = lpdo_1_conj.reindex({f's{i}': f'sp{i}'})
                lpdo_2_conj = lpdo_2_conj.reindex({f's{i}': f'sp{i}'})
        elif is_acl == 1:
            for i in range(n):
                lpdo_1_conj = lpdo_1_conj.reindex({f's{i}': f'sp{i}'})
                lpdo_2_conj = lpdo_2_conj.reindex({f's{i}': f'sp{i}'})
                lpdo_1_conj = lpdo_1_conj.reindex({f'a{i}': f'ap{i}'})
                lpdo_2_conj = lpdo_2_conj.reindex({f'a{i}': f'ap{i}'})

        lpdo_1_torch = lpdo_1.copy()
        lpdo_2_torch = lpdo_2.copy()
        lpdo_1_torch.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.complex128))
        lpdo_2_torch.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.complex128))
        lpdo_1_conj.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.complex128))
        lpdo_2_conj.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.complex128))

        self.torch_params, self.param_metadata = torch.nn.ParameterDict(), {}
        for i, initial in params.items():
            d = int(np.sqrt(initial.numel()))
            self.torch_params[str(i)] = torch.nn.Parameter(torch.randn(d**2, dtype=torch.float64)*0.05)
            self.param_metadata[str(i)] = (d, initial.shape)

        self._loss_fn = lambda x: full_contraction_td(x, lpdo_1_torch, lpdo_2_torch, lpdo_1_conj, lpdo_2_conj)

    def _get_complex_generators(self):
        """Generate anti-Hermitian matrices from real parameters."""
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
        params = self._get_complex_generators()
        pqc = qtn.unpack(params, self.skeleton)
        return self._loss_fn(pqc.isometrize(method='exp'))
    
    def get_unitary_circuit(self):
        """Extract the optimized unitary quantum circuit."""
        params = self._get_complex_generators()
        pqc = qtn.unpack(params, self.skeleton)
        return pqc.isometrize(method='exp')


def get_pqc_torch(n, depth, framework, is_acl, rand=True, val_iden=0.0, is_td=0, pbc=True):
    """
    Get parameterized quantum circuit as torch-compatible tensor network.
    
    Args:
        n: Number of system qubits
        depth: Circuit depth
        framework: 'brickwall', 'staircase', or 'mixed'
        is_acl: 0 for no ancilla, 1 for ancilla
        rand: Use random unitaries
        val_iden: Perturbation from identity
        is_td: 0 for overlap, 1 for trace distance
        pbc: Use periodic boundary conditions
    """
    if is_acl == 0:
        num_qubits = n
        psi_pqc = qmps_f(num_qubits, in_depth=depth, val_iden=val_iden, rand=rand, 
                        framework=framework, pbc=pbc)
        pqc = extract_unitary_circuit_acl(psi_pqc, num_qubits, is_td=is_td)
    
    return pqc


def get_lpdo_torch(M1, M2, is_acl):
    """
    Get LPDO tensor networks as torch-compatible objects for overlap calculation.
    
    Args:
        M1: First MPS data
        M2: Second MPS data
        is_acl: 0 for no ancilla, 1 for ancilla
    """
    n = len(M1)
    lpdo_1 = array_to_lpdo(M1, ('M1',))

    lpdo_2 = array_to_lpdo(M2, ('M2',))
    lpdo_2 = lpdo_2.H
    for i in range(n):
        lpdo_2 = lpdo_2.reindex({f'e{i}': f'ep{i}'})

    lpdo_1_acl = add_ancilla(lpdo_1, "a")
    lpdo_2_acl = add_ancilla(lpdo_2, "ap")

    if is_acl == 0:
        lpdo_1_torch = lpdo_1.copy()
        lpdo_2_torch = lpdo_2.copy()
    elif is_acl == 1:
        lpdo_1_torch = lpdo_1_acl.copy()
        lpdo_2_torch = lpdo_2_acl.copy()

    return lpdo_1_torch, lpdo_2_torch


def get_lpdo_torch_td(M1, M2, is_acl):
    """
    Get LPDO tensor networks as torch-compatible objects for trace distance calculation.
    
    Args:
        M1: First MPS data
        M2: Second MPS data
        is_acl: 0 for no ancilla, 1 for ancilla
    """
    n = len(M1)
    lpdo_1 = array_to_lpdo(M1, ('M1',))
    lpdo_2 = array_to_lpdo(M2, ('M2',))

    lpdo_1_acl = add_ancilla(lpdo_1, "a")
    lpdo_2_acl = add_ancilla(lpdo_2, "a")

    if is_acl == 0:
        lpdo_1_torch = lpdo_1.copy()
        lpdo_2_torch = lpdo_2.copy()
    elif is_acl == 1:
        lpdo_1_torch = lpdo_1_acl.copy()
        lpdo_2_torch = lpdo_2_acl.copy()

    return lpdo_1_torch, lpdo_2_torch


# 4. Optimization Functions
def run_single_optimization(model, num_steps=1000, lr=0.01):
    """Original single-start optimization logic."""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    pbar = tqdm.tqdm(range(num_steps))
    for step in pbar:
        optimizer.zero_grad()
        loss = model.forward()
        loss.backward()
        optimizer.step()
        pbar.set_description(f"Loss={loss.item():.8f}")
    return loss.item()


# 5. PROTECTED MAIN BLOCK
if __name__ == "__main__":
    # Load Data
    M1, M2 = read_data("M1_a2"), read_data("M2_a2")
    n = len(M1)
    
    # Pre-calculate LPDDs (once)
    l1, l2 = array_to_lpdo(M1, ('M1',)), array_to_lpdo(M2, ('M2',)).H
    for i in range(n): l2 = l2.reindex({f'e{i}': f'ep{i}'})
    l1_acl, l2_acl = add_ancilla(l1, "a"), add_ancilla(l2, "ap")
    for obj in [l1_acl, l2_acl]: 
        obj.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.complex128))

    psi_pqc = qmps_f(2*n, 2, is_acl=1, pbc=True)
    pqc_init = extract_unitary_circuit_acl(psi_pqc, 2*n)
    pqc_init.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.complex128))
    model = TNModel(pqc_init, l1_acl, l2_acl)
    run_single_optimization(model,num_steps=2000,lr=0.01)