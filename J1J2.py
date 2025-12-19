# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 11:36:51 2025

@author: yliu
"""

import numpy as np
import time
from math import pi
import matplotlib.pyplot as plt
# plt.style.use('seaborn-whitegrid')


from hamiltonian.spin_tool import Spin_hamiltonian, Sx, Sy, Sz
from toolkit.file_io import File_access
from entangle.ent_many_body import get_ent_many_total


def get_spectrum(N=10, plot=0,knum=50, is_save=0):
    start_time = time.time()
    Dir = File_access()
    PBC = 1
    bands = 1
  
    # Using the following choice gives the correct entanglement. 
    # The small offset is necessary.
    J1s = 1
    J2s = 0.241167
    J1 = np.tile([J1s],N)
    J2 = np.tile([J2s],N)
    
    # Generate hamiltonian and h2
    # Notice H=\sum_i (-e_i) in the paper. Need to take care of the minus sign.    
    couple_off = [[J1,Sx,Sx,1],[J1,Sy,Sy,1]]
    couple_off = [[J2,Sx,Sx,2],[J2,Sy,Sy,2]]
    
    couple_diag = [[J1,Sz,Sz,1],
                   [J2,Sz,Sz,2]]
    const_term = 0

          
    J1J2 = Spin_hamiltonian(N, couple_diag, couple_off, PBC, bands, const_term = const_term)
    J1J2.label = 'hermitian'
    
    # Solve the eigensyste
    eigval, eigvec, S = J1J2.sort_P(knum, is_sum_Sz = 1, is_prod_Sz=0)  
 

    #-----Output-----
    print("--- %s seconds ---\n" % (time.time() - start_time))  
  
    # plt.scatter(eigval.real,eigval.imag, color='blue')
    # plt.xlim(min(eigval.real)*1.1,max(eigval.real)*1.1)
    # plt.ylim(-0.5,0.5)
    
    # save the variables
    if is_save == 1: Dir.save_data(J1J2)
    
    return J1J2
  
  

if __name__ == "__main__":
    is_get_spectrum = 1
    is_from_new = 1
    
    is_extract_spectrum = 1 - is_get_spectrum
    
    if is_get_spectrum:
        N = 8
        J1J2 = get_spectrum(N=N, plot=1, knum=8)
        int_tot, ent_tot, coeffs = get_ent_many_total(J1J2, level=0, renyi=1,
                                              even_odd = 'odd')

    
    elif is_extract_spectrum:
        Dir = File_access()
        J1J2 = Dir.get_back_ext(is_from_new)
        # Use LY.__dict__.keys() to see all the instant variables
        
        int_tot, ent_tot, coeffs = get_ent_many_total(J1J2, level=0, renyi=1,
                                             even_odd = 'even')
        
 
        # x = 4
        # q = np.exp(1j*pi/(x+1))  
        # v1 =  1/np.sqrt(q+1/q)*(q**(-1/2))
        # v2 =  1/np.sqrt(q+1/q)*(q**(1/2))
        # vg = np.array([0,v1,-v2,0])
        
        # S = get_ent_many(vg, vg.conj(), 2 ,1,q=q)
        # print(S)
        # print(np.log(q+1/q))
        
        # S = get_ent_many(vg, vg.conj(), 2 ,1)
        # print(S)
        # S_t = -(q/(q+1/q))*np.log(q/(q+1/q))-(1/q/(q+1/q))*np.log(1/q/(q+1/q))
        # print(S_t)
        
# (old result)        
#   x = 1, N = 12 (6 unitcells), GS EE        
#####################################
#   L_sub   | open chain (from left)
#-----------------------------------
#   2       |       -7.5048
#   4       |       -7.7073
#   6       |       -7.7580
#   8       |       -7.7073
#   10      |       -7.5048
#-----------------------------------
#           |       c/2=-1.1007
#####################################
# c/2 because it is open chain.
