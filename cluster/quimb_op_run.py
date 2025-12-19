import h5py
import quimb as qu
import quimb.tensor as qtn
import numpy as np
import copy

import torch
from torch import optim
import tqdm
import cotengra as ctg

from quimb_op_cluster import *
from file_io import *

from time import time

class ModelPara:
    
    def __init__(self,framework, depth):
        self.framework = framework
        self.depth = depth


class OptimizePara:
    
    def __init__(self,lr, num_steps, sample):
        self.lr = lr
        self.num_steps = num_steps
        self.sample = sample


def build_model(file1, file2, model_para):
    M1 = read_data(file1)
    M2 = read_data(file2)
    n = len(M1)
    is_acl = 1
    rand = True
    val_iden = 0.0

     # model_para.depth should be twice the depth in the note
    pqc_init = get_pqc_torch(n, model_para.depth, model_para.framework, is_acl, rand=rand, val_iden=val_iden)
    lpdo_1_torch, lpdo_2_torch = get_lpdo_torch(M1, M2, is_acl)

    pqc_torch = pqc_init.copy()
    pqc_init.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.complex128))
    pqc_torch.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.complex128))
    lpdo_1_torch.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.complex128))
    lpdo_2_torch.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.complex128))


    model = TNModel(pqc_torch, lpdo_1_torch, lpdo_2_torch)
    print("Model created! Initial loss:", model.forward().item())

    return model


def optimize(file1, file2, model_para, optimize_para, save_name="test",is_mute=1):

    Dir = File_access()
    loss_save = np.zeros(optimize_para.sample)

    for i_sample in range(optimize_para.sample):

        model = build_model(file1, file2, model_para)
        optimizer = optim.Adam(model.parameters(), lr=optimize_para.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=200, gamma=0.5)
        pbar = tqdm.tqdm(range(optimize_para.num_steps), disable=is_mute)
        previous_loss = torch.inf
        losses = []

        for step in pbar:
            optimizer.zero_grad()
            loss = model.forward()
            losses.append(loss.detach().numpy())
            loss.backward()
            optimizer.step()
            pbar.set_description(f"Loss={loss} - LR={lr}")
            if step > 100 and torch.abs(previous_loss - loss) < 1e-10:
                print("Early stopping loss difference is smaller than 1e-10")
                break
            previous_loss = loss.clone()
            
        print(f'traning loss: {loss}')
        loss_save[i_sample] = min(losses)
        Dir.save_data(loss_save, save_name)


    return 0


if __name__ == "__main__":

    task = 'run'

    if task == 'run':

        start_time = time()

        lr = 0.01
        num_steps = 2000
        sample = 1
        N = 10
        file1 = "M1_a0_Xnoise_p03_N"+str(N)
        file2 = "M1_a2_Xnoise_p03_N"+str(N)
        framework = 'staircase'
        depth = 2

        model_para = ModelPara(framework, depth)
        optimize_para = OptimizePara(lr, num_steps, sample)

        save_name = "N"+str(N)+"lr"+str(lr)+"num_steps"+str(num_steps)+"sample"+str(sample)+framework+"depth"+str(depth)
        optimize(file1, file2, model_para, optimize_para, save_name=save_name, is_mute=0)

        print("running time: ", time()-start_time)

        Dir = File_access()
        test = Dir.get_back(save_name)
        print(test)
    
    elif task == 'read':

        save_name = "N14lr0.01num_steps6000sample120staircasedepth2"
        Dir = File_access()
        test = Dir.get_back(save_name)
        print(test)
        print(min(test))
    