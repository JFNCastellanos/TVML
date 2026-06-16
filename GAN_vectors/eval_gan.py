import torch.nn as nn
import torch
import parameters as var
import operators_torch as op
var.init()

def check_metric(pred,target):
    #pred.shape = [Nv,2,Nt,Nx] 
    #targe.shape = [Nv,2,Nt,Nx]
    
    #Build P, P^+ with preds
    ops = op.Operators(var.BLOCKS_X, var.BLOCKS_T, pred)  
    nv = pred.shape[0]
    loss = 0
    def ind_loss(corrected,target):
        return torch.linalg.norm(target - corrected)**2
        
    for tv in range(nv):
        loss += ind_loss(ops.P_Pdagg(target[tv]),target[tv])
        #corrected = ops.P_Pdagg(target[tv])                # shape (2, NT, NX)
        #diff = target[tv] - corrected
        #loss = loss + torch.linalg.norm(diff)**2
    return loss/nv
 