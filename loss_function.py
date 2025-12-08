import torch.nn as nn
import torch
import parameters as var
import operators_torch as op

class CustomLossTorch(nn.Module):
    """
    Custom loss function
    sum_n || ( I-PP^+(fn[Uμ]) )v_lambda ||₂
    The class is defined as subclass of nn.Module
    """
    def __init__(self):
        super().__init__() 

    def forward(self, pred, target):
        """
        pred  : Tensor of shape (B, NV, 2, NT, NX)   (complex numbers stored as complex128)
        target: Tensor of shape (B, NV, 2, NT, NX)   (the “near kernel” vectors)
        Returns a scalar loss Tensor (requires_grad=True)
        """
        batch_size = pred.shape[0]
        nv = target.shape[1]
        loss = 0.0
        for i in range(batch_size):
            ops = op.Operators(var.BLOCKS_X, var.BLOCKS_T, pred[i])   
            # loop over the NV dimension (vectorise this later)
            for tv in range(nv):
                corrected = ops.P_Pdagg(target[i, tv])                # shape (2, NT, NX)
                diff = target[i, tv] - corrected
                loss = loss + torch.linalg.norm(diff)
        return loss/(batch_size*nv)