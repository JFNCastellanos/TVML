import numpy as np
import torch.nn as nn
import torch
import parameters as var
import operators_torch as op

class CustomLoss(nn.Module):
    """
    Custom loss function
    sum_n ||  ( I-PP^+(fn[Uμ]) )fn[Uμ] ||₂
    The class is defined as subclass of nn.Module
    """
    def __init__(self,pred_test_vectors,test_vectors):
        super(CustomLoss, self).__init__() #Calls the constructor of the base class, i.e. Module

        self.test_vectors = test_vectors           #[batch_size,NV,2,NT,NX]
        self.pred_test_vectors = pred_test_vectors #[batch_size,NV,2,NT,NX]
        self.batch_size = self.test_vectors.shape[0]
    
    def forward(self):
        #P_Pdagg(f_n) = P(f_n)P^+(f_n)f_n
        loss = 0
        for i in range(self.batch_size):
            ops = op.Operators(var.BLOCKS_X,var.BLOCKS_T,self.pred_test_vectors[i])
            for tv in range(var.NV):
                #Evaluate 
                loss += np.linalg.norm( (self.test_vectors[i,tv] - ops.P_Pdagg(self.test_vectors[i,tv])) )
            #loss = np.linealg.norm(test_vectors - P_Pdagg(test_vectors)) #|| . ||₂ (l2-norm)
            #We normalize the loss function, otherwise the minimization is unstable.
        return loss/(self.batch_size*var.NV)
 
class CustomLossTorch(nn.Module):
    """
    Custom loss function
    sum_n || ( I-PP^+(fn[Uμ]) )fn[Uμ] ||₂
    The class is defined as subclass of nn.Module
    """
    def __init__(self):
        super().__init__()                     

    def forward(self, pred, target):
        """
        pred  : Tensor of shape (B, NV, 2, NT, NX)   (complex numbers stored as complex dtype)
        target: Tensor of shape (B, NV, 2, NT, NX)   (the “near kernel” vectors)
        Returns a scalar loss Tensor (requires_grad=True)
        """
        batch_size = pred.shape[0]
        loss = 0.0
        for i in range(batch_size):
            # the Operators class must accept a **torch** tensor, not a numpy array
            ops = op.Operators(var.BLOCKS_X, var.BLOCKS_T, pred[i])   # <-- pred[i] is a torch Tensor

            # loop over the NV dimension (vectorise this later)
            for tv in range(var.NV):
                #  P_Pdagg returns a tensor with the same shape as its input
                corrected = ops.P_Pdagg(target[i, tv])                # shape (2, NT, NX)
                diff = target[i, tv] - corrected

                # torch.norm with p=2 gives the Frobenius (L2) norm for tensors
                loss = loss + torch.linalg.norm(diff)   # square to match ∥·∥₂² if you like

        # loss is a scalar tensor (still attached to the graph)
        return loss/(batch_size*var.NV)