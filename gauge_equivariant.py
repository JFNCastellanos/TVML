#gauge equivariant layers
import torch
import torch.nn as nn
import parameters as var
var.init()


#TODO:
# Check that the parallel transport is correctly implemented. Basically verify that roll is working fine
# Check if "cat" is ok for my case
# Check that einsum is fine 
# See how the batches affect this layer 

#f([U],[W]) --> U are the gauge configurations and W the locally transforming quantites
# T(U) = Omega(x) U(x) Omega^+(x+mu),   T(W) = Omega(x) W(x) Omega^+(x)
class LConv(torch.nn.Module):
    """
    A lattice gauge equivariant convolution
    f_i(U,W) = Sum_{j,mu,k} Omega_{i,j,mu,k} W_j(x+k hat{mu})

    The kernel size in this example is 1 and the total number of (complex) weights is 5.
    o     o     o
          |
    o --- o --- o
          |
    o     o     o
    """
    def __init__(self, w, kernel_size, n_out):
        super(LConv, self).__init__()
        self.w = w #W = [Re(U01),Im(U01), ...] 
        #I could include more locally transforming quantities in the future
        #self.w[i].shape = (Batch,NT,NX)
        self.n_in = len(self.w)
        self.n_out = n_out
        self.kernel_size = kernel_size
        self.dims = 2 #We hardcode two dimensions for the Schwinger model

        #Initialize weights Omega_{i,j,mu,k} (j,mu,k) are the in features, (i) is the out feature 
        w_in_size = self.n_in * (2 * self.kernel_size  * self.dims + 1)
        w_out_size = self.n_out

        self.weight = nn.Parameter(torch.empty(w_out_size, w_in_size))
        std = 1.0
        nn.init.normal_(self.weight, mean=0.0, std=std)


    def forward(self, w):
        # local term
        transported_terms = [w.clone()] #k = 0, we don't transport anything in this case
        for orientation in [+1, -1]:
            for mu in range(len(self.dims)):
                w_transport = w.clone()
                for k in range(1, self.kernel_size):
                    # get transported terms
                    #Compute W_j(x +- k mu) (the sign depends on the orientation)
                    w_transport = torch.roll(w,shift=k*orientation,axis=mu)
                    # and add to list
                    transported_terms.append(w_transport)

        # combine terms into a single tensor
        t_w = torch.cat(transported_terms, dim=0)

        # perform multiplication and apply weights
        w = einsum('ij, bjxt -> bixt', self.weight, t_w)
        return w