#gauge equivariant layers
import torch
import torch.nn as nn
import parameters as var
var.init()

#f([U],[W]) --> U are the gauge configurations and W the locally transforming quantites
# T(U) = Omega(x) U(x) Omega^+(x+mu),   T(W) = Omega(x) W(x) Omega^+(x)
class LConv(nn.Module):
    """
    A lattice gauge equivariant convolution
    I don't really need the gauge links in the case of the Schwinger model. The U(1) gauge 
    group makes many thigns gauge invariant quite easily.
    f_i(U,W) = Sum_{j,mu,k} Omega_{i,j,mu,k} W_j(x+k hat{mu})

    The kernel size in this example is 2 and the total number of (complex) weights is 5.
    o     o     o
          |
    o --- o --- o
          |
    o     o     o
    """
    def __init__(self, n_in, n_out, kernel_size):
        super(LConv, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.kernel_size = kernel_size
        self.dims = 2 #We hardcode two dimensions for the Schwinger model
        #Initialize weights Omega_{i,j,mu,k} (j,mu,k) are the in features, (i) is the out feature 
        w_in_size = self.n_in * (2 * (self.kernel_size-1)  * self.dims + 1)
        w_out_size = self.n_out

        self.weight = nn.Parameter(torch.empty(w_out_size, w_in_size,dtype=var.PREC,device=var.DEVICE))
        std = 1.0
        nn.init.normal_(self.weight, mean=0.0, std=std)

    def forward(self, w):
        #W = [Re(U01),Im(U01), ...] 
        #I could include more locally transforming quantities in the future
        #w.shape = (Batch,n_in,NT,NX)
        #u.shape = (Batch,4,NT,NX)
        
        #Reshape u into a complex tensor ...
        
        transported_terms = [w.clone()] #k = 0
        for orientation in [+1, -1]:
            for mu in range(self.dims):
                w_transport = w.clone()
                for k in range(1, self.kernel_size):
                    #transported terms
                    #Compute W_j(x - orientation * k mu) (the sign depends on the orientation)
                    w_transport = torch.roll(w,shifts=k*orientation,dims=2+mu)  #Roll automatically considers periodic boundaries
                    transported_terms.append(w_transport)

        # combine terms into a single tensor (along dim = 1 because of the batches)
        t_w = torch.cat(transported_terms, dim=1)
        
        # f_i(W) = w_{ij} TW_{j}(t,x)
        w = torch.einsum('ij, bjxt -> bixt', self.weight, t_w)
        return w