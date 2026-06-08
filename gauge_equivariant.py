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
    f_i(U,W) = Sum_{j,mu,k} Omega_{i,j,mu,k} U_{x,kmu}W_j(x+k hat{mu})
    
              = U_mu(x) U_mu(x+mu) ... U_mu(x+(K-1)mu), if k>0
    U_{x,kmu} =  U_mu(x), if k = 0
              = U_mu^+(x-mu) U_mu^+(x-2mu) ... U_mu^+(x-|k|mu), if k<0

    n_in and n_out should consider that u and w are complex tensors
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

        std = 1.0
        real = torch.empty(w_out_size, w_in_size, dtype=var.PREC, device=var.DEVICE)
        imag = torch.empty(w_out_size, w_in_size, dtype=var.PREC, device=var.DEVICE)
        nn.init.normal_(real, mean=0.0, std=std)
        nn.init.normal_(imag, mean=0.0, std=std)
        self.weight = nn.Parameter(torch.complex(real, imag))
        #print("Weight shape",self.weight.shape)

    def forward(self, u, w):
        #We assume that the inputs are given as complex tensors
        #w.shape = (Batch,n_in,NT,NX) 
        #u.shape = (Batch,2,NT,NX)
        
        transported_terms = [w.clone()] #k = 0
        for orientation in [+1, -1]:
            for mu in range(self.dims):
                w_transport = w.clone()
                for k in range(1, self.kernel_size):
                    #transported terms
                    #Compute W_j(x - orientation * k mu) (the sign depends on the orientation)
                    w_transport = torch.roll(w,shifts=k*orientation,dims=2+mu)  #Roll automatically considers periodic boundaries
                    u_transporter = transporter(u,orientation,mu,k) #U_{x,kmu}
                    #print("w transport shape",w_transport.shape)
                    #print("u transporter shape",u_transporter.shape)
                    transported_terms.append(w_transport*u_transporter)

        # combine terms into a single tensor (along dim = 1 because of the batches)
        t_w = torch.cat(transported_terms, dim=1)
        # f_i(W) = w_{ij} TW_{j}(t,x)
        w = torch.einsum('ij, bjxt -> bixt', self.weight, t_w)
        return u, w

def transporter(u,orientation,mu,k):
    p_transporter = 1
    u_mu = u[:,mu] #(B,NT,NX)
    #x+mu
    if orientation == -1:
        #We are just multiplying U(1) numbers, so the order of operations is not relevant. For a different group this 
        #function would have to be rewritten respecting the order. 
        for i in range(k):
            p_transporter *= torch.roll(u_mu,shifts=i*orientation,dims=1+mu)
    #x-mu
    elif orientation == 1:
        for i in range(1,k+1):
            p_transporter *= torch.roll(u_mu.conj(),shifts=i*orientation,dims=1+mu)
    return p_transporter.unsqueeze(1)    


class Linear(nn.Module):
    """
    A lattice gauge invariant linear layer convolution
    f_i(U,W) = Sum_{j,} Omega_{i,j} W_j(x)
    
    n_in and n_out should consider that w is a complex tensors

    """
    def __init__(self, n_in, n_out):
        super(Linear, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.dims = 2 #We hardcode two dimensions for the Schwinger model
        #Initialize weights Omega_{i,j,mu,k} (j,mu,k) are the in features, (i) is the out feature 
        w_in_size  = self.n_in
        w_out_size = self.n_out

        std = 1.0
        real = torch.empty(w_out_size, w_in_size, dtype=var.PREC, device=var.DEVICE)
        imag = torch.empty(w_out_size, w_in_size, dtype=var.PREC, device=var.DEVICE)
        nn.init.normal_(real, mean=0.0, std=std)
        nn.init.normal_(imag, mean=0.0, std=std)
        self.weight = nn.Parameter(torch.complex(real, imag))
        
    # f_i(W) = w_{ij} W_{j}(t,x)
    def forward(self, w):
        w = torch.einsum('ij, bjxt -> bixt', self.weight, w)
        return w


class LPTConv(nn.Module):
    """
    A lattice gauge equivariant convolution
    f_i(U,W) = Sum_{j,p} Omega_{i,j,p} T_p W_j(x),
    where p denotes a path and T_p W_j(x) is the parallel transported version of W_j(x)

    n_in and n_out should consider that u and w are complex tensors
    """
    def __init__(self, n_in, n_out, paths):
        super(LPTConv, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.dims = 2 #We hardcode two dimensions for the Schwinger model
        self.paths = paths
        #Initialize weights Omega_{i,j,mu,k} (j,mu,k) are the in features, (i) is the out feature 
        w_in_size = self.n_in  * len(self.paths)
        w_out_size = self.n_out

        std = 1.0
        real = torch.empty(w_out_size, w_in_size, dtype=var.PREC, device=var.DEVICE)
        imag = torch.empty(w_out_size, w_in_size, dtype=var.PREC, device=var.DEVICE)
        nn.init.normal_(real, mean=0.0, std=std)
        nn.init.normal_(imag, mean=0.0, std=std)
        self.weight = nn.Parameter(torch.complex(real, imag))
        #print("Weight shape",self.weight.shape)

    def forward(self, u, w):
        #We assume that the inputs are given as complex tensors
        #w.shape = (Batch,n_in,NT,NX) 
        #u.shape = (Batch,2,NT,NX) 
        transported_terms = [] #k = 0
        for path in self.paths:
            w_transport = w.clone()
            #Let us transport w
            for p in path:
                mu  = abs(p)-1
                if p > 0:
                    w_transport = torch.roll(w_transport, shifts=1, dims=2+mu) 
                elif p < 0:
                    w_transport = torch.roll(w_transport, shifts=-1, dims=2+mu)             
            pt = Tp(u,path)
            transported_terms.append(w_transport*pt)
            #pt.shape = [Batch, 1, Nt, Nx]
            #w_transport.shape = [Batch,n_in,Nt,Nx]
            #print("w_transport",w_transport.shape)
        # combine terms into a single tensor (along dim = 1 because of the batches)
        t_w = torch.cat(transported_terms, dim=1)
        #t_w.shape = [Batch,n_in*paths,Nt,Nx]
        #transported_terms.shape = [path,Batch,n_in,Nt,Nx]
        #print("t_w",t_w.shape)
        #print("transported terms",transported_terms[0].shape)
        #print("weight",self.weight.shape)
        # f_i(W) = w_{ij} TW_{j}(t,x)
        w = torch.einsum('ij, bjxt -> bixt', self.weight, t_w)
        return u, w

def Tp(u,path):
    """
    Parallel transport along an arbitrary path given as a list. For instance
    path = [-1,-2,-1,+2,+2] 1 = hat{t}, 2 = hat{x}, I don't use +0 -0 for obvious reasons
    u.shape = (Batch,2,NT,NX)
    """
    p_transporter = 1
    #For torch.roll bear in mind that
    #shift = 1  --> x-1
    #shift = -1 --> x+1
    for p in path:
        mu  = abs(p)-1
        if p > 0:
            #Hp = U^+_p(x-p)
            #x-p
            #u[:,0] = torch.roll(u[:,0],shifts=1,dims=1+mu)
            #u[:,1] = torch.roll(u[:,1],shifts=1,dims=1+mu)
            #The following line is equivalent to the two previous lines
            u = torch.roll(u, shifts=1, dims=2+mu) 
            p_transporter *= u[:,mu].conj()
        elif p < 0:
            #Hp = U_p(x)
            #x+p
            p_transporter *= u[:,mu] #We roll both components ...
            #u[:,0] = torch.roll(u[:,0],shifts=-1,dims=1+mu)
            #u[:,1] = torch.roll(u[:,1],shifts=-1,dims=1+mu)
            u = torch.roll(u, shifts=-1, dims=2+mu) 
        else:
            raise("p should be positive or negative, not zero")
    return p_transporter.unsqueeze(1)