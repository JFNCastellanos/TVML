import torch
import torch.nn as nn
import parameters as var
import gauge_equivariant as ge
var.init()

#Generator
#PCT layers that yield a 2-component test vector
#-------------------------------------------#
lcnn_layers_gen = var.MultiInputSequential(
            ge.LConv( 1, 2, 6),      
            ge.LConv( 2, 4, 6),      
            ge.LConv( 4, 8, 6),    
            ge.LConv( 8, 16,6),
            ge.LConv( 16, 18,6),
            ge.LConv( 18, 2, 6),
)


paths_gen = [[1,1],[1,2],[1,-2],
         [-1,-1],[-1,2],[-1,-2],
         [2,1],[2,-1],[2,2],
         [-2,1],[-2,-1],[-2,-2],
        [1],[-1],[2],[-2],
        [1,1,1], [1,1,2], [1,1,-2],
        [1,2,1], [1,2,-1], [1,2,2]]

ptc_layers_gen = var.MultiInputSequential(
            ge.LPTConv( 1, 16,paths_gen),
            ge.LPTConv( 16, 2,paths_gen)
            #ge.LPTConv( 2, 4,paths),
            #ge.LPTConv( 4, 8,paths),
            #ge.LPTConv( 8, 16,paths),
            #ge.LPTConv( 16, 18,paths),
            #ge.LPTConv( 18, 2*var.NV_PRED,paths),
            )

class Generator(nn.Module):
    """
    U is a gauge configuration
    W is random noise 
    """
    def __init__(self, ngpu,batch_size):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.lcnn_layers = ptc_layers_gen
        self.batch_size = batch_size
    def forward(self, u, w):
        u, w = self.lcnn_layers(u,w)
        w = w.squeeze() #We remove the trivial dimensions
        return w.view(self.batch_size,2,var.NT,var.NX)    

class Discriminator(nn.Module):
    """
    U is a gauge configuration              (B,2,Nt,Nx)
    W is random noise or a smooth test vector (B,2,Nt,Nx)
    """
    def __init__(self, ngpu,batch_size):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.batch_size = batch_size
        self.sigmoid = nn.Sigmoid()
        self.lcnn_layers = var.MultiInputSequential(
            ge.LConv( 2, 2, 6),      
            ge.LConv( 2, 4, 6),      
            ge.LConv( 4, 8, 6),    
            ge.LConv( 8, 16,6),
            ge.LConv( 16, 18,6),
            ge.LConv( 18, 1, 6),
        )
    def forward(self, u, w):
        u, w = self.lcnn_layers(u,w)
        w = w.squeeze() #We remove the trivial dimensions
        #Build a two-component tensor with the real and imaginary part
        #stack them and apply sigmoid.
        real = torch.real(w)   # (B,NT,NX) (real number)
        imag = torch.imag(w)   # (B,NT,NX) (real number)
        w = torch.stack([real, imag], dim=1)
        w = self.sigmoid(w)    # (B,2,NT,NX)
        return w 
