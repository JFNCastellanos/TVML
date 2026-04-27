import torch
from opendataset import ConfsDataset, read_binary_plaquette #class for opening gauge confs
import utils
import parameters as var
import numpy as np
import gauge_equivariant as ge
var.init()


class TestLCNN():
    """
    Loading the configurations and the near-kernel test vectors
    """
    def __init__(self,batchsize):
        self.batch_size = batchsize
        dataset = ConfsDataset()                     
        workers    = 1
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,          
            num_workers=workers,
            pin_memory=True
        )

        #----returns a list  [ [batch_size,6,Nt,Nx], [batch_size,Nv,2,Nt,Nx], [batch_size]]----#
        #gauge_conf = [Re(U0),Re(U1),Im(U0),Im(U1)]
        #    gconf is float64 and tvectors complex128. The last entry has the indices of the confs.
        self.first_batch = next(iter(dataloader))
        self.confID = 0 
        self.data = self.first_batch[0].to(var.DEVICE) #[Re(U0),Re(U1),Im(U0),Im(U1), Re(W),Im(W)], where W'(x) = Omega(x) W(x) Omega^+(x)
        self.local_trans_obj = self.data.shape[1]-4
        print("Number of local transforming real variables",self.local_trans_obj)
        conf = self.data[:,0:4]
        local_t_object = self.data[:,4:]
        

        # Build a complex tensor of shape (B,  2, NT, NX)
        real = torch.stack([conf[:,0], conf[:,1]], dim=1)   # (B,2,NT,NX) (real number)
        imag = torch.stack([conf[:,2], conf[:,3]], dim=1)   # (B,2,NT,NX) (real number)
        self.u = torch.complex(real, imag)                  # (B,2,NT,NX) (complex number)

        if self.local_trans_obj == 2:
            real = local_t_object[:,0].unsqueeze(1)          #(B,1,NT,NX)
            imag = local_t_object[:,1].unsqueeze(1)          #(B,1,NT,NX)
        #else:
            #We have to stack them. I leave the line just in case I add more things, like Polyakov loops.
            

        self.w = torch.complex(real,imag)
        print("U shape",self.u.shape)
        print("Type",self.u.dtype)
        print("W shape",self.w.shape)
        print("Type",self.w.dtype)

    def check_u1_vars(self):
        for ID in range(len(self.u)):
            for x in range(var.NX):
                for t in range(var.NT):
                    for mu in range(2):
                        assert np.abs( np.abs(self.u[ID,mu,t,x].item()) - 1) < 1e-5, "U of conf={0}, mu={1}, x={2}, t={3} is not a U(1) variable".format(ID,mu,x,t)
        print("Variables are all elements of U(1)")

    def check_parallel_transport(self):
        k, mu, orientation = 1, 0, 1
        wkxmu = torch.roll(self.w,shifts=k*orientation,dims=mu+2)   #w[(t,x)-k * orientation *hat{mu}]

        #assert self.w[0,0,0] == wkxmu[0,k*mu,0], "W(0,0) = {0} and W(k mu,0) = {1}".format(self.w[0,0,0],wkxmu[0,k*mu,0])
        print("W(-1,0)={0}, Wkxmu(0,0)={1}".format(self.w[self.confID,0,-1,0],wkxmu[self.confID,0,0,0]))
        print("W(-1,0)={0}, Wkxmu(0,0)={1}".format(self.w[self.confID,0,-1,0],wkxmu[self.confID,0,0,0]))
        print("W(-1,0)={0}, Wkxmu(0,0)={1}".format(self.w[self.confID,0,-1,0],wkxmu[self.confID,0,0,0]))
        print("W(-1,0)={0}, Wkxmu(0,0)={1}".format(self.w[self.confID,0,-1,0],wkxmu[self.confID,0,0,0]))
        print("")
        k, mu, orientation = 1, 1, 1
        wkxmu = torch.roll(self.w,shifts=k*orientation,dims=mu+2)   #w[(t,x)-k * orientation *hat{mu}]
        print("W(0,-1)={0}, Wkxmu(0,0)={1}".format(self.w[self.confID,0,0,-1],wkxmu[self.confID,0,0,0]))
        print("W(0,-1)={0}, Wkxmu(0,0)={1}".format(self.w[self.confID,0,0,-1],wkxmu[self.confID,0,0,0]))
        print("W(0,-1)={0}, Wkxmu(0,0)={1}".format(self.w[self.confID,0,0,-1],wkxmu[self.confID,0,0,0]))
        print("W(0,-1)={0}, Wkxmu(0,0)={1}".format(self.w[self.confID,0,0,-1],wkxmu[self.confID,0,0,0]))
        return wkxmu

    def print_parallel_transport(self):
        w = self.w
        print("Transporting W(x) to W(x-k hat{0})\n")
        k, mu, orientation = 1, 0, +1
        wkxmu = torch.roll(w,shifts=k*orientation,dims=2+mu)   #w[(t,x)- k * orientation *hat{mu}]
        for x in range(5):
            for t in range(5):
                print(np.round(w[0,0,t-1,x].item(),3),end="  ")
            print()
        print()
        for x in range(5):
            for t in range(5):
                print(np.round(wkxmu[0,0,t,x].item(),3),end="  ")
            print()
        print("Transporting W(x) to W(x-k hat{1}) \n")
        k, mu, orientation = 1, 1, 1
        wkxmu = torch.roll(w,shifts=k*orientation,dims=2+mu)   #w[(t,x)- k * orientation *hat{mu}]
        for x in range(5):
            for t in range(5):
                print(np.round(w[0,0,t,x-1].item(),3),end="  ")
            print()
        print()
        for x in range(5):
            for t in range(5):
                print(np.round(wkxmu[0,0,t,x].item(),3),end="  ")
            print()
        
    def LConv_test(self):
        kernel_size = 3
        n_out = 4
        n_in = 1
        print("Gauge equivariant convolution, in_features={0}, out_features={1}, ksize={2}".format(n_in,n_out,kernel_size))
        lconv = ge.LConv( n_in, n_out,kernel_size)
        out = lconv(self.u,self.w)
        print("Input shape",self.w.shape)
        print("Output shape",out[1].shape)
        return out


def test_LCNN_layers():
    batch_size = 100
    w = torch.rand((batch_size,1,var.NT,var.NX),dtype=var.PREC_COMPLEX).to(var.DEVICE) 
    u = torch.rand((batch_size,2,var.NT,var.NX),dtype=var.PREC_COMPLEX).to(var.DEVICE)
    #gauge equivariant convolutional layers
    print("Input shape",w.shape)
    lcnn_layers = var.MultiInputSequential(
            ge.LConv( 1, 4,3),
            ge.LConv( 4, 16,3),
            ge.LConv( 16,2*var.NV_PRED ,3),
            #torch.nn.PReLU(4*var.NV_PRED,dtype=var.PREC,device=var.DEVICE)
    )
    u, w = lcnn_layers(u,w)
    print("W shape after convolution",w.shape)
    print("U shape after convolution",u.shape)
    #gauge equivariant convolutional layers

    
    
    #w = w.flatten(start_dim=1)
    #print("Output shape after flatten",w.shape)
    #ge_linear_layers = torch.nn.Sequential(
    #        torch.nn.Linear(w.shape[1], 4*var.NV_PRED*var.NT*var.NX,bias=False,device=var.DEVICE,dtype=var.PREC),
    #)
    #w = ge_linear_layers(w)
    #print("Output shape after linear layer",w.shape)