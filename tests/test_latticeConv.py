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
        self.data = self.first_batch[self.confID].to(var.DEVICE) #[Re(U0),Re(U1),Im(U0),Im(U1), Re(W),Im(W)], where W'(x) = Omega(x) W(x) Omega^+(x)
        print(len(self.data))
        self.conf = self.data[:,0:4]
        self.w = self.data[:,4:]


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
                print(np.round(w[0,0,t-1,x].item(),3),"+i",np.round(w[0,1,t-1,x].item(),3),end="  ")
            print()
        print()
        for x in range(5):
            for t in range(5):
                print(np.round(wkxmu[0,0,t,x].item(),3),"+i",np.round(wkxmu[0,1,t,x].item(),3),end="  ")
            print()
        print("Transporting W(x) to W(x-k hat{1}) \n")
        k, mu, orientation = 1, 1, 1
        wkxmu = torch.roll(w,shifts=k*orientation,dims=2+mu)   #w[(t,x)- k * orientation *hat{mu}]
        for x in range(5):
            for t in range(5):
                print(np.round(w[0,0,t,x-1].item(),3),"+i",np.round(w[0,1,t,x-1].item(),3),end="  ")
            print()
        print()
        for x in range(5):
            for t in range(5):
                print(np.round(wkxmu[0,0,t,x].item(),3),"+i",np.round(wkxmu[0,1,t,x].item(),3),end="  ")
            print()
        
    def LConv_test(self):
        kernel_size = 3
        n_out = 4
        n_in = 2
        print("Gauge equivariant convolution, in_features={0}, out_features={1}, ksize={2}".format(n_in,n_out,kernel_size))
        lconv = ge.LConv( n_in, n_out,kernel_size)
        out = lconv(self.w)
        print("Input shape",self.w.shape)
        print("Output shape",out.shape)

        
