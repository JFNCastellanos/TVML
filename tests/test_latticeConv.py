import torch
from opendataset import ConfsDataset, read_binary_plaquette #class for opening gauge confs
import utils
import parameters as var
import numpy as np
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