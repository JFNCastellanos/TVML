import sys
sys.path.append("../")  # Goes up to "project/"
import parameters as var
import data_loader as dl
import model_gan as mg
import torch
import torch.nn as nn
import numpy as np
var.init()
var.print_parameters()

class Test():
    def __init__(self,batchsize):
        self.batch_size = batchsize
        self.confID = 0 
        self.num_examples = 500
        dataset = dl.ConfsDataset(self.confID,self.num_examples)                     
        workers = 1
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,          
            num_workers=workers,
            pin_memory=True
        )
      
        self.first_batch = next(iter(dataloader))
        self.U, self.Tv, self.IDX = self.first_batch
        self.U = self.U.to(var.DEVICE)
        self.Tv = self.Tv.to(var.DEVICE)
        self.IDX = self.IDX.to(var.DEVICE)
        #U.shape = (B,2,Nt,Nx) (already complex)
        #Tv.shape = (B,2,Nt,Nx)
        
        print("U shape",self.U.shape)
        print("Type, device",self.U.dtype,self.U.device)
        print("Tv shape",self.Tv.shape)
        print("Type, device",self.Tv.dtype,self.Tv.device)

    def check_u1_vars(self):
        for ID in range(len(self.U)):
            for x in range(var.NX):
                for t in range(var.NT):
                    for mu in range(2):
                        assert np.abs( np.abs(self.U[ID,mu,t,x].item()) - 1) < 1e-5, "U of conf={0}, mu={1}, x={2}, t={3} is not a U(1) variable".format(ID,mu,x,t)

    def test_G_and_D(self):
        D = mg.Discriminator(var.NGPU,self.batch_size).to(var.DEVICE)
        std = 1.0
        n_in = 2
        Dw = D(self.U,self.Tv)
        #Tv.shape = (Batch,1,NT,NX) 
        #u.shape = (Batch,2,NT,NX) 
        print("Discriminator shape ",Dw.shape)
        print("Discriminator dtype ",Dw.dtype)
        
        G = mg.Generator(var.NGPU,self.batch_size).to(var.DEVICE)
        real = torch.empty(self.batch_size, 1, var.NT, var.NX, dtype=var.PREC, device=var.DEVICE)
        imag = torch.empty(self.batch_size, 1, var.NT, var.NX, dtype=var.PREC, device=var.DEVICE)
        nn.init.normal_(real, mean=0.0, std=std)
        nn.init.normal_(imag, mean=0.0, std=std)
        random_noise = torch.complex(real, imag)
        Gw = G(self.U,random_noise)
        print("Generator with random noise shape",Gw.shape)
        print("Dtype",Gw.dtype)
        DGw = D(self.U,Gw)
        print("Discriminator with generator D(G(w)) shape",DGw.shape)
        print("Dtype",DGw.dtype)