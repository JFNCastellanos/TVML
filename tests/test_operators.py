import torch
import random
import operators_torch as op #Interpolator and prolongator given a set of test vectors
import utils
import parameters as var
import numpy as np
var.init()
var.DEVICE = "cpu"
#poetry run python -m tests.test_operators

class TestOperators():
    def __init__(self,nv,nx,nt,blocks_x,blocks_t):
        self.nv = nv
        self.nx = nx
        self.nt = nt
        self.n = 2 * nt * nx
        self.blocks_x = blocks_x
        self.blocks_t = blocks_t
        self.na = blocks_x*blocks_t*2
        print("Nv={0}, Nx={1}, Nt={2}, Blocks X={3}, Blocks T={4}".format(self.nv,self.nx,self.nt,self.blocks_x,self.blocks_t))
        self.operators = op.Operators.test_tv(self.nv,self.nx,self.nt,self.blocks_x,self.blocks_t,orth=False)
        self.P = torch.zeros(self.n,self.nv*self.na,dtype=torch.complex128)
        self.Pdagg = torch.zeros(self.nv*self.na,self.n,dtype=torch.complex128)
    
    def flatten_col(self,col):
        flat = torch.zeros(self.n,dtype=torch.complex128)
        for t in range(self.nt):
            for x in range(self.nx):
                for s in range(2):
                    flat[2*(x*self.nt+t)+s] = col[s,t,x] 
        return flat

    def flatten_colV2(self,col):
        flat = torch.zeros(self.nv*self.na,dtype=torch.complex128)
        for tv in range(self.nv):
            for t in range(self.blocks_t):
                for x in range(self.blocks_x):
                    for s in range(2):
                        flat[tv*2*self.blocks_x*self.blocks_t + x*2*self.blocks_t+2*t+s] = col[tv,s,t,x]    
                        #NV*(2*(x*BLOCKS_T + t) + s)+tv] = col[tv,s,t,x]
        return flat
        
    def assembleP(self):
        #Assemble P
        for tv in range(self.nv):
            for bx in range(self.blocks_x):
                for bt in range(self.blocks_t):
                    for s in range(2):
                        vc = torch.zeros((self.nv,2,self.blocks_t,self.blocks_x),dtype=torch.complex128)
                        vc[tv,s,bt,bx] = 1
                        self.P[:,tv*self.blocks_x*self.blocks_t*2 + bx*self.blocks_t*2+bt*2+s] = self.flatten_col(self.operators.P_vc(vc)) 

    def printP(self):
        for row in range(self.n):
            for col in range(self.nv*self.na):
                print(self.P[row,col].real.item(),end="       ")
            print("")

    def assemblePdagg(self):
        for x in range(self.nx):
            for t in range(self.nt):
                for s in range(2):   
                    v = torch.zeros((2,self.nt,self.nx),dtype=torch.complex128)
                    v[s,t,x] = 1
                    self.Pdagg[:,2*(x*self.nt+t)+s] = self.flatten_colV2(self.operators.Pdagg_v(v))
    def printPdagg(self):
         for row in range(self.nv*self.na):
            for col in range(self.n):
                print(self.Pdagg[row,col].real.item(),end="       ")
            print("")

    def testP_Pdagg(self):
        Pconj_trans =  torch.conj(torch.transpose(self.P,0,1))
        if torch.all(self.Pdagg == Pconj_trans):
            print("P*^T = Pdagg, i.e. all good.")
        else:
            print("Conjugate transpose of P does not coincide with Pdagg")

    def testOrthonormalization(self):
        self.operators.tv_orth()
        self.operators.check_orth()