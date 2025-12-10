import numpy as np
import torch
import parameters as var
import random

class Operators():
    """
    Receives: 
        * set of COMPLEX test vectors [Nv,2,Nt,Nx] 
        * blocks_x, blocks_t: number of lattice blocks on each direction
    The class performs the operations 
        * P vc
        * P^+ v
        * P P^+ vc
        * P^+ P v (this is the identitiy operation, but it is good to cross-check
    The test vectors are orthonormalized when an instance is created.
    To check that orthonormalziation was done right call check_orth()
    The implementations rely on torch 
    """
    def __init__(self, blocks_x,blocks_t,test_vectors,orth=True):
        """
        blocks_x, blocks_t: lattice blocking
        test_vectors.shape [Nv,2,Nx,Nt]
        """
        self.test_vectors = test_vectors
        self.nv, _, self.nx, self.nt = self.test_vectors.shape  
        self.blocks_x = blocks_x
        self.blocks_t = blocks_t
        self.nb = self.blocks_x*self.blocks_t
        self.x_elements = self.nx // self.blocks_x
        self.t_elements = self.nt // self.blocks_t
        if orth == True:
            self.tv_orth() #Orthonormalizes the set of test vectors
        else:
            print("Test vectors not orthonormalized")
        #after orthonormalization, every test vector's norm is (globally) sqrt(2*number_of_lattice_blocks)
        self.device = var.DEVICE

    def getTestVectors(self):
        return self.test_vectors

    """
    In case the test vectors are not provided we generate them randomly
    """
    @classmethod
    def rand_tv(cls,nv,nx,nt,blocks_x,blocks_t,orth=True):
        test_vectors = torch.zeros(nv,2,nt,nx,dtype=var.PREC_COMPLEX,device = var.DEVICE)
        #random.seed(0)
        for tv in range(nv):
            for t in range(nt):
                for x in range(nx):
                    for s in range(2):
                        x = random.random()
                        y = random.random()
                        z = complex(x, y)
                        test_vectors[tv,s,t,x] = z   #2*(nx*NT + nt) + s + 1 + tv*2*NX*NT
        return cls(blocks_x,blocks_t, test_vectors,orth)

    @classmethod
    def test_tv(cls,nv,nx,nt,blocks_x,blocks_t,orth=True):
        test_vectors = torch.zeros(nv,2,nt,nx,dtype=var.PREC_COMPLEX,device = var.DEVICE)
        for tv in range(nv):
            for t in range(nt):
                for x in range(nx):
                    for s in range(2):
                        test_vectors[tv,s,t,x] = 2*(x*nt + t) + s + 1 + tv*2*nx*nt
        return cls(blocks_x,blocks_t, test_vectors,orth)
    

    def P_vc(self,vc):
        """
        Interpolator P = (v1^(1)|v2^(1)|...|v_Nv^(1)|...|v1^(Nagg)|...|v_Nv^(Nagg)) times a vector on the coarse grid
        [vi^(a)]_k = [vi]_k if k in Agg(a), else zero 
        Receives:
            * vector on the coarse grid vc.shape = [Nv,2,BlocksT,BlocksX] (test vectors, spin, blockT, blockX)
        Returns:
            * vector on the fine grid v.shape = [2,Nt,Nx] 
        0_ _ _ _t_ _ _ _ Nt
        |       |       |
        |block 0| block1|
      x |_ _ _ _|_ _ _ _|
        |       |       |
        |block 2| block3|
        |_ _ _ _|_ _ _ _| 
        Nx 
    
        block: vectorized lattice block index
        block_t: block index on t-direction
        block_x: block index on x-direction
        """   
        v = torch.zeros(2,self.nt,self.nx,dtype=var.PREC_COMPLEX,device = self.device)
        for block in range(self.nb):
            block_x = block // self.blocks_x
            block_t = block % self.blocks_t 
            #----Coordinates of elements inside block----#
            xini, tini = self.x_elements * block_x, self.t_elements * block_t
            xfin = xini + self.x_elements
            tfin = tini + self.t_elements
            #--------------------------------------------#

            # broadcast the (batch, channel) mask onto the (t, x) slice
            #   vc[:, :, block_t, block_x]                          → shape (B, C)
            #   vc[:, :, block_t, block_x][:, :, None, None]        → shape (B, C, 1, 1)
            #   test_vectors[:, :, tini:tfin, xini:xfin]            → shape (B, C, T, X)
            v[:, tini:tfin, xini:xfin] = torch.sum(
            self.test_vectors[:, :, tini:tfin, xini:xfin] * vc[:, :, block_t, block_x][:, :, None, None],
            axis=0
            )
            #Equivalent to this
            #v[0,tini:tfin,xini:xfin] = np.sum(test_vectors[:,0,tini:tfin,xini:xfin] * vc[:,0,block_t,block_x],axis=0)
            #v[1,tini:tfin,xini:xfin] = np.sum(test_vectors[:,1,tini:tfin,xini:xfin] * vc[:,1,block_t,block_x],axis=0)	   
        return v

    def Pdagg_v(self,v):
        """
        Restriction operator P^+ = (v1^(1)|v2^(1)|...|v_Nv^(1)|...|v1^(Nagg)|...|v_Nv^(Nagg))^+ times a vector on the fine grid
        Receives:
            * vector on the fine grid v.shape = [2,Nt,Nx] 
        Returns:
            * vector on the coarse grid vc.shape = [Nv,2,BlocksT,BlocksX]
        0_ _ _ _t_ _ _ _ Nt
        |       |       |
        |block 0| block1|
      x |_ _ _ _|_ _ _ _|
        |       |       |
        |block 2| block3|
        |_ _ _ _|_ _ _ _| 
        Nx 
    
        block: vectorized lattice block index
        block_t: block index on t-direction
        block_x: block index on x-direction
        """   
        vc = torch.zeros(self.nv,2,self.blocks_t,self.blocks_x,dtype=var.PREC_COMPLEX,device = self.device)
        for block in range(self.nb):
            block_x = block // self.blocks_x
            block_t = block % self.blocks_t
            #----Coordinates of elements inside block----#
            xini, tini = self.x_elements * block_x, self.t_elements * block_t
            xfin = xini + self.x_elements
            tfin = tini + self.t_elements 
            #--------------------------------------------#
            #vc[cc,s,block_t,block_x] = np.sum(np.conj(test_vectors[cc,s,tini:tfin, xini:xfin]) * v[s,tini:tfin, xini:xfin],axis=(-1,-2))
            vc[:, :, block_t, block_x] = torch.sum(
                torch.conj(self.test_vectors[:, :, tini:tfin, xini:xfin]) * v[:, tini:tfin, xini:xfin],axis=(-1, -2)
            )
        return vc

    def P_Pdagg(self,v):
        """
        Applies PP^+ v to a fine grid vector
        Receives:
            * vector on the fine grid v.shape = [2,BlocksT,BlocksX] 
        Returns:
            * vector on the coarse grid
        """
        vc = torch.zeros(self.nv,2,self.blocks_t,self.blocks_x,dtype=var.PREC_COMPLEX,device = self.device)
        temp = self.Pdagg_v(v)
        return self.P_vc(temp)

    def Pdagg_P(self,vc):
        """
        Applies P^+P vc to a coarse grid vector
        Receives:
            * vector on the coarse grid vc.shape = [Nv,2,BlocksT,BlocksX] (test vectors, spin, blockT, blockX)
        Returns:
            * vector on the fine grid
        NOTE: Given the definition of the interpolator, this has to return vc, i.e. it acts as the identity. 
        """
        v = torch.zeros(2,self.nt,self.nx,dtype=var.PREC_COMPLEX,device = self.device)
        temp = self.P_vc(vc)
        return self.Pdagg_v(temp)

    def tv_orth(self):
        """
        Given a set of test vectors, returns a local orthonormalization.
        This means that for the set of test vectors (v1^(1)|v2^(1)|...|v_Nv^(1)|...|v1^(Nagg)|...|v_Nv^(Nagg)), we
        orthonormalize the sets {v1^{a}, ..., vn^(a)} for each aggregate.
        This follows the steps from Section 3.1 of A. Frommer et al "Adaptive Aggregation-Based Domain Decomposition 
    	Multigrid for the Lattice Wilson-Dirac Operator", SIAM, 36 (2014).
        Receives:
            * Test vectors test_vectors.shape = [Nv,2,Nt,Nx] 
        Returns: 
            * Orthonormalized test_vectors.shape = [Nv,2,Nt,Nx]
        """
        result = torch.clone(self.test_vectors)
        for block in range(self.nb):
            block_x = block // self.blocks_x
            block_t = block % self.blocks_t
            #----Coordinates of elements inside block----#
            xini, tini = self.x_elements * block_x, self.t_elements * block_t
            xfin = xini + self.x_elements
            tfin = tini + self.t_elements 
            #--------------------------------------------#
            for s in range(2):
                # shape: [Nv, t_elements, x_elements]
                block_tensor = result[:, s, tini:tfin, xini:xfin]

                # Flatten the spatial dimensions → matrix A of shape
                # [t_elements * x_elements, Nv]
                # (the QR routine expects the *columns* to be the vectors we want to
                # orthonormalise, hence the transpose)
                A = block_tensor.reshape(self.nv, -1).T   # shape (M, Nv) where M = t_elements*x_elements

                # ---------------------------------------------------------
                # 3) QR decomposition (torch.linalg.qr works on any device)
                # ---------------------------------------------------------
                # We only need the orthonormal factor Q; the upper‑triangular R is
                # discarded.  Using `mode='reduced'` returns Q with shape (M, Nv)
                # which is exactly what we need.
                Q, _ = torch.linalg.qr(A, mode='reduced')   # Q: (M, Nv)

                # ---------------------------------------------------------
                # 4) Write the orthonormalised block back
                # ---------------------------------------------------------
                # Qᵀ has shape (Nv, M) → reshape to the original block shape
                block_tensor.copy_(Q.T.reshape(self.nv,
                                               self.t_elements,
                                               self.x_elements))
        self.test_vectors = result #check whether I actually need results array
	
    def check_orth(self):
        """
        Checks local orthogonality of the test vectors
        """
        for block in range(self.nb):
            block_x = block // self.blocks_x
            block_t = block % self.blocks_t
            #----Coordinates of elements inside block----#
            xini, tini = self.x_elements * block_x, self.t_elements * block_t
            xfin = xini + self.x_elements
            tfin = tini + self.t_elements 
            #--------------------------------------------#
            for s in range(2):
                for tv in range(self.nv):
                    for tvv in range(self.nv):
                        #np.vdot(a,b) = a^+ . b
                        dot_prod = torch.vdot(self.test_vectors[tv,s,tini:tfin, xini:xfin].flatten(), 
                                              self.test_vectors[tvv,s,tini:tfin, xini:xfin].flatten())
                        if torch.abs(dot_prod) > 1e-5 and tv != tvv:
                            print("Block",block,"spin",s)
                            print("Test vectors {0} and {1} are not orthogonal".format(tv,tvv),dot_prod)
                            return 
                        elif torch.abs(dot_prod-1.0) > 1e-5 and tv == tvv:
                            print("Test vectors {0} not orthonormalized".format(tv,tv),dot_prod)
                            return
        print("Test vectors are locally orthonormalized")
