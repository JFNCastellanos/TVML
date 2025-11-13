import numpy as np
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
    """
    def __init__(self, blocks_x,blocks_t,test_vectors):
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
        self.tv_orth() #Orthonormalizes the set of test vectors
        #after orthonormalization, every test vector's norm is (globally) sqrt(2*number_of_lattice_blocks)

    def getTestVectors(self):
        return self.test_vectors

    """
    In case the test vectors are not provided we generate them randomly
    """
    @classmethod
    def rand_tv(cls,blocks_x,blocks_t):
        test_vectors = np.zeros((var.NV,2,var.NT,var.NX),dtype=complex)
        #random.seed(0)
        for tv in range(var.NV):
            for nt in range(var.NT):
                for nx in range(var.NX):
                    for s in range(2):
                        x = random.random()
                        y = random.random()
                        z = complex(x, y)
                        test_vectors[tv,s,nt,nx] = z   #2*(nx*NT + nt) + s + 1 + tv*2*NX*NT
        return cls(blocks_x,blocks_t, test_vectors)

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
        v = np.zeros((2,self.nt,self.nx),dtype=complex)
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
            v[:, tini:tfin, xini:xfin] = np.sum(
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
        vc = np.zeros((self.nv,2,self.blocks_t,self.blocks_x),dtype=complex)
        for block in range(self.nb):
            block_x = block // self.blocks_x
            block_t = block % self.blocks_t
            #----Coordinates of elements inside block----#
            xini, tini = self.x_elements * block_x, self.t_elements * block_t
            xfin = xini + self.x_elements
            tfin = tini + self.t_elements 
            #--------------------------------------------#
            #vc[cc,s,block_t,block_x] = np.sum(np.conj(test_vectors[cc,s,tini:tfin, xini:xfin]) * v[s,tini:tfin, xini:xfin],axis=(-1,-2))
            vc[:, :, block_t, block_x] = np.sum(
                np.conj(self.test_vectors[:, :, tini:tfin, xini:xfin]) * v[:, tini:tfin, xini:xfin],axis=(-1, -2)
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
        vc = np.zeros((self.nv,2,self.blocks_t,self.blocks_x),dtype=complex)
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
        v = np.zeros((2,self.nt,self.nx),dtype=complex)
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
        result = np.copy(self.test_vectors)
        for block in range(self.nb):
            block_x = block // self.blocks_x
            block_t = block % self.blocks_t
            #----Coordinates of elements inside block----#
            xini, tini = self.x_elements * block_x, self.t_elements * block_t
            xfin = xini + self.x_elements
            tfin = tini + self.t_elements 
            #--------------------------------------------#
            for s in range(2):
                # Extract block for all Nv vectors: shape (Nv, T_ELEMENTS, X_ELEMENTS)
                block_vectors = result[:, s, tini:tfin, xini:xfin] 
                # Flatten spatial dimensions
                flat = block_vectors.reshape(self.nv, -1).T  # shape (T_ELEMENTS*X_ELEMENTS, Nv) for a fixed s
                # QR orthonormalization (matrix Q has the orthonormal vectors in columns)
                q, _ = np.linalg.qr(flat)
                # Reshape back and assign
                result[:, s, tini:tfin, xini:xfin] = q.T.reshape(self.nv, self.t_elements, self.x_elements)
        self.test_vectors = result #check whether I actually need results array
        #Equivalent Gram-Schmidt orthonormalization
            #for s in range(2):
            #    for tv in range(NV):
            #        for tvv in range(tv):
                        #projection over same aggregate
                        #np.vdot(a,b) = a^+ . b
            #            proj = np.vdot(test_vectors[tvv,s,tini:tfin, xini:xfin],test_vectors[tv,s,tini:tfin, xini:xfin])
            #            test_vectors[tv,s,tini:tfin, xini:xfin] -= proj * test_vectors[tvv,s,tini:tfin, xini:xfin]
        			#normalize the test vectors tv on aggregate block x {s}
            #        norm = np.linalg.norm(test_vectors[tv,s,tini:tfin, xini:xfin])
            #        test_vectors[tv,s,tini:tfin, xini:xfin] /= norm
        #return test_vectors
	
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
                        dot_prod = np.vdot(self.test_vectors[tv,s,tini:tfin, xini:xfin], self.test_vectors[tvv,s,tini:tfin, xini:xfin])
                        if np.abs(dot_prod) > 1e-8 and tv != tvv:
                            print("Block",block,"spin",s)
                            print("Test vectors {0} and {1} are not orthogonal".format(tv,tvv),dot_prod)
                            return 
                        elif np.abs(dot_prod-1.0) > 1e-8 and tv == tvv:
                            print("Test vectors {0} not orthonormalized".format(tv,tv),dot_prod)
                            return
        print("Test vectors are locally orthonormalized")

"""
Equivalent implementations of P and P^+ that don't rely on numpy (and therefore are probably slower)
def P_vc_equivalent(test_vectors,vc): 
    v = np.zeros((2,NT,NX),dtype=complex)
    for cc in range(NV):
        for block in range(NB):
            block_x = int(block / BLOCKS_T)
            block_t = int(block % BLOCKS_T) 
            for tb in range(T_ELEMENTS):
                for xb in range(X_ELEMENTS):
                    #given (tb,xb) in block, the values of (t,x) in V are
                    x = X_ELEMENTS * block_x + xb
                    t = T_ELEMENTS * block_t + tb
                    v[:,t,x] += test_vectors[cc,:,t,x] * vc[cc,:,block_t,block_x]			
    return v

def Pdagg_v_equivalent(test_vectors,v):
    vc = np.zeros((nv,2,BLOCKS_T,BLOCKS_X),dtype=complex)
    for block in range(NB):
        block_x = int(block / BLOCKS_T)
        block_t = int(block % BLOCKS_T) 
        for cc in range(NV):
            for tb in range(T_ELEMENTS):         
                for xb in range(X_ELEMENTS):      
                    #given (tb,xb) in block, the values of (t,x) in V are
                    x = X_ELEMENTS * block_x + xb
                    t = T_ELEMENTS * block_t + tb
                    for s in range(2):
                        vc[cc,s,block_t,block_x] += np.conj(test_vectors[cc,s,t,x]) * v[s,t,x]	
"""