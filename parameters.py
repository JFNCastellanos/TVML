import utils as utils
import numpy as np

def init():
    """
    Lattice parameters, gauge coupling, bare mass, number of configurations to load, lattice blocking, number of test vectors, etc.
    """
    global BETA, NX, NT, M0, M0_STRING, NO_CONFS, BLOCKS_X, BLOCKS_T, X_ELEMENTS, T_ELEMENTS, NB, NV, M0_FOLDER, N
    BETA, NX, NT= 2, 32, 32
    M0 = -0.1884 
    M0_STRING = utils.formatt(M0) #format string
    NO_CONFS = 1000 #number of confs to load
    M0_FOLDER = "m-018" #folder with confs
    BLOCKS_X, BLOCKS_T = 2, 2
    X_ELEMENTS, T_ELEMENTS = int(NX/BLOCKS_X), int(NT/BLOCKS_T) #elements per block
    NB = BLOCKS_X*BLOCKS_T #number of lattice blocks
    NV = 14    #test vectors
    N = 2*NX*NT

def print_parameters():
    print("*********** Configuration parameters ***********")
    print("* β={0}, Nx={1}, Nt={2}".format(BETA,NX,NT))
    print("* Lattice sites={0}".format(N))
    print("* m0={0}".format(np.round(M0,4)))
    print("* blocks_x={0}, blocks_t={1} (for the aggregation)".format(BLOCKS_X,BLOCKS_T))
    print("* Nv={0}".format(NV))
    print("* Number of confs={0}".format(NO_CONFS))
    print("************************************************")