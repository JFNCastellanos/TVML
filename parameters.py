import utils as utils
import numpy as np
import torch

def init():
    """
    Lattice parameters, gauge coupling, bare mass, number of configurations to load, lattice blocking, number of test vectors, etc.
    """
    global BETA, NX, NT, M0, M0_STRING, NO_CONFS, BLOCKS_X, BLOCKS_T, X_ELEMENTS, T_ELEMENTS, NB, NV, M0_FOLDER, N, NGPU, DEVICE, TRAIN_PROP
    global TRAIN_LEN, TEST_LEN, NV_PRED
    BETA, NX, NT= 2, 32, 32
    M0 = -0.18840579710144945 
    M0_STRING = utils.formatt(M0) #format string
    NO_CONFS = 1000 #number of confs to load
    M0_FOLDER = "m-018" #folder with confs
    BLOCKS_X, BLOCKS_T = 2, 2
    X_ELEMENTS, T_ELEMENTS = int(NX/BLOCKS_X), int(NT/BLOCKS_T) #elements per block
    NB = BLOCKS_X*BLOCKS_T #number of lattice blocks
    NV = 30   #SAP test vectors used in the loss function 
    NV_PRED = 5 #Number of test vectors to predict (NV_PRED < NV unless I have many training examples)
    N = 2*NX*NT
    NGPU = 1
    DEVICE = torch.device("cuda:0" if (torch.cuda.is_available() and NGPU > 0) else "cpu")
    TRAIN_PROP = 0.9 #Proportion of total examples used for training
    TRAIN_LEN = int(NO_CONFS*0.9)
    TEST_LEN = NO_CONFS - TRAIN_LEN 
    
def print_parameters():
    print("*********** Configuration parameters ***********")
    print("* β={0}, Nx={1}, Nt={2}".format(BETA,NX,NT))
    print("* Variables={0}".format(N))
    print("* m0={0}".format(np.round(M0,4)))
    print("* blocks_x={0}, blocks_t={1} (for the aggregation)".format(BLOCKS_X,BLOCKS_T))
    print("* SAP vectors for the loss function Nv={0}".format(NV))
    print("* Fake test vectors generated Nv={0}".format(NV_PRED))
    print("* Number of confs={0}".format(NO_CONFS))
    print("* Confs used for training={0}".format(int(TRAIN_PROP*NO_CONFS)))  
    print("************************************************")