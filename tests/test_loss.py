import torch
from opendataset import ConfsDataset, read_binary_conf #class for opening gauge confs
import utils
import parameters as var
import loss_function as lf
import numpy as np
import operators_torch as opt
var.init()
var.DEVICE = "cpu"

class TestLoss():
    """
    Loading the configurations and the near-kernel test vectors
    """
    def __init__(self,batchsize):
        self.criterion = lf.CustomLossTorch().to(var.DEVICE) 
        self.batch_size = batchsize
        dataset = ConfsDataset()                     
        workers    = 1
        batch_size = self.batch_size
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,          
            num_workers=workers,
            pin_memory=True
        )

        #----returns a tensor of size [ [batch_size,4,Nt,Nx], [batch_size,Nv,2,Nt,Nx], [batch_size]]----#
        #gauge_conf = [Re(U0),Re(U1),Im(U0),Im(U1)]
        #    gconf is float64 and tvectors complex128. The last entry has the indices of the confs.
        self.first_batch = next(iter(dataloader)) 
        #--------------------------------------

    
    def torch_loss_ind(self,pred, target):
        """
        For testing the loss function on individual training examples, not batches.
        """   
        ops = opt.Operators(var.BLOCKS_X, var.BLOCKS_T, pred)
        loss = torch.linalg.norm( (target - ops.P_Pdagg(target)) ) / torch.linalg.norm(target)
        return loss
    
    def testTrivialCase(self):
        loss = self.criterion(self.first_batch[1], self.first_batch[1]) 
        assert torch.abs(loss) < 1e-8, "When the interpolator is assembled with tv and is evaluated on tv, the loss should be zero, instead it is {0}".format(loss)
        print("Trivial case succesful")
    
    def testRandomCase(self):
        rand_tv = torch.rand(self.batch_size,var.NV,2,var.NT,var.NX,dtype=torch.complex128)
        loss = self.criterion(self.first_batch[1],rand_tv)
        print("Loss when interpolator is assembled with SAP vectors and evaluated on random vectors:",loss)
        loss = self.criterion(rand_tv,self.first_batch[1])
        print("Loss when interpolator is assembled with random vectors and evaluated on SAP vectors:",loss)

    def testRemainderTV(self):
        """
        We assemble P, P^+ with nv < NV SAP test vectors and evaluate the loss function on the remainder
        vectors.
        """
        loss = 0.0
        remTV = 30-var.NV
        assert remTV != 0, "NV has to be smaller than 30, otherwise there are not enough test vectors for this test"
        print("Number of test vectors used to assemble operators",var.NV)
        print("Number of test vectors used to evaluate the loss",remTV)
        print("Computing loss on remainder vectors ...")
        for confID in range(self.batch_size):
            #print("confID",confID)
            for i in range(remTV):
                path = '/wsgjsc/home/nietocastellanos1/Documents/TVML/sap/near_kernel/b{0}_{1}x{2}/{3}/tvector_{1}x{2}_b{0}0000_m{4}_nconf{5}_tv{6}.tv'.format(
                int(var.BETA),var.NX,var.NT,var.M0_FOLDER,var.M0_STRING,confID,29-i)
                tvector = torch.tensor(read_binary_conf(None,path))
                #print("Test vector ",29-i)
                loss_ind = self.torch_loss_ind(self.first_batch[1][confID],tvector)
                loss += loss_ind
                #print("Loss",loss_ind)
            #print("----------------------")
        print("Final loss on the remainder test vectors",loss/(self.batch_size*remTV))
        