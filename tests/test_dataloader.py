import torch
from opendataset import ConfsDataset #class for opening gauge confs
import utils
import parameters as var
var.init()
var.DEVICE = "cpu"


class TestDataloader():
    """
    Loading the configurations and the near-kernel test vectors
    """
    def __init__(self,batchsize):
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
    def testDimensions(self):
        assert len(self.first_batch) == 3, "The batch should have three objects (gconfs, tvectors,confID)"
        batch_size, no_real, nt, nx = self.first_batch[0].shape
        assert batch_size == self.batch_size, "Issue with the batch size for gconfs"
        assert no_real == 4, "gconfs[1] should be of dim 4"
        assert nt == var.NT, "gconfs[2] should match Nt"
        assert nx == var.NX, "gconfs[2] should match Nx"
        batch_size, nv, spins, nt, nx = self.first_batch[1].shape
        assert batch_size == self.batch_size, "Issue with the batch size for tvectors"
        assert nv == var.NV, "tvectors[1] should match Nv"
        assert spins == 2, "tvectors[2] should be 2 (spin components)"
        assert nt == var.NT, "tvectors[3] should match Nt"
        assert nx == var.NX, "tvectors[4] should match Nx"
        print("gconfs and tvectors dimensions correct")
    def testDataType(self):
        print("dtype of gconfs",self.first_batch[0].dtype)
        print("dtype of tvectors",self.first_batch[1].dtype)
    def testConfZero(self):
        confID = 0
        print("Check that conf {0} is correctly read. Compare with the actual numbers from the conf file.".format(confID))
        conf = self.first_batch[0][confID]
        assert self.first_batch[2][confID] == confID, "First conf should be ID {0}".format(confID)
        print("- - - Uμ(t,x) - - -")
        nx, nt, mu = 0, 0, 0
        val = conf[mu,nt,nx] + 1j*conf[mu+2,nt,nx,]
        print("U{0}({1},{2})=".format(mu,nt,nx),val.item(),val.dtype)
        nx, nt, mu = 0, 0, 1
        val = conf[mu,nt,nx] + 1j*conf[mu+2,nt,nx]
        print("U{0}({1},{2})=".format(mu,nt,nx),val.item(),val.dtype)
        nx, nt, mu = 0, 1, 0
        val = conf[mu,nt,nx] + 1j*conf[mu+2,nt,nx]
        print("U{0}({1},{2})=".format(mu,nt,nx),val.item(),val.dtype)
        nx, nt, mu = 0, 1, 1
        val = conf[mu,nt,nx] + 1j*conf[mu+2,nt,nx]
        print("U{0}({1},{2})=".format(mu,nt,nx),val.item(),val.dtype)
        nx, nt, mu = 10, 15, 1
        val = conf[mu,nt,nx] + 1j*conf[mu+2,nt,nx]
        print("U{0}({1},{2})=".format(mu,nt,nx),val.item(),val.dtype)
        
    def testTVector(self):
        print("Check that test vector 0 from conf 0 coincides with the info in the file")
        confID = 0
        nv = 0
        tvector = self.first_batch[1][confID][nv]
        assert self.first_batch[2][confID] == confID, "First conf should be ID {0}".format(confID)
        print("- - - Phi_μ(t,x) - - -")
        nx, nt, mu = 0, 0, 0
        val = tvector[mu,nt,nx]
        print("Phi_{0}({1},{2})=".format(mu,nt,nx),val.item(),val.dtype)
        nx, nt, mu = 0, 0, 1
        val = tvector[mu,nt,nx]
        print("Phi_{0}({1},{2})=".format(mu,nt,nx),val.item(),val.dtype)
        nx, nt, mu = 0, 1, 0
        val = tvector[mu,nt,nx]
        print("Phi_{0}({1},{2})=".format(mu,nt,nx),val.item(),val.dtype)
        nx, nt, mu = 0, 1, 1
        val = tvector[mu,nt,nx]
        print("Phi_{0}({1},{2})=".format(mu,nt,nx),val.item(),val.dtype)
        nx, nt, mu = 10, 15, 1
        val = tvector[mu,nt,nx]
        print("Phi_{0}({1},{2})=".format(mu,nt,nx),val.item(),val.dtype)