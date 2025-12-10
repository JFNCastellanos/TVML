import torch
from opendataset import ConfsDataset, read_binary_plaquette #class for opening gauge confs
import utils
import parameters as var
import numpy as np
var.init()


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

        #----returns a list  [ [batch_size,4,Nt,Nx], [batch_size,Nv,2,Nt,Nx], [batch_size]]----#
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
        conf = self.first_batch[0][confID].to(var.DEVICE)
        assert self.first_batch[2][confID] == confID, "First conf should be ID {0}".format(confID)
        print("- - - Uμ(t,x) - - -")
        nx, nt, mu = 0, 0, 0
        val = conf[mu,nt,nx] + 1j*conf[mu+2,nt,nx,]
        print("U{0}({1},{2})=".format(mu,nt,nx),val.item(),val.dtype,val.device)
        nx, nt, mu = 0, 0, 1
        val = conf[mu,nt,nx] + 1j*conf[mu+2,nt,nx]
        print("U{0}({1},{2})=".format(mu,nt,nx),val.item(),val.dtype,val.device)
        nx, nt, mu = 0, 1, 0
        val = conf[mu,nt,nx] + 1j*conf[mu+2,nt,nx]
        print("U{0}({1},{2})=".format(mu,nt,nx),val.item(),val.dtype,val.device)
        nx, nt, mu = 0, 1, 1
        val = conf[mu,nt,nx] + 1j*conf[mu+2,nt,nx]
        print("U{0}({1},{2})=".format(mu,nt,nx),val.item(),val.dtype,val.device)
        nx, nt, mu = 10, 15, 1
        val = conf[mu,nt,nx] + 1j*conf[mu+2,nt,nx]
        print("U{0}({1},{2})=".format(mu,nt,nx),val.item(),val.dtype,val.device)
        for nt in range(conf.shape[0]):
            for nx in range(conf.shape[1]):
                for mu in range(2):
                    val = conf[mu,nt,nx] + 1j*conf[mu+2,nt,nx]
                    norm = torch.abs(val)
                    assert torch.abs(norm-1) < 1e-9, "Norm should be one, instead it is {0} for mu={1} nt={2}, nx={3}".format(norm,mu,nt,nx)
        print("All elements of the gauge conf are U(1) variables")

        
    def testTVector(self):
        print("Check that test vector 0 from conf 0 coincides with the info in the file")
        confID = 0
        nv = 0
        tvector = self.first_batch[1][confID][nv].to(var.DEVICE)
        assert self.first_batch[2][confID] == confID, "First conf should be ID {0}".format(confID)
        print("- - - Phi_μ(t,x) - - -")
        nx, nt, mu = 0, 0, 0
        val = tvector[mu,nt,nx]
        print("Phi_{0}({1},{2})=".format(mu,nt,nx),val.item(),val.dtype,val.device)
        nx, nt, mu = 0, 0, 1
        val = tvector[mu,nt,nx]
        print("Phi_{0}({1},{2})=".format(mu,nt,nx),val.item(),val.dtype,val.device)
        nx, nt, mu = 0, 1, 0
        val = tvector[mu,nt,nx]
        print("Phi_{0}({1},{2})=".format(mu,nt,nx),val.item(),val.dtype,val.device)
        nx, nt, mu = 0, 1, 1
        val = tvector[mu,nt,nx]
        print("Phi_{0}({1},{2})=".format(mu,nt,nx),val.item(),val.dtype,val.device)
        nx, nt, mu = 10, 15, 1
        val = tvector[mu,nt,nx]
        print("Phi_{0}({1},{2})=".format(mu,nt,nx),val.item(),val.dtype,val.device)

    def testPlaquette(self):
        print("Check that plaquette from conf 0 coincides with the info in the file")
        print("(This test is on the CPU because I am opening the binaries directly with numpy)")
        confID = 0
        path =  '/wsgjsc/home/nietocastellanos1/Documents/TVML/sap/near_kernel' + \
            '/b{0}_{1}x{2}/{3}/plaquettes/plaquette_{1}x{2}_b{0}0000_m{4}_nconf{5}.plaq'.format(int(var.BETA),var.NX,var.NT,var.M0_FOLDER,var.M0_STRING,confID)
        plaquette = read_binary_plaquette(None,path)
        print("- - - U_01(t,x) - - -")
        nx, nt = 0, 0
        val = plaquette[nt,nx]
        print("U_01({0},{1})=".format(nt,nx),val,val.dtype)
        nx, nt = 0, 1
        val = plaquette[nt,nx]
        print("U_01({0},{1})=".format(nt,nx),val,val.dtype)
        nx, nt = 1, 0
        val = plaquette[nt,nx]
        print("U_01({0},{1})=".format(nt,nx),val,val.dtype)
        nx, nt = 1, 1
        val = plaquette[nt,nx]
        print("U_01({0},{1})=".format(nt,nx),val,val.dtype)
        nx, nt = 10, 15
        val = plaquette[nt,nx]
        print("U_01({0},{1})=".format(nt,nx),val,val.dtype)
        for nt in range(plaquette.shape[0]):
            for nx in range(plaquette.shape[1]):
                norm = np.abs(plaquette[nt,nx])
                assert np.abs(norm-1) < 1e-9, "Norm should be one, instead it is {0} for nt={1}, nx={2}".format(norm,nt,nx)
        print("All elements of the plaquette are U(1) variables")
