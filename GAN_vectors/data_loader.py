import torch
import parameters as var
import struct #For opening binary data
import numpy as np
import h5py
var.init()

def read_binary_conf(self,path):
    """
    Function used for opening a gauge configuration in binary format.
    It can also be used to open a .tv file with a test vector, necessary
    for training.
    """
    var.init()
    x, t, mu, vals = np.zeros(var.N), np.zeros(var.N), np.zeros(var.N), np.zeros(var.N,dtype=complex)
    #U[μ,t,x]
    #conf = np.zeros((2,var.NT,var.NX),dtype=complex)
    data = np.fromfile(path, dtype=[('x', 'i4'),
                ('t', 'i4'),
                ('mu', 'i4'),
                ('re','f8'),
                ('im','f8')])
    conf = np.zeros((2, var.NT, var.NX), dtype=np.complex128)
    conf[data['mu'], data['t'], data['x']] = data['re'] + 1j * data['im']            
    return conf

def read_binary_plaquette(self,path):
    """
    Function used for opening a plaquette in binary format.
    """
    var.init()
    x, t, vals = np.zeros(var.N), np.zeros(var.N), np.zeros(var.N,dtype=complex)
    #U[μ,t,x]
    #conf = np.zeros((2,var.NT,var.NX),dtype=complex)
    data = np.fromfile(path, dtype=[('x', 'i4'),
                ('t', 'i4'),
                ('re','f8'),
                ('im','f8')])
    plaquette = np.zeros((var.NT, var.NX), dtype=np.complex128)
    plaquette[data['t'], data['x']] = data['re'] + 1j * data['im']            
    return plaquette

class ConfsDataset(torch.utils.data.Dataset):
    """
    Class for opening the gauge confs, test vectors and plaquettes
    gconf and tvector files layout:
        x, t, μ, Re(Uμ), Im(Uμ)
    plaquette files layout:
        x, t, Re(U01), Im(U01)
        
    Each conf has the torch tensor shape U[μ,t,x] (4 x NX x NT) (we split real and imaginary part)
    gauge_conf = [Re(U0),Re(U1),Im(U0),Im(U1)]
    plaquette = [Re(U01),Im(U01)]
    data = [Re(U0),Re(U1),Im(U0),Im(U1),Re(U01),Im(U01)]
    
    For the near kernel vectors we don't do the splitting, since we will evaluate them
    with complex tensors.
    tvectors.shape = [NV,2,NT,NX], with a complex number format.
    
    The class is defined as a subclass of Dataset from pytorch
    PATH might have to be adjusted
    """
    def __init__(self,confID,num_examples):
        self.confID = confID
        self.tv_files = []
        self.num_examples = num_examples
        self.PATH_CONF = '/wsgjsc/home/nietocastellanos1/Documents/SchwingerModel/fermions/SchwingerModel' + \
            '/confs/b{0}_{1}x{2}/{3}/2D_U1_Ns{1}_Nt{2}_b{0}0000_m{4}_{5}.ctxt'.format(int(var.BETA),var.NX,var.NT,var.M0_FOLDER,var.M0_STRING,confID)
        for tvID in range(num_examples):
            PATH_TV='/wsgjsc/home/nietocastellanos1/Documents/TVML/real_tv/b{0}_{1}x{2}/{3}/tvector_{1}x{2}_b{0}0000_m{4}_nconf{5}_tv{6}.tv'.format(int(var.BETA),var.NX,var.NT,var.M0_FOLDER,var.M0_STRING,confID,tvID)
            self.tv_files.append(PATH_TV) 
        
    def __getitem__(self, tvID):
        #idx is the conf index i.e. the conf file ends with ..._idx.ctxt
        conf_file = self.PATH_CONF
        tvectors = torch.zeros(var.NV,2,var.NT,var.NX,dtype=var.PREC_COMPLEX)

        #Loading gauge conf
        gauge_conf = torch.load(conf_file.replace('.ctxt','.pt'), mmap=True,weights_only=True)
        #gauge_conf = torch.cat([torch.real(gauge_conf), torch.imag(gauge_conf)], dim=0).to(dtype=var.PREC)   # [4, NT, NX]
        tvector = torch.load(self.tv_files[tvID].replace('.tv','.pt'), mmap=True,weights_only=True)
        return (gauge_conf,tvector,tvID)

    def __len__(self):
        return self.num_examples

"""
class ConfsDatasetHDF5(torch.utils.data.Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path

        # IMPORTANT: do NOT keep file open here for multi-worker loaders
        with h5py.File(self.h5_path, "r") as f:
            self.length = f["confs"].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Open per worker (safe for multiprocessing)
        with h5py.File(self.h5_path, "r") as f:
            conf = f["confs"][idx]           # (2, NT, NX) complex
            plaq = f["plaquettes"][idx]      # (NT, NX) complex
            tv   = f["tvectors"][idx]        # (NV, 2, NT, NX) complex
        # Convert to torch
        conf = torch.from_numpy(conf)
        plaq = torch.from_numpy(plaq)
        tv   = torch.from_numpy(tv)

        # Match your original preprocessing
        gauge_conf = torch.cat(
            [torch.real(conf), torch.imag(conf)], dim=0
        ).to(dtype=var.PREC)   # [4, NT, NX]

        plaquette = torch.stack(
            [torch.real(plaq), torch.imag(plaq)], dim=0
        ).to(dtype=var.PREC)   # [2, NT, NX]

        data = torch.cat([gauge_conf, plaquette], dim=0)  # [6, NT, NX]

        return (data, tv.to(var.PREC_COMPLEX), idx)
"""