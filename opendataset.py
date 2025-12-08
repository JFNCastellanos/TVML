import torch
import parameters as var
import struct #For opening binary data
import numpy as np

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


class ConfsDataset(torch.utils.data.Dataset):
    """
    Class for opening the binary gauge confs, which have the following layout:
    x, t, μ, Re(Uμ), Im(Uμ)
    Each conf has the format U[μ,t,x] (4 x NX x NT) (we split real and imaginary part)
    gauge_conf = [Re(U0),Re(U1),Im(U0),Im(U1)]
    
    For the near kernel vectors we don't do the splitting, since we will evaluate them
    with complex tensors.
    tvectors.shape = [NV,2,NT,NX], with a complex number format.
    
    The class is defined as a subclass of Dataset from pytorch
    PATH might have to be adjusted
    """
    def __init__(self):
        self.conf_files = [] #List with files
        self.tv_files = []
        self.no_confs = var.NO_CONFS
        for confID in range(self.no_confs):
            PATH = '/wsgjsc/home/nietocastellanos1/Documents/SchwingerModel/fermions/SchwingerModel' + \
            '/confs/b{0}_{1}x{2}/{3}/2D_U1_Ns{1}_Nt{2}_b{0}0000_m{4}_{5}.ctxt'.format(int(var.BETA),var.NX,var.NT,var.M0_FOLDER,var.M0_STRING,confID)
            #PATH = 'confs/confs/b{0}_{1}x{2}/{3}/2D_U1_Ns{1}_Nt{2}_b{0}0000_m{4}_{5}.ctxt'.format(int(var.BETA),var.NX,var.NT,var.M0_FOLDER,var.M0_STRING,confID)
            self.conf_files.append(PATH)
            self.tv_files.append([])
            for tvID in range(var.NV):
                PATH='/wsgjsc/home/nietocastellanos1/Documents/TVML/sap/near_kernel/b{0}_{1}x{2}/{3}/tvector_{1}x{2}_b{0}0000_m{4}_nconf{5}_tv{6}.tv'.format(int(var.BETA),var.NX,var.NT,var.M0_FOLDER,var.M0_STRING,confID,tvID)
                #PATH =           'confs/near_kernel/b{0}_{1}x{2}/{3}/tvector_{1}x{2}_b{0}0000_m{4}_nconf{5}_tv{6}.tv'.format(int(var.BETA),var.NX,var.NT,var.M0_FOLDER,var.M0_STRING,confID,tvID)           
                self.tv_files[confID].append(PATH) 
        
    def __getitem__(self, idx):
        #idx is the conf index i.e. the conf file ends with ..._idx.ctxt
        conf_file = self.conf_files[idx]
        tvectors = torch.zeros(var.NV,2,var.NT,var.NX,dtype=torch.complex128)

        #tvectors = torch.load(self.tv_files[idx].replace('.tv','.pt'), mmap=True)

        #Loading gauge conf
        gauge_conf = torch.load(conf_file.replace('.ctxt','.pt'), mmap=True,weights_only=True)
        gauge_conf = torch.cat([torch.real(gauge_conf), torch.imag(gauge_conf)], dim=0).to(dtype=torch.float64)   # [4, NT, NX]

        
        for tvID in range(var.NV):
            #vector = self.read_binary_conf(self.tv_files[idx][tvID])
            #real = torch.tensor(np.real(vector), dtype=torch.float64)
            #imag = torch.tensor(np.imag(vector), dtype=torch.float64)
            #tvectors[tvID] = torch.complex(real,imag).reshape(2, var.NT, var.NX) 
            tvectors[tvID] = torch.load(self.tv_files[idx][tvID].replace('.tv','.pt'), mmap=True,weights_only=True)
        return (gauge_conf,tvectors,idx)

    def __len__(self):
        return self.no_confs