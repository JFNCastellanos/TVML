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
    x, t, mu, vals = np.zeros(var.N), np.zeros(var.N), np.zeros(var.N), np.zeros(var.N,dtype=complex)
    #U[μ,t,x]
    conf = np.zeros((2,var.NT,var.NX),dtype=complex)
    with open(path, 'rb') as infile:
        for i in range(var.N):
            #Real int values
            x, t, mu = struct.unpack('3i', infile.read(12))
            #Read double values
            re, im = struct.unpack('2d', infile.read(16))
            conf[mu,t,x] = complex(re,im)
    return conf


class ConfsDataset(torch.utils.data.Dataset):
    """
    Class for opening the binary gauge confs, which have the following layout:
    x, t, μ, Re(Uμ), Im(Uμ)
    Each conf has the format U[μ,t,x] (4 x NX x NT) (we split real and imaginary part)
    gauge_conf = [Re(U0),Re(U1),Im(U0),Im(U1)]
    
    For the near kernel vectors we don't do the splitting, since they won't be evaluated with
    pytorch. 
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
            self.conf_files.append(PATH)
            self.tv_files.append([])
            for tvID in range(var.NV):
                PATH = 'sap/near_kernel/b{0}_{1}x{2}/{3}/tvector_{1}x{2}_b{0}0000_m{4}_nconf{5}_tv{6}.tv'.format(int(var.BETA),var.NX,var.NT,var.M0_FOLDER,var.M0_STRING,confID,tvID)
                self.tv_files[confID].append(PATH) 
    #Method for opening the binary confs
    read_binary_conf = read_binary_conf
        
    def __getitem__(self, idx):
        conf_file = self.conf_files[idx]
        gauge_conf = self.read_binary_conf(conf_file)
        tvectors = torch.zeros(var.NV,2,var.NT,var.NX,dtype=torch.complex128)
             
        real = torch.tensor(np.real(gauge_conf), dtype=torch.float32)
        imag = torch.tensor(np.imag(gauge_conf), dtype=torch.float32)
        gauge_conf = torch.tensor(np.array([real,imag])).reshape(4, var.NT, var.NX)  
        
        for tvID in range(var.NV):
            vector = self.read_binary_conf(self.tv_files[idx][tvID])
            real = torch.tensor(np.real(vector), dtype=torch.float64)
            imag = torch.tensor(np.imag(vector), dtype=torch.float64)
            tvectors[tvID] = torch.complex(real,imag).reshape(2, var.NT, var.NX) 
        return (gauge_conf,tvectors)

    def __len__(self):
        return self.no_confs