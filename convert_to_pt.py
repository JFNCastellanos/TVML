# Functions for converting .ctxt and .plaq binary files to .pt files, which are supposed to be better handled by PyTorch.
import parameters as var
import torch
from glob import glob
from opendataset import read_binary_conf, read_binary_plaquette


def binaryConf2ptConf():
    PATH_CONF = '/wsgjsc/home/nietocastellanos1/Documents/SchwingerModel/fermions/SchwingerModel' + \
        '/confs/b{0}_{1}x{2}/{3}'.format(int(var.BETA),var.NX,var.NT,var.M0_FOLDER)
    print("Converting binary files of gconfs to .pt files")
    n = 0
    for path in sorted(glob(PATH_CONF+"/*.ctxt")):
        arr = read_binary_conf("",path)
        torch.save(torch.from_numpy(arr), path.replace(".ctxt", ".pt"))
        if ( n % (var.NO_CONFS // 10) == 0 and n > 0):
            print("Conf {0}/{1}".format(n,var.NO_CONFS))
        n +=1
    print("Done")

def binaryPlaq2ptPlaq():
    PATH_PLAQUETTE = '/wsgjsc/home/nietocastellanos1/Documents/TVML/sap/near_kernel' + \
            '/b{0}_{1}x{2}/{3}/plaquettes'.format(int(var.BETA),var.NX,var.NT,var.M0_FOLDER,var.M0_STRING)
    print("Converting binary files of plaquettes to .pt files")
    n = 0
    for path in sorted(glob(PATH_PLAQUETTE+"/*.plaq")): 
        arr = read_binary_plaquette("",path)
        torch.save(torch.from_numpy(arr), path.replace(".plaq", ".pt"))
        if ( n % (var.NO_CONFS // 10) == 0 and n > 0):
            print("Plaquette {0}/{1}".format(n,var.NO_CONFS))
        n +=1
    print("Done")

#idx = 0
#conf_file = conf_files[idx]
#gauge_conf = read_binary_conf(None,conf_file)
#tvectors = torch.zeros(var.NV,2,var.NT,var.NX,dtype=torch.complex128)

#gauge_conf = torch.load(self.conf_files[idx].replace('.ctxt','.pt'), mmap=True)
#tvectors = torch.load(self.tv_files[idx].replace('.tv','.pt'), mmap=True)
        
#real = torch.tensor(np.real(gauge_conf), dtype=torch.float32)
#imag = torch.tensor(np.imag(gauge_conf), dtype=torch.float32)
#gauge_conf = torch.tensor(np.array([real,imag])).reshape(4, var.NT, var.NX)  
        
#for tvID in range(var.NV):
#    vector = read_binary_conf(None,tv_files[idx][tvID])
#    real = torch.tensor(np.real(vector), dtype=torch.float64)
#    imag = torch.tensor(np.imag(vector), dtype=torch.float64)
#    tvectors[tvID] = torch.complex(real,imag).reshape(2, var.NT, var.NX) 