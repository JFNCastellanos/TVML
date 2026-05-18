# Functions for converting .ctxt and .plaq binary files to .pt files, which are supposed to be better handled by PyTorch.
# Functions for saving everything on a single .h5 file 
import parameters as var
import torch
from glob import glob
from opendataset import read_binary_conf, read_binary_plaquette
import h5py
import numpy as np
import os

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
    PATH_PLAQUETTE = '/wsgjsc/home/nietocastellanos1/Documents/TVML/plaquettes' + \
            '/b{0}_{1}x{2}/{3}'.format(int(var.BETA),var.NX,var.NT,var.M0_FOLDER)
    print("Converting binary files of plaquettes to .pt files")
    n = 0
    for path in sorted(glob(PATH_PLAQUETTE+"/*.plaq")): 
        arr = read_binary_plaquette("",path)
        torch.save(torch.from_numpy(arr), path.replace(".plaq", ".pt"))
        if ( n % (var.NO_CONFS // 10) == 0 and n > 0):
            print("Plaquette {0}/{1}".format(n,var.NO_CONFS))
        n +=1
    print("Done")

def binaryTestV2ptTestV():
    PATH_TV = '/wsgjsc/home/nietocastellanos1/Documents/TVML/real_tv' + \
            '/b{0}_{1}x{2}/{3}/'.format(int(var.BETA),var.NX,var.NT,var.M0_FOLDER)
    print("Converting binary files of test vectors to .pt files")
    n = 0
    for path in sorted(glob(PATH_TV+"/*.tv")): 
        arr = read_binary_conf("",path)
        torch.save(torch.from_numpy(arr), path.replace(".tv", ".pt"))
        if ( n % (var.NO_CONFS*var.NV // 10) == 0 and n > 0):
            print("Test vector {0}/{1}".format(n,var.NO_CONFS*var.NV))
        n +=1
    print("Done")

    
def MakeH5_File(output_path):
    PATH_CONF = '/wsgjsc/home/nietocastellanos1/Documents/SchwingerModel/fermions/SchwingerModel' + \
        '/confs/b{0}_{1}x{2}/{3}/'.format(int(var.BETA),var.NX,var.NT,var.M0_FOLDER)

    PATH_TV = '/wsgjsc/home/nietocastellanos1/Documents/TVML/real_tv' + \
        '/b{0}_{1}x{2}/{3}/'.format(int(var.BETA),var.NX,var.NT,var.M0_FOLDER)

    PATH_PLAQ = '/wsgjsc/home/nietocastellanos1/Documents/TVML/plaquettes' + \
        '/b{0}_{1}x{2}/{3}/'.format(int(var.BETA),var.NX,var.NT,var.M0_FOLDER)


    Nconf = var.NO_CONFS
    NV = var.NV
    NT = var.NT
    NX = var.NX

    conf_files = [] #List with files
    plaq_files = []
    tv_files = []
    for confID in range(Nconf):
        conf_files.append(PATH_CONF+"2D_U1_Ns{0}_Nt{1}_b{2}0000_m{3}_{4}.ctxt".format(NX,NT,var.BETA,var.M0_STRING,confID))
        plaq_files.append(PATH_PLAQ+"plaquette_{0}x{1}_b{2}0000_m{3}_nconf{4}.plaq".format(NX,NT,var.BETA,var.M0_STRING,confID))
        tv_files.append([])
        for tvID in range(NV):
             tv_files[confID].append(PATH_TV+"tvector_{0}x{1}_b{2}0000_m{3}_nconf{4}_tv{5}.tv".format(NX,NT,
                var.BETA,var.M0_STRING,confID,tvID                                                  
             ))


    # ---- Create HDF5 file ----
    with h5py.File(output_path, "w") as f:
        f.attrs["NX"] = NX
        f.attrs["NT"] = NT
        f.attrs["NV"] = NV
        f.attrs["beta"] = var.BETA
        d_confs = f.create_dataset(
            "confs",
            shape=(Nconf, 2, NT, NX),
            dtype=np.complex128,
            chunks=(1, 2, NT, NX),
            compression="gzip"
        )

        d_tv = f.create_dataset(
            "tvectors",
            shape=(Nconf, NV, 2, NT, NX),
            dtype=np.complex128,
            chunks=(1, NV, 2, NT, NX),
            compression="gzip"
        )

        d_plaq = f.create_dataset(
            "plaquettes",
            shape=(Nconf, NT, NX),
            dtype=np.complex128,
            chunks=(1, NT, NX),
            compression="gzip"
        )

        # Write configurations
        print("Writing configurations in H5 file")

        for n, path in enumerate(conf_files):
            arr = read_binary_conf("", path)  # (2, NT, NX)
            d_confs[n] = arr.astype(np.complex128)
            if (n % max(1, (Nconf // 10)) == 0 and n > 0):
                print(f"Conf {n}/{Nconf}")

        # Write plaquettes
        print("Writing plaquettes in H5 file")
        for n, path in enumerate(plaq_files):
            arr = read_binary_plaquette("", path)  # (NT, NX)
            d_plaq[n] = arr.astype(np.complex128)
            if (n % max(1, (Nconf // 10)) == 0 and n > 0):
                print(f"Conf {n}/{Nconf}")

        # Write test vectors
        print("Writing test vectors in H5 file")

        counter = 0
        for confID in range(Nconf):
            for tvID in range(NV):
                path = tv_files[confID][tvID]
                arr = read_binary_conf("", path)  # (2, NT, NX)
                d_tv[confID, tvID] = arr.astype(np.complex128)
                if ( counter % (var.NO_CONFS*var.NV // 10) == 0 and n > 0):
                    print("Test vector {0}/{1}".format(counter,var.NO_CONFS*var.NV))
                counter +=1

    print("HDF5 file written to:", output_path)