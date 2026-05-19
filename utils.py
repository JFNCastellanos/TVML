import numpy as np
import parameters as var
import torch
import h5py

def formatt(x):
    """
    Format strings (rounds number to four digits and replaces decimal point) 
    For instance -0.7884 --> -07884
    """
    x = str(np.round(x,4)).replace(".","")
    return x

def flatten_col(col):
    flat = np.zeros(var.N,dtype=complex)
    for t in range(var.NT):
        for x in range(var.NX):
            for s in range(2):
                flat[2*(x*var.NT + t) + s] = col[s,t,x]
    return flat

def flatten_colV2(col):
    flat = np.zeros(2*var.NV*var.BLOCKS_X*var.BLOCKS_T,dtype=complex)
    for tv in range(var.NV):
        for t in range(var.BLOCKS_T):
            for x in range(var.BLOCKS_X):
                for s in range(2):
                    flat[tv*2*var.BLOCKS_X*var.BLOCKS_T + x*2*var.BLOCKS_T+2*t+s] = col[tv,s,t,x]    
                    #NV*(2*(x*BLOCKS_T + t) + s)+tv] = col[tv,s,t,x]
    return flat

def output_size(in_size,kernel,stride,padding,transpose=False):
    """
    Check the output size of a convolutional layer
    """
    if transpose == True:
        return (in_size-1)*stride - 2*padding + kernel
    else:
        return (in_size + 2*padding - kernel)/stride + 1

def conv_params(in_channel,out_channel,kernelx,kernely):
    """
    Number of parameters involved in a convolutional layer
    """
    return (in_channel*kernelx*kernely + 1) * out_channel

def ll_parameters(in_channel,out_channel):
    """
    Number of parameters involved in a linear layer
    """
    return (in_channel + 1) * out_channel


import torch
import struct

def SavePredictions(dataloader, model, dataname, device):
    """
    Saves test vectors predicted with the model into binary files.
    One file per test vector. The data layout is the same as for
    the near-kernel vectors used for the training.
    x, t, μ, Re(Uμ), Im(Uμ)
    """       
    with torch.no_grad():
        for batch_id, batch in enumerate(dataloader):
            data_batch = batch[0].to(device)          # (B, …)
            confsID = batch[2]
            if var.GAUGE_EQ == False:
                pred = model(data_batch)                  # (B, 4*NV, NT, NX)
                B = pred.shape[0]
                pred = pred.view(B, var.NV_PRED, 4, var.NT, var.NX)   # (B,NV,4,NT,NX)
    
                # Build complex tensor (B,NV,2,NT,NX)
                real = torch.stack([pred[:, :, 0], pred[:, :, 1]], dim=2)   # (B,NV,2,NT,NX)
                imag = torch.stack([pred[:, :, 2], pred[:, :, 3]], dim=2)   # (B,NV,2,NT,NX)
                pred_complex = torch.complex(real, imag)
            else:
                local_trans_obj = data_batch.shape[1]-4
                local_t_object = data_batch[:,4:]
        
                # Build a complex tensor of shape (B,  2, NT, NX)
                real = torch.stack([data_batch[:,0], data_batch[:,1]], dim=1)   # (B,2,NT,NX) (real number)
                imag = torch.stack([data_batch[:,2], data_batch[:,3]], dim=1)   # (B,2,NT,NX) (real number)
                u = torch.complex(real, imag)                  # (B,2,NT,NX) (complex number)
                if local_trans_obj == 2:
                    real = local_t_object[:,0].unsqueeze(1)          #(B,1,NT,NX)
                    imag = local_t_object[:,1].unsqueeze(1)          #(B,1,NT,NX)
                #else:
                    #We have to stack them. I leave the line just in case I add more things, like Polyakov loops.
                w = torch.complex(real,imag)
                pred_complex = model(u,w)       #pred is already a complex number
                B = pred_complex.shape[0]                                        #Batch size
                pred_complex = pred_complex.view(B, var.NV_PRED, 2, var.NT, var.NX)      # (B,NV,0,NT,NX)

            
            norms = torch.linalg.vector_norm(pred_complex[:,:],dim=(-3,-2, -1)).view(B, var.NV_PRED, 1, 1, 1)
            norms_broadcastable = norms.view(B, var.NV_PRED, 1, 1, 1)
            pred_complex_normalized = pred_complex / norms_broadcastable
            pred_complex_normalized = pred_complex_normalized.cpu().detach().numpy()
            for i in range(B):
                for tv in range(var.NV_PRED):
                    file_path = "fake_tv/b{0}_{1}x{2}/{3}/{4}/conf{5}_fake_tv{6}.tv".format(var.BETA,var.NX,var.NT,var.M0_FOLDER,
                                                                        dataname,confsID[i],tv)
                    fmt = "<3i2d"
                    with open(file_path, "wb") as f:
                        for x in range(var.NX):
                            for t in range(var.NT):
                                for mu in range(2):
                                    value = pred_complex_normalized[i,tv,mu,t,x]
                                    Re = np.real(value)
                                    Im = np.imag(value)
                                    data = struct.pack(fmt, int(x), int(t), int(mu), float(Re), float(Im))
                                    f.write(data)


import json
from datetime import datetime

class MetadataSaver:
    """Saves model metadata in a human-readable JSON format."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        model_name: str,
        epochs: int,
        lr: float,
        ratio: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        lamb: float = 0.0,
        loss_train: float = None,
        loss_test: float = None,
        train_examples: int = None,
    ):
        self.model = model
        self.model_name = model_name
        self.epochs = epochs
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.lamb = lamb
        self.loss_train = loss_train
        self.loss_test = loss_test
        self.ratio = ratio
        self.train_examples = var.TRAIN_LEN
        self.m0 = var.M0
        self.BLOCKS_X = var.BLOCKS_X
        self.BLOCKS_T = var.BLOCKS_T
        self.NV = var.NV
        self.NV_PRED = var.NV_PRED
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def _get_architecture(self) -> str:
        """Extracts the model architecture as a readable string."""
        return str(self.model)
    
    def _get_layer_summary(self) -> list:
        """Returns a summary of each layer with parameters."""
        summary = []
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                params = sum(p.numel() for p in module.parameters())
                if params > 0:
                    summary.append({
                        "layer": name,
                        "type": module.__class__.__name__,
                        "parameters": params
                    })
        return summary
    
    def to_dict(self) -> dict:
        """Converts metadata to a dictionary."""
        return {
            "metadata": {
                "model_name": self.model_name,
                "training_date": self.timestamp,
                "training_examples": self.train_examples,
                "parameters_data_ratio":self.ratio,
                "hyperparameters": {
                    "epochs": self.epochs,
                    "lr": self.lr,
                    "beta1": self.beta1,
                    "beta2": self.beta2,
                    "lamb": self.lamb
                },
                "m0": self.m0,
                "interpolator_parameters":{
                    "BLOCKS_X": self.BLOCKS_X,
                    "BLOCKS_T": self.BLOCKS_T,
                    "NV_SAP": self.NV,
                    "NV_PRED": self.NV_PRED
                },
                "final_losses": {
                    "train": self.loss_train,
                    "test": self.loss_test
                }
            },
            "architecture": self._get_architecture(),
            "layer_summary": self._get_layer_summary()
        }
    
    def save(self, filepath: str = "metadata.json"):
        """Saves metadata to a JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=4)
        print(f"Metadata saved to '{filepath}'")
    
    @classmethod
    def load(cls, filepath: str) -> dict:
        """Loads metadata from a JSON file."""
        with open(filepath, "r") as f:
            return json.load(f)



def SavePredictionsHDF5(dataloader, data_name, model, device, output_file):
    """
    Saves all predicted test vectors into a single HDF5 file.

    Dataset layout:
    - predictions: (N_total, NV, 2, NT, NX) complex64
    - confsID: (N_total,)
    """

    model.eval()

    all_preds = []
    all_confs = []

    with torch.no_grad():
        for batch_id, batch in enumerate(dataloader):
            data_batch = batch[0].to(device)
            confsID = batch[2]

            if not var.GAUGE_EQ:
                pred = model(data_batch)  # (B, 4*NV, NT, NX)
                B = pred.shape[0]

                pred = pred.view(B, var.NV_PRED, 4, var.NT, var.NX)

                real = torch.stack([pred[:, :, 0], pred[:, :, 1]], dim=2)
                imag = torch.stack([pred[:, :, 2], pred[:, :, 3]], dim=2)

                pred_complex = torch.complex(real, imag)

            else:
                local_trans_obj = data_batch.shape[1] - 4
                local_t_object = data_batch[:, 4:]

                real = torch.stack([data_batch[:, 0], data_batch[:, 1]], dim=1)
                imag = torch.stack([data_batch[:, 2], data_batch[:, 3]], dim=1)
                u = torch.complex(real, imag)

                if local_trans_obj == 2:
                    real = local_t_object[:, 0].unsqueeze(1)
                    imag = local_t_object[:, 1].unsqueeze(1)

                w = torch.complex(real, imag)

                pred_complex = model(u, w)
                B = pred_complex.shape[0]

                pred_complex = pred_complex.view(
                    B, var.NV_PRED, 2, var.NT, var.NX
                )

            # Normalize
            norms = torch.linalg.vector_norm(
                pred_complex, dim=(-3, -2, -1)
            ).view(B, var.NV_PRED, 1, 1, 1)

            pred_complex = pred_complex / norms

            # Move to CPU numpy
            pred_np = pred_complex.cpu().numpy()

            all_preds.append(pred_np)
            all_confs.append(np.array(confsID))

    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_confs = np.concatenate(all_confs, axis=0)

    # Write to HDF5
    with h5py.File(output_file, "w") as f:
        f.create_dataset(
            "predictions",
            data=all_preds,
            dtype=np.complex128,
            compression="gzip"
        )

        f.create_dataset(
            "confsID",
            data=all_confs,
            dtype=np.int32
        )

        # Optional metadata
        f.attrs["NX"] = var.NX
        f.attrs["NT"] = var.NT
        f.attrs["NV_PRED"] = var.NV_PRED
        f.attrs["beta"] = var.BETA
        f.attrs["data_name"] = data_name