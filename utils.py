import numpy as np
import parameters as var

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

def SavePredictions(dataloader, model, device):
    """
    Saves test vectors predicted with the model into binary files.
    One file per test vector. The data layout is the same as for
    the near-kernel vectors used for the training.
    x, t, μ, Re(Uμ), Im(Uμ)
    """
    with torch.no_grad():
        for batch_id, batch in enumerate(dataloader):
            data_batch = batch[0].to(device)          # (B, …)
            pred = model(data_batch)                  # (B, 4*NV, NT, NX)
            confsID = batch[2]

            B = pred.shape[0]
            pred = pred.view(B, var.NV_PRED, 4, var.NT, var.NX)   # (B,NV,4,NT,NX)

            # Build complex tensor (B,NV,2,NT,NX)
            real = torch.stack([pred[:, :, 0], pred[:, :, 1]], dim=2)   # (B,NV,2,NT,NX)
            imag = torch.stack([pred[:, :, 2], pred[:, :, 3]], dim=2)   # (B,NV,2,NT,NX)
            pred_complex = torch.complex(real, imag)
            norms = torch.linalg.vector_norm(pred_complex[:,:],dim=(-3,-2, -1)).view(B, var.NV_PRED, 1, 1, 1)
            norms_broadcastable = norms.view(B, var.NV_PRED, 1, 1, 1)
            pred_complex_normalized = pred_complex / norms_broadcastable
            pred_complex_normalized = pred_complex_normalized.cpu().detach().numpy()
            for i in range(B):
                for tv in range(var.NV_PRED):
                    file_path = "fake_tv/b{0}_{1}x{2}/{3}/conf{4}_fake_tv{5}.tv".format(var.BETA,var.NX,var.NT,var.M0_FOLDER,confsID[i],tv)
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
