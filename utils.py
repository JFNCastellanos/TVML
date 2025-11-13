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
