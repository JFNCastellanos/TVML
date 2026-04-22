import torch
import torch.nn as nn
import parameters as var
import gauge_equivariant as ge
var.init()

#torch.nn.Conv2d(in_channels, out_channels, kernel_size,
#stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', 
#device=None, dtype=None)
#torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1,
#padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', 
#device=None, dtype=None)

conv_layers = nn.Sequential(
            #Conv2D(in_chan,out_chan,kernel,stride,padding)
            #size 6 x NT x NX
            nn.CircularPad2d(1), #We pad to include periodic boundaries
            #size 6 x (NT+2) x (NX+2)    
            nn.Conv2d(6, 64, 2, 1, 0,dtype=var.PREC),
            nn.BatchNorm2d(64,dtype=var.PREC),
            nn.PReLU(64,dtype=var.PREC),
            #size 12 x (NT+1) x (NX+1)  
            nn.CircularPad2d(1), 
            #size 12 x (NT+3) x (NX+3)  
            nn.Conv2d(64, 128, 2, 1, 0,dtype=var.PREC),
            nn.BatchNorm2d(128,dtype=var.PREC),
            nn.PReLU(128,dtype=var.PREC),
            #size 24 x (NT+2) x (NX+2)  
            #nn.CircularPad2d(1), 
            #size 12 x (NT+4) x (NX+4)  
            #nn.Conv2d(128, 4*var.NV_PRED, 5, 1, 0),    
            nn.AdaptiveAvgPool2d((1, 1))
            #size 24 x 1 x 1  
)
linear_layers = nn.Sequential(
            #state size 24
            nn.Linear(128, 256,dtype=var.PREC), #We multiply by the number of output channels
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(256,dtype=var.PREC),
            nn.PReLU(256,dtype=var.PREC),
    
            #nn.Linear(256, 512,dtype=var.PREC), #We multiply by the number of output channels
            #nn.Dropout(p=0.1),
            #nn.BatchNorm1d(512,dtype=var.PREC),
            #nn.PReLU(512,dtype=var.PREC),
            nn.Linear(256, 4*var.NV_PRED*var.NT*var.NX,dtype=var.PREC),
    #The state is later reshaped into (B,NV_PRED,4,NT,NX) (real) and then (B,NV_PRED,2,NT,NX) (complex)
)

conv_layers_v2 = nn.Sequential(
            #Conv2D(in_chan,out_chan,kernel,stride,padding)
            #size 6 x NT x NX
            nn.CircularPad2d(1), #We pad to include periodic boundaries
            #size 6 x (NT+2) x (NX+2)    
            nn.Conv2d(6, 32, 2, 1, 0,dtype=var.PREC),
            nn.BatchNorm2d(32,dtype=var.PREC),
            nn.PReLU(32,dtype=var.PREC),
            #size 12 x (NT+1) x (NX+1)  
            nn.CircularPad2d(1), 
            #size 12 x (NT+3) x (NX+3)  
            nn.Conv2d(32, 64, 2, 1, 0,dtype=var.PREC),
            nn.BatchNorm2d(64,dtype=var.PREC),
            nn.PReLU(64,dtype=var.PREC),
            #size 24 x (NT+2) x (NX+2)    
            nn.AdaptiveAvgPool2d((1, 1))
            #size 24 x 1 x 1  
)
linear_layers_v2 = nn.Sequential(
            #state size 24
            nn.Linear(64, 128,dtype=var.PREC), #We multiply by the number of output channels
            #nn.Dropout(p=0.1),
            nn.BatchNorm1d(128,dtype=var.PREC),
            nn.PReLU(128,dtype=var.PREC),
            nn.Linear(128, 4*var.NV_PRED*var.NT*var.NX,dtype=var.PREC),
    #The state is later reshaped into (B,NV_PRED,4,NT,NX) (real) and then (B,NV_PRED,2,NT,NX) (complex)
)


#gauge equivariant convolutional layers
lcnn_layers = nn.Sequential(
            ge.LConv( 2, 8,3),
            nn.PReLU(8,dtype=var.PREC,device=var.DEVICE),
                
            ge.LConv( 8, 16,3),
            nn.PReLU(16,dtype=var.PREC,device=var.DEVICE),
                
            ge.LConv( 16, 32,3),
            nn.PReLU(32,dtype=var.PREC,device=var.DEVICE),
                
            ge.LConv( 32, 64,3),
            nn.PReLU(64,dtype=var.PREC,device=var.DEVICE),
                
            ge.LConv( 64, 128,3),
    
            ge.LConv( 128, 4*var.NV_PRED,3),
            nn.PReLU(4*var.NV_PRED,dtype=var.PREC,device=var.DEVICE)
)

class TvGenerator(nn.Module):
    """
    CNN for generating test vectors
    """
    def __init__(self, ngpu,batch_size):
        super(TvGenerator, self).__init__()
        self.ngpu = ngpu
        if var.GAUGE_EQ == False:
            self.conv_layers = conv_layers
            self.linear_layers = linear_layers
        else:
            self.lcnn_layers = lcnn_layers
        self.batch_size = batch_size
    def forward(self, input):
        if var.GAUGE_EQ == False:
            x = self.conv_layers(input)
            x = x.squeeze() #We remove the trivial dimensions
            x = self.linear_layers(x)
        else:  
            x = self.lcnn_layers(input)
            x = x.squeeze() #We remove the trivial dimensions
        return x.view(self.batch_size,var.NV_PRED,4,var.NT,var.NX)