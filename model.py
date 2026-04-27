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
            nn.Conv2d(32, 24, 2, 1, 0,dtype=var.PREC),
            nn.BatchNorm2d(24,dtype=var.PREC),
            nn.PReLU(24,dtype=var.PREC),
            #size 24 x (NT+2) x (NX+2)    

            nn.AdaptiveAvgPool2d((1, 1))
            #size 24 x 1 x 1  
)
linear_layers_v2 = nn.Sequential(
            #state size 24
            nn.PReLU(24,dtype=var.PREC),
            nn.Linear(24, 4*var.NV_PRED*var.NT*var.NX,dtype=var.PREC),
    #The state is later reshaped into (B,NV_PRED,4,NT,NX) (real) and then (B,NV_PRED,2,NT,NX) (complex)
)


#gauge equivariant convolutional layers
lcnn_layers = var.MultiInputSequential(
            ge.LConv( 1, 8,3),             
            ge.LConv( 8, 16,3),
            ge.LConv( 16,2*var.NV_PRED ,3),
           # nn.AdaptiveAvgPool2d((1, 1))
)

lcnn_linear_layer = nn.Sequential(
       
)

class TvGenerator(nn.Module):
    """
    CNN for generating test vectors
    """
    def __init__(self, ngpu,batch_size):
        super(TvGenerator, self).__init__()
        self.ngpu = ngpu
        if var.GAUGE_EQ == False:
            self.conv_layers = conv_layers_v2
            self.linear_layers = linear_layers_v2
        else:
            self.lcnn_layers = lcnn_layers
            #self.lcnn_linear_layer = lcnn_linear_layer
        self.batch_size = batch_size
    def forward(self, u,w):
        if var.GAUGE_EQ == False:
            x = self.conv_layers(input)
            x = x.squeeze() #We remove the trivial dimensions
            x = self.linear_layers(x)
            return x.view(self.batch_size,var.NV_PRED,4,var.NT,var.NX)
        else:  
            u,w = self.lcnn_layers(u,w)
            w = w.squeeze() #We remove the trivial dimensions
            #x = self.lcnn_linear_layer(x)
            return w.view(self.batch_size,var.NV_PRED,2,var.NT,var.NX)
        