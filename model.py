import torch.nn as nn
import parameters as var

#torch.nn.Conv2d(in_channels, out_channels, kernel_size,
#stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', 
#device=None, dtype=None)
#torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1,
#padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', 
#device=None, dtype=None)

neural_net = nn.Sequential(
            #Conv2D(in_chan,out_chan,kernel,stride,padding)
            #state size 4 x NT x NX
            nn.Conv2d(4, 128, 2, 2, 0),
            #nn.Dropout(p=0.1),
            nn.BatchNorm2d(128),
            nn.PReLU(128),
            #state size 128 x NT/2 x NX/2
            nn.Conv2d(128, 64, 2, 2, 0),
            #nn.Dropout(p=0.1),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
            #state size 64 x NT/4 x NX/4
            nn.Conv2d(64, 32, 2, 2, 0),
            #nn.Dropout(p=0.1),
            nn.BatchNorm2d(32),
            nn.PReLU(32),
            #state size 32 x NT/8 x NX/8
            nn.ConvTranspose2d(32, 64, 2, 2, 0),
            #nn.Dropout(p=0.1),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
            # state size 64 x NT/4 x NX/4
            nn.ConvTranspose2d(64, 64, 2, 2, 0),
            #nn.Dropout(p=0.1),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
            # state size 64 x NT/2 x NX/2
            nn.ConvTranspose2d(64, 4*var.NV, 2, 2, 0),
            nn.Hardtanh(min_val=-2.0, max_val=2.0)
            # state size. 4*Nv x NT x NX, (real, imaginary part and two spin components)
)

neural_net2 = nn.Sequential(
            #Conv2D(in_chan,out_chan,kernel,stride,padding)
            #state size 4 x NT x NX
            nn.Conv2d(4, 128, 2, 2, 0),
            nn.BatchNorm2d(128),
            nn.PReLU(128),
            #state size 128 x NT/2 x NX/2
            nn.Conv2d(128, 64, 2, 2, 0),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
            #state size 64 x NT/4 x NX/4
            nn.ConvTranspose2d(64, 64, 2, 2, 0),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
            # state size 64 x NT/2 x NX/2
            nn.ConvTranspose2d(64, 4*var.NV, 2, 2, 0),
            #nn.Hardtanh(min_val=-1.0, max_val=1.0)
            #nn.Tanh()
            # state size. 4*Nv x NT x NX, (real, imaginary part and two spin components)
            #print(Re(Psi_0),Re(Psi_1),Im(Psi_0),Im(Psi_1)). 

            #TODO:
            #Create a layer to normalize the output without changing its shape

    #The state is later reshaped into (B,NV,4,NT,NX) (real) and then (B,NV,2,NT,NX) (complex)
)

neural_net3 = nn.Sequential(
            #Conv2D(in_chan,out_chan,kernel,stride,padding)
            #state size 4 x NT x NX
            nn.Conv2d(4, 64, 2, 2, 0),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
            #state size 128 x NT/2 x NX/2
            nn.Conv2d(64, 128, 2, 2, 0),
            nn.BatchNorm2d(128),
            nn.PReLU(128),
            #state size 64 x NT/4 x NX/4
            nn.Flatten(), #We flatten the output after the convolution
    
            nn.Linear(128 * var.NT//4 * var.NX//4, 256), #We multiply by the number of output channels
            nn.BatchNorm1d(256),
            nn.PReLU(256),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(512),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(512),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(512),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.PReLU(256),

            nn.Linear(256, 128), 
            nn.BatchNorm1d(128),
            nn.PReLU(128),

            nn.Linear(128, 64), 
            nn.BatchNorm1d(64),
            nn.PReLU(64),
            #nn.Hardtanh(min_val=-0.5, max_val=0.5),
    

            nn.Linear(64, 4*var.NV*var.NT*var.NX),
            #nn.Hardtanh(min_val=-1.0, max_val=1.0),
            #nn.PReLU(4*var.NV*var.NT*var.NX),

            
    #The state is later reshaped into (B,NV,4,NT,NX) (real) and then (B,NV,2,NT,NX) (complex)
)


class TvGenerator(nn.Module):
    """
    CNN for generating test vectors
    """
    def __init__(self, ngpu,batch_size):
        super(TvGenerator, self).__init__()
        self.ngpu = ngpu
        self.main = neural_net3 #neural_net
        self.batch_size = batch_size
    def forward(self, input):
        x = self.main(input)
        return x.view(self.batch_size,var.NV,4,var.NT,var.NX)