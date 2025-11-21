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
            nn.Tanh()
            # state size. 4*Nv x NT x NX, (real, imaginary part and two spin components)
)

class TvGenerator(nn.Module):
    """
    CNN for generating test vectors
    """
    def __init__(self, ngpu):
        super(TvGenerator, self).__init__()
        self.ngpu = ngpu
        self.main = neural_net2 #neural_net
    def forward(self, input):
        return self.main(input)