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
            nn.Conv2d(4, 64, 2, 2, 0),
            nn.BatchNorm2d(64),
            #state size 64 x NT/2 x NX/2
            nn.PReLU(64),
            nn.Conv2d(64, 128, 2, 2, 0),
            nn.BatchNorm2d(128),
            nn.PReLU(128),
            # state size. ``128 x NT/4 x NX/4``
            nn.ConvTranspose2d(128, 64, 2, 2, 0),
            nn.BatchNorm2d(64),
            # state size. ``64 x NT/2 x NX/2``,
            nn.PReLU(64),
            nn.ConvTranspose2d(64, 4*var.NV, 2, 2, 0),
            nn.PReLU(4*var.NV)
            # state size. ``4*Nv x NT x NX``, (real, imaginary part and two spin components)

)

class TvGenerator(nn.Module):
    """
    CNN for generating test vectors
    """
    def __init__(self, ngpu):
        super(TvGenerator, self).__init__()
        self.ngpu = ngpu
        self.main = neural_net
    def forward(self, input):
        return self.main(input)