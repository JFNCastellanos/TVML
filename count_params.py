#For this particular model we count the number of parameters manually
conv_layers = nn.Sequential(
            #Conv2D(in_chan,out_chan,kernel,stride,padding)
            #size 4 x NT x NX
            nn.CircularPad2d(1), #We pad to include periodic boundaries
            #size 4 x (NT+1) x (NX+1)    
            nn.Conv2d(4, 64, 2, 1, 0),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
            #size 12 x NT x NX  
            nn.CircularPad2d(1), 
            #size 12 x (NT+1) x (NX+1)  
            nn.Conv2d(64, 128, 2, 1, 0),
            nn.BatchNorm2d(128),
            nn.PReLU(128),
            #size 24 x NT x NX  
            nn.AdaptiveAvgPool2d((1, 1))
            #size 24 x 1 x 1  
)
linear_layers = nn.Sequential(
            #state size 24
            nn.Linear(128, 256), #We multiply by the number of output channels
            #nn.Dropout(p=0.1),
            nn.BatchNorm1d(256),
            nn.PReLU(256),
    
            nn.Linear(256, 512), #We multiply by the number of output channels
            #nn.Dropout(p=0.1),
            nn.BatchNorm1d(512),
            nn.PReLU(512),
            nn.Linear(512, 4*var.NV*var.NT*var.NX),
    #The state is later reshaped into (B,NV,4,NT,NX) (real) and then (B,NV,2,NT,NX) (complex)
)



params = ut.conv_params(4,64,2,2) #conv layer
params += 64 #PReLU
params += 64*2 #Batch norm
params += ut.conv_params(64,128,2,2)
params += 128
params += 128*2
params += ut.ll_parameters(128,256)
params += 256
params += 256*2
params += ut.ll_parameters(256,512) #linear layer
params += 512
params += 512*2
params += ut.ll_parameters(512,4*var.NV*var.NT*var.NX)
#Both outputs agree
print("Model parameters",params)
print(sum(p.numel() for p in model.parameters() if p.requires_grad))