#training loop
# Training Loop



# Lists to keep track of progress
conf_list = []
G_losses = []
D_losses = []
iters = 0

import numpy as np
import torch
import torch.optim as optim
import torch.nn.parallel
import parameters as var #Configuration and coarsening parameters
var.init() #initializes parameter
import loss_function as lf #Custom loss function


#We train model for one configuration and many test vectors
def train(dataloader, Gen, Disc, criterion, optimizerG,optimizerD,conf,lossesG,lossesD):
    Gen.train()
    Disc.train()
    #u = conf.repeat(batch_size,2,var.NT,var.NX) #gauge configuration
    real_label = 1
    fake_label = 0 
    for batch_id, batch in enumerate(dataloader):
        # Load the data
        w   = batch[0].to(var.DEVICE)      # shape (B, 2, NT, NX) #Near kernel 
        batch_size = w.shape[0]

        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        Disc.zero_grad()
        label = torch.full((batch_size,), real_label, dtype=var.PREC, device=device)
        # Forward pass real batch through D
        Dw_real = Disc(u,w)
        # Calculate loss on all-real batch
        errD_real = criterion(Dw_real, label)
        errD_real.backward()
        Dw = Dw_real.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        
        #Random noise for the generator
        std = 0.5 
        real = torch.empty(batch_size, 1, var.NT, var.NX, dtype=var.PREC, device=var.DEVICE) #(B,1,NT,NX)
        imag = torch.empty(batch_size, 1, var.NT, var.NX, dtype=var.PREC, device=var.DEVICE)
        nn.init.normal_(real, mean=0.0, std=std)
        nn.init.normal_(imag, mean=0.0, std=std)
        random_noise = torch.complex(real, imag)

        fake = Gen(u,random_noise)
        label.fill_(fake_label)

        # Classify all fake batch with D
        D_Gw = Disc(fake)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(D_Gw, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = D_Gw.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step() #Use gradients to optimize D
    
        # (2) Update G network: maximize log(D(G(z)))

        Gen.zero_grad()
        label.fill_(real_label)
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = Disc(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()
        
       
        # Output training stats
        if i % 50 == 0:
            print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (i, len(dataloader),
                     errD.item(), errG.item(), Dw, D_G_z1, D_G_z2))
        # Save Losses for plotting later
        lossesG.append(errG.item())
        lossesD.append(errD.item())
