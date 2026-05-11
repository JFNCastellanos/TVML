#!/usr/bin/env python

# In[1]:


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.parallel
import random
import matplotlib.pyplot as plt
import parameters as var #Configuration and coarsening parameters
var.init() #initializes parameters
import utils as ut #Some utility functions 
import loss_function as lf #Custom loss function

import operators_torch as op #Interpolator and prolongator given a set of test vectors
from opendataset import ConfsDataset #class for opening gauge confs
import model as mod #import machine learning model
from train import train, evaluate

var.print_parameters()
device = var.DEVICE

model_name = "model3"
# In[2]:


"""
Loading the configurations and the near-kernel test vectors
We split train and test set
"""
dataset = ConfsDataset()                     
total_len = len(dataset)                    
train_len = int(var.TRAIN_PROP * total_len) 
test_len  = total_len - train_len   
torch.manual_seed(42)                       # <-- any integer you like

train_set, test_set = torch.utils.data.random_split(
    dataset,
    [train_len,  test_len]          # lengths in the same order
)

workers    = 2
# Batch size
batch_size = 10

#train dataloader
train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,          # we usually want shuffling *only* for training
    num_workers=workers,
    pin_memory=True
)

#test dataloader
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    num_workers=workers,
    pin_memory=True
)

# In[3]:


"""
Custom weights initialization
"""  
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 1.0) #nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 0.0, 1.0)
        nn.init.constant_(m.bias.data, 0.0) #nn.init.constant_(m.bias.data, 0.0)


# In[4]:


"""
Declare the model
"""
model = mod.TvGenerator(var.NGPU,batch_size).to(device)
if (device.type == 'cuda') and (var.NGPU > 1):
    model = nn.DataParallel(model, list(range(var.NGPU)))

if var.LOAD == False:
    model.apply(weights_init)
else:
    PATH = "model_weights/model{0}_b2_{1}x{2}.pt".format(model_name,var.NT,var.NX)
    model.load_state_dict(torch.load(PATH, map_location=device))

#print("")
# Print the model
print(model)
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
if var.GAUGE_EQ==False:
    data_points = 6*var.NX*var.NT*train_len
else:
    data_points = 2*var.NX*var.NT*train_len
predicted_data = 4*var.NX*var.NV_PRED*var.NT
ratio = np.round(params/data_points,4)
print("Total number of (trainable) model parameters",params)
print("Total number of real data points (6·Nx·Nt·train_len)",data_points)
print("Total number of expected predicted real numbers",predicted_data)
print("Parameters/data points = {0}".format(ratio))
if ratio > 1.0:
    print("Model needs more data to prevent overfit")
#summary(model, input_size=(6, var.NT, var.NX))


# In[10]:


# Learning rate for optimizers
lr = 0.01
# Beta1 hyperparameter for Adam optimizers
beta1 = 0.9
beta2 = 0.999
lamb = 0
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2),weight_decay=lamb)
#optimizer = torch.optim.SGD(model.parameters(), lr=1e-3,momentum=0.9)


# In[11]:


import time

start = time.time()

epochs = 2
losses = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_loader, model, optimizer,losses,var.VERSION)
print("Done!")

end = time.time()
print(end - start,"s")
#20 epochs double precision 921s , NV = 30, NV_PRED= 15
#20 epochs single precision 759 s, NV = 30, NV_PRED= 15


# # Check loss on test set

# In[13]:

print("Computing loss on test set...")
test_loss, dtest_loss, test_batch_losses = evaluate(test_loader, model, device,var.VERSION)
print(f'Test average loss: {test_loss:.6f} +- {dtest_loss:.6f}')


if var.SAVE_W == True:
    model_name = "{0}_b{1}_{2}x{3}".format(model_name,var.BETA,var.NT,var.NX)
    # --- Save metadata ---
    saver = ut.MetadataSaver(
        model,
        model_name,
        epochs,
        lr,
        ratio,
        beta1,
        beta2,
        lamb,
        losses[-1],
        test_loss,
        )    
    saver.save("model_weights/"+model_name+".json")
    PATH = "model_weights/"+model_name+".pt"
    torch.save(model.state_dict(), PATH)     
    print("Model saved to", PATH)


#ut.SavePredictions(test_loader, model, device)
#ut.SavePredictions(train_loader, model, device)

