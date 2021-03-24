import os
import math
import glob
import torch
import random
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import visdom


from data import *
from models import *
from utils import *

# ---------------------------------------------------------------------------------
# Train Parameters [Gated-GAN]
# ---------------------------------------------------------------------------------

epoch = 0
n_epochs = 100
decay_epoch=50
batchSize = 1
dataroot = './photo2fourcollection'
loadSize = 143
fineSize = 128
ngf = 64
ndf = 64    
in_nc = 3 
out_nc = 3 
lr = 0.0002 
gpu = 1
lambda_A = 10.0
pool_size = 50
resize_or_crop = 'resize_and_crop'
autoencoder_constrain = 10 
n_styles = 4
cuda=True
tv_strength=1e-6

# ---------------------------------------------------------------------------------
# Dataloader
dataloader = DataLoader( ImageDataset('./photo2fourcollection')
                        ,  batch_size=1
                        , shuffle=True
                        , num_workers=1 )
                        
batch = next(iter(dataloader))

# Batch Information
batch_info = f""" 

\n--------------------------------------------
Batch Info : ( Each item from data loader )
______________________________________________

Dictionary : \n{batch.keys()}\n
Content Shape : \n{batch['content'].shape}\n
Style Shape : \n{batch['style'].shape}\n
St_Label Shape : \n{batch['style_label'].shape}
--------------------------------------------\n
"""
print(batch_info)

# ---------------------------------------------------------------------------------
# Init G, D
generator = Generator(in_nc, out_nc, n_styles, ngf)
discriminator= Discriminator(in_nc, n_styles, ndf)
print('-> Discriminator and Generator  Initialized [x]\n')

# Load previous params.
generator.load_state_dict(torch.load('./netG5.pth'))
print('-> Dictionary State Loaded [x]\n')


if cuda:
    generator.cuda()
    discriminator.cuda()

# ---------------------------------------------------------------------------------
#Losses Init
use_lsgan=True

if use_lsgan:
    criterion_GAN = nn.MSELoss()
else: 
    criterion_GAN = nn.BCELoss()
    
    
criterion_ACGAN = nn.CrossEntropyLoss()
criterion_Rec = nn.L1Loss()
criterion_TV = TVLoss(TVLoss_weight=tv_strength)

# ---------------------------------------------------------------------------------
# Optimization and Learning Rate Schedulers
#Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(generator.parameters(),
                                lr=lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), 
                               lr=lr, betas=(0.5, 0.999))


lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G
                                                   , lr_lambda=LambdaLR(n_epochs
                                                                        , epoch
                                                                        , decay_epoch).step)
lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D
                                                   , lr_lambda=LambdaLR(n_epochs
                                                                        ,epoch
                                                                        , decay_epoch).step)
print('-> Optimizer and LR Scheduler initialized [x]\n')

# ---------------------------------------------------------------------------------
#Set vars for training
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

input_A = Tensor(batchSize, in_nc, fineSize, fineSize)
input_B = Tensor(batchSize, out_nc, fineSize, fineSize)

target_real = Variable(Tensor(batchSize).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(batchSize).fill_(0.0), requires_grad=False)

D_A_size = discriminator(input_A.copy_(batch['style']))[0].size()  
D_AC_size = discriminator(input_B.copy_(batch['style']))[1].size()

class_label_B = Tensor(D_AC_size[0],D_AC_size[1],D_AC_size[2]).long()

autoflag_OHE = Tensor(1,n_styles+1).fill_(0).long()
autoflag_OHE[0][-1] = 1

fake_label = Tensor(D_A_size).fill_(0.0)
real_label = Tensor(D_A_size).fill_(0.99) 

rec_A_AE = Tensor(batchSize,in_nc,fineSize,fineSize)

fake_buffer = ReplayBuffer()

# ---------------------------------------------------------------------------------
##Init Weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

print('-> Dictionary State Loaded [x]\n')

# ---------------------------------------------------------------------------------
# TRAIN LOOP
# ---------------------------------------------------------------------------------

for epoch in range(epoch,n_epochs):
    for i, batch in enumerate(dataloader):
        ## Unpack minibatch
        
        # source content, target style, and style label
        real_content = Variable(input_A.copy_(batch['content']))
        real_style = Variable(input_B.copy_(batch['style']))
        style_label = batch['style_label']
        
        # one-hot encoded style
        style_OHE = F.one_hot(style_label,n_styles).long()
        
        # style Label mapped over 1x19x19 tensor for patch discriminator 
        class_label = class_label_B.copy_(label2tensor(style_label,class_label_B)).long()
        
        # ---------------------------------------------------------------------------------
        # Update Discriminator
        # ---------------------------------------------------------------------------------
        
        optimizer_D.zero_grad()
        
        # Generate style-transfered image
        genfake = generator({
            'content':real_content,
            'style_label': style_OHE})
        
        # Add generated image to image pool and randomly sample pool 
        fake = fake_buffer.push_and_pop(genfake)
        
        # Discriminator forward pass with sampled fake 
        out_gan, out_class = discriminator(fake)
        
        # Discriminator Fake loss (correctly identify generated images)
        errD_fake = criterion_GAN(out_gan, fake_label)
        
        # Backward pass and parameter optimization
        errD_fake.backward()
        optimizer_D.step()
        
        optimizer_D.zero_grad()
        # Discriminator forward pass with target style
        out_gan, out_class = discriminator(real_style)
        
        # Discriminator Style Classification loss
        errD_real_class = criterion_ACGAN(out_class.transpose(1,3),class_label)*lambda_A
        
        # Discriminator Real loss (correctly identify real style images)
        errD_real = criterion_GAN(out_gan, real_label)        
        errD_real_total = errD_real + errD_real_class
        
        # Backward pass and parameter optimization
        errD_real_total.backward()
        optimizer_D.step()
        
        
        errD = (errD_real+errD_fake)/2.0
        
        
        # ---------------------------------------------------------------------------------        
        # Generator Update
        # ---------------------------------------------------------------------------------
        
        ## Style Transfer Loss
        optimizer_G.zero_grad()
        # Discriminator forward pass with generated style transfer
        out_gan, out_class = discriminator(genfake)
        
        # ---------------------------------------------------------------------------------
        # Generator Losses
        
        # Generator gan (real/fake) loss
        err_gan = criterion_GAN(out_gan, real_label)
        # Generator style class loss
        err_class = criterion_ACGAN(out_class.transpose(1,3), class_label)*lambda_A
        # Total Variation loss
        err_TV = criterion_TV(genfake)
        
        # Final Loss
        errG_tot = err_gan + err_class + err_TV 
        errG_tot.backward()
        optimizer_G.step()
        
        # ---------------------------------------------------------------------------------
        ## Auto-Encoder (Reconstruction ) Loss
        optimizer_G.zero_grad()
        identity = generator({
            'content': real_content,
            'style_label': autoflag_OHE,
        })
        err_ae = criterion_Rec(identity,real_content)*autoencoder_constrain
        err_ae.backward()
        optimizer_G.step()
        
        # ---------------------------------------------------------------------------------
        
        # pring losses
        if i % 1000 == 0:
            print(
                f"Epoch [{epoch}/{n_epochs}] Batch {i}/{len(dataloader)} \
                  Loss D: {errD:.4f}, Loss G: {errG_tot:.4f}, Loss AE: {err_ae:.4f} \
                  , Loss AC: {err_class:.4f} " 
            )
            
    ##update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D.step()
    
    #Save model
    torch.save(generator.state_dict(), 'model-states/netG.pth')
    torch.save(discriminator.state_dict(), 'model-states/netD.pth')
