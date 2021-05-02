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
# import visdom

from utils import *
from data import ImageDataset, ReplayBuffer
from models import Generator \
                   , Discriminator \
                   , Identity \
                   , ResidualBlock \
                   , Encoder \
                   , Transformer \
                   , Decoder \
                   , TVLoss \
                   , weights_init_normal

print('\n>>> Dependencies Loaded')
torch.manual_seed(111)

# Device Info : 
#-------------------------------------------------------------------
print('\n------------------------<< Device Info >>--------------------------------')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cuda_info = f"""
Device     : GPU {torch.cuda.device(device)}
Properties :
{torch.cuda.get_device_properties(device)}
----------------------------------------------------------------------
"""

if device == torch.device('cuda'):
    print(cuda_info)
elif device == torch.device('cpu'):
    print('Device : CPU')

# ---------------------------------------------------------------------------------
# Train Parameters [Gated-GAN]
# ---------------------------------------------------------------------------------

DATASET_PATH = '../../../dataset/photo2fourcollection/'

NUM_EPOCHS = 100
DECAY_EPOCHS=50
BATCH_SIZE = 1

loadSize = 143
fineSize = 128

FEATURES_GEN = 64
FEATURES_DISC = 64  

INPUT_CHANNELS = 3 
OUTPUT_CHANNELS = 3 

LEARNING_RATE= 0.0002 
BETA_ADAM = ( 0.5, 0.999 )
LAMBDA_A = 10.0
TV_STRENGTH=1e-6
AE_CONSTRAIN = 10 
NUM_STYLES = 4

epoch = 0
gpu = 1
pool_size = 50
resize_or_crop = 'resize_and_crop'
cuda=True

print('>>> Parameters Defined ')

# Image Transformations [ Resize, Convert2Tensor, and Normalzie ]
#-------------------------------------------------------------------
transform = transforms.Compose(
    [
        transforms.Resize(int(143), Image.BICUBIC),
        transforms.RandomCrop(128),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ]
)

# ---------------------------------------------------------------------------------
# Dataloader
dataloader = DataLoader( ImageDataset( DATASET_PATH, transform_=transform )
                       , batch_size=1
                       , shuffle=True
                       , num_workers=1 )
                        
batch = next(iter(dataloader))
print('\n>>> Data Loader with Transformation is Ready')

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

# Initialize Generator, Discriminator, and Optimizers
#-------------------------------------------------------------------
generator = Generator(INPUT_CHANNELS, OUTPUT_CHANNELS, NUM_STYLES, FEATURES_GEN)
discriminator= Discriminator(INPUT_CHANNELS, NUM_STYLES, FEATURES_DISC)
print('\n>>> Network [D]isciminator & [G]enerator Initialized')

# Load previous params.
# generator.load_state_dict(torch.load('./netG5.pth'))
# print('-> Dictionary State Loaded [x]\n')


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
criterion_TV = TVLoss(TVLoss_weight=TV_STRENGTH)

# ---------------------------------------------------------------------------------
# Optimization and Learning Rate Schedulers
#Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(generator.parameters(),
                                lr= LEARNING_RATE, betas = BETA_ADAM )
optimizer_D = torch.optim.Adam(discriminator.parameters(), 
                               lr=LEARNING_RATE, betas = BETA_ADAM )
print('\n>>> Optimizers for [D] & [G] Initialized')

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G
                                                   , lr_lambda=LambdaLR(NUM_EPOCHS
                                                                        , epoch
                                                                        , DECAY_EPOCHS).step)
lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D
                                                   , lr_lambda=LambdaLR(NUM_EPOCHS
                                                                        ,epoch
                                                                        , DECAY_EPOCHS).step)
print('\n>>> LR Scheduler initialized')

#Set vars for training
# ---------------------------------------------------------------------------------

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

input_A = Tensor(BATCH_SIZE, INPUT_CHANNELS, fineSize, fineSize)
input_B = Tensor(BATCH_SIZE, OUTPUT_CHANNELS, fineSize, fineSize)

target_real = Variable(Tensor(BATCH_SIZE).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(BATCH_SIZE).fill_(0.0), requires_grad=False)

D_A_size = discriminator(input_A.copy_(batch['style']))[0].size()  
D_AC_size = discriminator(input_B.copy_(batch['style']))[1].size()

class_label_B = Tensor(D_AC_size[0],D_AC_size[1],D_AC_size[2]).long()

autoflag_OHE = Tensor(1,NUM_STYLES+1).fill_(0).long()
autoflag_OHE[0][-1] = 1

fake_label = Tensor(D_A_size).fill_(0.0)
real_label = Tensor(D_A_size).fill_(0.99) 

rec_A_AE = Tensor(BATCH_SIZE,INPUT_CHANNELS,fineSize,fineSize)

fake_buffer = ReplayBuffer()

# Initialize weights
# ---------------------------------------------------------------------------------
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)
print('\n>>> Network weights initialized ')


arch_info = f"""
-------------------------------------------------------------------
Network Architectures :
--------------------<< Generator >>--------------------------------
{generator}
--------------------<< Discriminator >>----------------------------
{discriminator}
"""

print(arch_info)

# ---------------------------------------------------------------------------------
# TRAIN LOOP
# ---------------------------------------------------------------------------------

for epoch in range(epoch,NUM_EPOCHS):
    
    for i, batch in enumerate(dataloader):
        ## Unpack minibatch
        
        # source content, target style, and style label
        real_content = Variable(input_A.copy_(batch['content']))
        real_style = Variable(input_B.copy_(batch['style']))
        style_label = batch['style_label']
        
        # one-hot encoded style
        style_OHE = F.one_hot(style_label,NUM_STYLES).long()
        
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
        errD_real_class = criterion_ACGAN(out_class.transpose(1,3),class_label)*LAMBDA_A
        
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
        err_class = criterion_ACGAN(out_class.transpose(1,3), class_label)*LAMBDA_A
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
        err_ae = criterion_Rec(identity,real_content)*AE_CONSTRAIN
        err_ae.backward()
        optimizer_G.step()
        
        # ---------------------------------------------------------------------------------
        
        # pring losses
        if i % 500 == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {i}/{len(dataloader)} \
                  Loss D: {errD:.4f}, Loss G: {errG_tot:.4f}, Loss AE: {err_ae:.4f} \
                  , Loss AC: {err_class:.4f} " 
            )
            
    ##update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D.step()
    
    print('>>> Saving gen and disc model States ')
    torch.save(generator.state_dict(), 'netG.pth')
    torch.save(discriminator.state_dict(), 'netD.pth')

print('--------------------<< Training Completed >>----------------------------')
