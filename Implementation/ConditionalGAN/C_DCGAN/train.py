import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.utils import make_grid

from torch.utils.data import DataLoader
import torchvision.datasets as datasets

from model import Generator, Discriminator, weights_init
from art_dataset import ArtDataset
from utils import *

print('-------------------------------------------------------------------')
print('>>> Dependencies Loaded')
torch.manual_seed(111)

# Device Info : 
#-------------------------------------------------------------------
print('>>> Computation Unit Info')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cuda_info = f"""
Device : GPU {torch.cuda.device(device)}
-------------------------------------------------------------------
    Count  : {torch.cuda.device_count()}
    Name   : {torch.cuda.get_device_name(device=None)}
    Arch   : {torch.cuda.get_arch_list()}
-------------------------------------------------------------------
{torch.cuda.get_device_properties(device)}
-------------------------------------------------------------------
"""

if device == torch.device('cuda'):
    print(cuda_info)
elif device == torch.device('cpu'):
    print('Device : CPU')

# Parameters etc.
#-------------------------------------------------------------------

ART_SHAPE = (3, 64, 64)
NUM_CLASSES = 4
IMAGE_SIZE = 64 # To resize the images
IMAGE_CHANNELS = 3
Z_DIM = 64

LEARNING_RATE = 2e-4
NUM_EPOCHS = 200

device = 'cuda'
display_step = 100
BATCH_SIZE = 64

# Image Transformation
#-------------------------------------------------------------------
transform = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(IMAGE_CHANNELS)], [0.5 for _ in range(IMAGE_CHANNELS)]),
    ]
)

# Datasets and Loader [./wikiArt]
#-------------------------------------------------------------------
dataset = datasets.ImageFolder('wikiArt', transform=transform)
# dataset = ArtDataset(dataset_dirs, transform=transform)
# dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms, download=False)

dataloader = DataLoader( dataset, batch_size=BATCH_SIZE, shuffle=True )
print('>>> Data Loader ready for N/W Input')

# Initialize Generator, Discriminator, and Optimizers
#-------------------------------------------------------------------

# Get Input dimension for G, D
    # G --> size of input vector [ concatenated noise, class vector ]
    # D --> add a channel for every class 
generator_input_dim, discriminator_im_chan = get_input_dimensions(Z_DIM, ART_SHAPE, NUM_CLASSES)

# Initialize the Network
gen = Generator(input_dim=generator_input_dim).to(device)
disc = Discriminator(im_chan=discriminator_im_chan, hidden_dim=64).to(device)
print('>>> Generator and Discriminator Initialized')

# Initialize Optimizers
gen_opt = torch.optim.Adam(gen.parameters(), lr=LEARNING_RATE)
disc_opt = torch.optim.Adam(disc.parameters(), lr=LEARNING_RATE)
print('>>> Optimizers Initialized')

# Initialize weights
gen = gen.apply(weights_init)
disc = disc.apply(weights_init)
print('>>> Network weights initialized ')

arch_info = f"""
Network Architectures :
-------------------------------Generator---------------------------
{gen}
-------------------------------Discriminator----------------------

{disc}
------------------------------------------------------------------
"""

print(arch_info)

# Define Loss Criterion
criterion = nn.BCEWithLogitsLoss()


# Initial Values
step = 0
generator_losses = []
discriminator_losses = []

result_folder = 'results'
if not os.path.exists(result_folder):
    os.makedirs(result_folder, exist_ok=True)
      

# Training
#----------------------------------------
for epoch in range(NUM_EPOCHS):
    
    # Dataloader returns the batches and the labels
    for batch_idx, (real, labels) in enumerate(dataloader):
        
        cur_batch_size = len(real)
        
        # Flatten the batch of real images from the dataset
        real = real.to(device)

        one_hot_labels = get_one_hot_labels(labels.to(device), NUM_CLASSES)
        
        image_one_hot_labels = one_hot_labels[:, :, None, None]
        image_one_hot_labels = image_one_hot_labels.repeat(1, 1, ART_SHAPE[1], ART_SHAPE[2])
        
        # Discriminator Training
        # ------------------------------------------------------------------------
        disc_opt.zero_grad()
        
        # Get noise corresponding to the current batch_size 
        fake_noise = get_noise(cur_batch_size, Z_DIM, device=device)      
        noise_and_labels = combine_vectors(fake_noise, one_hot_labels)
        
        fake = gen(noise_and_labels)
        
        fake_image_and_labels = combine_vectors(fake, image_one_hot_labels)
        real_image_and_labels = combine_vectors(real, image_one_hot_labels)

        disc_fake_pred = disc(fake_image_and_labels.detach())
        disc_real_pred = disc(real_image_and_labels)
               
        disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
        disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
        disc_loss = (disc_fake_loss + disc_real_loss) / 2
        disc_loss.backward(retain_graph=True)
        disc_opt.step() 

        # Keep track of the average discriminator loss
        discriminator_losses += [disc_loss.item()]

        # Generator Training
        # ------------------------------------------------------------------------
        gen_opt.zero_grad()

        fake_image_and_labels = combine_vectors(fake, image_one_hot_labels)
                                                 
        disc_fake_pred = disc(fake_image_and_labels)
        gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
        gen_loss.backward()
        gen_opt.step()

        # Keep track of the generator losses
        generator_losses += [gen_loss.item()]

        # Statistics and Images to be saved
        # ------------------------------------------------------------------------
        if step % display_step == 0 and step > 0:
                                        
            gen_mean = sum(generator_losses[-display_step:]) / display_step
            disc_mean = sum(discriminator_losses[-display_step:]) / display_step
            
            # Training Log to the console
            # -------------------------------------------------------------------
            print(
                f"Step: {step}, Epoch: [{epoch}/{NUM_EPOCHS}] Batch: {batch_idx}/{len(dataloader)} \
                  Loss D: {disc_mean:.4f}, loss G: {gen_mean:.4f}"
            )            
            
            # Save Loss function Plot
            # -------------------------------------------------------------------
            step_bins = 20
            x_axis = sorted([i * step_bins for i in range(len(generator_losses) // step_bins)] * step_bins)
            num_examples = (len(generator_losses) // step_bins) * step_bins
            
            fig = plt.gcf()
            fig.set_size_inches(8,6)
            
            plt.figure()
            plt.plot(
                range(num_examples // step_bins), 
                torch.Tensor(generator_losses[:num_examples]).view(-1, step_bins).mean(1),
                label="Generator Loss"
            )
            
            plt.plot(
                range(num_examples // step_bins), 
                torch.Tensor(discriminator_losses[:num_examples]).view(-1, step_bins).mean(1),
                label="Discriminator Loss"
            )
            
            plt.legend()
            plt.savefig(result_folder + "/loss%03d.png" % epoch)
            plt.close()   

            # Save real and fake images
            # -------------------------------------------------------------------
            show_tensor_images(fake, figure_name=(result_folder + "/sample%03d.png" % epoch), show=False)
            show_tensor_images(real, figure_name=(result_folder + "/real%03d.png" % epoch), show=False)
            
        elif step == 0:
            print('--------------------<< Training Started >>---------------------------')
            print('>>> It may take a while ...  c[_] ')
        step += 1


print('>>> Saving gen and disc checkpoints')
torch.save(gen.state_dict(), 'G.ckpt')
torch.save(disc.state_dict(), 'D.ckpt')

print('--------------------<< Training Completed >>----------------------------')