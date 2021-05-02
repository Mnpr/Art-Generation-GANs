import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import make_grid
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter

from c_dcgan_model import Generator, Discriminator, weights_init
from art_dataset import ArtDataset
from utils import *

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

# Parameters
# ----------------------------------------------------
DATASET_PATH = '../../dataset/wikiart_conditional/'
SAMPLES_DIR = 'gen_samples'
STATS_DIR = 'statistics'

DISPLAY_STEPS = 100 # Display/ Log  Steps

IMG_SHAPE = ( 3, 64, 64 )
IMG_CHANNELS = IMG_SHAPE[0]
NUM_CLASSES = 4

# Hyper-parameters
# ----------------------------------------------------
LEARNING_RATE = 1e-4
BETA_ADAM = ( 0.5, 0.999 )

BATCH_SIZE = 16
NUM_EPOCHS = 100

Z_DIM = 100
FEATURES_GEN = 64
FEATURES_DISC = 64

print('>>> Parameters Defined ')


# Image Transformations [ Resize, Convert2Tensor, and Normalzie ]
#-------------------------------------------------------------------
transform = transforms.Compose(
    [
        transforms.Resize( IMG_SHAPE[1] ),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(IMG_CHANNELS)], [0.5 for _ in range(IMG_CHANNELS)]),
    ]
)

# Dataset and Loader
# ----------------------------------------------------
"""[ for MNIST Dataset ]

- Change Image Shape and Dimension ( 1 x 28 x 28 )
- uncomment the following dataset.MNIST line and comment custom dataset
"""

# dataset = datasets.MNIST(
#     root="dataset/"
#     , train=True
#     , transform=transforms
#     , download=True )


dataset = datasets.ImageFolder( root=DATASET_PATH, transform=transform )
data_loader = DataLoader( dataset, batch_size=BATCH_SIZE, shuffle=True )
print('\n>>> Data Loader with Transformation is Ready')

# Initialize Generator, Discriminator, and Optimizers
#-------------------------------------------------------------------

# Get Input dimension for G, D
    # G --> size of input vector [ concatenated noise, class vector ]
    # D --> add a channel for every class 
generator_input_dim, discriminator_im_chan = get_input_dimensions(Z_DIM, IMG_SHAPE, NUM_CLASSES)

# Initialize the Network
gen = Generator(input_dim=generator_input_dim).to(device)
disc = Discriminator(im_chan=discriminator_im_chan, hidden_dim=64).to(device)
print('\n>>> Network [D]isciminator & [G]enerator Initialized')

# Initialize Optimizers
gen_opt = torch.optim.Adam(gen.parameters(), lr=LEARNING_RATE)
disc_opt = torch.optim.Adam(disc.parameters(), lr=LEARNING_RATE)
print('\n>>> Optimizers for [D] & [G] Initialized')


# Initialize weights
gen = gen.apply(weights_init)
disc = disc.apply(weights_init)
print('\n>>> Network weights initialized ')


arch_info = f"""
-------------------------------------------------------------------
Network Architectures :
--------------------<< Generator >>--------------------------------
{gen}
--------------------<< Discriminator >>----------------------------
{disc}
"""

print(arch_info)

# Define Loss Criterion
criterion = nn.BCEWithLogitsLoss()

# Statistics to be saved
# ----------------------------------------------------
if not os.path.exists(SAMPLES_DIR):
    os.makedirs(SAMPLES_DIR)

if not os.path.exists(STATS_DIR):
    os.makedirs(STATS_DIR)

generator_losses = []
discriminator_losses = []

# Tensorboard writer
# ----------------------------------------------------
fixed_noise = torch.randn(( BATCH_SIZE, Z_DIM, 1, 1 )).to(device)

writer_fake = SummaryWriter(f"logs/fake")
writer_real = SummaryWriter(f"logs/real")


# ----------------------------------------------------
# Training
# ----------------------------------------------------

step = 0
for epoch in range(NUM_EPOCHS):
    
    # Dataloader returns the batches and the labels
    for batch_idx, (real, labels) in enumerate(data_loader):
        
        cur_batch_size = len(real)
        
        # Flatten the batch of real images from the dataset
        real = real.to(device)

        one_hot_labels = get_one_hot_labels(labels.to(device), NUM_CLASSES)
        
        image_one_hot_labels = one_hot_labels[:, :, None, None]
        image_one_hot_labels = image_one_hot_labels.repeat(1, 1, IMG_SHAPE[1], IMG_SHAPE[2])
        
        # Discriminator Training
        # ------------------------------------------------------------------------
        
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

        # Keep track of the average discriminator loss
        discriminator_losses += [disc_loss.item()]

        # Reset gradients
        disc_opt.zero_grad()

        # backward propagation
        disc_loss.backward(retain_graph=True)
        disc_opt.step() 

        

        # Generator Training
        # ------------------------------------------------------------------------
        
        fake_image_and_labels = combine_vectors(fake, image_one_hot_labels)
                                                 
        disc_fake_pred = disc(fake_image_and_labels)
        gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))

        # Keep track of the generator losses
        generator_losses += [gen_loss.item()]

        # Reset gradient 
        gen_opt.zero_grad()

        # Backward Propagation
        gen_loss.backward()
        gen_opt.step()

        

        # Statistics and Images to be saved
        # ------------------------------------------------------------------------
        if step % DISPLAY_STEPS == 0 and step > 0:
                                        
            gen_mean = sum(generator_losses[-DISPLAY_STEPS:]) / DISPLAY_STEPS
            disc_mean = sum(discriminator_losses[-DISPLAY_STEPS:]) / DISPLAY_STEPS
            
            # Training Log to the console
            # -------------------------------------------------------------------
            print(
                f"Step: {step}, Epoch: [{epoch}/{NUM_EPOCHS}] Batch: {batch_idx}/{len(data_loader)} \
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
            plt.savefig(STATS_DIR + "/loss%03d.png" % epoch)
            plt.close()

            # Tensorboard Logging
            # with torch.no_grad():
            #     fake = gen(fixed_noise).reshape(-1, 3, 64, 64)
            #     data = real.reshape(-1, 3, 64, 64)

            #     # Image Grid [ ]
            #     img_grid_fake = torchvision.utils.make_grid(fake[:16], normalize=True)
            #     img_grid_real = torchvision.utils.make_grid(data[:16], normalize=True)

            #     # Write Images to the grids
            #     writer_fake.add_image(
            #         "Fake Art Images", img_grid_fake, global_step=step, 
            #     )
            #     writer_real.add_image(
            #         "Real Art Images", img_grid_real, global_step=step
            #     )

            # Save real and fake images
            # -------------------------------------------------------------------
            show_tensor_images(fake, figure_name=(SAMPLES_DIR + "/sample%03d.png" % epoch), show=False)
            show_tensor_images(real, figure_name=(SAMPLES_DIR + "/real%03d.png" % epoch), show=False)
            
        elif step == 0:
            print('--------------------<< Training Started >>---------------------------')
            print('>>> It may take a while ...  c[_] ')
        step += 1


print('>>> Saving gen and disc checkpoints')
torch.save(gen.state_dict(), 'G.ckpt')
torch.save(disc.state_dict(), 'D.ckpt')

print('--------------------<< Training Completed >>----------------------------')