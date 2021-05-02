import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard

from mlp_model import Generator, Discriminator, weights_init
from mlp_utils import *

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
DATASET_PATH = '../../../dataset/celebA/'
SAMPLES_DIR = 'gen_samples_celeb'
STATS_DIR = 'statistics_celeb'

DISPLAY_STEPS = 100 # Display/ Log  Steps

IMG_DIM = 128 * 128 * 3 # 49152
IMG_SHAPE = ( 3, 128, 128 )
IMG_CHANNELS = IMG_SHAPE[0]

# Hyper-parameters
# ----------------------------------------------------
LEARNING_RATE = 1e-4
BETA_ADAM = ( 0.5, 0.999 )

BATCH_SIZE = 32
NUM_EPOCHS = 100

Z_DIM = 100
FEATURES_GEN = 100
FEATURES_DISC = 100

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
- uncomment the following line and comment custom dataset
"""

# dataset = datasets.MNIST(
#     root="dataset/"
#     , train=True
#     , transform=transforms
#     , download=True )


dataset = datasets.ImageFolder( root=DATASET_PATH, transform=transform )

data_loader = DataLoader( dataset, batch_size=BATCH_SIZE, shuffle=True )

print('\n>>> Data Loader with Transformation is Ready')

# Network, Optimizers, and Weight Initialization
# ----------------------------------------------------

# Discriminator and Generator init
gen = Generator( Z_DIM, IMG_DIM, FEATURES_GEN ).to(device)
disc = Discriminator( IMG_DIM, FEATURES_DISC ).to(device)
print('\n>>> Network [D]isciminator & [G]enerator Initialized')

# Optimizer init 
opt_disc = optim.Adam( disc.parameters(), lr = LEARNING_RATE, betas = BETA_ADAM )
opt_gen = optim.Adam( gen.parameters(), lr = LEARNING_RATE, betas = BETA_ADAM )
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

# Loss criterion
criterion = nn.BCELoss()


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
fixed_noise = torch.randn(( BATCH_SIZE, Z_DIM )).to(device)

writer_fake = SummaryWriter(f"logs_celeb/fake")
writer_real = SummaryWriter(f"logs_celeb/real")

# ----------------------------------------------------
# Training
# ----------------------------------------------------

step = 0
for epoch in range( NUM_EPOCHS ):

    for batch_idx, (real, _) in enumerate(data_loader):

        noise = torch.randn( BATCH_SIZE, Z_DIM ).to(device)
        
        fake = gen(noise)
        real = real.view(-1, IMG_DIM).to(device) # Flatten 

        # Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        # ----------------------------------------------------
        disc_real = disc(real).view(-1) # flatten
        disc_loss_real = criterion(disc_real, torch.ones_like(disc_real))

        disc_fake = disc(fake).view(-1)
        disc_loss_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

        # Total disc loss = Agerage of Disc(fake) & Disc(real)
        disc_loss = (disc_loss_real + disc_loss_fake) / 2

        # keep track of discriminator
        discriminator_losses += [disc_loss.item()]

        # Reset Gradients
        disc.zero_grad()

        # Back propagation
        disc_loss.backward( retain_graph=True )
        opt_disc.step()

        ### Train Generator: min log(1 - D(G(z))) <--equivalent-to--> max log(D(G(z))
        # The second option of maximizing is used which doesn't suffer from saturating gradients
        # ----------------------------------------------------
        
        output = disc(fake).view(-1)
        gen_loss = criterion(output, torch.ones_like(output))

        # keep track of generator loss
        generator_losses += [gen_loss.item()]

        # Reset Gradients
        gen.zero_grad()

        # Backward Propagation
        gen_loss.backward()
        opt_gen.step()


        # Logging, saving loss function and Images
        if step % DISPLAY_STEPS == 0 and step > 0:

            gen_mean = sum(generator_losses[-DISPLAY_STEPS:]) / DISPLAY_STEPS
            disc_mean = sum(discriminator_losses[-DISPLAY_STEPS:]) / DISPLAY_STEPS

            print(
                f"Steps: {step}, Epoch: [{epoch}/{NUM_EPOCHS}] Batch: {batch_idx}/{len( data_loader )} \
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

            # Save generated samples
            # show_tensor_images(fake, figure_name=(SAMPLES_DIR + "/sample%03d.png" % epoch), show=False)
            # show_tensor_images(real, figure_name=(SAMPLES_DIR + "/real%03d.png" % epoch), show=False)

            # Tensorboard Logging
            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 3, 128, 128)
                data = real.reshape(-1, 3, 128, 128)

                # Image Grid [ ]
                img_grid_fake = torchvision.utils.make_grid(fake[:16], normalize=True)
                img_grid_real = torchvision.utils.make_grid(data[:16], normalize=True)

                # Write Images to the grids
                writer_fake.add_image(
                    "Fake Art Images", img_grid_fake, global_step=step, 
                )
                writer_real.add_image(
                    "Real Art Images", img_grid_real, global_step=step
                )

        elif step == 0:
            print('-------------------<< Training Started >>---------------------------')
            print('>>> It may take a while ...  c[_] \n')

        step += 1

print('\n>>> Saving gen and disc checkpoints')

torch.save( gen.state_dict(), 'G.ckpt' )
torch.save( disc.state_dict(), 'D.ckpt' )

print('\n--------------------<< Training Completed >>----------------------------')