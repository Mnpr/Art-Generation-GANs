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
from torch.utils.tensorboard import SummaryWriter

from model import Critic, Generator, weights_init

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
DATASET_PATH = '../../../dataset/wikiart/abstract/'
SAMPLES_DIR = 'gen_samples_wc'
STATS_DIR = 'statistics_wc'

DISPLAY_STEPS = 100 # Display/ Log  Steps

IMG_DIM = 128 * 128 * 3 # 49152
IMG_SHAPE = ( 3, 128, 128 )
IMG_CHANNELS = IMG_SHAPE[0]

# Hyper-parameters
# ----------------------------------------------------
LEARNING_RATE = 5e-4
# BETA_ADAM = ( 0.5, 0.999 )

BATCH_SIZE = 16
NUM_EPOCHS = 10

Z_DIM = 100
FEATURES_GEN = 64
FEATURES_DISC = 64

CRITIC_ITERATIONS = 5
WEIGHT_CLIP = 0.01

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

# Network, Optimizers, and Weight Initialization
# ----------------------------------------------------

# Discriminator and Generator init
gen = Generator( Z_DIM, IMG_CHANNELS, FEATURES_GEN ) .to(device)
critic = Critic( IMG_CHANNELS, FEATURES_CRITIC ).to(device)
print('\n>>> Network [D]isciminator & [G]enerator Initialized')

# Optimizers Initializationi
opt_gen = optim.RMSprop(gen.parameters(), lr=LEARNING_RATE)
opt_critic = optim.RMSprop(critic.parameters(), lr=LEARNING_RATE)
print('\n>>> Optimizers for [D] & [G] Initialized')

# Initialize weights
gen = gen.apply(weights_init)
critic = critic.apply(weights_init)
print('\n>>> Network weights initialized ')


arch_info = f"""
-------------------------------------------------------------------
Network Architectures :
--------------------<< Generator >>--------------------------------
{gen}
--------------------<< Discriminator >>----------------------------
{critic}
"""

print(arch_info)


# Statistics to be saved
# ----------------------------------------------------
if not os.path.exists(SAMPLES_DIR):
    os.makedirs(SAMPLES_DIR)

if not os.path.exists(STATS_DIR):
    os.makedirs(STATS_DIR)

generator_losses = []
critic_losses = []

# Tensorboard writer
# ----------------------------------------------------
fixed_noise = torch.randn(( BATCH_SIZE, Z_DIM, 1, 1 )).to(device)

writer_fake = SummaryWriter(f"logs_wc/fake")
writer_real = SummaryWriter(f"logs_wc/real")

step = 0

# ----------------------------------------------------
# Training
# ----------------------------------------------------
for epoch in range(NUM_EPOCHS):

    for batch_idx, (real, _) in enumerate(data_loader):
        
        real = real.to(device)
        
        # Train Critic
        # ----------------------------------------------------
        for _ in range(CRITIC_ITERATIONS):

            noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1).to(device)
            fake = gen(noise)
            
            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)

            # - ve of minimization from optimization = + maximization
            critic_loss = -(torch.mean(critic_real) - torch.mean(critic_fake))

            # Keep track of critic losses
            critic_losses += [critic_loss.item()]

            # Reset Gradient
            critic.zero_grad()
            
            # Back propagation
            critic_loss.backward(retain_graph=True) # fake computation for generator
            opt_critic.step()

            # Weight Clipping
            for p in critic.parameters():
                p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP) # _ = Inplace
        
        # Train Generator: min -E[critic(gen_fake)]
        # ----------------------------------------------------
        output = critic(fake).reshape(-1)

        gen_loss = -torch.mean(output)

        # keep track of generator loss
        generator_losses += [gen_loss.item()]

        # Reset Gradient
        gen.zero_grad()

        # Backward propagation
        gen_loss.backward()
        opt_gen.step()


        # Logging, saving loss function and Images
        if step % DISPLAY_STEPS == 0 and step > 0:

            gen_mean = sum(generator_losses[-DISPLAY_STEPS:]) / DISPLAY_STEPS
            critic_mean = sum(critic_losses[-DISPLAY_STEPS:]) / DISPLAY_STEPS

            print(
                f"Steps: {step}, Epoch: [{epoch}/{NUM_EPOCHS}] Batch: {batch_idx}/{len( data_loader )} \
                  Loss D: {critic_mean:.4f}, loss G: {gen_mean:.4f}"
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
                torch.Tensor(critic_losses[:num_examples]).view(-1, step_bins).mean(1),
                label="Critic Loss"
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
torch.save( critic.state_dict(), 'C.ckpt' )

print('\n--------------------<< Training Completed >>----------------------------')

