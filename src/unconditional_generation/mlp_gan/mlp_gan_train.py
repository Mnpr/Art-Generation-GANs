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

from model import Generator, Discriminator


print('\n>>> Dependencies Loaded')

# Parameters
# ----------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_PATH = '../../../dataset/wikiart/portrait/'
SAMPLES_DIR = 'gen_samples'
STATS_DIR = 'statistics'

IMG_DIM = 128 * 128 * 3 
IMAGE_CHANNELS = 3

# Hyper-parameters
LR = 2e-4
Z_DIM = 100
BATCH_SIZE = 16
NUM_EPOCHS = 10
HIDDEN_DIM = 128

print('\n>>> Parameters Defined')


# Image Transformations [ Resize, Convert2Tensor, and Normalzie ]
# ----------------------------------------------------
transform = transforms.Compose(
    [
        transforms.Resize( 128 ),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(IMAGE_CHANNELS)], [0.5 for _ in range(IMAGE_CHANNELS)]),
    ]
)


# Dataset and Loader
# ----------------------------------------------------
dataset = datasets.ImageFolder( root=DATASET_PATH, transform=transform )
data_loader = DataLoader( dataset, batch_size=BATCH_SIZE, shuffle=True )

print('\n>>> Data Loader with Transformation is Ready')
# Network, Optimizers Initialization
# ----------------------------------------------------

# Discriminator and Generator init
gen = Generator( Z_DIM, IMG_DIM, HIDDEN_DIM ).to(device)
disc = Discriminator( IMG_DIM, HIDDEN_DIM ).to(device)
print('\n>>> Network [D]isciminator & [G]enerator Initialized')

# Optimizer init 
opt_disc = optim.Adam(disc.parameters(), lr=LR)
opt_gen = optim.Adam(gen.parameters(), lr=LR)

print('\n>>> Optimizers for [D] & [G] Initialized')

# Loss criterion
criterion = nn.BCELoss()


# Dir of Statistics to Save
# ----------------------------------------------------
if not os.path.exists(SAMPLES_DIR):
    os.makedirs(SAMPLES_DIR)

if not os.path.exists(STATS_DIR):
    os.makedirs(STATS_DIR)

# Tensorboard writer
# ----------------------------------------------------
fixed_noise = torch.randn(( BATCH_SIZE, Z_DIM )).to(device)

writer_fake = SummaryWriter(f"logs/fake")
writer_real = SummaryWriter(f"logs/real")

generator_losses = []
discriminator_losses = []

step = 0
display_step = 300

# ----------------------------------------------------
# Training
# ----------------------------------------------------

for epoch in range( NUM_EPOCHS ):

    for batch_idx, (real, _) in enumerate(data_loader):

        real = real.view(-1, IMG_DIM ).to(device) # Flatten

        batch_size = real.shape[0]
        print(real.shape)

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        # ----------------------------------------------------
        noise = torch.randn( batch_size, Z_DIM ).to(device)
        fake = gen(noise)

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
        output = disc(fake).view(-1)
        gen_loss = criterion(output, torch.ones_like(output))

        # Reset Gradients
        gen.zero_grad()

        # Backward Propagation
        gen_loss.backward()
        opt_gen.step()

        # keep track of generator loss
        generator_losses += [gen_loss.item()]

        # Logging, saving loss function and Images
        if step % display_step == 0 and step > 0:

            gen_mean = sum(generator_losses[-display_step:]) / display_step
            disc_mean = sum(discriminator_losses[-display_step:]) / display_step

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

            # Tensorboard Logging

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 3, 128, 128)
                data = real.reshape(-1, 3, 128, 128)

                # Image Grid [ ]
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                # Write Images to the grids
                writer_fake.add_image(
                    "Fake Art Images", img_grid_fake, global_step=step, 
                )
                writer_real.add_image(
                    "Real Art Images", img_grid_real, global_step=step
                )

        elif step == 0:
            print('\n-------------------<< Training Started >>---------------------------')
            print('>>> It may take a while ...  c[_] ')

        step += 1

print('\n>>> Saving gen and disc checkpoints')

torch.save( gen.state_dict(), 'G.ckpt' )
torch.save( disc.state_dict(), 'D.ckpt' )

print('\n--------------------<< Training Completed >>----------------------------')