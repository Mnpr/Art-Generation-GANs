import os
import numpy as np
from pathlib import Path

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

DATASET_PATH = Path('../../../dataset/wikiart/portrait/')
SAMPLES_DIR = Path('./gen_samples')
STATS_DIR = Path('./statistics')

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

print('\n>>> Data Loader is Ready')
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


# Statistics to Save
# ----------------------------------------------------

# Saving Directories
if not os.path.exists(SAMPLES_DIR):
    os.makedirs(SAMPLES_DIR)

if not os.path.exists(STATS_DIR):
    os.makedirs(STATS_DIR)

d_losses = np.zeros(NUM_EPOCHS)
g_losses = np.zeros(NUM_EPOCHS)
real_scores = np.zeros(NUM_EPOCHS)
fake_scores = np.zeros(NUM_EPOCHS)

# Tensorboard Logging
# ----------------------------------------------------

fixed_noise = torch.randn(( BATCH_SIZE, Z_DIM )).to(device)

writer_fake = SummaryWriter(f"logs/fake")
writer_real = SummaryWriter(f"logs/real")

step = 0

# ----------------------------------------------------
# Training
# ----------------------------------------------------
print('\n>>> Start Training <<<\n')

for epoch in range( NUM_EPOCHS ):

    for batch_idx, (real, _) in enumerate(data_loader):

        real = real.view(-1, 128 * 128 * 3).to(device) # Flatten

        batch_size = real.shape[0]

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        # ----------------------------------------------------
        noise = torch.randn( batch_size, Z_DIM ).to(device)
        fake = gen(noise)

        disc_real = disc(real).view(-1) # flatten
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))

        disc_fake = disc(fake).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

        # Total disc loss = Agerage of Disc(fake) & Disc(real)
        lossD = (lossD_real + lossD_fake) / 2

        # Reset Gradients
        disc.zero_grad()

        # Back propagation
        lossD.backward( retain_graph=True )
        opt_disc.step()

        ### Train Generator: min log(1 - D(G(z))) <--equivalent-to--> max log(D(G(z))
        # The second option of maximizing is used which doesn't suffer from saturating gradients
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))

        # Reset Gradients
        gen.zero_grad()

        # Backward Propagation
        lossG.backward()
        opt_gen.step()

        # Console Logging
        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{ NUM_EPOCHS }] Batch {batch_idx}/{len(data_loader)} \
                      Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 3, 128, 128)
                data = real.reshape(-1, 3, 128, 128)

                # Image Grid [ ]
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                # Write Images to the grids
                writer_fake.add_image(
                    "Fake Art Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Real Art Images", img_grid_real, global_step=step
                )

                step += 1

print('\n>>> Training Complete ')
torch.save(G.state_dict(), 'G.ckpt')
torch.save(D.state_dict(), 'D.ckpt')

print('\n>>> Model checkpoint Saved')