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

from utils import gradient_penalty, save_checkpoint, load_checkpoint
from model import Critic, Generator, initialize_weights


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
DATASET_PATH = '../../../../dataset/wikiart/abstract/'
SAMPLES_DIR = 'gen_samples_gp_abstract'
STATS_DIR = 'statistics_gp_abstract'

DISPLAY_STEPS = 100 # Display/ Log  Steps

# Hyperparameters etc.
LEARNING_RATE = 1e-4
BETA_ADAM = ( 0.0, 0.9 )

BATCH_SIZE = 16
NUM_EPOCHS = 100

IMAGE_SIZE = 128
CHANNELS_IMG = 3

FEATURES_CRITIC = 64
FEATURES_GEN = 64
Z_DIM = 100

CRITIC_ITERATIONS = 5
LAMBDA_GP = 10

print('>>> Parameters Defined ')

# Image Transformations [ Resize, Convert2Tensor, and Normalzie ]
#-------------------------------------------------------------------
transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]),
    ]
)

# dataset = datasets.MNIST(root="data/", transform=transforms, download=True)
dataset = datasets.ImageFolder(root=DATASET_PATH, transform=transforms)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize Generator and Critic
gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
critic = Critic(CHANNELS_IMG, FEATURES_CRITIC).to(device)

initialize_weights(gen)
initialize_weights(critic)

print('\n>>> Network weights initialized ')


# initializate optimizer
opt_gen = optim.Adam( gen.parameters(), lr=LEARNING_RATE, betas=BETA_ADAM )
opt_critic = optim.Adam( critic.parameters(), lr=LEARNING_RATE, betas=BETA_ADAM )


print('\n>>> Optimizers initialized ')

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

writer_real = SummaryWriter(f"logs_gp_abstract/real")
writer_fake = SummaryWriter(f"logs_gp_abstract/fake")

step = 0

gen.train()
critic.train()

# ----------------------------------------------------
# Training
# ----------------------------------------------------
for epoch in range(NUM_EPOCHS):

    for batch_idx, (real, _) in enumerate(data_loader):

        real = real.to(device)
        cur_batch_size = real.shape[0]

        # Train Critic: max E[critic(real)] - E[critic(fake)]
        for _ in range(CRITIC_ITERATIONS):

            noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
            fake = gen(noise)

            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)

            # Gradient penalty
            gp = gradient_penalty(critic, real, fake, device=device)

            # maximize = -(to minimize : since optimizer minimizes cost)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
            )

            # Keep track of critic losses
            critic_losses += [loss_critic.item()]

            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        gen_fake = critic(fake).reshape(-1)

        loss_gen = -torch.mean(gen_fake)

        # keep track of generator loss
        generator_losses += [loss_gen.item()]


        gen.zero_grad()
        loss_gen.backward()
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


