"""
Training of WGAN-GP
"""


import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import os
import numpy as np
import matplotlib.pyplot as plt
import pylab

from utils import gradient_penalty, save_checkpoint, load_checkpoint
from model import Discriminator, Generator, initialize_weights

# Device cpu or gpu
device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters etc.
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
IMAGE_SIZE = 64
CHANNELS_IMG = 3 
Z_DIM = 100
NUM_EPOCHS = 5
FEATURES_CRITIC = 64
FEATURES_GEN = 64
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10

# Results to be saved
sample_dir = 'samples'
save_dir = 'save'

# Create a directory if not exists
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]),
    ]
)

# dataset = datasets.MNIST(root="data/", transform=transforms, download=True)
dataset = datasets.ImageFolder(root="art/", transform=transforms)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

# initialize gen and disc, note: discriminator should be called critic,
# according to WGAN paper (since it no longer outputs between [0, 1])
gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
critic = Discriminator(CHANNELS_IMG, FEATURES_CRITIC).to(device)

# initialize weights
initialize_weights(gen)
initialize_weights(critic)

# initializate optimizer
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

gen.train()
critic.train()

# Statistics to be saved:
d_losses = np.zeros(NUM_EPOCHS)
g_losses = np.zeros(NUM_EPOCHS)
real_scores = np.zeros(NUM_EPOCHS)
fake_scores = np.zeros(NUM_EPOCHS)

# Tensorboard Summary
fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")

# Denormalize Image
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

# Tensorboard steps
step = 0
total_step = len(loader)

# Training 
for epoch in range(NUM_EPOCHS):
    # Target labels not needed! <3 unsupervised

    for batch_idx, (real, _) in enumerate(loader):
        
        real_labels = torch.ones(BATCH_SIZE, 1)
        real_labels = Variable(real_labels)
        fake_labels = torch.zeros(BATCH_SIZE, 1)
        fake_labels = Variable(fake_labels)
        
        real = real.to(device)
        cur_batch_size = real.shape[0]

        # Train Critic: max E[critic(real)] - E[critic(fake)]
        # equivalent to minimizing the negative of that
        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
            fake = gen(noise)

            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)

            gp = gradient_penalty(critic, real, fake, device=device)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
            )
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

            # Statistics to be saved
            fake_score = critic_fake
            real_score = critic_real

            # Update statistics (from Critic)
            d_losses[epoch] = d_losses[epoch]*(batch_idx/(batch_idx+1.)) + loss_critic.data*(1./(batch_idx+1.))
            real_scores[epoch] = real_scores[epoch]*(batch_idx/(batch_idx+1.)) + real_score.mean().data*(1./(batch_idx+1.))
            fake_scores[epoch] = fake_scores[epoch]*(batch_idx/(batch_idx+1.)) + fake_score.mean().data*(1./(batch_idx+1.))

        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        gen_fake = critic(fake).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Update statistics ( from Generator)
        g_losses[epoch] = g_losses[epoch]*(batch_idx/(batch_idx+1.)) + loss_gen.data*(1./(batch_idx+1.))
        
        # Print losses occasionally and print to tensorboard
        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                  Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
            )

            # Tensorboard Images in batches
            with torch.no_grad():
                fake = gen(fixed_noise)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1
    
    # Save real Images
    if(epoch+1) == 1:
        images = real.view(real.size(0), CHANNELS_IMG, IMAGE_SIZE, IMAGE_SIZE)
        save_image(denorm(images[0].data), os.path.join(sample_dir, 'real_images.png'))

    # Save sampled images
    fake_images = fake.view(fake.size(0), CHANNELS_IMG, IMAGE_SIZE, IMAGE_SIZE)
    save_image(denorm(fake_images[0].data), os.path.join(sample_dir, 'fake_images-{}.png'.format(epoch+1)))
    # Save and plot Statistics
    np.save(os.path.join(save_dir, 'd_losses.npy'), d_losses)
    np.save(os.path.join(save_dir, 'g_losses.npy'), g_losses)
    np.save(os.path.join(save_dir, 'fake_scores.npy'), fake_scores)
    np.save(os.path.join(save_dir, 'real_scores.npy'), real_scores)
    
    # Save loss pdf
    plt.figure()
    pylab.xlim(0, NUM_EPOCHS + 1)
    plt.plot(range(1, NUM_EPOCHS + 1), d_losses, label='d loss')
    plt.plot(range(1, NUM_EPOCHS + 1), g_losses, label='g loss')    
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss.pdf'))
    plt.close()

    # Save accuracy pdf
    plt.figure()
    pylab.xlim(0, NUM_EPOCHS + 1)
    pylab.ylim(0, 1)
    plt.plot(range(1, NUM_EPOCHS + 1), fake_scores, label='fake score')
    plt.plot(range(1, NUM_EPOCHS + 1), real_scores, label='real score')    
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'accuracy.pdf'))
    plt.close()

    # Save model at checkpoints
    if (epoch+1) % 50 == 0:
        torch.save(gen.state_dict(), os.path.join(save_dir, 'G--{}.ckpt'.format(epoch+1)))
        torch.save(critic.state_dict(), os.path.join(save_dir, 'C--{}.ckpt'.format(epoch+1)))

# Save the model checkpoints 
torch.save(gen.state_dict(), 'G.ckpt')
torch.save(critic.state_dict(), 'D.ckpt')
