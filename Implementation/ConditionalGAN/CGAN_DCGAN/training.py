import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import Discriminator, Generator
from torch.utils.tensorboard import SummaryWriter
import utils_train

torch.manual_seed(0)

# Parameters
# ------------------------------------------------
mnist_shape = (1, 28, 28)
n_classes = 10
n_epochs = 10
z_dim = 64
display_step = 500
batch_size = 128
lr = 0.0002
device = 'cuda'

# Img Transform, Dataloader
# ------------------------------------------------

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

dataloader = DataLoader(
    MNIST('.', download=False, transform=transform),
    batch_size=batch_size,
    shuffle=True)


# Init Weight, Optimizer, and Model
# ------------------------------------------------

criterion = nn.BCEWithLogitsLoss()
generator_input_dim, discriminator_im_chan = utils_train.get_input_dimensions(z_dim, mnist_shape, n_classes)

gen = Generator(input_dim=generator_input_dim).to(device)
disc = Discriminator(im_chan=discriminator_im_chan).to(device)

gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)
        
gen = gen.apply(weights_init)
disc = disc.apply(weights_init)

# for tensorboard plotting
fixed_noise = torch.randn(32, , 1, 1).to(device)
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0


# Training
# ------------------------------------------------

cur_step = 0
generator_losses = []
discriminator_losses = []


for epoch in range(n_epochs):
    
    # Batches, Labels from the dataloader
    for batch_idx, (real, labels) in enumerate(dataloader):
        
        cur_batch_size = real.shape[0]
        
        # Flatten the batch of real images
        real = real.to(device)
        
        # One Hot Labels
        one_hot_labels = get_one_hot_labels(labels.to(device), num_classes)
        img_one_hot_labels = one_hot_labels[:,:, None, None]
        img_one_hot_labels = img_one_hot_labels.repeat(1,1,mnist_shape[1], mnist_shape[2])
        
        
        # --------------------------------------------------
        # Discriminator Training
        # --------------------------------------------------
                
        # Reset Gradients for Discriminator
        disc_opt.zero_grad()
                
        # Get noise for current batch size
        fake_noise = get_noise(cur_batch_size, z_dim, device=device)
        
        # Combine Noise vector + One-hot labels for the generator
        noise_n_labels = combine_vectors(fake_noise, one_hot_labels)
        
        # Discriminar Inputs
        fake_image_and_labels = combine_vectors(fake, img_one_hot_labels)
        real_image_and_labels = combine_vectors(real, img_one_hot_labels)
        
        # Discriminator prediction on fake and real images
        disc_fake_pred = disc(fake_image_and_labels.detach())
        disc_real_pred = disc(real_image_and_labels)
        disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
        disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_perd))
        disc_loss = (disc_fake_loss + disc_real_loss) / 2
        disc_opt.step()
        
        # Keep track of Average discriminator losses
        discriminator_losses += [disc_loss.item()]
        
        # --------------------------------------------------
        # Generator Training
        # --------------------------------------------------
        
        # Reset Gradients for Generator
        gen_opt.zero_grad()
        
        fake_image_and_labels = combine_vectors(fake, img_one_hot_labels)
        disc_fake_pred = disc(fake_image_and_labels)
        
        gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
        gen_loss.backward()
        gen_opt.step()
        
        generator_losses += [gen_loss.item()]
        
        if cur_step % display_step == 0 and cur_step > 0:
            gen_mean = sum(generator_losses[-display_step:]) / display_step
            disc_mean = sum(discriminator_losses[-display_step:]) / display_step
            print(f"Step {cur_step}: Generator loss: {gen_mean}, discriminator loss: {disc_mean}")
            show_tensor_images(fake)
            show_tensor_images(real)
            step_bins = 20
            x_axis = sorted([i * step_bins for i in range(len(generator_losses) // step_bins)] * step_bins)
            num_examples = (len(generator_losses) // step_bins) * step_bins
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
            plt.show()
        elif cur_step == 0:
            print("Start Update --- *** ---")
        cur_step += 1
        
        
        
        
        