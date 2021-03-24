import torch
import torch.nn as nn
import torch.nn.functional as F

from MapNetwork import MappingLayers
from Noise import InjectNoise
from AdaIN import AdaIN

class StyleGenBlock(nn.Module):

    def __init__(self, in_ch, out_ch, w_dim, kernel_size, starting_size, use_upsample=True):
        super().__init__()

        self.use_upsample = use_upsample

        if use_upsample:
            self.use_upsample = nn.Upsample((starting_size), mode='bilinear')
        
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=1)
        self.inject_noise = InjectNoise(out_ch)
        self.adain = AdaIN(out_ch, w_dim)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x, w):

        if self.use_upsample:
            x = self.use_upsample(x)

        x = self.conv(x)
        x = self.inject_noise(x)
        x = self.activation(x)
        x = self.adain(x, w)
        return x

class StyleGen(nn.Module):
    
    def __init__(self
                , z_dim
                , map_hidden_dim
                , w_dim
                , in_ch
                , out_ch
                , kernel_size
                , hidden_chan ):
        super().__init__()

        self.map = MappingLayers(z_dim, map_hidden_dim, w_dim)
        self.starting_constant = nn.Parameter(torch.randn(1, in_ch, 4, 4))

        self.block0 = StyleGenBlock(in_ch, hidden_chan, w_dim, kernel_size, 4, use_upsample=False)
        self.block1 = StyleGenBlock(hidden_chan, hidden_chan, w_dim, kernel_size, 8)
        self.block2 = StyleGenBlock(hidden_chan, hidden_chan, w_dim, kernel_size, 16)

        self.block1_to_image = nn.Conv2d(hidden_chan, out_ch, kernel_size=1)
        self.block2_to_image = nn.Conv2d(hidden_chan, out_ch, kernel_size=1)
        self.alpha = 0.2

    def upsample_to_match_size(self, smaller_image, bigger_image):
        return F.interpolate(smaller_image, size = bigger_image.shape[-2:], mode='bilinear')

    def forward(self, noise, return_intermediate=False):

        x = self.starting_constant
        w = self.map(noise)
        x = self.block0(x, w)

        # First generator run output
        x_small = self.block1(x, w)
        x_small_image = self.block2(x_small, w)
        # Second generator run output        
        x_big = self.block2(x_small, w)
        x_big_image = self.block2_to_image(x_big)
        # Upsample first generator run output size = second generator run output
        x_small_upsample = self.upsample_to_match_size(x_small_image, x_big_image)
        # Interpolate between the upsampled image and the image from generator (using alpha)
        interploation = self.alpha * (x_big_image) + (1-self.alpha) * (x_small_upsample)

        if return_intermediate:
            return interploation, x_small_upsample, x_big_image

        return interploation
        