"""
WGAN 128 x 128 (IN/OUT)
"""

import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, channels_img, features_d):

        super(Critic, self).__init__()
        self.disc = nn.Sequential(

            # input: N x channels_img x 128 x 128
            nn.Conv2d(
                channels_img
                , features_d
                , kernel_size=4
                , stride=2
                , padding=1)
            , nn.LeakyReLU(0.2)

            , self._block(features_d, features_d * 2, 4, 2, 1)
            , self._block(features_d * 2, features_d * 4, 4, 2, 1)
            , self._block(features_d * 4, features_d * 8, 4, 2, 1)
            , self._block(features_d * 8, features_d * 16, 4, 2, 1) # 4 x 4

            , nn.Conv2d(
                features_d * 16
                , 1
                , kernel_size=4
                , stride=2
                , padding=0 ) 
            # output : N x 1 X 1 x 1
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        
        return nn.Sequential(
            nn.Conv2d( 
                in_channels 
                , out_channels
                , kernel_size 
                , stride
                , padding 
                , bias=False)
            , nn.InstanceNorm2d(out_channels, affine=True)
            , nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g):

        super(Generator, self).__init__()
        self.net = nn.Sequential(

            # Input: N x z_dim x 1 x 1
            self._block(z_dim, features_g * 32, 4, 1, 0)  # img: 4x4
            , self._block(features_g * 32, features_g * 16, 4, 2, 1)  # img: 8x8
            , self._block(features_g * 16, features_g * 8, 4, 2, 1)  # img: 16x16
            , self._block(features_g * 8, features_g * 4, 4, 2, 1)  # img: 32x32
            , self._block(features_g * 4, features_g * 2, 4, 2, 1)  # img: 64x64

            , nn.ConvTranspose2d( 
                features_g * 2
                , channels_img
                , kernel_size=4
                , stride=2
                , padding=1
            )
            , nn.Tanh() # [-1, 1]
            # Output: N x channels_img x 128 x 128
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):

        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels
                , out_channels
                , kernel_size
                , stride
                , padding
                , bias=False
            )
            , nn.BatchNorm2d(out_channels)
            , nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


def weights_init(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def test():
    N, in_channels, H, W = 8, 3, 128, 128
    noise_dim = 100

    x = torch.randn((N, in_channels, H, W))
    z = torch.randn((N, noise_dim, 1, 1))

    cri = Critic(in_channels, 8)
    gen = Generator(noise_dim, in_channels, 8)

    assert cri(x).shape == (N, 1, 1, 1), "Critic test failed"
    assert gen(z).shape == (N, in_channels, H, W), "Generator test failed"

    print('Test passed ...')

test()

