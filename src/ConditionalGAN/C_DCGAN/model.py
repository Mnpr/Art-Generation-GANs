import torch
import torch.nn as nn

class Discriminator(nn.Module):

    def __init__(self, im_chan=3, hidden_dim=64):
        super(Discriminator, self).__init__()
        
        self.disc = nn.Sequential(
            self. _block(im_chan, hidden_dim),
            self. _block(hidden_dim, hidden_dim * 2),
            self. _block(hidden_dim * 2, 1, final_layer=True),
        )

    def  _block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
       
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
            )

    def forward(self, image):
        disc_pred = self.disc(image)

        return disc_pred.view(len(disc_pred), -1)
    

class Generator(nn.Module):
    def __init__(self, input_dim=10, im_chan=3, features_g=64):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        
        self.net = nn.Sequential(

            # Input: N x input_dim x 1 x 1
            self._block(input_dim, features_g * 16, 4, 1, 0),
            self._block(features_g * 16, features_g * 8, 4, 2, 1),
            self._block(features_g * 8, features_g * 4, 4, 2, 1),
            self._block(features_g * 4, features_g * 2, 4, 2, 1),
            nn.ConvTranspose2d(
                features_g * 2
                , im_chan
                , kernel_size=4
                , stride=2
                , padding=1
            ),
            # Output: N x channels_img x 64 x 64
            nn.Tanh(), # [-1, 1]
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, noise):
        
        x = noise.view(len(noise), self.input_dim, 1, 1)
        return self.net(x)
    
    
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)