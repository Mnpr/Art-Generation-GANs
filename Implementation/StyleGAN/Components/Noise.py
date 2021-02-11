import torch
import torch.nn as nn

class InjectNoise(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(channels)[None, : , None, None])

    def forward(self, image):
        noise_shape = (image.shape[0], 1, image.shape[0], image.shape[3])
        noise = torch.randn(noise_shape, device=image.device)
        return image + self.weight * noise

    