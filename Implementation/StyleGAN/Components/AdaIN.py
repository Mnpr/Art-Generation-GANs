import torch
import torch.nn as nn

class AdaIN(nn.Module):

    def __init__(self, channels, w_dim):
        super().__init__()

        self.instance_norm = nn.InstanceNorm2d(channels)

        self.style_scale_transform = nn.Linear(w_dim, channels)
        self.style_shift_transform = nn.Linear(w_dim, channels)

    def forward(self, image, w):
        
        normalized_image = self.instance_norm(image)
        style_scale = self.style_scale_transform(w)[:, : , None, None]
        style_shift = self.style_shift_transform(w)[:, : , None, None]

        transformed_image = style_scale * normalized_image + style_shift
        return transformed_image

