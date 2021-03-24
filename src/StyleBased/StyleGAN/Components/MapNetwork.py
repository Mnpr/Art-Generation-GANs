import torch
import torch.nn as nn

class MappingLayers(nn.Module):
    '''
    z_dim: dimension of noise vector
    hidden_dim : the inner dimension scalar
    w_dim: the dimension of the intermediate noise vector (scalar)
    '''
    def __init__(self, z_dim, hidden_dim, w_dim):
        super().__init__()

        self.mapping = nn.Sequential(
            nn.Linear(z_dim, hidden_dim)
            , nn.ReLU()
            , nn.Linear(hidden_dim, hidden_dim)
            , nn.ReLU()
            , nn.Linear(hidden_dim, w_dim)
        )

    def forward(self, noise):
        return self.mapping(noise)
