import torch
import torch.nn as nn

class Generator(nn.Module):

    def __init__(self, z_dim, img_dim, hidden_dim):
        super().__init__()
        
        self.gen = nn.Sequential(
            
            self._block(z_dim, hidden_dim),
            self._block(hidden_dim, hidden_dim * 2),
            self._block(hidden_dim * 2 , hidden_dim * 4),
            self._block(hidden_dim * 4, hidden_dim * 8),
            nn.Linear(hidden_dim * 8 , img_dim),
            
            # normalize inputs to [-1, 1] so make outputs [-1, 1]
            nn.Tanh()
        )

    def _block(self, input_dim, output_dim):

        return nn.Sequential(

            nn.Linear(input_dim, output_dim),
            # nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )

    # Forward Propagation
    def forward(self, x):
        return self.gen(x)

class Discriminator(nn.Module):

    def __init__(self, img_dim, hidden_dim ):
        super().__init__()

        self.disc = nn.Sequential(

            self._block( img_dim, hidden_dim * 8  ),
            self._block( hidden_dim*8 , hidden_dim * 4 ),
            self._block( hidden_dim * 4 , hidden_dim * 2 ),
            self._block( hidden_dim * 2, hidden_dim ),

            nn.Linear( hidden_dim, 1 ),
            nn.Sigmoid()
        )

    # Repeating Block
    def _block(self, input_dim, output_dim):

        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LeakyReLU(0.2) 
        )

    # Forward Propagation
    def forward(self, x):
        return self.disc(x)

def weights_init(model):
    if type(model) == nn.Linear:
        nn.init.xavier_uniform_(model.weight)
        model.bias.data.fill_(0.01)