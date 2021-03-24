import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [15, 15]

from utils import  get_truncated_noise, show_tensor_images
from ProGAN import StyleGen

# Parameters:

z_dim = 128
out_ch = 3
truncation = 0.7

my_stylegan = StyleGen(
    z_dim = z_dim
    , map_hidden_dim = 1024
    , w_dim = 496
    , out_ch = out_ch
    , in_ch = 512
    , kernel_size = 3
    , hidden_chan = 256
)

test_samples = 10
test_result = my_stylegan(get_truncated_noise(test_samples, z_dim, truncation))

viz_samples = 10
viz_noise = get_truncated_noise(viz_samples, z_dim, truncation) * 10

my_stylegan.eval()

images = []
for alpha in np.linspace(0,1,num=5):
    my_stylegan.alpha = alpha
    viz_results, _, _ = my_stylegan(
        viz_noise,
        return_intermediate=True )
    images += [tensor for tensor in viz_results]

show_tensor_images(torch.stack(images), nrow=viz_samples, num_images=len(images))
my_stylegan = my_stylegan.train()

