import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def show_tensor_images(image_tensor, figure_name='default_name', num_images=16, size=(3, 128 ,128), nrow=4, show=True):

    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=nrow)
    
    fig = plt.gcf()
    fig.set_size_inches(8,8)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    
    if show:
        plt.show()
        plt.close()
    else:
        plt.savefig(figure_name)
        plt.close() 