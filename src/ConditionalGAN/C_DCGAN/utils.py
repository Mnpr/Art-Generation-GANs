import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.utils import make_grid


def show_tensor_images(image_tensor, figure_name='default_name', num_images=16, size=(3, 64 ,64), nrow=4, show=True):

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

def get_noise(n_samples, input_dim, device='cpu'):
    return torch.randn(n_samples, input_dim, device=device)

def get_one_hot_labels(labels, n_classes): 
    return F.one_hot(labels,n_classes)
    
def combine_vectors(x, y):
    combined = torch.cat((x.float(),y.float()),dim=1)
    
    return combined

def get_input_dimensions(z_dim, img_shape, n_classes):
    generator_input_dim = z_dim + n_classes
    discriminator_im_chan = img_shape[0] + n_classes

    return generator_input_dim, discriminator_im_chan

