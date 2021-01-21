import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F


# --------------------------------------------------------------------------------------------
def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28), nrow=5, show=True):
    
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=nrow)
    
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    if show:
        plt.show()
        
# --------------------------------------------------------------------------------------------
def get_noise(n_samples, input_dim, device='cpu'):
    
    '''
    Function for creating noise vectors: Given the dimensions (n_samples, input_dim)
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
        n_samples: the number of samples to generate, a scalar
        input_dim: the dimension of the input vector, a scalar
        device: the device type
    '''
    
    return torch.randn(n_samples, input_dim, device=device)

# --------------------------------------------------------------------------------------------
def get_one_hot_labels(labels, n_classes):
    
    '''
    Function for creating one-hot vectors for the labels, returns a tensor of shape (?, num_classes).
    Parameters:
        labels: tensor of labels from the dataloader, size (?)
        n_classes: the total number of classes in the dataset, an integer scalar
    '''
    
    return F.one_hot(labels,n_classes)
# --------------------------------------------------------------------------------------------
def combine_vectors(x, y):
    
    '''
    Function for combining two vectors with shapes (n_samples, ?) and (n_samples, ?).
    Parameters:
      x: (n_samples, ?) the first vector. 
        Here, this will be the noise vector of shape (n_samples, z_dim), 
        
      y: (n_samples, ?) the second vector.
        here this will be the one-hot class vector with the shape (n_samples, n_classes)
    '''
    
    # Note: Make sure this function outputs a float no matter what inputs it receives
    
    combined = torch.cat((x.float(),y.float()), 1)
    return combined
# --------------------------------------------------------------------------------------------
def get_input_dimensions(z_dim, mnist_shape, n_classes):

    '''
    Function for getting the size of the conditional input dimensions 
    from z_dim, the image shape, and number of classes.
    Parameters:
        z_dim: the dimension of the noise vector, a scalar
        mnist_shape: the shape of each MNIST image as (C, W, H), which is (1, 28, 28)
        n_classes: the total number of classes in the dataset, an integer scalar
                (10 for MNIST)
    Returns: 
        generator_input_dim: the input dimensionality of the conditional generator, 
                          which takes the noise and class vectors
        discriminator_im_chan: the number of input channels to the discriminator
                            (e.g. C x 28 x 28 for MNIST)
    '''
    generator_input_dim = z_dim + n_classes
    discriminator_im_chan = mnist_shape[0] + n_classes

    return generator_input_dim, discriminator_im_chan
# --------------------------------------------------------------------------------------------

