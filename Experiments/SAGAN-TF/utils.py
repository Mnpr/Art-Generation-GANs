import scipy.misc
import imageio
import numpy as np
import os
from glob import glob
import tensorflow as tf
import tensorflow.contrib.slim as slim

class ImageData:

    def __init__(self, load_size, channels):
        self.load_size = load_size
        self.channels = channels

    def image_processing(self, filename):
        x = tf.read_file(filename)
        x_decode = tf.image.decode_jpeg(x, channels=self.channels)
        img = tf.image.resize_images(x_decode, [self.load_size, self.load_size])
        img = tf.cast(img, tf.float32) / 127.5 - 1

        return img

incept_count=0

def load_data(dataset_name, size=64) :
    x = glob(os.path.join("./dataset", dataset_name, '*.*'))
    return x

def preprocessing(x, size):
    x = imageio.imread(x, mode='RGB')
    x = scipy.misc.imresize(x, [size, size])
    x = normalize(x)
    return x

def normalize(x) :
    return x/127.5 - 1

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    global incept_count
    incept_count = incept_count + 1
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
            # code to save individual image in the cwd 'results' folder
            data = image*255
            img_incep = data.astype(np.uint8)
            x = str(incept_count) + '_' + str(idx)
            imageio.imwrite(os.getcwd() + '/results/%s.png' % x, img_incep)
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')

def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    data= 255*image
    img = data.astype(np.uint8)
    return imageio.imwrite(path, img)
    # return imageio.imsave(path, merge(images, size))

def inverse_transform(images):
    return (images+1.)/2.

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def str2bool(x):
    return x.lower() in ('true')
