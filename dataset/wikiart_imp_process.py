import os
import PIL
from PIL import Image

# ------------------------------------------------------------------------
# path : where image to be processed
path = '/home/mnpr_term/Documents/VersionControlled/Art-Generation-GANs/Dataset/WikiArt/symbolic-painting/'

dirs = os.listdir( path )
dir_length = len(dirs)
print(f'\nNumber of Contents in Directory: {dir_length}\n')

# Collect image names in the directory
images = [file for file in dirs if file.endswith(('.jpg','.jpeg'))]
print(f'Number of Images: {len(images)}\n')
print(f'Sample Image name : \n\t{images[0]}\n')

# ------------------------------------------------------------------------

# change to working directory path
os.chdir(path)

# ------------------------------------------------------------------------
# O/P img resized dimension
img_shape = (200, 200)

# ------------------------------------------------------------------------
# process all images in folder
for image in images:
    
    if os.path.isfile(path+image):
#         fname, extension = os.path.splitext(image)

        try:
            img = Image.open(image)
            img = img.convert('RGB')
            img = img.resize(img_shape, Image.ANTIALIAS)
            img.save(image, optimize=True, quality=90)

        except IOError as e:
            print('Exception : "%s"' %e)

# ------------------------------------------------------------------------


