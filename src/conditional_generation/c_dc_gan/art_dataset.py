
import os
import random
from PIL import Image
from torch.utils.data import Dataset

class ArtDataset(Dataset):
    def __init__(self, image_dirs, transform=None):
        
        def get_images(class_name):
            images = [x for x in os.listdir(image_dirs[class_name]) if x.endswith(('.jpg', '.jpeg'))]
            print(f'{len(images)} examples Found in {class_name}')
            return images
        
        self.images = {}
        self.class_names = [ 'abstract', 'cityscape', 'landscape', 'portrait' ]
        
        for class_name in self.class_names:
            self.images[class_name] = get_images(class_name)
        
        self.transform = transform
        self.image_dirs = image_dirs
        
    def __len__(self):
        
        return sum([len(self.images[class_name]) for class_name in self.class_names])
    
    def __getitem__(self, index):
        
        class_name = random.choice(self.class_names)
        index = index % len(self.images[class_name])
        image_name = self.images[class_name][index]
        image_path = os.path.join(self.image_dirs[class_name], image_name)
        image = Image.open(image_path)
        return self.transform(image), self.class_names.index(class_name)