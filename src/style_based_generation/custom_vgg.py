import torch
import torch.nn as nn
import torchvision.models as models

model = models.vgg19(pretrained=True).features

class FeatureVGG19(nn.Module):
    def __init__(self):
        super(FeatureVGG19, self).__init__()

        # chosen feature extraction layers
        self.chosen_features  = ['0','5','10','19','28']
        self.model = models.vgg19(pretrained=True).features[:29]

    def forward(self, x):
        features = []

        # Forward propagation with feature layers
        for layer_num, layer in enumerate(self.model):
            x = layer(x)

            # chosen feature extraction layers
            if str(layer_num) in self.chosen_features:
                features.append(x)

        return features