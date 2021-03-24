import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image 


model = models.vgg19(pretrained=True).features

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()

        self.chosen_features  = ['0','5','10','19','28']
        self.model = models.vgg19(pretrained=True).features[:29]

    def forward(self, x):
        features = []

        for layer_num, layer in enumerate(self.model):
            x = layer(x)

            if str(layer_num) in self.chosen_features:
                features.append(x)

        return features

def load_image(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device)

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

image_size = 400

loader = transforms.Compose(
    [
        transforms.Resize((image_size, 800))
        , transforms.ToTensor()
    ]
)

original_img = load_image('frankfurt.jpg')
style_img = load_image('graffiti.jpeg')

model = VGG().to(device).eval()

# generated = torch.randn(original_img.shape, device=device, requires_grad=True)\
generated = original_img.clone().requires_grad_(True)
# Parameters:

total_steps = 6000
learning_rate = 0.001
alpha = 1
beta = 0.01


optimizer = optim.Adam([generated], lr=learning_rate)

for step in range(total_steps):
    generated_features = model(generated)
    original_img_features = model(original_img)
    style_features = model(style_img)

    style_loss = original_loss = 0


    for gen_feat, orig_feat, style_feat in zip(generated_features, original_img_features, style_features ):
        
        batch_size, channel, height, width = gen_feat.shape
        original_loss += torch.mean((gen_feat - orig_feat) **2 )

        # compute Gram matrix
        G = gen_feat.view(channel, height*width).mm(
            gen_feat.view(channel, height*width).t()
        )
        
        A = style_feat.view(channel, height*width).mm(
            style_feat.view(channel, height*width).t()
        )

        style_loss += torch.mean((G-A)**2)

    total_loss = alpha*original_loss + beta * style_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 200 == 0:
        print(total_loss)
        save_image(generated, 'generated.png')