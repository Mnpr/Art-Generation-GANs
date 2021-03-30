import os
import torch
import torchvision
import torch.nn as nn
from PIL import Image
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms

from custom_vgg import FeatureVGG19

print('\n>>> Dependencies Loaded')
torch.manual_seed(111)

# Device Info : 
#-------------------------------------------------------------------
print('\n------------------------<< Device Info >>--------------------------------')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cuda_info = f"""
Device     : GPU {torch.cuda.device(device)}
Properties :
{torch.cuda.get_device_properties(device)}
----------------------------------------------------------------------
"""

if device == torch.device('cuda'):
    print(cuda_info)
elif device == torch.device('cpu'):
    print('Device : CPU')


# Parameters
# ----------------------------------------------------
SAMPLES_DIR = 'samples'

# Create if !exist
if not os.path.exists(SAMPLES_DIR):
    os.makedirs(SAMPLES_DIR)


# Optimization Steps
OPTIM_STEPS = 500
DISPLAY_STEPS = 200 # Steps to show generated image update

IMG_SHAPE = ( 3, 512, 512 )

LEARNING_RATE = 0.001

ALPHA = 1
BETA = 0.01
print('>>> Parameters Defined ')

# Image Transformations [ Resize and Convert2Tensor ]
#------------------------------------------------------
loader = transforms.Compose(
    [
        transforms.Resize(( IMG_SHAPE[1], IMG_SHAPE[2]))
        , transforms.ToTensor()
    ]
)

# Load images style/ content to device
#------------------------------------------------------
def load_image(image_name, data_loader):

    image = Image.open(image_name)
    image = data_loader(image).unsqueeze(0)

    return image.to(device)


# Style and Content Images
CONTENT_IMG = load_image('content/hallstatt.jpg', loader)
STYLE_IMG = load_image('style/thewowstyle.com.jpeg', loader)


# Load Model
# ----------------------------------------------------
model = FeatureVGG19().to(device).eval()
print('\n>>> Custom VGG-19 model Loaded ')

# Generated Image [ Noise of Image Shape ]
# ----------------------------------------------------
generated = CONTENT_IMG.clone().requires_grad_(True)
# generated = torch.randn(original_img.shape, device=device, requires_grad=True)\


# Initialize Optimizer
# ----------------------------------------------------
optimizer = optim.Adam( [ generated ], lr=LEARNING_RATE )
print('\n>>> Optimizers for model Initialized')

arch_info = f"""
----------------------------------------------------------------------

Network Architecture :
----------------------<< VGG-19 >>-------------------------------------
{model}
----------------------------------------------------------------------
"""

print(arch_info)

# ----------------------------------------------------
# Gradient Optimization
# ----------------------------------------------------
for step in range( OPTIM_STEPS ):

    # Original Simantics, Style and Generated Features From model
    generated_features = model( generated )
    original_img_features = model( CONTENT_IMG )
    style_features = model( STYLE_IMG )

    # Initialize Style Loss
    style_loss = original_loss = 0

    # Style Transfer
    # ----------------------------------------------------
    for gen_feat, orig_feat, style_feat in zip( generated_features, original_img_features, style_features ):
        
        batch_size, channel, height, width = gen_feat.shape

        # Original Loss
        original_loss += torch.mean((gen_feat - orig_feat) **2 )

        # compute Gram matrix
        G = gen_feat.view(channel, height*width).mm(
            gen_feat.view(channel, height*width).t()
        )
        
        A = style_feat.view(channel, height*width).mm(
            style_feat.view(channel, height*width).t()
        )

        # Style Loss
        style_loss += torch.mean((G-A)**2)

    # Total Loss
    total_loss = ALPHA * original_loss + BETA * style_loss

    # Reset Optimizer
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()


    if step % DISPLAY_STEPS == 0:
        print(total_loss)

        # Save image every display_step
        torchvision.utils.save_image(generated, 'generated.png')

print('\n--------------------<< Generation Completed >>----------------------------')
