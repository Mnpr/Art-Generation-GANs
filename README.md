This repository contains Master Projectwork :arrow_down: undertaken for the partial fulfillment of M.S. in Information Engineering @Fachhochschule-Kiel under the supervision of  Prof. Dr. Hauke Schramm

# Exploration of Art Generation using Deep Generative models (GANs).

Objective : Series of progressive exploration and experimentation of Deep Generative Models on subset of WikiArt dataset to produce Realistic art Images. 

## Datasets :

*Source* : [Wiki-Art : Visual Art Encyclopedia](https://www.wikiart.org/)

- [Download Here @Kaggle :arrow_down:](https://www.kaggle.com/ipythonx/wikiart-gangogh-creating-art-gan/download)

**Dataset Contents**

```json
'abstract': 14999,
'animal-painting': 1798,
'cityscape': 6598,
'figurative': 4500,
'flower-painting': 1800,
'genre-painting': 14997,
'landscape': 15000,
'marina': 1800,
'mythological-painting': 2099,
'nude-painting-nu': 3000,
'portrait': 14999,
'religious-painting': 8400,
'still-life': 2996,
'symbolic-painting': 2999
```

**Resized Subsets**

- `abstract`,`cityscape`, `landscape`,`portrait` ( no. of samples > 5000 )

## Implementation:

- [x] [FullyConnected-GAN](https://github.com/Mnpr/MS-Project/tree/main/Implementation/VanillaGAN)

- [x] [DeepConvolutional-GAN](https://github.com/Mnpr/MS-Project/tree/main/Implementation/DCGAN)

- [x] [Wasserstein-GAN](https://github.com/Mnpr/MS-Project/tree/main/Implementation/WGAN)
  
  - [x] Weight Clipping
  - [x] Gradient Penalty

- [x] Conditional-GAN (MNIST)

- [x] [Neural Style Transfer](https://github.com/Mnpr/Art-Generation-GANs/tree/main/Implementation/NeuralStyleTransfer)

- [x] [Multi Collection Style Transfer( GatedGAN )](https://github.com/Mnpr/Art-Generation-GANs/tree/main/Implementation/GatedGAN)

- [x] Wikiart Classes ( Unconditional )

- [x] Wikiart Classes ( Conditional )
  - [x] C_DCGAN

- [ ] Refactor and Restructure
- [ ] Final report Draft/Diagrams/Template
- [ ] Clean Observations Networks



***
