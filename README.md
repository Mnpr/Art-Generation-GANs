This repository contains Master Projectwork :arrow_down: undertaken for the partial fulfillment of M.S. in Information Engineering @Fachhochschule-Kiel under the supervision of  Prof. Dr. Hauke Schramm

# Exploration of Art Generation using Deep Generative models (GANs).

Objective : Series of progressive exploration and experimentation of Deep Generative Models on subset of WikiArt dataset to produce Realistic art Images. 

****

## :art: Datasets

*Source* : [Wiki-Art : Visual Art Encyclopedia](https://www.wikiart.org/)


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

**Resized subsets for Conditional Generation**

- `abstract`,`cityscape`, `landscape`,`portrait` ( no. of samples > 5000 )

## :computer: Implementation:

- [x] [Unconditional Generation](https://github.com/Mnpr/Art-Generation-GANs/tree/main/src/unconditional_generation)

  - [x] FullyConnected-GAN
  - [x] DeepConvolutional-GAN
  - [x] Wasserstein-GAN
  
    - [x] Weight Clipping
    - [x] Gradient Penalty

  - [x] MNIST Digits
  - [x] CelebA Face
  - [x] Wikiart  

- [x] [Conditional Generation](https://github.com/Mnpr/Art-Generation-GANs/tree/main/src/conditional_generation/)
  
  - [x] Conditional-DCGAN
  - [x] MNIST Digits 
  - [x] Wikiart

- [x] [Style Based Generation](https://github.com/Mnpr/Art-Generation-GANs/tree/main/src/style_based_generation)

  - [x] Neural Style Transfer
  - [x] Multi Collection Style Transfer( GatedGAN )

- [x] Report/Observations

  - [ ] Refactor and Restructure
  - [ ] Final report Draft/Diagrams/Template
  - [ ] Clean and Complete re-Observations

## :bookmark_tabs: References

1. [Generative Adversarial Nets](https://papers.nips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf)
2. [Unsupervised Representation Learning with Deep Convolutional GANs ](https://arxiv.org/pdf/1511.06434.pdf)
3. [Improved Techniques for Training GANs](https://papers.nips.cc/paper/2016/file/8a3363abe792db2d8761d6403605aeb7-Paper.pdf)
4. [Wasserstein GAN]
5. [Improved Training with WGAN-GP]
6. [Gated GAN]
7. [Style Transfer]
8. [Conditional GAN]
9. ...

***
