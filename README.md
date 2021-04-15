This repository contains Master Projectwork :arrow_down: undertaken for the partial fulfillment of M.S. in Information Engineering @Fachhochschule-Kiel under the supervision of  Prof. Dr. Hauke Schramm

# Exploration of Art Generation using Deep Generative models (GANs).

Objective : Series of progressive exploration and experimentation of Deep Generative Models on subset of Wiki-Art dataset to produce Realistic art Images. 

## :art: Datasets

*Source* : [Wiki-Art : Visual Art Encyclopedia](https://www.wikiart.org/)

**Dataset Contents for Unconditional Generation**

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

**photo2fourcollection for multi-collection style transfer**

- `train_content`, `test_content`, `train_styles`

## :information_source: Instructions / Information

- [System Setup](https://github.com/Mnpr/Art-Generation-GANs/blob/main/src/README.md)

- [ Dataset Instructions (Extract[ Download ], Transform, & Load)](https://github.com/Mnpr/Art-Generation-GANs/blob/main/dataset/README.md)

- [Training and Logging](https://github.com/Mnpr/Art-Generation-GANs/blob/main/src/README.md)

## :computer: Explored / Implemented Architectures

- [x] [Unconditional Generation](https://github.com/Mnpr/Art-Generation-GANs/tree/main/src/unconditional_generation)
  
  - [x] FullyConnected MLP-GAN
  
  - [x] DeepConvolutional-GAN
  
  - [x] Wasserstein-GAN
    
    - [x] Weight Clipping
    - [x] Gradient Penalty

- [x] [Conditional Generation](https://github.com/Mnpr/Art-Generation-GANs/tree/main/src/conditional_generation/)
  
  - [x] Conditional-DCGAN

- [x] [Style Based Generation](https://github.com/Mnpr/Art-Generation-GANs/tree/main/src/style_based_generation)
  
  - [x] Neural Style Transfer
  - [x] Multi Collection Style Transfer( GatedGAN )

## :bookmark_tabs: References

- [ 1 ] [ Generative Adversarial Nets [ arXiv:1406.2661v1  [stat.ML]  10 Jun 2014 ]](https://papers.nips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf)

- [ 2 ] [ Conditional Generative Adversarial Nets [arXiv:1411.1784v1  [cs.LG]  6 Nov 2014]](https://arxiv.org/pdf/1411.1784.pdf)

- [ 3 ] [ Unsupervised Representation Learning with Deep Convolutional GANs [ arXiv:1511.06434v2  [cs.LG]  7 Jan 2016 ]](https://arxiv.org/pdf/1511.06434.pdf)


- [ 4 ] [ Improved Techniques for Training GANs [ arXiv:1606.03498v1  [cs.LG]  10 Jun 2016 ]](https://papers.nips.cc/paper/2016/file/8a3363abe792db2d8761d6403605aeb7-Paper.pdf)


- [ 5 ] [ Wasserstein GAN [ arXiv:1701.07875v3  [stat.ML]  6 Dec 2017 ]](https://arxiv.org/pdf/1701.07875.pdf)


- [ 6 ] [ Improved Training with WGAN-GP [ arXiv:1704.00028v3  [cs.LG]  25 Dec 2017 ]](https://arxiv.org/pdf/1704.00028.pdf)


- [ 7 ] [ Gated GAN: Adversarial Gated Networks for Multi-Collection Style Transfer [ arXiv:1904.02296v1  [cs.CV]  4 Apr 2019 ]](https://arxiv.org/pdf/1904.02296.pdf)


- [ 8 ] [ VGG-19 as Feature(style/semantics) Extractor - PyTorch Documentation ](https://pytorch.org/hub/pytorch_vision_vgg/)


- [ 9 ] [ Coursera (Deeplearning.ai) - Build Basic GANs [ Online Course ] ](https://www.coursera.org/learn/build-basic-generative-adversarial-networks-gans/)


- [ 10 ] [ GitHub - colemiller94/gatedgan: PyTorch Implementation of Lua [https://github.com/xinyuanc91/Gated-GAN]](https://github.com/colemiller94/gatedgan)



***