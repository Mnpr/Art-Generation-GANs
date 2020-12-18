# Exploration of  Art Generation using Deep Generative models.

***

**>>> Biweekly Report 1.** ( $25^{th} Nov - 9^{th} Dec : 2020$ )

***

**Contents**

1. Implementation Status/ Progress
2. Observations
   - MNIST Dataset
   - Art Dataset
3. Image Generation with Generative models ( Exploration )
   - Generative Models
   - Generative Adversarial Learning
4. Multilayer Perceptron GAN using PyTorch ( Implementation )
   - MNIST Dataset ( Ref: `gan.py` or `VanillaGAN.ipynb` )
   - Art Dataset ( Ref: `ganArt.py`)
5. Conclusion
   - Criterias, Drawbacks, and Possible Solutions ( Insight next iteration )
6. Reference/ Candidate Papers

***

**>>> Submitted by :** Sudesh Acharya (932253) @FH-Kiel

***

## 1. Implementation Status/ Progress

### Current Iteration

**[ :white_check_mark: Completed ] [ :stopwatch: Time Approx(Hours)  ] [ :computer: Implementaiton | :spiral_notepad: Documentation ]**

- [x] [ 7 ]  Generative Models Concepts and Documentation

- [x] [ 6 ]  Exploration/ Documentation of Mathematics behind GAN

- [x] [ 8 ]  Implementation of MLP-GAN ( MNIST Dataset )

- [x] [ 2.5 ]  Hyper/Parameter Optimization

- [x] [ 4 ]  Implementation of MLP-GAN ( Art Dataset )

- [x] [ 4.5 ]  Observation on different Hyperparameters  ( Art Dataset ).

- [x] [ 3.5 ]  Exploration of Different Candidate Methods for Art Generation.

- [x] [ 4 ]  Report Documentation.

- [ ] [ ]  Stability in GANs, Optimization Techniques ( Add to next iteration )

**Total Time (Hours)** : 39.5

## 2. Observations

### Generated Samples ( MNIST Dataset )

#### Hyperparameters

```python
latent_size = 64
hidden_size = 256
image_size = 784
num_epochs = 5
batch_size = 32
```

##### Epoch 1. (Random noise)

![](./sample-Epoch1.png)

##### Epoch 10. ( simple representation )

![](./sample-Epoch10.png)

##### Epoch 50. ( Almost realistic )

![](./sample-Epoch49.png)

### Generated Samples (ART Dataset )

```python
latent_size = 256
hidden_size = 64
image_size = 49152 # 128*128*3
num_epochs = 50
batch_size = 4
```

##### Epoch 1. (Random RGB noise )

![](./fake_art-1.png)

##### Epoch 10.

![](./fake_art-10.png)

##### Epoch 50

![](./fake_art-49.png)

## 3. Image Generation( Sythesis ) with Generative Models.

Image Generation is the task of genetating new image | given existing images.

- Generating image unconditionally from dataset. i.e. $y$ is termed as **Unconditional image generation**. whereas generating samples conditionally( Subtask ), based on labels i.e. $p(y|x)$, as **Conditional image generation**.

### Generative Models.

- **Generative modeling** tries to form the representation of probability distribution(i.e. Density Estimation), which explains( | represents) collection of input training examples. 
- Generetive Adversarial Network generate new samples from generator network given the collection of inputs without estimating Density functions improving with adversarial discriminator network supervising the generated samples.

#### Maximum Likelihood

*Generative Modeling* based on Performing Maximum Likelihood: [[Reference 1.]](https://arxiv.org/pdf/1701.00160.pdf)

![](./generative.png)

Maximum likelihood is Defined as :

$$
\boxed{\theta^* = arg_\theta max\mathbb{E}_{x\sim p_{data}}\log p_{model}(x | \theta )}
$$

Where :

- $x$ is the vector describing the input.
- $p_{model}(x)$ is the density function that the model describes. 
- $p_{model}(x | \theta)$ is the distibution controlled by parameter$\theta$, which describes the concentratration/spread of the data.
- Performing maximum likelihood consists of measuring the log probability that ($p_{model}(x)$) assigns to each input ($x$) data points and adjusting the paramter $\theta$ to increase that probability.

### Generative Adversarial Network

- **Generative Adversarial Network(GAN)** [[Reference 2.]](https://arxiv.org/pdf/1406.2661.pdf) is composed of : a generator(G) & a discriminator (D). It is used to generate new samples from learned latent space.

![](./gan.png)
[[@imgsource]](https://www.kdnuggets.com/wp-content/uploads/generative-adversarial-network.png)

The generator output is connected directly to the discriminator input. Through backpropagation, the discriminator's classification provides a signal that the generator uses to update its weight

**Generative vs Discriminative models**

- Generative models capture the joint probability $p(X, Y)$, or just $p(X)$ if there are no labels.
- Discriminative models capture the conditional probability $p(Y | X)$
- Discriminative models try to draw boundaries in the data space, while generative models try to model how data is placed throughout the space.

#### Generator

- It tries to generate fake data from randomly generated noise G(z), which are harder to discriminate each iteration, from real ones.

- It learns to generate plausible data. The generated instances become negative training examples for the discriminator

#### Discriminator

- It learns to distinguish the generator's fake data from real data. The discriminator penalizes the generator for producing implausible results.

### MiniMax Game | *Adversarial Learning*

- Gererator(G) wins (i.e. learns to create realistic data ) when Discriminator(D) can't differentiate generated data from the real one.
- The loss function below maximizes the function $D(x)$, and also minimizes $D(G(z))$. where $x$: real data, $G(z)$: generrated data. 

$$
\boxed{\min_G \max_D V(D, G)= \mathbb{E}_{x\sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z\sim p_z(z)}[\log(1 - D(G(z)))]} 
$$

**Generator :**

$\because x$ is the actual image, $D(x) = 1$ ,  generator tries to increase the value of $D(G(z))$ (i.e. Probability of being real data )

- Training G : Maximizing the probability of $D$ making mistakes by generating data as realistic as possible.

**Discriminator :**

$\because x$ is the actual image, $D(x) = 1$, discriminator tries to decrease the value of $D(G(z))$ towards 0 (i.e. fake data )

#### Binary Cross Entropy Loss( `BCE-Loss` ):

- v : inputs, w: weights, y : targets, N : batch size

$$
\boxed{L = {\{l_1, ... , l_N\}}^T, l_i = -w_i\left[ y_i \cdot \log(v_i) + (1-y)\cdot \log(1-v_i)\right]}
$$

#### Discriminator Loss:

$$
\boxed{D_{o}=\frac{1}{m} \sum_{i=1}^{m}\left[ \log D(x^{(i)}) + \log (1-D(G(z^{(i)}))) \right]} 
$$

1. If $v_i = D(x_i)$ and $y_i=1 \forall i$ in the `BCE-Loss above ⬆️` : Loss related to real images.
2. If $v_i = D(G(x))$ and $y_i=0 \forall i$ : Loss related to fake images.
3. Sum of `1` and `2` : **`minibatch-loss`** for the Discriminator.

#### Generator Loss:

$$
\boxed{G_{o}=\frac{1}{m} \sum_{i=1}^{m}\log \left({1 - D\left(G\left(z^{(i)}\right)\right)}\right)} 
$$

- If $v_i = D(G(z_i))$ and $y_i = 1 \forall i$ : Loss needed to be minimized.
- Train the generator to maximize $\log \left(D(G(z)))\right)$ ( provides stronger gradients early in training [[Reference 1.](https://arxiv.org/pdf/1406.2661.pdf)] rather than minimizing $\log \left( 1- D(G(z))\right)$ 

## 4. Multilayer Perceptron GAN ( PyTorch )

Implementation Reference : [ Notebook or Python files attached ]

### Instruction :

**Datasets**

1. MNIST Handwritten digits : Downloaded from Torch Datasets 
2. Art Dataset : 
   - ( `./data/1/< images here >` ) `|` create if !exists already
   - ( `/1/` due to recursive reading from Dataloader - Pytorch )

*Note:* Given Datasets reside in their own directory containing `gan.py | ganArt.py`

**Dependencies**

- PyTorch : `pip install torch torchvision`
- Matplotlib: `python -m pip install -U pip`
- Numpy: `pip install numpy`

**Jupyter-notebook | Open in Colab**

```
< Run `jupyter-lab` | Upload to googleColabFolder >

< Navigate to VanillaGAN.ipynb >
```

**Python Interpreter**

```python
python gan.py 

|OR| python ganArt.py
```

## 5. Conclusion

The ability of Generative Adversarial Networks to consume unstructured data makes it applicable to several interesting problem as soon as it can be extended beyond training data.  It can be used as Data augmentation to train better Classifiers, Generate uncertain events, and the learned representation can be used for other tasks.

Training a GAN requires finding a Nash equilibrium of a Minimax game ( Documented below `⬇️` ). Sometimes gradient descent finds it, sometimes doesn’t and there is not a a good equilibrium finding algorithm yet, so GAN training is unstable. This is because the function GANs try to optimize is a loss function that essentially has no closed form (unlike standard loss functions like log-loss or squared error), optimizing this loss function is very hard and requires a lot of **trial-and-error** regarding the network structure and training protocol [@Ref-Quora](https://www.quora.com/What-are-the-pros-and-cons-of-using-generative-adversarial-networks-a-type-of-neural-network-Could-they-be-applied-to-things-like-audio-waveform-via-RNN-Why-or-why-not?share=1) 

Although MLP GANs were able to generate realistic samples for single channel MNIST handwritten digit set, it is hard to learn representation and generate realistic samples due to the sensitivity of Hyperparameters on the model, Size of training set, and the similiraty of each training samples representing a single class. Different combination of hyperparameter set were considered and the samples observed were not close to realistic. The time during next iteration will be spent on adding Convolutional layers ( works well with images ) in the GANs and the different hyperparameter optimization concepts.

## 6.Reference | Candidate Papers

#### Candidate Papers

1. [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
   - [Deep Convolutional GAN](https://arxiv.org/pdf/1511.06434.pdf)
   - [Self Attention GAN](https://github.com/heykeetae/Self-Attention-GAN)
   - [Super Resolution GAN](https://arxiv.org/pdf/1609.04802.pdf)
2. [Denoising Diffusion Probabilistic models](https://github.com/lucidrains/denoising-diffusion-pytorch) 
   - ( Interesting generative results but need to research about Diffusion probabilistic models)

### References

- [[ Reference 1.] NIPS 2016 - Generative Adversarial Networks, Ian Goodfellow OpenAI](https://arxiv.org/pdf/1701.00160.pdf)

- [[ Reference 2.] Generative Adversarial Networks, Ian J Goodfellow and et. al. ( June 2014)](https://arxiv.org/pdf/1406.2661.pdf)

****

### Next iteraton ( Implementation Plan )

- [ ] GAN with Convolutional Layers ( Improve on MLP-GAN )
- [ ] Effect of Different Hyperparameters on Model Performance.
- [ ] Implementation of Art Dataset with DC-GAN
- [ ] Exploration of Candidate Methods ( From Previous iteration )

***
