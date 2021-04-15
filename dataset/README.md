# Instruction - Dataset Usage [ ETL ]

Contents:

> 1. Extract [ pre-processing ]
> 
> 2. Transform [ Image Processing ]
> 
> 3. Data Loaders

## 1. Extract

**Wiki-Art** [ [Download](https://www.kaggle.com/ipythonx/wikiart-gangogh-creating-art-gan/download) ]

* `unzip wikiart.zip` inside `datasets/` directory
- `tree ./datasets` [ directory structure ]

```shell
Output >>> :

dataset/
    - wikiart/                         # (14)_child_dirs
        - abstract
        - ...
        - ...
        - symbolic-painting
    - photo2fourcollection/
    - wikiart_img_process.py
```

**Photo2fourcollection** [[ Download ]([photo2fourcollection – Google Drive](https://drive.google.com/drive/folders/10N972-REqb1R0rqkAB4jRFuNnFijTEgC))] ( used in [Gated-GAN](https://ieeexplore.ieee.org/document/8463508) paper )

- `unzip photo2fourcollection.zip` inside `datasets/` directory

- `tree ./datasets` [ directory structure ]

```shell
Output >>> :

dataset/
    - photo2fourcollection/
        - test_content/ ...
        - train_content/ ...
        - style_content/
            - cezanne
            - monet
            - ukiyoe
            - vangogh
    - wikiart/ ...
    - wikiart_img_process.py
```

## 2. Transform

**Pre-processing** [ before run-time]

1. wikiart dataset to [ 3 x 200 x 200 ]
   
   - Open `wikiart_img_process.py` script and change `path` variable to directories e.g. `./wikiart/abstract`
   
   - Run `python wikiart_img_process.py`
   
   - Repeat 1, 2, 3 for each desired processing directory.
   
   - ! *may need to perform some manual processing in case transformation exceptions were to occur*

2. photo2fourcollection [ Already Pre-processed ]

**processing** [ during run-time ] 

*Implemented with `train.py` script*

1. wikiart dataset to 
   
   - Resize to `[ 3 x 128 x 128 ]`
   
   - Convert to Tensor
   
   - Normalize( 0.5)

2. photo2fourcollection
   
   - Resize/ Random Crop to `[ 3 x 128 x 128 ]`
   
   - Convert to Tensor
   
   - Normalize

## 3. Load

- Data Loader [ pytorch ] `torch.utils.data.DataLoader` 

- Specify the Batch Size for Training data loader.



For Next step >> [[  Train, Log and System Setup ]]()

****
