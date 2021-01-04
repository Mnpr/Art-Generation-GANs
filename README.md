# Art Generation( Deep Generative models )

This Projectwork is series of Exploration and Experimentation with Deep Generative models to produce Realistic art Images.

## Datasets :

- Dimension $( 128 * 128 * 3 )$

- 9349 Art Samples

## Implementation:

- [x] [FullyConnectedMLP-GAN]()

- [x] [DeepConvolutionalGAN]()

- [ ] WGAN

- [ ] StyleGAN

- [ ] SAGAN

- [ ] Denoising Diffusion  ...

## Development Instructions

- Remote GPU-Server

- Local Setup

- Dependencies

### Remote GPU-Server setup

**Login**

```
ssh sacharya@149.222.24.125
```

**Info ( GPU )**

```shell
nvidia-smi 
```

**FIle transfer**

```shell
# copy files from server
scp -r <source> <destination>
scp -r sacharya@149.222.24.125:~/sacharya/server ~/mnpr_term/local

# Upload files to server

scp -r ~/mnpr_term/local sacharya@149.222.24.125:~/sacharya/server 
```

**Tmux**

```shell
# new session
tmux
tmux new -s <session_name>
tmux new -sg #add ganTorch

# list-session
tmux list-sessions

<ctrl><b>:
    - c : create window
    - % : split vertical
    - p | n : switch previous << and next >> windows
    - " : split horizontal
    - b | o : switch between down | up respectively

    - :<commands>
    - :detach-client

#attach session
tmux a -t ganTorch

# kill & switch sessions
tmux switch -t [0]
tmux kill-session -t [2]
```

### Local Setup

**Virtualenv-Python**

```shell
python3 -m venv <directory-name>
source <directory-name>/bin/activate

# path
>>> import sys
>>> sys.path
[''
, '/home/mnpr_term/miniconda3/lib/python38.zip'
, '/home/mnpr_term/miniconda3/lib/python3.8'
, '/home/mnpr_term/venvs/ai/lib/python3.8/site-packages']

# run file
python <file_name.py>

# exit from venv
deactivate
```

**Managing Packages with `pip`**

```python
# list
pip list

# search
pip search <package>

# install
pip install <package_name> # ==<version> (optional)
pip install torch torchvision
pip install requests==2.6.0

# upgrade
pip install --upgrade <pachage_name>

# info
pip show <package_name>

# generate requirements.txt
pip freeze > requirements.txt
```

**Managing Environments/ Packages with `Conda`**

```python
# list
conda list

# search
conda search <package>

# install 
conda install --yes <package1> <package2>

# update packages
conda update conda 

# Info
conda info --envs

# Manage Python
conda create --name aipy python=3.8

# Activate
conda activate aipy

# Deactivate
conda deactivate
```

### Dependencies

- NumPy : *Numerical( Vectorized ) computation*

- Matplotlib : *Visualization*

- PyTorch : *Neural Network Models Implementation*

- TensorBoard : *Logging and Observations*

```python
# Pytorch
pip install torch torchvision
# |OR|
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

# Matplotlib
pip install matplotlib
# |OR|
conda install -c conda-forge matplotlib

# Numpy
pip install numpy
|OR|
conda install numpy

# Tensorboard
pip install tensorboard
# |OR|
conda install -c conda-forge tensorboard
```

****
