# Development/Reproduction Instructions

*>>> : ETL instructions [[Here]](https://github.com/Mnpr/Art-Generation-GANs/blob/main/dataset/README.md)*

### Contents

1. Local Setup

2. Remote Setup

3. Dependencies

4. Training and Logging
 

## 1. Local Setup

**Virtualenv-Python**

```shell
python3 -m venv <directory-name>
source <directory-name>/bin/activate

# path
>>> import sys
>>> sys.path
[''
, '/home/username/miniconda3/lib/python38.zip'
, '/home/username/miniconda3/lib/python3.8'
, '/home/username/venvs/ai/lib/python3.8/site-packages']

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
pip search <package name>

# install
pip install torch torchvision
pip install <package_name> # ==<version> (optional)
pip install requests==2.6.0

# upgrade
pip install --upgrade <pachage_name>

# package info
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

## 2. Remote Setup

**Login**

```
ssh username@server_ip
```

**Info ( GPU )**

```shell
nvidia-smi 
```

**FIle transfer**

```shell
scp -r <source> <destination>
```

**Tmux** 

```shell
# new session
tmux
tmux new -s <session_name>
tmux new -sg #add ganTorch

# list-session
tmux list-sessions

# master key 
    # - options    
<ctrl><b> (+)
    - c : create window
    - % : split vertical
    - p | n : switch previous << and next >> windows
    - " : split horizontal
    - b | o : switch between down | up respectively

    - :<commands>
    - e.g. <ctrl><b> :detach-client


# attach previously detached session
tmux a -t ganTorch

# switch and/or kill sessions
tmux switch -t [0]
tmux kill-session -t [2]
```

## 3. Dependencies

- NumPy : *Numerical( Vectorized ) computation*

- Matplotlib : *Visualization*

- PyTorch : *Neural Network Models Implementation*

- TensorBoard : *Logging and Observations*

**Install Dependencies**

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

## 4. Training and Logging Instructions


### 4.1 Training

For training and observing :

- Training step is performed after [ Dataset Processing ](https://github.com/Mnpr/Art-Generation-GANs/blob/main/dataset/README.md).

- Navigate to the folder containing the implementation of specific network.

- Change/Create Virtual environment with installed dependencies.

- Change Hyper/Parameters if necessary.

- Run `python train_script.py`

### 4.2 Logging

- Simple information about the states and successful execution of model is summarized as console output logs in `train_script.py`

- Tensorboard Logging can be done ( In case the script contains Tensorboard observation elements ) by executing training script and running Tensorboard `tensorboard --logdir=./logs_folder/`

- Navigate to `http://localhost:6006/` >> `Images`

****