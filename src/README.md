## Development Instructions

1. Dependencies

2. Remote GPU-Server

3. Local Setup

### 1. Dependencies

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

### 2. Remote GPU-Server setup

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

### 3. Local Setup

**Instructions :**

- Changing the dataset directory and IMG_CHANNELS in `train.py`

- Dataset residing in a recursive folder `e.g. data/1/<images-here>`

- To train the model  : `python train.py`

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