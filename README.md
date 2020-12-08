# Development Instruction MS-Project

## Gpu Server  Setup

**Info :**

- Arch Linux

- Nvidia GPUS ( 2 * 12GB )

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
tmux new -sg#add anTorch

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

## Local Setup
