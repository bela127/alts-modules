## Installation:

install pyenv dependencies:

```bash
sudo apt update
sudo apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \
libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
xz-utils tk-dev libffi-dev liblzma-dev python-openssl git
```

install pyenv:

```bash
curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash
```

add it to bash, by adding the following to the bottom of the file  `~/.bashrc`:

```bash
export PATH="/home/user/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

install python version:

```bash
pyenv install 3.9.6
```

install poetry:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

set poetry to use pyenv:

```bash
poetry config virtualenvs.prefer-active-python true
```

GPy still wants an old scipy version, that needs to be build from source:

install build dependencies for scipy (or just let it fail):

```bash
sudo apt-get install gfortran libopenblas-dev liblapack-dev
```

install project dependencies:

```bash
poetry install
```

wait for scipy to build and all other dependencies to install.

Now install manually via pip a new version of scipy (1.8)

```
pip install scipy==1.8
pip install scipy>=1.10.0
```

