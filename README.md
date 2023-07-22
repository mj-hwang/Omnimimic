# Omnimimic
[**Omni**Gibson](https://behavior.stanford.edu/omnigibson/) + [robo**mimic**](https://robomimic.github.io/)

This repository provides necessary wrappers & pipeline scripts for collecting data and training imitation learning (IL) agents in the [**OmniGibson**](https://behavior.stanford.edu/omnigibson/) environment under the [**robomimic**](https://robomimic.github.io/) framework. 

## Installation
Preliminary: Conda, Issac Sim platform

For installation of OmniGibson and robomimic, installing from source is highly recommended.

If you encounter an error when installing [**egl_probe**](https://github.com/StanfordVL/egl_probe), please install this library from source, instead of from pypi.

### [Install **OmniGibson**](https://behavior.stanford.edu/omnigibson/getting_started/installation.html#setup)

```bash
git clone https://github.com/StanfordVL/OmniGibson.git
cd OmniGibson

source setup_conda_env.sh
conda activate omnigibson

OMNIGIBSON_NO_OMNIVERSE=1 python omnigibson/scripts/setup.py
```

### [Install **robomimic**](https://robomimic.github.io/docs/introduction/installation.html)

```bash
git clone https://github.com/ARISE-Initiative/robomimic.git
cd robomimic
pip install -e .
```



## Exemplary project
[**Distilling MOMA**](https://github.com/mj-hwang/distilling-moma/tree/main) project uses this repository