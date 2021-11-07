# Simulated TUNL Experiments and Data Analysis

This repository contains the code needed to train Deep Reinforcement Learning (DRL) agents on the simulated Trial-Unique, Nonmatch-to-Location (TUNL) memory task, collect responses of LSTM neurons in trained DRL agents, and analyze the data.
&nbsp;

The simulated TUNL experiments are conceptualized by [Dongyan Lin](http://dongyanl1n.github.io) and [Blake Richards](http://linclab.org/) and coded by Dongyan Lin.

The experiment details, analyses and results are published in [Lin & Richards, 2021, _bioRxiv_](https://www.biorxiv.org/content/10.1101/2021.07.15.452557v1).
&nbsp;

## Installation
### Dependencies:
- python 3.7
- torch 1.9.0
- scipy 1.7.1
- scikit-learn 0.24.2
- numpy 1.21.2
- matplotlib 3.4.3
- gym 0.20.0
- [linclab-utils-0.0.1](https://github.com/linclab/linclab_utils)

## Contents
- `model.py` contains actor-critic agent backbone used for solving TUNL task in 2D (i.e. birds-eyes view) environment. 
You can customize the architecture of the network by changing the arguments. It also contains functions used for backpropagation.
- `world.py` contains environments TUNL task in 2D (i.e. birds-eyes view) environments and its variations:
  - `Tunl`: original, mnemonic TUNL task
  - `Tunl_nomem`: non-mnemonic TUNL task
  - `Tunl_vd`: mnemonic TUNL task with variable delays 
  - `Tunl_nomem_vd`: non-mnemonic TUNL task with variable delays
- `run.py`: script for training the agent and collecting data
- `analysis.py`: script for running the analysis on collected data to reproduce figures
  - `analysis_helper.py` contains functions used in `analysis.py`
- `1d` directory contains mirroring scripts for collecting data from non-spatial TUNL experiments:
  - `1d/world1d.py`: TUNL tasks in non-spatial environment
    - `TunlEnv`: mnemonic
    - `TunlEnv_nomem`: non-mnemomic
  - `1d/model1d.py`: actor-critic network without CNN
  - `1d/run1d.py`: script for training the agent and collecting data 
  - `1d/analysis1d.py`: script for running the analysis on collected data to reproduce figures

    
## Run
```angular2html
#!/bin/bash
#SBATCH --job-name=tunl
#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt
#SBATCH --ntasks=1
#SBATCH --partition=unkillable
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G

module load python/3.
module load python/3.7/cuda/10.2/cudnn/7.6/pytorch/1.5.0

source venv/bin/activate

python run.py
```

## Contact
For all inquiries regarding the code, please contact **Dongyan Lin** ([dongyan.lin@mail.mcgill.ca](mailto:dongyan.lin@mail.mcgill.ca)).

