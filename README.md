# [Learning Domain-specific Causal Discovery from Time-series](https://openreview.net/pdf?id=JFaZ94tT8M)
Could a Neural Network Understand Microprocessor?

## Table Of Contents
-  [Introduction](#introduction)
-  [Requirements](#requirements)
-  [Codebase](#codebase)
-  [Usage](#usage)

## Introduction  
Causal discovery (CD) from time-varying data is important in neuroscience, medicine, 
and machine learning. Techniques for CD encompass randomized experiments, which are 
generally unbiased but expensive, and algorithms such as Granger causality, 
conditional- independence-based, structural-equation-based, and score-based methods 
that are only accurate under strong assumptions made by human designers. However, 
as demonstrated in other areas of machine learning, human expertise is often not 
entirely accurate and tends to be outperformed in domains with abundant data. In 
this study, we examine whether we can enhance domain-specific causal discovery for 
time series using a data-driven ap- proach. Our findings indicate that this procedure 
significantly outperforms human-designed, domain-agnostic causal discovery methods, 
such as Mutual Information, VAR-LiNGAM, and Granger Causality on the MOS 6502 
microprocessor, the NetSim fMRI dataset, and the Dream3 gene dataset. We argue that, 
when feasible, the causality field should consider a supervised approach in which 
domain-specific CD procedures are learned from extensive datasets with known causal 
relationships, rather than being designed by human specialists. Our findings promise 
a new approach toward improving CD in neural and medical data and for the broader 
machine learning community.
## Requirements
Clone the repo:
```
mkdir learning_causal_discovery
git clone https://github.com/CharonWangg/LearningCausalDiscovery.git learning_causal_discovery
```
### For MOS 6502 Simulation (Modified from and inspired by [Sim2600](https://github.com/ericmjonas/Sim2600)):  

* Create a Python 2.7 env
* Setup:
    ```
    conda activate env_py2.7
    cd learning_causal_discovery/nmos_simulation && pip install -r requirements.txt
    cd learning_causal_discovery/nmos_simulation && pip install -e .
    ```
  
* Simulated data has been open-sourced (wget and unzip):
  * [Donkey Kong (from half-clock 0 to 40960)](https://mos-6502.s3.us-east-2.amazonaws.com/compressed/DonkeyKong.zip)
  * [Pitfall (from half-clock 0 to 1024)](https://mos-6502.s3.us-east-2.amazonaws.com/compressed/Pitfall.zip)
  * [SpaceInvaders (from half-clock 0 to 1024)](https://mos-6502.s3.us-east-2.amazonaws.com/compressed/SpaceInvaders.zip)

  ```
  # take DonkeyKong.zip as an example
  ├── DonkeyKong.zip
  │   ├── window_0_128 # period of 128 half-clocks, from half-clock 0 to 128
  │   │   ├── sequence_step_128_rec_30.npy # sequences of all 3510 transistors, reconstructed with a 30 within half-clock steps
  │   │   ├── adjacency_matrix.pkl # cause-effect dictionary, {cause: [effects]}, only have unique cause, but might have effects have duplicated sequences
  │   ├── window_256_384 # period of 128 half-clocks, from half-clock 256 to 384
  ...
  ```

### For NMOS 6502 Inference
* Create a Python 3.9 env
* Install requirements:
    ```
    conda activate env_py3.9
    cd learning_causal_discovery/run && pip install -r requirements.txt
    ```
  
### For Experiments on NetSim
* Download NetSim dataset
```
cd learning_causal_discovery && mkdir .cache
cd .cache
wget https://www.fmrib.ox.ac.uk/datasets/netsim/sims.tar.gz
tar -xzvf sims.tar.gz
```

## Codebase (after preparation)
```
├── README.md
├── work_dir  # work directory for checkpoints and corresponding configs
├── nmos_simulation  # MOS 6502 simulation
│   ├── build # build directory
│   ├── causal_simulation # scripts for simulation
│   ├── sim2600 # core simulation engine code
├── run  # learning procedure related
│   ├── scripts  # scripts for tests and dataset generation
│   ├── configs  # network configs
│   ├── train_all.sh  # shell script for training
│   ├── tools  # deep learning tool
```
## Usage
### MOS 6502 Simulation
First, to acquire multiple simulations of different periods on Donkey Kong and test periods on other two games:
```
conda activate env_py2.7
cd learning_causal_discovery
bash nmos_simulation/causal_simulation/collect_pretrain_data.sh
```
### Learning Procedure
First, to generate the train/val/test .csv files from the state sequences and adjacency matrix acquired from
MOS 6502 Simulation.
```
conda activate env_py3.9
cd learning_causal_discovery
```
```
python run/scripts/csv_generation_v2.py 
```
Train all the networks for MOS 6502, NetSim and Dream3 mentioned in the paper
```
cd learning_causal_discovery
bash run/train_all.sh
```
All saved checkpoints and configs should be in `learning_discovery/work_dir` and simulation/analysis results 
should be in `learning_discovery/.cache`

---
<p align=center><b>Made with 💚 at <a href="https://kordinglab.com"><img alt="KordingLab" src="https://avatars.githubusercontent.com/u/7226053?s=200&v=4" height="23px" /></a></b></p>

