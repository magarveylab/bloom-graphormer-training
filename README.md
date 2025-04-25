# bloom-graphormer-training
Training scripts for BLOOM Graphormer models for publication

## Installation

### Training-Only Installation
1. Install the Package via Pip Symlinks.
```
    conda env create -f bloom-training.yml
    conda activate bloom-training
    pip install -e .
```
2. Set Up Weights & Biases (wandb)
    - Follow the [official quickstart guide](https://docs.wandb.ai/quickstart/) to configure Weights & Biases for experiment tracking.