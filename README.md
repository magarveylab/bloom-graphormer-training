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

## BLOOM-MOL Training

## Dataset preparation
BLOOM-DOS generates biosynthetic breakdowns from SMILES entries in the [molecular dataset table](https://github.com/magarveylab/bloom-graphormer-training/blob/main/omnicons/datasets/molecule_training_data.csv). It constructs molecular graphs that include both atomic structure and higher-order biosynthetic substructures.
```python
from omnicons.datasetprep import prepare_molecular_graphs

prepare_molecular_graphs(cpus=10)
```
## Training and Model Deployment
Masked Language Modeling (MLM) – Biosynthetic units and atoms are randomly masked, and the model is trained to predict their identities. This process enables the model to learn contextual relationships between biosynthetic substructures and atomic features within the molecule, generating embeddings that reflect the underlying biosynthetic ontology. `save.py` converts DeepSpeed checkpoints into standard PyTorch checkpoint format. `export.py` converts the trained PyTorch model into TorchScript format for deployment.
```
cd training/BloomMOL
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -logger_entity new_user
python save.py
python export.py
```