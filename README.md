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

### Dataset Preparation
BLOOM-DOS generates biosynthetic breakdowns from SMILES entries in the [molecular dataset table](https://github.com/magarveylab/bloom-graphormer-training/blob/main/omnicons/datasets/molecule_training_data.csv). It constructs molecular graphs that include both atomic structure and higher-order biosynthetic substructures.
```python
from omnicons.datasetprep import prepare_molecular_graphs

prepare_molecular_graphs(cpus=10)
```
### Training and Model Deployment
Masked Language Modeling (MLM) – Biosynthetic units and atoms are randomly masked, and the model is trained to predict their identities. This process enables the model to learn contextual relationships between biosynthetic substructures and atomic features within the molecule, generating embeddings that reflect the underlying biosynthetic ontology. `save.py` converts DeepSpeed checkpoints into standard PyTorch checkpoint format. `export.py` converts the trained PyTorch model into TorchScript format for deployment.
```
cd training/BloomMOL
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -logger_entity new_user
python save.py
python export.py
```

## BLOOM-BGC Training

### Dataset Preparation
Generate BGC graphs from IBIS outputs. The training dataset includes high-quality biosynthetic clusters identified across 40,000 genomes. Download and extract `ibis_quality.zip` from the accompanying Zenodo repository, and place the contents in this [directory](https://github.com/magarveylab/bloom-graphormer-training/tree/main/omnicons/datasets).
```python
from omnicons.datasetprep import prepare_bgc_graphs

prepare_bgc_graphs()
```

### Training and Model Deployment
The model is trained to predict biosynthetic subtypes, domain architectures, and EC classifications from ORF- and domain-level embeddings.
```
cd training/BloomBGC
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -logger_entity new_user
python save.py
python export.py
```