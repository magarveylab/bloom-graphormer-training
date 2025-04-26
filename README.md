# bloom-graphormer-training
Training scripts for BLOOM Graphormer models for publication

## Installation

### Training-Only Installation
1. Install the Package via Pip Symlinks.
**NOTE** The training environment is only compatible with CUDA 12.
```
    conda env create -f bloom-training.yml
    conda activate bloom-training
    pip install -e .
```
2. Set Up Weights & Biases (wandb)
    - Follow the [official quickstart guide](https://docs.wandb.ai/quickstart/) to configure Weights & Biases for experiment tracking.
3. Download the datasets from Zenodo and place the extracted contents in this [directory](https://github.com/magarveylab/bloom-graphormer-training/tree/main/omnicons/datasets).

## BLOOM-MOL Training

### Dataset Preparation
BLOOM-DOS generates biosynthetic breakdowns from SMILES entries in the [molecular dataset table](https://github.com/magarveylab/bloom-graphormer-training/blob/main/omnicons/datasets/molecule_training_data.csv). It constructs molecular graphs that include both atomic structure and higher-order biosynthetic substructures.
```python
from omnicons.datasetprep import prepare_molecular_graphs

prepare_molecular_graphs(cpus=10)
```
### Training and Model Deployment
**Masked Language Modeling (MLM)** – Trains the model to predict masked biosynthetic units and atoms. This process enables the model to learn contextual relationships between biosynthetic substructures and atomic features within the molecule, generating embeddings that reflect the underlying biosynthetic ontology. `save.py` converts DeepSpeed checkpoints into standard PyTorch checkpoint format. `export.py` converts the trained PyTorch model into TorchScript format for deployment.
```
cd training/BloomMOL
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -logger_entity new_user
python save.py
python export.py
```

## BLOOM-BGC Training

### Dataset Preparation
Generate BGC graphs from IBIS outputs, following the train–validation–test split defined in the [BGC dataset table](https://github.com/magarveylab/bloom-graphormer-training/blob/main/omnicons/datasets/bgc_training_data.csv). The training dataset includes high-quality biosynthetic clusters identified across 40,000 genomes. Download and extract `ibis_quality.zip` from the accompanying Zenodo repository.

```python
from omnicons.datasetprep import prepare_bgc_graphs

prepare_bgc_graphs()
```

### Training and Model Deployment
**Supervised Enzyme Classification** - Trains the model to predict biosynthetic subtypes, domain architectures, and EC classifications from ORF- and domain-level embeddings.
```
cd training/BloomBGC
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -logger_entity new_user
python save.py
python export.py
```

## BLOOM-RXN Training

### Dataset Preparation
Generate reaction graphs and corresponding tensors locally from SMILES found in the [reaction dataset table](https://github.com/magarveylab/bloom-graphormer-training/blob/main/omnicons/datasets/reaction_ec.csv). Each reaction is annotated with its corresponding Enzyme Commission (EC) number. Download `reaction_fewshot_ec.csv` from Zenodo — this file contains a pairwise Siamese dataset for contrastive learning, with reaction pairs sampled based on shared EC level 4 classifications.

```python
from omnicons.datasetprep import (
    prepare_reaction_ec_dataset,
    prepare_reaction_ec_fewshot_dataset
)

prepare_reaction_ec_dataset()
prepare_reaction_ec_fewshot_dataset()
```

### Training and Model Deployment
**Masked Language Modeling (MLM)** – Trains the model to predict masked atoms within molecular graphs.
```
cd training/BloomRXN/MLMTraining
CUDA_VISIBLE_DEVICES=0 python train.py -logger_entity new_user
python save.py
python export.py
```
**Supervised Learning of EC Hierarchy** – Trains the model to predict enzyme classifications across all three EC levels using parallel classification heads. This multi-task setup captures hierarchical relationships and improves functional resolution across diverse enzymatic reactions.
```
cd training/BloomRXN/ECTraining
CUDA_VISIBLE_DEVICES=0 python train.py -logger_entity new_user
python save.py
python export.py
```
**Contrastive Learning of EC Hierarchy** – Trains the model to distinguish whether pairs of reactions share the same EC level 4 classification. This is performed in conjunction with supervised parallel classification across EC levels 1 to 3, enabling the model to learn both fine-grained and hierarchical enzymatic relationships.
```
cd training/BloomRXN/ECFewShotTraining
CUDA_VISIBLE_DEVICES=0 python train.py -logger_entity new_user
python save.py
python export.py
```

## BLOOM-LNK Training

### Dataset Preparation
Download and extract `ibis_known_bgcs.zip` from Zenodo, which contains all known BGC results from IBIS. Additionally, download and extract `sm_dags.zip` and `sm_graphs.zip` to construct the corresponding BGC-metabolite graphs. Training pairs follow a k-fold split defined in this [directory](https://github.com/magarveylab/bloom-graphormer-training/tree/main/omnicons/datasets/bloom-lnk-datasets).

```python
from omnicons.datasetprep import prepare_lnk_graphs

prepare_lnk_graphs()
```

### Training and Model Deployment
**Supervised Classification of Pairs** – Trains the model to predict whether a given BGC–metabolite pair represents a true biosynthetic link. Ground truth pairs are derived from known associations, while negatives are sampled to maintain class balance during training.
```
cd training/BloomLNK
CUDA_VISIBLE_DEVICES=0 python train.py -logger_entity new_user
python save.py
python export.py
```