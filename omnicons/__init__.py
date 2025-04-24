import os

# package location
curdir = os.path.abspath(os.path.dirname(__file__))
# dataset directory (after downloaded from zenodo)
dataset_dir = f"{curdir}/datasets"
# default directory for training experiments
experiment_dir = f"{curdir}/experiments"
