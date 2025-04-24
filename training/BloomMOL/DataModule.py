from typing import List, Optional

import pandas as pd
from Bloom.BloomEmbedder.graphs.MoleculeGraph import (
    get_edge_vocab,
    get_node_vocab,
)
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from tqdm import tqdm
from TrainingDataset import TrainingDataset

from omnicons import dataset_dir
from omnicons.collators.MaskCollators import NodeWordMaskCollator
from omnicons.collators.MixedCollators import MixedCollator


class MolGraphDataModule(LightningDataModule):

    def __init__(
        self,
        dataset_fp: str = f"{dataset_dir}/molecule_training_data.csv",
        graph_dir: str = f"{dataset_dir}/molecular_graphs",
        batch_size: int = 30,
        num_workers: int = 0,
        persistent_workers: bool = False,
        subset: Optional[int] = None,
        node_types_to_consider: Optional[List[str]] = None,
        edge_types_to_consider: Optional[List[str]] = None,
    ):
        super().__init__()
        self.dataset_fp = dataset_fp
        self.graph_dir = graph_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.subset = subset
        # vocab
        self.node_vocab = get_node_vocab()
        self.edge_vocab = get_edge_vocab()
        # graph parameters to create tensors
        self.node_types_to_consider = node_types_to_consider
        self.edge_types_to_consider = edge_types_to_consider
        # collators
        self.collator = MixedCollator(
            collators=(
                NodeWordMaskCollator(
                    mask_id=self.node_vocab["MetaboliteMolecularRegion"][
                        "[MASK]"
                    ],
                    p=0.30,
                    mask_name="unit_mask",
                    node_types_to_consider=("MetaboliteMolecularRegion"),
                ),
                NodeWordMaskCollator(
                    mask_id=self.node_vocab["Atom"]["[MASK]"],
                    p=0.30,
                    mask_name="atom_mask",
                    node_types_to_consider=("Atom"),
                ),
            )
        )

    def setup(self, stage: str = "fit"):
        # load datapoints
        datapoints = {"train": [], "val": [], "test": []}
        for r in tqdm(pd.read_csv(self.dataset_fp).to_dict("records")):
            datapoints[r["split"]].append(r["metabolite_id"])
        # setup dynamic datasets
        if stage == "fit":
            self.train = TrainingDataset(
                datapoints=datapoints["train"],
                root=self.graph_dir,
                subset=self.subset,
                node_types_to_consider=self.node_types_to_consider,
                edge_types_to_consider=self.edge_types_to_consider,
                node_vocab=self.node_vocab,
                edge_vocab=self.edge_vocab,
            )
            self.val = TrainingDataset(
                datapoints=datapoints["val"],
                root=self.graph_dir,
                subset=self.subset,
                node_types_to_consider=self.node_types_to_consider,
                edge_types_to_consider=self.edge_types_to_consider,
                node_vocab=self.node_vocab,
                edge_vocab=self.edge_vocab,
            )
        if stage == "test":
            self.test = TrainingDataset(
                datapoints=datapoints["test"],
                root=self.graph_dir,
                subset=self.subset,
                node_types_to_consider=self.node_types_to_consider,
                edge_types_to_consider=self.edge_types_to_consider,
                node_vocab=self.node_vocab,
                edge_vocab=self.edge_vocab,
            )

    def train_dataloader(self):
        train_dl = DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collator,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )
        return train_dl

    def test_dataloader(self):
        test_dl = DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collator,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )
        return test_dl

    def val_dataloader(self):
        val_dl = DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collator,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )
        return val_dl
