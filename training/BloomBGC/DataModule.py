import json
import pickle
from typing import List, Optional

import numpy as np
import pandas as pd
from Bloom.BloomEmbedder.graphs.BGCGraph import get_node_vocab
from pytorch_lightning import LightningDataModule
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from tqdm import tqdm
from TrainingDataset import TrainingDataset

from omnicons import dataset_dir
from omnicons.collators.StandardCollators import StandardCollator


class BGCGraphDataModule(LightningDataModule):

    def __init__(
        self,
        dataset_fp: str = f"{dataset_dir}/bgc_training_data.csv",
        graph_dir: str = f"{dataset_dir}/bgc_graphs",
        class_dict_fp: str = f"{dataset_dir}/bgc_class_dict.json",
        batch_size: int = 60,
        num_workers: int = 0,
        persistent_workers: bool = False,
        subset: Optional[int] = None,
        node_types_to_consider: Optional[List[str]] = None,
        edge_types_to_consider: Optional[List[str]] = None,
    ):
        super().__init__()
        self.dataset_fp = dataset_fp
        self.class_dict = json.load(open(class_dict_fp))
        self.graph_dir = graph_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.subset = subset
        # vocab
        self.node_vocab = get_node_vocab()
        # graph parameters to create tensors
        self.node_types_to_consider = node_types_to_consider
        self.edge_types_to_consider = edge_types_to_consider
        # collators
        self.collator = StandardCollator(
            variables_to_adjust_by_precision=[
                ("Orf", "x"),
                ("Domain", "x"),
            ]
        )

    def setup(self, stage: str = "fit"):
        # load datapoints
        datapoints = {"train": [], "val": [], "test": []}
        for r in tqdm(pd.read_csv(self.dataset_fp).to_dict("records")):
            datapoints[r["split"]].append(r["cluster_id"])
        # setup dynamic datasets
        if stage == "fit":
            self.train = TrainingDataset(
                datapoints=datapoints["train"],
                root=self.graph_dir,
                subset=self.subset,
                node_types_to_consider=self.node_types_to_consider,
                edge_types_to_consider=self.edge_types_to_consider,
                node_vocab=self.node_vocab,
                class_dict=self.class_dict,
            )
            self.val = TrainingDataset(
                datapoints=datapoints["val"],
                root=self.graph_dir,
                subset=self.subset,
                node_types_to_consider=self.node_types_to_consider,
                edge_types_to_consider=self.edge_types_to_consider,
                node_vocab=self.node_vocab,
                class_dict=self.class_dict,
            )
        if stage == "test":
            self.test = TrainingDataset(
                datapoints=datapoints["test"],
                root=self.graph_dir,
                subset=self.subset,
                node_types_to_consider=self.node_types_to_consider,
                edge_types_to_consider=self.edge_types_to_consider,
                node_vocab=self.node_vocab,
            )

    def calculate_class_weights(self):
        # load datapoints
        freq = {"ec4": [], "ec3": [], "ec2": [], "ec1": [], "module_tag": []}
        for r in tqdm(pd.read_csv(self.dataset_fp).to_dict("records")):
            G = pickle.load(
                open(f"{self.graph_dir}/{r['cluster_id']}.pkl", "rb")
            )
            for n, labels in G.node_label_lookup.items():
                for k, v in labels.items():
                    if v in self.class_dict[k]:
                        freq[k].append(self.class_dict[k][v])
        # calculate class weights
        weights = {}
        for label_name, y in freq.items():
            weights[label_name] = compute_class_weight(
                class_weight="balanced", classes=np.unique(y), y=y
            )
        return weights

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
