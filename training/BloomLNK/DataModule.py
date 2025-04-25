from typing import List, Optional

import numpy as np
import pandas as pd
from Bloom.BloomLNK.graph import get_vocab
from pytorch_lightning import LightningDataModule
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from tqdm import tqdm
from TrainingDataset import TrainingDataset

from omnicons import dataset_dir
from omnicons.collators.MixedCollators import MixedCollator
from omnicons.collators.StandardCollators import StandardCollator


class LNKDataModule(LightningDataModule):

    def __init__(
        self,
        dataset_fp: str = f"{dataset_dir}/bloom-lnk-datasets/final.csv",
        graph_dir: str = f"{dataset_dir}/bloom-lnk-graphs",
        batch_size: int = 30,
        num_workers: int = 0,
        persistent_workers: bool = False,
        subset: Optional[int] = None,
        node_types_to_consider: Optional[List[str]] = None,
        edge_types_to_consider: Optional[List[str]] = None,
        ds_multiplier: int = 1,
    ):
        super().__init__()
        self.dataset_fp = dataset_fp
        self.graph_dir = graph_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.subset = subset
        self.ds_multiplier = ds_multiplier
        # vocab
        self.node_vocab = get_vocab()
        self.node_types_to_consider = node_types_to_consider
        self.edge_types_to_consider = edge_types_to_consider
        self.collator = MixedCollator(
            standard_collator=StandardCollator(
                variables_to_adjust_by_precision=[
                    ("Domain", "x"),
                    ("Orf", "x"),
                    ("Reaction", "x"),
                ]
            ),
        )

    def setup(self, stage: str = "fit"):
        # load datapoints
        datapoints = {"train": [], "val": [], "test": []}
        for r in tqdm(
            pd.read_csv(self.dataset_fp).to_dict("records"),
            desc="Loading datapoints",
        ):
            split = r["split"]
            datapoints[split].append(r)
        # setup dynamic datasets
        if stage == "fit":
            self.train = TrainingDataset(
                datapoints=datapoints["train"],
                root=self.graph_dir,
                subset=self.subset,
                node_vocab=self.node_vocab,
                node_types_to_consider=self.node_types_to_consider,
                edge_types_to_consider=self.edge_types_to_consider,
                in_memory=True,
                dynamic_tensor_render=False,
                ds_multiplier=self.ds_multiplier,
            )
            self.val = TrainingDataset(
                datapoints=datapoints["val"],
                root=self.graph_dir,
                subset=self.subset,
                node_vocab=self.node_vocab,
                node_types_to_consider=self.node_types_to_consider,
                edge_types_to_consider=self.edge_types_to_consider,
                in_memory=True,
                dynamic_tensor_render=False,
                ds_multiplier=self.ds_multiplier,
            )
        if stage == "test":
            self.test = TrainingDataset(
                datapoints=datapoints["test"],
                root=self.graph_dir,
                subset=self.subset,
                node_vocab=self.node_vocab,
                edge_vocab=self.edge_vocab,
                node_types_to_consider=self.node_types_to_consider,
                edge_types_to_consider=self.edge_types_to_consider,
                in_memory=True,
                dynamic_tensor_render=False,
                ds_multiplier=self.ds_multuplier,
            )

    def calculate_weights(self):
        cls_bins = {
            "tanimoto_bin_1": {},
            "tanimoto_bin_2": {},
            "tanimoto_bin_3": {},
            "tanimoto_bin_4": {},
            "tanimoto_bin_5": {},
            "binary_9": {},
            "binary_8": {},
            "binary_7": {},
            "binary_6": {},
        }

        for r in pd.read_csv(self.dataset_fp).to_dict("records"):
            split = r["split"]
            if split == "train":
                for label_name in cls_bins:
                    value = r[label_name]
                    if value not in cls_bins[label_name]:
                        cls_bins[label_name][value] = set()
                    cls_bins[label_name][value].add(r["classification_bin"])
        weights = {}
        for label_name in cls_bins:
            y = []
            for v, cls_set in cls_bins[label_name].items():
                y.extend([v] * len(cls_set))
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
