from typing import Dict

import pandas as pd
from Bloom.BloomRXN.utils import vocab_dir
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from omnicons import dataset_dir
from omnicons.collators.MaskCollators import (
    EdgeWordMaskCollator,
    NodeWordMaskCollator,
)
from omnicons.collators.MixedCollators import MixedCollator
from omnicons.collators.StandardCollators import StandardCollator
from omnicons.data.DatasetWrapper import GraphInMemoryDataset


def get_vocab(fp: str) -> Dict[str, int]:
    df = pd.read_csv(fp)
    return dict(zip(df.word, df.index))


atom_vocab = get_vocab(f"{vocab_dir}/atom_vocab.csv")
bond_vocab = get_vocab(f"{vocab_dir}/bond_vocab.csv")


class MoleculeMLMDataModule(LightningDataModule):

    def __init__(
        self,
        data_dir: str = f"{dataset_dir}/reaction_tensors",
        atom_vocab: Dict[str, int] = atom_vocab,
        bond_vocab: Dict[str, int] = bond_vocab,
        p: float = 0.15,  # masked percentage
        batch_size: int = 100,
        num_workers: int = 0,
        persistent_workers: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.train_fh = f"{self.data_dir}/train"
        self.val_fh = f"{self.data_dir}/val"
        self.test_fh = f"{self.data_dir}/test"
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.collator = MixedCollator(
            collators=(
                NodeWordMaskCollator(mask_id=atom_vocab["[MASK]"], p=p),
                EdgeWordMaskCollator(mask_id=bond_vocab["[MASK]"], p=p),
            ),
            standard_collator=StandardCollator(
                variables_to_adjust_by_precision=["extra_x", "extra_edge_attr"]
            ),
        )

    def setup(self, stage: str = "fit"):
        if stage == "fit":
            self.train = GraphInMemoryDataset(root=self.train_fh)
            self.val = GraphInMemoryDataset(root=self.val_fh)
        if stage == "test":
            self.test = GraphInMemoryDataset(root=self.test_fh)

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
