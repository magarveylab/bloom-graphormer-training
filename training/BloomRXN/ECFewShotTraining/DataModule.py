from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from tqdm import tqdm

from omnicons import dataset_dir
from omnicons.class_dicts import get_ec_class_dict
from omnicons.collators.StandardCollators import StandardCollator
from omnicons.data.DatasetWrapper import GraphInMemoryDataset
from omnicons.helpers import get_single_label_class_weight


class ReactionDataModule(LightningDataModule):

    def __init__(
        self,
        data_dir: str = f"{dataset_dir}/reaction_fewshot_tensors",
        in_memory: bool = True,
        batch_size: int = 100,
        num_workers: int = 0,
        persistent_workers: bool = False,
        subset: Optional[int] = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.train_fh = f"{self.data_dir}/train"
        self.val_fh = f"{self.data_dir}/val"
        self.test_fh = f"{self.data_dir}/test"
        self.in_memory = in_memory
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.subset = subset
        self.persistent_workers = persistent_workers
        self.collator = StandardCollator()

    def setup(self, stage: str = "fit"):
        if stage == "fit":
            self.train = GraphInMemoryDataset(
                root=self.train_fh, subset=self.subset
            )
            self.val = GraphInMemoryDataset(
                root=self.val_fh, subset=self.subset
            )
        if stage == "test":
            self.test = GraphInMemoryDataset(root=self.test_fh)

    def compute_class_weight(self):
        ec1_labels = []
        ec2_labels = []
        ec3_labels = []
        pair_match_labels = []
        l = self.train.len()
        if isinstance(self.subset, int):
            ec1_labels.extend(get_ec_class_dict(level=1).keys())
            ec2_labels.extend(get_ec_class_dict(level=2).keys())
            ec3_labels.extend(get_ec_class_dict(level=3).keys())
            pair_match_labels.extend([0, 1])
        for idx in tqdm(range(l), total=l):
            d = self.train.get(idx=idx)
            # ec1 labels
            ec1_labels.extend(d.graphs["a"].ec1.tolist())
            ec1_labels.extend(d.graphs["b"].ec1.tolist())
            # ec2 labels
            ec2_labels.extend(d.graphs["a"].ec2.tolist())
            ec2_labels.extend(d.graphs["b"].ec2.tolist())
            # ec3 labels
            ec3_labels.extend(d.graphs["a"].ec3.tolist())
            ec3_labels.extend(d.graphs["b"].ec3.tolist())
            # pair match labels
            pair_match_labels.extend(d.common_y.pair_match___a___b.tolist())
        weights = {
            "ec1": get_single_label_class_weight(ec1_labels),
            "ec2": get_single_label_class_weight(ec2_labels),
            "ec3": get_single_label_class_weight(ec3_labels),
            "pair_match": get_single_label_class_weight(pair_match_labels),
        }
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
