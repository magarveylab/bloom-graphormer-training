from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from tqdm import tqdm

from omnicons import dataset_dir
from omnicons.collators.StandardCollators import StandardCollator
from omnicons.data.DatasetWrapper import GraphInMemoryDataset
from omnicons.helpers import get_single_label_class_weight


class ReactionDataModule(LightningDataModule):

    def __init__(
        self,
        data_dir: str = f"{dataset_dir}/reaction_ec_tensors",
        in_memory: bool = True,
        batch_size: int = 100,
        num_workers: int = 0,
        persistent_workers: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.train_fh = f"{self.data_dir}/train"
        self.val_fh = f"{self.data_dir}/val"
        self.test_fh = f"{self.data_dir}/test"
        self.in_memory = in_memory
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.collator = StandardCollator()

    def setup(self, stage: str = "fit"):
        if stage == "fit":
            self.train = GraphInMemoryDataset(root=self.train_fh)
            self.val = GraphInMemoryDataset(root=self.val_fh)
        if stage == "test":
            self.test = GraphInMemoryDataset(root=self.test_fh)

    def compute_class_weight(self):
        ec1_labels = []
        ec2_labels = []
        ec3_labels = []
        l = self.train.len()
        for idx in tqdm(range(l), total=l):
            d = self.train.get(idx=idx)
            ec1_labels.extend(d.graphs["a"].ec1.tolist())
            ec2_labels.extend(d.graphs["a"].ec2.tolist())
            ec3_labels.extend(d.graphs["a"].ec3.tolist())
        weights = {
            "ec1": get_single_label_class_weight(ec1_labels),
            "ec2": get_single_label_class_weight(ec2_labels),
            "ec3": get_single_label_class_weight(ec3_labels),
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
