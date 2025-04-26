import pickle
import random
from typing import List, Optional

from torch_geometric.data import Dataset
from tqdm import tqdm


class TrainingDataset(Dataset):

    def __init__(
        self,
        datapoints: List[int],
        root: str,
        subset: Optional[int] = None,
        node_types_to_consider: Optional[List[str]] = None,
        edge_types_to_consider: Optional[List[str]] = None,
        node_vocab: dict = {},
        edge_vocab: dict = {},
        class_dict: dict = {},
    ):
        self.root = root
        self.node_types_to_consider = node_types_to_consider
        self.edge_types_to_consider = edge_types_to_consider
        self.node_vocab = node_vocab
        self.edge_vocab = edge_vocab
        self.class_dict = class_dict
        self.prepare_data(datapoints=datapoints, subset=subset)
        super().__init__(
            root=root, transform=None, pre_transform=None, pre_filter=None
        )

    @property
    def processed_file_names(self):
        return []

    def prepare_data(
        self,
        datapoints: List[int],
        subset: Optional[int] = None,
    ):
        if subset:
            datapoints = random.sample(datapoints, subset)
        self.tensor_cache = {}
        for sample_id in tqdm(datapoints):
            G = pickle.load(open(f"{self.root}/{sample_id}.pkl", "rb"))
            self.tensor_cache[sample_id] = G.get_tensor_data(
                node_vocab=self.node_vocab,
                edge_vocab=self.edge_vocab,
                node_types_to_consider=self.node_types_to_consider,
                edge_types_to_consider=self.edge_types_to_consider,
                node_label_class_dict=self.class_dict,
                apply_edge_attr=False,
                apply_multigraph_wrapper=True,
            )
        self.datapoints = sorted(self.tensor_cache)

    def len(self):
        return len(self.datapoints)

    def get(self, input_idx):
        return self.tensor_cache[self.datapoints[input_idx]]
