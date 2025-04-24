import pickle
import random
from typing import Dict, List, Optional, TypedDict, Union

import torch
from Bloom.CommonUtils.HeteroGraph import EdgeVocab, NodeVocab
from torch_geometric.data import Dataset
from tqdm import tqdm


class ClassificationDataPoint(TypedDict):
    # also contains labels
    graph_id: str
    classification_bin: str


class DynamicDataset(Dataset):

    def __init__(
        self,
        root: str,
        node_types_to_consider: Optional[List[str]] = None,
        edge_types_to_consider: Optional[List[str]] = None,
        node_vocab: NodeVocab = {},
        edge_vocab: EdgeVocab = {},
        filename_lookup: Dict[str, str] = {},
        in_memory: bool = True,
        dynamic_tensor_render: bool = True,
    ):
        self.node_types_to_consider = node_types_to_consider
        self.edge_types_to_consider = edge_types_to_consider
        self.node_vocab = node_vocab
        self.edge_vocab = edge_vocab
        self.filename_lookup = (
            filename_lookup  # links ids in anchors to graph paths or tensors
        )
        self.in_memory = in_memory
        self.dynamic_tensor_render = dynamic_tensor_render
        if self.in_memory:
            if self.dynamic_tensor_render:
                self.graph_cache = self.get_graph_cache()
                self.tensor_cache = None
            else:
                self.graph_cache = None
                self.tensor_cache = self.get_tensor_cache()
        else:
            self.graph_cache = None
            self.tensor_cache = None
        super().__init__(
            root=root, transform=None, pre_transform=None, pre_filter=None
        )

    @property
    def processed_file_names(self):
        return list(self.filename_lookup.values())

    def get_graph_cache(self):
        cache = {}
        for sample_id, fp in tqdm(self.filename_lookup.items()):
            cache[sample_id] = pickle.load(open(fp, "rb"))
        return cache

    def get_tensor_cache(self):
        cache = {}
        for sample_id, fp in tqdm(self.filename_lookup.items()):
            G = pickle.load(open(fp, "rb"))
            tensor = G.get_tensor_data(
                node_vocab=self.node_vocab,
                edge_vocab=self.edge_vocab,
                node_types_to_consider=self.node_types_to_consider,
                edge_types_to_consider=self.edge_types_to_consider,
                apply_edge_attr=False,
            )
            cache[sample_id] = tensor
        return cache

    def get_tensor(self, graph_id: Union[int, str]):
        if self.dynamic_tensor_render:
            # load graph
            if self.in_memory:
                G = self.graph_cache[graph_id]
            else:  # filepath corresponds to HeterGraph
                G = pickle.load(open(self.filename_lookup[graph_id], "rb"))
            # convert to tensor
            return G.get_tensor_data(
                node_vocab=self.node_vocab, edge_vocab=self.edge_vocab
            )
        else:
            # load tensor
            if self.in_memory:
                return self.tensor_cache[graph_id]
            else:
                return torch.load(self.filename_lookup[graph_id])


class TrainingDataset(DynamicDataset):

    def __init__(
        self,
        datapoints: List[ClassificationDataPoint],
        root: str,
        subset: Optional[int] = None,
        node_types_to_consider: Optional[List[str]] = None,
        edge_types_to_consider: Optional[List[str]] = None,
        node_vocab: NodeVocab = {},
        edge_vocab: EdgeVocab = {},
        in_memory: bool = True,
        dynamic_tensor_render: bool = True,
        ds_multiplier: int = 1,
    ):
        self.root = root
        self.ds_multiplier = ds_multiplier
        self.prepare_data(datapoints=datapoints, subset=subset)
        self.graph_len = len(self.graphs_sorted)
        super().__init__(
            root=root,
            node_types_to_consider=node_types_to_consider,
            edge_types_to_consider=edge_types_to_consider,
            node_vocab=node_vocab,
            edge_vocab=edge_vocab,
            filename_lookup=self.filename_lookup,
            in_memory=in_memory,
            dynamic_tensor_render=dynamic_tensor_render,
        )

    def prepare_data(
        self,
        datapoints: List[ClassificationDataPoint],
        subset: Optional[int],
    ):
        graphs_sorted = {d["classification_bin"]: [] for d in datapoints}
        self.label_lookup = {}
        for d in tqdm(datapoints):
            classification_bin = d["classification_bin"]
            metabolite_id = d["metabolite_id"]
            cluster_id = d["cluster_id"]
            graph_id = f"{metabolite_id}-{cluster_id}"
            graphs_sorted[classification_bin].append(graph_id)
            self.label_lookup[graph_id] = d
        # take subset
        if isinstance(subset, int):
            for b in graphs_sorted:
                graphs_sorted[b] = graphs_sorted[b][:subset]
        # prepare filename lookup
        self.filename_lookup = {}
        for b in graphs_sorted:
            for graph_id in graphs_sorted[b]:
                self.filename_lookup[graph_id] = f"{self.root}/{graph_id}.pkl"
        self.graphs_sorted = [
            {"bin": dp_bin, "graphs": dp}
            for dp_bin, dp in graphs_sorted.items()
        ]

    def len(self):
        return self.graph_len * self.ds_multiplier

    def get(self, input_idx):
        idx = input_idx % self.graph_len
        graph_id = random.choice(self.graphs_sorted[idx]["graphs"])
        tensor = self.get_tensor(graph_id)
        for label in [
            "tanimoto_bin_1",
            "tanimoto_bin_2",
            "tanimoto_bin_3",
            "tanimoto_bin_4",
            "tanimoto_bin_5",
            "binary_9",
            "binary_8",
            "binary_7",
            "binary_6",
        ]:
            tensor.graphs["a"].__setattr__(
                label, torch.LongTensor([[self.label_lookup[graph_id][label]]])
            )
        return tensor
