import math
from dataclasses import dataclass
from typing import Dict, Union

import torch
from torch_geometric.data import Data, HeteroData

from omnicons.collators.BaseCollators import BaseCollator


@dataclass
class NodeWordMaskCollator(BaseCollator):
    # in word graphs, each node is represented by a word
    # random subset of words are replaced with masked nodes
    # mask_name should correspond to a SingleNodelLabelClassificationHead
    mask_id: int = 1
    p: int = 0.15
    mask_name: str = "node_mask"
    apply_batch: bool = False
    node_types_to_consider: tuple = ()

    def prepare_individual_data(
        self, data: Union[Data, HeteroData]
    ) -> Union[Data, HeteroData]:
        data = data.clone()
        if isinstance(data, Data):
            out = self.process(data.x)
            for k, v in out.items():
                setattr(data, k, v)
        elif isinstance(data, HeteroData):
            for node_type in data.node_types:
                if node_type not in self.node_types_to_consider:
                    continue
                out = self.process(data[node_type].x)
                for k, v in out.items():
                    setattr(data[node_type], k, v)
        return data

    def process(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        l = x.shape[0]
        # generate mask
        masked_nodes = math.ceil(l * self.p)
        mask = torch.cat(
            [
                torch.ones(masked_nodes, dtype=torch.bool),
                torch.zeros(l - masked_nodes, dtype=torch.bool),
            ]
        )
        mask = mask.index_select(0, torch.randperm(mask.shape[0]))
        # introduce mask nodes
        y = x.reshape(-1).clone()
        x[mask] = torch.tensor([self.mask_id])
        y[~mask] = -100  # non-masked words are ignored
        return {"x": x, self.mask_name: y}


@dataclass
class EdgeWordMaskCollator(BaseCollator):
    # in word graphs, each node is represented by a word
    # random subset of words are replaced with masked nodes
    # mask_name should correspond to a SingleEdgeLabelClassificationHead
    mask_id: int = 1
    p: int = 0.15
    mask_name: str = "edge_mask"
    apply_batch: bool = False
    edge_types_to_consider: tuple = ()

    def prepare_individual_data(
        self, data: Union[Data, HeteroData]
    ) -> Union[Data, HeteroData]:
        data = data.clone()
        if isinstance(data, Data):
            out = self.process(data.edge_index, data.edge_attr)
            for k, v in out.items():
                setattr(data, k, v)
        elif isinstance(data, HeteroData):
            for edge_type in data.edge_types:
                if edge_type not in self.edge_types_to_consider:
                    continue
                out = self.process(
                    data[edge_type].edge_index, data[edge_type].edge_attr
                )
                for k, v in out.items():
                    setattr(data[edge_type], k, v)

        return data

    def process(
        self, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        links = edge_index.transpose(-1, -2)
        l = links.shape[0]
        # generate mask
        masked_nodes = int(l * self.p)
        mask = torch.cat(
            [
                torch.ones(masked_nodes, dtype=torch.bool),
                torch.zeros(l - masked_nodes, dtype=torch.bool),
            ]
        )
        mask = mask.index_select(0, torch.randperm(mask.shape[0]))
        indices = mask.nonzero().reshape(-1)
        # introduce mask nodes
        y = edge_attr.reshape(-1).clone()
        y = y.index_select(0, indices)
        links = links.index_select(0, indices)
        edge_attr[mask] = torch.tensor([self.mask_id])
        return {
            "edge_attr": edge_attr,
            self.mask_name: y,
            f"{self.mask_name}_links": links,
        }
