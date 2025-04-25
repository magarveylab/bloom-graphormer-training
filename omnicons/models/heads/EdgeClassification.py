from typing import List, Optional

import numpy as np
import torch
from torch import nn

from omnicons.models import DataStructs, helpers


class SingleLabelEdgeClassificationHead(nn.Module):

    def __init__(
        self,
        hidden_size: int = 768,
        hidden_dropout_prob: float = 0.1,
        num_labels: int = 2,
        class_weight: Optional[List[float]] = None,
        analyze_inputs: List[str] = ["a"],
        edge_type: tuple = None,
        **kwargs,
    ):
        super().__init__()
        self.training_task = "edge_classification"
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)
        if class_weight == None:
            weight = None
        else:
            weight = np.array(class_weight)
            weight = torch.tensor(weight).to(torch.float32)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100, weight=weight)
        self.analyze_inputs = analyze_inputs
        self.edge_type = edge_type

    def forward(self, edge_logits: torch.Tensor) -> torch.Tensor:
        edge_logits = self.dropout(edge_logits)
        edge_logits = self.classifier(edge_logits)
        return edge_logits

    def classification(
        self, edge_logits: torch.Tensor, labels: torch.Tensor
    ) -> DataStructs.ClassificationOutput:
        # x is accessed by f'x' in data.graphs[inp]
        # labels is accessed by f'{head_name}' in data.graphs[inp]
        # links is accessed by f'{head_name}___links' in data.graphs[inp]
        # prepare logits
        consider_index, labels = helpers.recast_labels_for_single_label(labels)
        edge_logits = helpers.recast_input_for_single_label(
            edge_logits, consider_index
        )
        edge_logits = self.forward(edge_logits)
        # calculate loss
        loss = self.loss_fct(edge_logits, labels)
        return {"logits": edge_logits, "labels": labels, "loss": loss}
