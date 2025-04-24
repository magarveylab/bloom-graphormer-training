from typing import Callable, Optional

import torch
from DataModule import atom_vocab, bond_vocab
from torch.nn import ModuleDict

from omnicons.lightning.GraphModelForMultiTask import (
    GraphModelForMultiTaskLightning,
)
from omnicons.metrics import ClassificationMetrics
from omnicons.optimizers.preconfigured import get_deepspeed_adamw


def get_node_encoder(atom_vocab: dict, embedding_dim: int):
    from omnicons.configs.EncoderConfigs import WordEncoderConfig

    # setup encoder
    node_encoder_config = WordEncoderConfig(
        num_embeddings=len(atom_vocab),
        embedding_dim=embedding_dim,
        extra_features=1,
        dropout=0.1,
        mlp_layers=1,
    )
    return node_encoder_config


def get_edge_encoder(bond_vocab: dict):
    from omnicons.configs.EncoderConfigs import WordEncoderConfig

    # setup encoder
    edge_encoder_config = WordEncoderConfig(
        num_embeddings=len(bond_vocab),
        embedding_dim=10,
        extra_features=1,
        dropout=0.1,
        mlp_layers=1,
    )
    return edge_encoder_config


def get_gnn(embedding_dim: int):
    from omnicons.configs.GNNConfigs import GATConfig

    gnn_config = GATConfig(
        num_layers=4,
        num_heads=4,
        embed_dim=embedding_dim,
        edge_dim=10,
        dropout=0.1,
    )
    return gnn_config


def get_transformer(embedding_dim: int):
    from omnicons.configs.TransformerConfigs import GraphormerConfig

    transformer_config = GraphormerConfig(
        num_layers=4,
        num_heads=4,
        embed_dim=embedding_dim,
        dropout=0.1,
        attention_dropout=0.1,
        mlp_dropout=0.1,
    )
    return transformer_config


def get_graph_pooler(embedding_dim: int):
    from omnicons.configs.GraphPoolerConfigs import NodeClsPoolerConfig

    graph_pooler_config = NodeClsPoolerConfig(hidden_channels=embedding_dim)
    return graph_pooler_config


def get_edge_pooler(embedding_dim: int):
    from omnicons.configs.EdgePoolerConfigs import EdgeMLPPoolerConfig

    edge_pooler_config = EdgeMLPPoolerConfig(
        node_embed_dim=embedding_dim,
        output_dim=embedding_dim,
        dropout=0.1,
        num_layers=1,
    )
    return edge_pooler_config


def get_heads(atom_vocab, bond_vocab, embedding_dim: int):
    from omnicons.configs.HeadConfigs import (
        EdgeClsTaskHeadConfig,
        NodeClsTaskHeadConfig,
    )

    heads = {}
    heads["node_mask"] = NodeClsTaskHeadConfig(
        hidden_size=embedding_dim,
        hidden_dropout_prob=0.1,
        num_labels=len(atom_vocab),
        multi_label=False,
    )
    heads["edge_mask"] = EdgeClsTaskHeadConfig(
        hidden_size=embedding_dim,
        hidden_dropout_prob=0.1,
        num_labels=len(bond_vocab),
    )
    return heads


def get_model(
    embedding_dim: int,
    checkpoint_path: Optional[str] = None,
    optimizer: Callable = get_deepspeed_adamw,
):
    # model setup
    node_encoder_config = get_node_encoder(
        atom_vocab, embedding_dim=embedding_dim
    )
    edge_encoder_config = get_edge_encoder(
        bond_vocab, embedding_dim=embedding_dim
    )
    gnn_config = get_gnn(embedding_dim=embedding_dim)
    transformer_config = get_transformer(embedding_dim=embedding_dim)
    graph_pooler_config = get_graph_pooler(embedding_dim=embedding_dim)
    edge_pooler_config = get_edge_pooler(embedding_dim=embedding_dim)
    heads = get_heads(atom_vocab, bond_vocab, embedding_dim=embedding_dim)
    # Metrics
    train_metrics = ModuleDict(
        {
            "node_mask___a": ClassificationMetrics.get(
                name="node_mask___a_train",
                num_classes=len(atom_vocab),
                task="multiclass",
            ),
            "edge_mask___a": ClassificationMetrics.get(
                name="edge_mask___a_train",
                num_classes=len(bond_vocab),
                task="multiclass",
            ),
        }
    )
    val_metrics = ModuleDict(
        {
            "node_mask___a": ClassificationMetrics.get(
                name="node_mask___a_val",
                num_classes=len(atom_vocab),
                task="multiclass",
            ),
            "edge_mask___a": ClassificationMetrics.get(
                name="edge_mask___a_val",
                num_classes=len(bond_vocab),
                task="multiclass",
            ),
        }
    )
    # Instantiate a PyTorch Lightning Module
    model = GraphModelForMultiTaskLightning(
        node_encoder_config=node_encoder_config,
        edge_encoder_config=edge_encoder_config,
        gnn_config=gnn_config,
        transformer_config=transformer_config,
        graph_pooler_config=graph_pooler_config,
        edge_pooler_config=edge_pooler_config,
        heads=heads,
        optimizer_fn=optimizer,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
    )
    # load checkpoint
    if checkpoint_path != None:
        states = torch.load(checkpoint_path)
        model.load_state_dict(states["state_dict"], strict=True)
    return model
