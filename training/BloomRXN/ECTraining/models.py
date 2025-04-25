from typing import Callable

import torch
from Bloom.BloomRXN.utils import get_atom_vocab, get_bond_vocab
from DataModule import ReactionDataModule
from torch.nn import ModuleDict

from omnicons.class_dicts import get_ec_class_dict
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


def get_heads(
    embedding_dim: int,
    weights: dict,
    ec1_dict: dict,
    ec2_dict: dict,
    ec3_dict: dict,
):
    from omnicons.configs.HeadConfigs import GraphClsTaskHeadConfig

    heads = {}
    heads["ec1"] = GraphClsTaskHeadConfig(
        hidden_size=embedding_dim,
        hidden_dropout_prob=0.1,
        num_labels=len(ec1_dict),
        class_weight=weights["ec1"],
        multi_label=False,
    )
    heads["ec2"] = GraphClsTaskHeadConfig(
        hidden_size=embedding_dim,
        hidden_dropout_prob=0.1,
        num_labels=len(ec2_dict),
        class_weight=weights["ec2"],
        multi_label=False,
    )
    heads["ec3"] = GraphClsTaskHeadConfig(
        hidden_size=embedding_dim,
        hidden_dropout_prob=0.1,
        num_labels=len(ec3_dict),
        class_weight=weights["ec3"],
        multi_label=False,
    )
    return heads


def get_model(
    dm: ReactionDataModule,
    mlm_checkpoint_fp: str,
    embedding_dim: int,
    optimizer: Callable = get_deepspeed_adamw,
):
    # vocab
    atom_vocab = get_atom_vocab()
    bond_vocab = get_bond_vocab()
    # data module
    class_weights = dm.compute_class_weight()
    # class dicts
    ec1_dict = get_ec_class_dict(level=1)
    ec2_dict = get_ec_class_dict(level=2)
    ec3_dict = get_ec_class_dict(level=3)
    # model setup
    node_encoder_config = get_node_encoder(
        atom_vocab, embedding_dim=embedding_dim
    )
    edge_encoder_config = get_edge_encoder(bond_vocab)
    gnn_config = get_gnn(embedding_dim=embedding_dim)
    transformer_config = get_transformer(embedding_dim=embedding_dim)
    graph_pooler_config = get_graph_pooler(embedding_dim=embedding_dim)
    edge_pooler_config = get_edge_pooler(embedding_dim=embedding_dim)
    heads = get_heads(
        embedding_dim=embedding_dim,
        weights=class_weights,
        ec1_dict=ec1_dict,
        ec2_dict=ec2_dict,
        ec3_dict=ec3_dict,
    )
    # Metrics
    train_metrics = ModuleDict(
        {
            "ec1___a": ClassificationMetrics.get(
                name="ec1___a_train",
                num_classes=len(ec1_dict),
                task="multiclass",
            ),
            "ec2___a": ClassificationMetrics.get(
                name="ec2___a_train",
                num_classes=len(ec2_dict),
                task="multiclass",
            ),
            "ec3___a": ClassificationMetrics.get(
                name="ec3___a_train",
                num_classes=len(ec3_dict),
                task="multiclass",
            ),
        }
    )
    val_metrics = ModuleDict(
        {
            "ec1___a": ClassificationMetrics.get(
                name="ec1___a_val",
                num_classes=len(ec1_dict),
                task="multiclass",
            ),
            "ec2___a": ClassificationMetrics.get(
                name="ec2___a_val",
                num_classes=len(ec2_dict),
                task="multiclass",
            ),
            "ec3___a": ClassificationMetrics.get(
                name="ec3___a_val",
                num_classes=len(ec3_dict),
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
    # load weights from mlm model
    states = torch.load(mlm_checkpoint_fp)
    model.load_state_dict(states["state_dict"], strict=False)
    return model
