from typing import Callable

from Bloom.BloomEmbedder.graphs.MoleculeGraph import (
    get_edge_vocab,
    get_node_vocab,
)
from torch.nn import ModuleDict

from omnicons.lightning.HeteroGraphModelForMultiTask import (
    HeteroGraphModelForMultiTaskLightning,
)
from omnicons.metrics import ClassificationMetrics
from omnicons.optimizers.preconfigured import get_deepspeed_adamw


def get_node_encoders(vocab: dict, embedding_dim: int = 128):
    from omnicons.configs.EncoderConfigs import WordEncoderConfig

    node_encoders = {}
    for node_type in vocab:
        node_encoders[node_type] = WordEncoderConfig(
            num_embeddings=len(vocab[node_type]),
            embedding_dim=embedding_dim,
            dropout=0.1,
            mlp_layers=1,
        )
    return node_encoders


def get_edge_encoders(vocab: dict, embedding_dim: int = 10):
    from omnicons.configs.EncoderConfigs import WordEncoderConfig

    edge_encoders = {}
    for edge_type in vocab:
        edge_encoders[edge_type] = WordEncoderConfig(
            num_embeddings=len(vocab[edge_type]),
            embedding_dim=embedding_dim,
            dropout=0.1,
            mlp_layers=1,
        )
    return edge_encoders


def get_edge_type_encoder(edge_types: list, embedding_dim: int = 10):
    from omnicons.configs.EncoderConfigs import WordEncoderConfig

    return WordEncoderConfig(
        num_embeddings=len(edge_types),
        embedding_dim=embedding_dim,
        extra_features=embedding_dim,
        dropout=0.1,
        mlp_layers=1,
    )


def get_gnn(
    node_embedding_dim: int = 128,
    edge_embedding_dim: int = 128,
    num_heads: int = 8,
):
    from omnicons.configs.GNNConfigs import GATConfig

    gnn_config = GATConfig(
        num_layers=num_heads,
        num_heads=num_heads,
        embed_dim=node_embedding_dim,
        edge_dim=edge_embedding_dim,
        dropout=0.1,
    )
    return gnn_config


def get_transformer(embedding_dim: int = 128, num_heads: int = 8):
    from omnicons.configs.TransformerConfigs import GraphormerConfig

    transformer_config = GraphormerConfig(
        num_layers=num_heads,
        num_heads=num_heads,
        embed_dim=embedding_dim,
        dropout=0.1,
        attention_dropout=0.1,
        mlp_dropout=0.1,
    )
    return transformer_config


def get_heads(vocab: dict, embedding_dim: int = 128):
    from omnicons.configs.HeadConfigs import NodeClsTaskHeadConfig

    heads = {}
    heads["unit_mask"] = NodeClsTaskHeadConfig(
        hidden_size=embedding_dim,
        hidden_dropout_prob=0.1,
        num_labels=len(vocab["MetaboliteMolecularRegion"]),
        multi_label=False,
        node_type="MetaboliteMolecularRegion",
        analyze_inputs=["a"],
    )
    return heads


def get_model(
    node_embedding_dim: int = 128,
    edge_embedding_dim: int = 10,
    num_gnn_heads: int = 8,
    num_transformer_heads: int = 8,
    optimizer: Callable = get_deepspeed_adamw,
):
    edge_types = [
        ("Atom", "bond", "Atom"),
        ("Atom", "atom_to_unit", "MetaboliteMolecularRegion"),
        (
            "MetaboliteMolecularRegion",
            "mol_region_adj",
            "MetaboliteMolecularRegion",
        ),
    ]
    # get vocab
    node_vocab = get_node_vocab()
    edge_vocab = get_edge_vocab()
    # model setup
    node_encoders = get_node_encoders(
        vocab=node_vocab,
        embedding_dim=node_embedding_dim,
    )
    edge_encoders = get_edge_encoders(
        vocab=edge_vocab, embedding_dim=edge_embedding_dim
    )
    edge_type_encoder_config = get_edge_type_encoder(
        edge_types=edge_types,
        embedding_dim=edge_embedding_dim,
    )
    gnn_config = get_gnn(
        node_embedding_dim=node_embedding_dim,
        edge_embedding_dim=edge_embedding_dim,
        num_heads=num_gnn_heads,
    )
    transformer_config = get_transformer(
        embedding_dim=node_embedding_dim, num_heads=num_transformer_heads
    )
    heads = get_heads(vocab=node_vocab, embedding_dim=node_embedding_dim)
    # Metrics
    train_metrics = ModuleDict()
    val_metrics = ModuleDict()
    num_classes = {
        "unit_mask": len(node_vocab["MetaboliteMolecularRegion"]),
    }
    for label_name in ["unit_mask"]:
        key = f"{label_name}___a"
        train_metrics[key] = ClassificationMetrics.get(
            name=f"{key}_train",
            num_classes=num_classes[label_name],
            task="multiclass",
        )
        val_metrics[key] = ClassificationMetrics.get(
            name=f"{key}_val",
            num_classes=num_classes[label_name],
            task="multiclass",
        )
    # Instantiate a PyTorch Lightning Module
    model = HeteroGraphModelForMultiTaskLightning(
        node_encoders=node_encoders,
        edge_encoders=edge_encoders,
        edge_type_encoder_config=edge_type_encoder_config,
        gnn_config=gnn_config,
        transformer_config=transformer_config,
        heads=heads,
        optimizer_fn=optimizer,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        inputs=["a"],
        edge_types=edge_types,
    )
    return model
