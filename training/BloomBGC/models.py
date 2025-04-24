from typing import Callable

from torch.nn import ModuleDict

from omnicons.lightning.HeteroGraphModelForMultiTask import (
    HeteroGraphModelForMultiTaskLightning,
)
from omnicons.metrics import ClassificationMetrics
from omnicons.optimizers.preconfigured import get_deepspeed_adamw


def get_node_encoders(embedding_dim: int = 128):
    from omnicons.configs.EncoderConfigs import (
        MLPEncoderConfig,
        WordEncoderConfig,
    )

    node_encoders = {}
    # setup encoder for nodes with labels
    for node_type in ["BGC", "Module"]:
        node_encoders[node_type] = WordEncoderConfig(
            num_embeddings=4,
            embedding_dim=embedding_dim,
            extra_features=0,
            dropout=0.1,
            mlp_layers=1,
        )
    # setup encoder for nodes with embedding
    for node_type in ["Orf", "Domain"]:
        node_encoders[node_type] = MLPEncoderConfig(
            input_dim=1024,
            output_dim=embedding_dim,
            dropout=0.1,
            num_layers=1,
        )
    return node_encoders


def get_edge_type_encoder(edge_types: list, embedding_dim: int = 10):
    from omnicons.configs.EncoderConfigs import WordEncoderConfig

    return WordEncoderConfig(
        num_embeddings=len(edge_types),
        embedding_dim=embedding_dim,
        extra_features=0,
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


def get_heads(class_dict: dict, weights: dict, embedding_dim: int = 128):
    from omnicons.configs.HeadConfigs import NodeClsTaskHeadConfig

    heads = {}
    for label_name in class_dict:
        if label_name == "module_tag":
            node_type = "Module"
        else:
            node_type = "Orf"
        heads[label_name] = NodeClsTaskHeadConfig(
            hidden_size=embedding_dim,
            hidden_dropout_prob=0.1,
            num_labels=len(class_dict[label_name]),
            class_weight=weights[label_name].tolist(),
            node_type=node_type,
            analyze_inputs=["a"],
        )
    return heads


def get_model(
    class_dict: dict,
    weights: dict,
    node_embedding_dim: int = 128,
    edge_embedding_dim: int = 10,
    num_gnn_heads: int = 8,
    num_transformer_heads: int = 8,
    optimizer: Callable = get_deepspeed_adamw,
):
    edge_types = [
        ("Orf", "orf_to_module", "Module"),
        ("Module", "module_adj", "Module"),
        ("Module", "module_to_domain", "Domain"),
    ]
    # model setup
    node_encoders = get_node_encoders(
        embedding_dim=node_embedding_dim,
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
    heads = get_heads(
        class_dict=class_dict,
        weights=weights,
        embedding_dim=node_embedding_dim,
    )
    # Metrics
    train_metrics = ModuleDict()
    val_metrics = ModuleDict()
    for label_name in class_dict:
        key = f"{label_name}___a"
        train_metrics[key] = ClassificationMetrics.get(
            name=f"{key}_train",
            num_classes=len(class_dict[label_name]),
            task="multiclass",
        )
        val_metrics[key] = ClassificationMetrics.get(
            name=f"{key}_val",
            num_classes=len(class_dict[label_name]),
            task="multiclass",
        )
    # Instantiate a PyTorch Lightning Module
    model = HeteroGraphModelForMultiTaskLightning(
        node_encoders=node_encoders,
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
