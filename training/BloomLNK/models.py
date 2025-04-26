from typing import Callable

from Bloom.BloomLNK.graph import get_vocab
from torch.nn import ModuleDict

from omnicons.lightning.HeteroGraphModelForMultiTask import (
    HeteroGraphModelForMultiTaskLightning,
)
from omnicons.metrics import ClassificationMetrics
from omnicons.optimizers.preconfigured import get_deepspeed_adamw

node_types_with_labels = [
    "Pair",
    "Module",
    "ModuleTag",
    "Substrate",
    "SubstrateFamilyTag",
    "PKSReactionTag",
    "EC4",
    "EC3",
    "Gene",
    "ProteinFamilyTag",
    "MetaboliteMolecularRegion",
    "UnitRule",
    "SugarReactionTag",
    "TailoringReactionTag",
]
node_types_with_embedding = {"Orf": 1024, "Domain": 1024, "Reaction": 256}
edge_types = [
    # genomic edges
    ("Orf", "orf_to_module", "Module"),
    ("Module", "module_adj", "Module"),
    ("Module", "module_to_domain", "Domain"),
    # module edges
    ("Module", "module_to_tag", "ModuleTag"),
    ("Module", "module_to_substrate", "Substrate"),
    ("Module", "module_to_substrate_family_tag", "SubstrateFamilyTag"),
    ("Module", "module_to_pks_reaction_tag", "PKSReactionTag"),
    # orf edges
    ("Orf", "orf_to_ec4", "EC4"),
    ("EC4", "ec4_to_ec3", "EC3"),
    ("Orf", "orf_to_gene", "Gene"),
    ("Gene", "gene_to_protein_family_tag", "ProteinFamilyTag"),
    # metabolite edges
    (
        "MetaboliteMolecularRegion",
        "mol_region_adj",
        "MetaboliteMolecularRegion",
    ),
    # unit to module connections (module_tag, pks reaction, substrate, substrate family)
    ("MetaboliteMolecularRegion", "unit_to_module_tag", "ModuleTag"),
    (
        "MetaboliteMolecularRegion",
        "unit_to_pks_reaction_tag",
        "PKSReactionTag",
    ),
    ("MetaboliteMolecularRegion", "unit_to_substrate", "Substrate"),
    (
        "MetaboliteMolecularRegion",
        "unit_to_substrate_family_tag",
        "SubstrateFamilyTag",
    ),
    # unit to rules
    ("MetaboliteMolecularRegion", "unit_to_rule", "UnitRule"),
    # unit to sugar connections
    ("UnitRule", "rule_to_sugar_reaction_tag", "SugarReactionTag"),
    ("SugarReactionTag", "sugar_reaction_tag_to_ec4", "EC4"),
    # unit to tailoring connections
    (
        "UnitRule",
        "rule_to_tailoring_reaction_tag",
        "TailoringReactionTag",
    ),
    ("TailoringReactionTag", "tailoring_reaction_tag_to_ec3", "EC3"),
    # unit to reaction connections
    ("UnitRule", "rule_to_reaction", "Reaction"),
    ("Reaction", "reaction_to_ec4", "EC4"),
    ("Reaction", "reaction_to_protein_family_tag", "ProteinFamilyTag"),
]

# label parameters
tanimoto_bins = {
    "tanimoto_bin_1": {
        "bins": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1],
        "labels": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    },
    "tanimoto_bin_2": {
        "bins": [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
        "labels": [0, 1, 2, 3, 4],
    },
    "tanimoto_bin_3": {
        "bins": [0, 0.3, 0.6, 0.9, 1.2],
        "labels": [0, 1, 2, 3],
    },
    "tanimoto_bin_4": {"bins": [0, 0.4, 0.8, 1.2], "labels": [0, 1, 2]},
    "tanimoto_bin_5": {"bins": [0, 0.5, 1.1], "labels": [0, 1]},
    "binary_9": {"bins": [0, 0.9, 1.1], "labels": [0, 1]},
    "binary_8": {"bins": [0, 0.8, 1.1], "labels": [0, 1]},
    "binary_7": {"bins": [0, 0.7, 1.1], "labels": [0, 1]},
    "binary_6": {"bins": [0, 0.6, 1.1], "labels": [0, 1]},
}


def get_node_encoders(vocab: dict, embedding_dim: int):
    from omnicons.configs.EncoderConfigs import (
        MLPEncoderConfig,
        WordEncoderConfig,
    )

    node_encoders = {}
    # setup encoder for nodes with labels
    for node_type in node_types_with_labels:
        node_encoders[node_type] = WordEncoderConfig(
            num_embeddings=len(vocab[node_type]),
            embedding_dim=embedding_dim,
            extra_features=0,
            dropout=0.1,
            mlp_layers=1,
        )
    # setup encoder for nodes with embedding
    for node_type in node_types_with_embedding:
        node_encoders[node_type] = MLPEncoderConfig(
            input_dim=node_types_with_embedding[node_type],
            output_dim=embedding_dim,
            dropout=0.1,
            num_layers=1,
        )
    return node_encoders


def get_edge_type_encoder(embedding_dim: int):
    from omnicons.configs.EncoderConfigs import WordEncoderConfig

    return WordEncoderConfig(
        num_embeddings=len(edge_types),
        embedding_dim=embedding_dim,
        extra_features=0,
        dropout=0.1,
        mlp_layers=1,
    )


def get_gnn(
    node_embedding_dim: int,
    edge_embedding_dim: int,
    num_heads: int,
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


def get_transformer(embedding_dim: int, num_heads: int):
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


def get_graph_pooler(embedding_dim: int):
    from omnicons.configs.GraphPoolerConfigs import HeteroNodeClsPoolerConfig

    graph_pooler_config = HeteroNodeClsPoolerConfig(
        node_type="Pair",
        index_selector=1,
        hidden_channels=embedding_dim,
    )
    return graph_pooler_config


def get_heads(embedding_dim: int, weights: dict):
    from omnicons.configs.HeadConfigs import GraphClsTaskHeadConfig

    heads = {}
    for label_name in tanimoto_bins:
        heads[label_name] = GraphClsTaskHeadConfig(
            hidden_size=embedding_dim,
            hidden_dropout_prob=0.1,
            num_labels=len(tanimoto_bins[label_name]["labels"]),
            class_weight=list(weights[label_name]),
            multi_label=False,
            analyze_inputs=["a"],
        )
    return heads


def get_model(
    weights: dict,
    embedding_dim: int,
    edge_embedding_dim: int,
    num_gnn_heads: int,
    num_transformer_heads: int,
    optimizer: Callable = get_deepspeed_adamw,
):
    # model setup
    vocab = get_vocab()
    node_encoders = get_node_encoders(
        vocab=vocab,
        embedding_dim=embedding_dim,
    )
    edge_type_encoder_config = get_edge_type_encoder(
        embedding_dim=edge_embedding_dim,
    )
    gnn_config = get_gnn(
        node_embedding_dim=embedding_dim,
        edge_embedding_dim=edge_embedding_dim,
        num_heads=num_gnn_heads,
    )
    transformer_config = get_transformer(
        embedding_dim=embedding_dim, num_heads=num_transformer_heads
    )
    graph_pooler_config = get_graph_pooler(embedding_dim=embedding_dim)
    heads = get_heads(embedding_dim=embedding_dim, weights=weights)
    # Metrics
    train_metrics = ModuleDict()
    val_metrics = ModuleDict()
    for label_name in tanimoto_bins:
        num_classes = len(tanimoto_bins[label_name]["labels"])
        key = f"{label_name}___a"
        train_metrics[key] = ClassificationMetrics.get(
            name=f"{key}_train",
            num_classes=num_classes,
            task="multiclass",
        )
        val_metrics[key] = ClassificationMetrics.get(
            name=f"{key}_val",
            num_classes=num_classes,
            task="multiclass",
        )
    # Instantiate a PyTorch Lightning Module
    model = HeteroGraphModelForMultiTaskLightning(
        node_encoders=node_encoders,
        edge_type_encoder_config=edge_type_encoder_config,
        gnn_config=gnn_config,
        transformer_config=transformer_config,
        graph_pooler_config=graph_pooler_config,
        heads=heads,
        optimizer_fn=optimizer,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        inputs=["a"],
        edge_types=edge_types,
    )
    return model
