import argparse
import os

import torch
from DataModule import ReactionDataModule
from models import get_model

from omnicons import experiment_dir
from omnicons.models.Compilers import compile_with_torchscript


def compile_model(
    pytorch_checkpoint_fp: str = f"{experiment_dir}/bloom-rxn-ec-fewshot/checkpoints/last.pt",
    torchscript_dir: str = f"{experiment_dir}/bloom-rxn/torchscript",
    embedding_dim: int = 128,
):
    # data module
    dm = ReactionDataModule()
    dm.setup()
    # model
    model = get_model(
        dm=dm,
        embedding_dim=embedding_dim,
        pretrained_checkpoint_fp=pytorch_checkpoint_fp,
    )
    # load weights
    states = torch.load(pytorch_checkpoint_fp)
    model.load_state_dict(states["state_dict"], strict=True)
    # export model
    os.makedirs(torchscript_dir, exist_ok=True)
    # compile models
    compile_with_torchscript(
        model=model.model.edge_encoder,
        model_fp=f"{torchscript_dir}/node_encoder.pt",
    )
    compile_with_torchscript(
        model=model.model.edge_encoder,
        model_fp=f"{torchscript_dir}/edge_encoder.pt",
    )
    compile_with_torchscript(
        model=model.model.gnn, model_fp=f"{torchscript_dir}/gnn.pt"
    )
    compile_with_torchscript(
        model=model.model.transformer,
        model_fp=f"{torchscript_dir}/transformer.pt",
    )


parser = argparse.ArgumentParser(
    description="Convert Bloom-RXN to torchscript format"
)
parser.add_argument(
    "-torchscript_dir",
    help="Directory to save torchscript models",
    default=f"{experiment_dir}/bloom-rxn/torchscript",
)
parser.add_argument(
    "-pytorch_checkpoint_fp",
    help="Pytorch checkpoint file path",
    default=f"{experiment_dir}/bloom-rxn-ec-fewshot/checkpoints/last.pt",
)

if __name__ == "__main__":
    args = parser.parse_args()
    compile_model(
        torchscript_dir=args.torchscript_dir,
        pytorch_checkpoint_fp=args.pytorch_checkpoint_fp,
    )
