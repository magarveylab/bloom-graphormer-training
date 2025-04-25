import argparse
import os
from multiprocessing import freeze_support

from DataModule import ReactionDataModule
from models import get_model

from omnicons import experiment_dir
from omnicons.trainers import get_trainer


def train(
    checkpoint_dir: str = f"{experiment_dir}/bloom-rxn-mlm/checkpoints",
    checkpoint_name: str = "bloom-rxn-mlm-{epoch:02d}-{val_loss:.2f}",
    logger_entity: str = "magarvey",
    logger_name: str = "bloom-rxn-mlm",
    logger_project: str = "BLOOM",
    trainer_strategy: str = "deepspeed_stage_3_offload",
    embedding_dim: int = 256,
):
    # setup directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    # data module
    dm = ReactionDataModule(
        data_dir="/home/gunam/storage/workspace/bearlinker_workspace/zenodo/reaction_ec_tensors"
    )
    # model
    model = get_model(embedding_dim=embedding_dim)
    # trainer
    trainer = get_trainer(
        checkpoint_dir=checkpoint_dir,
        checkpoint_name=checkpoint_name,
        logger_entity=logger_entity,
        logger_name=logger_name,
        logger_project=logger_project,
        trainer_strategy=trainer_strategy,
    )
    trainer.fit(model, dm)


parser = argparse.ArgumentParser(description="Train Bloom-RXN MLM")
parser.add_argument(
    "-checkpoint_dir",
    help="Directory to save checkpoints",
    default=f"{experiment_dir}/bloom-rxn-mlm/checkpoints",
)
parser.add_argument(
    "-checkpoint_name",
    help="checkpoint name for wandb",
    default="bloom-rxn-mlm-{epoch:02d}-{val_loss:.2f}",
)
parser.add_argument(
    "-logger_entity",
    help="wandb entity",
    default="magarvey",
)
parser.add_argument(
    "-logger_name",
    help="wandb entity",
    default="bloom-rxn-mlm",
)
parser.add_argument(
    "-embedding_dim",
    help="node embedding dimension",
    default=256,
)

if __name__ == "__main__":
    args = parser.parse_args()
    freeze_support()
    train(
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name=args.checkpoint_name,
        logger_entity=args.logger_entity,
        logger_name=args.logger_name,
        embedding_dim=args.embedding_dim,
    )
