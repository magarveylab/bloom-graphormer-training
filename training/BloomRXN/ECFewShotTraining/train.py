import argparse
import os
from multiprocessing import freeze_support

from DataModule import ReactionDataModule
from models import get_model

from omnicons import experiment_dir
from omnicons.trainers import get_trainer


def train(
    checkpoint_dir: str = f"{experiment_dir}/bloom-rxn-ec-fewshot/checkpoints",
    pretrained_checkpoint_fp: str = f"{experiment_dir}/bloom-rxn-ec/checkpoints/last.pt",
    checkpoint_name: str = "bloom-rxn-ec-fewshot-{epoch:02d}-{val_loss:.2f}",
    logger_entity: str = "magarvey",
    logger_name: str = "bloom-rxn-ec-fewshot",
    logger_project: str = "BLOOM",
    trainer_strategy: str = "deepspeed_stage_3_offload",
    embedding_dim: int = 256,
):
    # setup directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    # data module
    dm = ReactionDataModule()
    dm.setup()
    # model
    model = get_model(
        dm=dm,
        embedding_dim=embedding_dim,
        pretrained_checkpoint_fp=pretrained_checkpoint_fp,
    )
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


parser = argparse.ArgumentParser(description="Train Bloom-RXN EC")
parser.add_argument(
    "-checkpoint_dir",
    help="Directory to save checkpoints",
    default=f"{experiment_dir}/bloom-rxn-ec-fewshot/checkpoints",
)
parser.add_argument(
    "-mlm_checkpoint_fp",
    help="Pretrained checkpoint for EC Training",
    default=f"{experiment_dir}/bloom-rxn-ec/checkpoints/last.pt",
)
parser.add_argument(
    "-checkpoint_name",
    help="checkpoint name for wandb",
    default="bloom-rxn-ec-{epoch:02d}-{val_loss:.2f}",
)
parser.add_argument(
    "-logger_entity",
    help="wandb entity",
    default="user",
)
parser.add_argument(
    "-logger_name",
    help="wandb entity",
    default="bloom-rxn-ec-fewshot",
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
