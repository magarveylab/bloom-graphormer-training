from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


def get_trainer(
    checkpoint_dir: str,
    checkpoint_name: str,
    logger_entity: str,
    logger_name: str,
    logger_project: str,
    trainer_strategy: str,
):
    # Checkpoint Setup
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=checkpoint_dir,
        filename=checkpoint_name,
        save_top_k=3,
        save_last=True,
        mode="min",
        every_n_train_steps=1000,
    )

    # wandb logger
    wandb_logger = WandbLogger(
        entity=logger_entity,
        name=logger_name,
        project=logger_project,
    )

    # Setup Trainer
    trainer = Trainer(
        max_epochs=100000,
        callbacks=[checkpoint_callback],
        strategy=trainer_strategy,
        precision="16-mixed",
        logger=wandb_logger,
        accelerator="gpu",
        devices="auto",
    )
    return trainer
