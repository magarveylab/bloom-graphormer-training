import argparse

from pytorch_lightning.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict,
)

from omnicons import experiment_dir


def save(
    checkpoint_dir: str = f"{experiment_dir}/bloom-bgc-mlm/checkpoints",
):
    convert_zero_checkpoint_to_fp32_state_dict(
        f"{checkpoint_dir}/last.ckpt/",
        f"{checkpoint_dir}/last.pt",
    )


parser = argparse.ArgumentParser(
    description="Convert Bloom-BGC MLM model to pytorch checkpoint."
)
parser.add_argument(
    "-checkpoint_dir",
    help="Directory to save checkpoints",
    default=f"{experiment_dir}/bloom-bgc-mlm/checkpoints",
)


if __name__ == "__main__":
    args = parser.parse_args()
    save(checkpoint_dir=args.checkpoint_dir)
