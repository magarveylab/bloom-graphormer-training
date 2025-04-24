from typing import Literal

import pandas as pd

from omnicons import dataset_dir


def get_ec_class_dict(level: Literal[1, 2, 3, 4] = 1) -> dict:
    class_fp = f"{dataset_dir}/class_labels/ec{level}.csv"
    df = pd.read_csv(class_fp)
    return dict(zip(df.index, df.annotation))
