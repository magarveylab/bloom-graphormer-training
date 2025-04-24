import os
import pickle
from glob import glob

import pandas as pd
from tqdm import tqdm

from omnicons import dataset_dir


def prepare_molecular_graphs(
    dataset_fp: str = f"{dataset_dir}/molecule_training_data.csv",
    bloom_dos_dir: str = f"{dataset_dir}/bloom_dos_annotations",
    output_dir: str = f"{dataset_dir}/molecular_graphs",
    cpus: int = 10,
):
    from Bloom import BloomDOS
    from Bloom.BloomEmbedder.graphs.MoleculeGraph import MoleculeGraph

    # get bloom annotations
    data = pd.read_csv(dataset_fp).to_dict("records")
    BloomDOS.multiprocess_subission(
        submission_list=data, output_dir=bloom_dos_dir, cpus=cpus
    )

    # get molecular graphs
    filenames = glob(f"{bloom_dos_dir}/*.json")
    for fp in tqdm(filenames, desc="Creating molecular graphs"):
        metabolite_id = int(fp.split("/")[-1].split(".")[0])
        output_fp = f"{output_dir}/{metabolite_id}.pkl"
        if os.path.exists(output_fp):
            continue
        # get bloom annotations
        out = MoleculeGraph.build_from_bloom_graph(
            graph_id=metabolite_id,
            bloom_graph_fp=fp,
        )
        G = out["graph"]
        pickle.dump(G, open(output_fp, "wb"))


def prepare_bgc_graphs(
    ibis_data_dir: str = f"{dataset_dir}/ibis_data",
):
    pass


def prepare_lnk_graphs(
    ibis_data_dir: str = f"{dataset_dir}/ibis_data",
    bloom_dos_dir: str = f"{dataset_dir}/bloom_dos_annotations",
):
    # prepare dags

    # preare mol graphs

    # preapre bgc graphs

    # unite graphs

    pass
