import json
import os
import pickle
from glob import glob

import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from omnicons import dataset_dir
from omnicons.data.DataClass import MultiInputData


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
    ibis_data_dir: str = f"{dataset_dir}/ibis_quality",
    output_dir: str = f"{dataset_dir}/bgc_graphs",
):
    from Bloom.BloomEmbedder.graphs.BGCGraph import BGCGraph

    filenames = glob(f"{ibis_data_dir}/*")
    for fp in tqdm(filenames):
        out = BGCGraph.build_from_ibis_output(fp)
        for cluster in out:
            graph = cluster["graph"]
            cluster_id = graph.graph_id
            output_fp = f"{output_dir}/{cluster_id}.pkl"
            with open(output_fp, "wb") as f:
                pickle.dump(graph, f)


def prepare_lnk_graphs(
    dataset_fp: str = f"{dataset_dir}/bloom-lnk-datasets/final.csv",
    ibis_data_dir: str = f"{dataset_dir}/ibis_data",
    sm_dag_dir: str = f"{dataset_dir}/sm_dags",
    sm_graph_dir: str = f"{dataset_dir}/sm_graphs",
    output_dir: str = f"{dataset_dir}/bloom-lnk-graphs",
):
    from Bloom.BloomLNK.graph import MetaboloGraph
    from Bloom.BloomLNK.local.bgc_graph import (
        get_bgc_graphs,
        get_embeddings_for_bgc_graph,
        get_orf_to_dags,
        quality_control_bgc_filtering,
    )
    from Bloom.BloomLNK.local.unison import unite_graphs

    # prepare bgc graphs
    filenames = glob(f"{ibis_data_dir}/*")
    bgc_graphs = {}
    bgc_embeddings = {}
    for ibis_fp in tqdm(filenames):
        orf_to_dags = get_orf_to_dags(
            ibis_dir=ibis_fp,
        )
        clusters_to_run = quality_control_bgc_filtering(
            ibis_dir=ibis_fp,
            min_orf_count=1,
            min_module_count=1,
        )
        out = get_bgc_graphs(
            ibis_dir=ibis_fp,
            orf_to_dags=orf_to_dags,
            clusters_to_run=clusters_to_run,
        )
        for c in out:
            cluster_id = c["cluster_id"]
            graph = c["graph"]
            bgc_graphs[cluster_id] = graph
        bgc_embeddings.update(get_embeddings_for_bgc_graph(ibis_fp))
    # unite graphs
    data = pd.read_csv(dataset_fp).to_dict("records")
    for r in data:
        cluster_id = r["cluster_id"]
        if cluster_id not in bgc_embeddings:
            continue
        metabolite_id = r["metabolite_id"]
        graph_id = f"{metabolite_id}-{cluster_id}"
        mol_dag_fp = f"{sm_dag_dir}/{metabolite_id}.pkl"
        mol_graph_fp = f"{sm_graph_dir}/{metabolite_id}.pkl"
        graph = unite_graphs(
            mol_G=pickle.load(open(mol_graph_fp, "rb")),
            bgc_G=bgc_graphs[cluster_id],
            mol_dags=json.load(open(mol_dag_fp)),
        )
        G = MetaboloGraph.build_graph(
            graph_id=graph_id,
            graph=graph,
            orf_embedding=bgc_embeddings[cluster_id]["orfs"],
            domain_embedding=bgc_embeddings[cluster_id]["domains"],
        )
        output_fp = f"{output_dir}/{graph_id}.pkl"
        if os.path.exists(output_fp):
            continue
        pickle.dump(G, open(output_fp, "wb"))


def prepare_reaction_ec_dataset(
    rxn_dataset_fp: str = f"{dataset_dir}/reaction_ec.csv",
    output_dir: str = f"{dataset_dir}/reaction_ec_tensors",
):
    from Bloom.BloomRXN.inference.Preprocess import rxnsmiles2tensor
    from Bloom.BloomRXN.utils import get_atom_vocab, get_bond_vocab

    from omnicons.class_dicts import get_ec_class_dict

    atom_vocab = get_atom_vocab()
    bond_vocab = get_bond_vocab()
    ec1_class_dict = get_ec_class_dict(level=1, reverse=True)
    ec2_class_dict = get_ec_class_dict(level=2, reverse=True)
    ec3_class_dict = get_ec_class_dict(level=3, reverse=True)

    os.makedirs(f"{output_dir}/train", exist_ok=True)
    os.makedirs(f"{output_dir}/val", exist_ok=True)
    os.makedirs(f"{output_dir}/test", exist_ok=True)

    data = pd.read_csv(rxn_dataset_fp).to_dict("records")
    for rec in tqdm(data):
        split = rec["split"]
        reaction_id = rec["reaction_id"]
        rxn_smiles = rec["rxn_smiles"]
        output_fp = f"{output_dir}/{split}/{reaction_id}.pt"
        if os.path.exists(output_fp):
            continue
        data = rxnsmiles2tensor(
            rxn_smiles,
            atom_vocab=atom_vocab,
            bond_vocab=bond_vocab,
        )
        data.graphs["a"].ec1 = torch.LongTensor(
            [ec1_class_dict.get(rec["ec1"], -100)]
        )
        data.graphs["a"].ec2 = torch.LongTensor(
            [ec2_class_dict.get(rec["ec2"], -100)]
        )
        data.graphs["a"].ec3 = torch.LongTensor(
            [ec3_class_dict.get(rec["ec3"], -100)]
        )
        torch.save(data, output_fp)


def prepare_reaction_ec_fewshot_dataset(
    rxn_dataset_fp: str = f"{dataset_dir}/reaction_ec_fewshot_dataset",
    output_dir: str = f"{dataset_dir}/reaction_ec_fewshot_tensors",
):
    from Bloom.BloomRXN.inference.Preprocess import rxnsmiles2tensor
    from Bloom.BloomRXN.utils import get_atom_vocab, get_bond_vocab

    from omnicons.class_dicts import get_ec_class_dict

    atom_vocab = get_atom_vocab()
    bond_vocab = get_bond_vocab()
    ec1_class_dict = get_ec_class_dict(level=1, reverse=True)
    ec2_class_dict = get_ec_class_dict(level=2, reverse=True)
    ec3_class_dict = get_ec_class_dict(level=3, reverse=True)

    os.makedirs(f"{output_dir}/train", exist_ok=True)
    os.makedirs(f"{output_dir}/val", exist_ok=True)
    os.makedirs(f"{output_dir}/test", exist_ok=True)

    data = pd.read_csv(rxn_dataset_fp).to_dict("records")
    for rec in tqdm(data):
        split = rec["split"]
        rxn_id_a = rec["rxn_id_a"]
        rxn_id_b = rec["rxn_id_b"]
        output_fp = f"{output_dir}/{split}/{rxn_id_a}_{rxn_id_b}.pt"
        if os.path.exists(output_fp):
            continue
        rxn_smiles_a = rec["rxn_smiles_a"]
        rxn_smiles_b = rec["rxn_smiles_b"]
        # create tensors
        data_a = rxnsmiles2tensor(
            rxn_smiles_a,
            atom_vocab=atom_vocab,
            bond_vocab=bond_vocab,
        )
        data_b = rxnsmiles2tensor(
            rxn_smiles_b,
            atom_vocab=atom_vocab,
            bond_vocab=bond_vocab,
        )
        common_y = Data(
            pair_match___a___b=torch.LongTensor([rec["pair_match"]])
        )
        data = MultiInputData(
            graphs={"a": data_a.graphs["a"], "b": data_b.graphs["a"]},
            common_y=common_y,
        )
        # add ec labels to graphs a
        data.graphs["a"].ec1 = torch.LongTensor(
            [ec1_class_dict["ec1"].get(rec["ec1_a"], -100)]
        )
        data.graphs["a"].ec2 = torch.LongTensor(
            [ec2_class_dict["ec2"].get(rec["ec2_b"], -100)]
        )
        data.graphs["a"].ec3 = torch.LongTensor(
            [ec3_class_dict["ec3"].get(rec["ec3_a"], -100)]
        )
        # add ec labels to graphs b
        data.graphs["b"].ec1 = torch.LongTensor(
            [ec1_class_dict["ec1"].get(rec["ec1_b"], -100)]
        )
        data.graphs["b"].ec2 = torch.LongTensor(
            [ec2_class_dict["ec2"].get(rec["ec2_b"], -100)]
        )
        data.graphs["b"].ec3 = torch.LongTensor(
            [ec3_class_dict["ec3"].get(rec["ec3_b"], -100)]
        )
        torch.save(data, output_fp)
