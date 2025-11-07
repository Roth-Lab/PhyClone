import h5py
import numpy as np
import pandas as pd
from phyclone.data.base import DataPoint
from phyclone.tree import Tree


def load_chain_trace_data_df(in_file):
    chain_df_list = []
    with h5py.File(in_file) as fh:
        result_chains = fh["trace"]["chains"]

        for chain, chain_grp in result_chains.items():
            chain_idx = chain_grp.attrs["chain_idx"]
            chain_trace_data = chain_grp["trace_data"]
            df_dict = {k:v[()] for k, v in chain_trace_data.items()}
            df = pd.DataFrame(df_dict)
            df["chain"] = chain_idx
            chain_df_list.append(df)

    final_df = pd.concat(chain_df_list, ignore_index=True)
    return final_df


def load_clusters_df_from_trace(in_file):
    df_dict = dict()
    with h5py.File(in_file) as fh:
        clusters_grp = fh["clusters"]
        # df_dict = {k: v[()] for k, v in clusters_grp.items()}
        #TODO: set this up to be more flexible on type, IDs could be strings or numbers for any of these
        df_dict["cluster_id"] = clusters_grp["cluster_id"][()]
        df_dict["mutation_id"] = clusters_grp["mutation_id"][()].astype('T')

    df = pd.DataFrame(df_dict)
    return df


def build_datapoints_dict_from_trace(in_file):
    datapoints = {}
    with h5py.File(in_file) as fh:
        datapoints_grp = fh["data"]["datapoints"]

        for dp_name, dp_grp in datapoints_grp.items():
            attrs_dict = dp_grp.attrs
            idx = attrs_dict["idx"]
            name = attrs_dict["name"]
            outlier_prob = attrs_dict["outlier_prob"]
            outlier_prob_not = attrs_dict["outlier_prob_not"]
            outlier_marginal_prob = attrs_dict["outlier_marginal_prob"]
            value = dp_grp["value"][()]
            datapoints[idx] = DataPoint(idx, value, name, outlier_prob, outlier_prob_not, outlier_marginal_prob)
    return datapoints


# def tmp_tree_build(in_file, chain=0, iteration=2):
#     datapoints = build_datapoints_dict_from_trace(in_file)
#
#     with h5py.File(in_file) as fh:
#         tree_dict = build_tree_dict_from_trace(chain, iteration, fh, datapoints)
#         tree = Tree.from_dict(tree_dict)
#         print('d')


def build_tree_dict_from_trace(chain, iteration, fh, datapoints):
    chain_template = "chain_{}"
    chain_name = chain_template.format(chain)
    tree_template = "tree_{}"
    tree_name = tree_template.format(iteration)
    tree_grp = fh["trace"]["chains"][chain_name]["trees"][tree_name]["tree"]

    node_idx_dict = _get_node_idx_dict(tree_grp)

    if tree_grp.attrs["is_node_last_added_to_null"]:
        node_last_added_to = None
    else:
        node_last_added_to = tree_grp.attrs["node_last_added_to"]

    tree_dict = {
        "graph": list(map(tuple, tree_grp["graph"][()])),
        "node_idx": node_idx_dict,
        "node_idx_rev": {v:k for k,v in node_idx_dict.items()},
        "node_data": _get_node_data_dict(datapoints, tree_grp),
        "grid_size": tree_grp.attrs["grid_size"],
        "node_last_added_to": node_last_added_to,
        "log_prior": tree_grp.attrs["log_prior"],
    }

    return tree_dict


def _get_node_data_dict(datapoints, tree_grp):
    node_data_grp = tree_grp["node_data"]
    node_data_dict = dict()
    val_dset = node_data_grp["values"][()]
    node_keys = node_data_grp["keys"][()]
    for i, node_data in enumerate(val_dset):
        node_data_dict[node_keys[i]] = [datapoints[dp_idx] for dp_idx in node_data]
    return node_data_dict


def _get_node_idx_dict(tree_grp):
    node_idx_grp = tree_grp["node_idx"]
    node_idx_dict = dict(
        zip(node_idx_grp["str_nodes"]["keys"][()].astype('T'), node_idx_grp["str_nodes"]["values"][()]),)
    node_idx_dict.update(dict(zip(node_idx_grp["int_nodes"]["keys"][()], node_idx_grp["int_nodes"]["values"][()])))
    return node_idx_dict