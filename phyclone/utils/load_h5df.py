import h5py
import numpy as np
import pandas as pd


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