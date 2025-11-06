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
