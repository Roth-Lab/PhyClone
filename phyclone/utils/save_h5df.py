import h5py
import numpy as np


def save_trace_to_h5df(results, out_file, minimal_cluster_df, rng_seed):
    num_chains = len(results)

    with h5py.File(out_file, "w", track_order=True) as fh:
        fh.create_dataset("samples", data=results[0]["samples"])
        run_info_grp = fh.create_group("run_info")
        run_info_grp.attrs["rng_seed"] = str(rng_seed)

        clusters_grp = fh.create_group("clusters")
        clusters_grp.create_dataset("cluster_id", data=minimal_cluster_df["cluster_id"].to_numpy())
        clusters_grp.create_dataset("mutation_id", data=minimal_cluster_df["mutation_id"].to_numpy())

        store_datapoints(fh, results)

        store_trace(fh, num_chains, results)


def store_trace(fh, num_chains, results):
    trace_grp = fh.create_group("trace")
    trace_grp.attrs["num_chains"] = num_chains
    chains_grp = trace_grp.create_group("chains")
    chain_template = "chain_{}"
    tree_template = "tree_{}"
    for chain, chain_results in results.items():
        chain_trace = chain_results["trace"]
        num_iters = len(chain_trace)
        chain_num = chain_results["chain_num"]
        curr_chain_grp = chains_grp.create_group(chain_template.format(chain_num))
        curr_chain_grp.attrs["chain_idx"] = chain_num
        curr_chain_grp.attrs["num_iters"] = num_iters

        store_chain_trace(chain_trace, curr_chain_grp, num_iters, tree_template)


def store_datapoints(fh, results):
    data_grp = fh.create_group("data")
    data = results[0]["data"]
    data_grp.attrs["num_datapoints"] = len(data)
    datapoints_grp = data_grp.create_group("datapoints")
    datapoint_template = "datapoint_{}"
    for datapoint in data:
        idx = datapoint.idx
        curr_dp_grp = datapoints_grp.create_group(datapoint_template.format(idx))
        curr_dp_grp.attrs["idx"] = idx
        curr_dp_grp.attrs["name"] = datapoint.name
        curr_dp_grp.attrs["outlier_prob"] = datapoint.outlier_prob
        curr_dp_grp.attrs["outlier_prob_not"] = datapoint.outlier_prob_not
        curr_dp_grp.attrs["outlier_marginal_prob"] = datapoint.outlier_marginal_prob
        curr_dp_grp.create_dataset("value", data=datapoint.value)


def store_chain_trace(chain_trace, curr_chain_grp, num_iters, tree_template):
    iters = np.empty(num_iters, dtype=np.int32)
    alpha = np.empty(num_iters)
    log_p_one = np.empty(num_iters)
    tree_hash = np.empty(num_iters, dtype=int)
    # tree_key_name = np.empty(num_iters, dtype=np.dtypes.StringDType)
    trees_grp = curr_chain_grp.create_group("trees")
    for i, iter_dict in enumerate(chain_trace):
        tree_iter = iter_dict["iter"]
        curr_tree_grp = trees_grp.create_group(tree_template.format(tree_iter))
        curr_tree_grp.attrs["iter"] = tree_iter
        curr_tree_grp.attrs["time"] = iter_dict["time"]
        curr_tree_grp.attrs["alpha"] = iter_dict["alpha"]
        curr_tree_grp.attrs["log_p_one"] = iter_dict["log_p_one"]
        curr_tree_grp.attrs["tree_hash"] = iter_dict["tree_hash"]

        iters[i] = tree_iter
        alpha[i] = iter_dict["alpha"]
        log_p_one[i] = iter_dict["log_p_one"]
        tree_hash[i] = iter_dict["tree_hash"]

        store_tree_dict(curr_tree_grp, iter_dict)
    chain_trace_data_grp = curr_chain_grp.create_group("trace_data")
    chain_trace_data_grp.create_dataset("iter", data=iters)
    chain_trace_data_grp.create_dataset("alpha", data=alpha)
    chain_trace_data_grp.create_dataset("log_p_one", data=log_p_one)
    chain_trace_data_grp.create_dataset("tree_hash", data=tree_hash)


def store_tree_dict(curr_tree_grp, iter_dict):
    subtree_grp = curr_tree_grp.create_group("tree")
    tree_dict = iter_dict["tree"]
    subtree_grp.create_dataset("graph", data=tree_dict["graph"])
    node_idx_grp = subtree_grp.create_group("node_idx")
    store_dict_mixed_type_keys(tree_dict["node_idx"], node_idx_grp)
    store_node_data(subtree_grp, tree_dict)
    subtree_grp.attrs["grid_size"] = tree_dict["grid_size"]
    node_last_added_to = tree_dict["node_last_added_to"]
    if node_last_added_to is None:
        subtree_grp.attrs["is_node_last_added_to_null"] = True
        subtree_grp.attrs["node_last_added_to"] = "None"
    else:
        subtree_grp.attrs["is_node_last_added_to_null"] = False
        subtree_grp.attrs["node_last_added_to"] = node_last_added_to
    subtree_grp.attrs["log_prior"] = tree_dict["log_prior"]


def store_node_data(subtree_grp, tree_dict):
    node_data_grp = subtree_grp.create_group("node_data")
    node_data = tree_dict["node_data"]
    num_nodes = len(node_data)
    node_data_grp.create_dataset("keys", data=list(node_data.keys()))
    val_dset = node_data_grp.create_dataset("values", shape=(num_nodes,), dtype=h5py.vlen_dtype(np.dtype('int32')), )
    for i, val in enumerate(node_data.values()):
        val_dset[i] = list(map(lambda x: x.idx, val))


def store_dict_mixed_type_keys(dict_to_store, parent_grp):
    str_nodes_grp = parent_grp.create_group("str_nodes")
    int_nodes_grp = parent_grp.create_group("int_nodes")
    str_nodes_keys = []
    int_nodes_keys = []
    str_nodes_values = []
    int_nodes_values = []
    for key, val in dict_to_store.items():
        if isinstance(key, str):
            str_nodes_keys.append(key)
            str_nodes_values.append(val)
        else:
            int_nodes_keys.append(key)
            int_nodes_values.append(val)
    str_nodes_grp.create_dataset("keys", data=str_nodes_keys, dtype=h5py.string_dtype())
    str_nodes_grp.create_dataset("values", data=str_nodes_values)
    int_nodes_grp.create_dataset("keys", data=int_nodes_keys)
    int_nodes_grp.create_dataset("values", data=int_nodes_values)