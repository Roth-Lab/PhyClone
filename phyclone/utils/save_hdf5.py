import h5py
import numpy as np
import click


def save_trace_to_h5df(results, out_file, minimal_cluster_df, rng_seed, samples, data):
    num_chains = len(results)
    click.echo()
    click.echo("#" * 20)
    click.echo("\nWriting sample trace to disk.")

    with h5py.File(out_file, "w", track_order=True) as fh:
        fh.create_dataset("samples", data=samples, compression="gzip")
        run_info_grp = fh.create_group("run_info")
        run_info_grp.attrs["rng_seed"] = str(rng_seed)

        if minimal_cluster_df is not None:
            clusters_grp = fh.create_group("clusters")
            clusters_grp.create_dataset(
                "cluster_id",
                data=minimal_cluster_df["cluster_id"].to_numpy(),
                compression="gzip",
            )
            clusters_grp.create_dataset(
                "mutation_id",
                data=minimal_cluster_df["mutation_id"].to_numpy(),
                compression="gzip",
            )

        idx_dtype = store_datapoints(fh, data)

        store_trace(fh, num_chains, results, idx_dtype)

    click.secho("\nFinished.", fg="green")


def store_trace(fh, num_chains, results, idx_dtype):
    trace_grp = fh.create_group("trace")
    trace_grp.attrs["num_chains"] = num_chains
    chains_grp = trace_grp.create_group("chains")
    chain_template = "chain_{}"
    tree_template = "tree_{}"
    tree_obj_dict = dict()
    for chain, chain_results in results.items():
        chain_trace = chain_results["trace"]
        num_iters = len(chain_trace)
        chain_num = chain_results["chain_num"]
        curr_chain_grp = chains_grp.create_group(chain_template.format(chain_num))
        curr_chain_grp.attrs["chain_idx"] = chain_num
        curr_chain_grp.attrs["num_iters"] = num_iters
        with click.progressbar(length=num_iters, label=f"Saving chain {chain_num} trace") as bar:
            store_chain_trace(chain_trace, curr_chain_grp, num_iters, tree_template, tree_obj_dict, bar, idx_dtype)
    click.echo(f"\nUnique trees sampled: {len(tree_obj_dict)}")


def store_datapoints(fh, data):
    data_grp = fh.create_group("data")
    data_grp.attrs["num_datapoints"] = len(data)
    datapoints_grp = data_grp.create_group("datapoints")
    datapoint_template = "datapoint_{}"
    min_idx = np.inf
    max_idx = -np.inf
    for datapoint in data:
        idx = datapoint.idx
        curr_dp_grp = datapoints_grp.create_group(datapoint_template.format(idx))
        curr_dp_grp.attrs["idx"] = idx
        curr_dp_grp.attrs["name"] = datapoint.name
        curr_dp_grp.attrs["outlier_prob"] = datapoint.outlier_prob
        curr_dp_grp.attrs["outlier_prob_not"] = datapoint.outlier_prob_not
        curr_dp_grp.attrs["outlier_marginal_prob"] = datapoint.outlier_marginal_prob
        curr_dp_grp.create_dataset("value", data=datapoint.value, compression="gzip")
        min_idx = min(idx, min_idx)
        max_idx = max(idx, max_idx)
    idx_dtype = np.result_type(np.min_scalar_type(max_idx), np.min_scalar_type(min_idx))
    return idx_dtype


def store_chain_trace(chain_trace, curr_chain_grp, num_iters, tree_template, tree_obj_dict, bar, idx_dtype):
    iters = np.empty(num_iters, dtype=np.uint32)
    time = np.empty(num_iters)
    alpha = np.empty(num_iters)
    log_p_one = np.empty(num_iters)
    tree_hash = np.empty(num_iters, dtype=int)

    trees_grp = curr_chain_grp.create_group("trees")
    for i, iter_obj in enumerate(chain_trace):
        tree_iter = iter_obj.iter
        curr_tree_grp = trees_grp.create_group(tree_template.format(tree_iter))

        tree_hash_val = hash(iter_obj.tree_hash)

        iters[i] = tree_iter
        alpha[i] = iter_obj.alpha
        time[i] = iter_obj.time
        log_p_one[i] = iter_obj.log_p_one
        tree_hash[i] = tree_hash_val

        tree_group_ref = store_tree_dict(curr_tree_grp, iter_obj, tree_hash_val, tree_obj_dict, idx_dtype)
        curr_tree_grp.attrs["tree_group"] = tree_group_ref
        bar.update(1)
    chain_trace_data_grp = curr_chain_grp.create_group("trace_data")
    chain_trace_data_grp.create_dataset("iter", data=iters, compression="gzip")
    chain_trace_data_grp.create_dataset("time", data=time, compression="gzip")
    chain_trace_data_grp.create_dataset("alpha", data=alpha, compression="gzip")
    chain_trace_data_grp.create_dataset("log_p_one", data=log_p_one, compression="gzip")
    chain_trace_data_grp.create_dataset("tree_hash", data=tree_hash, compression="gzip")


def store_tree_dict(curr_tree_grp, iter_obj, tree_hash_val, tree_obj_dict, idx_dtype):
    if tree_hash_val in tree_obj_dict:
        return tree_obj_dict[tree_hash_val]
    else:
        subtree_grp = curr_tree_grp.create_group("tree")
        tree_obj_dict[tree_hash_val] = subtree_grp.ref
        tree_storage_obj = iter_obj.tree
        subtree_grp.create_dataset("graph", data=_downcast_edge_list(tree_storage_obj.graph))
        node_idx_grp = subtree_grp.create_group("node_idx")
        store_dict_mixed_type_keys(tree_storage_obj.node_idx, node_idx_grp)
        store_node_data(subtree_grp, tree_storage_obj, idx_dtype)
        subtree_grp.attrs["grid_size"] = tree_storage_obj.grid_size
        subtree_grp.attrs["log_prior"] = tree_storage_obj.log_prior
        return subtree_grp.ref


def _downcast_edge_list(edge_list):
    arr = np.asarray(edge_list)
    if len(arr) > 0:
        smallest_dtype_max_val = np.min_scalar_type(np.nanmax(arr))
        smallest_dtype_min_val = np.min_scalar_type(np.nanmin(arr))
        new_dtype = np.result_type(smallest_dtype_max_val, smallest_dtype_min_val)
        arr = arr.astype(new_dtype)
    return arr


def store_node_data(subtree_grp, tree_storage_obj, idx_dtype):
    node_data_grp = subtree_grp.create_group("node_data")
    node_data = tree_storage_obj.node_data

    int_nodes_keys, int_nodes_values, str_nodes_keys, str_nodes_values = _split_dict_by_key_type(node_data)
    str_nodes_grp = node_data_grp.create_group("str_nodes")
    int_nodes_grp = node_data_grp.create_group("int_nodes")

    _store_node_data_typed_dict(h5py.string_dtype(), str_nodes_grp, str_nodes_keys, str_nodes_values, idx_dtype)
    _store_node_data_typed_dict(None, int_nodes_grp, int_nodes_keys, int_nodes_values, idx_dtype)


def _store_node_data_typed_dict(dtype_to_use, dict_grp, dict_keys, dict_values, idx_dtype):
    num_vals = len(dict_keys)
    dict_grp.create_dataset("keys", data=dict_keys, dtype=dtype_to_use)
    val_dset = dict_grp.create_dataset(
        "values",
        shape=(num_vals,),
        dtype=h5py.vlen_dtype(np.dtype(idx_dtype)),
    )
    for i, val in enumerate(dict_values):
        val_dset[i] = val


def store_dict_mixed_type_keys(dict_to_store, parent_grp):
    str_nodes_grp = parent_grp.create_group("str_nodes")
    int_nodes_grp = parent_grp.create_group("int_nodes")
    int_nodes_keys, int_nodes_values, str_nodes_keys, str_nodes_values = _split_dict_by_key_type(dict_to_store)
    str_nodes_grp.create_dataset("keys", data=str_nodes_keys, dtype=h5py.string_dtype())
    str_nodes_grp.create_dataset("values", data=str_nodes_values)
    int_nodes_grp.create_dataset("keys", data=int_nodes_keys)
    int_nodes_grp.create_dataset("values", data=int_nodes_values)


def _split_dict_by_key_type(dict_to_store):
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
    return int_nodes_keys, int_nodes_values, str_nodes_keys, str_nodes_values
