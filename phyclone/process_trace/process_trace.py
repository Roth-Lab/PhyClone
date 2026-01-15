import os
import tarfile
import tempfile

import networkx as nx
import numpy as np
import pandas as pd

from phyclone.process_trace.consensus import get_consensus_tree
from phyclone.process_trace.map import get_map_node_ccfs_and_clonal_prev_dicts
from phyclone.process_trace.utils import print_string_to_file
from phyclone.tree import Tree
from phyclone.utils.math_utils import exp_normalize
from phyclone.utils.load_hdf5 import (
    load_chain_trace_data_df,
    load_clusters_df_from_trace,
    build_map_tree_from_trace,
    build_datapoints_dict_from_trace,
    load_samples_from_trace,
    build_df_trees_from_trace,
)
import click

from phyclone.utils.utils import print_command_header


def write_map_results(
    in_file,
    out_table_file,
    out_tree_file,
    out_sample_prev_table,
    map_type="joint-likelihood",
):

    print_command_header("Write MAP Tree Results")

    click.echo("Loading MAP tree from trace.")

    chain_trace_df = load_chain_trace_data_df(in_file)

    if map_type == "frequency":
        df = create_topology_dataframe(chain_trace_df)
        max_idx = df["count"].idxmax()
        map_iter = df.loc[max_idx, "iter"]
        chain_num = df.loc[max_idx, "chain_num"]
    else:
        max_idx = chain_trace_df["log_p"].idxmax()
        map_iter = chain_trace_df.loc[max_idx, "iter"]
        chain_num = chain_trace_df.loc[max_idx, "chain_num"]

    datapoints = build_datapoints_dict_from_trace(in_file)
    tree = build_map_tree_from_trace(in_file, chain_num, map_iter, datapoints)

    clusters = load_clusters_df_from_trace(in_file)
    samples = load_samples_from_trace(in_file)
    table, sample_prevs_df = get_clone_table(datapoints, samples, tree, clusters=clusters)

    _create_results_output_files(out_table_file, out_tree_file, out_sample_prev_table, table, tree, sample_prevs_df)
    print_finished_command_footer()


def print_finished_command_footer():
    click.secho("\nFinished.", fg="green")
    click.echo("#" * 100)


def write_consensus_results(
    in_file,
    out_table_file,
    out_tree_file,
    out_sample_prev_table,
    consensus_threshold=0.5,
    weight_type="joint-likelihood",
):

    print_command_header("Write Consensus Tree Results")

    click.echo(f"Weight type: {weight_type}")

    datapoints = build_datapoints_dict_from_trace(in_file)
    chain_trace_df = load_chain_trace_data_df(in_file)

    df = create_topology_dataframe(chain_trace_df)
    build_df_trees_from_trace(in_file, df, datapoints)

    probs = None

    if weight_type == "counts":
        weighted_consensus = False
    else:
        weighted_consensus = True

    trees = df["tree_obj"].to_numpy()
    counts = df["count"].to_numpy()

    if weighted_consensus:
        probs = df["log_p"].to_numpy()
        probs, _ = exp_normalize(probs)

    click.echo("\nBuilding consensus tree from trace.")

    graph = get_consensus_tree(
        trees,
        counts,
        data=datapoints,
        threshold=consensus_threshold,
        weighted=weighted_consensus,
        tree_probs=probs,
    )

    tree = get_tree_from_consensus_graph(datapoints, graph)
    clusters = load_clusters_df_from_trace(in_file)
    samples = load_samples_from_trace(in_file)
    table, sample_prevs_df = get_clone_table(datapoints, samples, tree, clusters=clusters)

    _create_results_output_files(out_table_file, out_tree_file, out_sample_prev_table, table, tree, sample_prevs_df)

    print_finished_command_footer()


def write_topology_report(in_file, out_file, topologies_archive=None, top_trees=float("inf")):
    if top_trees is None:
        top_trees = float("inf")

    print_command_header("Topology Report")

    datapoints = build_datapoints_dict_from_trace(in_file)
    chain_trace_df = load_chain_trace_data_df(in_file)

    click.echo("Extracting unique topologies from sample trace.")

    topology_df = create_topology_dataframe(chain_trace_df)
    topo_df_to_save = topology_df.drop(columns=["tree_hash", "time"], errors="ignore")
    topo_df_to_save.to_csv(out_file, index=False, sep="\t")
    num_unique_tree = len(topology_df)

    click.echo("Topology report created, saved as:\n{}".format(out_file))

    if topologies_archive is not None:
        click.echo()
        click.echo("#" * 50)

        if top_trees == float("inf"):
            top_trees_statement = "for all {}".format(num_unique_tree)
        else:
            top_trees_statement = "for the top {}".format(top_trees)
            if top_trees > num_unique_tree:
                click.echo(
                    "Warning: Number of top trees requested ({}) "
                    "is greater than the total number of "
                    "uniquely sampled topologies ({}).".format(top_trees, num_unique_tree)
                )
                top_trees_statement = "for all {}".format(num_unique_tree)
        click.echo("\nBuilding PhyClone topologies archive {} uniquely sampled topologies.".format(top_trees_statement))
        if top_trees >= num_unique_tree:
            topology_df = topology_df
            top_trees = int(num_unique_tree)
        else:
            top_trees = int(top_trees)
            topology_df = topology_df.head(n=top_trees)
        build_df_trees_from_trace(in_file, topology_df, datapoints)
        create_topologies_archive(topology_df, in_file, top_trees, topologies_archive, datapoints)
        click.echo("Topologies archive created, saved as:\n{}".format(topologies_archive))

    print_finished_command_footer()


def create_topologies_archive(topology_df, in_file, top_trees, topologies_archive, data):
    filename_template = "{}_results_table.tsv"
    nwk_template = "{}.nwk"
    sample_prevs_template = "{}_sample_prevs_table.tsv"
    clusters = load_clusters_df_from_trace(in_file)
    samples = load_samples_from_trace(in_file)
    with tarfile.open(topologies_archive, "w:gz") as archive:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with click.progressbar(length=top_trees, label="Building Trees") as bar:
                for idx, row in topology_df.iterrows():
                    tree = row["tree_obj"]
                    topology_id = row["topology_id"]
                    topology_rank = int(topology_id[2:])
                    if topology_rank >= top_trees:
                        continue
                    table, sample_prevs_df = get_clone_table(data, samples, tree, clusters=clusters)
                    filename = filename_template.format(topology_id)
                    filepath = os.path.join(tmp_dir, filename)
                    table.to_csv(filepath, index=False, sep="\t")
                    archive.add(filepath, arcname=str(os.path.join(topology_id, filename)))
                    nwk_filename = nwk_template.format(topology_id)
                    nwk_path = os.path.join(tmp_dir, nwk_filename)
                    print_string_to_file(tree.to_newick_string(), nwk_path)
                    archive.add(nwk_path, arcname=str(os.path.join(topology_id, nwk_filename)))
                    sample_prevs_filename = sample_prevs_template.format(topology_id)
                    sample_prevs_filepath = os.path.join(tmp_dir, sample_prevs_filename)
                    sample_prevs_df.to_csv(sample_prevs_filepath, index=False, sep="\t")
                    archive.add(sample_prevs_filepath, arcname=str(os.path.join(topology_id, sample_prevs_filename)))
                    bar.update(1)


def create_topology_dataframe(chain_trace_df):
    grouped = chain_trace_df.groupby("tree_hash")
    df_list = []
    for grp_name, grp in grouped:
        count = len(grp)
        best_one = grp.loc[grp["log_p"].idxmax()].copy()
        best_one["count"] = count
        df_list.append(best_one)
    df = pd.concat(df_list, axis=1, ignore_index=True).T
    df = df.convert_dtypes()
    df = df.sort_values(by="log_p", ascending=False, ignore_index=True)
    df.insert(0, "topology_id", "t_" + df.index.astype(str))
    return df


def _create_results_output_files(out_table_file, out_tree_file, out_sample_prev_table, table, tree, sample_prevs_df):
    table.to_csv(out_table_file, index=False, sep="\t")
    print_string_to_file(tree.to_newick_string(), out_tree_file)
    sample_prevs_df.to_csv(out_sample_prev_table, index=False, sep="\t")


def get_tree_from_consensus_graph(data, graph):

    graph2 = graph.copy()
    nodes = list(graph2.nodes)
    root_node_name = Tree._ROOT_NODE_NAME

    for node in nodes:
        if len(list(graph2.predecessors(node))) == 0:
            graph2.add_edge(root_node_name, node)

    tree = build_phyclone_tree_from_nx(graph2, data, root_node_name)
    tree.relabel_nodes()

    return tree


def build_phyclone_tree_from_nx(nx_tree, data_list, root_name):

    phyclone_tree = Tree(data_list[0].grid_size)

    post_order_nodes = nx.dfs_postorder_nodes(nx_tree, root_name)

    if isinstance(data_list, list):
        dp_dict = dict(zip([x.idx for x in data_list], data_list))
    else:
        dp_dict = data_list
        data_list = list(dp_dict.values())

    node_map = {}

    included_datapoints = set()

    for node in post_order_nodes:
        if node == root_name:
            continue

        translated_children = [node_map[child] for child in nx_tree.successors(node)]

        data = [dp_dict[idx] for idx in nx_tree.nodes[node]["idxs"]]
        included_datapoints.update(data)
        new_node = phyclone_tree.create_root_node(translated_children, data)
        node_map[node] = new_node

    outlier_datapoints = set(data_list) - included_datapoints

    for dp in outlier_datapoints:
        phyclone_tree.add_data_point_to_outliers(dp)

    return phyclone_tree


def get_clone_table(data, samples, tree, clusters=None):
    labels = get_labels_table(data, tree, clusters=clusters)
    ccfs, clonal_prev_dict = get_map_node_ccfs_and_clonal_prev_dicts(tree)

    outlier_node_name = tree.outlier_node_name

    if outlier_node_name not in ccfs:
        ccfs[outlier_node_name] = np.zeros(len(samples))

    if outlier_node_name not in clonal_prev_dict:
        clonal_prev_dict[outlier_node_name] = np.zeros(len(samples))

    prev_df_list = []
    for clone, sample_prevs in clonal_prev_dict.items():
        curr_df = pd.DataFrame(
            {
                "sample_id": samples,
                "clone_id": clone,
                "ccf": ccfs[clone],
                "clonal_prev": sample_prevs,
            }
        )
        prev_df_list.append(curr_df)

    sample_prevs_df = pd.concat(prev_df_list, ignore_index=True)
    sample_prevs_df.sort_values(by=["sample_id", "clone_id"], inplace=True, ignore_index=True)

    res_df = pd.merge(labels, sample_prevs_df, on="clone_id")

    sample_prevs_df = sample_prevs_df.loc[sample_prevs_df["clone_id"] != outlier_node_name]

    return res_df, sample_prevs_df


def get_labels_table(data, tree, clusters=None):
    df_records_list = []
    clone_muts = set()
    outlier_node_name = tree.outlier_node_name

    if clusters is None:
        tree_labels = tree.labels
        for idx in tree_labels:
            df_records_list.append(
                {
                    "mutation_id": data[idx].name,
                    "clone_id": tree_labels[idx],
                }
            )

            clone_muts.add(data[idx].name)

        for x in data.values():
            if x.name not in clone_muts:
                df_records_list.append({"mutation_id": x.name, "clone_id": outlier_node_name})

        df = pd.DataFrame(df_records_list)
        df = df.sort_values(by=["clone_id", "mutation_id"])

    else:
        tree_labels = tree.labels
        clusters_grouped = clusters.groupby("cluster_id")

        for idx in tree_labels:

            cluster_id = data[idx].name
            clone_id = tree_labels[idx]

            muts = clusters_grouped.get_group(cluster_id)["mutation_id"]
            muts_set = muts.unique()

            curr_muts_records = [
                {"mutation_id": mut, "clone_id": clone_id, "cluster_id": cluster_id} for mut in muts_set
            ]

            clone_muts.update(muts_set)

            df_records_list.extend(curr_muts_records)

        missing_muts_df = clusters.loc[~clusters["mutation_id"].isin(clone_muts)]
        missing_muts_df = missing_muts_df.copy()
        missing_muts_df["clone_id"] = outlier_node_name

        df_records_list.extend(missing_muts_df.to_dict("records"))

        df = pd.DataFrame(df_records_list)
        df = df.sort_values(by=["clone_id", "cluster_id", "mutation_id"], ignore_index=True)

    return df
