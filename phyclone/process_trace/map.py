import numpy as np
from numba import njit
from phyclone.process_trace.utils import convert_rustworkx_to_networkx


def get_map_node_ccfs_and_clonal_prev_dicts(tree):
    root_node_name = tree.root_node_name
    graph = compute_map_tree_features(tree, root_node_name)

    ccf_dict = get_map_ccfs(graph, root_node_name)
    clonal_prev_dict = get_map_clonal_prev_dict(graph, root_node_name)

    return ccf_dict, clonal_prev_dict


def compute_map_tree_features(tree, root_node_name):
    graph = tree._graph.copy()
    graph = convert_rustworkx_to_networkx(graph)
    compute_max_likelihood(graph, root_node_name)
    set_max_assignment(graph, root_node_name)
    return graph


def get_map_clonal_prev_dict(tree, root_node):

    num_vals = tree.nodes[root_node]["log_R"].shape[1] - 1

    result_dict = {}

    for node in tree.nodes:
        clonal_prev = tree.nodes[node]["max_idx"].astype(np.float64)

        for child in tree.successors(node):
            clonal_prev -= tree.nodes[child]["max_idx"]

        clonal_prev /= num_vals

        result_dict[node] = clonal_prev

    return result_dict


def compute_max_likelihood(graph, node_id):
    children = list(graph.successors(node_id))

    for child_id in children:
        compute_max_likelihood(graph, child_id)

    node = graph.nodes[node_id]

    if len(children) == 0:
        node["log_S_max"] = np.zeros(node["log_p"].shape)
        node["log_R_max"] = node["log_p"]

    else:
        child_log_R = [graph.nodes[child_id]["log_R_max"] for child_id in children]

        node["log_D_choice"], node["log_S_choice"], node["log_S_max"] = compute_log_S(child_log_R)
        node["log_R_max"] = node["log_p"] + node["log_S_max"]


def compute_log_S(child_log_R_values):
    log_D_choice, log_D = compute_log_D(child_log_R_values)

    log_S = np.zeros(log_D.shape)

    log_S_choice = np.zeros(log_D.shape, dtype=int)

    _compute_log_S(log_D, log_S, log_S_choice)

    return log_D_choice, log_S_choice, log_S


@njit
def _compute_log_S(log_D, log_S, log_S_choice):
    grid_size = log_D.shape[1]
    num_dims = log_D.shape[0]
    log_S[:, 0] = log_D[:, 0]
    for i in range(num_dims):
        for j in range(1, grid_size):
            if log_D[i, j] > log_S[i, j - 1]:
                log_S[i, j] = log_D[i, j]

                log_S_choice[i, j] = j

            else:
                log_S[i, j] = log_S[i, j - 1]

                log_S_choice[i, j] = log_S_choice[i, j - 1]


def compute_log_D(child_log_R_values):
    log_D = np.zeros(child_log_R_values[0].shape)
    log_D_choice = []

    num_dims = log_D.shape[0]
    log_d_shape = log_D.shape

    for child_log_R in child_log_R_values:

        child_choices = np.zeros(log_d_shape, dtype=np.int32)
        child_results = np.full(log_d_shape, -np.inf)

        for i in range(num_dims):
            _compute_log_D_n(child_log_R[i, :], log_D[i, :], child_choices[i], child_results[i])
            log_D[i, :] = child_results[i]

        log_D_choice.append(child_choices)

    return log_D_choice, log_D


@njit
def _compute_log_D_n(child_log_R, prev_log_D_n, choice, result):
    grid_size = len(prev_log_D_n)

    for i in range(grid_size):
        for j in range(i + 1):
            val = child_log_R[j] + prev_log_D_n[i - j]

            if val >= result[i]:
                choice[i] = j

                result[i] = val


def set_max_assignment(graph, root_node_name):
    num_dims, num_vals = graph.nodes[root_node_name]["log_R"].shape

    comp_vals = num_vals - 1

    graph.nodes[root_node_name]["max_idx"] = np.full(num_dims, comp_vals, dtype=int)

    _set_max_assignment(graph, np.full(num_dims, comp_vals, dtype=int), root_node_name)


def _set_max_assignment(graph, idxs, node):
    num_dims = len(idxs)

    children = list(graph.successors(node))

    if len(children) == 0:
        return

    child_total_idx = [graph.nodes[node]["log_S_choice"][d, idxs[d]] for d in range(num_dims)]

    for i in range(len(children) - 1, -1, -1):
        child = children[i]

        graph.nodes[child]["max_idx"] = np.zeros(num_dims, dtype=int)

        for d in range(num_dims):
            graph.nodes[child]["max_idx"][d] = graph.nodes[node]["log_D_choice"][i][d, child_total_idx[d]]

            child_total_idx[d] -= graph.nodes[child]["max_idx"][d]

        _set_max_assignment(graph, graph.nodes[child]["max_idx"], children[i])


def get_map_ccfs(graph, root_node):
    num_vals = graph.nodes[root_node]["log_R"].shape[1] - 1

    result = {}

    for node in graph.nodes:

        result[node] = graph.nodes[node]["max_idx"] / num_vals

    return result
