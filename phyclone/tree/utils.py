import numpy as np

from phyclone.utils import two_np_arr_cache, list_of_np_cache
from phyclone.utils.math import fft_convolve_two_children, conv_over_dims, np_conv_dims, _np_conv_dims


@list_of_np_cache(maxsize=4096)
def compute_log_S(child_log_R_values):
    """Compute log(S) recursion.

    Parameters
    ----------
    child_log_R_values: ndarray
        log_R values from child nodes.
    """
    if len(child_log_R_values) == 0:
        return 0.0

    log_D = compute_log_D(child_log_R_values)
    log_S = np.empty_like(log_D)
    log_S = np.logaddexp.accumulate(log_D, out=log_S, axis=-1)

    return np.ascontiguousarray(log_S)


# def _sub_compute_S(log_D):
#     log_S = np.empty_like(log_D)
#     num_dims = log_D.shape[0]
#     for i in range(num_dims):
#         np.logaddexp.accumulate(log_D[i, :], out=log_S[i, :])
#     return log_S


def compute_log_D(child_log_R_values):
    num_children = len(child_log_R_values)

    if num_children == 0:
        return 0

    if num_children == 1:
        return child_log_R_values[0]

    all_maxes = np.max(child_log_R_values, axis=-1, keepdims=True)
    normed_children = np.empty_like(child_log_R_values, order="C")

    np.subtract(child_log_R_values, all_maxes, order="C", dtype=np.float64, out=normed_children)

    np.exp(normed_children, order="C", dtype=np.float64, out=normed_children)

    # normed_children = np.ascontiguousarray(normed_children)

    conv_res = _convolve_two_children(normed_children[0], normed_children[1])
    for j in range(2, num_children):
        conv_res = _convolve_two_children(normed_children[j], conv_res)

    # conv_res[conv_res <= 0] = 1e-100
    log_d = conv_res.copy()
    log_d[log_d <= 0] = 1e-100
    # log_d = np.clip(conv_res, a_min=1e-100, a_max=None, out=log_d)
    # conv_res.clip(min=1e-100, max=None, order="C", dtype=np.float64, out=log_d)
    # np.maximum(1e-100, conv_res, order="C", dtype=np.float64, out=log_d)

    np.log(log_d, order="C", dtype=np.float64, out=log_d)

    log_d += all_maxes.sum(0)

    return log_d



# def compute_log_D(child_log_R_values):
#     num_children = len(child_log_R_values)
#
#     if num_children == 0:
#         return 0
#
#     if num_children == 1:
#         return child_log_R_values[0]
#
#     # child_log_R_values = np.ascontiguousarray(child_log_R_values)
#
#     conv_res = _convolve_two_children(child_log_R_values[0], child_log_R_values[1])
#     for j in range(2, num_children):
#         conv_res = _convolve_two_children(child_log_R_values[j], conv_res)
#
#     # log_D = conv_res
#     return conv_res


@two_np_arr_cache(maxsize=2048)
def _convolve_two_children(child_1, child_2):
    grid_size = child_1.shape[-1]
    if grid_size < 1000:
        # res_arr = conv_over_dims(child_1, child_2, np.zeros_like(child_1, order="C"))
        # res_arr = _np_conv_dims(child_1, child_2)
        res_arr = np_conv_dims(child_1, child_2)
        # res_arr.clip(min=1e-100, max=None, out=res_arr)
        # res_arr[res_arr <= 0] = 1e-100
    else:
        res_arr = np.ascontiguousarray(fft_convolve_two_children(child_1, child_2))
    return res_arr


def get_clades(tree):
    result = set()

    for root in tree.roots:
        _clades(result, root, tree)

    return frozenset(result)


def _clades(clades, node, tree):
    current_clade = set()

    for mutation in tree.get_data(node):
        current_clade.add(mutation.idx)

    for child in tree.get_children(node):
        for mutation in _clades(clades, child, tree):
            current_clade.add(mutation)

    clades.add(frozenset(current_clade))

    return current_clade
