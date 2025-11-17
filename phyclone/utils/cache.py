from phyclone.smc.kernels.base import (
    get_cached_dp_added_to_new_node_builder,
    get_cached_built_tree_holder,
    # get_cached_dp_added_to_node_builder,
    get_cached_dp_added_to_outliers_builder,
    get_cached_dp_added_to_empty_tree_builder,
    get_cached_dp_added_to_new_node_tree_holder
)
from phyclone.smc.kernels.fully_adapted import _get_cached_full_proposal_dist
from phyclone.smc.kernels.semi_adapted import _get_cached_semi_proposal_dist
from phyclone.tree.utils import compute_log_S, _convolve_two_children


def clear_proposal_dist_caches():
    # get_cached_new_tree_adder.cache_clear()
    # get_cached_dp_added_to_new_node_tree_holder.cache_clear()
    get_cached_built_tree_holder.cache_clear()
    _get_cached_semi_proposal_dist.cache_clear()
    _get_cached_full_proposal_dist.cache_clear()


def clear_all_caches():
    get_cached_dp_added_to_empty_tree_builder.cache_clear()
    get_cached_dp_added_to_outliers_builder.cache_clear()
    # get_cached_dp_added_to_new_node_tree_holder.cache_clear()
    # get_cached_dp_added_to_node_builder.cache_clear()
    get_cached_dp_added_to_new_node_builder.cache_clear()
    get_cached_built_tree_holder.cache_clear()
    _get_cached_semi_proposal_dist.cache_clear()
    _get_cached_full_proposal_dist.cache_clear()
    compute_log_S.cache_clear()
    _convolve_two_children.cache_clear()


def clear_convolution_caches():
    compute_log_S.cache_clear()
    _convolve_two_children.cache_clear()


def print_cache_info():
    print("\n***********************************************************")
    print(
        "get_cached_dp_added_to_new_node_tree_holder cache info: {}, hit ratio: {}".format(
            get_cached_dp_added_to_new_node_tree_holder.cache_info(),
            _cache_ratio(get_cached_dp_added_to_new_node_tree_holder.cache_info()),
        )
    )
    print(
        "get_cached_dp_added_to_empty_tree_builder cache info: {}, hit ratio: {}".format(
            get_cached_dp_added_to_empty_tree_builder.cache_info(),
            _cache_ratio(get_cached_dp_added_to_empty_tree_builder.cache_info()),
        )
    )
    print(
        "get_cached_dp_added_to_outliers_builder cache info: {}, hit ratio: {}".format(
            get_cached_dp_added_to_outliers_builder.cache_info(),
            _cache_ratio(get_cached_dp_added_to_outliers_builder.cache_info()),
        )
    )
    # print(
    #     "get_cached_dp_added_to_node_builder cache info: {}, hit ratio: {}".format(
    #         get_cached_dp_added_to_node_builder.cache_info(),
    #         _cache_ratio(get_cached_dp_added_to_node_builder.cache_info()),
    #     )
    # )
    print(
        "get_cached_built_tree_holder cache info: {}, hit ratio: {}".format(
            get_cached_built_tree_holder.cache_info(),
            _cache_ratio(get_cached_built_tree_holder.cache_info()),
        )
    )
    print(
        "get_cached_dp_added_to_new_node_builder cache info: {}, hit ratio: {}".format(
            get_cached_dp_added_to_new_node_builder.cache_info(),
            _cache_ratio(get_cached_dp_added_to_new_node_builder.cache_info()),
        )
    )
    print(
        "_get_cached_full_proposal_dist cache info: {}, hit ratio: {}".format(
            _get_cached_full_proposal_dist.cache_info(),
            _cache_ratio(_get_cached_full_proposal_dist.cache_info()),
        )
    )
    print(
        "_get_cached_semi_proposal_dist cache info: {}, hit ratio: {}".format(
            _get_cached_semi_proposal_dist.cache_info(),
            _cache_ratio(_get_cached_semi_proposal_dist.cache_info()),
        )
    )
    print(
        "compute_log_S cache info: {}, hit ratio: {}".format(
            compute_log_S.cache_info(), _cache_ratio(compute_log_S.cache_info())
        )
    )
    print(
        "_convolve_two_children cache info: {}, hit ratio: {}".format(
            _convolve_two_children.cache_info(),
            _cache_ratio(_convolve_two_children.cache_info()),
        )
    )
    print("***********************************************************")


def _cache_ratio(cache_obj):
    try:
        ratio = cache_obj.hits / (cache_obj.hits + cache_obj.misses)
    except ZeroDivisionError:
        ratio = 0.0
    return ratio
