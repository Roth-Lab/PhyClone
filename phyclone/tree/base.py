from __future__ import annotations

from collections import defaultdict

import numpy as np
import rustworkx as rx

from phyclone.tree.visitors import GraphToCladesVisitor
from phyclone.utils.math_utils import cached_log_factorial


class BaseTree(object):
    _OUTLIER_NODE_NAME = -1
    _ROOT_NODE_NAME = "root"

    __slots__ = (
        "grid_size",
        "_data",
        "_log_prior",
        "_graph",
        "_node_indices",
        "_node_indices_rev",
        "_root_node_name",
        "_outlier_node_name",
    )

    def __init__(self, grid_size: tuple[int, int]):
        self._log_prior = -np.log(grid_size[1])
        self._node_indices_rev = dict()
        self._node_indices = dict()
        self._graph = rx.PyDiGraph(multigraph=False)
        self._outlier_node_name = self.__class__._OUTLIER_NODE_NAME
        self.grid_size = grid_size
        self._root_node_name = self.__class__._ROOT_NODE_NAME
        self._data = defaultdict(list)

    def __hash__(self):
        return hash(self.get_hash_id_obj())

    def __eq__(self, other):
        self_key = self.get_hash_id_obj()
        other_key = other.get_hash_id_obj()
        return self_key == other_key

    def get_hash_id_obj(self):
        return self.get_clades(), frozenset([dp.idx for dp in self.outliers])

    @property
    def outliers(self):
        return list(self._data[self._outlier_node_name])

    def get_clades(self) -> frozenset:
        visitor = GraphToCladesVisitor(self._node_indices_rev, self._data, self._root_node_name)
        root_idx = self._node_indices[self._root_node_name]
        rx.dfs_search(self._graph, [root_idx], visitor)
        vis_clades = frozenset(visitor.clades)
        return vis_clades

    def to_dict(self):
        tree_dict = {
            "graph": self._graph.edge_list(),
            "node_idx": self._node_indices.copy(),
            "node_data": {k: v.copy() for k, v in self._data.items()},
            "grid_size": self.grid_size,
            "log_prior": self._log_prior,
        }
        return tree_dict

    @property
    def root_node_name(self):
        return self._root_node_name

    @property
    def outlier_node_name(self):
        return self._outlier_node_name

    @property
    def labels(self):
        return {dp.idx: k for k, l in self._data.items() for dp in l}

    @property
    def node_data(self):
        result = self._data.copy()

        if self._root_node_name in result:
            del result[self._root_node_name]

        return result

    def _compute_multiplicity(self):
        mult = sum(
            map(
                cached_log_factorial,
                map(self._graph.out_degree, self._graph.node_indices()),
            )
        )
        return mult
