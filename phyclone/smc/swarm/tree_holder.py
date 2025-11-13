from __future__ import annotations

import numpy as np

from phyclone.tree import Tree


class TreeHolder(object):
    __slots__ = (
        "_tree_dist",
        "log_p",
        "_hash_val",
        "_tree",
        "log_pdf",
        "log_p_one",
        "_perm_dist",
        "tree_nodes",
        "tree_roots",
        "labels",
        "node_last_added_to",
        "num_children_on_node_that_matters",
        "outlier_node_name",
    )

    def __init__(self, tree, tree_dist, perm_dist):

        self._tree_dist = tree_dist

        self._perm_dist = perm_dist

        self.log_p = 0

        self.log_pdf = 0

        self.log_p_one = 0

        self._hash_val = 0

        self.tree = tree

    def __hash__(self):
        return self._hash_val

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __copy__(self):
        return self.copy()

    # def copy(self) -> TreeHolder:
    #     return TreeHolder(self.tree, self._tree_dist, self._perm_dist)

    def copy(self) -> TreeHolder:
        cls = self.__class__
        new = cls.__new__(cls)
        new._tree_dist = self._tree_dist
        new.log_p = self.log_p
        new._hash_val = self._hash_val
        new._tree = self._tree.copy()
        new.log_pdf = self.log_pdf
        new.log_p_one = self.log_p_one
        new._perm_dist = self._perm_dist
        new.tree_nodes = list(self.tree_nodes)
        new.tree_roots = self.tree_roots.copy()
        new.labels = self.labels.copy()
        new.node_last_added_to = self.node_last_added_to
        new.num_children_on_node_that_matters = self.num_children_on_node_that_matters
        new.outlier_node_name = self.outlier_node_name
        return new

    @property
    def tree(self):
        return self._tree

    @tree.setter
    def tree(self, tree):

        self.outlier_node_name = tree.outlier_node_name

        if self._perm_dist is None:
            self.log_pdf = 0.0
        else:
            self.log_pdf = self._perm_dist.log_pdf(tree)

        # self.log_p = self._tree_dist.log_p(tree)
        # self.log_p_one = self._tree_dist.log_p_one(tree)

        self.log_p, self.log_p_one = self._tree_dist.compute_log_p_and_log_p_one(tree)

        self.tree_roots = np.asarray(tree.roots)
        self.tree_nodes = tree.nodes
        self._hash_val = hash(tree)
        self._tree = tree.to_dict()
        self.labels = tree.labels
        self.node_last_added_to = tree.node_last_added_to
        if self.node_last_added_to != tree.outlier_node_name:
            self.num_children_on_node_that_matters = tree.get_number_of_children(self.node_last_added_to)
        else:
            self.num_children_on_node_that_matters = 0

    @tree.getter
    def tree(self) -> Tree:
        return Tree.from_dict(self._tree)
