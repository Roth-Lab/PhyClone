from __future__ import annotations

import numpy as np
from phyclone.tree import Tree


class TreeHolder(object):
    __slots__ = (
        "_tree_dist",
        # "_log_p",
        "_hash_val",
        "_tree",
        "log_pdf",
        # "_log_p_one",
        "_perm_dist",
        "tree_nodes",
        "tree_roots",
        "labels",
        "node_last_added_to",
        "num_children_on_node_that_matters",
        "outlier_node_name",
        "multiplicity",
        # "_likelihood_parts_dict",
        "_partial_log_p",
        "_partial_log_p_one",
        "_num_nodes",
        "_alpha_prior",
        "_curr_alpha_val"
    )

    def __init__(self, tree, tree_dist, perm_dist):

        self._tree_dist = tree_dist

        self._perm_dist = perm_dist

        # self.log_p = 0

        self.log_pdf = 0

        # self.log_p_one = 0

        self._hash_val = 0

        self._alpha_prior = 0.0

        self._curr_alpha_val = 0.0

        self.tree = tree

    def __hash__(self):
        return self._hash_val

    def __eq__(self, other):
        return hash(self) == hash(other)

    def copy(self) -> TreeHolder:
        return TreeHolder(self.tree, self._tree_dist, self._perm_dist)
        # TODO: re-write this? building tree unnecessarily here

    @property
    def tree(self):
        return self._tree

    @property
    def alpha_prior(self):
        return self._alpha_prior


    @alpha_prior.getter
    def alpha_prior(self):
        curr_alpha = self._tree_dist.prior.alpha
        if curr_alpha != self._curr_alpha_val:
            self._curr_alpha_val = curr_alpha
            self._alpha_prior = self._compute_alpha_prior()
        return self._alpha_prior

    @property
    def log_p(self):
        return self.alpha_prior + self._partial_log_p

    def _compute_alpha_prior(self):
        log_alpha = self._tree_dist.prior.log_alpha
        alpha_prior = 0.0
        alpha_prior += self._num_nodes * log_alpha
        return alpha_prior

    @property
    def log_p_one(self):
        return self.alpha_prior + self._partial_log_p_one

    @tree.setter
    def tree(self, tree):

        self.outlier_node_name = tree.outlier_node_name

        if self._perm_dist is None:
            self.log_pdf = 0.0
        else:
            self.log_pdf = self._perm_dist.log_pdf(tree)

        self.multiplicity = tree.multiplicity

        self._partial_log_p, self._partial_log_p_one = self._tree_dist.compute_likelihood_parts_for_tree_holder(tree)

        self._num_nodes = tree.get_number_of_nodes()

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

        self._curr_alpha_val = self._tree_dist.prior.alpha
        self._alpha_prior = self._compute_alpha_prior()


    @tree.getter
    def tree(self) -> Tree:
        return Tree.from_dict(self._tree)
