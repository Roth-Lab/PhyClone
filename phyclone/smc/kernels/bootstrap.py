import numpy as np

from phyclone.smc.kernels.base import Kernel, ProposalDistribution
from phyclone.utils.math_utils import log_binomial_coefficient
from functools import lru_cache


class BootstrapProposalDistribution(ProposalDistribution):
    """Bootstrap proposal distribution.

    A simple proposal from the prior distribution.
    """

    __slots__ = (
        "log_half",
        "_cached_log_old_num_roots",
        "_computed_prob",
        "_outlier_tree",
        "_half_val",
        "_log_outlier_prob",
        "_log_1_minus_outlier_prob",
        "_extant_node_trees",
        "parent_is_empty_tree",
        "_tree_root_set",
    )

    def __init__(
        self,
        data_point,
        kernel,
        parent_particle,
        outlier_modelling_active=False,
    ):
        super().__init__(data_point, kernel, parent_particle, outlier_modelling_active)
        self.parent_tree = None
        self._init_dist()

    def _init_dist(self):
        if self._empty_tree():
            self.parent_is_empty_tree = True
        else:
            self.parent_is_empty_tree = False

        self._half_val = (1 - self.outlier_proposal_prob) / 2
        self.log_half = np.log(self._half_val)

        if self.outlier_modelling_active:
            self._log_outlier_prob = np.log(self.outlier_proposal_prob)
            self._log_1_minus_outlier_prob = np.log1p(-self.outlier_proposal_prob)
            self._outlier_tree = self._get_outlier_tree()
        else:
            self._log_outlier_prob = -np.inf
            self._log_1_minus_outlier_prob = 0.0

        self._extant_node_trees = dict()
        self._cached_log_old_num_roots = np.log(self._num_roots + 1)
        self._tree_root_set = set(self._tree_roots)

    def log_p(self, tree):
        """Get the log probability of the tree."""
        # First particle
        if self.parent_particle is None:
            if tree.labels[self.data_point.idx] == tree.outlier_node_name:
                log_p = self._log_outlier_prob
            else:
                log_p = self._log_1_minus_outlier_prob

        # Particles t=2 ...
        else:
            node = tree.labels[self.data_point.idx]

            # Outlier
            if node == tree.outlier_node_name:
                log_p = self._log_outlier_prob
            # Node in tree
            elif node in self._tree_root_set:
                num_nodes = self._num_roots
                log_p = self.log_half - np.log(num_nodes)
            # New node
            else:
                old_num_roots = self._num_roots
                log_p = self.log_half
                num_children = tree.get_number_of_children(node)
                log_p -= self._cached_log_old_num_roots + log_binomial_coefficient(old_num_roots, num_children)

        return log_p

    def sample(self):
        """Sample a new tree from the proposal distribution."""
        u = self._rng.random()

        if self.parent_is_empty_tree:
            if u < self.outlier_proposal_prob:
                tree = self._propose_outlier()
            else:
                tree = self._propose_new_node()
        # Nodes in the tree
        else:
            if u < self.outlier_proposal_prob:
                tree = self._propose_outlier()
            elif self.outlier_proposal_prob < u < self._half_val:
                tree = self._propose_existing_node()
            else:
                tree = self._propose_new_node()

        return tree

    def _propose_existing_node(self):

        root_node = self._rng.choice(self._tree_roots)

        if root_node in self._extant_node_trees:
            tree_holder = self._extant_node_trees[root_node]
        else:
            tree_builder = self._tree_shell_node_adder.create_tree_holder_with_datapoint_added_to_node(
                root_node,
                self.data_point,
            )
            tree_holder = tree_builder.build()
            self._extant_node_trees[root_node] = tree_holder

        return tree_holder

    def _propose_new_node(self):
        num_roots = self._num_roots
        roots = self._tree_roots

        num_children = self._rng.integers(0, num_roots + 1)

        if num_children == 0:
            children = []
        elif num_children == num_roots:
            children = roots
        else:
            children = self._rng.choice(roots, num_children, replace=False)

        tree_holder_builder = self._tree_shell_node_adder.create_tree_holder_with_new_node(
            children,
            self.data_point,
        )

        return tree_holder_builder.build()

    def _propose_outlier(self):
        return self._outlier_tree


class BootstrapKernel(Kernel):
    __slots__ = "outlier_modelling_active"

    def __init__(self, tree_prior_dist, rng, outlier_modelling_active=False, perm_dist=None):
        super().__init__(tree_prior_dist, rng, perm_dist=perm_dist)

        self.outlier_modelling_active = outlier_modelling_active

    def get_proposal_distribution(self, data_point, parent_particle):
        return _get_cached_bootstrap_proposal_dist(
            data_point,
            self,
            parent_particle,
            self.outlier_modelling_active,
            self.tree_dist.prior.alpha,
        )


@lru_cache(maxsize=4096)
def _get_cached_bootstrap_proposal_dist(data_point, kernel, parent_particle, outlier_modelling_active, alpha):
    ret = BootstrapProposalDistribution(
        data_point,
        kernel,
        parent_particle,
        outlier_modelling_active=outlier_modelling_active,
    )
    return ret
