import numpy as np

from phyclone.smc.kernels.base import Kernel, ProposalDistribution
from phyclone.tree import Tree
from phyclone.utils.math import log_binomial_coefficient


class BootstrapProposalDistribution(ProposalDistribution):
    """Bootstrap proposal distribution.

    A simple proposal from the prior distribution.
    """

    def __init__(
        self,
        data_point,
        kernel,
        parent_particle,
        outlier_modelling_active=False,
    ):
        super().__init__(data_point, kernel, parent_particle, outlier_modelling_active)

    def log_p(self, tree):
        """Get the log probability of the tree."""
        # First particle
        if self.parent_particle is None:
            if tree.labels[self.data_point.idx] == tree.outlier_node_name:
                log_p = np.log(self.outlier_proposal_prob)

            else:
                log_p = np.log(1 - self.outlier_proposal_prob)
        # Particles t=2 ...
        else:
            node = tree.labels[self.data_point.idx]

            # Outlier
            if node == tree.outlier_node_name:
                log_p = np.log(self.outlier_proposal_prob)

            # Node in tree
            elif node in self.parent_tree.nodes:
                num_nodes = len(self.parent_particle.tree_roots)

                log_p = np.log((1 - self.outlier_proposal_prob) / 2) - np.log(num_nodes)

            # New node
            else:
                old_num_roots = len(self.parent_particle.tree_roots)

                log_p = np.log((1 - self.outlier_proposal_prob) / 2)

                if old_num_roots > 0:
                    if isinstance(tree, Tree):
                        num_children = tree.get_number_of_children(node)
                    else:
                        num_children = tree.num_children_on_node_that_matters

                    log_p -= np.log(old_num_roots + 1) + log_binomial_coefficient(old_num_roots, num_children)

        return log_p

    def sample(self):
        """Sample a new tree from the proposal distribution."""
        u = self._rng.random()

        # First particle
        if self.parent_particle is None:
            tree = Tree(self.data_point.grid_size)

            if u < (1 - self.outlier_proposal_prob):
                node = tree.create_root_node([])

                tree.add_data_point_to_node(self.data_point, node)

            else:
                tree.add_data_point_to_outliers(self.data_point)

        # Particles t=2 ...
        # Only outliers in tree
        elif len(self.parent_tree.nodes) == 0:
            if u < (1 - self.outlier_proposal_prob):
                tree = self._propose_new_node()

            else:
                tree = self._propose_outlier()

        # Nodes in the tree
        else:
            if u < (1 - self.outlier_proposal_prob) / 2:
                tree = self._propose_existing_node()

            elif u < (1 - self.outlier_proposal_prob):
                tree = self._propose_new_node()

            else:
                tree = self._propose_outlier()

        return tree

    def _propose_existing_node(self):
        nodes = self.parent_particle.tree_roots

        node = self._rng.choice(list(nodes))

        tree = self.parent_tree.copy()

        tree.add_data_point_to_node(self.data_point, node)

        return tree

    def _propose_new_node(self):
        num_roots = len(self.parent_particle.tree_roots)

        num_children = self._rng.integers(0, num_roots + 1)

        children = self._rng.choice(self.parent_particle.tree_roots, num_children, replace=False)

        tree = self.parent_tree.copy()

        tree.create_root_node(children=children, data=[self.data_point])

        return tree

    def _propose_outlier(self):
        tree = self.parent_tree.copy()

        tree.add_data_point_to_outliers(self.data_point)

        return tree


class BootstrapKernel(Kernel):
    __slots__ = "outlier_modelling_active"

    def __init__(self, tree_prior_dist, rng, outlier_modelling_active=False, perm_dist=None):
        super().__init__(tree_prior_dist, rng, perm_dist=perm_dist)

        self.outlier_modelling_active = outlier_modelling_active

    def get_proposal_distribution(self, data_point, parent_particle):
        return BootstrapProposalDistribution(
            data_point,
            self,
            parent_particle,
            outlier_modelling_active=self.outlier_modelling_active,
        )
