import itertools
from functools import lru_cache

import numpy as np

from phyclone.smc.kernels.base import Kernel, ProposalDistribution, get_cached_new_tree_adder, \
    get_cached_new_tree_adder_datapoint
from phyclone.smc.swarm import TreeHolder
from phyclone.tree import Tree
from phyclone.utils.math import log_normalize, discrete_rvs
from phyclone.smc.swarm.tree_shell_node_adder import TreeShellNodeAdder


class FullyAdaptedProposalDistribution(ProposalDistribution):
    """Fully adapted proposal density.

    Considers all possible proposals and weight according to log probability.
    """

    # __slots__ = "_log_p"

    def __init__(
        self,
        data_point,
        kernel,
        parent_particle,
        outlier_modelling_active=False,
        parent_tree=None,
    ):
        super().__init__(data_point, kernel, parent_particle, outlier_modelling_active, parent_tree)

        self._init_dist()

    def log_p(self, tree):
        """Get the log probability of the tree."""
        # if isinstance(tree, Tree):
        #     tree_particle = TreeHolder(tree, self.tree_dist, self.perm_dist)
        # else:
        #     tree_particle = tree
        return self._log_p[tree]

    def sample(self):
        """Sample a new tree from the proposal distribution."""
        # p = np.exp(np.array(list(self._log_p.values())))
        #
        # idx = discrete_rvs(p, self._rng)
        #
        # tree = list(self._log_p.keys())[idx]
        q = self._q_dist

        idx = self._rng.multinomial(1, q).argmax()

        tree = self._curr_trees[idx]

        return tree

    def _init_dist(self):
        self._log_p = {}
        # if not self._empty_tree():
        #     self._tree_shell_node_adder = TreeShellNodeAdder(self.parent_tree, self.perm_dist)
        # trees = self._get_existing_node_trees()
        # trees.extend(self._get_new_node_trees())
        trees = list()

        if self._empty_tree():
            if self.parent_particle is None:
                tree = Tree(self.data_point.grid_size)
            else:
                tree = self.parent_tree.copy()

            tree.create_root_node(children=[], data=[self.data_point])
            tree_particle = TreeHolder(tree, self.tree_dist, self.perm_dist)
            trees.append(tree_particle)
        else:
            self._tree_shell_node_adder = TreeShellNodeAdder(self.parent_tree, self.perm_dist)
            trees.extend(self._get_new_node_trees())

        if self.outlier_modelling_active:
            trees.append(self._get_outlier_tree())
        # log_q = np.array([x.log_p for x in trees])
        #
        # log_q = log_normalize(log_q)
        #
        # self._log_p = dict(zip(trees, log_q))
        trees.extend(self._get_existing_node_trees())

        self._set_log_p_dist(trees)

        self.parent_tree = None

    # def _get_existing_node_trees(self):
    #     """Enumerate all trees obtained by adding the data point to an existing node."""
    #     trees = []
    #
    #     if self.parent_particle is None:
    #         return trees
    #
    #     nodes = self.parent_particle.tree_roots
    #
    #     for node in nodes:
    #         tree = self.parent_tree.copy()
    #         tree.add_data_point_to_node(self.data_point, node)
    #         tree_particle = TreeHolder(tree, self.tree_dist, self.perm_dist)
    #         trees.append(tree_particle)
    #
    #     return trees

    def _get_new_node_trees(self):
        """Enumerate all trees obtained by adding the data point to a new node."""
        trees = []

        if self.parent_particle is None:
            tree = Tree(self.data_point.grid_size)

            tree.create_root_node(children=[], data=[self.data_point])
            tree_particle = TreeHolder(tree, self.tree_dist, self.perm_dist)

            trees.append(tree_particle)

        else:
            num_roots = len(self.parent_particle.tree_roots)

            for r in range(0, num_roots + 1):
                for children in itertools.combinations(self.parent_particle.tree_roots, r):
                    # tree = self.parent_tree.copy()
                    #
                    # tree.create_root_node(children=children, data=[self.data_point])
                    # tree_particle = TreeHolder(tree, self.tree_dist, self.perm_dist)
                    #
                    # trees.append(tree_particle)
                    frozen_children = frozenset(children)

                    tree_container = get_cached_new_tree_adder(
                        self._tree_shell_node_adder,
                        self.data_point,
                        frozen_children,
                        self.tree_dist,
                    )
                    trees.append(tree_container)

        return trees

    # def _get_outlier_tree(self):
    #     """Get the tree obtained by adding data point as outlier"""
    #     if self.parent_particle is None:
    #         tree = Tree(self.data_point.grid_size)
    #
    #     else:
    #         tree = self.parent_tree.copy()
    #
    #     tree.add_data_point_to_outliers(self.data_point)
    #
    #     tree_particle = TreeHolder(tree, self.tree_dist, self.perm_dist)
    #
    #     return [tree_particle]


class FullyAdaptedKernel(Kernel):
    __slots__ = "outlier_modelling_active"

    def __init__(self, tree_prior_dist, rng, outlier_modelling_active=False, perm_dist=None):
        super().__init__(tree_prior_dist, rng, perm_dist=perm_dist)

        self.outlier_modelling_active = outlier_modelling_active

    def get_proposal_distribution(self, data_point, parent_particle):
        return _get_cached_full_proposal_dist(
            data_point,
            self,
            parent_particle,
            self.outlier_modelling_active,
            self.tree_dist.prior.alpha,
        )


@lru_cache(maxsize=1024)
def _get_cached_full_proposal_dist(data_point, kernel, parent_particle, outlier_modelling_active, alpha):
    ret = FullyAdaptedProposalDistribution(
        data_point,
        kernel,
        parent_particle,
        outlier_modelling_active=outlier_modelling_active,
        parent_tree=None,
    )
    return ret
