from functools import lru_cache

import numpy as np
from phyclone.smc.kernels.base import Kernel, ProposalDistribution, get_cached_new_tree_adder
from phyclone.smc.swarm import TreeHolder
from phyclone.smc.swarm.tree_shell_node_adder import TreeShellNodeAdder
from phyclone.tree import Tree
from phyclone.utils.math import cached_log_binomial_coefficient


class SemiAdaptedProposalDistribution(ProposalDistribution):
    """Semi adapted proposal density.

    Considers all possible choice of existing nodes and one option for a new node proposed at random. This
    should provide a computational advantage over the fully adapted proposal.
    """

    __slots__ = ("log_half", "parent_is_empty_tree", "_cached_log_old_num_roots",
                 "_computed_prob", "_max_samples", "_sample_idx", "_sample_arr", "_u_draws", "_u_idx")

    def __init__(
        self,
        data_point,
        kernel,
        parent_particle,
        outlier_modelling_active=False,
    ):
        super().__init__(data_point, kernel, parent_particle, outlier_modelling_active)

        self.log_half = kernel.log_half

        self.parent_is_empty_tree = False

        self._max_samples = 100

        self._sample_idx = self._max_samples

        self._u_draws = None

        self._u_idx = self._max_samples

        self._sample_arr = None

        self._computed_prob = dict()

        self._init_dist()


    def log_p(self, tree):
        """Get the log probability of proposing the tree."""

        if self.parent_is_empty_tree:
            log_p = self._get_log_p(tree)

        else:

            node = tree.labels[self.data_point.idx]
            assert node == tree.node_last_added_to

            # Existing node
            if node in self.parent_particle.tree_nodes or node == tree.outlier_node_name:
                log_p = self.log_half + self._get_log_p(tree)

            # New node
            else:
                if tree in self._computed_prob:
                    log_p = self._computed_prob[tree]
                else:
                    old_num_roots = len(self.parent_particle.tree_roots)

                    log_p = self.log_half

                    num_children = tree.num_children_on_node_that_matters

                    log_p -= self._cached_log_old_num_roots + cached_log_binomial_coefficient(old_num_roots, num_children)
                    self._computed_prob[tree] = log_p

        return log_p

    def _get_log_p(self, tree):
        """Get the log probability of the given tree. From stored dict, using TreeHolder intermediate."""
        return self._log_p[tree]

    def sample(self):
        """Sample a new tree from the proposal distribution."""
        if self.parent_is_empty_tree:
            tree = self._propose_existing_node()
        else:
            # u = self._rng.random()
            if self._u_idx == self._max_samples:
                self._u_draws = self._rng.random(size=self._max_samples, dtype=np.float32)
                self._u_idx = 0

            u = self._u_draws[self._u_idx]
            self._u_idx += 1

            if u < 0.5:
                tree = self._propose_existing_node()
            else:
                tree = self._propose_new_node()

        return tree

    def _init_dist(self):
        self._log_p = {}

        trees = list()

        if self.outlier_modelling_active:
            trees.append(self._get_outlier_tree())

        if self._empty_tree():
            self.parent_is_empty_tree = True
            if self.parent_particle is None:
                tree = Tree(self.data_point.grid_size)
            else:
                tree = self.parent_tree.copy()

            tree.create_root_node(children=[], data=[self.data_point])
            tree_particle = TreeHolder(tree, self.tree_dist, self.perm_dist)
            trees.append(tree_particle)
        else:
            self._tree_shell_node_adder = TreeShellNodeAdder(self.parent_tree, self.perm_dist)
            old_num_roots = len(self.parent_particle.tree_roots)
            self._cached_log_old_num_roots = np.log(old_num_roots + 1)

        trees.extend(self._get_existing_node_trees())

        self._set_log_p_dist(trees)

        self.parent_tree = None

    def _propose_existing_node(self):
        # q = self._q_dist

        # idx = self._rng.multinomial(1, q).argmax()
        #
        # tree = self._curr_trees[idx]

        # tree = self._rng.choice(trees, size=None, p=q)
        if self._sample_idx == self._max_samples:
            self._sample_arr = self._rng.multinomial(1, self._q_dist, size=self._max_samples).argmax(1)
            self._sample_idx = 0

        tree = self._curr_trees[self._sample_arr[self._sample_idx]]
        self._sample_idx += 1

        return tree

    def _propose_new_node(self):
        roots = self.parent_particle.tree_roots
        num_roots = len(roots)

        num_children = self._rng.integers(0, num_roots + 1)

        if num_children == 0:
            children = []
        elif num_children == num_roots:
            children = roots
        else:
            children = self._rng.choice(roots, num_children, replace=False)

        frozen_children = frozenset(children)

        tree_container = get_cached_new_tree_adder(
            self._tree_shell_node_adder,
            self.data_point,
            frozen_children,
            self.tree_dist,
        )

        return tree_container


@lru_cache(maxsize=1024)
def get_cached_new_tree(parent_particle, data_point, children, tree_dist, perm_dist):
    tree = parent_particle.tree

    tree.create_root_node(children=children, data=[data_point])

    tree_container = TreeHolder(tree, tree_dist, perm_dist)

    return tree_container


class SemiAdaptedKernel(Kernel):
    __slots__ = ("outlier_modelling_active", "log_half")

    def __init__(self, tree_dist, rng, outlier_modelling_active=False, perm_dist=None):
        super().__init__(tree_dist, rng, perm_dist=perm_dist)

        self.log_half = np.log(0.5)

        self.outlier_modelling_active = outlier_modelling_active

    def get_proposal_distribution(self, data_point, parent_particle):
        return _get_cached_semi_proposal_dist(
            data_point,
            self,
            parent_particle,
            self.outlier_modelling_active,
            self.tree_dist.prior.alpha,
        )


@lru_cache(maxsize=2048)
def _get_cached_semi_proposal_dist(data_point, kernel, parent_particle, outlier_modelling_active, alpha):
    ret = SemiAdaptedProposalDistribution(
        data_point,
        kernel,
        parent_particle,
        outlier_modelling_active=outlier_modelling_active,
    )
    return ret
