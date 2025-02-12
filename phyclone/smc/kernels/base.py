from functools import lru_cache

import numpy as np

from phyclone.data.base import DataPoint
from phyclone.smc.swarm import Particle, TreeHolder
from phyclone.smc.swarm.tree_shell_node_adder import TreeShellNodeAdder
from phyclone.tree import Tree
from phyclone.utils.math import log_normalize


class Kernel(object):
    """Abstract class representing an SMC kernel targeting the marginal FS-CRP distribution.

    Subclasses should implement the get_proposal_distribution method.
    """

    __slots__ = ("tree_dist", "perm_dist", "_rng")

    @property
    def rng(self):
        return self._rng

    def get_proposal_distribution(self, data_point, parent_particle):
        """Get proposal distribution given the current data point and parent particle."""
        raise NotImplementedError

    def __init__(self, tree_dist, rng, perm_dist=None):
        """
        Parameters
        ----------
        tree_dist: TreeJointDistribution
            Joint distribution of tree
        # outlier_proposal_prob: float
        #     Probability of proposing an outlier.
        perm_dist: PermutationDistribution
            The permutation distribution used in a particle Gibbs sampler to reorder data points. Set to None if single
            pass SMC is being performed.
        """
        self.tree_dist = tree_dist

        self.perm_dist = perm_dist

        self._rng = rng

    def create_particle(self, log_q, parent_particle, tree):
        """Create a new particle from a parent particle."""
        particle = Particle(0, parent_particle, tree, self.tree_dist, self.perm_dist)

        if self.perm_dist is None:
            if parent_particle is None:
                log_w = particle.log_p - log_q

            else:
                log_w = particle.log_p - parent_particle.log_p - log_q

        else:
            if parent_particle is None:
                log_w = particle.log_p + particle.log_pdf - log_q

            else:
                log_w = particle.log_p - parent_particle.log_p + particle.log_pdf - parent_particle.log_pdf - log_q

        particle.log_w = log_w
        return particle

    def propose_particle(self, data_point, parent_particle):
        """Propose a particle for t given a particle from t - 1 and a data point."""
        proposal_dist = self.get_proposal_distribution(data_point, parent_particle)

        tree = proposal_dist.sample()

        log_q = proposal_dist.log_p(tree)

        return self.create_particle(log_q, parent_particle, tree)

    def _get_log_p(self, tree):
        """Compute joint distribution."""
        return self.tree_dist.log_p(tree)


class ProposalDistribution(object):
    """Abstract class for proposal distribution."""

    __slots__ = (
        "data_point",
        "tree_dist",
        "perm_dist",
        "outlier_proposal_prob",
        "parent_particle",
        "_rng",
        "parent_tree",
        "outlier_modelling_active",
        "_q_dist",
        "_tree_shell_node_adder",
        "_log_p",
        "_curr_trees"
    )

    def __init__(
        self,
        data_point,
        kernel,
        parent_particle,
        outlier_modelling_active=False,
    ):
        self._q_dist = None
        self._tree_shell_node_adder = None
        self._log_p = {}
        self._curr_trees = None
        self.data_point = data_point

        self.tree_dist = kernel.tree_dist

        self.perm_dist = kernel.perm_dist

        self.outlier_modelling_active = outlier_modelling_active

        if outlier_modelling_active:
            self.outlier_proposal_prob = 0.1
        else:
            self.outlier_proposal_prob = 0.0

        self.parent_particle = parent_particle

        self._rng = kernel.rng

        self._set_parent_tree()

    def _empty_tree(self):
        """Tree has no nodes"""
        return (self.parent_particle is None) or (len(self.parent_particle.tree_roots) == 0)

    def _set_parent_tree(self):
        if self.parent_particle is not None:
            self.parent_tree = self.parent_particle.tree
        else:
            self.parent_tree = None

    def log_p(self, state):
        """Get the log probability of proposing a tree."""
        raise NotImplementedError

    def sample(self):
        """Sample a new tree from the proposal distribution."""
        raise NotImplementedError

    def _get_existing_node_trees(self):
        """Enumerate all trees obtained by adding the data point to an existing node."""

        if self.parent_particle is None:
            return []

        nodes = self.parent_particle.tree_roots

        # trees = [get_cached_new_tree_adder_datapoint(self._tree_shell_node_adder, self.data_point, node, self.tree_dist,) for node in nodes]
        trees = [
            self._build_existing_node_tree(node)
            for node in nodes]
        # tree_adder = self._tree_shell_node_adder
        # dp = self.data_point
        # tree_dist = self.tree_dist
        #
        # trees = [tree_adder.create_tree_holder_with_datapoint_added_to_node(node, dp, tree_dist).build() for node in nodes]

        return trees

    def _build_existing_node_tree(self, node_id):
        tree_holder_builder = self._tree_shell_node_adder.create_tree_holder_with_datapoint_added_to_node(node_id, self.data_point,)
        return tree_holder_builder.build()

    def _get_outlier_tree(self):
        """Get the tree obtained by adding data point as outlier"""
        if self.parent_particle is None:
            tree = Tree(self.data_point.grid_size)

        else:
            tree = self.parent_tree.copy()

        tree.add_data_point_to_outliers(self.data_point)

        tree_particle = TreeHolder(tree, self.tree_dist, self.perm_dist)

        return tree_particle

    def _set_log_p_dist(self, trees):
        log_q = np.array([x.log_p for x in trees])
        log_q = log_normalize(log_q)
        self._curr_trees = np.array(trees)
        self._set_q_dist(log_q)
        self._log_p = dict(zip(trees, log_q))

    def _set_q_dist(self, log_q):
        q = np.exp(log_q)
        q_sum = q.sum()
        assert abs(1 - q_sum) < 1e-6
        q /= q_sum
        self._q_dist = q


@lru_cache(maxsize=4096)
def get_cached_new_tree_adder(tree_shell_node_adder: TreeShellNodeAdder, data_point: DataPoint, children):
    tree_holder_builder = tree_shell_node_adder.create_tree_holder_with_new_node(children, data_point)
    tree_container = tree_holder_builder.build()

    return tree_container


@lru_cache(maxsize=2048)
def get_cached_new_tree_adder_datapoint(tree_shell_node_adder: TreeShellNodeAdder, data_point: DataPoint, node_id, tree_dist):
    tree_holder_builder = tree_shell_node_adder.create_tree_holder_with_datapoint_added_to_node(node_id, data_point)
    tree_container = tree_holder_builder.build()

    return tree_container
