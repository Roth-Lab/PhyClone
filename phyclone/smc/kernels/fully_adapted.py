from functools import lru_cache
from itertools import combinations

from phyclone.smc.kernels.base import Kernel, ProposalDistribution
from phyclone.smc.swarm.tree_shell_node_adder import TreeShellNodeAdder


class FullyAdaptedProposalDistribution(ProposalDistribution):
    """Fully adapted proposal density.

    Considers all possible proposals and weight according to log probability.
    """

    __slots__ = "_sample_idx"

    def __init__(
        self,
        data_point,
        kernel,
        parent_particle,
        outlier_modelling_active=False,
    ):
        super().__init__(data_point, kernel, parent_particle, outlier_modelling_active)

        self._sample_idx = 0

        self._init_dist()

    def log_p(self, tree):
        """Get the log probability of the tree."""
        return self._log_p[tree]

    def sample(self):
        """Sample a new tree from the proposal distribution."""

        idx = self._rng.multinomial(1, self._q_dist).argmax()
        tree = self._curr_trees[idx]

        return tree

    def _init_dist(self):
        self._log_p = {}
        trees = []

        self._tree_shell_node_adder = TreeShellNodeAdder(self.parent_tree, self.tree_dist, self.perm_dist)
        trees.extend(self._get_new_node_trees())

        if self.outlier_modelling_active:
            trees.append(self._get_outlier_tree())

        trees.extend(self._get_existing_node_trees())

        self._set_log_p_dist(trees)

        self.parent_tree = None

    def _get_new_node_trees(self):
        """Enumerate all trees obtained by adding the data point to a new node."""
        trees = []

        num_roots = self._num_roots
        tree_roots = self._tree_roots

        for r in range(0, num_roots + 1):
            for children in combinations(tree_roots, r):

                tree_container = self._tree_shell_node_adder.create_tree_holder_with_new_node(
                    children,
                    self.data_point,
                )
                trees.append(tree_container.build())

        return trees


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


@lru_cache(maxsize=2048)
def _get_cached_full_proposal_dist(data_point, kernel, parent_particle, outlier_modelling_active, alpha):
    ret = FullyAdaptedProposalDistribution(
        data_point,
        kernel,
        parent_particle,
        outlier_modelling_active=outlier_modelling_active,
    )
    return ret
