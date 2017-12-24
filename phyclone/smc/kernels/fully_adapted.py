from __future__ import division

import itertools
import numpy as np

from phyclone.math_utils import log_normalize
from phyclone.smc.kernels.base import Kernel, ProposalDistribution


class FullyAdaptedProposalDistribution(ProposalDistribution):
    """ Fully adapted proposal density.

    Considers all possible proposals and weight according to log probability.
    """

    def __init__(self, data_point, kernel, parent_particle, use_outliers=False):
        self.data_point = data_point

        self.kernel = kernel

        self.parent_particle = parent_particle

        self.use_outliers = use_outliers

        self._init_dist()

    def log_p(self, state):
        return self.log_q[state.node_idx]

    def sample(self):
        q = np.exp(np.array(list(self.log_q.values())))

        assert abs(1 - sum(q)) < 1e-6

        q = q / sum(q)

        idx = np.random.multinomial(1, q).argmax()

        node_idx = list(self.log_q.keys())[idx]

        return self.states[node_idx]

    def _init_dist(self):
        self.states = {}

        self._propose_new_node()

        if self.use_outliers:
            self._propose_outlier_node()

        if self.parent_particle is not None:
            self._propose_existing_node()

        log_q = [x.log_p for x in self.states.values()]

        log_q = log_normalize(np.array(log_q))

        self.log_q = dict(zip(self.states.keys(), log_q))

    def _propose_existing_node(self):
        for node_idx in self.parent_particle.state.root_idxs:
            self.states[node_idx] = self.kernel.create_state(
                self.data_point,
                self.parent_particle,
                node_idx,
                self.parent_particle.state.root_idxs
            )

    def _propose_new_node(self):
        if self.parent_particle is None:
            self.states[0] = self.kernel.create_state(self.data_point, self.parent_particle, 0, set([0, ]))

        else:
            node_idx = max(list(self.parent_particle.state.root_idxs) + [-1, ]) + 1

            num_roots = len(self.parent_particle.state.root_idxs)

            for r in range(0, num_roots + 1):
                for child_idxs in itertools.combinations(self.parent_particle.state.root_idxs, r):
                    root_idxs = set(self.parent_particle.state.root_idxs - set(child_idxs))

                    root_idxs.add(node_idx)

                    self.states[node_idx] = self.kernel.create_state(
                        self.data_point, self.parent_particle, node_idx, root_idxs
                    )

                    node_idx += 1

    def _propose_outlier_node(self):
        if self.parent_particle is None:
            root_idxs = set()

        else:
            root_idxs = self.parent_particle.state.root_idxs

        self.states[-1] = self.kernel.create_state(
            self.data_point,
            self.parent_particle,
            -1,
            root_idxs
        )


class FullyAdaptedKernel(Kernel):

    def get_proposal_distribution(self, data_point, parent_particle):
        return FullyAdaptedProposalDistribution(data_point, self, parent_particle)
