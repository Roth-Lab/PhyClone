from phyclone.smc.samplers import ConditionalSMCSampler
from phyclone.smc.swarm import ParticleSwarm
from phyclone.smc.utils import RootPermutationDistribution
from phyclone.utils.math import discrete_rvs


class ParticleGibbsTreeSampler(object):
    """Particle Gibbs sampler targeting sampling a full tree."""

    __slots__ = ("kernel", "num_particles", "resample_threshold", "_rng")

    def __init__(self, kernel, rng, num_particles=10, resample_threshold=0.5):
        self.kernel = kernel

        self.num_particles = num_particles

        self.resample_threshold = resample_threshold

        self._rng = rng

    def sample_tree(self, tree):
        """Sample a new tree"""
        swarm = self.sample_swarm(tree)

        return self._sample_tree_from_swarm(swarm)

    def sample_swarm(self, tree):
        """Sample a new SMC swarm"""

        data_sigma = RootPermutationDistribution.sample(tree, self._rng)

        sampler = ConditionalSMCSampler(
            tree,
            data_sigma,
            self.kernel,
            num_particles=self.num_particles,
            resample_threshold=self.resample_threshold,
        )

        return sampler.sample()

    def _sample_tree_from_swarm(self, swarm):
        """Given an SMC swarm sample a tree"""
        particle_idx = discrete_rvs(swarm.weights, self._rng)

        particle = swarm.particles[particle_idx]

        return particle.tree


class ParticleGibbsSubtreeSampler(ParticleGibbsTreeSampler):
    """Particle Gibbs sampler which resamples a sub-tree."""

    def sample_tree(self, tree):
        nodes = []

        outlier_node_name = tree.outlier_node_name
        for label in tree.labels.values():
            if label != outlier_node_name:
                nodes.append(label)

        subtree_root_child = self._rng.choice(nodes)

        subtree_root = tree.get_parent(subtree_root_child)

        parent = tree.get_parent(subtree_root)

        subtree = tree.get_subtree(subtree_root)

        tree.remove_subtree(subtree)

        for data_point in tree.outliers:
            tree.remove_data_point_from_outliers(data_point)

            subtree.add_data_point_to_outliers(data_point)

        swarm = self.sample_swarm(subtree)

        swarm = self._correct_weights(parent, swarm, tree)

        return self._sample_tree_from_swarm(swarm)

    # TODO: Check that this targets the correct distribution.
    # Specifically do we need a term for the random choice of node.
    def _correct_weights(self, parent, swarm, tree):
        """Correct weights so target is the distribution on the full tree"""
        new_swarm = ParticleSwarm()

        for p, w in zip(swarm.particles, swarm.unnormalized_log_weights):
            subtree = p.tree

            w -= p.log_p_one

            new_tree = tree.copy()

            new_tree.add_subtree(subtree, parent=parent)

            for data_point in subtree.outliers:
                new_tree.add_data_point_to_outliers(data_point)

            new_tree.update()

            p.tree = new_tree

            w += p.log_p_one

            new_swarm.add_particle(w, p)

        return new_swarm
