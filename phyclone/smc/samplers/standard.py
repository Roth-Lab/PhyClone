from itertools import repeat

import numpy as np

from phyclone.smc.samplers.base import AbstractSMCSampler
from phyclone.smc.swarm import ParticleSwarm


class SMCSampler(AbstractSMCSampler):
    """Standard SMC sampler with adaptive resampling."""

    __slots__ = ()

    def _init_swarm(self):
        self.swarm = ParticleSwarm()

        uniform_weight = -np.log(self.num_particles)

        for _ in range(self.num_particles):
            self.swarm.add_particle(uniform_weight, None)

    def _resample_swarm(self):
        if self.swarm.relative_ess <= self.resample_threshold:
            new_swarm = ParticleSwarm()

            log_uniform_weight = -np.log(self.num_particles)

            multiplicities = self._rng.multinomial(self.num_particles, self.swarm.weights)

            neg_inf = -np.inf

            particle_list = self.swarm.particles

            kept_particle_indices = np.where(multiplicities > 0)[0]

            for particle_idx in kept_particle_indices:
                particle = particle_list[particle_idx]
                assert particle.log_w != neg_inf
                multiplicity = multiplicities[particle_idx]
                new_swarm.add_particles_from_iterators(
                    repeat(log_uniform_weight, multiplicity), repeat(particle, multiplicity)
                )

            self.swarm = new_swarm

    def _update_swarm(self):
        new_swarm = ParticleSwarm()

        for parent_log_W, parent_particle in zip(self.swarm.log_weights, self.swarm.particles):
            particle = self._propose_particle(parent_particle)

            new_swarm.add_particle(parent_log_W + self._get_log_w(particle), particle)

        self.swarm = new_swarm
