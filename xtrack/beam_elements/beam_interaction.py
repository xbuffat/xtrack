import xobjects as xo

from ..particles import Particles

class BeamInteraction:

    def __init__(self, interaction_process):
        self.interaction_process = interaction_process

    def track(self, particles):

        assert isinstance(particles._buffer.context, xo.ContextCpu)
        assert particles._num_active_particles >= 0

        # Assumes active particles are contiguous
        products = self.interaction_process.interact(particles)

        # TODO: This should work also when no products are there
        #       Particles reorganization should still happen

        if products is None or products['x'].size == 0:
            particles.reorganize()
        else:
            new_particles = Particles(_context=particles._buffer.context,
                    p0c = particles.p0c[0], # TODO: Should we check that 
                                            #       they are all the same?
                    s = products['s'],
                    x = products['x'],
                    px = products['px'],
                    y = products['y'],
                    py = products['py'],
                    zeta = products['zeta'],
                    delta = products['delta'],
                    mass_ratio = products['mass_ratio'],
                    charge_ratio = products['charge_ratio'],
                    parent_particle_id = products['parent_particle_id'])

            particles.add_particles(new_particles)
