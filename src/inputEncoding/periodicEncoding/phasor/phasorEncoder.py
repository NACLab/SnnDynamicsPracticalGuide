from ngclearn.components import JaxComponent
from ngclearn import Compartment, compilable
from jax import numpy as jnp

class PhasorEncoder(JaxComponent):
    """
    period units: time steps per event
    """
    def __init__(self, name, n_units, period, **kwargs):
        super().__init__(name, **kwargs)
        self.n_units = n_units
        self.rad_per_timestep = 2 * jnp.pi / period

        self.inputs = Compartment(jnp.ones((1, n_units), dtype=jnp.float16), display_name="inputs")
        self.currentAngle = Compartment(jnp.zeros((1, n_units), dtype=jnp.float32),
                                        display_name="Angle", units="rad")
        self.spikeOutput = Compartment(jnp.zeros((1, n_units), dtype=jnp.bool), display_name="Spike Output")

    @compilable
    def advance(self):
        currentAngle = self.currentAngle.get() + self.rad_per_timestep * self.inputs.get()
        spikeOutput = currentAngle > (2 * jnp.pi)
        currentAngle = jnp.where(spikeOutput, currentAngle - 2 * jnp.pi, currentAngle)
        self.currentAngle.set(currentAngle)
        self.spikeOutput.set(spikeOutput)


    @compilable
    def reset(self):
        self.currentAngle.set(jnp.zeros((1, self.n_units), dtype=jnp.float32))
        self.spikeOutput.set(jnp.zeros((1, self.n_units), dtype=jnp.bool))
