from ngclearn.components import JaxComponent
from ngclearn import Compartment, compilable
from jax import numpy as jnp, random

class PoissonEncoder(JaxComponent):
    def __init__(self, name, n_units, time_steps_per_event, **kwargs):
        super().__init__(name, **kwargs)
        self.time_steps_per_event = time_steps_per_event # ~15.7 for 63.75 hertz and dt=1

        self.inputs = Compartment(jnp.zeros((1, n_units)), display_name="Intensity", units="%")

        self.spikeOutput = Compartment(jnp.zeros((1, n_units), dtype=jnp.bool), display_name="spikes")


    @compilable
    def emit(self):
        key, subkey = random.split(self.key.get(), 2)
        probabilities = self.inputs.get() / self.time_steps_per_event
        eps = random.uniform(subkey, self.inputs.get().shape, minval=0., maxval=1.,dtype=jnp.float32)

        self.spikeOutput.set(probabilities > eps)
        self.key.set(key)