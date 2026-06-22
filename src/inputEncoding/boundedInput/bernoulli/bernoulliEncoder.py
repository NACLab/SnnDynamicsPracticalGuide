from ngclearn.components import JaxComponent
from ngclearn import Compartment, compilable
from jax import numpy as jnp, random


class BernoulliEncoder(JaxComponent):
    def __init__(self, name, n_units, **kwargs):
        super().__init__(name, **kwargs)

        self.inputs = Compartment(jnp.zeros((1, n_units)), display_name="Intensity", units="%")
        self.spikeOutput = Compartment(jnp.zeros((1, n_units), dtype=jnp.bool), display_name="spikes")


    @compilable
    def emit(self):
        key, subkey = random.split(self.key.get(), 2)
        eps = random.uniform(subkey, self.inputs.get().shape, minval=0., maxval=1.,dtype=jnp.float32)
        self.spikeOutput.set(self.inputs > eps)
        self.key.set(key)