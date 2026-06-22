from ngclearn.components import JaxComponent
from ngclearn import Compartment, compilable
from jax import numpy as jnp


class EventEncoder(JaxComponent):
    def __init__(self, name, n_units, thr_high, thr_low, **kwargs):
        super().__init__(name, **kwargs)
        self.n_units = n_units
        self.thr_high = thr_high
        self.thr_low = thr_low

        self.inputs = Compartment(jnp.zeros((1, n_units)))
        self.level = Compartment(jnp.zeros((1, n_units)), display_name="level")
        self.spikeHigh = Compartment(jnp.zeros((1, n_units), dtype=jnp.bool), display_name="Spike Out: High")
        self.spikeLow = Compartment(jnp.zeros((1, n_units), dtype=jnp.bool), display_name="Spike Out: Low")


    @compilable
    def emit(self,):
        deltas = self.inputs.get() - self.level.get()
        self.level.set(self.inputs.get())
        self.spikeHigh.set((deltas >= self.thr_high) & (deltas > 0))
        self.spikeLow.set((jnp.abs(deltas) >= self.thr_low) & (deltas < 0))

