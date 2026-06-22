from ngclearn.components import JaxComponent
from ngclearn import Compartment, compilable
from jax import numpy as jnp


class LatencyEncoder(JaxComponent):
    def __init__(self, name, n_units, R, C, thr, **kwargs):
        super().__init__(name, **kwargs)

        self.n_units = n_units
        self.R = R
        self.C = C
        self.thr = thr

        self.inputs = Compartment(jnp.zeros((1, n_units), jnp.float32))
        self.integrator = Compartment(jnp.zeros((1, n_units), dtype=jnp.float32), display_name="Integrator")
        self.spikeHistory = Compartment(jnp.zeros((1, n_units), dtype=jnp.bool), display_name="History")
        self.spikeOutput = Compartment(jnp.zeros((1, n_units), dtype=jnp.bool), display_name="Spike Output")

    @compilable
    def step(self, dt):
        integrator = self.integrator.get() + (dt / (self.R * self.C)) * (self.inputs.get() * self.R - self.integrator.get())

        spikes = (integrator > self.thr) & (~self.spikeHistory.get())
        history = spikes | self.spikeHistory.get()

        self.integrator.set(integrator)
        self.spikeOutput.set(spikes)
        self.spikeHistory.set(history)


    @compilable
    def reset(self):
        self.integrator.set(jnp.zeros((1, self.n_units), dtype=jnp.bool))
        self.spikeHistory.set(jnp.zeros((1, self.n_units), dtype=jnp.bool))
