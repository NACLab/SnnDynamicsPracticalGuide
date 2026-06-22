from ngclearn.components import JaxComponent
from ngclearn import Compartment, compilable
from jax import numpy as jnp


class WaveEncoder(JaxComponent):
    def __init__(self, name, n_units, inter_event_refractory_period,
                 inter_spike_refractory_period, spike_cost, **kwargs):
        super().__init__(name, **kwargs)

        self.n_units = n_units
        self.spike_cost = spike_cost
        self.inter_event_refractory_period = inter_event_refractory_period
        self.inter_spike_refractory_period = inter_spike_refractory_period

        self.inputs = Compartment(jnp.zeros((1, n_units)), display_name="Input Wave")

        self.recharge = Compartment(jnp.zeros((1, n_units)), display_name="Recharge")
        self.stored_power = Compartment(jnp.zeros((1, n_units)), display_name="Stored Power")
        self.last_input = Compartment(jnp.zeros((1, n_units)), display_name="Last Input")
        self.increasing = Compartment(jnp.zeros((1, n_units), dtype=jnp.bool), display_name="Increasing")
        self.spikeOut = Compartment(jnp.zeros((1, n_units), dtype=jnp.bool), display_name="Spike Out")

    @compilable
    def process(self):
        recharge_mask = self.recharge.get() > 0
        recharge = jnp.where(recharge_mask, self.recharge.get() - 1, 0)

        should_spike = jnp.where((self.last_input.get() > self.inputs.get()) & self.increasing.get(), 1, 0)
        self.increasing.set(self.inputs.get() > self.last_input.get())

        stored_power = self.stored_power.get() + (should_spike * self.inputs.get())
        spikes = (stored_power >= self.spike_cost.get()) & ~recharge_mask

        recharge = jnp.where(spikes, self.inter_spike_refractory_period.get(), recharge)

        stored_power = stored_power - (self.spike_cost.get() * spikes)

        self.recharge.set(jnp.where(spikes & (stored_power < self.spike_cost.get()), self.inter_event_refractory_period.get(), recharge))
        self.stored_power.set(jnp.where(stored_power < self.spike_cost.get(), 0, stored_power))
        self.last_input.set(self.inputs.get())
        self.spikeOut.set(spikes)
