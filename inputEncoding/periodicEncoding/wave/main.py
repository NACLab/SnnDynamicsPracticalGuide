from ngclearn import Context, numpy as jnp, MethodProcess
from waveEncoder import WaveEncoder
from matplotlib import pyplot as plt
from ngclearn.utils.viz.raster import create_raster_plot
import matplotlib as mpl
from ngcsimlib.global_state import stateManager as gsm

mpl.rcParams['font.size'] = 14

output_dir = "../../../exp/inputEncoding/periodicEncoding/wave"
import os
os.makedirs(output_dir, exist_ok=True)


def get_distinct_colors(n, cmap_name='tab10'):
    cmap = plt.get_cmap(cmap_name)
    return [cmap(i) for i in range(n)]

def make_plots(inputs, spikes, ax=None, ticks=False):

    if ax is None:
        ax = plt
        plt.figure(figsize=(12, 8))

    inp_count = inputs.shape[0]
    colors = get_distinct_colors(inp_count)

    for i in range(inp_count):
        inps = inputs[i, :]
        ax.plot(inps, label=f"neuron {i+1}", color=colors[i])

        neuron_spikes = (spikes[:, 0, i]).nonzero()[0]

        if ticks:
            for spike_time in neuron_spikes:
                ax.vlines(spike_time, inps[spike_time] - 0.1,
                           inps[spike_time] + 0.1,
                           color="black")
        else:
            ax.scatter(neuron_spikes,
                        [inps[spike_time] for spike_time in neuron_spikes],
                        color=colors[i])


def viz_spikes():
    length = 350

    wave1 = jnp.array([jnp.sin(x * 0.15) for x in range(length)])
    wave2 = jnp.array([jnp.sin(x * 0.05) * jnp.sin(x * 0.1) for x in range(length)])
    wave3 = jnp.array([jnp.sin(x * 0.15) + 1.5 * jnp.sin(x * 0.05) - 0.3 * jnp.sin(x * 0.2) for x in range(length)])

    wave4 = jnp.array([2 *jnp.sin(x * 0.01) + 0.15 * jnp.sin(x * 0.15) for x in range(length)])


    waves = jnp.vstack([wave1, wave2, wave3, wave4])

    neuron_count = waves.shape[0]
    with Context("waveEncoder_ctx") as ctx:
        encoder = WaveEncoder("E1", neuron_count, 5, 2, 0.6)

        process = MethodProcess("process") >> encoder.process
        process.watch(encoder.inputs)
        process.watch(encoder.spikeOut)
        process.watch(encoder.stored_power)
        process.watch(encoder.recharge)


    spikes = []

    for inp in range(length):
        encoder.inputs.set(waves[:, inp])
        gsm.state, output = process.run()
        spikes.append(output[1])

    spikes = jnp.vstack(spikes)

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

    needed_terms = int(jnp.max(waves) // encoder.spike_cost)

    for n in range(needed_terms):
        ax[0].axhline(encoder.spike_cost * (n+1), color="gray")

    create_raster_plot(spikes, ax=ax[1])
    ax[1].set_ylabel("Neuron Index")
    ax[1].set_title("Encoded Spikes")
    ax[1].set_yticks([i for i in range(neuron_count)])
    ax[1].set_yticklabels([str(i+1) for i in range(neuron_count)])


    make_plots(waves, spikes.reshape((spikes.shape[0], 1, spikes.shape[1])), ax[0])

    ax[0].set_ylabel("Amplitude")
    ax[0].set_title("Wave Input")
    ax[0].legend()

    xmin, xmax = ax[0].get_xlim()

    terms = ["No\nActivity", "Singlet", "Doublet", "Triplet"]
    gap = encoder.spike_cost / 2

    for idx, term in enumerate(terms[:needed_terms+1]):
        ax[0].text(xmin, gap + idx * encoder.spike_cost, term, va='center', ha='left', color='black',
             fontsize=10)
    fig.tight_layout()
    fig.savefig(output_dir + "/wave_plot.png")
    plt.close(fig)

if __name__ == '__main__':
    viz_spikes()

