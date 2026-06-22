from matplotlib.lines import Line2D
from ngclearn import Context, MethodProcess, numpy as jnp
from .phasorEncoder import PhasorEncoder
from matplotlib import pyplot as plt
from ngclearn.utils.viz.raster import create_raster_plot
from src.visualization import get_distinct_colors, plot_with_events

output_dir = "exp/inputEncoding/periodicEncoding/phasor"

def OmegaVisualization():
    with Context("basic_viz"):
        neuron_count = 5
        omegas = [15 * (1 + i) for i in range(neuron_count)]
        encoder = PhasorEncoder("E1", n_units=neuron_count, period=jnp.array(omegas))

        advance_process = MethodProcess("advance") >> encoder.advance
        advance_process.watch(encoder.currentAngle)
        advance_process.watch(encoder.spikeOutput)


    _, outputs = advance_process.scan(inputs=jnp.array(advance_process.pack_rows(length=200)))

    angles = outputs[0]
    spikes = outputs[1]

    for k in range(neuron_count):
        data = jnp.sin(angles[:, 0, k]) * 0.25 + k
        plt.plot(data)
    create_raster_plot(spikes, ax=plt)

    colors = get_distinct_colors(neuron_count)
    legend_elements = [Line2D([0], [0], color=colors[idx], lw=4, label=f"{omega}") for idx, omega in enumerate(omegas)]
    plt.legend(title=r'$\omega$', handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))


    plt.title("Phasor Spike Timing")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/omega_comparison.png")
    plt.close()

def IntensityVisualization():
    with Context("basic_viz"):
        neuron_count = 5
        omegas = [30 for i in range(neuron_count)]
        encoder = PhasorEncoder("E1", n_units=neuron_count, period=jnp.array(omegas))

        advance_process = MethodProcess("advance") >> encoder.advance
        advance_process.watch(encoder.currentAngle)
        advance_process.watch(encoder.spikeOutput)

    inputs = [0.25 * i for i in range(neuron_count)]
    encoder.inputs.set(jnp.array([inputs], dtype=jnp.float16))

    _, outputs = advance_process.scan(inputs=jnp.array(advance_process.pack_rows(length=200)))

    angles = outputs[0]
    spikes = outputs[1]

    for k in range(neuron_count):
        data = jnp.sin(angles[:, 0, k]) * 0.25 + k
        plt.plot(data)
    create_raster_plot(spikes, ax=plt)

    colors = get_distinct_colors(neuron_count)
    legend_elements = [Line2D([0], [0], color=colors[idx], lw=4, label=f"{omega}") for idx, omega in enumerate(omegas)]
    plt.legend(title=r'$\omega$', handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))


    plt.title("Velocity Modulated Spike Timing")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/velocity_timing.png")

if __name__ == '__main__':
    OmegaVisualization()
    IntensityVisualization()
