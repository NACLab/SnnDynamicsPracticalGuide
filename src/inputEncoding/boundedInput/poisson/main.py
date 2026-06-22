from ngclearn import Context, numpy as jnp, MethodProcess
from .poissonEncoder import PoissonEncoder
from ngclearn.utils.viz.raster import create_raster_plot

from matplotlib import pyplot as plt

output_dir = "exp/inputEncoding/boundedInput/poisson"

def basicVisualization():
    with Context("basic_viz"):
        neuron_count = 5
        encoder = PoissonEncoder("E1", n_units=neuron_count, time_steps_per_event=30)
        advance_process = MethodProcess("advance") >> encoder.emit
        advance_process.watch(encoder.spikeOutput)

    encoder.inputs.set(jnp.array([[0.2, 0.4, 0.6, 0.8, 1.0]]))
    _, output = advance_process.scan(inputs=jnp.array(advance_process.pack_rows(length=200)))

    create_raster_plot(output[0], plt.gca())
    plt.xlabel("Time step")
    plt.ylabel("Neuron Index")
    plt.title("Poisson Process Spike Timing")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/basic_viz.png")
    plt.close()


if __name__ == '__main__':
    basicVisualization()
