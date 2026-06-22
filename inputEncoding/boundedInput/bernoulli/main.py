from ngclearn import Context, numpy as jnp, MethodProcess
from bernoulliEncoder import BernoulliEncoder
from matplotlib import pyplot as plt
from ngclearn.utils.viz.raster import create_raster_plot
output_dir = "../../../exp/inputEncoding/boundedInput/bernoulli"
import os
os.makedirs(output_dir, exist_ok=True)

def basicVisualization():
    with Context("basic_viz"):
        neuron_count = 5
        encoder = BernoulliEncoder("E1", n_units=neuron_count)

        advance_process = MethodProcess("advance") >> encoder.emit
        advance_process.watch(encoder.spikeOutput)

    encoder.inputs.set(jnp.array([[0.2, 0.4, 0.6, 0.8, 1.0]]))
    _, output = advance_process.scan(inputs=jnp.array(advance_process.pack_rows(length=200)))

    create_raster_plot(output[0], ax=plt.gca())
    plt.xlabel("Time step")
    plt.ylabel("Neuron Index")
    plt.title("Bernoulli Encoding Spike Timing")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/basic_viz.png")


if __name__ == '__main__':
    basicVisualization()
