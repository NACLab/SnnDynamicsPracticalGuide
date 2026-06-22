from matplotlib.lines import Line2D
from ngclearn import Context, numpy as jnp, MethodProcess
from latencyEncoder import LatencyEncoder
from matplotlib import pyplot as plt
from visualization import get_distinct_colors
from ngcsimlib.global_state import stateManager as gsm
import os

output_dir = "../../../exp/inputEncoding/unboundedInput/latency"
import os
os.makedirs(output_dir, exist_ok=True)


def basicVisualization():
    with Context("basic_viz"):
        neuron_count = 21
        alphas = [25, 50, 75, 100, 125, 150, 175, 200]

        encoders = [LatencyEncoder(f"E{alpha}", n_units=neuron_count, R=1, C=alpha, thr=15) for alpha in alphas]

        advance_process = MethodProcess("advance")
        for encoder in encoders:
            advance_process >> encoder.step
            advance_process.watch(encoder.spikeOutput)

    for encoder in encoders:
        encoder.inputs.set(jnp.array([[i * 10. for i in range(neuron_count)]], dtype=jnp.float32))

    _, outputs = advance_process.scan(inputs=jnp.array(advance_process.pack_rows(200, dt=0.1)))

    for alpha in encoders[0].inputs.get()[0]:
        plt.plot([i for i in range(200)], [alpha for _ in range(200)], color="#888888")

    plt.ylabel("Sensor Magnitude")




    spikes = outputs
    inputs = encoders[0].inputs.get()[0]

    colors = get_distinct_colors(neuron_count)
    for i in range(spikes[0].shape[0]):
        for k in range(spikes[0].shape[2]):
            for idx, spike in enumerate(spikes):
                if spike[i, 0, k]:
                    plt.vlines(i, inputs[k] - 10 / 2, inputs[k] + 10 / 2,
                           color=colors[idx])

    legend_elements = [Line2D([0], [0], color=colors[idx], lw=4, label=f"{alpha}") for idx, alpha in enumerate(alphas)]

    plt.legend(title="C", handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))

    plt.title("Capacitance on Spike Timing")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/constant.png")
    plt.close()


def dynamicVisualization():

    with Context("dynamic"):
        neuron_count = 10
        encoder = LatencyEncoder(f"E{50}", n_units=neuron_count, R=1, C=50, thr=15)

        advance_process = MethodProcess("advance")
        advance_process >> encoder.step
        advance_process.watch(encoder.spikeOutput, encoder.inputs)

    dist_deltas = jnp.array([[2 * i for i in range(1, 1 + neuron_count)]])
    distances = jnp.ones((1, neuron_count))


    outs = [[], []]
    for i in range(200):
        encoder.inputs.set(distances)
        gsm.state, outputs = advance_process.run(dt=0.1)

        distances += dist_deltas

        for idx, o in enumerate(outputs):
            outs[idx].append(o)

    outs = [jnp.vstack(o) for o in outs]



    tracked_dist = outs[1]

    for k in range(neuron_count):
        plt.plot(tracked_dist[:, k], label=f"{dist_deltas[0][k]}")

    spikes = outs[0]
    inputs = tracked_dist

    for i in range(spikes.shape[0]):
        for k in range(spikes.shape[1]):
            if spikes[i, k]:
                plt.vlines(i, inputs[i, k] - 20 / 2, inputs[i, k] + 20 / 2,
                           color='black')



    plt.legend(title="Speed", loc='upper left', bbox_to_anchor=(1, 1))

    plt.ylabel("Sensor Magnitude")
    plt.xlabel("Time Step")
    plt.title("Spike Timing with Dynamic Magnitudes")
    plt.ylim(0, 750)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/dynamic.png")
    plt.close()



if __name__ == '__main__':
    basicVisualization()
    # dynamicVisualization()
