import math

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from ngclearn import Context, MethodProcess
from jax import numpy as jnp
from eventEncoder import EventEncoder
from visualization import plot_with_events
from matplotlib.colors import ListedColormap, BoundaryNorm
from ngcsimlib.global_state import stateManager as gsm

output_dir = "../../../exp/inputEncoding/sparceInput/event"
import os
os.makedirs(output_dir, exist_ok=True)


def plot_events(spikeHigh, spikeLow, ax, titleGenerator):
    hSpikes = jnp.transpose(spikeHigh, [2, 0, 1])
    lSpikes = jnp.transpose(spikeLow, [2, 0, 1])
    neuron_count = hSpikes.shape[0]
    highEvents = []
    lowEvents = []
    for t in range(hSpikes.shape[0]):
        highEvents.append(hSpikes[t, :].nonzero()[0])
        lowEvents.append(lSpikes[t, :].nonzero()[0])

    highOffsets = [1.5 + 3*i for i in range(neuron_count)]
    lowOffsets = [0.5 + 3*i for i in range(neuron_count)]

    print(len(highOffsets), len(highEvents))
    ax.eventplot(highEvents, lineoffsets=highOffsets, linelengths=1, color="blue")
    ax.eventplot(lowEvents, lineoffsets=lowOffsets, linelengths=1, color="red")

    ax.yticks([1 + 3 * i for i in range(neuron_count)], [titleGenerator(i) for i in range(neuron_count)])
    legend_elements = [Line2D([0], [0], color="blue", lw=4, label="High"),
                       Line2D([0], [0], color="red", lw=4, label="Low")]

    ax.legend(title="Events", handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))

def show_events(ax, shape, pos_events, neg_events, a=1.0, overlay=True):
    events = jnp.where(pos_events, 1, jnp.where(neg_events, -1, 0))
    events = events.reshape(shape)

    cmap = ListedColormap(['blue', 'black', 'white'])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = BoundaryNorm(bounds, cmap.N)

    if overlay:
        alpha = jnp.where(events != 0, a, 0.)
    else:
        alpha = 1.0

    ax.imshow(events, cmap=cmap, norm=norm, alpha=alpha, interpolation='none')

def show_ball(ax, frame):
    yellow = jnp.array([0.8, 0.8, 0.2])
    binary_expanded = jnp.expand_dims(frame, axis=-1)
    binary_rgb = binary_expanded * yellow
    ax.imshow(binary_rgb)

def show_motion(ax, frame, pos_events, neg_events):
    show_ball(ax, frame)
    show_events(ax, frame.shape, pos_events, neg_events, 0.9)


def basicVis():
    neuron_count = 4
    with Context("eventEncoder_ctx") as ctx:
        encoder = EventEncoder("E1", n_units=neuron_count, thr_high=0.05, thr_low=0.05)
        advance = MethodProcess("adv") >> encoder.emit
        advance.watch(encoder.inputs)
        advance.watch(encoder.spikeHigh)
        advance.watch(encoder.spikeLow)

    changes = []
    changes.extend([(10 + 10 * i, 1, 0.1) for i in range(9)])
    changes.extend([(100 + 10 * i, 1, -0.1) for i in range(9)])

    changes.extend([(12 + 10 * i, 2, 0.25) for i in range(4)])
    changes.extend([(120 + 10 * i, 2, -0.25) for i in range(4)])

    changes.extend([(5 + 5 * i, 3, 0.2 * ((-1) ** i)) for i in range(200)])

    inputs = jnp.zeros((neuron_count, 200))
    for ts, idx, delta in changes:
        inputs = inputs.at[idx, ts].set(delta)

    for i in range(1, 200):
        inputs = inputs.at[:, i].set(inputs[:, i-1] + inputs[:, i])

    highSpikes = []
    lowSpikes = []
    inps = []
    for i in range(200):
        encoder.inputs.set(inputs[:, i, None].T)
        gsm.state, outputs = advance.run()

        inps.append(outputs[0])
        highSpikes.append(outputs[1])
        lowSpikes.append(outputs[2])

    inps = jnp.vstack(inps)
    highSpikes = jnp.vstack(highSpikes)
    lowSpikes = jnp.vstack(lowSpikes)

    plot_with_events(inps, highSpikes, lowSpikes, event_colors=['black', 'blue'])
    plt.tight_layout()
    plt.show()

def drawBall(plane, ball, r=3):
    y, x = ball
    lb_shift = math.floor(r / 2)
    ub_shift = math.ceil(r / 2)

    return plane.at[y-lb_shift:y+ub_shift, x-lb_shift:x+ub_shift].set(1)

def bouncingBall():
    ball = (3, 2) # y, x
    plane = jnp.zeros((10, 10))
    frames = [drawBall(plane, ball)]
    while ball[0] < 8:
        ball = (ball[0] + 1, ball[1])
        if ball[0] % 2 == 0:
            ball = (ball[0], ball[1] + 1)
        frames.append(drawBall(plane, ball))
    frames.append(drawBall(plane, ball))
    while ball[0] > 3:
        ball = (ball[0] - 1, ball[1])
        if ball[0] % 2 == 0:
            ball = (ball[0], ball[1] + 1)
        frames.append(drawBall(plane, ball))
    frames.append(drawBall(plane, ball))

    duration = len(frames)

    fig, ax = plt.subplots(3, duration, figsize=(duration, 3), gridspec_kw={'wspace': 0.05, 'hspace': 0.05})
    for i in range(duration):
        show_ball(ax[0][i], frames[i])
        ax[0][i].set_xticks([])
        ax[0][i].set_yticks([])


    with Context("eventEncoder_ctx") as ctx:
        encoder = EventEncoder("E1", n_units=100, thr_high=0.05,
                               thr_low=0.05)
        advance = MethodProcess("adv") >> encoder.emit
        advance.watch(encoder.spikeHigh)
        advance.watch(encoder.spikeLow)
        encoder.level.set(frames[0].reshape((1, 100)))

    highSpikes = []
    lowSpikes = []
    for i in range(duration):
        encoder.inputs.set(frames[i].reshape((1, 100)))
        gsm.state, outputs = advance.run()

        highSpikes.append(outputs[0])
        lowSpikes.append(outputs[1])

    highSpikes = jnp.vstack(highSpikes)
    lowSpikes = jnp.vstack(lowSpikes)

    for i in range(duration):
        show_events(ax[1][i], frames[1].shape, highSpikes[i], lowSpikes[i], overlay=False)
        ax[1][i].set_xticks([])
        ax[1][i].set_yticks([])

    for i in range(duration):
        show_motion(ax[2][i], frames[i], highSpikes[i], lowSpikes[i])
        ax[2][i].set_xticks([])
        ax[2][i].set_yticks([])
        ax[2][i].set_xlabel(f"Frame {i}")

    fig.suptitle("Simulated Event Sensor Results: Bouncing Ball", fontsize=14)
    ax[0][0].set_ylabel("Raw Frames", rotation=0, labelpad=45, fontsize=14)
    ax[1][0].set_ylabel("Raw Events", rotation=0, labelpad=43, fontsize=14)
    ax[2][0].set_ylabel("Overlaid", rotation=0, labelpad=32, fontsize=14)
    plt.subplots_adjust(left=0.1, right=0.99, top=0.90, bottom=0.10,
                        wspace=0.05, hspace=0.05)
    plt.savefig(output_dir + "/ball.png")
    plt.show()
    plt.close()

if __name__ == '__main__':
    bouncingBall()