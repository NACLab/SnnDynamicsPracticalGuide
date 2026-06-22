import math
from collections.abc import Iterable

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy
from visualization import get_distinct_colors, legend_builder


def explode_check(xbs, ybs, dt, can_explode):
    for xb, yb in zip(xbs, ybs):
        test_val = (xb ** 2 + yb ** 2) / xb
        bound = (2 / dt)
        # print(f"Explode Check: {test_val} < {bound}")
        if not can_explode and test_val >= bound:
            raise RuntimeError(
                f"The attractor will cause this to be unstable and explode")


def dp(_p, b, omega):
    """Delta of the point p"""
    return (-b + (omega * 1j)) * _p


def previous(p, b, omega, dt):
    """Location of the previous point p"""
    return (1 / (1 - b * dt + (omega * 1j * dt))) * p

def plot_boundary(*points, b=1.0, omega=7.5, dt=0.015,
                  point_labels=None, output_dir=None, title=None, can_explode=True, spike_terminate=True, save_title=None,
                  cycles=3, current_inject=None):
    """

    :param points: The initial current (x-axis) for each point being tracked
    :param b: rate of attraction
    :param omega: frequency of oscillation
    :param dt: length of time step
    :param point_labels: labels for each point
    :param output_dir: directory to save plot to
    :param title: title of the plot
    :param can_explode: whether to stop if the neuron is unstable
    :param spike_terminate: whether to stop the neuron when it reaches the spike
    threshold
    :param cycles: number of oscillations to simulate
    :param current_inject: list of (point_idx, step_idx, strength)
    """

    inp_spikes = {}
    for i in range(len(points)):
        inp_spikes[i] = {}

    if current_inject is not None:
        for idx, step, strength in current_inject:
            inp_spikes[idx][step]=strength

    homogenous = True

    if not isinstance(b, Iterable):
        bs = [b for _ in range(len(points))]
    else:
        bs = b
        homogenous = False

    if not isinstance(omega, Iterable):
        omegas = [omega for _ in range(len(points))]
    else:
        omegas = omega
        homogenous = False

    explode_check(bs, omegas, dt, can_explode)

    angles = [numpy.arccos(
        (1 - _b * dt) / math.sqrt((1 - _b * dt) ** 2 + (_omega * dt) ** 2)) for _b, _omega in zip(bs, omegas)]

    points_list = points[:]
    tracked_points = [points_list]
    i = 0
    while len(tracked_points) < math.ceil(2 * math.pi / min(angles)) * cycles:
        new_points_list = []
        for idx, (p, _b, _omega) in enumerate(zip(points_list, bs, omegas)):
            if idx in inp_spikes.keys():
                if i in inp_spikes[idx].keys():
                    p = p.real + inp_spikes[idx][i] + (p.imag * 1j)

            if p.imag >= 1 and i > 5 and spike_terminate:
                if p.imag > 1:
                    p = p.real + 1j
                new_points_list.append(p)
                continue

            p = dp(p, _b, _omega) * dt + p
            if p.imag > 1:
                p = p.real + 1j
            new_points_list.append(p)

        i += 1
        points_list = new_points_list
        tracked_points.append(points_list)

    ## Plot bounds
    if homogenous:
        colors = get_distinct_colors(len(points) + 1)

        pb = 1j
        i = 0
        bound = [pb]

        while len(bound) < math.ceil(2 * math.pi / min(angles)):
            pb = previous(pb, b, omega, dt)
            i += 1
            if pb.imag > 1:
                pb = pb.real + 1j
            bound.append(pb)

        breals = numpy.array([i.real for i in bound])
        bimags = numpy.array([i.imag for i in bound])
        plt.plot(breals, bimags, linewidth=2, zorder=1, linestyle="dashed",
                 color=colors[-1])
    else:
        colors = get_distinct_colors(len(points))

        for idx, (b, o, a) in enumerate(zip(bs, omegas, angles)):
            pb = 1j
            i = 0
            bound = [pb]

            while len(bound) < math.ceil(2 * math.pi / a):
                pb = previous(pb, b, o, dt)
                i += 1
                if pb.imag > 1:
                    pb = pb.real + 1j
                bound.append(pb)

            # thr_r =
            breals = numpy.array([i.real for i in bound])
            bimags = numpy.array([i.imag for i in bound])
            plt.plot(breals, bimags, color=colors[idx], linewidth=2, zorder=1,
                     linestyle="dashed")

    # print("Total points =", len(tracked_points))
    # print("Total Time =", len(tracked_points) * dt)
    x_min = 0
    x_max = 0
    y_min = 0
    c = 0
    for p in range(len(points)):
        reals = [i[p].real for i in tracked_points]
        imags = [i[p].imag for i in tracked_points]
        x_min = min(x_min, min(reals))
        x_max = max(x_max, max(reals))
        y_min = min(y_min, min(imags))

        plt.plot(reals, imags, color=colors[c], zorder=2)
        plt.scatter(reals[0], imags[0], color=colors[c])
        plt.scatter(reals[-1], imags[-1], color=colors[c], marker="X")
        c += 1
    plt.axhline(y=1, linewidth=2, zorder=1, linestyle="dashed",
                color=colors[-1])

    plt.xlim(x_min*1.2, x_max*1.2)
    plt.ylim(y_min*1.2, 1.5)
    if title is not None:
        plt.title(title)

    if point_labels is not None:
        elements = [(lab, c, "-") for lab, c in zip(point_labels, colors)]
        elements.append(("Spike Boundary", colors[-1] if homogenous else "black", "--"))
        legend_builder(*elements)

    if output_dir is not None:
        if title is None and save_title is None:
            print("No title to save image under, showing instead")
            plt.show()
        else:
            plt.savefig(output_dir + f'/{save_title or title}.png')
    else:
        plt.show()