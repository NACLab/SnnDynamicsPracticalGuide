"""
Contains all the utility methods needed to create the visualizations in this
project.
"""
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


def _is_plt(ax):
    return ax == plt


def _extract_events(event_data):
    return event_data.nonzero()[0]


def _make_tick(ax, x, y, height, color):
    """Logic to draw a tick on a plot"""
    ax.vlines(x, y - height / 2, y + height / 2, color=color)


def _make_ticks(ax, points, height, colors=None, **kwargs):
    """Logic to draw ticks at all provided points"""
    for x, y, c in points:
        if colors is None:
            c = "black"
        elif isinstance(colors, (str, tuple)):
            c = colors
        else:
            c = colors[c]
        _make_tick(ax, x, y, height, c)


def _make_dots(ax, dots, colors=None, **kwargs):
    """Logic to draw dots at specified points"""
    x, y, c = zip(*dots)
    if colors is None:
        c = "black"
    elif isinstance(colors, (str, tuple)):
        c = colors
    else:
        c = [colors[i] for i in c]
    ax.scatter(x, y, color=c, **kwargs)


def _plot(plot_data, ax, **kwargs):
    """Helper function for plotting data returned from NGC-Learn"""
    xs = [x for x in range(plot_data.shape[0])]
    ys = [plot_data[k] for k in xs]
    ax.plot(xs, ys, **kwargs)


def _plot_events(plot_data, event_data, ax, style, **kwargs):
    """Helper function to plot a series of events on a plot"""
    events = _align_events(plot_data, event_data)
    match style:
        case "dots":
            _make_dots(ax, events, **kwargs)
        case "ticks":
            _make_ticks(ax, events, **kwargs)

def _align_events(plot_data, event_data):
    """Places events at the correct real value (x,y) of the data when the event
    happened"""
    eevents = [_extract_events(event_data[:, i]) for i in range(event_data.shape[1])]

    events = []
    for idx, ee in enumerate(eevents):
        for e in ee:
            events.append((e, plot_data[e, idx], idx))

    return events


def get_distinct_colors(n, cmap_name='tab10'):
    """
    Gets a list of distinct colors allowing for a dynamic number of differently
    colors plots. the cmap_name will need to be changed if the number of inputs
    exceeds the number of colors in the cmap.
    """
    cmap = plt.get_cmap(cmap_name)
    return [cmap(i) for i in range(n)]


def plot_with_events(plot_data, *event_data, ax=None, colors=None, event_colors=None):
    """Creates a plot based on the data, while also placing the events on the
    produced plot as dots."""
    num_points = plot_data.shape[0]
    num_lines = plot_data.shape[1]
    if ax is None:
        ax = plt

    if colors is None:
        colors = get_distinct_colors(num_lines, cmap_name='tab10')

    if event_colors is None:
        event_colors = get_distinct_colors(len(event_data), cmap_name='tab10')

    for i in range(num_lines):
        _plot(plot_data[:, i], ax, color=colors[i], zorder=1)

    for eData, c in zip(event_data, event_colors):
        _plot_events(plot_data, eData, ax, colors=c, style='dots', zorder=1, s=15)


    return colors


def legend_builder(*lines, ax=None, legend_title=None, loc="upper left"):
    if ax is None:
        ax = plt
    if len(lines) == 0:
        return

    custom_lines = []
    labels = []
    for label, color, style in lines:
        custom_lines.append(Line2D([0], [0], color=color, lw=2, ls=style or "-"))
        labels.append(label)


    ax.legend(custom_lines, labels, title=legend_title, loc=loc)

