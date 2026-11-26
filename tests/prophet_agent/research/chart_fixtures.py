"""Synthetic chart image with a known series, rendered by matplotlib, and the tick proposal derived from its own axes.

The tick positions come from the axes transform, so they are exact pixel positions. Tests use them as the offline tick
proposal; the live test lets the vision model propose the ticks instead.
"""
import io

import numpy as np

COLOUR = (31, 119, 180)          # matplotlib's default C0
WIDTH, HEIGHT, DPI = 800, 500, 100


def known_series():
    x = np.linspace(0, 10, 60)
    y = 40 + 25 * np.sin(x * 0.7) + 3 * x
    return x, y


def render(x, y, *, ylog=False):
    """PNG bytes and the proposal (ticks as pixel positions with labels, colour, axis titles)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator

    fig = plt.figure(figsize=(WIDTH / DPI, HEIGHT / DPI), dpi=DPI)
    ax = fig.add_axes([0.12, 0.14, 0.84, 0.78])
    ax.plot(x, y, color=np.array(COLOUR) / 255.0, linewidth=2.0)
    ax.set_xlim(0, 10)
    if ylog:
        ax.set_yscale("log")
        ax.set_ylim(1, 1000)
    else:
        ax.set_ylim(0, 120)
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    yticks = [0, 20, 40, 60, 80, 100, 120] if not ylog else [1, 10, 100, 1000]
    ax.yaxis.set_major_locator(FixedLocator(yticks))
    ax.yaxis.set_minor_locator(FixedLocator([]))
    ax.grid(True, color="#dddddd", linewidth=0.8)
    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel("Units", fontsize=11)
    fig.canvas.draw()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI)

    def pixel_x(v):
        return float(ax.transData.transform((v, 1.0))[0])

    def pixel_y(v):
        return HEIGHT - float(ax.transData.transform((1.0, v))[1])

    proposal = {
        "chart_type": "line",
        "x_axis": {"title": "Time", "unit": "", "scale": "linear",
                   "ticks": [{"pixel": pixel_x(v), "label": f"{v:g}"} for v in [0, 2, 4, 6, 8, 10]]},
        "y_axis": {"title": "Units", "unit": "", "scale": "log" if ylog else "linear",
                   "ticks": [{"pixel": pixel_y(v), "label": f"{v:g}"} for v in yticks]},
        "series": [{"name": "Known", "colour": list(COLOUR)}],
    }
    plt.close(fig)
    return buf.getvalue(), proposal


def truth_range(ylog=False):
    return (1.0, 1000.0) if ylog else (0.0, 120.0)
