import subprocess
import tempfile
import os

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec


def efficiency_to_hex(e, cmap_name="viridis"):
    """
    This method maps an efficiency e in [0,1] to a hex color using a
    Matplotlib colormap.

    Parameters
    ----------
    e: float
    cmap_name: str
    """
    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(e)  # (r, jj, b, a) in [0,1]
    return mcolors.to_hex(rgba)  # '#rrggbb'


def plot_dot_with_colorbar(
        dot_file,
        caption,
        cmap_name="viridis",
        vmin=0.0,
        vmax=1.0,
        figsize=(8, 4),
        colorbar_label="Efficiency",
        engine="neato"):
    """
    This method plots a dot file using graphviz. The plot includes a color bar
    on the right side

    Parameters
    ----------
    dot_file: str
        only dot_file and caption is changed in this study
    caption: str
        only dot_file and caption is changed in this study
    cmap_name: str
    vmin: float
    vmax: float
    figsize: tuple[float]
    colorbar_label: str
    engine: str

    Returns
    -------

    """
    with tempfile.TemporaryDirectory() as tmp:
        png_file = os.path.join(tmp, "graph.png")

        # Render DOT -> PNG
        subprocess.run(
            [engine, "-Tpng", dot_file, "-o", png_file],
            check=True
        )

        # Load PNG
        img = mpimg.imread(png_file)

        fig = plt.figure(figsize=figsize)
        gs = GridSpec(1, 2, width_ratios=[4, 0.3])

        ax_graph = fig.add_subplot(gs[0])
        ax_graph.imshow(img)
        ax_graph.axis("off")

        ax_cbar = fig.add_subplot(gs[1])
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(
            cm.ScalarMappable(norm=norm, cmap=cmap_name),
            cax=ax_cbar
        )
        cbar.set_label(colorbar_label)
        if caption is not None:
            fig.text(
                0.5, 0.01,
                caption,
                ha="center",
                va="bottom"
            )

        plt.tight_layout()
        plt.show()


def plot_parametric_curve(param_to_x_y):
    """
    This method plots points in the (x,y) plane. Each point has a parameter
    param (like time)

    Parameters
    ----------
    param_to_x_y: dict[float, tuple(float, float)]
        dictionary mapping a param to an (x,y) tuple.

    Returns
    -------
    None

    """
    # Sort dictionary by parameter p0
    sorted_items = sorted(param_to_x_y.items())

    p_vals = [item[0] for item in sorted_items]
    x_vals = [item[1][0] for item in sorted_items]
    y_vals = [item[1][1] for item in sorted_items]

    plt.figure()

    # Plot curve
    plt.plot(x_vals, y_vals, marker='o', label="Parametric Curve")

    # Plot line x = y
    min_val = min(min(x_vals), min(y_vals))
    max_val = max(max(x_vals), max(y_vals))
    plt.plot([min_val, max_val], [min_val, max_val], 'r-', label="x = y")

    # Label first and last points
    if len(x_vals) > 0:
        # First point
        plt.annotate(
            f"{p_vals[0]:.2f}",
            (x_vals[0], y_vals[0]),
            textcoords="offset points",
            xytext=(5, 5)
        )

        # Last point
        plt.annotate(
            f"{p_vals[-1]:.2f}",
            (x_vals[-1], y_vals[-1]),
            textcoords="offset points",
            xytext=(5, 5)
        )

    plt.xlabel("av_ent")
    plt.ylabel("av_cond_info")
    plt.gca().set_aspect('equal', adjustable='box')

    plt.legend()
    plt.grid(True)
    plt.show()


def plot_x_to_y(x_to_y,
                xlabel="beta_hat",
                ylabel="mag"):
    """
    Plots a dictionary mapping x -> y.

    Parameters
    ----------
    x_to_y: dict
    xlabel: str
    ylabel: str
    """

    # Sort by x so the curve is ordered
    x_vals = sorted(x_to_y.keys())
    y_vals = [x_to_y[x] for x in x_vals]

    plt.figure()
    plt.plot(x_vals, y_vals, marker='o')

    # Axis labels
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Vertical line at x = 1
    plt.axvline(x=1, color='red')

    plt.show()


if __name__ == "__main__":
    def main():
        param_to_x_y = {
            0.0: (0.0, 0.0),
            0.2: (0.1, 0.15),
            0.4: (0.25, 0.3),
            0.6: (0.5, 0.45),
            0.8: (0.7, 0.75),
            1.0: (1.0, 1.0),
        }

        plot_parametric_curve(param_to_x_y)


    main()
