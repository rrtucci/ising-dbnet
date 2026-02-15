import subprocess
import tempfile
import os

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec


def plot_dot_with_colorbar(
    dot_file,
    caption,
    cmap_name="viridis",
    vmin=0.0,
    vmax=1.0,
    figsize=(8, 4),
    colorbar_label="Efficiency",
    engine="neato"
):
    with tempfile.TemporaryDirectory() as tmp:
        png_file = os.path.join(tmp, "graph.png")

        # Render DOT -> PNG
        subprocess.run(
            [engine , "-Tpng", dot_file, "-o", png_file],
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


import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt


def plot_parametric_curve(param_to_x_y):
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
