import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def coin_toss_entropy(prob):
    if prob < 1e-6 or prob> .999999:
        return 0
    else:
        return -prob*np.log(prob) -(1-prob)*np.log(1-prob)

def efficiency_to_hex(e, cmap_name="viridis"):
    """
    Map efficiency e in [0,1] to a hex color using a Matplotlib colormap.
    """
    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(e)                 # (r, jj, b, a) in [0,1]
    return mcolors.to_hex(rgba)    # '#rrggbb'