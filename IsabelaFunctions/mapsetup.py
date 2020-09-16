import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np


def discrete_cmap(N, base_cmap=None):
    """ Create an N-bin discrete colormap from the specified input map.
    
    Parameters:
        N: integer
            Desired number of colors.
        
        base_map: string, optional
            Name of the base map, e.g., 'viridis'.
    """

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return colors.LinearSegmentedColormap.from_list(cmap_name, color_list, N)


def resample_by_mean(x, y, z, xplim, yplim):
    """ Resamples by nan-mean

    Parameters:

        x: list or array
            X coordinates.

        y: list or array
            Y coordinates.

        z: 2D array
            Data points.

        xplim: list or array
            Limits of x, such that [0, 1, 2] returns values at 0.5 and 1.5.

        yplim: list or array
            Limits of y, such that [0, 1, 2, 3] returns values at 0.5, 1.5, and 2.5.

    Returns:

        xp: array
            Coordinates of resampled x.

        yp: array
            Coordinates of resampled y.

        zp: array
            Values after nan-mean of the smaller areas.
    """
    nx = len(xplim)-1
    ny = len(yplim)-1
   
    zp = np.zeros(shape = (ny, nx))
    xp = np.zeros(shape = (nx))
    yp = np.zeros(shape = (ny))
    for i in range(nx):
        for j in range(ny):
            zp[j, i] = np.nanmean(z[np.logical_and(y<yplim[j+1], y>=yplim[j])][:, np.logical_and(x<xplim[i+1], x>=xplim[i])])
    for i in range(nx):
        xp[i] = (xplim[i]+xplim[i+1])/2
    for j in range(ny):
        yp[j] = (yplim[j]+yplim[j+1])/2

    return xp, yp, zp
