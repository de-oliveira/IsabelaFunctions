"""
Useful for the visualization of maps.
"""

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np


def discrete_cmap(N, base_cmap = None):
    """ Create an N-bin discrete colormap from the specified input colormap.
    
    Parameters:
        N: integer
            Desired number of colors.
        
        base_map: string, optional
            Name of the base map, e.g., 'viridis'.
            
    Returns:
        A colormap object.
    """
    base = plt.cm.get_cmap(base_cmap)
    cmap_name = base.name + str(N)
        
    color_list = base(np.linspace(0, 1, N))
    discrete_map = colors.LinearSegmentedColormap.from_list(cmap_name, color_list, N)
   
    return discrete_map


def resample_by_mean(x, y, z, xplim, yplim, median = False):
    """ Resamples a grid by nan-mean. Example: a 30 x 30 datapoints grid can be reduced to a 10 x 10 grid.

    Parameters:
        x: list or array
            X original coordinates.

        y: list or array
            Y original coordinates.

        z: 2D array
            Data points.

        xplim: list or array
            New limits of x, such that [0, 1, 2] returns values at 0.5 and 1.5.

        yplim: list or array
            New limits of y, such that [0, 1, 2, 3] returns values at 0.5, 1.5, and 2.5.
        
        median: bool, optional
            If True, resamples by median instead of by mean. Default is False.
            
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
    
    if median is False:
        for i in range(nx):
            for j in range(ny):
                zp[j, i] = np.nanmean(z[np.logical_and(y<yplim[j+1], y>=yplim[j])][:, np.logical_and(x<xplim[i+1], x>=xplim[i])])
        for i in range(nx):
            xp[i] = (xplim[i]+xplim[i+1])/2
        for j in range(ny):
            yp[j] = (yplim[j]+yplim[j+1])/2

    else:
        for i in range(nx):
            for j in range(ny):
                zp[j, i] = np.nanmedian(z[np.logical_and(y<yplim[j+1], y>=yplim[j])][:, np.logical_and(x<xplim[i+1], x>=xplim[i])])
        for i in range(nx):
            xp[i] = (xplim[i]+xplim[i+1])/2
        for j in range(ny):
            yp[j] = (yplim[j]+yplim[j+1])/2
            
    return xp, yp, zp


def resample_1D(x, y, xplim, median = False):
    """ Resamples a 1D array by nan-mean, considering the coordinates of each point in the array.
        Example: an array with 30 datapoints can be reduced to 10 datapoints.

    Parameters:
        x: list or array
            X original coordinates.

        y: list or array
            Datapoints.

        xplim: list or array
            New limits of x, such that [0, 1, 2] returns values at 0.5 and 1.5.
 
        median: bool, optional
            If True, resamples by median instead of by mean. Default is False.
            
    Returns:
        xp: array
            Coordinates of resampled x.

        yp: array
            Values after resample.
    """
    nx = len(xplim)-1
   
    yp = np.empty(nx)
    xp = np.empty(nx)
    
    if median is False:
        for i in range(nx):
           yp[i] = np.nanmean(y[np.logical_and(x < xplim[i+1], x >= xplim[i])])
           xp[i] = (xplim[i] + xplim[i+1]) / 2

    else:
        for i in range(nx):
           yp[i] = np.nanmedian(y[np.logical_and(x < xplim[i+1], x >= xplim[i])])
           xp[i] = (xplim[i] + xplim[i+1]) / 2
            
    return xp, yp


def down_sample(x, n, median = False):
    """ Resamples a 1D array by nan-mean. Example: an array with 30 datapoints can be reduced to 10 datapoints.

    Parameters:
        x: array
            The array that will be resampled.

        n: integer
            The number of datapoints by which the array will be resampled.
 
        median: bool, optional
            If True, resamples by median instead of by mean. Default is False.
            
    Returns:
        xp: array
            The resampled values.
    """
    yp = np.r_[x, np.nan * np.zeros((-len(x) % n,))]
    
    if median is False:
        xp = np.nanmean(yp.reshape(-1, n), axis = -1)
    else:
        xp = np.nanmedian(yp.reshape(-1, n), axis = -1)
        
    return xp


def zoom_in(file, lonlim, latlim, binsize = 0.5):
    """ Zoom in a specific region in a lon x lat map.

    Parameters:

        file: N x M array
            Matrix of latitutde and longitude coordinates.

        binsize: float, optional
            The bin size of the array in degrees (bin size of N = bin size of M). Default is 0.5.

        lonlim: [a, b] array
            Longitude coordinates to be zoomed in.

        latlim: [c, d] array
            Latitude coordinates to be zoomed in.

    Returns:

        zoom_file: N' < N x M' < M array
            Matrix of latitutde and longitude coordinates of the zoomed region.
            
        coords: N x M array
            Matrix of booleans, where True correspond to the coordinates of the zoomed region.
    """
    longitude = np.linspace(0., 360., int(360./binsize+1.))
    latitude = np.linspace(-90., 90., int(180./binsize+1.))
    
    zoom_lon = np.logical_and(longitude >= lonlim[0], longitude <= lonlim[1])
    zoom_lat = np.logical_and(latitude >= latlim[0], latitude <= latlim[1])
    zoom_file = np.array(file[zoom_lat][:, zoom_lon])
    
    lats, lons = np.meshgrid(zoom_lon, zoom_lat)
    coords = np.logical_and(lats, lons)
    
    return zoom_file, coords


def crop(file, x, y): 
    cropped = file[y[0]:y[1]+1, x[0]:x[1]+1]
    
    return cropped


def shift_map_longitude(mapdata, lonshift, spline_order=1):
    """ Simple shift of the map by wrapping it around the edges
    
    Internally uses scipy's ndimage.shift with spline interpolation order as
    requested for interpolation

    Parameters
    ----------
    mapdata : 2D Numpy array
        A map with the second dimension the longutide stretched fully along the
        map
        
    lonshift : float
        A simple float representing the longitude shift of the array
        
    spline_order: int [1, 5]

    Returns
    -------
    A shifted map

    """
    from scipy.ndimage import shift
    
    # Constant
    degrees = 360.0
    
    # Check the map and compute the relative shift
    assert len(mapdata.shape) == 2, "Only for 2D maps"
    assert mapdata.shape[1] > 1, "Map has only one longitudinal coordinate"
    
    n = (mapdata.shape[1] - 1) 
    x = degrees * lonshift / n      # The number of pixels to shift
    
    # Use scipy for the rest
    mapdata_shift = shift(mapdata, [0, x], mode='wrap', order=spline_order)
    
    return mapdata_shift
