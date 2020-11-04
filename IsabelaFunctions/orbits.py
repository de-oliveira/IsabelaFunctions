import numpy as np
import scipy as sp
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from copy import copy
import datetime
import os
import shutil


def plot_altitude_model(new_alt, alt, lon, lat, br, anomaly, brange = None, lim = [20, 85, -15, 15], dawn = True):
    """ Plots the crustal magnetic field model (Langlais et al 2019) and overplots the measurements for a specific altitude.

   Parameters:
        new_alt: int
            The altitude at which the data will be plotted.

        alt: 1D array
            An array containing the altitude data.
            
        lon: 1D array
            An array containing the longitude data.
            
        lat: 1D array
            An array containing the latitude data.
            
        br: 1D array
            An array containing the magnetic field data.
        
        anomaly: string
            The anomaly index, e. g., A1, A2, A6, etc. This string is used to find the directory where the model matrices are located.
        
        brange: int, optional
            The maximum magnetic field values used for scaling the colorbar. Default is None, so it computes based on the data.
        
        lim: 4-elements array, optional
            An array cointaining the limits for latitude and longitude data, in which: [lon_min, lon_max, lat_min, lat_max].
            Default is the anomalous region [20, 85, -15, 15].
        
        dawn: bool, optional
            If True, assumes dawn. If False, assumes dusk. Default is true.
                
    Returns:
        None.
    """
    one_alt = np.around(alt) == new_alt
    
    data = sp.io.readsav('/home/oliveira/ccati_mexuser/LANGLAIS_Matrices/' + anomaly + '/LANGLAIS_BR_ALT_' + str(new_alt) + '_RES_01.bin')
    brmodel = data['zbins']
    
    f = 12

    my_cmap = copy(plt.cm.RdBu_r)
    my_cmap.set_over('r', 1.0)
    my_cmap.set_under('b', 1.0)
    
    if brange is None:
        bmax = np.max(br)
    else:
        bmax = brange
    bmin = -bmax
    
    if dawn == True:
        title = 'Dawn'
    else:
        title = 'Dusk'
    
    fig, axes = plt.subplots(1, 1, sharex = True, sharey = True, figsize = [6, 6])
    plt.subplots_adjust(hspace = 0, wspace = 0)
    
    plt.subplot(111)
    plt.scatter(lon[one_alt], lat[one_alt], c = br[one_alt], cmap = my_cmap, norm = colors.Normalize(vmin = bmin, vmax = bmax), \
                edgecolors = 'black')
    im = plt.imshow(brmodel, extent = lim, origin = 'lower', cmap = my_cmap, norm = colors.Normalize(vmin = bmin, vmax = bmax))
    plt.xlim(lim[0], lim[1])
    plt.ylim(lim[2], lim[3])
    plt.ylabel('Latitude ($\degree$)')
    plt.xlabel('Longitude ($\degree$)')
    plt.title(title + '-side, Altitude = ' + str(new_alt) + ' km')
    
    plt.subplots_adjust(right = 0.8)
    cbar_ax = fig.add_axes([0.82, 0.35, 0.03, 0.3])
    cbar = fig.colorbar(im, cax = cbar_ax, extend = 'both')
    cbar.set_label('$B_r$ (nT)', fontsize = f)
    cbar.ax.tick_params(labelsize = f)
    
    plt.savefig('/home/oliveira/PythonScripts/Orbits/Altitudes/Orbit_' + anomaly + '_alt_' + str(new_alt) + '_' + title + '.pdf', bbox_inches = 'tight')


def plot_time_range(start, delta, date, alt, lon, lat, br, lt, sza, vaz, vpol, brange = None, lim = [20, 85, -15, 15], anomaly = 'A6', dawn = True, save = True):
    """ Plots magnetic field and altitude data for a specific time range.

   Parameters:
        start: 6-elements array
            The time start at which the data will be plotted, in which: [YYYY, MM, DD, hh, mm, ss].
        
        delta: 6-elements array
            The time range to be computed from the time start, in which: [YYYY, MM, DD, hh, mm, ss].
        
        date: 1D array
            An array containing the date data.
        
        alt: 1D array
            An array containing the altitude data.
            
        lon: 1D array
            An array containing the longitude data.
            
        lat: 1D array
            An array containing the latitude data.
            
        br: 1D array
            An array containing the magnetic field data.
        
        brange: int, optional
            The maximum magnetic field values used for scaling the colorbar. Default is None, so it computes based on the data.
        
        lim: 4-elements array, optional
            An array cointaining the limits for latitude and longitude data, in which: [lon_min, lon_max, lat_min, lat_max].
            Default is the anomalous region [20, 85, -15, 15].
         
        dawn: bool, optional
            If True, assumes dawn. If False, assumes dusk. Default is true.
        
        save: bool, optional
            If True, saves the plot as a .pdf file. Default is True.       
            
    Returns:
        The number of data points.
        The list of selected longitudes.
        The list of selected latitudes.
        The list of selected altitudes.
        The list of selected magnetic field data.
        The string of the start date.
        The string of the end date.
    """
    time1 = datetime.datetime(start[0], start[1], start[2], start[3], start[4], start[5])
    time2 = datetime.datetime(start[0]+delta[0], start[1]+delta[1], start[2]+delta[2], start[3]+delta[3], start[4]+delta[4], \
                              start[5]+delta[5])
    
    date_range = np.logical_and(date >= time1, date < time2)
    
    f = 12

    my_cmap = copy(plt.cm.RdBu_r)
    my_cmap.set_over('r', 1.0)
    my_cmap.set_under('b', 1.0)
    my_cmap2 = copy(plt.cm.viridis_r)
    
    if brange is None:
        bmax = np.max(br)
    else:
        bmax = brange
    bmin = -bmax
    
    if dawn == True:
        title = 'Dawn'
    else:
        title = 'Dusk'
    
    fig, axes = plt.subplots(2, 1, sharex = True, sharey = True, figsize = [6, 6])
    plt.subplots_adjust(hspace = 0, wspace = 0)
    
    ax = plt.subplot(211)
    im1 = plt.scatter(lon[date_range], lat[date_range], c = br[date_range], cmap = my_cmap, norm = colors.Normalize(vmin = bmin, vmax = bmax))
    plt.xlim(lim[0], lim[1])
    plt.ylim(lim[2], lim[3])
    plt.xticks(visible = False)
    plt.ylabel('Latitude ($\degree$)')
    plt.title(title + '-side, ' + str(time1) + ' -- ' + str(time2))
    
    plt.subplot(212, sharex = ax, sharey = ax)
    im2 = plt.scatter(lon[date_range], lat[date_range], c = alt[date_range], cmap = my_cmap2, norm = colors.Normalize())
    plt.ylabel('Latitude ($\degree$)')
    plt.xlabel('Longitude ($\degree$)')
    
    plt.subplots_adjust(right = 0.8)
    cbar_ax = fig.add_axes([0.82, 0.52, 0.03, 0.35])
    cbar = fig.colorbar(im1, cax = cbar_ax, extend = 'both')
    cbar.set_label('$B_r$ (nT)', fontsize = f)
    cbar.ax.tick_params(labelsize = f)
    
    plt.subplots_adjust(right = 0.8)
    cbar_ax = fig.add_axes([0.82, 0.14, 0.03, 0.35])
    cbar = fig.colorbar(im2, cax = cbar_ax)
    cbar.set_label('Altitude (km)', fontsize = f)
    cbar.ax.tick_params(labelsize = f)
    
    if save is True:
        plt.savefig('/home/oliveira/PythonScripts/Orbits/Single Orbits/'+ anomaly +'/Orbit_' + anomaly + '_' + str(time1) + ' -- ' + str(time2) + '_' + title + '.pdf', \
                bbox_inches = 'tight')
    
    return len(br[date_range]), lon[date_range], lat[date_range], alt[date_range], br[date_range], lt[date_range], sza[date_range], vaz[date_range], vpol[date_range], str(time1), str(time2)


def plot_single_data(alt, lon, lat, br, lt, sza, n_title, anomaly, vaz = None, vpol = None, brange = None, sigma = 0.1, lim = [20, 85, -15, 15], dawn = True, time1 = '0', time2 = '1', length = None):
    """ Plots a single measurement on a model magnetic field contour map. For many single measurements, it creates an animation. 

   Parameters:
        start: 6-elements array
            The time start at which the data will be plotted, in which: [YYYY, MM, DD, hh, mm, ss].
        
        delta: 6-elements array
            The time range to be computed from the time start, in which: [YYYY, MM, DD, hh, mm, ss].
        
        date: 1D array
            An array containing the date data.
        
        alt: 1D array
            An array containing the altitude data.
            
        lon: 1D array
            An array containing the longitude data.
            
        lat: 1D array
            An array containing the latitude data.
            
        br: 1D array
            An array containing the magnetic field data.
            
        n_title: string
            A string used in the title and in the new directory.
            
        anomaly: string
            The anomaly index, e. g., A1, A2, A6, etc. This string is used to find the directory where the model matrices are located.
        
        brange: integer, optional
            The maximum magnetic field values used for scaling the colorbar. Default is None, so it computes based on the data.
        
        sigma: float, optional
            The thickness of the magnetic field contour, in percent. Default is 0.1 (10% of the value).
            Example: if the measurement is 10 nT, the contour includes the values (10 - 0.1*10) <= B <= (10 + 0.1*10), which is 9 <= B <= 11. 
        
        lim: 4-elements array, optional
            An array cointaining the limits for latitude and longitude data, in which: [lon_min, lon_max, lat_min, lat_max].
            Default is the anomalous region [20, 85, -15, 15].
         
        dawn: bool, optional
            If True, assumes dawn. If False, assumes dusk. Default is true.
        
        time1: string, optional
            The start date. Default is 0.
        
        time2: string, optional
            The end date. Default is 1.
        
        length: integer, optional
            The number of data points to use. If None, then use the total length of the data. Default is None.
            
    Returns:
    """    
    f = 12

    my_cmap = copy(plt.cm.RdBu_r)
    my_cmap.set_over('r', 1.0)
    my_cmap.set_under('b', 1.0)
    
    if brange is None:
        bmax = np.max(br)
    else:
        bmax = brange
    bmin = -bmax
    
    if dawn == True:
        title = 'Dawn'
    else:
        title = 'Dusk'
    
    new_dir = '/home/oliveira/PythonScripts/Orbits/Single Orbits/' + anomaly + '/' + n_title
    if os.path.isdir(new_dir) == True:
        shutil.rmtree(new_dir)
    
    os.makedirs(new_dir)
    
    if length is None:
        length = len(br)
    
    for i in range(length):
    
        data = sp.io.readsav('/home/oliveira/ccati_mexuser/LANGLAIS_Matrices/'+anomaly+'/LANGLAIS_BR_ALT_' + str(int(round(alt[i]))) + '_RES_01.bin')
        brmodel = data['zbins']
        
        window = np.around(abs(br[i]))*sigma
        
        br_single = np.logical_and(np.around(brmodel) >= (np.around(br[i]) - window), \
                                   np.around(brmodel) <= (np.around(br[i]) + window))
        a = np.copy(brmodel)
        a[np.logical_not(br_single)] = np.nan
        
        fig, axes = plt.subplots(1, 1, sharex = True, sharey = True, figsize = [6, 6])
        plt.subplots_adjust(hspace = 0, wspace = 0)
        
        plt.subplot(111)
        plt.scatter(lon[i], lat[i], c = br[i], cmap = my_cmap, norm = colors.Normalize(vmin = bmin, vmax = bmax), \
                    edgecolors = 'black', s = 2.0)
        im = plt.imshow(a, extent = lim, origin = 'lower', cmap = my_cmap, norm = colors.Normalize(vmin = bmin, vmax = bmax))
        plt.xlim(lim[0], lim[1])
        plt.ylim(lim[2], lim[3])
        plt.ylabel('Latitude ($\degree$)')
        plt.xlabel('Longitude ($\degree$)')
        plt.title(title + '-side, ' + time1 + ' -- ' + time2 + '\nLocal Time: ' + str(np.around(lt[i], 1)) + 'h \t SZA = ' + \
                  str(int(round(sza[i]))) + '$\degree$' + '\t Alt = ' + str(np.around(alt[i])) + ' km')
        
        if vaz is not None:
            if vpol is not None:
                plt.quiver(lon[i], lat[i], vaz[i], vpol[i])#, scale = 100, headlength = 3, headaxislength = 2.5)
                #plt.quiverkey(q, X = 0.8, Y = 0.1, U = 2, label = '2 km/s', transform = fig.transFigure)
                      
        plt.subplots_adjust(right = 0.8)
        cbar_ax = fig.add_axes([0.82, 0.35, 0.03, 0.3])
        cbar = fig.colorbar(im, cax = cbar_ax, extend = 'both')
        cbar.set_label('$B_r$ (nT)', fontsize = f)
        cbar.ax.tick_params(labelsize = f)
    
        if i < 10:
            plt.savefig(new_dir + '/0' + str(i) + '.png', bbox_inches = 'tight', dpi = 500)
        else:
            plt.savefig(new_dir + '/' + str(i) + '.png', bbox_inches = 'tight', dpi = 500)