"""
Useful for the analysis and visualization of individual datapoints.
"""

import numpy as np
import scipy as sp
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from copy import copy
import datetime
import os
import IsabelaFunctions as isa
from tqdm import tqdm


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
    
    plt.savefig('/home/oliveira/PythonScripts/Orbits/Altitudes/'+ anomaly +'/Orbit_' + anomaly + '_alt_' + str(new_alt) + '_' + title + '.pdf', bbox_inches = 'tight')
    return


def plot_altitude_range(alt_range, alt, lon, lat, br, anomaly, lim, brange = None, vaz = None, vpol = None, dawn = True):
    """ Plots the measurements for a specific altitude range and the crustal magnetic field model (Langlais et al 2019) at an average height.

   Parameters:
        alt_range: 2-elements array
            The altitude range at which the data will be plotted.

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
        
        lim: 4-elements array
            An array cointaining the limits for latitude and longitude data, in which: [lon_min, lon_max, lat_min, lat_max].
        
        brange: int, optional
            The maximum magnetic field values used for scaling the colorbar. Default is None, so it computes based on the data.
        
        vaz: 1D array, optional
            An array containing the azimuthal velocity data.
            
        vpol: 1D array, optional
            An array containing the polar velocity data. If vaz AND vpol ARE NOT None, arrows showing the horizontal velocity will be plotted.
        
        dawn: bool, optional
            If True, assumes dawn. If False, assumes dusk. Default is true.
                
    Returns:
        None.
    """
    h = np.logical_and(np.around(alt) >= alt_range[0], np.around(alt) <= alt_range[1]) 
    
    avg = (alt_range[0] + alt_range[1]) // 2
    
    data = sp.io.readsav('/home/oliveira/ccati_mexuser/LANGLAIS_Matrices/' + anomaly + '/LANGLAIS_BR_ALT_' + str(avg) + '_RES_01.bin')
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
    plt.scatter(lon[h], lat[h], c = br[h], cmap = my_cmap, norm = colors.Normalize(vmin = bmin, vmax = bmax), \
                edgecolors = 'black')
    im = plt.imshow(brmodel, extent = lim, origin = 'lower', cmap = my_cmap, norm = colors.Normalize(vmin = bmin, vmax = bmax))
    
    if vaz is not None and vpol is not None:
        vaz[vaz == 0.0] = np.nan
        vpol[vpol == 0.0] = np.nan
        plt.quiver(lon[h], lat[h], vaz[h], vpol[h], scale = 50.0)
        title = title + '_VelArrows'
    
    plt.xlim(lim[0], lim[1])
    plt.ylim(lim[2], lim[3])
    plt.ylabel('Latitude ($\degree$)')
    plt.xlabel('Longitude ($\degree$)')
    plt.title(title + ', Altitude range = ' + str(alt_range[0]) + ' - ' + str(alt_range[1]) + ' km')
    
    plt.subplots_adjust(right = 0.8)
    cbar_ax = fig.add_axes([0.82, 0.25, 0.03, 0.5])
    cbar = fig.colorbar(im, cax = cbar_ax, extend = 'both')
    cbar.set_label('$B_r$ (nT)', fontsize = f)
    cbar.ax.tick_params(labelsize = f)
    
    plt.savefig('/home/oliveira/PythonScripts/Orbits/Altitudes/'+ anomaly +'/' + title + '_Orbit_' + anomaly + '_alt_' + \
                str(alt_range[0]) + '_' + str(alt_range[1]) + '.png', bbox_inches = 'tight', dpi = 500)
        
    return 


def plot_time_range(start, delta, date, alt, lon, lat, br, lt, sza, vaz, vpol, arrows = True, brange = None, lim = [20, 85, -15, 15], anomaly = 'A6', dawn = True, save = True):
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
        
        lt: 1D array
            An array containing the local time data.
            
        sza: 1D array
            An array containing the solar zenith angle data.
        
        vaz: 1D array, optional
            An array containing the azimuthal velocity data.
            
        vpol: 1D array, optional
            An array containing the polar velocity data. If vaz AND vpol ARE NOT None, arrows showing the horizontal velocity will be plotted.
         
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
        Lists with the input data for the selected time range.
        The strings of the start and the end dates.
    """
    time1 = datetime.datetime(start[0], start[1], start[2], start[3], start[4], start[5])
    time2 = datetime.datetime(start[0]+delta[0], start[1]+delta[1], start[2]+delta[2], start[3]+delta[3], start[4]+delta[4], \
                              start[5]+delta[5])
    
    date_range = np.logical_and(date >= time1, date < time2)
    
    date1 = date[date_range][0]
    date2 = date[date_range][-1]
    
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
    vaz[vaz == 0.0] = np.nan
    vpol[vpol == 0.0] = np.nan
    if arrows == True:
        plt.quiver(lon[date_range], lat[date_range], vaz[date_range], vpol[date_range], scale = 50.0)
    
    plt.xlim(lim[0], lim[1])
    plt.ylim(lim[2], lim[3])
    plt.xticks(visible = False)
    plt.ylabel('Latitude ($\degree$)')
    plt.title(title + '-side, ' + str(date1) + ' -- ' + str(date2))
    
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
    
    return len(br[date_range]), lon[date_range], lat[date_range], alt[date_range], br[date_range], lt[date_range], sza[date_range], vaz[date_range], vpol[date_range], str(date1), str(date2)


def plot_many_data(alt, lon, lat, br, lt, sza, n_title, anomaly, lim, model, vaz = None, vpol = None, one_map = True, contour = False, sigma = 0.1, brange = None, time1 = '0', time2 = '1', shifteast = 0.0, shiftnorth = 0.0, rotate = 0.0, p = 0, h = None, title = None):
    """ Plots many measurements on an average height model magnetic field map.

   Parameters:
        alt: 1D array
            An array containing the altitude data.
            
        lon: 1D array
            An array containing the longitude data.
            
        lat: 1D array
            An array containing the latitude data.
            
        br: 1D array
            An array containing the magnetic field data.
        
        lt: 1D array
            An array containing the local time data.
            
        sza: 1D array
            An array containing the solar zenith angle data.
            
        n_title: string
            A string used in the title and in the new directory.
            
        anomaly: string
            The anomaly index, e. g., A1, A2, A6, etc. This string is used to find the directory where the model matrices are located.
        
        lim: 4-elements array
            An array cointaining the limits for latitude and longitude data, in which: [lon_min, lon_max, lat_min, lat_max].
        
        model: function
            A function that calculates the interpolated magnetic field model at a specific point. Can be created by IsabelaFunctions.read.crustal_model_files.
            
        vaz: 1D array, optional
            An array containing the azimuthal velocity data.
            
        vpol: 1D array, optional
            An array containing the polar velocity data. If vaz AND vpol ARE NOT None, arrows showing the horizontal velocity will be plotted.
         
        one_map: bool, optional
            If True, this function generates a single map, containing all the measurements. If False, it generates a map for each single measurement. Default is True.
        
        contour: bool, optional
            Only used if one_map is False. If True, plots the magnetic field model contour in the background. If False, plots the whole magnetic field model map. Default is False.
        
        sigma: float, optional
            Only used if contour is True. The thickness of the magnetic field contour, in percent. Default is 0.1 (10% of the value).
            Example: if the measurement is 10 nT, the contour includes the values (10 - 0.1*10) <= B <= (10 + 0.1*10), which is 9 <= B <= 11. 
        
        brange: integer, optional
            The maximum magnetic field values used for scaling the colorbar. Default is None, so it computes based on the data.
        
        time1: string, optional
            The start date. Default is 0.
        
        time2: string, optional
            The end date. Default is 1.
        
        shifteast: double, optional
            The value in degrees by which the data will be shifted longitudinally (negative values are westward). Default is 0.0.
            
        shiftnorth: double, optional
            The value in degrees by which the data will be shifted latitudinally (negative values are southward). Default is 0.0.
        
        rotate: double, optional
            The value in degrees by which the data will be rotated counterclockwise.
        
        p: integer, optional
            The index of the point of rotation. This value is required if rotate is called. Default is 0, which means it takes the first point in the array.
        
        h: float, optional
            The heigth of the crustal magnetic field model if one_map is True. If None, the mean height is used.
            
        title: string, optional
            The title (bold) that will appear in the maps.
    Returns:
    """    
    f = 14

    my_cmap = copy(plt.cm.RdBu_r)
    my_cmap.set_over('r', 1.0)
    my_cmap.set_under('b', 1.0)
    
    if brange is None:
        bmax = np.max(br)
    else:
        bmax = brange
    bmin = -bmax
    
    if vaz is not None and vpol is not None:
        vaz[vaz == 0.0] = np.nan
        vpol[vpol == 0.0] = np.nan
        avg = np.sqrt(abs(np.nanmedian(vaz))**2 + abs(np.nanmedian(vpol))**2)
    else:
        vaz = np.full_like(br, np.nan)
        vpol = np.full_like(br, np.nan)
    
    res_total = 0.0
    if shifteast != 0.0 or shiftnorth != 0.0 or rotate != 0.0:
        rot_lon, rot_lat = rot2D(lon+shifteast, lat+shiftnorth, rotate, p)
        res, res_total = isa.shifting.orbit_residual(rot_lon, rot_lat, alt, br, lim, model)
    res2, res_total2 = isa.shifting.orbit_residual(lon, lat, alt, br, lim, model)
    
    if one_map == True:
        fig, axes = plt.subplots(1, 1, sharex = True, sharey = True, figsize = [8, 8])
        plt.subplots_adjust(hspace = 0, wspace = 0)
        plt.subplot(111)
        
        #plt.text(lim[1]+0.5, lim[3]-1.0, '$|\overline{V}_h|$ = ' + str(np.around(avg, 1)) + ' km/s', fontsize = f, verticalalignment = 'center')
        #plt.text(lim[1]+0.5, lim[3], 'Rotation = ' + str(round(rotate)) + '$\degree$', fontsize = f)
        #plt.text(lim[1]+0.5, lim[2]+0.5, '$\Delta B_{Shift}$ = ' + str(np.around(res_total, 1)) + ' nT', fontsize = f)
        #plt.text(lim[1]+0.5, lim[2]+1.5, '$\Delta B_{Orig}$ = ' + str(np.around(res_total2, 1)) + ' nT', fontsize = f)
        
        plt.xlim(lim[0], lim[1])
        plt.ylim(lim[2], lim[3])
        plt.ylabel('Latitude ($\degree$)', fontsize = f)
        plt.yticks(fontsize = f)
        plt.xlabel('Longitude ($\degree$)', fontsize = f)
        plt.xticks(fontsize = f)
        
        if h == None:
            h = int((np.max(alt) + np.min(alt)) // 2)
        brmodel = isa.fieldmodel.model_map([lim[0], lim[1]], [lim[2], lim[3]], h, 'Br')
    
        p1 = plt.scatter(lon, lat, c = br, cmap = my_cmap, norm = colors.Normalize(vmin = bmin, vmax = bmax), edgecolors = 'black')
        im = plt.imshow(brmodel, extent = lim, origin = 'lower', cmap = my_cmap, norm = colors.Normalize(vmin = bmin, vmax = bmax))
        
        if shifteast != 0.0 or shiftnorth != 0.0 or rotate != 0.0: 
            plt.quiver(lim[1]-6., lim[2]+1.0, -shifteast, -shiftnorth, color = 'white', edgecolor = 'black', pivot = 'middle', \
                       lw = 1.0)    
            plt.quiver(lim[1]-6., lim[2]+2.0, np.nanmedian(vaz), np.nanmedian(vpol), pivot = 'middle', edgecolor = 'black', lw = 1.0)
            plt.text(lim[1]-0.3, lim[2]+0.6, '$\mathbf{\overline{V}_h}$ direction\nFit direction', fontsize = f, linespacing = 2, ha = 'right')
        
            p2 = plt.scatter(rot_lon, rot_lat, c = br, cmap = my_cmap, norm = colors.Normalize(vmin = bmin, vmax = bmax),\
                             edgecolors = 'black', marker = 'D')
            plt.legend([p1, p2], ['Data', 'Fit'], fancybox = True, loc = 'lower left', edgecolor = 'inherit', framealpha = 1.0, fontsize = f,\
                       handletextpad = 0.4)
        
        hours = str(np.around((np.min(lt)+np.max(lt))/2))[0:2]
        if hours[1] == '.':
            hours = '0' + hours[0]
        minutes = str(int((((np.min(lt) + np.max(lt)) % 2) / 2 * 60)))
        plt.title(time1 + ' $-$ ' + time2[-8:] + '\n~' + hours + ':' + minutes + ' LT\tSZA = ' \
                  + str(int(np.min(sza))) + '$\degree - $' + str(int(np.max(sza))) + \
                  '$\degree$', fontsize = f)
        
        if title is not None:
            plt.text(lim[0] + 0.5, lim[3] - 1.0, title, weight = 'bold', fontsize = f, va = 'center')
            
        plt.subplots_adjust(right = 0.8)
        cbar_ax = fig.add_axes([0.82, 0.2, 0.03, 0.6])
        cbar = fig.colorbar(im, cax = cbar_ax, extend = 'both')
        cbar.set_label('$B_r$ (nT)', fontsize = f)
        cbar.ax.tick_params(labelsize = f)
        
        plt.savefig('/home/oliveira/PythonScripts/Orbits/Single Orbits/'+ anomaly +'/Orbit_' + anomaly + '_' + n_title + '.pdf', bbox_inches = 'tight')
    
        print(' res orig = ' + str(res_total2) + ' nT\n res shift = ' + str(res_total) + ' nT\n Vh mean = ' + str(avg) + ' km/s')
        
    else:
        #assert type(model) is np.ndarray, "Please enter a matrix containing the model data."
        new_dir = '/home/oliveira/PythonScripts/Orbits/Single Orbits/' + anomaly + '/' + n_title
        if os.path.isdir(new_dir) == False:
            os.makedirs(new_dir)
        length = len(br)
        
        if contour == True:   
            for i in range(length):
                fig, axes = plt.subplots(1, 1, sharex = True, sharey = True, figsize = [8, 8], num = 1)
                plt.subplots_adjust(hspace = 0, wspace = 0)
                plt.subplot(111)
                
                plt.xlim(lim[0], lim[1])
                plt.ylim(lim[2], lim[3])
                plt.ylabel('Latitude ($\degree$)', fontsize = f)
                plt.yticks(fontsize = f)
                plt.xlabel('Longitude ($\degree$)', fontsize = f)
                plt.xticks(fontsize = f)
                
                data = sp.io.readsav('/home/oliveira/ccati_mexuser/LANGLAIS_Matrices/' + anomaly + '/LANGLAIS_BR_ALT_' + str(int(alt[i])) + '_RES_01.bin')
                brmodel = data['zbins']
                
                window = np.around(abs(br[i]))*sigma
                br_single = np.logical_and(np.around(brmodel) >= (np.around(br[i]) - window), \
                                           np.around(brmodel) <= (np.around(br[i]) + window))
                a = np.copy(brmodel)
                a[np.logical_not(br_single)] = np.nan
                
                p1 = plt.scatter(lon[i], lat[i], c = br[i], cmap = my_cmap, norm = colors.Normalize(vmin = bmin, vmax = bmax), edgecolors = 'black')
                im = plt.imshow(a, extent = lim, origin = 'lower', cmap = my_cmap, norm = colors.Normalize(vmin = bmin, vmax = bmax))
                p2 = plt.scatter(rot_lon[i], rot_lat[i], c = br[i], cmap = my_cmap, norm = colors.Normalize(vmin = bmin, vmax = bmax),\
                             edgecolors = 'black', marker = 'D')
                    
                if shifteast != 0.0 or shiftnorth != 0.0: 
                    plt.legend([p1, p2], ['Data', 'Fit'], fancybox = True, loc = 'lower left', edgecolor = 'inherit', framealpha = 1.0, fontsize = f,\
                       handletextpad = 0.4)
                    plt.quiver(lim[1]-5.5, lim[2]+1.0, -shifteast, -shiftnorth, color = 'white', edgecolor = 'black', pivot = 'middle', \
                       lw = 1.0)    
                    plt.text(lim[1]-0.3, lim[2]+1., 'Fit direction', fontsize = f, linespacing = 2, ha = 'right', va = 'center')
        
                hours = str(np.around(lt[i]))[0:2]
                if hours[1] == '.':
                     hours = '0' + hours[0]
                minutes = str(lt[i] % 1 * 60)[0:2]
                plt.title(time1 + ' $-$ ' + time2[-8:] + '\n' + hours + ':' + minutes + \
                          ' LT \tSZA = ' + str(int(round(sza[i]))) + '$\degree$', fontsize = f)
                
                if vaz is not None and vpol is not None:
                    plt.quiver(lon[i], lat[i], vaz[i], vpol[i], edgecolor = 'black', lw = 1.0)
                
                if title is not None:
                    plt.text(lim[0] + 0.5, lim[3] - 1.0, title, weight = 'bold', fontsize = f, va = 'center')
        
                plt.subplots_adjust(right = 0.8)
                cbar_ax = fig.add_axes([0.82, 0.2, 0.03, 0.6])
                cbar = fig.colorbar(im, cax = cbar_ax, extend = 'both')
                cbar.set_label('$B_r$ (nT)', fontsize = f)
                cbar.ax.tick_params(labelsize = f)
            
                if i < 10:
                    plt.savefig(new_dir + '/contour_0' + str(i) + '.png', bbox_inches = 'tight', dpi = 500)
                else:
                    plt.savefig(new_dir + '/contour_' + str(i) + '.png', bbox_inches = 'tight', dpi = 500)
                plt.clf()
    
        else:
            for i in range(length):
                fig, axes = plt.subplots(1, 1, sharex = True, sharey = True, figsize = [8, 8], num = 1)
                plt.subplots_adjust(hspace = 0, wspace = 0)
                plt.subplot(111)
             
                plt.xlim(lim[0], lim[1])
                plt.ylim(lim[2], lim[3])
                plt.ylabel('Latitude ($\degree$)', fontsize = f)
                plt.yticks(fontsize = f)
                plt.xlabel('Longitude ($\degree$)', fontsize = f)
                plt.xticks(fontsize = f)
                
                data = sp.io.readsav('/home/oliveira/ccati_mexuser/LANGLAIS_Matrices/'+anomaly+'/LANGLAIS_BR_ALT_' + str(int(round(alt[i]))) + '_RES_01.bin')
                brmodel = data['zbins']
                        
                p1 = plt.scatter(lon[i], lat[i], c = br[i], cmap = my_cmap, norm = colors.Normalize(vmin = bmin, vmax = bmax), edgecolors = 'black')
                im = plt.imshow(brmodel, extent = lim, origin = 'lower', cmap = my_cmap, norm = colors.Normalize(vmin = bmin, vmax = bmax))
                
                hours = str(np.around(lt[i]))[0:2]
                if hours[1] == '.':
                     hours = '0' + hours[0]
                minutes = str(lt[i] % 1 * 60)[0:2]
                plt.title(time1 + ' $-$ ' + time2[-8:] + '\n' + hours + ':' + minutes + \
                          ' LT \tSZA = ' + str(int(round(sza[i]))) + '$\degree$', fontsize = f)
                
                if vaz is not None and vpol is not None:
                    plt.quiver(lon[i], lat[i], vaz[i], vpol[i], edgecolor = 'black', lw = 1.0)
                
                p2 = plt.scatter(rot_lon[i], rot_lat[i], c = br[i], cmap = my_cmap, norm = colors.Normalize(vmin = bmin, vmax = bmax), \
                            edgecolors = 'black', marker = 'D')
                    
                if shifteast != 0.0 or shiftnorth != 0.0:
                    plt.legend([p1, p2], ['Data', 'Fit'], fancybox = True, loc = 'lower left', edgecolor = 'inherit', framealpha = 1.0, fontsize = f,\
                       handletextpad = 0.4)
                    plt.quiver(lim[1]-5.5, lim[2]+1.0, -shifteast, -shiftnorth, color = 'white', edgecolor = 'black', pivot = 'middle', \
                       lw = 1.0)    
                    plt.text(lim[1]-0.3, lim[2]+1., 'Fit direction', fontsize = f, linespacing = 2, ha = 'right', va = 'center')
                
                if title is not None:
                    plt.text(lim[0] + 0.5, lim[3] - 1.0, title, weight = 'bold', fontsize = f, va = 'center')
        
                plt.subplots_adjust(right = 0.8)
                cbar_ax = fig.add_axes([0.82, 0.2, 0.03, 0.6])
                cbar = fig.colorbar(im, cax = cbar_ax, extend = 'both')
                cbar.set_label('$B_r$ (nT)', fontsize = f)
                cbar.ax.tick_params(labelsize = f)
            
                if i < 10:
                    plt.savefig(new_dir + '/full_0' + str(i) + '.png', bbox_inches = 'tight', dpi = 500)
                else:
                    plt.savefig(new_dir + '/full_' + str(i) + '.png', bbox_inches = 'tight', dpi = 500)
                plt.clf()
    
    return res_total2, res_total


def plot_many_altitude(alt1, lon1, lat1, lim, hrange = None, alt2 = None, lon2 = None, lat2 = None, title = None, h = None):
    """ Plots many altitudes.

   Parameters:
        alt1: 1D array
            An array containing the altitude data.
            
        lon1: 1D array
            An array containing the longitude data.
            
        lat1: 1D array
            An array containing the latitude data.
            
        lim: 4-elements array
            An array cointaining the limits for latitude and longitude data, in which: [lon_min, lon_max, lat_min, lat_max].
        
        hrange: 2-elements array, optional
            An array containing the lower and upper limits of the altitude range. If None, it computes based on the first dataset.
            
        alt2: 1D array, optional
            An array containing the altitude data (second dataset).
            
        lon2: 1D array, optional
            An array containing the longitude data (second dataset).
            
        lat2: 1D array, optional
            An array containing the latitude data (second dataset).
            
        title: string or 2-elements array of strings, optional
            The title that will appear next to the measurements of altitude.
            
        h: float, optional
            If you want to plot the contour lines of the radial magnetic field in the background, set this parameter to a float equal to the desired height of the magnetic field model. Default is not plot anything. 
    Returns:
    """
    f = 14
    my_cmap = copy(plt.cm.viridis_r)
    
    if hrange is None:
        hrange = [np.min(alt1), np.max(alt1)]

    fig, axes = plt.subplots(1, 1, figsize = [8, 8])
    plt.subplots_adjust(hspace = 0, wspace = 0)
    plt.subplot(111)
      
    plt.xlim(lim[0], lim[1])
    plt.ylim(lim[2], lim[3])
    plt.ylabel('Latitude ($\degree$)', fontsize = f)
    plt.yticks(fontsize = f)
    plt.xlabel('Longitude ($\degree$)', fontsize = f)
    plt.xticks(fontsize = f)
    
    if h is not None:
        brmodel = isa.fieldmodel.model_map([lim[0], lim[1]], [lim[2], lim[3]], h, 'Br')
        
        lengthx = (lim[1] - lim[0]) / (np.size(brmodel, 0)-1)
        lengthy = (lim[3] - lim[2]) / (np.size(brmodel, 1)-1)
        
        x = np.arange(lim[0], lim[1] + lengthx, lengthx)
        y = np.arange(lim[2], lim[3] + lengthy, lengthy)
        X, Y = np.meshgrid(x, y)
        C = plt.contour(X, Y, brmodel, 4, colors = 'k', alpha = 0.7)
        plt.clabel(C, inline = True, fmt = '%1.0f', fontsize = 12)
    
    p1 = plt.scatter(lon1, lat1, c = alt1, cmap = my_cmap, norm = colors.Normalize(vmin = hrange[0], vmax = hrange[1]), \
                     edgecolors = 'black', alpha = 1, s = 50)
    
    if alt2 is not None:
        p2 = plt.scatter(lon2, lat2, c = alt2, cmap = my_cmap, norm = colors.Normalize(vmin = hrange[0], vmax = hrange[1]), \
            edgecolors = 'black', marker = 'D', alpha = 1, s = 50)
        
        if title is not None:
            plt.legend([p1, p2], [title[0], title[1]], fancybox = True, loc = 'lower left', edgecolor = 'inherit', \
                       framealpha = 1.0, fontsize = f, handletextpad = 0.4)
    
    plt.subplots_adjust(right = 0.8)
    cbar_ax = fig.add_axes([0.82, 0.2, 0.03, 0.6])
    cbar = fig.colorbar(p1, cax = cbar_ax)
    cbar.set_label('Altitude (km)', fontsize = f)
    cbar.ax.tick_params(labelsize = f)
    
    plt.savefig('altitudes.pdf', bbox_inches = 'tight')
    return


def rot2D(lon, lat, angle, p):
    """ Rotates an array counterclockwise around a point in a longitude vs. latitude map.
    
   Parameters:
        lon: 1D array
            An array containing the longitude data.
            
        lat: 1D array
            An array containing the latitude data.
        
        angle: double
            The angle of rotation, in degrees.
            
        p: integer
            The index of the point of rotation.
            
    Returns:
        Two arrays containing the rotated longitude and latitude.
    """
    angle = np.deg2rad(angle)
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    
    p0 = [lon[p], lat[p]]
    new_lon = []
    new_lat = []
    
    for i in range(len(lon)):
        lon0, lat0 = [lon[i] - p0[0], lat[i] - p0[1]]
        lon1, lat1 = R @ [lon0, lat0]
        
        new_lon.append(lon[i] + (lon1 - lon0))
        new_lat.append(lat[i] + (lat1 - lat0))
        
    new_lon = np.array(new_lon)
    new_lat = np.array(new_lat)
    
    return new_lon, new_lat
    

def minres(alt, lon, lat, br, fn, lim, Erange, Nrange, Rrange, Estep = 0.1, Nstep = 0.1, Rstep = 1.0, p = None):
    """ Calculates the residuals in nT for a range of horizontal and vertical shifts.
    
   Parameters:
       alt: 1D array
            An array containing the altitude data.
        
        lon: 1D array
            An array containing the longitude data.
            
        lat: 1D array
            An array containing the latitude data.
        
        br: 1D array
            An array containing the magnetic field data.
        
        fn: function or str
            A function that calculates the interpolated magnetic field model at a specific point. Can be created by IsabelaFunctions.read.crustal_model_files.
            If str, the magnetic field components are calculated by IsabelaFunctions.fieldmodel.mag_components, and the component is defined by the string ('Br', 'Btheta', or 'Bphi').
        
        lim: 4-elements array
            An array cointaining the limits for latitude and longitude data, in which: [lon_min, lon_max, lat_min, lat_max].
        
        Nrange: 2-elements array
            An array containing the northern shift range in degrees.
        
        Erange: 2-elements array
            An array containing the eastern shift range in degrees.
        
        Rrange: 2-elements array
            An array containing the rotation range in degrees.
        
        Nstep: float, optional
            The angular step in the northern shift. Default is 0.1. 
        
        Estep: float, optional
            The angular step in the eastern shift. Default is 0.1. 
        
        Rstep: float, optional
            The angular step in the rotation angle. Default is 1.0. 
        
        p: integer, optional
            The index of the point of rotation. Default is the middle point.
            
    Returns:
        A matrix containing all of the residuals in the 3 dimensions that are varied.
        4 floats, corresponding to the minimum residual, and its respective latitudinal and longitudinal shifts and the rotation angle.
    """
    Nlen = int((Nrange[1] - Nrange[0]) / Nstep + 1.0)
    Elen = int((Erange[1] - Erange[0]) / Estep + 1.0)
    Rlen = int((Rrange[1] - Rrange[0]) / Rstep + 1.0)
    length = len(br)
    
    if p is None:
        p = length // 2
    
    angle = np.linspace(Rrange[0], Rrange[1], Rlen)
    east = np.linspace(Erange[0], Erange[1], Elen)
    north = np.linspace(Nrange[0], Nrange[1], Nlen)
    matrix = np.empty((Elen, Nlen, Rlen))
    
    test = isinstance(fn, str)
    if test is True:
        model = np.zeros((Elen, Nlen, Rlen, length))
        for i in tqdm(range(Elen)):
            for j in range(Nlen):
                for k in range(Rlen):
                    rot_lons, rot_lats = isa.orbits.rot2D(lon+east[i], lat+north[j], angle[k], p)
                    model[i, j, k] = isa.fieldmodel.mag_components(rot_lons, rot_lats, alt, fn)
                    res = np.double(abs((br) - model[i, j, k]))
                    count = np.count_nonzero(~np.isnan(res))
                    matrix[i, j, k] = np.nansum(res)/count
    
    else:
        for i in tqdm(range(Elen)):
            for j in range(Nlen):
                for k in range(Rlen):
                    rot_lons, rot_lats = isa.orbits.rot2D(lon+east[i], lat+north[j], angle[k], p)
                    res, total = isa.shifting.orbit_residual(rot_lons, rot_lats, alt, br, lim, fn)
                    matrix[i, j, k] = total
        
    min0 = np.where(matrix == np.nanmin(matrix))
    lon0 = min0[0][0]*Estep + Erange[0]
    lat0 = min0[1][0]*Nstep + Nrange[0]
    rot0 = min0[2][0]*Rstep + Rrange[0]
    
    return lon0, lat0, rot0


def minres_finer(alt, lon, lat, br, fn, lim, Erange, Nrange, Rrange, p = None):
    """ Calculates the residuals in nT for a range of horizontal and vertical shifts, using first a coarse grid, and then a finer grid. It is probably faster than minres.
    
   Parameters:
       alt: 1D array
            An array containing the altitude data.
        
        lon: 1D array
            An array containing the longitude data.
            
        lat: 1D array
            An array containing the latitude data.
        
        br: 1D array
            An array containing the magnetic field data.
        
        fn: function or str
            A function that calculates the interpolated magnetic field model at a specific point. Can be created by IsabelaFunctions.read.crustal_model_files.
            If str, the magnetic field components are calculated by IsabelaFunctions.fieldmodel.mag_components, and the component is defined by the string ('Br', 'Btheta', or 'Bphi').
        
        lim: 4-elements array
            An array cointaining the limits for latitude and longitude data, in which: [lon_min, lon_max, lat_min, lat_max].
        
        Nrange: 2-elements array
            An array containing the northern shift range in degrees.
        
        Erange: 2-elements array
            An array containing the eastern shift range in degrees.
        
        Rrange: 2-elements array
            An array containing the rotation range in degrees.
         
        p: integer, optional
            The index of the point of rotation. Default is the middle point.
            
    Returns:
        A matrix containing all of the residuals in the 3 dimensions that are varied.
        4 floats, corresponding to the minimum residual, and its respective latitudinal and longitudinal shifts and the rotation angle.
    """
    length = len(br)
    if p is None:
        p = np.where(alt == np.nanmin(alt))
    
    step1 = 1.0
    Nlen = int((Nrange[1] - Nrange[0]) / step1 + 1.0)
    Elen = int((Erange[1] - Erange[0]) / step1 + 1.0)
    Rlen = int((Rrange[1] - Rrange[0]) / step1 + 1.0)
    
    angle = np.linspace(Rrange[0], Rrange[1], Rlen)
    east = np.linspace(Erange[0], Erange[1], Elen)
    north = np.linspace(Nrange[0], Nrange[1], Nlen)
    matrix = np.empty((Elen, Nlen, Rlen))
    
    test = isinstance(fn, str)
    if test is True:
        model = np.zeros((Elen, Nlen, Rlen, length))
        for i in tqdm(range(Elen)):
            for j in range(Nlen):
                for k in range(Rlen):
                    rot_lons, rot_lats = isa.orbits.rot2D(lon+east[i], lat+north[j], angle[k], p)
                    model[i, j, k] = isa.fieldmodel.mag_components(rot_lons, rot_lats, alt, fn)
                    res = np.double(abs((br) - model[i, j, k]))
                    count = np.count_nonzero(~np.isnan(res))
                    matrix[i, j, k] = np.nansum(res)/count
    
    else:
        for i in tqdm(range(Elen)):
            for j in range(Nlen):
                for k in range(Rlen):
                    rot_lons, rot_lats = isa.orbits.rot2D(lon+east[i], lat+north[j], angle[k], p)
                    res, total = isa.shifting.orbit_residual(rot_lons, rot_lats, alt, br, lim, fn)
                    matrix[i, j, k] = total
        
    min0 = np.where(matrix == np.nanmin(matrix))
    lon0 = min0[0][0]*step1 + east[0]
    lat0 = min0[1][0]*step1 + north[0]
    rot0 = min0[2][0]*step1 + angle[0]
    
    step2 = 0.5
    Nlen = 9
    Elen = 9
    Rlen = 9
    
    angle = np.linspace(-2.0, 2.0, Rlen) + rot0
    east = np.linspace(-2.0, 2.0, Elen) + lon0
    north = np.linspace(-2.0, 2.0, Nlen) + lat0
    matrix = np.empty((Elen, Nlen, Rlen))
    
    test = isinstance(fn, str)
    if test is True:
        model = np.zeros((Elen, Nlen, Rlen, length))
        for i in tqdm(range(Elen)):
            for j in range(Nlen):
                for k in range(Rlen):
                    rot_lons, rot_lats = isa.orbits.rot2D(lon+east[i], lat+north[j], angle[k], p)
                    model[i, j, k] = isa.fieldmodel.mag_components(rot_lons, rot_lats, alt, fn)
                    res = np.double(abs((br) - model[i, j, k]))
                    count = np.count_nonzero(~np.isnan(res))
                    matrix[i, j, k] = np.nansum(res)/count

    else:
        for i in tqdm(range(Elen)):
            for j in range(Nlen):
                for k in range(Rlen):
                    rot_lons, rot_lats = isa.orbits.rot2D(lon+east[i], lat+north[j], angle[k], p)
                    res, total = isa.shifting.orbit_residual(rot_lons, rot_lats, alt, br, lim, fn)
                    matrix[i, j, k] = total

    min00 = np.where(matrix == np.nanmin(matrix))
    lon00 = min00[0][0]*step2 + east[0]
    lat00 = min00[1][0]*step2 + north[0]
    rot00 = min00[2][0]*step2 + angle[0]
 
    return lon00, lat00, rot00


def index_orbit(date, alt = None, lon = None, lat = None, br = None, lt = None, sza = None, vaz = None, vpol = None, dens = None, short_lists = True, time = 7200):
    """ Separates the data set in sets of orbits.
    
    Parameters:
        date: 1D array
            An array containing the date/time data.
            
        alt: 1D array, optional
            An array containing the altitude data.
            
        lon: 1D array, optional
            An array containing the longitude data.
            
        lat: 1D array, optional
            An array containing the latitude data.
            
        br: 1D array, optional
            An array containing the magnetic field data.
        
        lt: 1D array, optional
            An array containing the local time data.
            
        sza: 1D array, optional
            An array containing the solar zenith angle data.
            
        vaz: 1D array, optional
            An array containing the azimuthal velocity data.
            
        vpol: 1D array, optional
            An array containing the polar velocity data.
        
        dens: 1D array, optional
            An array containing the density data.
        
        short_lists: bool, optional
            If true, remove the orbits with less than 10 data points. Default is True.
        
        time: int, optional
            The time difference, in seconds, that defines two different orbits. Default is 7200 seconds (2 hours).
    
    Returns:
        9 lists containing the orbits data for each input parameter.
    """
    list_date = []
    list_alt = []
    list_lon = []
    list_lat = []
    list_br = []
    list_sza = []
    list_lt = []
    list_vaz = []
    list_vpol = []
    list_dens = []
    
    j = 0
    for i in range(len(date)-1):
        delta = date[i+1] - date[i]
        
        # If the time difference is greater than 2 hours
        if delta.total_seconds() > time:
            list_date.append(date[j:i+1])
            list_alt.append(alt[j:i+1])
            list_lon.append(lon[j:i+1])
            list_lat.append(lat[j:i+1])
            list_sza.append(sza[j:i+1])
            list_lt.append(lt[j:i+1])
            list_vaz.append(vaz[j:i+1])
            list_vpol.append(vpol[j:i+1])
            list_br.append(br[j:i+1])
            list_dens.append(dens[j:i+1])
            
            j = i+1
    
    # Append the last orbit
    list_date.append(date[j:-1])
    list_alt.append(alt[j:-1])
    list_lon.append(lon[j:-1])
    list_lat.append(lat[j:-1])
    list_sza.append(sza[j:-1])
    list_lt.append(lt[j:-1])
    list_vaz.append(vaz[j:-1])
    list_vpol.append(vpol[j:-1])
    list_br.append(br[j:-1])
    list_dens.append(dens[j:-1])
    
    # If there are less than 10 elements in the orbit, remove it
    if short_lists is True:
        for i in reversed(range(len(list_date))):
            if len(list_date[i]) < 10:
                list_date.pop(i)
                list_alt.pop(i)
                list_lon.pop(i)
                list_lat.pop(i)
                list_sza.pop(i)
                list_lt.pop(i)
                list_vaz.pop(i)
                list_vpol.pop(i)
                list_br.pop(i)
                list_dens.pop(i)
            
    return list_date, list_alt, list_lon, list_lat, list_br, list_lt, list_sza, list_vaz, list_vpol, list_dens

