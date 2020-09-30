import numpy as np
import IsabelaFunctions as isa
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from copy import copy


def B_sum(BM, BO):
    """ Equation 3.2 in Master Dissertation.

    Parameters:
        BM: array
            Model magnetic field.

        BO: array
            Observed magnetic field (rolled or not).

    Returns:
        total_sum: float
            Sum of the absolute difference from the subtraction of the two magnetic fields, normalized by the number of data points.
    """
    res = abs(BM - BO)
    count = np.count_nonzero(~np.isnan(res))
    total_sum = np.nansum(res)/count
    return total_sum


def shifting_technique(BM, BO, N = 10, region = None):
    """ Equation 3.2 in Master Dissertation, with shifting applied

   Parameters:
        BM: array
            Model magnetic field.

        BO: array
            Observed magnetic field.

        N: integer, optional
            Total number of positive longitudinal shifts.

        region: bool (default is False), optional
            If False, elements that roll beyond the last position are re-introduced at the first. If True, elements that roll beyond the last position are counted as np.nan.

    Returns:
        total_sum: float
            B_sum for a range of longitudinal shifts
    """
    BM_summed = []
    for i in np.arange(-N, N+1):
        BM_shifted = np.roll(BM, i, axis = 1)
        if region is None:
            BM_summed.append(B_sum(BM_shifted, BO))
        else:
            BM_summed.append(B_sum(BM_shifted[region], BO[region]))
        
    total_sum = np.array(BM_summed) 

    return total_sum


def shifting_region(model, dawn, dusk, lon, lat, vmax, name, binsize = 0.5):
    """ Shifting technique applied to specific anomalous regions. Used for MAVEN data (4 altitude ranges as input).

   Parameters:
        model: list of 4 arrays
            Model magnetic field for 4 altitude ranges.

        dawn: list of 4 arrays
            Observed magnetic field at dawn-side for 4 altitude ranges.
            
        dusk: list of 4 arrays
            Observed magnetic field at dusk-side for 4 altitude ranges.
        
        binsize: float, optional
            The bin size of the maps in degrees (bin size of lon = bin size of lat). Default is 0.5.

        lon: [a, b] array
            Longitude limits of the anomaly.

        lat: [c, d] array
            Latitude limits of the anomaly.
        
        vmax: float
            Maximum value of the magnetic field to be shown in the colorbar.
        
        name: string
            The generated figures will be named after this string.
            
    Returns:
        mins: array
        error: array
    """
    zoom, maps = isa.mapsetup.zoom_in(model[0], lon, lat, binsize)
    
    my_cmap = copy(plt.cm.RdBu_r)
    my_cmap.set_over('r', 1.0)
    my_cmap.set_under('b', 1.0)
    vmin = -vmax
    f = 12
    f2 = 10

    fig, axes = plt.subplots(4, 2, sharex = True, sharey = True, figsize = [4.5, 8])
    plt.subplots_adjust(hspace = 0.1, wspace = 0.1)

    ax = plt.subplot(421)
    plt.minorticks_on()
    plt.imshow(dawn[0], extent = [0, 360, -90, 90], origin = 'lower', interpolation = 'none', 
           cmap = my_cmap, norm = colors.Normalize(vmin = vmin, vmax = vmax), aspect = 'auto')
    plt.tick_params(axis = 'both', direction = 'in', bottom = True, top = True, right = True, left = True, length = 5.0)
    plt.tick_params(which = 'minor', axis = 'both', direction = 'in', bottom = True, top = True, right = True, left = True, length = 2.0)
    plt.ylabel('Latitude ($\degree$)', fontsize = f)
    plt.xticks(visible = False)
    plt.text(0.5, 1.05, 'Dawn-side', transform = ax.transAxes, horizontalalignment = 'center', fontsize = f)
    plt.xlim(lon[0], lon[1])
    plt.ylim(lat[0], lat[1])

    plt.subplot(423, sharex = ax, sharey = ax)
    plt.imshow(dawn[1], extent = [0, 360, -90, 90], origin = 'lower', interpolation = 'none', 
           cmap = my_cmap, norm = colors.Normalize(vmin = vmin, vmax = vmax), aspect = 'auto')
    plt.tick_params(axis = 'both', direction = 'in', bottom = True, top = True, right = True, left = True, length = 5.0)
    plt.tick_params(which = 'minor', axis = 'both', direction = 'in', bottom = True, top = True, right = True, left = True, length = 2.0)
    plt.ylabel('Latitude ($\degree$)', fontsize = f)
    plt.xticks(visible = False)

    plt.subplot(425, sharex = ax, sharey = ax)
    plt.imshow(dawn[2], extent = [0, 360, -90, 90], origin = 'lower', interpolation = 'none', 
           cmap = my_cmap, norm = colors.Normalize(vmin = vmin, vmax = vmax), aspect = 'auto')
    plt.tick_params(which = 'minor', axis = 'both', direction = 'in', bottom = True, top = True, right = True, left = True, length = 2.0)
    plt.tick_params(axis = 'both', direction = 'in', bottom = True, top = True, right = True, left = True, length = 5.0)
    plt.ylabel('Latitude ($\degree$)', fontsize = f)
    plt.xticks(visible = False)

    plt.subplot(427, sharex = ax, sharey = ax)
    plt.imshow(dawn[3], extent = [0, 360, -90, 90], origin = 'lower', interpolation = 'none', 
           cmap = my_cmap, norm = colors.Normalize(vmin = vmin, vmax = vmax), aspect = 'auto')
    plt.tick_params(axis = 'both', direction = 'in', bottom = True, top = True, right = True, left = True, length = 5.0)
    plt.tick_params(which = 'minor', axis = 'both', direction = 'in', bottom = True, top = True, right = True, left = True, length = 2.0)
    plt.ylabel('Latitude ($\degree$)', fontsize = f)
    plt.xlabel('Longitude ($\degree$)', fontsize = f)

    ax2 = plt.subplot(422, sharex = ax, sharey = ax)
    plt.imshow(dusk[0], extent = [0, 360, -90, 90], origin = 'lower', interpolation = 'none', 
           cmap = my_cmap, norm = colors.Normalize(vmin = vmin, vmax = vmax), aspect = 'auto')
    plt.tick_params(axis = 'both', direction = 'in', bottom = True, top = True, right = True, left = True, length = 5.0)
    plt.tick_params(which = 'minor', axis = 'both', direction = 'in', bottom = True, top = True, right = True, left = True, length = 2.0)
    plt.xticks(visible = False)
    plt.yticks(visible = False)
    plt.text(0.5, 1.05, 'Dusk-side', transform = ax2.transAxes, horizontalalignment = 'center', fontsize = f)
    plt.text(1.025, 0.5, '200-400 km', transform = ax2.transAxes, verticalalignment = 'center', rotation = 'vertical', fontsize = f)

    ax4 = plt.subplot(424, sharex = ax, sharey = ax)
    plt.imshow(dusk[1], extent = [0, 360, -90, 90], origin = 'lower', interpolation = 'none', 
           cmap = my_cmap, norm = colors.Normalize(vmin = vmin, vmax = vmax), aspect = 'auto')
    plt.tick_params(axis = 'both', direction = 'in', bottom = True, top = True, right = True, left = True, length = 5.0)
    plt.tick_params(which = 'minor', axis = 'both', direction = 'in', bottom = True, top = True, right = True, left = True, length = 2.0)
    plt.xticks(visible = False)
    plt.yticks(visible = False)
    plt.text(1.025, 0.5, '400-600 km', transform = ax4.transAxes, verticalalignment = 'center', rotation = 'vertical', fontsize = f)

    ax6 = plt.subplot(426, sharex = ax, sharey = ax)
    plt.imshow(dusk[2], extent = [0, 360, -90, 90], origin = 'lower', interpolation = 'none', 
           cmap = my_cmap, norm = colors.Normalize(vmin = vmin, vmax = vmax), aspect = 'auto')
    plt.tick_params(axis = 'both', direction = 'in', bottom = True, top = True, right = True, left = True, length = 5.0)
    plt.tick_params(which = 'minor', axis = 'both', direction = 'in', bottom = True, top = True, right = True, left = True, length = 2.0)
    plt.xticks(visible = False)
    plt.yticks(visible = False)
    plt.text(1.025, 0.5, '600-800 km', transform = ax6.transAxes, verticalalignment = 'center', rotation = 'vertical', fontsize = f)

    ax8 = plt.subplot(428, sharex = ax, sharey = ax)
    im = plt.imshow(dusk[3], extent = [0, 360, -90, 90], origin = 'lower', interpolation = 'none', 
           cmap = my_cmap, norm = colors.Normalize(vmin = vmin, vmax = vmax), aspect = 'auto')
    plt.tick_params(axis = 'both', direction = 'in', bottom = True, top = True, right = True, left = True, length = 5.0)
    plt.tick_params(which = 'minor', axis = 'both', direction = 'in', bottom = True, top = True, right = True, left = True, length = 2.0)
    plt.yticks(visible = False)
    plt.xlabel('Longitude ($\degree$)', fontsize = f)
    plt.text(1.025, 0.5, '800-1000 km', transform = ax8.transAxes, verticalalignment = 'center', rotation = 'vertical', fontsize = f)

    fig.subplots_adjust(right = 0.8)
    cbar_ax = fig.add_axes([0.86, 0.22, 0.04, 0.55])
    cbar = fig.colorbar(im, cax = cbar_ax, extend = 'both')
    cbar.set_label('$\mathrm{B_r}$ (nT)', fontsize = f)
    cbar.ax.tick_params(labelsize = f)

    plt.savefig(name + '.pdf', bbox_inches = 'tight')
    
    deltaB = []

    for i in range(4):
        deltaB.append(isa.shifting.shifting_technique(np.array(model[i]), np.array(dawn[i]), region = maps, N = 20))
    
    for i in range(4):
        deltaB.append(isa.shifting.shifting_technique(np.array(model[i]), np.array(dusk[i]), region = maps, N = 20))

    x = np.linspace(-10.0, 10.0, 41)
    popt = []
    perr = []
    
    for i in range(8):
        popt.append(isa.fit.gauss_fit(x, deltaB[i])[0])
        perr.append(isa.fit.gauss_fit(x, deltaB[i])[2])
    
    mins = np.array(popt)[:,2]
    error = np.array(perr)[:,2]
    
    fig, axes = plt.subplots(4, 2, figsize = [9, 8], sharex = True)
    plt.subplots_adjust(hspace = 0.0)

    p0 = plt.subplot(421)
    plt.plot(x, isa.fit.gauss_function(x, *np.array(popt)[0,:]), color = 'crimson'), \
        plt.plot(x, np.array(deltaB)[0], '.', color = 'dodgerblue')
    plt.tick_params(axis = 'both', direction = 'in', bottom = True, top = True, left = True, right = True, length = 5.0)
    plt.ylabel("$\Delta B'$ (nT)", fontsize = f2)
    plt.xticks(visible = False)
    plt.text(0.5, 0.85, '200-400 km', transform = p0.transAxes, horizontalalignment = 'center', fontsize = f2)
    plt.text(0.5, 1.1, 'Dawn-side', transform = p0.transAxes, horizontalalignment = 'center', fontsize = f)

    p1 = plt.subplot(423, sharex = p0)
    plt.plot(x, isa.fit.gauss_function(x, *np.array(popt)[1,:]), color = 'crimson'), \
        plt.plot(x, np.array(deltaB)[1], '.', color = 'dodgerblue')
    plt.tick_params(axis = 'both', direction = 'in', bottom = True, top = True, left = True, right = True, length = 5.0)
    plt.ylabel("$\Delta B'$ (nT)", fontsize = f2)
    plt.xticks(visible = False)
    plt.text(0.5, 0.85, '400-600 km', transform = p1.transAxes, horizontalalignment = 'center', fontsize = f2)

    p2 = plt.subplot(425, sharex = p0)
    plt.plot(x, isa.fit.gauss_function(x, *np.array(popt)[2,:]), color = 'crimson'), \
        plt.plot(x, np.array(deltaB)[2], '.', color = 'dodgerblue')
    plt.tick_params(axis = 'both', direction = 'in', bottom = True, top = True, left = True, right = True, length = 5.0)
    plt.ylabel("$\Delta B'$ (nT)", fontsize = f2)
    plt.xticks(visible = False)
    plt.text(0.5, 0.85, '600-800 km', transform = p2.transAxes, horizontalalignment = 'center', fontsize = f2)

    p3 = plt.subplot(427, sharex = p0)
    plt.plot(x, isa.fit.gauss_function(x, *np.array(popt)[3,:]), color = 'crimson'), \
        plt.plot(x, np.array(deltaB)[3], '.', color = 'dodgerblue')
    plt.tick_params(axis = 'both', direction = 'in', bottom = True, top = True, left = True, right = True, length = 5.0)
    plt.ylabel("$\Delta B'$ (nT)", fontsize = f2)
    plt.xlabel('Shift value ($\degree$)', fontsize = f2)
    plt.text(0.5, 0.85, '800-1000 km', transform = p3.transAxes, horizontalalignment = 'center', fontsize = f2)

    p4 = plt.subplot(422, sharex = p0)
    plt.plot(x, isa.fit.gauss_function(x, *np.array(popt)[4,:]), color = 'crimson'), \
        plt.plot(x, np.array(deltaB)[4], '.', color = 'dodgerblue')
    plt.tick_params(axis = 'both', direction = 'in', bottom = True, top = True, left = True, right = True, length = 5.0)
    plt.xticks(visible = False)
    plt.text(0.5, 0.85, '200-400 km', transform = p4.transAxes, horizontalalignment = 'center', fontsize = f2)
    plt.text(0.5, 1.1, 'Dusk-side', transform = p4.transAxes, horizontalalignment = 'center', fontsize = f)

    p5 = plt.subplot(424, sharex = p0)
    plt.plot(x, isa.fit.gauss_function(x, *np.array(popt)[5,:]), color = 'crimson'), \
        plt.plot(x, np.array(deltaB)[5], '.', color = 'dodgerblue')
    plt.tick_params(axis = 'both', direction = 'in', bottom = True, top = True, left = True, right = True, length = 5.0)
    plt.xticks(visible = False)
    plt.text(0.5, 0.85, '400-600 km', transform = p5.transAxes, horizontalalignment = 'center', fontsize = f2)

    p6 = plt.subplot(426, sharex = p0)
    plt.plot(x, isa.fit.gauss_function(x, *np.array(popt)[6,:]), color = 'crimson'), \
        plt.plot(x, np.array(deltaB)[6], '.', color = 'dodgerblue')
    plt.tick_params(axis = 'both', direction = 'in', bottom = True, top = True, left = True, right = True, length = 5.0)
    plt.xticks(visible = False)
    plt.text(0.5, 0.85, '600-800 km', transform = p6.transAxes, horizontalalignment = 'center', fontsize = f2)

    p7 = plt.subplot(428, sharex = p0)
    plt.plot(x, isa.fit.gauss_function(x, *np.array(popt)[7,:]), color = 'crimson'), \
        plt.plot(x, np.array(deltaB)[7], '.', color = 'dodgerblue')
    plt.tick_params(axis = 'both', direction = 'in', bottom = True, top = True, left = True, right = True, length = 5.0)
    plt.xlabel('Shift value ($\degree$)', fontsize = f2)
    plt.text(0.5, 0.85, '800-1000 km', transform = p7.transAxes, horizontalalignment = 'center', fontsize = f2)
    plt.xticks(fontsize = f2)

    plt.legend(['Gaussian fit', 'Shift data'], bbox_to_anchor = (0.5, 0.05), loc = 10, fontsize = f2, \
               bbox_transform = fig.transFigure)

    plt.savefig('sigma_' + name + '.pdf', bbox_inches = 'tight')
    
    return mins, error


def shifting_ribbon(model, dawn, dusk, lat, name = None, binsize = 0.5, thickness = 1.0, plot = False):
    """ Shifting technique applied to ribbon regions. A ribbon is defined by a very narrow latitude range and an extense longitude range. Used for MAVEN data (4 altitude ranges as input).

   Parameters:
        model: list of 4 arrays
            Model magnetic field for 4 altitude ranges.

        dawn: list of 4 arrays
            Observed magnetic field at dawn-side for 4 altitude ranges.
            
        dusk: list of 4 arrays
            Observed magnetic field at dusk-side for 4 altitude ranges.
        
        lat: [c, d] array
            Latitude limits of the anomaly.
        
        name: string, optional
            The generated figures will be named after this string.
        
         binsize: float, optional
            The bin size of the maps in degrees (bin size of lon = bin size of lat). Default is 0.5.
        
        thickness: float, optional
            The positive latitude range around lat value. Default is 1.0 degree. 
        
        plot: bool, optional
            If plot is True, then a figure is generated. Default is False.
            
    Returns:
        mins: array
        error: array
    """
    latitudes = [lat - thickness, lat + thickness]
    zoom, maps = isa.mapsetup.zoom_in(model[0], [0, 360], latitudes, binsize)

    f = 12
    f2 = 10
    
    deltaB = []

    for i in range(4):
        deltaB.append(isa.shifting.shifting_technique(np.array(model[i]), np.array(dawn[i]), region = maps, N = 20))
    
    for i in range(4):
        deltaB.append(isa.shifting.shifting_technique(np.array(model[i]), np.array(dusk[i]), region = maps, N = 20))

    x = np.linspace(-10.0, 10.0, 41)
    popt = []
    perr = []
    
    for i in range(8):
        popt.append(isa.fit.gauss_fit(x, deltaB[i])[0])
        perr.append(isa.fit.gauss_fit(x, deltaB[i])[2])
    
    mins = np.array(popt)[:,2]
    error = np.array(perr)[:,2]
    
    if plot == True:
        fig, axes = plt.subplots(4, 2, figsize = [9, 8], sharex = True)
        plt.subplots_adjust(hspace = 0.0)

        p0 = plt.subplot(421)
        plt.plot(x, isa.fit.gauss_function(x, *np.array(popt)[0,:]), color = 'crimson'), \
            plt.plot(x, np.array(deltaB)[0], '.', color = 'dodgerblue')
        plt.tick_params(axis = 'both', direction = 'in', bottom = True, top = True, left = True, right = True, length = 5.0)
        plt.ylabel("$\Delta B'$ (nT)", fontsize = f2)
        plt.xticks(visible = False)
        plt.text(0.5, 0.85, '200-400 km', transform = p0.transAxes, horizontalalignment = 'center', fontsize = f2)
        plt.text(0.5, 1.1, 'Dawn-side', transform = p0.transAxes, horizontalalignment = 'center', fontsize = f)

        p1 = plt.subplot(423, sharex = p0)
        plt.plot(x, isa.fit.gauss_function(x, *np.array(popt)[1,:]), color = 'crimson'), \
            plt.plot(x, np.array(deltaB)[1], '.', color = 'dodgerblue')
        plt.tick_params(axis = 'both', direction = 'in', bottom = True, top = True, left = True, right = True, length = 5.0)
        plt.ylabel("$\Delta B'$ (nT)", fontsize = f2)
        plt.xticks(visible = False)
        plt.text(0.5, 0.85, '400-600 km', transform = p1.transAxes, horizontalalignment = 'center', fontsize = f2)

        p2 = plt.subplot(425, sharex = p0)
        plt.plot(x, isa.fit.gauss_function(x, *np.array(popt)[2,:]), color = 'crimson'), \
            plt.plot(x, np.array(deltaB)[2], '.', color = 'dodgerblue')
        plt.tick_params(axis = 'both', direction = 'in', bottom = True, top = True, left = True, right = True, length = 5.0)
        plt.ylabel("$\Delta B'$ (nT)", fontsize = f2)
        plt.xticks(visible = False)
        plt.text(0.5, 0.85, '600-800 km', transform = p2.transAxes, horizontalalignment = 'center', fontsize = f2)

        p3 = plt.subplot(427, sharex = p0)
        plt.plot(x, isa.fit.gauss_function(x, *np.array(popt)[3,:]), color = 'crimson'), \
            plt.plot(x, np.array(deltaB)[3], '.', color = 'dodgerblue')
        plt.tick_params(axis = 'both', direction = 'in', bottom = True, top = True, left = True, right = True, length = 5.0)
        plt.ylabel("$\Delta B'$ (nT)", fontsize = f2)
        plt.xlabel('Shift value ($\degree$)', fontsize = f2)
        plt.text(0.5, 0.85, '800-1000 km', transform = p3.transAxes, horizontalalignment = 'center', fontsize = f2)

        p4 = plt.subplot(422, sharex = p0)
        plt.plot(x, isa.fit.gauss_function(x, *np.array(popt)[4,:]), color = 'crimson'), \
            plt.plot(x, np.array(deltaB)[4], '.', color = 'dodgerblue')
        plt.tick_params(axis = 'both', direction = 'in', bottom = True, top = True, left = True, right = True, length = 5.0)
        plt.xticks(visible = False)
        plt.text(0.5, 0.85, '200-400 km', transform = p4.transAxes, horizontalalignment = 'center', fontsize = f2)
        plt.text(0.5, 1.1, 'Dusk-side', transform = p4.transAxes, horizontalalignment = 'center', fontsize = f)

        p5 = plt.subplot(424, sharex = p0)
        plt.plot(x, isa.fit.gauss_function(x, *np.array(popt)[5,:]), color = 'crimson'), \
            plt.plot(x, np.array(deltaB)[5], '.', color = 'dodgerblue')
        plt.tick_params(axis = 'both', direction = 'in', bottom = True, top = True, left = True, right = True, length = 5.0)
        plt.xticks(visible = False)
        plt.text(0.5, 0.85, '400-600 km', transform = p5.transAxes, horizontalalignment = 'center', fontsize = f2)

        p6 = plt.subplot(426, sharex = p0)
        plt.plot(x, isa.fit.gauss_function(x, *np.array(popt)[6,:]), color = 'crimson'), \
            plt.plot(x, np.array(deltaB)[6], '.', color = 'dodgerblue')
        plt.tick_params(axis = 'both', direction = 'in', bottom = True, top = True, left = True, right = True, length = 5.0)
        plt.xticks(visible = False)
        plt.text(0.5, 0.85, '600-800 km', transform = p6.transAxes, horizontalalignment = 'center', fontsize = f2)

        p7 = plt.subplot(428, sharex = p0)
        plt.plot(x, isa.fit.gauss_function(x, *np.array(popt)[7,:]), color = 'crimson'), \
            plt.plot(x, np.array(deltaB)[7], '.', color = 'dodgerblue')
        plt.tick_params(axis = 'both', direction = 'in', bottom = True, top = True, left = True, right = True, length = 5.0)
        plt.xlabel('Shift value ($\degree$)', fontsize = f2)
        plt.text(0.5, 0.85, '800-1000 km', transform = p7.transAxes, horizontalalignment = 'center', fontsize = f2)
        plt.xticks(fontsize = f2)

        plt.legend(['Gaussian fit', 'Shift data'], bbox_to_anchor = (0.5, 0.05), loc = 10, fontsize = f2, \
               bbox_transform = fig.transFigure)
        
        if name is not None:
            plt.savefig('sigma_' + name + '.pdf', bbox_inches = 'tight')
        else:
            plt.savefig('sigma_latxx.pdf', bbox_inches = 'tight')
    
    return mins, error