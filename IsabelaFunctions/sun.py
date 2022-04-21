"""
    Functions useful for the Sun project.
"""

import numpy as np
from tqdm import tqdm
import IsabelaFunctions as isa
import scipy.interpolate as interp
import scipy.integrate as integrate


def filling_factors(B, sat_fac = 250., min_spot = 60., max_spot = 700., umb = 0.2, pen = 0.8):
    """
    Calculation of filling factors of faculae and spots based on an input magnetic field map.
    
    
    Parameters
    ----------
    B : 2D-array
        The magnetic field at each pixel.
    sat_fac : float, optional
        The saturation threshold for the magnetic field of the faculae, in G. The default is 250.
    min_spots : float, optional
        The lower cut-off value for the magnetic field of the spots, in G. The default is 60.
    max_spots : float, optional
        The upper cut-off value for the magnetic field of the spots, in G. The default is 700.
    umb : float, optional
        The ratio of umbra in the spots, between 0 and 1. The default is 0.2.
    pen : float, optional
        The ratio of penumbra in the spots, between 0 and 1. The default is 0.8.

    Returns
    -------
    3 arrays, with the same shape as B, containing the filling factors of the faculae, umbra, and penumbra, respectively.

    """   
    ff_spots = np.zeros_like(B)
    between = np.logical_and(abs(B) < max_spot, abs(B) >= min_spot)
    
    ff_spots[abs(B) >= max_spot] = 1.
    ff_spots[between] = (abs(B[between]) - min_spot) / (max_spot - min_spot)
    
    ff_umbra = umb * ff_spots
    ff_penumbra = pen * ff_spots
    
    ff_faculae = np.zeros_like(B)
    under = abs(B) < sat_fac
    
    ff_faculae[abs(B) >= sat_fac] = 1.
    ff_faculae[under] = abs(B[under]) / sat_fac
    
    # Ignore bins already covered by spots, not allowing "mixed" bins
    ff_faculae[ff_spots != 0.] = 0.
    
    # Ignoring bins outside full-disc magnetogram
    ff_faculae[B == 0] = np.nan
    ff_umbra[B == 0] = np.nan
    ff_penumbra[B == 0] = np.nan

    return ff_faculae, ff_umbra, ff_penumbra    


def visible(y_obs, y_pos, delta_lambda):
    """
    The cosine of the heliocentric angle (mu = cos(theta)).
    
    """
    return (np.sin(np.deg2rad(y_obs)) * np.sin(np.deg2rad(y_pos)) + \
                 np.cos(np.deg2rad(y_obs)) * np.cos(np.deg2rad(y_pos)) * np.cos(np.deg2rad(delta_lambda)))


def irradiance_from_Br(B, ff_faculae, ff_umbra, ff_penumbra, interp_qs, interp_fac, \
                    interp_um, interp_penum, x_obs = None, y_obs = None):
    """
    Calculation of solar spectral irradiance for 1 wavelength, for multiple days, from a magnetic field map.
    
    
    Parameters
    ----------
    B : 3D-array
        The magnetic field at each pixel. Must be in the shape (ndays, nlat, nlon).
    ff_fac : 3D array
        Filling factors of the faculae, same shape as B.
    ff_umbra : 3D array
        Filling factors of the umbra, same shape as B.
    ff_penumbra : 3D array
        Filling factors of the penumbra, same shape as B.
    interp_qs : function
        Interpolation function (flux) for the quiet Sun.
    interp_fac : function
        Interpolation function (flux) for the faculae.
    interp_um : function
        Interpolation function (flux) for the umbra.
    interp_penum : function
        Interpolation function (flux) for the penumbra.
    x_obs : float, optional
        The longitude of the observer. The default is 0.
    y_obs : float, optional
        The latitude of the observer. The default is 0.

    Returns
    -------
    An array with length (nday).
  
    """
    nday, nlat, nlon = B.shape
    
    if x_obs is None:
        x_obs = np.zeros(nday)
        
    if y_obs is None:
        y_obs = np.zeros(nday)
    
    x = np.linspace(1., 360., 360)
    y = np.linspace(-90., 90., 181)
    x_not_used, y_pos = np.meshgrid(x, y)    
    
    irradiance = np.zeros(nday)
    mask_qs = np.zeros_like(B)
    mask_fac = np.zeros_like(B)
    mask_umb = np.zeros_like(B)
    mask_pen = np.zeros_like(B)
    vis = np.zeros_like(B)
    
    for i in tqdm(range(nday)):
        
        # Check if the whole map is nan
        if np.all(np.isnan(B[i])):
            continue
        
        # Take the (~) central colatitude of the map y0, and calculate the index of the central longitude x0
        y0 = 1. * B[i][90]
        lon = np.linspace(0, 359, 360)
        
        while (y0 == y0)[-1]:
            lon = np.roll(lon, 1)
            y0 = np.roll(y0, 1)
        
        equals = y0 == y0
        N = sum(equals)
        ind = N // 2
        
        if (N % 2) == 0:
            ind2 = ind - 1
            x1 = lon[np.where(equals)[0][ind]]
            x2 = lon[np.where(equals)[0][ind2]]
            
            if np.logical_and(x1 == 0.0, x2 == 359.):
                x0 = 359.5
            else:
                x0 = (x1 + x2) / 2
        else:
            x0 = lon[np.where(equals)[0][ind]]
        
        # Create the arrays of grid (I had to flip it to be correct)
        x_lon = np.flip((np.linspace(0, 359, 360) + x0) % 360)
        x_pos, y_not_used = np.meshgrid(x_lon, y)
        
        delta_lambda = abs(x_pos - x_obs[i])
        vis[i] = isa.sun.visible(y_obs[i], y_pos, delta_lambda)
        
        # an observer will see only half of the sphere
        vis[vis < 0.] = 0.
        
        # solid angle of each pixel in the visible disk, accounts for the reduction in the pixel area with latitude
        mask_qs[i] = vis[i] * np.cos(np.deg2rad(y_pos))
        
        # filling factors multiplied with solid angle of the pixel
        mask_fac[i] = mask_qs[i] * ff_faculae[i]
        mask_umb[i] = mask_qs[i] * ff_umbra[i]
        mask_pen[i] = mask_qs[i] * ff_penumbra[i]
        
        # Calculation of irradiance                
        irr_qs = np.nansum(mask_qs[i] * interp_qs(vis[i]))
        irr_fac = np.nansum(mask_fac[i] * (interp_fac(vis[i]) - interp_qs(vis[i])))
        irr_umb = np.nansum(mask_umb[i] * (interp_um(vis[i]) - interp_qs(vis[i])))
        irr_pen = np.nansum(mask_pen[i] * (interp_penum(vis[i]) - interp_qs(vis[i])))
        
        irradiance[i] = irr_qs + irr_fac + irr_umb + irr_pen

    irradiance[irradiance == 0.] = np.nan
    
    return irradiance


def irradiance_full_surface(ff_faculae, ff_umbra, ff_penumbra, interp_qs, interp_fac, \
                    interp_um, interp_penum, x_obs = 0., y_obs = 0.):
    """
    Calculation of irradiance from a full surface map (e.g. from simulation), from filling factors (no Br).
    Contribution by Sowmya Krishnamurthy.

    Parameters
    ----------
    ff_fac : 3D array
        Filling factors of the faculae, in the shape (nday, nlat, nlon).
    ff_umbra : 3D array
        Filling factors of the umbra, same shape as ff_fac.
    ff_penumbra : 3D array
        Filling factors of the penumbra, same shape as ff_fac.
    interp_qs : function
        Interpolation function (flux) for the quiet Sun.
    interp_fac : function
        Interpolation function (flux) for the faculae.
    interp_um : function
        Interpolation function (flux) for the umbra.
    interp_penum : function
        Interpolation function (flux) for the penumbra.
    x_obs : float, optional
        The longitude of the observer. The default is 0.
    y_obs : float, optional
        The latitude of the observer. The default is 0.

    Returns
    -------
    An array with length (nday).

    """
    nday, nlat, nlon = ff_faculae.shape
    days = np.linspace(0, nday-1, nday)
    
    # Defining the visible disk
    x = np.linspace(1., 360., nlon)
    y = np.linspace(-90., 90., nlat)
    x_pos, y_pos = np.meshgrid(x, y)    
    
    delta_lambda = abs(x_pos - x_obs)
    vis = isa.sun.visible(y_obs, y_pos, delta_lambda)
    vis[vis < 0.] = 0.                          # an observer will see only half of the sphere
    vis_corr = vis * np.cos(np.deg2rad(y_pos))  # solid angle of each pixel in the visible disk
    
    # Calculation of irradiance
    irradiance = np.zeros(nday)
    irr_qs = np.nansum(vis_corr * interp_qs(vis))
    
    for i in range(nday):  
        
        # Rotation by 13.38 degrees per day
        lonshift = 13.38 * days[i]

        # Rotated filling factors multiplied with solid angle of the pixel
        mask_fac = vis_corr * isa.mapsetup.shift_map_longitude(ff_faculae[i], lonshift)
        mask_umb = vis_corr * isa.mapsetup.shift_map_longitude(ff_umbra[i], lonshift)
        mask_pen = vis_corr * isa.mapsetup.shift_map_longitude(ff_penumbra[i], lonshift)
        
        # Calculation of irradiance                
        delta_fac = np.nansum(mask_fac * (interp_fac(vis) - interp_qs(vis)))
        delta_umb = np.nansum(mask_umb * (interp_um(vis) - interp_qs(vis)))
        delta_pen = np.nansum(mask_pen * (interp_penum(vis) - interp_qs(vis)))        
                       
        irradiance[i] = irr_qs + delta_fac + delta_umb + delta_pen

    irradiance[irradiance == 0.] = np.nan
    
    return irradiance


def s_index(ff_faculae, interp_qsh, interp_qsk, interp_qsr, interp_qsv, interp_fach, \
            interp_fack, interp_facr, interp_facv, x_obs = 0., y_obs = 0.):
    """
    Calculation of S-index from a full surface map (e.g. from simulation), from filling factors (no Br).
    Contribution by Sowmya Krishnamurthy.

    Parameters
    ----------
    ff_faculae : 3D array
        Filling factors of the faculae, in the shape (nday, nlat, nlon).
    interp_qsh : function
        Interpolation function (flux x mu) of the quiet Sun, for Ca II H line.
    interp_qsk : function
        Interpolation function (flux x mu) of the quiet Sun, for Ca II K line.
    interp_qsr : function
        Interpolation function (flux x mu) of the quiet Sun, for Ca II R band.
    interp_qsv : function
        Interpolation function (flux x mu) of the quiet Sun, for Ca II V band.
    interp_fach : function
        Interpolation function (flux x mu) of the faculae, for Ca II H line.
    interp_fack : function
        Interpolation function (flux x mu) of the faculae, for Ca II K line.
    interp_facr : function
        Interpolation function (flux x mu) of the faculae, for Ca II R band.
    interp_facv : function
        Interpolation function (flux x mu) of the faculae, for Ca II V band.
    x_obs : float, optional
        The longitude of the observer. The default is 0.
    y_obs : float, optional
        The latitude of the observer. The default is 0.

    Returns
    -------
    None.

    """
    nday, nlat, nlon = ff_faculae.shape
    days = np.linspace(0, nday-1, nday)
    
    # Defining the visible disk
    x = np.linspace(1., 360., nlon)
    y = np.linspace(-90., 90., nlat)
    x_pos, y_pos = np.meshgrid(x, y)    
    
    delta_lambda = abs(x_pos - x_obs)
    vis = isa.sun.visible(y_obs, y_pos, delta_lambda)
    vis[vis < 0.] = 0.                          # an observer will see only half of the sphere
    vis_corr = vis * np.cos(np.deg2rad(y_pos))  # solid angle of each pixel in the visible disk

    s_index = np.zeros(nday)
    
    for i in range(nday):  
        
        # Rotation by 13.38 degrees per day
        lonshift = 13.38 * days[i]

        # Rotated filling factors multiplied with solid angle of the pixel
        mask_fac = vis_corr * isa.mapsetup.shift_map_longitude(ff_faculae[i], lonshift)
        
        # Calculation of the contribution of each band  
        qsh = np.nansum(vis_corr * interp_qsh(vis))
        qsk = np.nansum(vis_corr * interp_qsk(vis))
        qsr = np.nansum(vis_corr * interp_qsr(vis))
        qsv = np.nansum(vis_corr * interp_qsv(vis))
        
        # The factor 3.2 accounts for the expansion of faculae at about 550 km above the photosphere, \
        #    where H and K are formed                
        delta_fach = np.nansum(mask_fac * 3.2 * (interp_fach(vis) - interp_qsh(vis)))
        delta_fack = np.nansum(mask_fac * 3.2 * (interp_fack(vis) - interp_qsk(vis)))
        # R and V form at the photosphere, so we do not need to take the expansion into consideration
        delta_facr = np.nansum(mask_fac * (interp_facr(vis) - interp_qsr(vis)))
        delta_facv = np.nansum(mask_fac * (interp_facv(vis) - interp_qsv(vis)))
        
        # Calculation of the flux
        fluxh = qsh + delta_fach
        fluxk = qsk + delta_fack
        fluxr = qsr + delta_facr
        fluxv = qsv + delta_facv
        
        # Calculation of the S-index
        fluxratio = (fluxh + fluxk) / (fluxr + fluxv)

        s_index[i] = 2.53 * 8. * fluxratio
        
    return s_index


def moving_average(a, nday, norm = True):
    """ Calculates the linear moving average for a data set.
    
   Parameters:
        a : array
            The array containing the data.
        nday : integer
            The window in which the mean is taken, in days.
        norm : bool, optional
            If True, it returns the array normalized by the average. 
            If false, it just subtracts the average from the array. Default is True.
            
    Returns:
        An array the same size as the input.
    """
    a_averaged = np.zeros_like(a)
    half_window = nday // 2
    
    for i in range(half_window, len(a) - half_window):
        d1 = i - half_window
        d2 = i + half_window
        mean = np.nanmean(a[d1:d2])
        
        a_averaged[i] = a[i] - mean 
        
        if norm:
            a_averaged[i] /= mean    

    a_averaged[:half_window] = np.nan
    a_averaged[-half_window:] = np.nan
    
    return a_averaged


def virgo(wavelengths, irradiance):
    """ Contribution by Nina Nèmec
        Calculation of irradiances within the VIRGO/SPM channels (blue, green and red).
    
   Parameters:
        wavelengths: array
            The array containing the wavelengths.
        irradiance: array
            The array containing the irradiance. Must have the same size as wavelengths.
                    
    Returns:
        The irradiances in all three VIRGO channels: blue, green and red, respectively.
    """

    # Make Gaussian functions for the different VIRGO channels
    # Blue channel
    b = 402                 # central wavelength
    a = 0.437               # maximum of transmissivity at central wavelength
    fwhm = 5                # full width half maximum (nm)
    sigma = fwhm /2.3548
    
    x_b = np.linspace(380, 420, num = 300)
    y_b = a*np.exp(-(x_b-b)**2./(2*sigma**2.)) # Gaussian function with 'a' being the maximum
    
    # Green channel
    b = 500
    a = 0.417
    fwhm = 5 
    sigma = fwhm /2.3548
    
    x_g = np.linspace(480, 520, num = 300)
    y_g = a*np.exp(-(x_g-b)**2./(2*sigma**2.))
    
    # Red channel
    b = 802
    a = 0.843
    fwhm = 5 
    sigma = fwhm /2.3548
    
    x_r = np.linspace(780, 820, num = 300)
    y_r = a*np.exp(-(x_r-b)**2./(2*sigma**2.))
    
    # Create interpolation function
    f = interp.interp1d(wavelengths, irradiance)
    
    blue_new = f(x_b)
    green_new = f(x_g)
    red_new = f(x_r)

    # Multiply irradiance with filter

    blue = blue_new * y_b
    green = green_new * y_g
    red = red_new * y_r
    
    # Integrate
    
    irr_blue = integrate.trapz(blue, x_b)
    irr_green = integrate.trapz(green, x_g)
    irr_red = integrate.trapz(red, x_r)
    
    return irr_blue, irr_green, irr_red


def residuals(x1, x2):
    """ Calculates the residuals between two data sets.
    
   Parameters:
        x1, x2: array
            The arrays containing the data. They must have the same dimensions.
                    
    Returns:
        An array of the same size of x1 containing the residuals.
        
        The total of the residuals (float).
    """
    res = x1 - x2
    count = np.count_nonzero(~np.isnan(res))
    res_total = np.nansum(res) / count
    
    return res, res_total


def flux_map(flux_quiet, flux_faculae, flux_umbra, flux_penumbra, x_obs):
    """
    NOT USED.
    Computes the general irradiance of faculae and sunspots in a 11-rings circle, 
    corrected for the corresponding solid angles of the pixels.

    Parameters
    ----------
    flux_quiet : 1D array of length n-1.
        Flux of the quiet Sun per ring.
    flux_faculae : 1D array of length n-1.
        Flux of the faculae per ring.
    flux_umbra : 1D array of length n-1.
        Flux of the umbra per ring.
    flux_penumbra : 1D array of length n-1.
        Flux of the penumbra per ring.
 
    Returns
    -------
    Flux maps of the quiet Sun, faculae, umbra, and penumbra, respectively.

    """
    # Defining the solid angles
    mu = [1, .95, .85, .75, .65, .55, .45, .35, .25, .15, .075, 0.0]    # The borders of the rings
    
    omega_sun = 6.7996873e-5            # This is used to scale it to 1 AU
    
    # See Methods in Nèmec et al. 2020
    # theta = arccos(mu)
    # omega = (theta_up - theta_low) * omega_sun
    omega = [6.6296969e-06, 1.2239433e-05, 1.0879501e-05 , 9.5195652e-06  , 8.1596210e-06, \
             6.7996889e-06, 5.4397487e-06, 4.0798125e-06, 2.7198764e-06, 1.1474451e-06, 3.8248306e-07]

    # Defining the grid
    x = np.linspace(1., 360., 360)
    y = np.linspace(-90., 90., 181)
    x_pos, y_pos = np.meshgrid(x, y)    

    delta_lambda = abs(x_pos - x_obs)
    vis = visible(0., y_pos, delta_lambda)
    vis[vis < 0.] = 0.

    # Calculating the irradiance of quiet sun, faculae and sunspots
    flux_qs_conv = 1. * vis
    flux_faculae_conv = 1. * vis
    flux_umbra_conv = 1. * vis
    flux_penumbra_conv = 1. * vis
    
    for i in range(len(mu) - 1):
        upp = mu[i]
        low = mu[i+1]
        my_range = np.logical_and(vis <= upp, vis > low)
        
        flux_qs_conv[my_range] = flux_quiet[i]
        flux_faculae_conv[my_range] = flux_faculae[i]
        flux_umbra_conv[my_range] = flux_umbra[i]
        flux_penumbra_conv[my_range] = flux_penumbra[i]
    
    # Defining the contrast
    quiet_sun = np.sum(flux_quiet * omega)
    delta_faculae = (flux_faculae_conv - flux_qs_conv) * omega_sun
    delta_umbra = (flux_umbra_conv - flux_qs_conv) * omega_sun
    delta_penumbra = (flux_penumbra_conv - flux_qs_conv) * omega_sun
    
    return quiet_sun, delta_faculae, delta_umbra, delta_penumbra


def varying_ff(B, interp_qs, interp_fac, interp_um, interp_penum, wavelengths,\
               fac_range, spot_min_range, spot_max_range, x_obs = None, y_obs = None):
    """
    This script does not work. It requires a big amount of space.
    """
    
    nday, nlat, nlon = B.shape
    
    # Preparing some arrays
    step1 = 10.
    fac_len = int((fac_range[1] - fac_range[0]) / step1 + 1.0)
    spot_min_len = int((spot_min_range[1] - spot_min_range[0]) / step1 + 1.0)
    spot_max_len = int((spot_max_range[1] - spot_max_range[0]) / step1 + 1.0)
    
    fac = np.linspace(fac_range[0], fac_range[1], fac_len)
    spot_min = np.linspace(spot_min_range[0], spot_min_range[1], spot_min_len)
    spot_max = np.linspace(spot_max_range[0], spot_max_range[1], spot_max_len)
    #matrix = np.empty((fac_len, spot_min_len, spot_max_len))
    
    ff_faculae = np.empty((nday, nlat, nlon, fac_len, spot_min_len, spot_max_len))
    ff_umbra = np.empty_like(ff_faculae)
    ff_penumbra = np.empty_like(ff_faculae)
    
    # Calculation of the filling factors
    for i in range(nday):
        for j in range(fac_len):
            for k in range(spot_min_len):
                for l in range(spot_max_len):
                    ff_faculae[i, :, :, j, k, l], ff_umbra[i, :, :, j, k, l], ff_penumbra[i, :, :, j, k, l] = \
                    isa.sun.filling_factors(B[i], fac[j], spot_min[k], spot_max[l])
                
    # Selecting only the regions where we have data    
    B[B == 0.0] = np.nan
    
    # Calculation of irradiance
    irradiance = np.zeros((nday, fac_len, spot_min_len, spot_max_len, len(wavelengths)))
    
    for j in range(fac_len):
        for k in range(spot_min_len):
            for l in range(spot_max_len):
                for m in range(len(wavelengths)):
                    irradiance[:, j, k, l, m] = isa.sun.ss_irradiance(B, wavelengths[m], ff_faculae[:, :, :, j, k, l], \
                    ff_umbra[:, :, :, j, k, l], ff_penumbra[:, :, :, j, k, l], \
                    interp_qs[m], interp_fac[m], interp_um[m], interp_penum[m], x_obs, y_obs)

    return irradiance


def angle_to_period(angle):
    """ Converts an angle within the ecliptic to its correspondent day in the past
    and in the future. Used in the irradiance extrapolation method (Thiemann et al., 2017).
    
   Parameters:
        angle: float
            The angle.
                    
    Returns:
        The day in the past (t1) and in the future (t2).
    """   
    t1 = angle * 27.27 / 360.
    t2 = 27.27 - t1
    
    return t1, t2


########################################################################################