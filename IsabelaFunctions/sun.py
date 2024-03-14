"""
    Functions useful for the Sun project.
"""

import numpy as np
import IsabelaFunctions as isa
import scipy.interpolate as interp
import scipy.integrate as integrate
import scipy.io as io
import pandas as pd
import julian
from datetime import datetime as dt
import subprocess as subp
from scipy.interpolate import RegularGridInterpolator


def read_filling_factors(file, cadence = 4):
    """
    Function for reading the filling factors of faculae/spots and the dates from a .npy file.

    Parameters
    ----------
    file : .npy file
        File containing the filling factors and the dates.
    cadence : int
        The cadence of the data is every 6h, so 4 measurements per day. For one measurement per day, we set cadence = 4.

    Returns
    -------
    Arrays of filling factors and dates.

    """
    loading_file = np.load(file, allow_pickle = True).item()
    dates = list(loading_file.keys())
    dates = np.array(dates)
    dates = dates[::cadence]

    data = list(loading_file.values())
    data = np.array(data)
    data = data[::cadence]
    
    # filling_factors = np.zeros((len(dates), 181, 360))
    # for i in range(len(filling_factors)):
    #     filling_factors[i] = data[i][1]
    
    # filling_factors = filling_factors[::cadence]
    # dates = dates[::cadence]
    
    return data, dates
        
        
def read_fluxes(file):
    """
    Function to read the fluxes and its parameters and to return them converted to physical units.

    Parameters
    ----------
    file : .sav file
        File with flux parameters.

    Returns
    -------
    Arrays with wavelengths, angles and fluxes of quiet Sun, faculae, umbra and penumbra.

    """
    flux = io.readsav(file)

    wavelengths_nm = flux['wav']
    angles = flux['mu_input']
    
    # The fluxes are in W/m**2/sr/nm
    flux_quiet = 1.* flux['intens_qs']        
    flux_faculae = 1.* flux['intens_fac']
    flux_umbra = 1.* flux['intens_um']
    flux_penumbra = 1.* flux['intens_pen']

    # Convert fluxes to physical units (W/m**2/nm)
    solar_radius_AU = 0.00465047
    pixel_size_at_equator = 2 * np.pi * solar_radius_AU / 360.
    solid_angle_pixel = pixel_size_at_equator**2

    flux_quiet *= solid_angle_pixel
    flux_faculae *= solid_angle_pixel
    flux_umbra *= solid_angle_pixel
    flux_penumbra *= solid_angle_pixel

    return wavelengths_nm, angles, flux_quiet, flux_faculae, flux_umbra, flux_penumbra
    

def compute_interpolations(flux, angles, number_of_wavelengths):
    interpolation_function = []
    
    for wavelength in range(number_of_wavelengths):
        interpolation_function.append(interp.interp1d(angles, flux[wavelength], fill_value = 'extrapolate'))
    
    return interpolation_function


def compute_filling_factors(B_input, sat_fac = 250., min_spot = 60., max_spot = 700., umbra = 0.2, penumbra = 0.8):
    """
    Calculation of filling factors of faculae and spots based on an input magnetic field map.
    
    Parameters
    ----------
    B_input : 2D-array
        The magnetic field at each pixel.
    sat_fac : float, optional
        The saturation threshold for the magnetic field of the faculae, in G. The default is 250.
    min_spots : float, optional
        The lower cut-off value for the magnetic field of the spots, in G. The default is 60.
    max_spots : float, optional
        The upper cut-off value for the magnetic field of the spots, in G. The default is 700.
    umbra : float, optional
        The ratio of umbra in the spots, between 0 and 1. The default is 0.2.
    penumbra : float, optional
        The ratio of penumbra in the spots, between 0 and 1. The default is 0.8.

    Returns
    -------
    3 arrays, with the same shape as B_input, containing the filling factors of the faculae, umbra, and penumbra, respectively.

    """   
    ff_spots = np.zeros_like(B_input)
    B_pixel = np.logical_and(abs(B_input) < max_spot, abs(B_input) >= min_spot)
    
    ff_spots[abs(B_input) >= max_spot] = 1.
    ff_spots[B_pixel] = (abs(B_input[B_pixel]) - min_spot) / (max_spot - min_spot)
    ff_spots[abs(B_input) < min_spot] = 0.

    ff_umbra = umbra * ff_spots
    ff_penumbra = penumbra * ff_spots
    
    ff_faculae = np.zeros_like(B_input)
    B_pixel = abs(B_input) < sat_fac
    
    ff_faculae[abs(B_input) >= sat_fac] = 1.
    ff_faculae[B_pixel] = abs(B_input[B_pixel]) / sat_fac
    
    # Ignore bins already covered by spots, not allowing "mixed" bins
    ff_faculae[ff_spots != 0.] = 0.
    
    # Ignoring bins outside full-disc magnetogram
    ff_faculae[B_input == 0] = np.nan
    ff_faculae[np.where(np.isnan(B_input))] = np.nan
    ff_umbra[B_input == 0] = np.nan
    ff_umbra[np.where(np.isnan(B_input))] = np.nan
    ff_penumbra[B_input == 0] = np.nan
    ff_penumbra[np.where(np.isnan(B_input))] = np.nan

    return ff_faculae, ff_umbra, ff_penumbra


def compute_ff_grids(B_input, B_sat = 250., B_min = 60., half_number_of_steps = 4, step_size = 5.):
    """
    Calculation of a grid of filling factors of faculae and spots based on an input magnetic field map.
    The grid will be centered on B_sat and B_max, with (half_number_of_steps * 2 + 1) steps.

    Parameters
    ----------
    B_input : 2D-array
        The magnetic field at each pixel.
    B_sat : float, optional
        The saturation threshold for the magnetic field of the faculae, in G. The default is 250.
    B_min : float, optional
        The lower cut-off value for the magnetic field of the spots, in G. The default is 60.
    half_number_of_steps : integer, optional
        The total number of steps in the grid is equal to half_number_of_steps * 2 + 1. The default is 4.
    step_size : float, optional
        The step size of the grid. The default is 5.0 G.
    
    Returns
    -------
    3 grid arrays, containing the filling factors of the faculae, umbra, and penumbra, respectively.

    """

    nday, nlat, nlon = B_input.shape
    length = half_number_of_steps * 2 + 1
    
    grid_B_sat = np.linspace(B_sat - half_number_of_steps * step_size, B_sat + half_number_of_steps * step_size, length)
    grid_B_min = np.linspace(B_min - half_number_of_steps * step_size, B_min + half_number_of_steps * step_size, length)

    ff_faculae = np.zeros((nday, nlat, nlon, length, length))
    ff_umbra = np.zeros_like(ff_faculae)
    ff_penumbra = np.zeros_like(ff_faculae)
        
    for day in range(nday):
        for i in range(length):
            for j in range(length):
                ff_faculae[day, :, :, i, j], ff_umbra[day, :, :, i, j], ff_penumbra[day, :, :, i, j] = \
                    isa.sun.compute_filling_factors(B_input[day], grid_B_sat[i], grid_B_min[j], 700.)

    return ff_faculae, ff_umbra, ff_penumbra


def visible(y_obs, y_pos, delta_lambda):
    """
    The visible function calculates the cosine of the heliocentric angle (mu = cos(theta)), also known as the cosine of the angle between the observer (y_obs) and the position of the sun (y_pos) in the sky. The heliocentric angle is an important parameter in astronomy that determines the visibility of celestial objects.

    The function takes three parameters:

    y_obs: The observer's position in the sky, specified in degrees.
    y_pos: The position of the sun in the sky, specified in degrees.
    delta_lambda: The difference in longitude between the observer and the sun, specified in degrees.
    
    """
    return (np.sin(np.deg2rad(y_obs)) * np.sin(np.deg2rad(y_pos)) + \
                 np.cos(np.deg2rad(y_obs)) * np.cos(np.deg2rad(y_pos)) * np.cos(np.deg2rad(delta_lambda)))


def compute_irradiance(ff_faculae, ff_umbra, ff_penumbra, interp_qs, interp_fac,
                    interp_um, interp_penum, x_obs, y_obs):
    """
    Calculation of irradiance from a full surface map of filling factors.
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
    x_obs : float
        The longitude of the observer.
    y_obs : float
        The latitude of the observer.

    Returns
    -------
    An array of len(nday).

    """
    nday, nlat, nlon = ff_faculae.shape
    days = np.linspace(0, nday-1, nday)
    irradiance = np.zeros(nday)
    
    x = np.linspace(1., 360., nlon)
    y = np.linspace(-90., 90., nlat)
    x_pos, y_pos = np.meshgrid(x, y)
    
    delta_lambda = abs(x_pos - x_obs)
    vis = isa.sun.visible(y_obs, y_pos, delta_lambda)
    vis[vis < 0.] = 0.                          # an observer will see only half of the sphere
    vis_corr = vis * np.cos(np.deg2rad(y_pos))  # solid angle of each pixel in the visible disk

    for i in range(nday):  

        # Rotation by 13.38 degrees per day
        lonshift = 13.38 * days[i]

        # Rotated filling factors multiplied with solid angle of the pixel
        mask_fac = vis_corr * isa.mapsetup.shift_map_longitude(ff_faculae[i], lonshift)
        mask_umb = vis_corr * isa.mapsetup.shift_map_longitude(ff_umbra[i], lonshift)
        mask_pen = vis_corr * isa.mapsetup.shift_map_longitude(ff_penumbra[i], lonshift)
        
        # Calculation of irradiance
        irr_qs = np.nansum(vis_corr * interp_qs(vis))              
        delta_fac = np.nansum(mask_fac * (interp_fac(vis) - interp_qs(vis)))
        delta_umb = np.nansum(mask_umb * (interp_um(vis) - interp_qs(vis)))
        delta_pen = np.nansum(mask_pen * (interp_penum(vis) - interp_qs(vis)))        

        irradiance[i] = irr_qs + delta_fac + delta_umb + delta_pen

    irradiance[irradiance == 0.] = np.nan
    
    return irradiance


def compute_simple_irradiance(ff_faculae, ff_umbra, ff_penumbra, interp_qs, interp_fac,
                    interp_um, interp_penum, x_obs, y_obs, grid = 1):
    """
    Calculation of irradiance from a full surface map of filling factors.
    Contribution by Sowmya Krishnamurthy.


    Parameters
    ----------
    ff_faculae : 2D array
        Filling factors of the faculae of shape (lat,lon).
    ff_umbra : 2D array
        Filling factors of the umbra of shape (lat,lon).
    ff_penumbra : 2D array
        Filling factors of the penumbra of shape (lat,lon).
    interp_qs : function
        Interpolation function (flux) for the quiet Sun.
    interp_fac : function
        Interpolation function (flux) for the faculae.
    interp_um : function
        Interpolation function (flux) for the umbra.
    interp_penum : function
        Interpolation function (flux) for the penumbra.
    x_obs : float
        The longitude of the observer.
    y_obs : float
        The latitude of the observer.

    Returns
    -------
    irradiance : float

    """
    
    x = np.linspace(1., 360., 360 * grid)
    y = np.linspace(-90., 90., 181 * grid)
    x_pos, y_pos = np.meshgrid(x, y)
    
    delta_lambda = abs(x_pos - x_obs)
    vis = isa.sun.visible(y_obs, y_pos, delta_lambda)
    vis[vis < 0.] = 0.                          # an observer will see only half of the sphere
    vis_corr = vis * np.cos(np.deg2rad(y_pos))  # solid angle of each pixel in the visible disk

    # Filling factors multiplied with solid angle of the pixel
    mask_fac = vis_corr * ff_faculae
    mask_umb = vis_corr * ff_umbra
    mask_pen = vis_corr * ff_penumbra
    
    # Calculation of irradiance
    irr_qs = np.nansum(vis_corr * interp_qs(vis))              
    delta_fac = np.nansum(mask_fac * (interp_fac(vis) - interp_qs(vis)))
    delta_umb = np.nansum(mask_umb * (interp_um(vis) - interp_qs(vis)))
    delta_pen = np.nansum(mask_pen * (interp_penum(vis) - interp_qs(vis)))        

    irradiance = irr_qs + delta_fac + delta_umb + delta_pen

    return irradiance
    

def compute_s_index(ff_faculae, interp_qsh, interp_qsk, interp_qsr, interp_qsv, interp_fach,
            interp_fack, interp_facr, interp_facv, x_obs, y_obs):
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
    x_obs : float
        The longitude of the observer.
    y_obs : float
        The latitude of the observer.

    Returns
    -------
    None.

    """
    nday, nlat, nlon = ff_faculae.shape
    days = np.linspace(0, nday-1, nday)
    s_index = np.zeros(nday)
    
    x = np.linspace(1., 360., nlon)
    y = np.linspace(-90., 90., nlat)
    x_pos, y_pos = np.meshgrid(x, y)    
    
    delta_lambda = abs(x_pos - x_obs)
    vis = isa.sun.visible(y_obs, y_pos, delta_lambda)
    vis[vis < 0.] = 0.                          
    vis_corr = vis * np.cos(np.deg2rad(y_pos))  

    for i in range(nday):  

        lonshift = 13.38 * days[i]

        mask_fac = vis_corr * isa.mapsetup.shift_map_longitude(ff_faculae[i], lonshift)
        
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
        
        fluxh = qsh + delta_fach
        fluxk = qsk + delta_fack
        fluxr = qsr + delta_facr
        fluxv = qsv + delta_facv
        
        fluxratio = (fluxh + fluxk) / (fluxr + fluxv)
        s_index[i] = 2.53 * 8. * fluxratio
        
    return s_index


def compute_tsi_from_ssi(ssi, wavelengths, number_of_days):
    """
    Computes total solar irradiance from spectral solar irradiance.

    Parameters
    ----------
    ssi : array
    wavelengths : array
    number_of_days : integer

    Returns
    -------
    tsi : array with length number_of_days

    """
    tsi = np.empty((number_of_days))

    for n in range(number_of_days):
        
        tmp = ssi[:, n]
        tmp2 = tmp[~np.isnan(tmp)]
        tmp3 = wavelengths[~np.isnan(tmp)]
        tsi[n] = integrate.trapz(tmp2, tmp3)
        
    return tsi


def load_satire_ssi(file, n_days, start_year = 2010, start_month = 6, start_day = 17):
    """
    Loads the SATIRE-S SSI data file for a specific time period

    Parameters
    ----------
    file : str
        Path to the data file.
    n_days : int
        Number of days since the start date to load the data.
    start_year : int, optional
        The default is 2010.
    start_month : int, optional
        The default is 6.
    start_day : int, optional
        The default is 17.

    Returns
    -------
    ssi : 2D array
        Array with teh SSI for each day adn wavelength.
    wl_lower : 2D array
        Array of wavelengths.

    """
    df = pd.read_csv(file, skiprows = 29, names = ['julian', 'bin_lower', 'bin_upper', 'SSI', 'index'],
                      delim_whitespace = True, skipinitialspace = True, infer_datetime_format = True)

    julian_days  = np.array(df.julian)
    wl_lower0 = np.array(df.bin_lower)
    ssi0 = np.array(df.SSI)

    start_date = julian.to_jd(dt(start_year, start_month, start_day, 12, 0))          
    end_date = start_date + n_days 

    condition = np.logical_and(julian_days >= start_date, julian_days < end_date)
    
    wl_lower = wl_lower0[condition]
    ssi = ssi0[condition]
    
    n_wl = len(np.unique(wl_lower))

    ssi = np.reshape(ssi, (n_days, n_wl))
    wl_lower = np.reshape(wl_lower, (n_days, n_wl)) 

    return ssi, wl_lower


def compute_irradiance_virgo(wavelengths, irradiance):
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


def compute_flux_map(flux_quiet, flux_faculae, flux_umbra, flux_penumbra, x_obs):
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


def irradiance_from_Br(B, ff_faculae, ff_umbra, ff_penumbra, interp_qs, interp_fac, \
                    interp_um, interp_penum, x_obs = None, y_obs = None):
    """
    Calculation of solar spectral irradiance for 1 wavelength, for multiple days, from a filling factors map.
    Only works for a map of the visible disk. For a full surface map, use irradiance_full_surface below.
    
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
    
    for i in range(nday):
        
        # Take the (~) central colatitude of the map y0, and calculate the index of the central longitude x0
        y0 = 1. * B[i][90] # <<<<<----- THIS IS WRONG!!
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


def limb_darkening_correction(data_hmi, keys):
    """
    Corrects for the limb darkening of a continuum intensity map. 
    Created by Dan Yang after Thompson (2006), https://doi.org/10.1051/0004-6361:20054262
    The limb darkening coefficients were provided by JSOC. They are from May 11th Mercury transit test run.

    Parameters
    ----------
    data_hmi : array
        Continuum intensity map.
    keys : TYPE
        Metadata from DRMS.

    Returns
    -------
    data_corrected : array
        The corrected data.
    smap: array
        The limb darkening map, without data.

    """
    x0     = float(keys['CRPIX1'])
    y0     = float(keys['CRPIX2'])
    DSUN   = float(keys['DSUN_OBS'])
    dx     = float(keys['CDELT1'])
    
    RSUN = 696e6

    x_raw = (np.arange(np.shape(data_hmi)[0]) + 1 - x0) # + 1 due to wcs
    y_raw = (np.arange(np.shape(data_hmi)[1]) + 1 - y0) # + 1 due to wcs

    xm_raw, ym_raw               = np.meshgrid(x_raw, y_raw)
    key_secant                   = {}
    key_secant["xm"]             = xm_raw
    key_secant["ym"]             = ym_raw
    key_secant["rad_sun_angle"]  = np.arcsin(RSUN / DSUN)
    key_secant["pix_sep_radian"] = np.deg2rad(dx) / 3600.
    key_secant["sun_rad_maj"]    = key_secant["rad_sun_angle"] / key_secant["pix_sep_radian"] 
    key_secant["sun_rad_min"]    = key_secant["sun_rad_maj"] 
    key_secant["maj_ax_ang"]     = 0 # angle of major axis, here assume to be zero. 
    key_secant["secant_thresh"]  = 4 
    
    LD_coeffs = [1.0, 0.429634631, 0.071182463, -0.02522375, -0.012669259, -0.001446241] 

    u = np.sin(key_secant["maj_ax_ang"])
    v = np.cos(key_secant["maj_ax_ang"])
    xm = key_secant["xm"]
    ym = key_secant["ym"]
    maj_ax_proj = (u * xm + v * ym) / key_secant["sun_rad_maj"]
    min_ax_proj = (v * xm - u * ym) / key_secant["sun_rad_min"]
    rho2 = maj_ax_proj ** 2 + min_ax_proj ** 2

    smap = np.zeros(np.shape(rho2))

    mu = np.sqrt(1.0 - rho2[rho2 < 1]) 
    xi = np.log(mu)
    zt = 1.0
    ld = 1.0
    
    for ord in np.arange(1, 6):
        zt *= xi
        ld += LD_coeffs[ord] * zt
    
    smap[rho2 < 1] = 1. / ld
    
    data_corrected = data_hmi * smap
    
    return data_corrected


def get_keys_from_drms(ds, keylist):
    
    knames = ','.join([nam for nam, typ in keylist])
    p = subp.Popen('show_info ds=%s key=%s -q' % (ds, knames),
                    shell=True, stdout=subp.PIPE, encoding='utf-8')
    lines = [line.rstrip() for line in p.stdout.readlines()]
    keys_str = np.array([line.split() for line in lines])
    keys = {}
    for i in range(keys_str.shape[0]):
        line = keys_str[i]
        keys[line[0]] = {}
        for j in range(1, keys_str.shape[1]):
            keys[line[0]][keylist[j][0]] = keylist[j][1](line[j])
    return keys


def get_paths_from_drms(ds):
    p = subp.Popen('show_info ds=%s -Pq' % (ds), shell=True,
                    stdout=subp.PIPE, encoding='utf-8')
    paths = [line.strip() for line in p.stdout.readlines()]
    return paths


def degrade_image(data, new_res):
    """
    Decreases the resolution of the data by averaging.

    Parameters
    ----------
    data : 2D array of shape (m, m)
        The original data.
        
    new_res : int
        The resolution desired for the image.

    Returns
    -------
    new_data : TYPE
        Degraded array with shape (new_res, new_res).

    """
            
    m = data.shape[0]
    y = np.linspace(0, 1.0/m, data.shape[0])
    x = np.linspace(0, 1.0/m, data.shape[1])
    interpolating_function = RegularGridInterpolator((y, x), data)

    yv, xv = np.meshgrid(np.linspace(0, 1.0/m, new_res), np.linspace(0, 1.0/m, new_res))

    return interpolating_function((xv, yv))

           
def create_map_mu_for_solar_disc(n_pixels, radius):
    """
    Creates a map of mu values (mu = cosine(heliocentric angle)) for a solar disc, with 11 concentric rings.

    Parameters
    ----------
    n_pixels : int
        The number of pixels in one row in the map. The map will have n_pixels x n_pixels pixels.
    radius : float
        The radius of the solar disc, in pixels.

    Returns
    -------
    mu_map : 2D array
        The map of mu values.

    """
    x0 = n_pixels // 2
    y0 = n_pixels // 2

    mu = [1, .95, .85, .75, .65, .55, .45, .35, .25, .15, .075, 0.0]
    mu_inverted = mu[::-1]
    full_disc = np.zeros((n_pixels, n_pixels))

    for i in range(len(mu)):
        r = mu[i] * radius
        [X, Y] = np.mgrid[0:n_pixels, 0:n_pixels]
        xpr = X - x0
        ypr = Y - y0

        reconstruction_circle = (xpr ** 2 + ypr ** 2) <= r ** 2
        full_disc[reconstruction_circle] = mu_inverted[i]
    
    return full_disc


########################################################################################