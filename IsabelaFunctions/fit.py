"""
Computes different types of statistics, means, fits/regression for the data.
"""

import numpy as np
from scipy.optimize import curve_fit
from numpy.fft import fft2, ifft2
from scipy.ndimage import gaussian_filter


def make_gaussian_kernel_2D(sigma, truncate):
    """ Creates a 2D Gaussian kernel to be used as a point spread function (psf).
        The sum of the psf is approximately 1. 
        The radius of the Gaussian kernel is round(truncate * sigma).
        The shape of the kernel is (2*radius + 1, 2*radius + 1).

    Parameters:
        sigma: float
            The standard deviation of the Gaussian.
        truncate: float
            The truncation of the Gaussian.

    Returns:
        psf: 2D-array
            The 2D Gaussian kernel.
    """
    radius = round(truncate * sigma)
    perfect_psf = np.zeros((2*radius + 1, 2*radius + 1))
    perfect_psf[radius, radius] = 1
    psf = gaussian_filter(perfect_psf, sigma = sigma, mode = 'constant', cval = 0, truncate = truncate)
    return psf


def convolve_image(image, psf):
    """ Convolves the image with the psf.
        The image should not contain NaNs.

    Parameters:
        image: 2D-array
            The image to be convolved.
        psf: array
            The point spread function.

    Returns:
        convolved_image: 2D-array
            The convolved image.
    """
    image_shape = np.array(image.shape)
    psf_shape = np.array(psf.shape)
    z_shape = image_shape + psf_shape - 1

    image_fft = fft2(image, z_shape)
    psf_fft = fft2(psf, z_shape)

    convolved_image_fft = image_fft * psf_fft
    convolved_image = ifft2(convolved_image_fft).real
    return convolved_image


def deconvolve_image(image, psf):
    """ Deconvolves an image using the psf.
        The image should not contain NaNs.

    Parameters:
        image: 2D-array
            The image to be deconvolved.
        psf: array
            The point spread function.

    Returns:
        deconvolved_image: 2D-array
            The deconvolved image.
    """
    image_shape = np.array(image.shape)
    psf_shape = np.array(psf.shape)
    new_shape = image_shape + psf_shape - 1

    image_fft = fft2(image, new_shape)
    psf_fft = fft2(psf, new_shape)

    deconvolved_image_fft = image_fft / psf_fft
    deconvolved_image = ifft2(deconvolved_image_fft).real

    cut = (deconvolved_image.shape[0] - image.shape[0]) // 2
    deconvolved_image_back_to_old_shape = deconvolved_image[cut:-cut, cut:-cut]

    return deconvolved_image_back_to_old_shape


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


def gauss_function(x, off, a, x_mean, sigma):
    """ Equation for a Gaussian curve.

    Parameters:
        x: array
            The independet variable where the data is measured.
        off: float
            Vertical offset of the Gaussian curve.
        a: float
            The height of the curve peak (negative for inverted bells).
        x_mean: float
            The mean in the x-axis, where the curve is centralized (x value for minimun y).
        sigma: float
            Standard deviation of the curve.
            
    Returns:
        f: array
            The Gaussian function.
    """
    f = off+a*np.exp(-(x-x_mean)**2/(2*sigma**2))
    return f


def gauss_fit(x, y):
    """ Best fit with the Gaussian function. Uses scipy.optimize.curve_fit.

    Parameters:
        x: array
            The independet variable where the data is measured.
        y: array
            The data points used for the best fit.

    Returns:
        popt: array
            The optimal values for the parameters. Respectively off, a, x_mean, and sigma.
        pcov: array
            The esimated covariance of popt. The diagonals provide the variance of the parameter estimate.
        perr: array
            One standard deviation errors on the parameters.
    """
    x_mean = sum(x*y)/sum(y)
    sigma = np.sqrt(sum(y*(x-x_mean)**2)/sum(y))
    p0 = (np.max(y), np.min(y)-np.max(y), x_mean, sigma)  # Initial guess for the curve parameters
    popt, pcov = curve_fit(gauss_function, x, y, p0, maxfev=100000)  # maxfev is the maximum number of calls to the function
    perr = np.sqrt(np.diag(pcov))

    return popt, pcov, perr


def linear_function(x, a, b):
    """ Equation for a line.

    Parameters:
        x: array
            The independet variable where the data is measured.
        a: float
            The linear coefficient.
        b: float
            The angular coefficient.

    Returns:
        f: array
            The linear function.
    """
    f = a + b*x
    return f


def linear_fit(x, y, y_error):
    """ Best fit with a linear function. Uses scipy.optimize.curve_fit.

    Parameters:
        x: array
            The independet variable where the data is measured.
        y: array
            The data points used for the best fit.
        y_error: array
            The error in the data.

    Returns:
        popt: array
            The optimal values for the parameters. Respectively linear and angular coefficients.
        pcov: array
            The esimated covariance of popt. The diagonals provide the variance of the parameter estimate.
        perr: array
            One standard deviation errors on the parameters.
    """
    popt, pcov = curve_fit(linear_function, x, y, sigma = y_error, absolute_sigma = True)
    perr = np.sqrt(np.diag(pcov))
    return popt, pcov, perr


def exp_function(x, a, b, c):
    """ Equation for an exponential curve.

    Parameters:
        x: array
            The independet variable where the data is measured.

        a, b, c: float
            The coefficients.

    Returns:
        f: array
            The exponential function.
    """
    f = a*np.exp(b*x)+c
    return f


def exp_fit(x, y, y_error, p0=(1, 1, 1)):
    """ Best fit with an exponential function. Uses scipy.optimize.curve_fit.

    Parameters:
        x: array
            The independet variable where the data is measured.
        y: array
            The data points used for the best fit.
        y_error: array
            The error in the data.
        p0: array
            The initial parameters. The default is p0 = (1, 1, 1)

    Returns:
        popt: array
            The optimal values for the parameters. Respectively linear and angular coefficients.
        pcov: array
            The esimated covariance of popt. The diagonals provide the variance of the parameter estimate.
        perr: array
            One standard deviation errors on the parameters.
    """
    popt, pcov = curve_fit(exp_function, x, y, p0=p0, sigma=y_error, absolute_sigma=True, maxfev=10000)
    perr = np.sqrt(np.diag(pcov))
    return popt, pcov, perr


def deg2km_mars(alt, deg, lat = 0):
    """ Converts lon/lat distances in degrees to kilometers, according to the desired altitude, on Mars.

    Parameters:
        alt: float or array
            The altitudes where the data are measured.
        deg: float or array
            The measured offsets in degree.
        lat: float or array
            The latitudes where the data are measured. This parameter is useful for when converting longitudes, since the longitude distance in km depends on the latitude.
            Default is zero. Leave it on default if converting latitude degrees to km.

    Returns:
        d: array
            The offset distances in kilometers.
    """
    r = 3393.5  # Mars' radius
    p = 2*np.pi*(r+alt) * np.cos(lat)
    d = p*deg/360
    return d
