'''
    Computes the magnetic field components from Langlais' spherical harmonic model at a certain altitude.
    
    References: 
        
        Whaler, K. A.; Gubbins, D., 1981. Spherical harmonic analysis of the geomagnetic field: an example of a 
        linear inverse problem, Geophysical Journal of the Royal Astronomical Society, 65, 645-693. 
        DOI: doi.org/10.1111/j.1365-246X.1981.tb04877.x
        
        Langlais, B.; Thébault, E.; Houliez, A.; Purucker, M.; Lillis, R. J., 2019. A new model of the crustal
        magnetic field of Mars using MGS and MAVEN, Journal of Geophysical Research: Planets, 124, 6, 1542-1569.
        DOI: doi.org/10.1029/2018JE005854
        
        Langel, R. A., The main field, In: Geomagnetism, 1987, ed. by J. A. Jacobs, Academic Press.
'''

import pyshtools as sh
import numpy as np
import scipy as sp
import scipy.io
import matplotlib.pyplot as plt
from tqdm import tqdm


def legendre_schmidt_Pyshtools(lat, nmax = 134):
    """
    Uses the pyshtools package to compute the Schmidt semi-normalized associated Legendre functions of the cosine of the colatitude.
    
    Parameters:
        lat: float
            The latitude, in degrees.
            
        nmax: integer, optional
            The maximum degree and order of the functions. Default (Langlais model) is 134.
            
    Returns:
        P: nmax+1 X nmax+1 array containing the associated Legendre functions.
        dP: nmax+1 X nmax+1 array containing the first derivatives of the functions.
        
    """
    theta = np.deg2rad(90.0 - lat)
    P0, dP0 = sh.legendre.PlmSchmidt_d1(nmax, np.cos(theta))
    
    P = np.zeros((nmax+1, nmax+1))
    dP = np.zeros((nmax+1, nmax+1))
    
    i = 0
    for n in range(nmax+1):
        for m in range(n+1):
            P[n, m] = P0[i]
            dP[n, m] = dP0[i]
            i += 1
    
    del P0, dP0
    
    return P, dP


def legendre_schmidt_Brain(lat, nmax = 134):
    """
    Uses the David Brain's approach (CCATi -  mars_crust_model.pro) to compute the Schmidt semi-normalized associated Legendre functions.
    Not used. Use legendre_schmidt_Pyshtools instead.
    
    Parameters:
        lat: float
            The latitude, in degrees.
            
        nmax: integer
            The maximum degree and order of the functions.
            
    Returns:
        P: nmax+1 X nmax+1 array containing the associated Legendre functions.
        
    """
    theta = np.deg2rad(90.0 - lat)
    x = np.cos(theta)
    P = np.zeros((nmax+1, nmax+1))
    
    P[0, 0] = 1.0
    P[1, 0] = x
    P[1, 1] = - np.sqrt(1 - x**2)
    
    for n in range(2, nmax+1):
        P[n, 0] = ((2*(n-1)+1) * x * P[n-1, 0] - (n-1) * P[n-2, 0])/n
        
    Cm = np.sqrt(2.0)
    for m in range(1, nmax+1):
        Cm /=  np.sqrt(2.0*m * (2.0*m - 1.0))
        P[m, m] = (1.0 - x**2)**(0.5*m) * Cm
        
        for i in range(1, m):
            P[m, m] *= (2.0*i + 1.0)
       
        if nmax > m:
            twoago = 0.0
            for n in range(m+1, nmax+1):
                P[n, m] = (x * (2.0*n - 1.0) * P[n-1, m] - np.sqrt((n+m-1.0) * (n-m-1.0)) * twoago) / np.sqrt((n**2 - m**2))
                twoago = P[n-1, m]
                
    return P
    

def legendre_schmidt_ChaosMagPy(theta, nmax):
    """
    Copy of the function developed in the ChaosMagPy package. I did not write this code!
    I do not use this, it is here only for reference on the equations!
    Use legendre_schmidt_Pyshtools instead.
    
    
    Returns associated Legendre polynomials `P(n,m)` (Schmidt quasi-normalized)
    and the derivative :math:`dP(n,m)/d\\theta` evaluated at :math:`\\theta`.
    Parameters
    ----------
    nmax : int, positive
        Maximum degree of the spherical expansion.
    theta : ndarray, shape (...)
        Colatitude in degrees :math:`[0^\\circ, 180^\\circ]`
        of arbitrary shape.
    Returns
    -------
    Pnm : ndarray, shape (n, m, ...)
          Evaluated values and derivatives, grid shape is appended as trailing
          dimensions. `P(n,m)` := ``Pnm[n, m, ...]`` and `dP(n,m)` :=
          ``Pnm[m, n+1, ...]``
    References
    ----------
    Based on Equations 26-29 and Table 2 in:
    Langel, R. A., "Geomagnetism - The main field", Academic Press, 1987,
    chapter 4
    """

    costh = np.cos(np.deg2rad(theta))
    sinth = np.sqrt(1-costh**2)

    Pnm = np.zeros((nmax+1, nmax+2) + costh.shape)
    Pnm[0, 0] = 1.  # is copied into trailing dimensions
    Pnm[1, 1] = sinth  # write theta into trailing dimenions via broadcasting

    rootn = np.sqrt(np.arange(2 * nmax**2 + 1))

    # Recursion relations after Langel "The Main Field" (1987),
    # eq. (27) and Table 2 (p. 256)
    for m in range(nmax):
        Pnm_tmp = rootn[m+m+1] * Pnm[m, m]
        Pnm[m+1, m] = costh * Pnm_tmp

        if m > 0:
            Pnm[m+1, m+1] = sinth*Pnm_tmp / rootn[m+m+2]

        for n in np.arange(m+2, nmax+1):
            d = n * n - m * m
            e = n + n - 1
            Pnm[n, m] = ((e * costh * Pnm[n-1, m] - rootn[d-e] * Pnm[n-2, m])
                         / rootn[d])

    # dP(n,m) = Pnm(m,n+1) is the derivative of P(n,m) vrt. theta
    Pnm[0, 2] = -Pnm[1, 1]
    Pnm[1, 2] = Pnm[1, 0]
    for n in range(2, nmax+1):
        Pnm[0, n+1] = -np.sqrt((n*n + n) / 2) * Pnm[n, 1]
        Pnm[1, n+1] = ((np.sqrt(2 * (n*n + n)) * Pnm[n, 0]
                       - np.sqrt((n*n + n - 2)) * Pnm[n, 2]) / 2)

        for m in np.arange(2, n):
            Pnm[m, n+1] = (0.5*(np.sqrt((n + m) * (n - m + 1)) * Pnm[n, m-1]
                           - np.sqrt((n + m + 1) * (n - m)) * Pnm[n, m+1]))

        Pnm[n, n+1] = np.sqrt(2 * n) * Pnm[n, n-1] / 2

    return Pnm
    

def mag_components(lon, lat, alt, comp, nmax = 134):
    """
    Calculates the magnetic field component (Br, Btheta or Bphi) of the crustal field model for one set of aerographic coordinates.
    
    Parameters:
        lon: float
            The longitude, in degrees.
        lat: float
            The latitude, in degrees.
        alt: float
            The altitude, in km.
        comp: string
            The desired magnetic field component, in spherical coordinates. Options are 'Br', 'Btheta', and 'Bphi'.
        nmax: integer, optional
            The maximum degree and order of the functions. Default (Langlais model) is 134.
        
    Returns:
        Three arrays containing the magnetic field components Br, Btheta and Bphi.        
    """
    # Raise an AssertionError if component is invalid
    assert comp == 'Br' or comp == 'Btheta' or comp == 'Bphi', "Check argument for comp"
    
    # Import the coefficient files
    from IsabelaFunctions.langlais_coeff import glm as g
    from IsabelaFunctions.langlais_coeff import hlm as h
    
    # Mars' radius
    a = 3393.5
    
    # Calculate r, theta, phi, and the Legendre functions
    r = a + alt
    phi = np.deg2rad(lon)
    theta = np.deg2rad(90.0 - lat)
    P, dP = legendre_schmidt_Pyshtools(lat, nmax)
    
    # Calculate Br, Btheta, Bphi
    if comp == 'Br':
        B = sum((n+1) * (a/r)**(n+2) * (g[n, m] * np.cos(m * phi) + h[n, m] * np.sin(m * phi)) * P[n, m] \
                for n in range(1, nmax+1) for m in range(n+1))
    elif comp == 'Btheta':
        B = sum((a/r)**(n+2) * (g[n, m] * np.cos(m * phi) + h[n, m] * np.sin(m * phi)) * \
                np.sin(theta) * dP[n, m] for n in range(1, nmax+1) for m in range(n+1))
    else:
        B = 1/np.sin(theta) * sum((a/r)**(n+2) * P[n, m] * m * (g[n, m] * np.sin(m * phi) - h[n, m] * \
                 np.cos(m * phi)) for n in range(1, nmax+1) for m in range(n+1))

    return B
            

def model_map(lon, lat, alt, comp, nmax = 134, binsize = 0.1):
    """
    Calculates a map of one component of the crustal magnetic field field model, for a given altitude.
    
    Parameters:
        lon: array
            The longitude range, in degrees.
        lat: array
            The latitude range, in degrees.
        alt: float
            The altitude in which the map will be computed, in km.
        comp: string
            The desired magnetic field component, in spherical coordinates. Options are 'Br', 'Btheta', and 'Bphi'.
        nmax: integer, optional
            The maximum degree and order of the functions. Default (Langlais model) is 134.
        binsize: float, list, optional
            The resolution of the grid. If a float, apply the same binsize for longitude and latitude. 
            If a list, the first value represents the longitude binsize and the second, the latitude binsize. 
        
    Returns:
        A lon X lat array containing the magnetic field component.        
    """
    # Raise an AssertionError if arguments are invalid
    assert comp == 'Br' or comp == 'Btheta' or comp == 'Bphi', "Check argument for comp"
    assert type(binsize) is float or type(binsize) is list, "Argument for binsize should be a float or a list"
    
    # Import the coefficient files
    from IsabelaFunctions.langlais_coeff import glm as g
    from IsabelaFunctions.langlais_coeff import hlm as h
    
    # Mars' radius
    a = 3393.5
    
    # Calculate r, theta, phi, and the Legendre functions
    r = a + alt
    
    if type(binsize) is float:
        binsize = [binsize, binsize]
    
    lat_len = int((lat[1] - lat[0]) / binsize[1] + 1.0)
    lon_len = int((lon[1] - lon[0]) / binsize[0] + 1.0)
        
    longitude = np.deg2rad(np.linspace(lon[0], lon[1], lon_len))
    latitude = np.linspace(lat[0], lat[1], lat_len)
    
    P = np.empty((nmax+1, nmax+1, lat_len)) * np.nan
    dP = np.empty_like(P) * np.nan
    for theta in range(lat_len):
        P[:, :, theta], dP[:, :, theta] = legendre_schmidt_Pyshtools(latitude[theta], nmax)
    
    cos = np.empty((nmax+1, lon_len)) * np.nan
    sen = np.empty_like(cos) * np.nan
    for phi in range(lon_len):
        for m in range(nmax+1):
            cos[m, phi] = np.cos(m * longitude[phi])
            sen[m, phi] = np.sin(m * longitude[phi])
        
    sen_theta = np.sin(np.deg2rad(90.0 - latitude))
    
    a_over_r = np.empty((nmax+1)) * np.nan
    for n in range(nmax+1):
        a_over_r[n] = (a/r)**(n+2)
        
    B = np.zeros((lon_len, lat_len))
    
    if comp == 'Br':
        for n in tqdm(range(1, nmax+1)):
            for m in range(n+1):
                tmp1 = g[n, m] * cos[m, :] + h[n, m] * sen[m, :]
                tmp2 = np.outer(tmp1, P[n, m, :])
                tmp3 = tmp2 * (n+1) * a_over_r[n]
                B += tmp3
    
    elif comp == 'Btheta':
        for n in tqdm(range(1, nmax+1)):
            for m in range(n+1):
                tmp1 = g[n, m] * cos[m, :] + h[n, m] * sen[m, :]
                tmp2 = np.outer(tmp1, dP[n, m, :] * sen_theta)
                tmp3 = tmp2 * a_over_r[n]
                B += tmp3
        
    else:
        for n in tqdm(range(1, nmax+1)):
            for m in range(n+1):
                tmp1 = g[n, m] * sen[m, :] + h[n, m] * cos[m, :]
                tmp2 = np.outer(tmp1, P[n, m, :])
                tmp3 = tmp2 * m * a_over_r[n]
                B += tmp3
        
        for theta in range(lat_len):
            B[:, theta] /= sen_theta[theta]
        
    return B.T


##############################################











