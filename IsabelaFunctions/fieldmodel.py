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
from tqdm import tqdm
from enum import Enum


def legendre_schmidt_Pyshtools(lat):
    """
    Uses the pyshtools package to compute the Schmidt semi-normalized associated Legendre functions of the cosine of the colatitude, with nmax = 134 (Langlais model),
    
    Parameters:
        lat: float or array
            The latitude, in degrees.
            
    Returns:
        P: 135 X 135 array containing the associated Legendre functions.
        dP: 135 X 135 array containing the first derivatives of the functions.
        
    """
    nmax = 134
    theta = np.deg2rad(90.0 - lat)
    
    if np.isscalar(lat):
        P0, dP0 = sh.legendre.PlmSchmidt_d1(nmax, np.cos(theta))
        
        P = np.zeros((nmax+1, nmax+1))
        dP = np.zeros((nmax+1, nmax+1))
        
        i = 0
        for n in range(nmax+1):
            for m in range(n+1):
                P[n, m] = P0[i]
                dP[n, m] = dP0[i]
                i += 1
    else:
        P = np.zeros((nmax+1, nmax+1, len(lat)))
        dP = np.zeros((nmax+1, nmax+1, len(lat)))
        
        for k in range(len(lat)):
            P0, dP0 = sh.legendre.PlmSchmidt_d1(nmax, np.cos(theta[k]))
            
            i = 0
            for n in range(nmax+1):
                for m in range(n+1):
                    P[n, m, k] = P0[i]
                    dP[n, m, k] = dP0[i]
                    i += 1
        
    return P, dP


def legendre_schmidt_Brain(lat, nmax = 134):
    """
    Uses the David Brain's approach (CCATi -  mars_crust_model.pro) to compute the Schmidt semi-normalized associated Legendre functions and its derivatives.
    
    Parameters:
        lat: float
            The latitude, in degrees.
            
        nmax: integer
            The maximum degree and order of the functions.
            
    Returns:
        P, dP: nmax+1 X nmax+1 arrays containing the associated Legendre functions and its derivatives, respectively.
        
    """
    theta = np.deg2rad(90.0 - lat)
    x = np.cos(theta)
    P = np.zeros((nmax+1, nmax+1))
    dP = np.zeros((nmax+1, nmax+1))
    
    P[0, 0] = 1.0
    P[1, 0] = x
    P[1, 1] = - np.sqrt(1 - x**2)
    dP[0, 0] = 0.0
    dP[1, 0] = 1.0
    
    for n in range(2, nmax+1):
        P[n, 0] = ((2*(n-1)+1) * x * P[n-1, 0] - (n-1) * P[n-2, 0])/n
    
    dP[nmax, 0] = nmax / (x**2 - 1) * (x * P[nmax, 0] - P[nmax-1, 0])
    for n in range(2, nmax):
        dP[n, 0] = (n+1) / (x**2 - 1) * (P[n+1, 0] - x * P[n, 0])
    
    Cm = np.sqrt(2.0)
    for m in range(1, nmax+1):
        Cm /=  np.sqrt(2.0*m * (2.0*m - 1.0))
        P[m, m] = (1.0 - x**2)**(0.5*m) * Cm
        
        for i in range(1, m):
            P[m, m] *= (2.0*i + 1.0)
        
        dP[m, m] = -P[m, m] * m * x / np.sqrt(1 - x**2)
                
        if nmax > m:
            twoago = 0.0
            for n in range(m+1, nmax+1):
                P[n, m] = (x * (2.0*n - 1.0) * P[n-1, m] - np.sqrt((n+m-1.0) * (n-m-1.0)) * twoago) / np.sqrt(n**2 - m**2)
                twoago = P[n-1, m]
    
    for n in range(2, nmax+1):
        for m in range(1, n):
            dP[n, m] = np.sqrt((n-m) * (n+m+1)) * P[n, m+1] - P[n, m] * m * x / np.sqrt(1 - x**2)
        
    return P, dP

    
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
    

def mag_components(lon, lat, alt, comp):
    """
    Calculates the magnetic field component (Br, Btheta, Bphi, or Bt) of the crustal field model for one set of aerographic coordinates.
    Obs.: Btheta is positive southwards (the origin is in the north pole).
    
    Parameters:
        lon: float or array
            The longitude, in degrees.
            
        lat: float or array
            The latitude, in degrees.
            
        alt: float or array
            The altitude, in km.
            
        comp: string
            The desired magnetic field component, in spherical coordinates. Options are 'Br', 'Btheta', 'Bphi', and 'Bt.
            
    Returns:
        A float or an array containing the magnetic field component Br, Btheta, Bphi, or Bt.        
    """
    # Raise an AssertionError if component is invalid
    assert comp == 'Br' or comp == 'Btheta' or comp == 'Bphi' or comp == 'Bt', "Check argument for comp"
    
    # Import the coefficient files
    from IsabelaFunctions.langlais_coeff import glm as g
    from IsabelaFunctions.langlais_coeff import hlm as h
    
    # Mars' radius
    a = 3393.5
    nmax = 134
    
    # Calculate r, theta, phi, and the Legendre functions
    r = a + alt
    if hasattr(lon, '__len__') is False:
        l = 1
    else:
        l = len(lon)
    
    P, dP = legendre_schmidt_Pyshtools(lat)
    
    cos = np.empty((nmax+1, l)) * np.nan
    sen = np.empty_like(cos) * np.nan
    for phi in range(l):
        for m in range(nmax+1):
            if np.isscalar(lon) is True:
                cos[m, phi] = np.cos(m * np.deg2rad(lon))
                sen[m, phi] = np.sin(m * np.deg2rad(lon))
            else:
                cos[m, phi] = np.cos(m * np.deg2rad(lon[phi]))
                sen[m, phi] = np.sin(m * np.deg2rad(lon[phi]))
   
    a_over_r = np.empty((nmax+1, l)) * np.nan
    if l == 1:
        for n in range(nmax+1):
            a_over_r[n] = (a/r)**(n+2)
    else:
        for radius in range(l):
            for n in range(nmax+1):
                a_over_r[n, radius] = (a/r[radius])**(n+2)
        
    # Calculate Br, Btheta, Bphi, Bt
    if comp == 'Bt':
        Br = np.zeros(l)
        Btheta = np.zeros(l)
        Bphi = np.zeros(l)
        sen_theta = np.sin(np.deg2rad(90.0 - lat))
        
        for n in range(1, nmax+1):
                for m in range(n+1):
                    Br += (g[n, m] * cos[m] + h[n, m] * sen[m]) * P[n, m] * (n+1) * a_over_r[n]
                    Btheta += (g[n, m] * cos[m] + h[n, m] * sen[m]) * dP[n, m] * sen_theta * a_over_r[n]
                    Bphi += (g[n, m] * sen[m] + h[n, m] * cos[m]) * P[n, m] * m * a_over_r[n]
        Bphi /= sen_theta 
        
        B = np.sqrt(Br**2 + Btheta**2 + Bphi**2)
        
    else:
        B = np.zeros(l)
        
        if comp == 'Br':
            for n in range(1, nmax+1):
                for m in range(n+1):
                    B += (g[n, m] * cos[m] + h[n, m] * sen[m]) * P[n, m] * (n+1) * a_over_r[n]
                    
        elif comp == 'Btheta':
            sen_theta = np.sin(np.deg2rad(90.0 - lat))
            for n in range(1, nmax+1):
                for m in range(n+1):
                    B += (g[n, m] * cos[m] + h[n, m] * sen[m]) * dP[n, m] * sen_theta * a_over_r[n]  
            
        elif comp == 'Bphi':
            sen_theta = np.sin(np.deg2rad(90.0 - lat))
            for n in range(1, nmax+1):
                for m in range(n+1):
                    B += (g[n, m] * sen[m] + h[n, m] * cos[m]) * P[n, m] * m * a_over_r[n]
            
            B /= sen_theta
    
    return B
            

def model_map(lon, lat, alt, comp, binsize = 0.1):
    """
    Calculates a map of one component of the crustal magnetic field field model, for a given altitude.
    
    Parameters:
        lon: array
            The longitude range, in degrees. Ex.: [20., 50.].
            
        lat: array
            The latitude range, in degrees.
            
        alt: float
            The altitude in which the map will be computed, in km.
            
        comp: string
            The desired magnetic field component, in spherical coordinates. Options are 'Br', 'Btheta', 'Bphi', and 'Bt'.
            
        binsize: float, list, optional
            The resolution of the grid. If a float, apply the same binsize for longitude and latitude. 
            If a list, the first value represents the longitude binsize and the second, the latitude binsize. 
        
    Returns:
        A lon X lat array containing the magnetic field component.        
    """
    # Raise an AssertionError if arguments are invalid
    assert comp == 'Br' or comp == 'Btheta' or comp == 'Bphi' or comp == 'Bt', "Check argument for comp"
    assert type(binsize) is float or type(binsize) is list, "Argument for binsize should be a float or a list"
    
    # Import the coefficient files
    from IsabelaFunctions.langlais_coeff import glm as g
    from IsabelaFunctions.langlais_coeff import hlm as h
    
    # Mars' radius
    a = 3393.5
    nmax = 134
    
    # Calculate r, theta, phi, and the Legendre functions
    r = a + alt
    
    if type(binsize) is float:
        binsize = [binsize, binsize]
    
    lat_len = int(round((lat[1] - lat[0]) / binsize[1] + 1.0))
    lon_len = int(round((lon[1] - lon[0]) / binsize[0] + 1.0))
        
    longitude = np.deg2rad(np.linspace(lon[0], lon[1], lon_len))
    latitude = np.linspace(lat[0], lat[1], lat_len)
    
    P = np.empty((nmax+1, nmax+1, lat_len)) * np.nan
    dP = np.empty_like(P) * np.nan
    for theta in range(lat_len):
        P[:, :, theta], dP[:, :, theta] = legendre_schmidt_Pyshtools(latitude[theta])
    
    cos = np.empty((nmax+1, lon_len)) * np.nan
    sen = np.empty_like(cos) * np.nan
    for phi in range(lon_len):
        for m in range(nmax+1):
            cos[m, phi] = np.cos(m * longitude[phi])
            sen[m, phi] = np.sin(m * longitude[phi])
    
    a_over_r = np.empty((nmax+1)) * np.nan
    for n in range(nmax+1):
        a_over_r[n] = (a/r)**(n+2)
    
    if comp == 'Bt':
        Br = np.zeros((lon_len, lat_len))
        Btheta = np.zeros((lon_len, lat_len))
        Bphi = np.zeros((lon_len, lat_len))
        sen_theta = np.sin(np.deg2rad(90.0 - latitude))
        
        for n in range(1, nmax+1):
                for m in range(n+1):
                    tmp1 = g[n, m] * cos[m, :] + h[n, m] * sen[m, :]
                    tmp2 = np.outer(tmp1, P[n, m, :])
                    tmp3 = tmp2 * (n+1) * a_over_r[n]
                    Br += tmp3
                    
                    tmp2 = np.outer(tmp1, dP[n, m, :] * sen_theta)
                    tmp3 = tmp2 * a_over_r[n]
                    Btheta += tmp3
                    
                    tmp1 = g[n, m] * sen[m, :] + h[n, m] * cos[m, :]
                    tmp2 = np.outer(tmp1, P[n, m, :])
                    tmp3 = tmp2 * m * a_over_r[n]
                    Bphi += tmp3
                    
        for theta in range(lat_len):
                Bphi[:, theta] /= sen_theta[theta]
        
        B = np.sqrt(Br**2 + Btheta**2 + Bphi**2)
        
    else:
        B = np.zeros((lon_len, lat_len))
        
        if comp == 'Br':
            for n in tqdm(range(1, nmax+1)):
                for m in range(n+1):
                    tmp1 = g[n, m] * cos[m, :] + h[n, m] * sen[m, :]
                    tmp2 = np.outer(tmp1, P[n, m, :])
                    tmp3 = tmp2 * (n+1) * a_over_r[n]
                    B += tmp3
        
        elif comp == 'Btheta':
            sen_theta = np.sin(np.deg2rad(90.0 - latitude))
            for n in tqdm(range(1, nmax+1)):
                for m in range(n+1):
                    tmp1 = g[n, m] * cos[m, :] + h[n, m] * sen[m, :]
                    tmp2 = np.outer(tmp1, dP[n, m, :] * sen_theta)
                    tmp3 = tmp2 * a_over_r[n]
                    B += tmp3
            
        else:
            sen_theta = np.sin(np.deg2rad(90.0 - latitude))
            for n in tqdm(range(1, nmax+1)):
                for m in range(n+1):
                    tmp1 = g[n, m] * sen[m, :] + h[n, m] * cos[m, :]
                    tmp2 = np.outer(tmp1, P[n, m, :])
                    tmp3 = tmp2 * m * a_over_r[n]
                    B += tmp3
            
            for theta in range(lat_len):
                B[:, theta] /= sen_theta[theta]
            
    return B.T


class DerivativesList(Enum):
    Lon = 1
    Lat = 2
    Alt = 3
    
    LonLon = 4
    LatLat = 5
    AltAlt = 6
    
    LonLat = 7
    LonAlt = 8
    
    LatLon = 9
    LatAlt = 10
    
    AltLon = 11
    AltLat = 12


def FieldDerivatives(hdegree, hheight, lon, lat, alt, comp, derivatives = None, binsize = 0.1):
    """
    Calculates the numerical derivatives of one component of the crustal magnetic field field model, following the equations:
        $f'(x) = \frac{f(x+h)-f(x)}{h}$
        $f''(x) = \frac{f(x+h)-2f(x)+f(x-h)}{h^2}$
    
    Parameters:
        hdegree: float
            A small value (h) for the degree change.
            
        hheight: float
            A small value (h) for the altitude change .

        lon: float, array
            The longitude range, or explicit values, in degrees.
            
        lat: float, array
            The latitude range, or explicit values, in degrees.
                        
        alt: float, array
            The altitude range, or explicit values, in km.
            
        comp: string
            The desired magnetic field component, in spherical coordinates. Options are 'Br', 'Btheta', 'Bphi', and 'Bt'.
        
        derivatives: list
            A list of the desired derivatives, that can be read by DerivativesList. Ex.:[MagFieldDerivative.Lon, MagFieldDerivative.LonLon]. 
            If None, the output will be every derivative. Default is None.
            
        binsize: float, list, optional
            The resolution of the grid. If a float, apply the same binsize for longitude and latitude. 
            If a list, the first value represents the longitude binsize and the second, the latitude binsize. 
        
    Returns:
        A list containing the desired derivatives, in the same order of the input.       
    """
    if derivatives is None:
        derivatives = [deriv for deriv in DerivativesList]
        return FieldDerivatives(hdegree, hheight, lon, lat, alt, comp, derivatives, binsize)
        
    else:
        if hasattr(lon, '__len__') is False or lon.size > 2:
            model = mag_components(lon, lat, alt, comp)
            
            lon1 = mag_components(lon + hdegree, lat, alt, comp)
            lon2 = mag_components(lon - hdegree, lat, alt, comp)
            
            colat1 = mag_components(lon, lat - hdegree, alt, comp)
            colat2 = mag_components(lon, lat + hdegree, alt, comp)
            
            alt1 = mag_components(lon, lat, alt + hheight, comp)
            alt2 = mag_components(lon, lat, alt - hheight, comp)
            
        else:
            model = model_map(lon, lat, alt, comp, binsize)
            
            lon1 = model_map(lon + hdegree, lat, alt, comp, binsize)
            lon2 = model_map(lon - hdegree, lat, alt, comp, binsize)
            
            colat1 = model_map(lon, lat - hdegree, alt, comp, binsize)
            colat2 = model_map(lon, lat + hdegree, alt, comp, binsize)
            
            alt1 = model_map(lon, lat, alt + hheight, comp, binsize)
            alt2 = model_map(lon, lat, alt - hheight, comp, binsize)
        
        out = []
        for deriv in derivatives:
            if deriv == DerivativesList.Lon:
                dlon = (lon1 - model) / hdegree
                out.append(dlon)
                
            if deriv == DerivativesList.Lat:
                dlat = (colat1 - model) / hdegree
                out.append(dlat)
                
            if deriv == DerivativesList.Alt:
                dalt = (alt1 - model) / hheight
                out.append(dalt)
            
            if deriv == DerivativesList.LonLon:
                dlon = (lon1 - model) / hdegree
                dlon2 = (dlon - (model - lon2) / hdegree) / hdegree
                out.append(dlon2)
            
            if deriv == DerivativesList.LatLat:
                dlat = (colat1 - model) / hdegree
                dlat2 = (dlat - (model - colat2) / hdegree) / hdegree
                out.append(dlat2)
            
            if deriv == DerivativesList.AltAlt:
                dalt = (alt1 - model) / hheight
                dalt2 = (dalt - (model - alt2) / hheight) / hheight
                out.append(dalt2)
            
            if deriv == DerivativesList.LonLat:
                dlon = (lon1 - model) / hdegree
                dlon_dlat = (dlon - (model - colat2) / hdegree) / hdegree
                out.append(dlon_dlat)
            
            if deriv == DerivativesList.LonAlt:
                dlon = (lon1 - model) / hdegree
                dlon_dalt = (dlon - (model - alt2) / hdegree) / hdegree
                out.append(dlon_dalt)
            
            if deriv == DerivativesList.LatLon:
                dlat = (colat1 - model) / hdegree
                dlat_dlon = (dlat - (model - lon2) / hdegree) / hdegree
                out.append(dlat_dlon)
             
            if deriv == DerivativesList.LatAlt:
                dlat = (colat1 - model) / hdegree
                dlat_dalt = (dlat - (model - alt2) / hdegree) / hdegree
                out.append(dlat_dalt)
            
            if deriv == DerivativesList.AltLon:
                dalt = (alt1 - model) / hheight
                dalt_dlon = (dalt - (model - lon2) / hheight) / hheight
                out.append(dalt_dlon)
            
            if deriv == DerivativesList.AltLat:
                dalt = (alt1 - model) / hheight
                dalt_dlat = (dalt - (model - colat2) / hheight) / hheight
                out.append(dalt_dlat)
                
        return out

    
    