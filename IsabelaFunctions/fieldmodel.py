'''
    Computes the magnetic field components from Langlais' spherical harmonic model at a certain altitude.
    
    References: 
        
        Whaler, K. A.; Gubbins, D., 1981. Spherical harmonic analysis of the geomagnetic field: an example of a 
        linear inverse problem, Geophysical Journal of the Royal Astronomical Society, 65, 645-693. 
        DOI: doi.org/10.1111/j.1365-246X.1981.tb04877.x
        
        Langlais, B.; Thébault, E.; Houliez, A.; Purucker, M.; Lillis, R. J., 2019. A new model of the crustal
        magnetic field of Mars using MGS and MAVEN, Journal of Geophysical Research: Planets, 124, 6, 1542-1569.
        DOI: doi.org/10.1029/2018JE005854
'''

import pyshtools as sh
import numpy as np
import scipy as sp
import scipy.io
import matplotlib.pyplot as plt


# Maximum number of degree and order of the spherical harmonics
nmax = 134

# given the input:
lat0 = -5.0
long0 = 35.0
alt0 = 300.

# Co-latitude, in rad
theta = np.deg2rad(90.0 - lat0)

# Longitude, in rad
phi = np.deg2rad(long0)

# Calculate the Schmidt semi-normalized associated Legendre polynomials P(cos(theta)) and first derivatives

x = np.cos(theta)

def legendre_schmidt(theta, nmax, x):
    P = np.zeros((nmax+1,nmax+1))
    
    P[0, 0] = 1.0
    
    twoago = 0.0
    for i in range(1, nmax+1):
        P[i, 0] = (x * (2.0*i - 1.0) * P[i-1, 0] - (i-1.0) * twoago) / i
        twoago = P[i-1, 0]
        
    Cm = np.sqrt(2.0)
    for m in range(1, nmax+1):
        Cm /=  np.sqrt(2.0*m * (2.0*m -1.0))
        P[m, m] = (1.0 - x**2)**(0.5 * m) * Cm
        
        for i in range(1, m):
            P[m, m] *= (2.0*i + 1.0)
       
        if nmax > m:
            twoago = 0.0
            for i in range(m+1, nmax+1):
                P[i, m] = (x * (2.0*i - 1.0) * P[i-1, m] - np.sqrt((i+m-1.0) * (i-m-1.0)) * twoago) / np.sqrt((i**2 - m**2))
                twoago = P[i-1, m]
                
    return P

P = legendre_schmidt(theta, nmax, 0.5)

def legendre_schmidt_2(theta, nmax, x):
    P = np.zeros((nmax+1,nmax+1))
    
    # Initial values for first recursion:
    P[0, 0] = 1.0
    P[1, 0] = x
    P[1, 1] = np.sqrt(1 - x**2)
    
    for m in range(2, nmax+1):
        P[m, m] = np.sqrt((2*m-1) / (2*m)) * P[1, 1] * P[m-1, m-1]
        
        for n in range(2, nmax+1):
            P[n, m] = 
            
    for n in range(2, nmax+1):
        P[n, 0] = ((2*(n-1)+1) * x * P[n-1, 0] - (n-1) * P[n-2, 0])/n
        
        #for m in range(1, n+1):
        #    P[n, m] = (n-(m-1) * x * P[n, m-1] - (n+(m-1)) * P[n-1, m-1]) / P[1, 1]
            
        #    # Schmidt semi-normalization: 
        #    P[n, m] *= (-1)**m * np.sqrt(2 * (np.math.factorial(n-m)) / np.math.factorial(n+m))
        
        for m in range(1, n):
            P[n, m] = (2*n-1) * x * P[n-1, m] - np.sqrt((n-1)**2-m**2)
            
    return P

P2 = legendre_schmidt_2(theta, nmax, 0.5)

##############################################




Br = []
Btheta = []
Bphi = []
for th in theta:
    pnm0, dpnm0 = sh.legendre.PlmSchmidt_d1(nmax, np.cos(th))
    
    pnm = dpnm = np.zeros((nmax+1, nmax+1))
    ind = np.tril_indices(len(pnm))
    
    pnm[ind] = pnm0
    dpnm[ind] = dpnm0
    
    del pnm0, dpnm0
    
    # Import the coefficient files
    from IsabelaFunctions.langlais_coeff import gnm, hnm
    
    # Mars' radius
    a = 3393.5
    r = alt0 + a
    
    # Calculate the potential
   # V = a * sum((a/r)**(l+1) * (glm[l, m]*np.cos(m*phi) + hlm[l, m]*np.sin(m*phi)) * plm[l, m] \
   #     for l in range(1, lmax+1) for m in range(l+1))
        
    # Calculate the derivatives
    
    Br.append(sum((n+1) * (a/r)**(n+2) * (gnm[n, m]*np.cos(m*phi) + hnm[n, m]*np.sin(m*phi)) * pnm[n, m] \
                for n in range(1, nmax+1) for m in range(n+1)))
    
    Btheta.append(np.sin(th) * sum((a/r)**(n+2) * (gnm[n, m]*np.cos(m*phi) + hnm[n, m]*np.sin(m*phi)) * \
                                        dpnm[n, m] for n in range(1, nmax+1) for m in range(n+1)))
    
    Bphi.append(-1/np.sin(th) * sum((a/r)**(n+2) * pnm[n, m] * m *(- gnm[n, m]*np.sin(m*phi) + \
                                        hnm[n, m]*np.cos(m*phi)) for n in range(1, nmax+1) for m in range(n+1)))
Br = np.array(Br)
Btheta = np.array(Btheta)
Bphi = np.array(Bphi)

plt.imshow(Br, extent = [30, 40, -10, 0], origin = 'lower', vmin = -200, vmax = 200)
plt.xlim(30, 40)
plt.ylim(-10, 0)
plt.colorbar()
# Test

data = sp.io.readsav('/home/oliveira/ccati_mexuser/LANGLAIS_Matrices/A6/LANGLAIS_BR_ALT_300_RES_01.bin')
brmodel = data['zbins']

im = plt.imshow(brmodel, extent = [20, 85, -15, 15], origin = 'lower')
plt.xlim(30, 40)
plt.ylim(-10, 0)
plt.colorbar()

#data = sp.io.readsav('/home/oliveira/ccati_mexuser/LANGLAIS_Matrices/A2/LANGLAIS_BR_ALT_200_RES_01.bin')
#brmodel = data['zbins']

#im = plt.imshow(brmodel, extent = [70, 90, 25, 45], origin = 'lower', vmin = -20, vmax = 20)
#plt.colorbar()







