#%%
# Original data: Ciddor 1996, https://doi.org/10.1364/AO.35.001566

# The original script is taken from : https://github.com/polyanskiy/refractiveindex.info-scripts/blob/master/scripts/Ciddor%201996%20-%20air.py

# CHANGELOG
# 2017-11-23 [Misha Polyanskiy] original version
# 2024-03-03 [Misha Polyanskiy] minor refactoring

###############################################################################
from math import exp

def Z(T, p, xw): #compressibility
    t  = T - 273.15
    a0 = 1.58123e-6   # K·Pa^-1
    a1 = -2.9331e-8   # Pa^-1
    a2 = 1.1043e-10   # K^-1·Pa^-1
    b0 = 5.707e-6     # K·Pa^-1
    b1 = -2.051e-8    # Pa^-1
    c0 = 1.9898e-4    # K·Pa^-1
    c1 = -2.376e-6    # Pa^-1
    d  = 1.83e-11     # K^2·Pa^-2
    e  = -0.765e-8    # K^2·Pa^-2
    return 1-(p/T)*(a0+a1*t+a2*t**2+(b0+b1*t)*xw+(c0+c1*t)*xw**2) + (p/T)**2*(d+e*xw**2)


def n_air(lmbd, t=20, p=1e5, h=0, xc=0):
    # lmbd: wavelength, 0.3 to 1.69 microns, in [m]
    # t:    temperature, -40 to +100 [C]
    # p:    pressure, 80000 to 120000 [Pa]
    # h:    fractional humidity, 0 to 1
    # xc:   CO2 concentration, 0 to 2000 [ppm]

    sigma = 1e-6 / lmbd # micron^-1
    
    T = t + 273.15    # Temperature C -> K
    R = 8.314510      # gas constant, J/(mol K)
    
    k0 = 238.0185     # μm^-2
    k1 = 5792105      # μm^-2
    k2 = 57.362       # μm^-2
    k3 = 167917       # μm^-2
 
    w0 = 295.235      # μm^-2
    w1 = 2.6422       # μm^-2
    w2 = -0.032380    # μm^-4
    w3 = 0.004028     # μm^-6
    
    A = 1.2378847e-5  # K^-2
    B = -1.9121316e-2 # K^-1
    C = 33.93711047
    D = -6.3431645e3  # K
    
    alpha = 1.00062
    beta  = 3.14e-8   #Pa^-1,
    gamma = 5.6e-7    #C^-2

    # saturation vapor pressure of water vapor in air at temperature T (Pa)
    svp = exp(A*T**2 + B*T + C + D/T) if t >= 0 else 10**(-2663.5/T + 12.537)
    
    # enhancement factor of water vapor in air
    f = alpha + beta*p + gamma*t**2
    
    # molar fraction of water vapor in moist air
    xw = f*h*svp/p
    
    # refractive index of standard air at 15 C, 101325 Pa, 0% humidity, 450 ppm CO2
    nas = 1 + (k1/(k0-sigma**2)+k3/(k2-sigma**2))*1e-8
    
    # refractive index of standard air at 15 C, 101325 Pa, 0% humidity, xc ppm CO2
    naxs = 1 + (nas-1) * (1+0.534e-6*(xc-450))
    
    # refractive index of water vapor at standard conditions (20 C, 1333 Pa)
    nws = 1 + 1.022*(w0+w1*sigma**2+w2*sigma**4+w3*sigma**6)*1e-8
    
    Ma = 1e-3*(28.9635 + 12.011e-6*(xc-400)) #molar mass of dry air, kg/mol
    Mw = 0.018015                            #molar mass of water vapor, kg/mol
    
    Za = Z(288.15, 101325, 0)                #compressibility of dry air
    Zw = Z(293.15, 1333, 1)                  #compressibility of pure water vapor
    
    # Eq.4 with (T, P, xw) = (288.15, 101325, 0)
    rhoaxs = 101325*Ma/(Za*R*288.15)           #density of standard air
    
    # Eq 4 with (T, P, xw) = (293.15, 1333, 1)
    rhows  = 1333*Mw/(Zw*R*293.15)             #density of standard water vapor
    
    # two parts of Eq.4: rho = rhoa + rhow
    rhoa   = p*Ma/(Z(T,p,xw)*R*T)*(1-xw)       #density of the dry component of the moist air    
    rhow   = p*Mw/(Z(T,p,xw)*R*T)*xw           #density of the water vapor component
    
    nprop = 1 + (rhoa/rhoaxs)*(naxs-1) + (rhow/rhows)*(nws-1)
    
    return nprop


#%%