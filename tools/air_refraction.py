#%%
# Original data: Ciddor 1996, https://doi.org/10.1364/AO.35.001566

# The original script is taken from : https://github.com/polyanskiy/refractiveindex.info-scripts/blob/master/scripts/Ciddor%201996%20-%20air.py

# CHANGELOG
# 2017-11-23 [Misha Polyanskiy] original version
# 2024-03-03 [Misha Polyanskiy] minor refactoring

###############################################################################

import torch
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
class AirRefractiveIndexCalculator:
    def __init__(self, device=None, dtype=torch.float64):
        # Use provided a device or CPU (default).
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype

        # Create constants as tensors on the target device.
        self.t_0   = torch.tensor(273.15,   device=self.device, dtype=self.dtype)  # 0°C offset (Kelvin)
        self.R     = torch.tensor(8.314510, device=self.device, dtype=self.dtype)  # Gas constant (J/(mol·K))

        # Refractivity coefficients.
        self.k0 = torch.tensor(238.0185,  device=self.device, dtype=self.dtype)
        self.k1 = torch.tensor(5792105.0, device=self.device, dtype=self.dtype)
        self.k2 = torch.tensor(57.362,    device=self.device, dtype=self.dtype)
        self.k3 = torch.tensor(167917.0,  device=self.device, dtype=self.dtype)

        self.w0 = torch.tensor(295.235,   device=self.device, dtype=self.dtype)
        self.w1 = torch.tensor(2.6422,    device=self.device, dtype=self.dtype)
        self.w2 = torch.tensor(-0.032380, device=self.device, dtype=self.dtype)
        self.w3 = torch.tensor(0.004028,  device=self.device, dtype=self.dtype)

        self.A = torch.tensor(1.2378847e-5,  device=self.device, dtype=self.dtype)
        self.B = torch.tensor(-1.9121316e-2, device=self.device, dtype=self.dtype)
        self.C = torch.tensor(33.93711047,   device=self.device, dtype=self.dtype)
        self.D = torch.tensor(-6.3431645e3,  device=self.device, dtype=self.dtype)

        self.alpha = torch.tensor(1.00062, device=self.device, dtype=self.dtype)
        self.beta  = torch.tensor(3.14e-8, device=self.device, dtype=self.dtype)
        self.gamma = torch.tensor(5.6e-7,  device=self.device, dtype=self.dtype)

        # Coefficients for compressibility.
        self.a0 = torch.tensor(1.58123e-6, device=self.device, dtype=self.dtype)
        self.a1 = torch.tensor(-2.9331e-8, device=self.device, dtype=self.dtype)
        self.a2 = torch.tensor(1.1043e-10, device=self.device, dtype=self.dtype)
        self.b0 = torch.tensor(5.707e-6,   device=self.device, dtype=self.dtype)
        self.b1 = torch.tensor(-2.051e-8,  device=self.device, dtype=self.dtype)
        self.c0 = torch.tensor(1.9898e-4,  device=self.device, dtype=self.dtype)
        self.c1 = torch.tensor(-2.376e-6,  device=self.device, dtype=self.dtype)
        self.d  = torch.tensor(1.83e-11,   device=self.device, dtype=self.dtype)
        self.e  = torch.tensor(-0.765e-8,  device=self.device, dtype=self.dtype)

        # A constant for base-10 calculations.
        self.ten = torch.tensor(10.0, device=self.device, dtype=self.dtype)

        # Precompute standard-state compressibility factors.
        # Dry air: T = 288.15 K, p = 101325 Pa, xw = 0.
        std_T_air = torch.tensor(288.15,   device=self.device, dtype=self.dtype)
        std_p_air = torch.tensor(101325.0, device=self.device, dtype=self.dtype)
        std_xw_air = torch.tensor(0.0,     device=self.device, dtype=self.dtype)
        self.Za = self._Z(std_T_air, std_p_air, std_xw_air)

        # Water vapor: T = 293.15 K, p = 1333 Pa, xw = 1.
        std_T_wv = torch.tensor(293.15, device=self.device, dtype=self.dtype)
        std_p_wv = torch.tensor(1333.0, device=self.device, dtype=self.dtype)
        std_xw_wv = torch.tensor(1.0,   device=self.device, dtype=self.dtype)
        self.Zw = self._Z(std_T_wv, std_p_wv, std_xw_wv)

        # Precompute standard densities.
        # For dry air, assume standard CO2 concentration of 450 ppm.
        std_xc = torch.tensor(450.0, device=self.device, dtype=self.dtype)
        self.Ma_std = 1e-3 * (28.9635 + 12.011e-6 * (std_xc - 400))  # Molar mass of dry air (kg/mol)
        self.Mw = torch.tensor(0.018015, device=self.device, dtype=self.dtype)  # Molar mass of water vapor (kg/mol)
        self.rhoaxs = std_p_air * self.Ma_std / (self.Za * self.R * std_T_air)
        self.rhows  = std_p_wv  * self.Mw     / (self.Zw * self.R * std_T_wv)
        
        self.t_def  = torch.tensor(20,  device=self.device, dtype=self.dtype)
        self.p_def  = torch.tensor(1e5, device=self.device, dtype=self.dtype)
        self.h_def  = torch.tensor(0,   device=self.device, dtype=self.dtype)
        self.xc_def = torch.tensor(0,   device=self.device, dtype=self.dtype)
        

    def _Z(self, T, p, xw):
        """
        Compute the compressibility factor Z using only tensor operations.
        Assumes T, p, and xw are already tensors.
        """
        t = T - self.t_0
        return 1 - (p / T) * (
            self.a0 + self.a1 * t + self.a2 * t**2 +
            (self.b0 + self.b1 * t) * xw +
            (self.c0 + self.c1 * t) * xw**2
        ) + (p / T)**2 * (self.d + self.e * xw**2)


    def n_air(self, lmbd, t=None, p=None, h=None, xc=None):
        """
        Compute the refractive index of air.

        Parameters:
          lmbd: wavelength in meters (0.3 to 1.69 microns expected)
          t: temperature in Celsius (default 20)
          p: pressure in Pa (default 1e5)
          h: fractional humidity (0 to 1; default 0)
          xc: CO2 concentration in ppm (default 0)

        Returns:
          Refractive index (tensor)
        """

        # Set default values for missing parameters.
        t  = self.t_def  if t  is None else t
        p  = self.p_def  if p  is None else p
        h  = self.h_def  if h  is None else h
        xc = self.xc_def if xc is None else xc

        # Compute sigma in μm^-1. (The division will use tensor math if lmbd is a tensor.)
        sigma = 1e-6 / lmbd

        # Temperature in Kelvin.
        T = t + self.t_0

        # Saturation vapor pressure (Pa). Use torch.where to select the proper branch.
        # (Note: All operations below use element‐wise tensor ops.)
        svp_positive = torch.exp(self.A * T**2 + self.B * T + self.C + self.D / T)
        svp_negative = self.ten**(-2663.5 / T + 12.537)
        svp = torch.where(t >= 0, svp_positive, svp_negative)

        # Enhancement factor.
        f = self.alpha + self.beta * p + self.gamma * t**2

        # Molar fraction of water vapor.
        xw = f * h * svp / p

        # Refractive index of standard dry air at 15°C, 101325 Pa, 0% humidity.
        nas = 1 + (self.k1 / (self.k0 - sigma**2) + self.k3 / (self.k2 - sigma**2)) * 1e-8
        naxs = 1 + (nas - 1) * (1 + 0.534e-6 * (xc - 450))
        # Refractive index of water vapor at standard conditions (20°C, 1333 Pa).
        nws = 1 + 1.022 * (self.w0 + self.w1 * sigma**2 +
                           self.w2 * sigma**4 + self.w3 * sigma**6) * 1e-8

        # Molar masses.
        Ma = 1e-3 * (28.9635 + 12.011e-6 * (xc - 400))
        Mw = self.Mw

        # Compute compressibility factor only once.
        Z_val = self._Z(T, p, xw)

        # Densities of the dry air and water vapor components.
        rhoa = p * Ma / (Z_val * self.R * T) * (1 - xw)
        rhow = p * Mw / (Z_val * self.R * T) * xw

        # Combine the contributions.
        nprop = 1 + (rhoa / self.rhoaxs) * (naxs-1) + (rhow / self.rhows) * (nws-1)
        return nprop


    def __call__(self, *args, **kwargs):
        return self.n_air(*args, **kwargs)



#%%
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create an instance of the calculator.
calc = AirRefractiveIndexCalculator(device=device, dtype=torch.float64)

# Define input parameters.
wavelength = 0.5e-6  # 0.5 microns in meters.
temperature = 20     # Celsius.
pressure = 1e5       # Pa.
humidity = 0.5       # 50% relative humidity.
co2 = 450            # ppm.

# Compute the refractive index.
n = n_air(lmbd=wavelength, t=temperature, p=pressure, h=humidity, xc=co2)

# print("Refractive index:", n.item())
print("Refractive index:", n)
"""
