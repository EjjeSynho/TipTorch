#%%
from scipy.special import jv
import numpy as np


#%%
#
#inputs (in SI units)
#

# wavelength = 1e-10 # 500e-9 # 1e-10
# distance = 1.0
# aperture_diameter = 1e-6 # 1e-3 # 1e-6

# wavelength = 5000e-10 # 500e-9 # 1e-10
# distance = 1.0
# aperture_diameter = 500e-6 # 1e-3 # 1e-6

wavelength = 1.24e-10 # 10keV
distance = 3.6
aperture_diameter = 40e-6 # 1e-3 # 1e-6


# sin(theta) ~ theta

sin_theta = 1.22*wavelength/aperture_diameter

print("Angular radius of first Airy ring: %15.10f rad"%sin_theta)
print("spatial position at first minimum: %15.10f mm"%(sin_theta*distance*1e3))

sin_theta_array = np.linspace(-3*sin_theta, 3*sin_theta, 600)
x = 2*np.pi/wavelength * aperture_diameter/2 * sin_theta_array

#2D array of sin(theta) values
x_, y_ = np.meshgrid(x, x)
r = np.sqrt(x_**2 + y_**2)

x_over_pi = x / np.pi
electric_field = 2*jv(1, x) / x
electric_field_2d = 2*jv(1, r) / r

intensity    = electric_field**2
intensity_2d = electric_field_2d**2


#%%
from matplotlib import pylab as plt

plt.figure(1)
plt.plot(x_over_pi, intensity**0.5)
plt.xlim(-3,3)
plt.yscale('linear')
plt.xlabel(r"$\frac{D}{\lambda} sin(\theta)$")
plt.ylabel(r"$\sqrt{I \, /\, I_0}$")
plt.title("Airy pattern profile")

plt.savefig("C:/Users/akuznets/Desktop/thesis_results/general/airy_pattern.pdf")

plt.show()
#%%
from matplotlib.colors import LogNorm

extent = [
    x_over_pi.min(), x_over_pi.max(),
    x_over_pi.min(), x_over_pi.max()
]

plt.figure(4)
# plt.imshow(intensity_2d, cmap='hot', interpolation='nearest', norm=LogNorm())
# Change extents
plt.imshow(1-(intensity_2d**.5), interpolation='bicubic', cmap='Reds', extent=extent)

plt.savefig("C:/Users/akuznets/Desktop/thesis_results/general/airy_pattern_2d.pdf")

plt.colorbar()
plt.show()

