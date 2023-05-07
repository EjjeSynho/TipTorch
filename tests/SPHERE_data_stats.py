#%%
%reload_ext autoreload
%autoreload 2

import sys
sys.path.insert(0, '..')

import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from globals import SPHERE_DATA_FOLDER

with open(SPHERE_DATA_FOLDER+'sphere_df.pickle', 'rb') as handle:
    psf_df = pickle.load(handle)

#%%
'Filename'
'Date'
'Observation'
'Airmass'
'r0 (SPARTA)'
'Seeing (SPARTA)'
'Seeing (MASSDIMM)'
'Seeing (DIMM)'
'FWHM'
'Strehl'
'Turb. speed'
'Wind direction (header)'
'Wind direction (MASSDIMM)'
'Wind speed (header)'
'Wind speed (SPARTA)'
'Wind speed (MASSDIMM)'
'Tau0 (header)'
'Tau0 (SPARTA)'
'Tau0 (MASSDIMM)'
'Tau0 (MASS)'
'Pressure'
'Humidity'
'Temperature'
'Nph WFS'
'Rate'
'λ left (nm)'
'λ right (nm)'
'Δλ left (nm)'
'Δλ right (nm)'
'mag V'
'mag R'
'mag G'
'mag J'
'mag H'
'mag K'
'Class A'
'Class B'
'Class C'
'LWE'
'doubles'
'No coronograph'
'invalid'

#%%
# print histogramm using sns
import seaborn as sns

entry = 'Wind speed (SPARTA)'
# sns.kdeplot(data=psf_df, x=entry)
sns.displot(data=psf_df, x=entry, kde=True)

#%%

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

# Generate some random data
np.random.seed(42)
data = np.random.randn(100)

# Reshape the data to be 2D (required by Scikit-learn)
data_reshaped = data[:, np.newaxis]

# Instantiate and fit the KernelDensity estimator
kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(data_reshaped)

# Evaluate the density on a set of points
x_points = np.linspace(-3, 3, 1000)[:, np.newaxis]
log_density = kde.score_samples(x_points)

# Plot the KDE
plt.plot(x_points, np.exp(log_density))
plt.show()
#%%


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

# Create a sample pandas DataFrame
data = {'column1': np.random.randn(100)}
df = pd.DataFrame(data)

# Reshape the data to be 2D (required by Scikit-learn)
data_reshaped = df['column1'].values[:, np.newaxis]

# Instantiate and fit the KernelDensity estimator
kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(data_reshaped)

# Evaluate the density on a set of points
x_points = np.linspace(min(df['column1']), max(df['column1']), 1000)[:, np.newaxis]
log_density = kde.score_samples(x_points)

# Convert log density to probability
probability = np.exp(log_density)

# Find the boundaries where the probability is below 1%
boundary_indices = np.where(probability < 0.025)

# Extract the boundary values
boundaries = x_points[boundary_indices]

print("Boundaries where probability is below 1%:")
print(boundaries)

# Plot the KDE and the boundaries
plt.plot(x_points, probability)
plt.axhline(y=0.01, color='r', linestyle='--')
plt.scatter(boundaries, [0.01]*len(boundaries), color='r', marker='x')
plt.show()



#%%
# Plotting the KDE Plot
sns.kdeplot(data = psf_df,
            x='Tau0 (header)',
            y='Wind speed (MASSDIMM)',
            color='r', fill=True, Label='Iris_Setosa',
            cmap="Reds", thresh=0.02)


#%%
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import poisson, nbinom
from scipy.optimize import curve_fit

# Generate sample data with a hump-like shape
data = np.concatenate((np.random.poisson(3, 2000), np.random.normal(8, 1, 1000)))

plt.hist(data, bins=200)

# # Fit Poisson distribution to data
# mu = data.mean()
# poisson_fit = poisson(mu)

# # Fit Negative Binomial distribution to data
# def nbinom_fit(x, r, p):
#     return nbinom(r, p).pmf(x)
# params, _ = curve_fit(nbinom_fit, np.arange(data.max()+1), np.histogram(data, bins=data.max().astype('uint')+2, density=True)[0])
# r, p = params
# nbinom_fit = nbinom(r, p)

# # Create probability density plot with fitted distributions
# sns.kdeplot(data, shade=True, label="Data")
# x = np.arange(0, data.max()+1)
# plt.plot(x, poisson_fit.pmf(x), label="Poisson")
# plt.plot(x, nbinom_fit.pmf(x), label="Negative Binomial")
# plt.legend()
# plt.show()

#%%


import numpy as np
from scipy.stats import gaussian_kde

# Generate sample data
data = np.random.normal(0, 1, 1000)

# Create kernel density estimate
kde = gaussian_kde(data)

# Evaluate kde at a set of points
x = np.linspace(-3, 3, 100)
y = kde(x)

# Print the density values as an array
plt.plot(x,y)
plt.hist(data, bins=100, density=True)
# %%

import numpy as np
import seaborn as sns
from scipy.stats import gamma
from scipy.optimize import curve_fit

# Generate sample data from a gamma distribution
data = gamma.rvs(a=2, size=1000)

# Create kernel density estimate
kde = gaussian_kde(data)

# Evaluate kde at a set of points
x = np.linspace(data.min(), data.max(), 100)
y = kde(x)

# plt.hist(data, bins=100)

plt.plot(x,y)

#%%

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Generate sample data
data = np.random.normal(0, 1, 10000)

counts, bins_raw = np.histogram(data, bins=1000, range=None, density=None, weights=None)
bins = 0.5*(bins_raw[1:]+bins_raw[:-1])

norma = np.trapz(counts, bins)
counts = counts/norma

# Define a function to fit the gamma distribution
gamma_fit = lambda x, a, loc, scale: gamma.pdf(x, a, loc, scale)

# params_0 = [10, -1, 0.15]

bins = np.linspace(-30, 30, 1000)

params_0 = [200, -10, 0.1]

# Fit the gamma distribution to the data
# params, cov = curve_fit(gamma_fit, bins, counts, p0=[10, -1, 0.15], method='trf')
# a, loc, scale = params
# gamma_fit = gamma(a, loc=loc, scale=scale)

# Plot the histogram of the data and the fitted gamma distribution
# sns.histplot(data, bins=100, kde=False, label="Data", common_norm=True)
plt.plot(bins, gamma_fit(bins, *params_0), label="Gamma Fit")
# plt.plot(bins, counts, label="Gamma Fit")
plt.legend()
plt.show()
