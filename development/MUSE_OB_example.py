#%%
%reload_ext autoreload
%autoreload 2

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tiptorch._config import default_device, project_settings
from tools.observations import MUSEObservation

from itertools import product
from pathlib import Path

# Define the location of your NFM data. It can be whenever. One option is to add it to the project config file 
MUSE_DATA_FOLDER = Path(project_settings["MUSE_data_folder"])
device = default_device

#%%
# To run properly, the model requires:
# - The raw MUSE NFM data cube (fits.fz)
# - The reduced MUSE NFM data cube (fits)

# Both should be temporally associated. If your reduced cube is combined from several OBs, it is not a deal breaker, just provide any relevant raw data file within the observation block. The model will extract the necessary metadata from it. In this case, the PSF guessing might be less accurate, but it can be later improved with PSF fitting (provided the field has point sources).
# Based on the raw headers and reduced cubes, it generates a cached data cube (pickle) that stores necessary OB metadata. Once generated, it is stored and subsequently re-used.

# Define the paths to the raw and reduced MUSE NFM cubes. The cached data cube will be generated based on them
data_folder = MUSE_DATA_FOLDER / 'clumpy_galaxies/' # change to your actual path with the MUSE NFM data

cube_path  = data_folder / "reduced_cubes/CUBE_0002.fits"
raw_path   = data_folder / "raw_data/MUSE.2023-04-27T05_11_59.783.fits.fz"
cache_path = data_folder / "cached_cubes/CUBE_0002.pickle"

ob = MUSEObservation(raw_path, cube_path, cache_path, device=device)

# MUSEObservation is a master class that implements the MUSE NFM observation block (OB) simulation. It is a swiss-knife tool that:
# - Reads the OB data
# - Cleans up the data from artifacts
# - Performs PSF simulations
# - Performs entire field simulation
# - Does spectral disentangling of the sources
# - Performs PSF fitting
# - Stores the results for later re-use and analysis.
# It is designed to be flexible and can be used in various ways, depending on the user's needs.

# The general workflow is as follows:
# - Init OB instance by providing raw and reduced cubes, save cache
# - Init sources
# - Init simulation (i.e predict initial PSFs, extract sources information, compute bad pixels mask)
# The steps below can be executed in an arbitrary order. But generally:
# - Simulate the field (sparse or full spectrum)
# - Optionally: fit PSF model to improve the PSF shape and re-simulate the field
# - Save the simulation results for later re-use
# - Do your analysis

#%%
ob.DetectSources(nsigma=35, threshold='auto')
# ob.DetectSources(nsigma=35, threshold=4e2, verbose=True)
# ob.AddSources([[100, 200]], weights=0.0)
# NOTE: If some sources must be filtered out, do it here by modifying the ob.sources_table before initializing the sources for simulation

# ob.DeleteSources([1, 2])  # Example: delete sources with IDs 1 and 2

# Sometimes, hot pixels are detected as sources. To filter them out, use the FilterHotPixels() function.
hot_pixels = ob.FilterHotPixels(filter_sources=True, verbose=True)

ob.DisplayField()

# Note, that the source detection in TipTorch is very crude, and it is better to offload this task to a more sophisticated tool. Alternatively, one can provide sources directly by setting 'ob.sources_table = <your_table>' before calling InitSimulation(). In this case, the source detection step can be skipped. The table must contain at least the following columns: ['id', 'x_peak', 'y_peak', 'XXX', 'weight'], where
# - 'id' is the source ID used to index it (not necessarily sequential or numerical)
# - 'x_peak' and 'y_peak' are the source coordinates in pixels (TODO: add WCS support)
# - 'XXX' is the averaged source flux. Can be approximate, and used only for technical purposes so have an initial guess on which sources are bright and which are faint. The model will re-estimate the fluxes later during the simulation.
# - 'weight' is the relative weight of the source in the PSF fitting (1.0 by default)

# Note, if you want to fit PSF in one place but simulate sources in another region of the field, just 'AddSource(..., weight=0.0)'. Zero weight tells the model that this source should not be used in fitting and must be only simulated. This is useful when the field has a bright point source, but the scientific interest is in another region of the field. The model will handle the extrapolation of the PSF shape to the source location automatically.

#%%
# The Swiss-knife function that:
# - Initializes the PSF model
# - Predicts PSF shape based on OB metadata and system telemetry
# - Extracts the sources' spectra and other information based on the coordinates from sources table
# - Computes bad pixels mask

# It must be called after detecting and optionally modifying the sources table. All later modification to the source data require re-running InitSimulation().
# Also note, that at this stage, source spectra are extracted approximately, as in many real cases, the sources' flux disentangling is required first to estimate the true spectra. This to be done later in this example.

ob.InitSimulation()

# Ability to blindly predict PSF based on the OB metadata is a powerful tool that allows to post-process the fields devoid of any point sources. However, if the field has point sources, the results can be further improved with fitting (see below).

#%%
# Provided the scientific field has point sources, the initial guess of the PSF from the previous step can be further improved with fitting.

ob.FitPSFModel(fit=['PSF', 'astrometry', 'photometry'], repeat=3, max_iter=200)

# In practice, repeating the PSF fitting several times is recommended to push the model out of the local minima and also update flux normalization based on the new updated shape. But this example is rather maximalistic, and in many cases, 2 repetitions with 'max_iter=75-100' can be sufficient.

# Note, that you can also limit fitting to astrometry and/or photometry only, for example:

# ob.FitPSFModel(fit=['astrometry'], repeat=3, max_iter=200)
# ob.FitPSFModel(fit=['astrometry', 'photometry'], repeat=3, max_iter=200)

#%%
# Note, the 'full_spectrum=False (by default)' flag. This is because simulation uses so called 'sparse' and 'full' spectra and cubes. All heavy computations (e.g. PSF fitting) are performed on the sparse cubes & spectra to save speed and compute resources. Sparse data contain 7 spectrally averaged bins of Δλ = XXX. Sparse data can also be used for fast previews like here. However, for the final scientific analysis, the full spectra should be used.

model = ob.SimulateField(full_spectrum=False)

#%%
# Display the simulation results. The 'plot_profiles' option allows to visualize the median sources' profiles and their fits.
ob.DisplaySimulation(plot_profiles=True)

#%%
# Now, the model values obtained with the sparse spectra can be used for the full-spectrum simulation. Note, the 'disentangle_spectra' option. By default, the function that simulates the field also performs the disentangling of the sources' by solving the linear system like: ∑_λ PSF_λ * F_λ + b_λ = I_data_λ w.r.t. to F_λ (approach similar to XXX). This helps when dealing with crowded fields. In this particular example, it's not that critical.

_ = ob.SimulateField(full_spectrum=True, disentangle_spectra=True)

# %%
ob.DisplaySimulation(plot_profiles=True, full_spectrum=True, focus_on_src=0)  # Focus on the first source

#%%
ob.PlotSourceSpectra()

#%%
ob.SaveState(data_folder/'metadata'/(cache_path.stem+'.pkl'), save_full_cubes=False)

#%%
ob = MUSEObservation(raw_path, cube_path, cache_path, device=device)

ob.LoadState(data_folder/'metadata'/(cache_path.stem+'.pkl'), device=device)
_ = ob.SimulateField(full_spectrum=True)
ob.DisplaySimulation(plot_profiles=True, full_spectrum=True, focus_on_src=0)  # Focus on the first source

# %%
