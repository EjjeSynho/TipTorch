#%%
try:
    ipy = get_ipython()        # NameError if not running under IPython
    if ipy:
        ipy.run_line_magic('reload_ext', 'autoreload')
        ipy.run_line_magic('autoreload', '2')
        import linecache
        ipy.events.register('post_execute', lambda: linecache.clearcache())
except NameError:
    pass

import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tiptorch._config import default_device, project_settings

import matplotlib.pyplot as plt

from tools.observations import MUSEObservation

from pathlib import Path

# Define the location of your NFM data. It can be whenever. One option is to add it to the project config file 
MUSE_DATA_FOLDER = Path(project_settings["MUSE_data_folder"])

device = default_device


#%%
# Define the paths to the raw and reduced MUSE NFM cubes. The cached data cube will be generated based on them
# data_folder = MUSE_DATA_FOLDER / 'quasars/' # change to your actual path with the MUSE NFM data
data_folder = MUSE_DATA_FOLDER / 'clumpy_galaxies/' # change to your actual path with the MUSE NFM data

if not isinstance(data_folder, Path):
    data_folder = Path(data_folder)

# raw_path   = data_folder / "J0259/MUSE.2024-12-05T03_15_37.598.fits.fz"
# cube_path  = data_folder / "J0259/J0259-0901_all.fits"
# cache_path = data_folder / "J0259/J0259-0901_all.pickle"

# cube_path  = data_folder / "J0144/J0144-5745.fits"
# cache_path = data_folder / "J0144/J0144-5745_cache.pickle"
# raw_path   = data_folder / "J0144/J0144_raw.114.fits.fz"

cube_path  = data_folder / "reduced_cubes/CUBE_0001.fits"
raw_path   = data_folder / "raw_data/MUSE.2023-04-27T04_56_21.169.fits.fz"
cache_path = data_folder / "reduced_telemetry/CUBE_0001.pickle"

# cube_path  = data_folder / "reduced_cubes/CUBE_0002.fits"
# raw_path   = data_folder / "raw_data/MUSE.2023-04-27T05_11_59.783.fits.fz"
# cache_path = data_folder / "reduced_telemetry/CUBE_0002.pickle"

# cube_path  = data_folder / "reduced_cubes/CUBE_0003.fits"
# raw_path   = data_folder / "raw_data/MUSE.2023-04-27T05_28_41.014.fits.fz"
# cache_path = data_folder / "reduced_telemetry/CUBE_0003.pickle"

# cube_path  = data_folder / "reduced_cubes/CUBE_0021.fits"
# raw_path   = data_folder / "raw_data/MUSE.2023-06-17T00_04_47.319.fits.fz"
# cache_path = data_folder / "reduced_telemetry/CUBE_0021.pickle"

ob = MUSEObservation(raw_path, cube_path, cache_path, device=device)

ob.DetectSources(nsigma=35, threshold='auto')
ob.AddSources([[100, 200]], weights=0.0)
# NOTE: If some sources must be filtered out, do it here by modifying the ob.sources_table before initializing the sources for simulation

#%%
ob.ExtractSources()
ob.DisplaySources(draw_box_size=20)

#%%
ob.PlotSourceSpectra()
ob.InitSimulation()

#%%
ob.FitPSFModel(repeat=3, max_iter=200)

#%%
ob.SimulateField()
ob.DisplaySimulation(plot_profiles=True)

#%%
Strehls_per_λ = ob.PSF_model.ComputeStrehl()
plt.title('Strehl ratio vs. λ (for the 1st source)')
plt.plot(ob.λ_sparse, 100.0 * Strehls_per_λ.flatten().cpu())
plt.ylabel('Strehl ratio, [%]')
plt.xlabel('Wavelength, [nm]')
plt.grid()
plt.show()

#%%
ob.SimulateField(full_spectrum=True)
ob.DisplaySimulation(plot_profiles=True, plot_full_spectrum=True)

#%%
ob.PlotSourceSpectra(title='Sources spectra (residual)', show_sparse=False, plot_residual=True, smooth_kernel=15)
# %%