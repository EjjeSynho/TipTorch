#%%
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tiptorch._config import WEIGHTS_FOLDER, default_torch_type

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm
from typing import Optional, Union, Sequence
from dataclasses import dataclass
from pathlib import Path

from tiptorch.PSF_models.NFM_wrapper import PSFModelNFM
from tiptorch.tools.utils import mask_square, generate_random_colors, rad2mas, rad2arc
from tools.plotting import PlotSpetralCubeInRGB
from data_processing.MUSE_data_utils import GetSpectrum, LoadCachedDataMUSE
from machine_learning.calibrators.NFM_calibrator import NFMCalibrator
from fitting.PSF_optimizer import OptimizePSFModel
from tools.multisources import (
    add_ROIs,
    AddSources,
    DetectSources,
    DisplaySources,
    ExtractSourceImages,
    VisualizeSources,
    ROI_from_valid_mask,
    PlotSourcesProfiles,
)


@dataclass
class SourcesSubset:
    ids: list[int]
    table: pd.DataFrame
    imgs_sparse: list
    slices_local: list
    slices_global: list
    spectra_sparse: torch.Tensor
    spectra_full: torch.Tensor
    spectra_res_sparse: Optional[torch.Tensor] = None
    spectra_res_full: Optional[torch.Tensor] = None


@dataclass
class SourcesData:
    """All source-indexed arrays. Row order is the canonical source ID order."""
    table: pd.DataFrame
    imgs_sparse: list
    slices_local: list
    slices_global: list
    spectra_full: torch.Tensor
    spectra_sparse: torch.Tensor
    spectra_res_full: Optional[torch.Tensor] = None
    spectra_res_sparse: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        self.table = self.table.reset_index(drop=True).copy()
        self.table.index.name = "src_id"
        self.table["src_id"] = np.arange(len(self.table), dtype=int)

    def __len__(self) -> int:
        return len(self.table)

    def index(self, src_ids: Optional[Union[int, np.integer, Sequence[int]]] = None) -> list[int]:
        """Normalize src_ids to an explicit validated list. None returns all source IDs."""
        if src_ids is None:
            return list(range(len(self)))
        if isinstance(src_ids, (int, np.integer)):
            src_ids = [int(src_ids)]
        ids = [int(i) for i in src_ids]
        bad = [i for i in ids if i < 0 or i >= len(self)]
        if bad:
            raise IndexError(f"Source IDs out of range: {bad}; valid range is [0, {len(self) - 1}]")
        return ids

    def select(self, src_ids: Optional[Union[int, Sequence[int]]]) -> "SourcesSubset":
        ids = self.index(src_ids)
        return SourcesSubset(
            ids   = ids,
            table = self.table.iloc[ids],
            imgs_sparse    = [self.imgs_sparse[i]   for i in ids],
            slices_local   = [self.slices_local[i]  for i in ids],
            slices_global  = [self.slices_global[i] for i in ids],
            spectra_sparse = self.spectra_sparse[ids],
            spectra_full   = self.spectra_full[ids],
            spectra_res_sparse = self.spectra_res_sparse[ids] if self.spectra_res_sparse is not None else None,
            spectra_res_full   = self.spectra_res_full[ids]   if self.spectra_res_full is not None   else None
        )

    def subset(self, src_ids: Optional[Union[int, Sequence[int]]] = None) -> "SourcesSubset":
        """Return a subset view for the requested source IDs."""
        return self.select(src_ids)

    def __getitem__(self, src_ids: Union[int, np.integer, Sequence[int], slice]) -> "SourcesSubset":
        """Enable bracket-based source subset selection."""
        if isinstance(src_ids, slice):
            return self.select(list(range(len(self)))[src_ids])
        return self.select(src_ids)


@dataclass
class FitWeights:
    subset:     SourcesSubset
    per_src:    torch.Tensor
    spectral:   torch.Tensor
    total:      torch.Tensor
    phase_bump: float
    LO:         float
    MSE:        float = 900.0
    MAE:        float = 2.6
    positive:   float = 2.0
    unit_flux:  float = 0.2


# TODO: implement a OB storage function (not only FITS cubes, but also sources data, fitting results, etc.)
class MUSEObservation:
    def __init__(self, raw_path, cube_path, cache_path, device=default_torch_type):
        self.raw_path   = raw_path
        self.cube_path  = cube_path
        self.cache_path = cache_path
        self.device     = device
        
        self.suppress_bump_flag = False # Flag to suppress fitting of the "phase bump" NCPA of MUSE NFM
        self.suppress_LO_flag   = False # Flag to suppress fitting of the quasi-static LO aberrations
        # Helper function during the fitting
        self.force_positive_coef = lambda x: torch.clamp(-x, min=0).pow(2).mean()
        
        # Initialize no sources in the OB yet
        self.sources_table = None
        self.sources = None
        self.N_src = 0
        
        # Define the half-width of the region for spectrum extraction = ~size of the PSF core
        self.flux_core_radius = 2 # [pix]
        # Define how many pixels are averaged when extracting the spectra from the PSF core
        self.N_core_pixels = (self.flux_core_radius*2 + 1)**2  # [pix²], this expression assumes a square mask for the core flux estimation

        try:
            self._load_data()
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise e
        
        # Simulated PSF size
        self.PSF_size = 111
        self.model_config['sensor_science']['FieldOfView'] = self.PSF_size # Sice of the PSF to simulate
        
        # Empty image to store the simulated PSFs while adding them to the right locations in the field. This is more memory efficient than storing all PSFs
        # separately and also allows to overlap sources on top of each other, which is important for a realistic simulation and fitting
        self.canvas_sparse = torch.zeros([self.N_wvl, self.cube_sparse.shape[-2], self.cube_sparse.shape[-1]], device=self.device)
        # Same idea here, but for the full-spectrum simulation. It's stored on CPU to save GPU memory since it's only used for visualization and evaluation
        self.canvas_full = torch.zeros([self.N_wvl_full, self.cube_full.shape[-2], self.cube_full.shape[-1]], device='cpu')

        # Region to plot when visualizing the sources and fitting results
        self.ROI_plot = ROI_from_valid_mask(self.valid_mask)["slice"]
        
        self.simulated_sparse = None # this will be used to store the simulated field
        self.simulated_full   = None # this will be used to store the simulated field for the full MUSE-NFM spectrum


    def _load_data(self):
        ''' Loads OB related parameters and data cubes. '''
        # We need to pre-process the data before using it with the model and asssociate the reduced telemetry - this is done by the LoadDataCache function
        # You need to run this function at least ones to generate the data cache file. Then, te function will automatically reduce it ones it's found
        spectral_cubes, spectral_info, data_cache, model_config = LoadCachedDataMUSE(self.raw_path, self.cube_path, self.cache_path, save_cache=True, device=self.device, verbose=True)   
        # Extract full and binned spectral cubes. Sparse cube selects a subset of binned wavelengths ranges
        self.cube_full, cube_binned, valid_mask = spectral_cubes["cube_full"], spectral_cubes["cube_binned"], spectral_cubes["mask"]

        #NOTE: MUSE cube flux units are [10^-20 erg s^-1 cm^-2 Å^-1], all cubes are normalized to this flux unit

        # Compute the center of mass for valid mask assuming it's the center of the science field
        yy, xx = torch.where(valid_mask.squeeze() > 0)
        self.field_center = np.stack([xx.float().mean().item(), yy.float().mean().item()])[None,...] # [pix]
        del yy, xx

        self.reduced_telemetry = data_cache['All data'] # this is the telemetry reduced to the format compatible with the model, it can be used to update the model config with the actual telemetry values
        
        del data_cache # free up memory, we won't need the rest of the data cache for now, but it can be useful for debugging and further analysis if needed

        # To save memory and compute time, we don't need neither the full spectral cube, nor even the binned one. It's enough to have a sparse subset of spectral
        # slices that cover the whole wavelength range, which is enough to constrain the chromatic behavior of PSFs. It is referred to as "sparse" cube.
        λ_full,   Δλ_full   = spectral_info['λ_full'],   spectral_info['Δλ_full']
        λ_binned, Δλ_binned = spectral_info['λ_binned'], spectral_info['Δλ_binned']

        # Here, it's assumed to be every 5th bin, but it can be changed to any other selection strategy
        ids_λ_sparse = np.arange(0, λ_binned.shape[-1], 5)
        λ_sparse  =  λ_binned[..., ids_λ_sparse]
        Δλ_sparse = Δλ_binned[..., ids_λ_sparse]

        self.cube_sparse = cube_binned[ids_λ_sparse, ...] # Select the sparse subset ofspectral slices
        self.N_wvl = self.cube_sparse.shape[0]
        self.N_wvl_full = self.cube_full.shape[0]

        # Since spectral bins are the sum, they need to be re-normalized to averages to be compatible with the full spectrum
        flux_λ_norm = torch.tensor(Δλ_full / Δλ_sparse, device=self.device, dtype=torch.float32)
        self.cube_sparse *= flux_λ_norm[:, None, None]
        
        self.λ_full   = λ_full
        self.λ_sparse = λ_sparse
        
        self.model_config = model_config #TODO: make config (re-)initialization more flexible
        self.valid_mask = valid_mask


    def AddSourcesToModelConfig(self):
        # TODO: properly manage per-source variables in NFM_wrapper if sources are added/deleted
        # TODO: implement sources deletion
        pixel_scale = self.model_config['sensor_science']['PixelScale'] # [mas/pix]
        
        # Sources coordinates that can be understood by TipTorch model
        # Computed relativeto the center of the field, which is assumed to be a CoG of valid pixels' mask
        sources_coords  = np.stack([self.sources.table['x_peak'].values, self.sources.table['y_peak'].values], axis=1)
        sources_coords -= self.field_center
        sources_coords  = sources_coords * pixel_scale / rad2mas  # [pix] -> [rad]

        # Convert to zenith and azimuth angles
        sources_zenith  = np.arctan(np.sqrt(sources_coords[:,0]**2 + sources_coords[:,1]**2)) * rad2arc # [arcsec]
        sources_azimuth = np.degrees(np.arctan2(sources_coords[:,1], sources_coords[:,0]))  # [deg]

        # Update the model config with the sources coordinates
        self.model_config['NumberSources'] = len(self.sources)
        self.model_config['sources_science']['Zenith']  = torch.tensor(sources_zenith,  device=self.device).unsqueeze(-1)
        self.model_config['sources_science']['Azimuth'] = torch.tensor(sources_azimuth, device=self.device).unsqueeze(-1)


    def ExtractSpectraFromCore(self, sources_table, cube_sparse, cube_full):
        if len(sources_table) == 0:
            raise ValueError("No sources detected. Please run DetectSources() or AddSources() before extracting spectra.")
        
        # Contains the spectrum per source AVERAGED across the PSF core pixels. The core mask here must match EXACTLY the one
        # used for the flux normalization factor estimation later        
        N_src = len(sources_table)
        # This one is stored in RAM to save VRAM since it's only used for visualization and evaluation
        spectra_full = [GetSpectrum(cube_full, sources_table.iloc[i], radius=self.flux_core_radius, mask_type='square') for i in range(N_src)]
        spectra_full = np.stack(spectra_full, axis=0)
        spectra_full = torch.tensor(spectra_full, device='cpu', dtype=torch.float32)
        # This one is stored in VRAM since it's used for fitting and needs to be on the same device as the model
        spectra_sparse = [GetSpectrum(cube_sparse, sources_table.iloc[i], radius=self.flux_core_radius, mask_type='square') for i in range(N_src)]
        spectra_sparse = torch.stack(spectra_sparse, dim=0)
        
        return spectra_sparse, spectra_full


    def InitSimulation(self) -> None:
        if self.sources is None:
             raise ValueError("Sources must be initialized before initializing the PSF model. Please run ExtractSources() first.")
        
        # The model config is also updated to simulate only sparse λs
        self.model_config['sources_science']['Wavelength'] = torch.tensor(self.λ_sparse, device=self.device, dtype=torch.float32) * 1e-9 #[m]
        
        if not isinstance(self.model_config['telescope']['PupilAngle'], torch.Tensor):
            self.model_config['telescope']['PupilAngle'] = torch.tensor(self.model_config['telescope']['PupilAngle'], device=self.device)

        self.AddSourcesToModelConfig()
        # Initialize the PSF model
        self.PSF_model = PSFModelNFM(
            self.model_config,
            multiple_obs    = False,
            LO_NCPAs        = True,
            chrom_defocus   = False,
            Moffat_absorber = False,
            N_spline_nodes  = 5,
            Z_mode_max      = 9,
            device          = self.device
        )
        #  Get the initial guess for the PSF model parameters
        calibrator = NFMCalibrator(WEIGHTS_FOLDER / 'NFM_calibrator/NFM_calibrator_bundle.pth', device=self.device)
        _ = calibrator.check_compatibility(self.PSF_model)
        # Change PSF model parameters based on NN's predictions
        calibrator.calibrate(self.reduced_telemetry, self.PSF_model)

        # Disable optimization of some variables
        self.PSF_model.inputs_manager.set_optimizable('bg_ctrl',  False)
        self.PSF_model.inputs_manager.set_optimizable('wind_dir', False)


    def DetectSources(self, nsigma=35, threshold='auto'):
        # Simple sources detector function. For now, make sure that only point sources are included. This function also defines the
        # order in which sources are indexed ad processed later, so it's important to use it before extracting the source images
        # and spectra. The order can be defined by the brightness of the sources in the descending order
        self.sources_table = DetectSources(
            self.cube_sparse,
            threshold = threshold,
            nsigma = nsigma,
            box_size = 11,
            sort_by_brightness = True,
            weight_from_flux = False
        )


    def AddSources(self, sources_coords, weights=0.0):
        # If some sources are missing from the automatic detection, they can be added manually by providing their coordinates in the same format as the sources dataframe
        self.sources_table = AddSources(self.cube_sparse, sources_coords, self.sources_table, weights=weights, weight_from_flux=False)


    def ExtractSources(self, verbose=False, max_nan_fraction=0.3):
        ''' Extract separate source images + other auxilliary data. It's necessary for later fitting and performance evaluation '''
        # Filtering means, that the sources on the edge of the filed will be removed since they don't have enough pixels
        srcs_image_data = ExtractSourceImages(self.cube_sparse, self.sources_table, box_size=self.PSF_size, filter_sources=True, max_nan_fraction=max_nan_fraction, debug_draw=False)
        # Extract spectra from the PSF core for each filtered source
        spectra_sparse, spectra_full = self.ExtractSpectraFromCore(srcs_image_data["src_data"], self.cube_sparse, self.cube_full)
        # These are the sources which are included in the model
        self.sources = SourcesData(
            table          = srcs_image_data["src_data"],
            imgs_sparse    = srcs_image_data["src_images"],
            slices_local   = srcs_image_data["ROI_local"],
            slices_global  = srcs_image_data["ROI_global"],
            spectra_full   = spectra_full,
            spectra_sparse = spectra_sparse,
        )
        self.N_src = len(self.sources)
        
        if verbose:
            print(f"Extracted {self.N_src} sources for fitting.")
            print(f"Num. of filtered sources on the edge of the field: {len(self.sources_table) - self.N_src}")
            

    @torch.no_grad()
    def _get_flux_factor(self, quasi_inf_PSF_size: int = 511) -> None:
        ''' Computes composite chromatic flux normalization factor. '''

        current_PSF_size = self.PSF_model.model.N_pix # the actual size of the simulated PSFs
        # Ensure that if PSF's small size is odd, then quasi-infinite PSF size is also odd to avoid sub-pixel shifts of the PSF core
        quasi_inf_PSF_size -= (1-current_PSF_size % 2)

        if self.PSF_model.use_splines:
            wvl_current = self.PSF_model.λ_sim.clone()
            self.PSF_model.SetWavelengths(self.PSF_model.λ_ctrl) # isntead of anchor λs, evaluate at spline nodes

        # Backup the original sources coordinates
        coords_backup = (
            self.PSF_model.inputs_manager['src_dirs_x'].clone(),
            self.PSF_model.inputs_manager['src_dirs_y'].clone()
        )
        # Assume on-axis PSF for computing the ratio
        self.PSF_model.inputs_manager['src_dirs_x'] *= 0.0
        self.PSF_model.inputs_manager['src_dirs_y'] *= 0.0

        PSF_small = self.PSF_model.forward(src_ids=0) # compute only for the first source ignoring the field variability (just for speed's sake)
        self.PSF_model.SetImageSize(quasi_inf_PSF_size) # quasi-infinite PSF image to compute how much flux is lost while cropping

        PSF_inf = self.PSF_model.forward(src_ids=0)
        self.PSF_model.SetImageSize(current_PSF_size)

        if self.PSF_model.use_splines:
            self.PSF_model.SetWavelengths(wvl_current) # switch back to the original wavelengths

        # Restore the original coordinates
        self.PSF_model.inputs_manager['src_dirs_x'], self.PSF_model.inputs_manager['src_dirs_y'] = coords_backup

        # How much flux is cropped by assuming the finite size of the PSF. Since PSFs are normalized to ∑PSF≈1 per wavelength,
        # the crop ratio is given by the ratio of the max pixel values in the small and quasi-infinite PSF images
        crop_ratio = (PSF_inf.amax(dim=(-2,-1)) / PSF_small.amax(dim=(-2,-1))).squeeze()

        # How much flux is spread out of the PSF core because PSF is not a single pixel but rather "a blob". In other words,
        # compute the ratio of the flux in the PSF core (defined by a mask) to the total flux in the quasi-infinite PSF image.
        core_mask_inf   = torch.tensor(mask_square(quasi_inf_PSF_size, self.flux_core_radius+1)[None,None,...], dtype=default_torch_type, device=self.device)
        core_flux_ratio = torch.squeeze((PSF_inf * core_mask_inf).sum(dim=(-2,-1), keepdim=True) / PSF_inf.sum(dim=(-2,-1), keepdim=True))

        # Compute composite normalization factor. Since the spectrum extracted from the PSF core is averaged over the core,
        # we need to multiply it by the number of core pixels to get the total flux in the PSF core
        PSF_norm_factor = self.N_core_pixels / core_flux_ratio / crop_ratio
        torch.cuda.empty_cache()
        
        if self.PSF_model.use_splines:
            self.PSF_model.inputs_manager['F_norm_λ_ctrl'] = PSF_norm_factor.clone() # store it, we'll need it later when simulating full spectrum
        else:
            self.PSF_model.inputs_manager['F_norm_λ'] = PSF_norm_factor.unsqueeze(0).clone() # we'll need it later when simulating full spectrum


    @torch.no_grad()
    def _update_flux_norm(self) -> None:
        #TODO: what about non-optimized sources?
        PSF_norm_factor_old = self.PSF_model.inputs_manager['F_norm_λ_ctrl'].clone()

        self._get_flux_factor() # update the core flux normalization factor based on the current PSF morphology
        PSF_norm_factor_new = self.PSF_model.inputs_manager['F_norm_λ_ctrl'].clone()
        
        # Compute the updated normalization factors
        F_norm_correction = (PSF_norm_factor_new / PSF_norm_factor_old).mean().item()
        self.PSF_model.inputs_manager['F_norm'] /= F_norm_correction

        # Since F_ctrl is the parameter that directly controls the overall flux normalization in the model, we can use its mean value to correct
        # for per-source flux normalization. This is a rather empirical correction
        F_mean_correction = self.PSF_model.inputs_manager['F_ctrl'].mean().item()
        self.PSF_model.inputs_manager['F_ctrl'] /= F_mean_correction
        self.PSF_model.inputs_manager['F_norm'] *= F_mean_correction


    @torch.no_grad()
    def InitFluxNorm(self) -> None:
        
        self._update_flux_norm()
        
        # """
        # Sets initial flux normalization from PSF morphology alone, without using image data.
        # Avoids the faint-source brightness bias that image-based matched-filter estimates
        # suffer from (contamination by background and brighter neighbours).

        # 1. Calls _get_flux_factor() to establish F_norm_λ_ctrl from the current PSF shape
        #    (core encircled-energy fraction and finite-image crop ratio).
        # 2. Normalises F_ctrl to unit mean, folding any residual overall scale into F_norm.
        # 3. Resets F_norm to 1.0 per source — the analytically correct initial value given
        #    that spectra_sparse already encodes the per-source flux and F_norm_λ_ctrl
        #    accounts for all PSF-morphology flux corrections.
        # """
        # self._get_flux_factor()

        # F_ctrl_mean = self.PSF_model.inputs_manager['F_ctrl'].mean().item()
        # if abs(F_ctrl_mean) > 1e-12:
        #     self.PSF_model.inputs_manager['F_ctrl'] /= F_ctrl_mean

        # self.PSF_model.inputs_manager['F_norm'] = torch.ones(
        #     self.N_src, device=self.device,
        #     dtype=self.PSF_model.inputs_manager['F_norm'].dtype
        # )


    @torch.no_grad()
    def InitFluxNormFromImage(self) -> None:
        """
        Estimates initial per-source flux normalization from the currently predicted PSF
        morphology using a matched-filter comparison against the observed image.

        NOTE: this approach can overestimate flux for faint sources because their local
        ROI is contaminated by background and spillover from brighter neighbours.
        Prefer InitFluxNorm() for an unbiased (image-free) initialisation.

        1. Calls _get_flux_factor() to establish chromatic PSF normalization factors
           (stored in F_norm_λ_ctrl) from the current PSF shape.
        2. Resets F_norm to 1.0 and normalizes F_ctrl to unit mean, so the subsequent
           per-source estimate is unbiased by any residual overall scale.
        3. Simulates each source individually and computes the optimal F_norm[i] as the
           matched-filter (least-squares) scale factor between the observed and predicted
           flux in that source's local field region.
        """
        self._get_flux_factor()

        F_ctrl_mean = self.PSF_model.inputs_manager['F_ctrl'].mean().item()
        if abs(F_ctrl_mean) > 1e-12:
            self.PSF_model.inputs_manager['F_ctrl'] /= F_ctrl_mean

        self.PSF_model.inputs_manager['F_norm'] = torch.ones(
            self.N_src, device=self.device,
            dtype=self.PSF_model.inputs_manager['F_norm'].dtype
        )

        F_norm_est = torch.empty(self.N_src, device=self.device,
                                 dtype=self.PSF_model.inputs_manager['F_norm'].dtype)

        for i in range(self.N_src):
            src_subset_i = self.sources.select(i)

            sim_i = self.simulate_sparse(x=None, src_subset=src_subset_i)

            (y_min, y_max), (x_min, x_max) = src_subset_i.slices_global[0]

            obs_roi = self.cube_sparse[:, y_min:y_max, x_min:x_max]  # [N_wvl, H_roi, W_roi]
            sim_roi = sim_i           [:, y_min:y_max, x_min:x_max]  # [N_wvl, H_roi, W_roi]

            valid = ~torch.isnan(obs_roi) & (sim_roi > 0)

            numerator   = (obs_roi[valid] * sim_roi[valid]).sum()
            denominator = (sim_roi[valid] * sim_roi[valid]).sum()
            F_norm_est[i] = (numerator / (denominator + 1e-12)).clamp(min=0.1)

        self.PSF_model.inputs_manager['F_norm'] = F_norm_est


    def _compute_fitting_weights(self) -> FitWeights:
        # Compute loss weighting factors per source based on total flux per source
        src_fluxes  = torch.tensor(self.sources.table['peak_value'].to_numpy(), device=self.device, dtype=torch.float32)
        src_weights = torch.tensor(self.sources.table['weight'].to_numpy(),     device=self.device, dtype=torch.float32)

        # Compute chromatic loss weighting factors per source based on its spectrum
        w_spectral  = self.sources.spectra_sparse.amax(dim=-1, keepdim=True) / self.sources.spectra_sparse
        w_spectral /= w_spectral.mean(dim=0, keepdim=True) # normalize to the mean to avoid changing the overall loss scale
        w_spectral  = torch.clamp(w_spectral, min=0.2, max=2.0) # limit the loss influence of certain wavelengths
        w_spectral  = w_spectral.view(len(self.sources), self.N_wvl, 1, 1)
        w_spectral  = w_spectral * 0.0 + 1.0 # For now, use uniform spectral weighting

        # ---------- Source selection for fitting ----------
        # Exclude sources with negligible weights; pre-select into a SourcesSubset so
        # simulate_sparse can use its fields directly without re-indexing every call.
        fit_ids = torch.where(src_weights > 1e-6)[0].tolist()
        fit_subset = self.sources.select(fit_ids if len(fit_ids) < len(self.sources) else None)

        fit_fluxes   = src_fluxes [fit_subset.ids]
        fit_weights  = src_weights[fit_subset.ids]
        fit_spectral = w_spectral [fit_subset.ids]

        fit_relative = torch.clamp(fit_fluxes / fit_fluxes.max().item(), min=0.1, max=1.0) * fit_weights
        # Weighted mean normalisation so loss stays within ~10⁰–10¹ range
        w_total = fit_relative.sum() / (fit_fluxes * fit_relative).sum()

        return FitWeights(
            subset     = fit_subset,
            per_src    = fit_weights,
            spectral   = fit_spectral,
            total      = w_total,
            LO         = 1e-7 if not self.suppress_LO_flag   else 1e3,
            phase_bump = 5e-5 if not self.suppress_bump_flag else 1e3,
            positive   = 2.0,
            unit_flux  = 0.2
        )

    
    def simulate_sparse(self, x=None, src_subset: Optional[SourcesSubset] = None) -> torch.Tensor:
        """
        Simulates the full-field PSF canvas for the given source subset (all sources by default).
        Accepts a pre-built SourcesSubset so spectra and ROI lists are already indexed,
        avoiding per-call list comprehensions in the fitting hot-path.
        """
        if src_subset is None:
            src_subset = self.sources.select(None) # None here = select all sources
            
        # Update model inputs from the stacked parameter vector. If None provided, use the current model parameters as they are
        x_dict = self.PSF_model.inputs_manager.unstack(x, include_all=True, update=True) if x is not None else self.PSF_model.inputs_manager.to_dict()
        # Pass None to NFM_wrapper when all sources are selected to avoid unnecessary indexing inside the model
        all_selected = len(src_subset.ids) == self.N_src
        # Index F_norm before forward() may modify x_dict in-place
        F_norm = x_dict['F_norm'] if all_selected else x_dict['F_norm'][src_subset.ids]
        PSFs_  = self.PSF_model(x_dict, src_ids=None if all_selected else src_subset.ids)
        PSF_norm_factor = self.PSF_model.evaluate_splines(self.PSF_model.inputs_manager['F_norm_λ_ctrl'], self.PSF_model.λ_sim_normed)
        
        flux_normalization = F_norm.view(-1, 1) * PSF_norm_factor * src_subset.spectra_sparse
        PSFs_ = PSFs_ * flux_normalization.view(-1, self.N_wvl, 1, 1)
        return add_ROIs(self.canvas_sparse * 0.0, PSFs_, src_subset.slices_local, src_subset.slices_global)


    def _loss_PSF(self, PSF_data: torch.Tensor, PSF_pred: torch.Tensor, fit_weights: FitWeights):
        diff = PSF_data - PSF_pred
        residuals = diff * fit_weights.spectral * fit_weights.per_src.view(-1, 1, 1, 1) # apply both spectral and source weights to the residuals
        MSE_loss = residuals.pow(2).mean() * fit_weights.MSE
        MAE_loss = residuals.abs().mean()  * fit_weights.MAE
        # Since x input in simulate_sparse() updates the values inside the PSF_model (including F_norm), they now can be used directly here
        F_penalty = (self.PSF_model.inputs_manager['F_ctrl'] - 1.0).abs().mean()
        # Soft non-negativity penalty: penalize when residuals (data - model) are negative to prevent ouversubtracting simulated PSFs
        force_positive_diff = torch.nn.functional.relu(-diff).mean()
        return (MSE_loss + MAE_loss) * fit_weights.total + F_penalty * fit_weights.unit_flux + force_positive_diff * fit_weights.positive


    def _loss_LO(self, fit_weights: FitWeights):
        ''' Loss function term focus on values of LO aberrations coefficients '''
        # L2 regularization on all LO coefficients
        LO_loss = self.PSF_model.inputs_manager['LO_coefs'].pow(2).sum(-1).mean() * fit_weights.LO
        # Constraint to enforce first element of LO_coefs to be positive
        phase_bump_positive = self.force_positive_coef(self.PSF_model.inputs_manager['LO_coefs'][:, 0]) * fit_weights.phase_bump
        # Force defocus to be positive to mitigate sign ambiguity
        first_defocus_penalty = self.force_positive_coef(self.PSF_model.inputs_manager['LO_coefs'][:, 2]) * fit_weights.LO  #NOTE: won't work with the chromatic defocus
        # Final composite loss
        return LO_loss + phase_bump_positive + first_defocus_penalty


    def loss(self, x_, data, model, fit_weights: FitWeights):
        model = model(x_) # generate the batch of PSFs
        LO_loss  = self._loss_LO(fit_weights) # LO static modes loss
        PSF_loss = self._loss_PSF(data, model, fit_weights=fit_weights) # loss related to PSF morphology
        return LO_loss + PSF_loss


    def _select_optimizable_variables(self, fit_params):
        
        if isinstance(fit_params, str):
            fit_params = [fit_params]
        
        # Determine which parameters must be fitted
        PSF_params        = ['r0', 'dn', 'LO_coefs', 'F_ctrl', 'J_ctrl', 'L0', 'wind_speed_single', 'wind_dir_single', 'Cn2_weights']
        astrometry_params = ['dx_ctrl', 'dy_ctrl']
        photometry_params = ['bg_ctrl', 'F_norm_λ_ctrl', 'F_norm']

        if 'PSF' in fit_params and 'astrometry' in fit_params and 'photometry' in fit_params:
            fitable_set = None # all PSF model parameters are optimized
        else:
            fitable_set = (PSF_params        if 'PSF'        in fit_params else []) + \
                          (astrometry_params if 'astrometry' in fit_params else []) + \
                          (photometry_params if 'photometry' in fit_params else [])
            # Check if selected parameters are fittable within the PSF model
            fitable_set = [param for param in fitable_set if self.PSF_model.inputs_manager.is_optimizable(param)]
        return fitable_set


    # TODO: provide sources selection from outside
    def FitPSFModel(self, fit=['PSF', 'astrometry', 'photometry'], repeat=2, max_iter=200) -> None:
        
        # optimizables_backup = self.PSF_model.get_optimizable_param_names() # backup the original optimizables settings to restore them later if needed
        # Vector to store the encoded PSF model parameters
        x_params = None
        
        optimizables_ = self._select_optimizable_variables(fit)
        weights_ = self._compute_fitting_weights()
        
        fit_subset = weights_.subset  # pre-built once, reused every iteration
        
        run_fn  = lambda x: self.simulate_sparse(x, fit_subset) # inference function used by the optimizer
        loss_fn = lambda x: self.loss(x, self.cube_sparse, run_fn, weights_) # composite loss function to minimize
        
        for _ in range(repeat):
            self._update_flux_norm()
            x_params, _ = OptimizePSFModel(
                self.PSF_model,
                loss_fn,
                x_initial  = x_params.clone() if x_params is not None else None,
                max_iter   = max_iter,
                n_attempts = 1,
                verbose    = True,
                force_bfgs = True,
                include_params = optimizables_
            )

        # self.PSF_model.inputs_manager.set_optimizable(optimizables_backup) # restore the original optimizables settings after fitting
        # NOTE: Intrinsic values inside PSF_model.inputs manager are updated automatically on every function call
        

    @torch.no_grad() #TODO: split output
    def SimulateField(self, full_spectrum=False) -> None:
        ''' Simulate all sources within the field. If full_spectrum is True, simulates the full spectrum instead of the sparse λ-subset. '''
        if full_spectrum:
            # No inputs since SimulateFullSpectrum() fully relies on the inputs stored in the inputs_manager, which are already up-to-date
            PSFs_combined = self.PSF_model.SimulateFullSpectrum(verbose=True)
            # Now, chromatic PSF normalization factor must be re-evaluated for the full spectrum
            PSF_norm_factor_full = self.PSF_model.evaluate_splines(self.PSF_model.inputs_manager['F_norm_λ_ctrl'], self.PSF_model.λ_full_normed).cpu()
            flux_normalization   = self.PSF_model.inputs_manager['F_norm'].unsqueeze(-1).cpu() * PSF_norm_factor_full * self.sources.spectra_full
            # Apply the flux normalization to the PSFs similar to func()
            PSFs_ = PSFs_combined * flux_normalization.unsqueeze(-1).unsqueeze(-1)
            self.simulated_full = add_ROIs(self.canvas_full, PSFs_, self.sources.slices_local, self.sources.slices_global)
            self.residue_full = (self.cube_full - self.simulated_full.numpy()) * self.valid_mask.cpu().numpy()
        else:
            self.simulated_sparse = self.simulate_sparse(x=None, src_subset=None)
            self.residue_sparse = (self.cube_sparse - self.simulated_sparse) * self.valid_mask


    def DisplaySources(self, draw_box_size=20):
        if self.sources_table is None:
            raise ValueError("Sources must be initialized before displaying. Please run DetectSources() or AddSources() first.")
        
        # If sources were not yet initialized or filtered, but just detected.
        # Otherwise, use the sources table from the sources data which is already filtered and has the right order of sources
        DisplaySources(
            self.cube_sparse,
            self.sources_table if self.sources is None else self.sources.table,
            src_box_size = draw_box_size,
            vmin = 10,
            vmax = (self.sources_table if self.sources is None else self.sources.table)['peak_value'].max() * 0.85,
            ROI  = self.ROI_plot
        )
   

    def PlotSourceSpectra(self, src_ids=None, title='Sources spectra', show_sparse=True, plot_residual=False, smooth_kernel=None, figsize=(10, 6)):
        """
            Plots the spectra of the sources extracted from the PSF core.
            If plot_residual is True, plots the spectra extracted from the residual cube.
            If smooth_kernel is provided, applies a simple boxcar smoothing to the full spectra for better visualization.
            
            Parameters
            ----------
            src_ids : int, list of int, or None
                Source IDs to plot. If None, plots all sources.
            title : str
                Plot title.
            show_sparse : bool
                Whether to show sparse wavelength samples as scatter points.
            plot_residual : bool
                Whether to plot residual spectra instead of the original spectra.
            smooth_kernel : int or None
                Boxcar smoothing kernel size for full spectra visualization.
            figsize : tuple
                Figure size (width, height) in inches.
        """
        if self.sources is None:
            raise ValueError("Sources must be initialized before plotting spectra. Please run InitSources() first.")
        
        from astropy.convolution import convolve, Box1DKernel

        if plot_residual:
            if self.sources.spectra_res_full is None or self.sources.spectra_res_sparse is None:
                print("Residual spectra are not available. Computing them now...")
                self.sources.spectra_res_sparse, self.sources.spectra_res_full = \
                    self.ExtractSpectraFromCore(self.sources.table, self.residue_sparse, self.residue_full)
                
        src_subset = self.sources.select(src_ids) # select a subset of sources to plot
        
        if plot_residual:
            src_spectra_full   = src_subset.spectra_res_full
            src_spectra_sparse = src_subset.spectra_res_sparse
        else:
            src_spectra_full   = src_subset.spectra_full
            src_spectra_sparse = src_subset.spectra_sparse
                  
        N = len(src_subset.ids)
        colors = generate_random_colors(N)

        show_smooth = smooth_kernel is not None and smooth_kernel > 1

        plt.figure(figsize=figsize)

        vmin = min(src_spectra_full.min().item(), src_spectra_sparse.min().item() if show_sparse else float('inf'))

        for i, src_id in enumerate(src_subset.ids):
            plt.plot(self.λ_full, src_spectra_full[i],
                     linewidth=0.25, alpha=(0.5 if show_smooth else 0.8), color=colors[i],
                     label=f'Source {src_id+1} (full spectrum)')
            
            if show_sparse:
                plt.scatter(self.λ_sparse, src_spectra_sparse[i].cpu().numpy(),
                            color=colors[i], marker='o', s=30, alpha=1.0,
                            label=f'Source {src_id+1} (sparse samples)')
            if show_smooth:
                spectrum_smooth = convolve(src_spectra_full[i].numpy(), Box1DKernel(smooth_kernel), boundary='extend')
                plt.plot(self.λ_full, spectrum_smooth,
                         linewidth=0.65, alpha=1, color=colors[i],
                         label=f'Source {src_id+1}')
                
            plt.axhline(0, color='gray', linewidth=0.8)

        plt.xlabel('Wavelength, [nm]')
        plt.xlim(self.λ_full.min(), self.λ_full.max())
        plt.ylim(vmin, None)
        plt.ylabel(r'Flux, [ $10^{-20} \frac{erg} {s \, \cdot \, cm^2 \, \cdot \, Å} ]$')
        plt.title(title)
        
        if N < 10: # Don't plot it when too many sources are displayed to avoid cluttering the legend
            plt.legend()
        
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


    def DisplaySimulation(self, plot_profiles=True, plot_full_spectrum=False) -> None:
        if plot_full_spectrum:
            if self.simulated_full is None:
                print("Full spectrum simulation not found, simulating now...")
                self.SimulateField(full_spectrum=True)
            
            # Mapping MUSE spectral range to visible spectrum range for RGB conversion
            λ_vis = np.linspace(440, 750, self.N_wvl_full)

            _ = PlotSpetralCubeInRGB(
                self.residue_full[self.ROI_plot],
                wavelengths=λ_vis,
                title="Difference",
                min_val=500, max_val=60000,
                scale='log',
                show=False
            )

            _ = PlotSpetralCubeInRGB(
                self.cube_full[self.ROI_plot],
                wavelengths=λ_vis,
                title=f"Data",
                min_val=500, max_val=200000,
                scale='log',
                show=True
            )

            _ = PlotSpetralCubeInRGB(
                self.simulated_full[self.ROI_plot],
                wavelengths=λ_vis,
                title=f"Model",
                min_val=500, max_val=200000,
                scale='log',
                show=True
            )
            
            if plot_profiles:
                PlotSourcesProfiles(self.cube_full, self.simulated_full, self.sources.table, radius=16, title='Source radial profiles (full spectrum)')                
        
        else:
            display_norm = LogNorm(vmin=1, vmax=self.cube_sparse.sum(dim=0).max()) # again, rather empirical values
            VisualizeSources(self.cube_sparse, self.simulated_sparse, norm=display_norm, mask=self.valid_mask, ROI=self.ROI_plot)
            
            if plot_profiles:
                PlotSourcesProfiles(self.cube_sparse, self.simulated_sparse, self.sources.table, radius=16, title='Source radial profiles (sparse spectrum)')
            
            
    def SaveModelCubeFITS(self, cube, output_path, λ_full, sources=None, compress=True, compression_type='GZIP_2'):
        raise NotImplementedError("This function is not properly implemented yet.")
        
        from astropy.io import fits
    
        """
        Save a 3D (N_wvl, H, W) or 4D (N_src, N_wvl, H, W) model cube to a FITS file.

        3D input → single image HDU (optionally compressed).

        4D input with sources=None → single zero-padded 4D image HDU (mostly zeros, poor use
        of space even with compression).

        4D input with sources=<DataFrame> → multi-extension FITS: one compact HDU per source
        (N_wvl, psf_H, psf_W), with the WCS in each extension encoding where the PSF belongs
        in the full field. No zeros are stored at all — this is the recommended format.
        Reading back: for source i, hdul[i+1].data gives its PSF cube; CRVAL1/2 give its
        1-based (x, y) centroid in the full-field pixel frame.

        Available compression types (all lossless for float32):
        'GZIP_2'  - byte-shuffle + gzip, best for float PSF data (default)
        'RICE_1'  - fast, good general-purpose
        'HCOMPRESS_1' - hierarchical, slightly lossy unless scale=0 enforced

        Parameters
        ----------
        cube             : array-like, shape (N_wvl, H, W) or (N_src, N_wvl, H, W)
        output_path      : str or Path
        λ_full           : 1-D array, wavelengths in nm
        sources          : pd.DataFrame with 'x_peak' and 'y_peak' columns (0-based pixel coords),
                        required for the compact multi-extension 4D format
        compress         : bool, apply tile compression (default True)
        compression_type : str, FITS tile compression algorithm (default 'GZIP_2')
        """
        data = np.asarray(cube, dtype=np.float32)
        is_4d = data.ndim == 4

        def _wcs_λ(hdr, λ_full):
            hdr['CRVAL3'] = float(λ_full[0]);  hdr['CDELT3'] = float(λ_full[1] - λ_full[0])
            hdr['CUNIT3'] = 'nm';              hdr['CTYPE3'] = 'WAVE';  hdr['CRPIX3'] = 1

        def _make_hdu(arr, hdr):
            if compress:
                return fits.CompImageHDU(arr, header=hdr, compression_type=compression_type)
            return fits.ImageHDU(arr, header=hdr)

        if is_4d and sources is not None:
            # --- compact multi-extension format: one HDU per source, no zeros stored ---
            primary = fits.PrimaryHDU()
            primary.header['NSOURCES'] = (data.shape[0], 'Number of source PSF extensions')
            primary.header['NWVL']     = (data.shape[1], 'Number of wavelength slices')
            primary.header['COMMENT']  = 'Each extension contains one source PSF cube (N_wvl, H, W).'
            primary.header['COMMENT']  = 'CRVAL1/2 gives the source centroid in full-field pixel coords (1-based).'
            hdul = fits.HDUList([primary])

            psf_h, psf_w = data.shape[2], data.shape[3]
            crpix_x = (psf_w + 1) / 2.0  # centre of the PSF cutout (1-based)
            crpix_y = (psf_h + 1) / 2.0

            for i in range(data.shape[0]):
                hdr = fits.Header()
                # Spatial WCS: reference pixel is the PSF centre; CRVAL encodes its position
                # in the full field (FITS 1-based convention, hence +1)
                hdr['CRPIX1'] = crpix_x;               hdr['CRVAL1'] = float(sources.iloc[i]['x_peak']) + 1
                hdr['CDELT1'] = 1;                      hdr['CUNIT1'] = 'pixel'; hdr['CTYPE1'] = 'PIXEL'
                hdr['CRPIX2'] = crpix_y;               hdr['CRVAL2'] = float(sources.iloc[i]['y_peak']) + 1
                hdr['CDELT2'] = 1;                      hdr['CUNIT2'] = 'pixel'; hdr['CTYPE2'] = 'PIXEL'
                _wcs_λ(hdr, λ_full)
                hdr['SRCIDX']  = (i,     'Source index (0-based)')
                hdul.append(_make_hdu(data[i], hdr))

            hdul.writeto(str(output_path), overwrite=True)
            comp_label = f' ({compression_type} compressed)' if compress else ''
            print(f"Saved {data.shape[0]}-source multi-extension FITS{comp_label} to {output_path}")

        else:
            # --- single HDU fallback (3D, or 4D without source positions) ---
            hdr = fits.Header()
            hdr['CRVAL1'] = 1; hdr['CDELT1'] = 1; hdr['CUNIT1'] = 'pixel'; hdr['CTYPE1'] = 'PIXEL'; hdr['CRPIX1'] = 1
            hdr['CRVAL2'] = 1; hdr['CDELT2'] = 1; hdr['CUNIT2'] = 'pixel'; hdr['CTYPE2'] = 'PIXEL'; hdr['CRPIX2'] = 1
            _wcs_λ(hdr, λ_full)
            if is_4d:
                hdr['CTYPE4'] = 'OBJECT'; hdr['CRPIX4'] = 1; hdr['CRVAL4'] = 1; hdr['CDELT4'] = 1

            if compress:
                hdul = fits.HDUList([fits.PrimaryHDU(), fits.CompImageHDU(data, header=hdr, compression_type=compression_type)])
            else:
                hdul = fits.HDUList([fits.PrimaryHDU(data, header=hdr)])

            hdul.writeto(str(output_path), overwrite=True)
            ndim_label = '4D' if is_4d else '3D'
            comp_label = f' ({compression_type} compressed)' if compress else ''
            print(f"Saved {ndim_label} model cube{comp_label} to {output_path}")


        # stem = os.path.splitext(os.path.basename(cube_path))[0]
        # suffix = '_modeled_cube_objects' if np.asarray(PSFs_separated).ndim == 4 else '_modeled_cube'
        # output_file = data_folder / f'{stem}{suffix}.fits'
        # Pass PSFs_separated (N_src, N_wvl, 111, 111) + source positions → compact multi-extension format, no zeros
        # SaveModelCubeFITS(PSFs_separated, output_file, λ_full, sources=sources)


    def cleanup(self):
        """Explicitly clean up GPU memory and delete large objects."""
        # Delete PSF model and its internal tensors
        if hasattr(self, 'PSF_model') and self.PSF_model is not None:
            del self.PSF_model
            self.PSF_model = None
        
        # Delete GPU tensors
        gpu_attrs = ['cube_sparse', 'canvas_sparse', 'valid_mask', 'simulated_sparse', 'residue_sparse']
        for attr in gpu_attrs:
            if hasattr(self, attr) and getattr(self, attr) is not None:
                tensor = getattr(self, attr)
                if isinstance(tensor, torch.Tensor) and tensor.device.type == 'cuda':
                    del tensor
                    setattr(self, attr, None)
        
        # Delete CPU tensors (to free RAM)
        cpu_attrs = ['cube_full', 'canvas_full', 'simulated_full', 'residue_full']
        for attr in cpu_attrs:
            if hasattr(self, attr) and getattr(self, attr) is not None:
                del_attr = getattr(self, attr)
                del del_attr
                setattr(self, attr, None)
        
        # Delete source data tensors
        if hasattr(self, 'sources') and self.sources is not None:
            for attr in ['spectra_sparse', 'spectra_full', 'spectra_res_sparse', 'spectra_res_full']:
                if hasattr(self.sources, attr):
                    tensor = getattr(self.sources, attr)
                    if tensor is not None:
                        del tensor
                        setattr(self.sources, attr, None)
        
        # Force GPU memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


    def __del__(self):
        """Destructor called when the object is garbage collected."""
        try:
            self.cleanup()
        except Exception:
            # Silently ignore errors during cleanup to avoid issues during interpreter shutdown
            pass



