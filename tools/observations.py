#%%
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tiptorch._config import WEIGHTS_FOLDER, default_torch_type

import torch
import gc
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from tqdm import tqdm
from matplotlib.colors import LogNorm
from typing import Optional
from dataclasses import dataclass

from data_processing.MUSE_data_utils import GetSpectrum, LoadCachedDataMUSE
from machine_learning.calibrators.NFM_calibrator import NFMCalibrator
from fitting.PSF_optimizer import OptimizePSFModel
from tiptorch.PSF_models.NFM_wrapper import PSFModelNFM
from tiptorch.PSF_models.NFM_Moffat  import MoffatPSFModelNFM
from tiptorch.tools.utils import mask_square, generate_random_colors, rad2mas, rad2arc
from tools.plotting import PlotSpetralCubeInRGB
from tools.multisources import (
    add_ROIs,
    AddSources,
    SourcesData,
    SourcesSubset,
    DetectSources,
    DisplaySources,
    VisualizeSources,
    add_ROIs_separately,
    ExtractSourceImages,
    ROI_from_valid_mask,
    PlotSourcesProfiles
)


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
    def __init__(self, raw_path, cube_path, cache_path, PSF_size=111, model_type='TipTorch', device=default_torch_type):
        self.raw_path   = raw_path
        self.cube_path  = cube_path
        self.cache_path = cache_path
        self.device     = device
        self.model_type = model_type.lower()
        self.suppress_bump_flag = False # Flag to suppress fitting of the "phase bump" NCPA of MUSE NFM
        self.suppress_LO_flag   = False # Flag to suppress fitting of the quasi-static LO aberrations
        # Helper function during the fitting
        self.force_positive_coef = lambda x: torch.clamp(-x, min=0).pow(2).mean()
        
        # Initialize no sources in the OB yet
        self.sources_table = None
        self.sources = None
        self.N_src = 0
        
        # The number of λ slices simulated per batch for the full-spectrum simulation
        self.λ_batch_size = 100
        
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
        self.PSF_size = PSF_size
        self.model_config['sensor_science']['FieldOfView'] = self.PSF_size # Sice of the PSF to simulate
        
        # Empty image to store the simulated PSFs while adding them to the right locations in the field. This is more memory efficient than storing all PSFs
        # separately and also allows to overlap sources on top of each other, which is important for a realistic simulation and fitting
        self.canvas_sparse = torch.zeros([self.N_wvl, self.cube_sparse.shape[-2], self.cube_sparse.shape[-1]], device=self.device)
        # Same idea here, but for the full-spectrum simulation. It's stored on CPU to save GPU memory since it's only used for visualization and evaluation
        self.canvas_full = torch.zeros([self.N_wvl_full, self.cube_full.shape[-2], self.cube_full.shape[-1]], device='cpu')

        # Region to plot when visualizing the sources and fitting results
        self.ROI_plot = ROI_from_valid_mask(self.valid_mask)["slice"]
        
        self.simulated_sparse = None # stores simulated field
        self.simulated_full   = None # stores simulated field for the full MUSE-NFM spectrum
        
        

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

        # Since spectral bins are the sum, they need to be re-normalized to averages to ≈ the full spectrum values range
        flux_λ_norm = torch.tensor(Δλ_full / Δλ_sparse, device=self.device, dtype=torch.float32)
        self.cube_sparse *= flux_λ_norm[:, None, None]
        
        self.λ_full   = λ_full
        self.λ_sparse = λ_sparse
        
        # Constant flux background wthin the field
        self.background_sparse = torch.zeros([self.N_wvl], device=self.device)
        self.background_full   = torch.zeros([self.N_wvl_full], device='cpu')
        
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
        self.model_config['NumberSources'] = len(self.sources)
        
        if not isinstance(self.model_config['telescope']['PupilAngle'], torch.Tensor):
            self.model_config['telescope']['PupilAngle'] = torch.tensor(self.model_config['telescope']['PupilAngle'], device=self.device)

        if self.model_type == 'tiptorch':
            self.AddSourcesToModelConfig()
            # Initialize the PSF model
            self.PSF_model = PSFModelNFM(
                self.model_config,
                multiple_obs    = False,
                LO_NCPAs        = True,
                chrom_defocus   = False,
                use_Moffat      = False,
                model_type      = 'physics-based',
                N_spline_nodes  = 5,
                Z_mode_max      = 9,
                device          = self.device,
                λ_min           = self.λ_full.min().item() * 1e-9, # [m]
                λ_max           = self.λ_full.max().item() * 1e-9, # [m]
                num_λ_slices    = len(self.λ_full)
            )
            #  Get the initial guess for the PSF model parameters
            calibrator = NFMCalibrator(WEIGHTS_FOLDER / 'NFM_calibrator/NFM_calibrator_bundle.pth', device=self.device)
            _ = calibrator.check_compatibility(self.PSF_model)
            # Change PSF model parameters based on NN's predictions
            calibrator.calibrate(self.reduced_telemetry, self.PSF_model)

            # Disable optimization of some variables
            self.PSF_model.inputs_manager.set_optimizable('bg_ctrl',  False)
            self.PSF_model.inputs_manager.set_optimizable('wind_dir', False)
        
        elif self.model_type == 'psfao':
            self.AddSourcesToModelConfig()
            # Initialize the PSF model
            self.PSF_model = PSFModelNFM(
                self.model_config,
                multiple_obs    = False,
                LO_NCPAs        = True,
                chrom_defocus   = False,
                model_type      = 'psfao',
                N_spline_nodes  = 5,
                Z_mode_max      = 9,
                device          = self.device,
                λ_min           = self.λ_full.min().item() * 1e-9, # [m]
                λ_max           = self.λ_full.max().item() * 1e-9, # [m]
                num_λ_slices    = len(self.λ_full)
            )
            self.PSF_model.inputs_manager.set_optimizable('bg_ctrl', False)
        
        elif self.model_type == 'moffat':
            self.PSF_model = MoffatPSFModelNFM(
                self.model_config,
                N_spline_nodes = 5,
                device         = self.device,
                λ_min          = self.λ_full.min().item() * 1e-9, # [m]
                λ_max          = self.λ_full.max().item() * 1e-9, # [m]
                num_λ_slices   = len(self.λ_full)
            )
            self.PSF_model.inputs_manager.set_optimizable('bg_ctrl',  False)
            
        elif self.model_type == 'gaussian':
            raise NotImplementedError("Gaussian PSF model is not implemented yet. Please use 'TipTorch' or 'Moffat' as the model type.")
        
        else:
            raise ValueError(f"Invalid model type: {self.model_type}. Supported types are: 'TipTorch', 'PSFAO', 'Moffat', 'Gaussian'.")
        
        self._update_flux_norm() # Update the flux normalization factor based on the initial guess for the PSF shape


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
            print(f"Extracted {self.N_src} source{'s' if self.N_src != 1 else ''}.")
            print(f"Num. of filtered sources on the edge of the field: {len(self.sources_table) - self.N_src}")
            

    @torch.no_grad()
    def _compute_flux_crop_factor(self, quasi_inf_PSF_size: int = 511) -> None:
        ''' Computes composite chromatic flux normalization factor. '''

        current_PSF_size = self.PSF_model.model.N_pix # the actual size of the simulated PSFs
        # Ensure that if PSF's small size is odd, then quasi-infinite PSF size is also odd to avoid sub-pixel shifts of the PSF core
        quasi_inf_PSF_size -= (1-current_PSF_size % 2)

        if self.PSF_model.use_splines:
            wvl_current = self.PSF_model.λ_sim.clone()
            self.PSF_model.SetWavelengths(self.PSF_model.λ_ctrl) # isntead of anchor λs, evaluate at spline nodes

        has_src_dirs = all(name in self.PSF_model.get_param_names() for name in ['src_dirs_x', 'src_dirs_y'])
        if has_src_dirs:
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

        if has_src_dirs:
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
        # # Since there is no calibration step involved when using purely empirical PSF models, the flux normalization can be quite far from optimum
        # # when optimization starts, which can lead to very high initial loss values and unstable optimization. To mitigate this issue, we can update
        # # the initial flux normalization factor based on the crop factor
        # if self.model_type in ('moffat', 'gaussian', 'psfao'):
        #     PSF_norm_factor_old = self.PSF_model.inputs_manager['F_norm_λ_ctrl'].clone()
        #     self._compute_flux_crop_factor() # update the core flux normalization factor based on the current PSF morphology
        #     PSF_norm_factor_new = self.PSF_model.inputs_manager['F_norm_λ_ctrl'].clone()
        #     # Compute the updated normalization factors
        #     F_norm_correction = (PSF_norm_factor_new / PSF_norm_factor_old).mean().item()
        #     self.PSF_model.inputs_manager['F_norm'] /= F_norm_correction
        # # Since for physical modelling, a data-driven calibrator gices already good-enough initial flux normalization, we can skip that step
        # else:
        self._compute_flux_crop_factor() # update the core flux normalization factor based on the current PSF morphology

        # Since F_ctrl is the parameter that directly controls the overall flux normalization in the model, we can use its mean value to correct
        # for per-source flux normalization. This is a rather empirical correction
        F_mean_correction = self.PSF_model.inputs_manager['F_ctrl'].mean().item() # Same for all PSFs
        self.PSF_model.inputs_manager['F_ctrl'] /= F_mean_correction
        self.PSF_model.inputs_manager['F_norm'] *= F_mean_correction


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
            LO         = 1e-7   if not self.suppress_LO_flag   else 1e3,
            phase_bump = 1.5e-4 if not self.suppress_bump_flag else 1e3,
            positive   = 2.0,
            unit_flux  = 0.2
        )

    
    def simulate_sparse(self, x=None, src_subset: Optional[SourcesSubset]=None, return_PSFs=False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        
        flux_normalization = F_norm.view(-1, 1) * PSF_norm_factor
        PSFs = PSFs_ * (flux_normalization * src_subset.spectra_sparse).view(-1, self.N_wvl, 1, 1)
        
        if not return_PSFs:
            return add_ROIs(self.canvas_sparse * 0.0, PSFs, src_subset.slices_local, src_subset.slices_global), None, flux_normalization
        
        return add_ROIs(self.canvas_sparse * 0.0, PSFs, src_subset.slices_local, src_subset.slices_global), PSFs_, flux_normalization


    @torch.no_grad()
    def simulate_full(self, src_subset: Optional[SourcesSubset]=None, return_PSFs=False, λ_batch_size=None, force_cpu=True, verbose=False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if src_subset is None:
            src_subset = self.sources.select(None)

        all_selected = len(src_subset.ids) == self.N_src
        # No inputs since SimulateFullSpectrum() fully relies on the inputs stored in the inputs_manager, which are already up-to-date
        PSFs_full = self.PSF_model.SimulateFullSpectrum(
            src_ids = None if all_selected else src_subset.ids,
            λ_batch_size = self.λ_batch_size if λ_batch_size is None else λ_batch_size,
            verbose = verbose,
            force_cpu = force_cpu
        )
        # Determine target device for all intermediate tensors
        target_device = 'cpu' if force_cpu else self.device
        # Now, chromatic PSF normalization factor must be re-evaluated for the full NFM spectrum
        PSF_norm_factor_full = self.PSF_model.evaluate_splines(self.PSF_model.inputs_manager['F_norm_λ_ctrl'], self.PSF_model.λ_full_normed)
        F_norm = self.PSF_model.inputs_manager['F_norm'] if all_selected else self.PSF_model.inputs_manager['F_norm'][src_subset.ids]

        flux_normalization = (F_norm.view(-1, 1) * PSF_norm_factor_full).to(target_device)
        # Apply the flux scaling; spectra_full is on CPU, move to target device if needed
        spectra_full = src_subset.spectra_full.to(target_device)
        PSFs_ = PSFs_full * (flux_normalization * spectra_full).view(-1, self.N_wvl_full, 1, 1)
        canvas = self.canvas_full.to(target_device)
        simulated_full = add_ROIs(canvas, PSFs_, src_subset.slices_local, src_subset.slices_global)

        if not return_PSFs:
            return simulated_full, None, flux_normalization
        
        return simulated_full, PSFs_full, flux_normalization


    @torch.no_grad()
    def simulate_range(
        self,
        λ_min        = None,
        λ_max        = None,
        src_subset: Optional[SourcesSubset] = None,
        return_PSFs  = False,
        sequential   = True,
        λ_batch_size = None,
        force_cpu    = True,
        verbose      = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Simulate the field over a spectral sub-range of the full MUSE NFM spectrum.
        Mirrors simulate_full but delegates to PSFModelNFM.SimulateSpectralRange so
        only the requested wavelength slice is computed.

        Parameters
        ----------
        λ_min, λ_max : float or None
            Wavelength bounds in the same units as PSF_model.λ_full (metres).
            None falls back to the respective end of the full spectrum.
        src_subset : SourcesSubset or None
            Sources to simulate. None selects all sources.
        return_PSFs : bool
            If True, the raw (un-scaled) PSF cube is included in the return tuple.
        sequential : bool
            Passed to SimulateSpectralRange. True = one source per forward call
            (lower VRAM); False = all sources in one call per λ-batch (faster).
        λ_batch_size : int or None
            Wavelength slices per forward pass. Defaults to self.λ_batch_size.
        force_cpu : bool
            Keep output tensors on CPU to avoid VRAM overflow.
        verbose : bool
            Show a progress bar over wavelength batches.

        Returns
        -------
        simulated_range  : torch.Tensor [N_λ_range, H, W]
        PSFs_range       : torch.Tensor [N_src_sim, N_λ_range, N_pix, N_pix] or None
        flux_normalization : torch.Tensor [N_src_sim, N_λ_range]
        λ_range          : torch.Tensor [N_λ_range]  - wavelengths that were simulated
        """
        if src_subset is None:
            src_subset = self.sources.select(None)

        all_selected = len(src_subset.ids) == self.N_src
        batch_size   = λ_batch_size if λ_batch_size is not None else self.λ_batch_size

        PSFs_range, λ_range = self.PSF_model.SimulateSpectralRange(
            λ_min        = λ_min,
            λ_max        = λ_max,
            src_ids      = None if all_selected else src_subset.ids,
            λ_batch_size = batch_size,
            sequential   = sequential,
            verbose      = verbose,
            force_cpu    = force_cpu,
        )

        target_device = 'cpu' if force_cpu else self.device
        N_λ_range     = len(λ_range)

        # Evaluate the chromatic PSF normalisation factor at the simulated sub-range wavelengths
        λ_range_normed       = self.PSF_model.norm_wvl(λ_range.to(self.PSF_model.device)).to(target_device)
        PSF_norm_factor_range = self.PSF_model.evaluate_splines(
            self.PSF_model.inputs_manager['F_norm_λ_ctrl'], λ_range_normed
        ).to(target_device)

        F_norm_raw = self.PSF_model.inputs_manager['F_norm'] if all_selected else self.PSF_model.inputs_manager['F_norm'][src_subset.ids]
        F_norm     = F_norm_raw.to(target_device)
        flux_normalization = F_norm.view(-1, 1) * PSF_norm_factor_range  # [N_src_sim, N_λ_range]

        # Slice the source spectra to the same wavelength sub-range
        wvl_mask    = (self.PSF_model.λ_full >= λ_range[0]) & (self.PSF_model.λ_full <= λ_range[-1])
        spectra_range = src_subset.spectra_full[:, wvl_mask].to(target_device)  # [N_src_sim, N_λ_range]

        PSFs_ = PSFs_range * (flux_normalization * spectra_range).view(-1, N_λ_range, 1, 1)

        # Build a zero canvas matching the sub-range spectral depth
        canvas = torch.zeros(
            (N_λ_range, self.canvas_full.shape[-2], self.canvas_full.shape[-1]),
            device=target_device, dtype=self.canvas_full.dtype
        )
        simulated_range = add_ROIs(canvas, PSFs_, src_subset.slices_local, src_subset.slices_global)

        if not return_PSFs:
            return simulated_range, None, flux_normalization, λ_range

        return simulated_range, PSFs_range, flux_normalization, λ_range


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
        if 'LO_coefs' not in self.PSF_model.get_param_names():
            return torch.zeros((), device=self.cube_sparse.device, dtype=self.cube_sparse.dtype)

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
        """ Helper function to select which parameters of the PSF model should be optimized based on the user's choice """
        if isinstance(fit_params, str):
            fit_params = [fit_params]
        
        # Determine which parameters must be fitted
        PSF_params        = ['r0', 'dn', 'LO_coefs', 'F_ctrl', 'J_ctrl', 'L0', 'wind_speed_single', 'wind_dir_single', 'Cn2_weights',
                             'alpha_x_ctrl', 'alpha_y_ctrl', 'beta_ctrl', 'theta', # pixel-space Moffat parameters
                             'amp', 'b', 'alpha', 'beta', 'ratio']  # Moffat PSD params (PSFAO / hybrid modes)
        astrometry_params = ['dx_ctrl', 'dy_ctrl']
        photometry_params = ['bg_ctrl', 'F_norm_λ_ctrl', 'F_norm']

        if 'PSF' in fit_params and 'astrometry' in fit_params and 'photometry' in fit_params:
            fitable_set = None # all currently selected model parameters are optimized
        else:
            fitable_set = (PSF_params        if 'PSF'        in fit_params else []) + \
                          (astrometry_params if 'astrometry' in fit_params else []) + \
                          (photometry_params if 'photometry' in fit_params else [])

            optimizable_names = set(self.PSF_model.get_optimizable_param_names())
            fitable_set = [param for param in fitable_set if param in optimizable_names]
        return fitable_set


    # TODO: provide sources selection from outside
    def FitPSFModel(self, fit=['PSF', 'astrometry', 'photometry'], repeat=2, max_iter=200) -> None:
        import gc
        # Vector to store the encoded PSF model parameters
        x_params = None
        
        optimizables_ = self._select_optimizable_variables(fit)
        weights_      = self._compute_fitting_weights()
        
        fit_subset = weights_.subset  # pre-built once, reused every iteration
        
        run_fn  = lambda x: self.simulate_sparse(x, fit_subset)[0] # inference function used by the optimizer
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
        # NOTE: Intrinsic values inside PSF_model.inputs manager are updated automatically on every function call
             
        # Clean up GPU memory after fitting
        gc.collect()
        torch.cuda.empty_cache()
        

    def SimulateField(self, full_spectrum=False, N_src_per_batch=None, N_λ_per_batch=None, disentangle_spectra=True, force_cpu=True) -> torch.Tensor:
        ''' Simulate all sources within the field. If full_spectrum is True, simulates the full spectrum instead of the sparse λ-subset. '''
        # ------------- In this branch, all spectral slices from the full range are simulated
        if full_spectrum:
            
            if N_λ_per_batch is None:
                N_λ_per_batch = self.λ_batch_size
            
            if disentangle_spectra:
                # This function simulates raw PSFs per λ-batch and disentangles in one pass
                I_sim_full, fluxes_full, bg_full = self.disentangle_flux_batched(
                    λ_batch_size = N_λ_per_batch,
                    solver = 'linear', # for saving resources, let it be linear
                    verbose = False,
                )
                simulated = I_sim_full - bg_full.view(-1, 1, 1)
                self.background_full   = bg_full
                
                # Compute PSF normalization factors for the full spectral range. Without disentangling, it is done inside simulate_full()
                PSF_norm_factor_full = self.PSF_model.evaluate_splines(self.PSF_model.inputs_manager['F_norm_λ_ctrl'], self.PSF_model.λ_full_normed)
                F_norm = self.PSF_model.inputs_manager['F_norm']
                # The F_PSF factor is used to correct for the overall flux normalization changes during fitting, but it has the same value for all sources,
                # so it doesn't affect the relative spectra shapes. However, it's important to apply it to get the correct absolute flux values in the spectra,
                # assuming that ∑ PSF(λ) ≡ 1 for ∀ PSF ∈ field
                F_PSF = self.PSF_model.evaluate_splines(self.PSF_model.inputs_manager['F_ctrl'], self.PSF_model.λ_full_normed)
                
                flux_normalization = (F_norm.view(-1, 1) * PSF_norm_factor_full).cpu()
                
                self.sources.spectra_full_true = fluxes_full.T / F_PSF.cpu()        # True astrophysical decoupled sources spectrum
                self.sources.spectra_full      = fluxes_full.T / flux_normalization # [N_src, N_wvl_full], on CPU
                
            else:
                with torch.no_grad():
                    if N_src_per_batch is None:
                        simulated, _, _ = self.simulate_full(λ_batch_size=N_λ_per_batch)
                    else:
                        source_indices = np.arange(len(self.sources))
                        np.random.shuffle(source_indices)
                        batches = np.array_split(source_indices, (len(self.sources) + N_src_per_batch - 1) // N_src_per_batch)

                        device_ = 'cpu' if force_cpu else self.device

                        simulated = torch.zeros([self.N_wvl_full, self.cube_full.shape[-2], self.cube_full.shape[-1]], device=device_, dtype=self.PSF_model.dtype)

                        for batch in tqdm(batches):
                            src_subset = self.sources.select(batch)
                            simulated += self.simulate_full(src_subset=src_subset, λ_batch_size=N_λ_per_batch, force_cpu=force_cpu)[0]

            self.simulated_full = simulated.cpu() if force_cpu else simulated
            self.residue_full   = (self.cube_full - self.simulated_full.cpu().numpy()) * self.valid_mask.cpu().numpy()
                
        
        # ------------- In this branch, only the fast simulation of the sparse subset is done
        else: #TODO: PSFs must add-up exactly to 1 per λ when getting the true spectrum
            with torch.no_grad():
                simulated, PSFs, flux_normalization = self.simulate_sparse(x=None, src_subset=None, return_PSFs=disentangle_spectra)
            
            if disentangle_spectra:
                I_sim, fluxes, bg_solved = self.disentangle_flux(self.sources.select(None), PSFs, self.cube_sparse, solver='nonlinear')
                simulated = I_sim - bg_solved.view(self.N_wvl, 1, 1) # Leave the pure PSF contribution in the simulated image
                self.background_sparse = bg_solved
                
                F_PSF = self.PSF_model.evaluate_splines(self.PSF_model.inputs_manager['F_ctrl'], self.PSF_model.λ_sim_normed)
                
                # spectra_sparse stores flux per PSF core, not full flux per source, that's why we need it here
                self.sources.spectra_sparse      = fluxes.T / flux_normalization # update the previous spectra estimates with the disentangling results
                self.sources.spectra_sparse_true = fluxes.T / F_PSF
                
            self.simulated_sparse = simulated
            self.residue_sparse = (self.cube_sparse - simulated) * self.valid_mask
        
        gc.collect()
        torch.cuda.empty_cache()

        return simulated if not force_cpu else simulated.cpu()


    def GetSpectrum(self, source_id=None):
        if self.sources is None:
            raise ValueError("Sources must be initialized before getting spectra. Please run ExtractSources() first.")
        
        if all(s is None for s in [self.sources.spectra_full_true]):
            raise ValueError("Spectra are not yet computed. Please simulate the full first field with spectra disentangling enabled by running SimulateField(full_spectrum=True)")
        
        return self.sources.spectra_full if source_id is None else self.sources.spectra_full[source_id]
        

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
   
    @staticmethod
    def disentangle_flux(
        srcs: SourcesSubset,
        PSFs,
        data_cube,
        solver           = 'linear',   # 'linear' | 'nonlinear'
        rcond            = 1e-2,
        n_iter           = 1000,
        lr               = 1e-2,
        init_from_lstsq  = True,
        bg_unconstrained = True,
        verbose          = False,
        nan_policy       = "zero",
        grad_clip        = None,
        preferred_device = None,
    ):
        """
        Reconstruct spectral cube by solving  P @ spectrum + bg ≈ I_data.
        Can do this with a single linear least-squares solve or with an iterative non-linear optimization
        enforcing non-negativity on the spectra and optionally on the background as well.

        Returns
        -------
        I_sim    : torch.Tensor  [N_wvl, H, W]
        spectrum : torch.Tensor  [N_wvl, N_src]
        bg       : torch.Tensor  [N_wvl]
        """
        N_src  = PSFs.shape[0]
        N_wvl  = PSFs.shape[1]
        device = preferred_device if preferred_device is not None else PSFs.device
        dtype  = PSFs.dtype

        PSFs = PSFs.to(device=device, dtype=dtype)
        data_cube = (data_cube if isinstance(data_cube, torch.Tensor)
                    else torch.tensor(data_cube)).to(device=device, dtype=dtype)

        def sanitize(x, name):
            if torch.isfinite(x).all(): return x
            if nan_policy == "raise":   raise FloatingPointError(f"{name} contains {(~torch.isfinite(x)).sum().item()} NaN/Inf values.")
            if nan_policy == "zero":    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            raise ValueError("nan_policy must be 'zero' or 'raise'.")

        # Scatter PSFs onto the field canvas → design matrix P
        with torch.no_grad():
            PSFs_ = PSFs if solver == 'linear' else sanitize(PSFs, "PSFs")
            canvas = torch.zeros_like(data_cube, device=device, dtype=dtype)
            srcs_stack = add_ROIs_separately(canvas, PSFs_, srcs.slices_local, srcs.slices_global)
            del canvas, PSFs_

            P      = srcs_stack.view(N_src, N_wvl, -1).permute(1, 2, 0).contiguous()  # [N_wvl, pixels, N_src]
            I_data = data_cube.view(N_wvl, -1, 1)                                      # [N_wvl, pixels, 1]
            del srcs_stack

            if solver == 'nonlinear':
                P      = sanitize(P,      "P")
                I_data = sanitize(I_data, "I_data")

        N_wvl, N_pix, N_src = P.shape

        # ======================== Linear solver ========================
        if solver == 'linear':
            ones  = torch.ones(N_wvl, N_pix, 1, device=device, dtype=dtype)
            P_aug = torch.cat([P, ones], dim=-1)
            sol   = torch.linalg.lstsq(P_aug, I_data, rcond=rcond).solution
            spectrum = sol[:, :N_src, :].clamp(min=0)
            bg       = sol[:, N_src:,  :]
            I_sim    = (torch.matmul(P, spectrum) + bg).squeeze(-1).view(N_wvl, data_cube.shape[-2], data_cube.shape[-1])
            return I_sim, spectrum.squeeze(-1), bg.view(N_wvl)

        # ======================== Non-linear solver ========================
        def softplus_inv(y):
            eps = torch.finfo(y.dtype).eps
            y   = sanitize(y, "softplus_inv input").clamp_min(eps)
            return torch.where(y > 20.0, y, torch.log(torch.expm1(y).clamp_min(eps)))

        # Initialize from lstsq solution (warm-starting across λ-batches is not
        # useful here: spectra vary strongly per wavelength while only PSF shape
        # is smooth, so lstsq gives a better starting point every time)
        if init_from_lstsq:
            ones  = torch.ones(N_wvl, N_pix, 1, device=device, dtype=dtype)
            P_aug = sanitize(torch.cat([P, ones], dim=-1), "P_aug")
            with torch.no_grad():
                sol           = sanitize(torch.linalg.lstsq(P_aug, I_data, rcond=rcond).solution, "lstsq init")
                spec_init     = sanitize(sol[:, :N_src, 0], "spec_init").clamp_min(0)
                bg_init       = sanitize(sol[:,  N_src, 0], "bg_init")
                raw_spec_init = softplus_inv(spec_init)
                raw_bg_init   = bg_init if bg_unconstrained else softplus_inv(bg_init.clamp_min(0))
        else:
            raw_spec_init = torch.zeros(N_wvl, N_src, device=device, dtype=dtype)
            raw_bg_init   = torch.zeros(N_wvl,        device=device, dtype=dtype)

        raw_spectrum = torch.nn.Parameter(raw_spec_init.clone())
        raw_bg       = torch.nn.Parameter(raw_bg_init.clone())
        # P and I_data are plain tensors (no requires_grad); gradients only flow
        # through raw_spectrum / raw_bg, keeping the computation graph minimal.
        optimizer = torch.optim.Adam([raw_spectrum, raw_bg], lr=lr)

        for i in range(n_iter):
            optimizer.zero_grad(set_to_none=True)
            spectrum = F.softplus(raw_spectrum)
            bg       = raw_bg if bg_unconstrained else F.softplus(raw_bg)
            loss     = torch.mean((torch.matmul(P, spectrum.unsqueeze(-1)) + bg.view(N_wvl, 1, 1) - I_data) ** 2)

            if not torch.isfinite(loss):
                if verbose: print(f"iter {i:05d} | non-finite loss, stopping")
                break

            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_([raw_spectrum, raw_bg], grad_clip)

            # Sanitize gradients in-place before the step
            with torch.no_grad():
                for p in [raw_spectrum, raw_bg]:
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0)

            optimizer.step()

            if verbose and (i % 100 == 0 or i == n_iter - 1):
                print(f"iter {i:05d} | loss = {loss.item():.6e}")

        with torch.no_grad():
            spectrum = sanitize(F.softplus(raw_spectrum), "final spectrum")
            bg       = sanitize(raw_bg if bg_unconstrained else F.softplus(raw_bg), "final bg")
            I_sim    = sanitize(
                (torch.matmul(P, spectrum.unsqueeze(-1)) + bg.view(N_wvl, 1, 1))
                .squeeze(-1).view(N_wvl, data_cube.shape[-2], data_cube.shape[-1]),
                "I_sim"
            )

        return I_sim.detach(), spectrum.detach(), bg.detach()


    def disentangle_flux_batched(
        self,
        λ_batch_size  = 100,
        solver        = 'nonlinear',
        n_iter        = 1000,
        verbose       = False,
        **disentangle_kwargs
    ):
        """
        Disentangle source fluxes for all sources simultaneously, batching only
        over wavelengths. Simpler than the tiled approach: no spatial subdivision,
        one DisentangleFlux call per λ-batch over the entire field.

        PSFs for each λ-batch are simulated on-the-fly via
        PSFModelNFM.SimulateSpectralRange so that only one batch of PSFs lives in
        memory at a time.

        Parameters
        ----------
        λ_batch_size  : int   - wavelength slices per DisentangleFlux call
        solver        : str   - 'linear' or 'nonlinear'
        n_iter        : int   - Adam iterations per λ-batch
        verbose       : bool  - show per-batch progress information
        **disentangle_kwargs  - forwarded verbatim to DisentangleFlux

        Returns
        -------
        I_sim_full   : torch.Tensor [N_wvl_full, H, W]  on CPU
        fluxes_full  : torch.Tensor [N_wvl_full, N_src] on CPU
        bg_full      : torch.Tensor [N_wvl_full]        on CPU
        """
        
        # Clear GPU memory before starting the batch processing loop
        gc.collect()
        torch.cuda.empty_cache()
            
        srcs       = self.sources.select(None)   # all sources, full-field coords
        dtype      = self.PSF_model.dtype
        λ_full     = self.PSF_model.λ_full       # [N_wvl_full], in metres
        N_wvl_full = self.N_wvl_full
        H, W       = self.cube_full.shape[-2], self.cube_full.shape[-1]

        I_sim_full  = torch.zeros([N_wvl_full, H, W],       device='cpu', dtype=dtype)
        fluxes_full = torch.zeros([N_wvl_full, self.N_src], device='cpu', dtype=dtype)
        bg_full     = torch.zeros([N_wvl_full],             device='cpu', dtype=dtype)

        for λ0 in tqdm(range(0, N_wvl_full, λ_batch_size), desc='Disentangling λ-batches'):
            λ1 = min(λ0 + λ_batch_size, N_wvl_full)

            # Simulate raw PSFs for this spectral sub-range on-the-fly
            PSFs_batch, _ = self.PSF_model.SimulateSpectralRange(
                λ_min        = λ_full[λ0].item(),
                λ_max        = λ_full[λ1 - 1].item(),
                src_ids      = None,
                λ_batch_size = λ1 - λ0,
                sequential   = False,
                force_cpu    = False,
                verbose      = False,
            )

            data_batch = torch.tensor(self.cube_full[λ0:λ1]).to(self.device)

            I_sim_batch, spec_batch, bg_batch = self.disentangle_flux(
                srcs,
                PSFs_batch,
                data_batch,
                solver  = solver.replace('non-', 'non'),
                n_iter  = n_iter,
                verbose = verbose,
                **disentangle_kwargs
            )

            I_sim_full [λ0:λ1] = I_sim_batch.cpu()
            fluxes_full[λ0:λ1] = spec_batch.cpu()
            bg_full    [λ0:λ1] = bg_batch.cpu()

            del PSFs_batch, data_batch, I_sim_batch, spec_batch, bg_batch
            gc.collect()
            torch.cuda.empty_cache()
            
        
        return I_sim_full, fluxes_full, bg_full


    def ExtractBackgroundFromResidue(self, residue, min_radius=2, max_radius=14, border_margin=15, show=False):
        """
        Estimate a per-wavelength background level from the residue cube by
        averaging over pixels that are not contaminated by sources or borders.

        Works on both the sparse cube (N_wvl slices) and the full cube (~4 K
        slices) without copying the input tensor — only a small 2-D boolean
        mask and a 1-D pixel-index vector are allocated.

        Parameters
        ----------
        residue      : torch.Tensor [N_λ, H, W]
            Residue cube (data - model).  Not modified in place.
        min_radius   : float
            Minimum masking radius around faint sources, in pixels.
        max_radius   : float
            Maximum masking radius around bright sources, in pixels.
        border_margin : int
            Width of the border strip to exclude, in pixels.
        show         : bool
            If True, display a diagnostic image of the masked residue sum.

        Returns
        -------
        bg : torch.Tensor [N_λ, 1, 1]
            Mean background per wavelength slice, ready to broadcast over (H, W).
        """
        if self.sources is None:
            raise ValueError("Sources must be initialised before calling ExtractBackgroundFromResidue().")

        N_λ, H, W = residue.shape
        device     = residue.device

        # ---- build the 2-D exclusion mask (True = exclude) ---------------
        src_positions = self.sources.table[['x_peak', 'y_peak']].values
        src_fluxes    = self.sources.table['peak_value'].values

        log_fluxes = np.log10(src_fluxes + 1e-10)
        log_min, log_max = log_fluxes.min(), log_fluxes.max()
        radii = min_radius + (log_fluxes - log_min) / (log_max - log_min + 1e-10) * (max_radius - min_radius)

        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )

        mask = torch.zeros(H, W, dtype=torch.bool, device=device)

        for (x_src, y_src), r in zip(src_positions, radii):
            dist_sq = (x_grid - x_src) ** 2 + (y_grid - y_src) ** 2
            mask |= dist_sq <= r ** 2

        mask |= (x_grid <  border_margin) | (x_grid >= W - border_margin)
        mask |= (y_grid <  border_margin) | (y_grid >= H - border_margin)

        # valid pixel indices as a 1-D flat index — allocated once, reused for all λ
        valid_idx = (~mask).flatten().nonzero(as_tuple=True)[0]  # [N_valid]

        if valid_idx.numel() == 0:
            raise RuntimeError("ExtractBackgroundFromResidue: no unmasked pixels remain. "
                               "Try reducing min_radius or border_margin.")

        # ---- background estimate: mean over valid pixels per λ, no copy -----
        # residue[:, valid_pixels] gathers only the unmasked columns — O(N_λ * N_valid)
        # instead of O(N_λ * H * W) for a full clone.
        residue_flat  = residue.view(N_λ, H * W)          # view, no copy
        bg_vals       = residue_flat[:, valid_idx].mean(dim=-1)  # [N_λ]

        if show:
            display_norm = LogNorm(vmin=1, vmax=float(self.cube_sparse.sum(dim=0).max()))
            masked_sum   = residue.sum(dim=0).clone()
            masked_sum[mask] = 0.0
            plt.figure()
            plt.imshow(masked_sum.cpu(), origin='lower', norm=display_norm, cmap='inferno')
            plt.title('Background mask (sources + border excluded)')
            plt.axis('off')
            plt.show()

        return bg_vals.view(N_λ, 1, 1)


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
            raise NotImplementedError("Residual spectra plotting is not properly implemented yet.")
            # if self.sources.spectra_res_full is None or self.sources.spectra_res_sparse is None:
            #     print("Residual spectra are not available. Computing them now...")
            #     self.sources.spectra_res_sparse, self.sources.spectra_res_full = \
            #         self.ExtractSpectraFromCore(self.sources.table, self.residue_sparse, self.residue_full)
                
        src_subset = self.sources.select(src_ids) # select a subset of sources to plot
        
        if plot_residual:
            raise NotImplementedError("Residual spectra plotting is not properly implemented yet.")
            # src_spectra_full   = src_subset.spectra_res_full
            # src_spectra_sparse = src_subset.spectra_res_sparse
        else:
            src_spectra_full   = src_subset.spectra_full_true
            src_spectra_sparse = src_subset.spectra_sparse_true
                  
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

            # Shared vmin/vmax across data, model, and residual
            def _to_np(x): return x.numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
            data_white = _to_np(self.cube_full     [self.ROI_plot]).sum(axis=0)
            sim_white  = _to_np(self.simulated_full[self.ROI_plot]).sum(axis=0)
            res_white  = np.abs(_to_np(self.residue_full[self.ROI_plot])).sum(axis=0)

            all_white = np.stack([data_white, sim_white, res_white])
            vmax = float(np.nanmax(all_white))
            pos  = all_white[all_white > 0]
            vmin = float(max(1.0, np.nanpercentile(pos, 1)) if pos.size > 0 else 1.0)

            for cube, title in [(self.residue_full, "Difference"), (self.cube_full, "Data"), (self.simulated_full, "Model")]:
                _ = PlotSpetralCubeInRGB(
                    cube[self.ROI_plot],
                    wavelengths=λ_vis,
                    title=title,
                    min_val=vmin, max_val=vmax,
                    scale='log',
                    show=title != "Difference"
                )
            
            if plot_profiles:
                PlotSourcesProfiles(self.cube_full, self.simulated_full, self.sources.table, radius=16, title='Source radial profiles (full spectrum)')                
        
        else:
            data_white = self.cube_sparse.sum(dim=0)
            sim_white  = self.simulated_sparse.sum(dim=0)
            vmax = float(max(data_white.max(), sim_white.max()))
            vmin = float(max(1.0, min(data_white[data_white > 0].min(), sim_white[sim_white > 0].min())))
            display_norm = LogNorm(vmin=vmin, vmax=vmax)
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
        in the full field. No zeros are stored at all - this is the recommended format.
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
        gpu_attrs = ['cube_sparse', 'canvas_sparse', 'valid_mask', 'simulated_sparse', 'residue_sparse', 'λ_sparse']
        for attr in gpu_attrs:
            if hasattr(self, attr) and getattr(self, attr) is not None:
                tensor = getattr(self, attr)
                if isinstance(tensor, torch.Tensor) and tensor.device.type == 'cuda':
                    tensor.detach().cpu()  # Move to CPU first to ensure cleanup
                    del tensor
                    setattr(self, attr, None)
        
        # Delete CPU tensors (to free RAM)
        cpu_attrs = ['cube_full', 'canvas_full', 'simulated_full', 'residue_full', 'λ_full', 'flux_λ_norm']
        for attr in cpu_attrs:
            if hasattr(self, attr) and getattr(self, attr) is not None:
                tensor = getattr(self, attr)
                if isinstance(tensor, torch.Tensor):
                    if tensor.device.type == 'cuda':
                        tensor.detach().cpu()
                del tensor
                setattr(self, attr, None)
        
        # Delete source data tensors
        if hasattr(self, 'sources') and self.sources is not None:
            for attr in ['spectra_sparse', 'spectra_full', 'spectra_res_sparse', 'spectra_res_full', 'imgs_sparse']:
                if hasattr(self.sources, attr):
                    tensor = getattr(self.sources, attr)
                    if tensor is not None:
                        if isinstance(tensor, torch.Tensor) and tensor.device.type == 'cuda':
                            tensor.detach().cpu()
                        del tensor
                        setattr(self.sources, attr, None)
        
        # Delete model config and other large objects
        if hasattr(self, 'model_config') and self.model_config is not None:
            self.model_config = None
        
        if hasattr(self, 'reduced_telemetry') and self.reduced_telemetry is not None:
            self.reduced_telemetry = None
        
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




# %%
import matplotlib.patches as patches


def split_field_into_tiles(image_shape: tuple, N_tiles_x: int, N_tiles_y: int, border_offset: int=0) -> list:
    """
    Split an image into N_tiles_x x N_tiles_y tiles with an optional border offset.

    Args:
        image_shape: Tuple of (height, width) for the image
        N_tiles_x: Number of tiles along x-axis
        N_tiles_y: Number of tiles along y-axis
        border_offset: Pixels to exclude from borders

    Returns:
        List of dictionaries containing tile information with x_range and y_range
    """
    height, width = image_shape[-2:]

    # Calculate effective dimensions after applying border offset
    eff_height = height - 2 * border_offset
    eff_width  = width  - 2 * border_offset

    # Calculate tile sizes
    tile_height = eff_height // N_tiles_y
    tile_width = eff_width // N_tiles_x

    tiles = []

    for i in range(N_tiles_y):
        for j in range(N_tiles_x):
            # Calculate tile boundaries with offset
            y_min = border_offset + i * tile_height
            y_max = border_offset + (i + 1) * tile_height
            x_min = border_offset + j * tile_width
            x_max = border_offset + (j + 1) * tile_width

            # Ensure the last tiles include any remaining pixels
            if i == N_tiles_y - 1:  y_max = height - border_offset
            if j == N_tiles_x - 1:  x_max = width - border_offset

            tile_info = {
                'ROI': (slice(y_min, y_max), slice(x_min, x_max)),
                'ID': (i, j)
            }
            tiles.append(tile_info)

    return tiles


def visualize_field_tiles(
    image,
    tiles: list,
    title: str = 'Field Tiles',
    cmap: str = 'gray',
    norm = None,
    alpha: float = 0.7
) -> None:
    """
    Visualize the tiling of an image.

    Args:
        image: The image data to display
        tiles: List of tile dictionaries as returned by split_field_into_tiles
        title: Plot title
        cmap: Colormap for the image display
        norm: Normalization for the image display
        alpha: Alpha value for the rectangle overlay
    """
    if torch.is_tensor(image):
        if image.dim() > 2:
            # If multi-channel/wavelength, take mean or sum
            display_img = image.mean(dim=0).cpu().numpy()
        else:
            display_img = image.cpu().numpy()
    else:
        if image.ndim > 2:
            # If multi-channel/wavelength, take mean or sum
            display_img = image.mean(axis=0)
        else:
            display_img = image

    plt.figure(figsize=(10, 8))
    plt.imshow(display_img, origin='lower', cmap=cmap, norm=norm)

    colors = plt.cm.tab10.colors

    for i, tile in enumerate(tiles):
        slice_y, slice_x = tile['ROI']
        y_min, y_max = slice_y.start, slice_y.stop
        x_min, x_max = slice_x.start, slice_x.stop
        width = x_max - x_min
        height = y_max - y_min

        color = colors[i % len(colors)]
        rect = patches.Rectangle(
            (x_min, y_min), width, height,
            linewidth=2, edgecolor=color, facecolor='none', alpha=alpha
        )
        plt.gca().add_patch(rect)

        # Add tile ID label
        plt.text(
            x_min + width/2, y_min + height/2,
            f"Tile {tile['ID']}", color=color,
            ha='center', va='center', fontweight='bold'
        )

    plt.title(title)
    plt.tight_layout()
    plt.show()
