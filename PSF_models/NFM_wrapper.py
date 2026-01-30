# %reload_ext autoreload
# %autoreload 2

from logging import warning
import sys

from zmq import has
sys.path.insert(0, '..')

import torch
import numpy as np
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline

from PSF_models.TipTorch import TipTorch
from tools.static_phase import ArbitraryBasis, PixelmapBasis, ZernikeBasis, MUSEPhaseBump
from tools.utils import PupilVLT
from managers.config_manager import MultipleTargetsInDifferentObservations
from managers.input_manager  import InputsManager, InputsManagersUnion
from tools.normalizers import Uniform, Uniform0_1, SoftmaxInv, Identity

from warnings import warn
from project_settings import device
import gc


class PSFModelNFM:
    def __init__(
        self,
        config,
        multiple_obs    = False,
        LO_NCPAs        = True,
        chrom_defocus   = False,
        use_splines     = False,
        Moffat_absorber = False,
        Z_mode_max      = 9,
        device          = device
    ):
        
        self.Z_mode_max  = Z_mode_max
        self.LO_N_params = None
        
        self.LO_NCPAs        = LO_NCPAs
        self.multiple_obs    = multiple_obs
        self.device          = device
        self.Moffat_absorber = Moffat_absorber
        self.use_splines     = use_splines
        self.chrom_defocus   = chrom_defocus and LO_NCPAs
        self.__config_raw    = config # Input config file
        
        _config = self._init_configs(self.__config_raw) # Processed and converted config
        self.wavelengths = _config['sources_science']['Wavelength'].squeeze()
        
        # Spectrum of MUSE NFM
        self.λ_min = 475.e-9
        self.λ_max = 935.e-9
        self.Δλ = 0.125e-9
        self.num_λ_slices = 3681
        self.λ_full = np.arange(self.num_λ_slices) * self.Δλ + self.λ_min
        
        # Initialize TipTorch model
        self._init_model(_config)
        # Initialize LO NCPAs
        if LO_NCPAs:
            self._init_NCPAs()
        
        self.polychromatic_params = ['F', 'dx', 'dy', 'bg'] + (['chrom_defocus'] if self.chrom_defocus else ['J'])

        if self.use_splines:
            self.N_wvl_ctrl = 5
            self.norm_wvl = Uniform0_1(a=self.λ_min, b=self.λ_max) # MUSE NFM wavelength range
            self.λ_ctrl   = torch.linspace(0, 1, self.N_wvl_ctrl, device=self.device) # normalized to [0...1]
            self.λ_ctrl_denorm = self.norm_wvl.inverse(self.λ_ctrl) # [nm]

        self._init_model_inputs()
        
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        if self.device.type == 'mps':
            torch.mps.empty_cache()
            torch.mps.synchronize()
    

    # def _cleanup_dict_recursive(self, obj):
    #     """Recursively clean up tensors in nested dictionaries"""
    #     if isinstance(obj, dict):
    #         for key in list(obj.keys()):
    #             self._cleanup_dict_recursive(obj[key])
    #             del obj[key]
    #     elif isinstance(obj, (list, tuple)):
    #         for item in obj:
    #             self._cleanup_dict_recursive(item)
    #     elif isinstance(obj, torch.Tensor):
    #         del obj
    
    
    def cleanup(self):
        """Explicitly clean up GPU memory"""       
        # Clean up model (which will trigger TipTorch cleanup)
        del self.inputs_manager, self.wavelengths
        
        if self.use_splines:
            del self.LO_basis, self.λ_ctrl, self.norm_wvl

        self.model.cleanup()
        gc.collect()
        
        # Clear cache
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        if self.device.type == 'mps':
            torch.mps.empty_cache()
            torch.mps.synchronize()
    
    
    def __del__(self):
        """Class destructor to ensure the GPU memory is freed"""
        try:
            self.cleanup()
        except Exception:
            pass  # Avoid errors during interpreter shutdown
    
 
    def copy(self):
        """
        Create a deep copy of the PSFModelNFM instance, including the internal TipTorch model.
        
        Returns:
        --------
        PSFModelNFM
            A new instance with copied parameters and model state
        """
        import copy
        
        # Create a new instance with the same configuration
        new_instance = PSFModelNFM(
            config          = copy.deepcopy(self.__config_raw),
            multiple_obs    = self.multiple_obs,
            LO_NCPAs        = self.LO_NCPAs,
            chrom_defocus   = self.chrom_defocus,
            use_splines     = self.use_splines,
            Moffat_absorber = self.Moffat_absorber,
            Z_mode_max      = self.Z_mode_max,
            device          = self.device
        )
        # Deep copy the TipTorch model state
        if hasattr(self.model, 'state_dict'):
            new_instance.model.load_state_dict(copy.deepcopy(self.model.state_dict()))
        
        new_instance.model.is_float   = self.model.is_float
        new_instance.model.dtype      = self.model.dtype
        new_instance.model.tomography = self.model.tomography
        new_instance.model.approx_noise_gain = self.model.approx_noise_gain
        
        # Copy wavelengths (already handled by config, but ensure reference is correct)
        new_instance.wavelengths = self.wavelengths.clone()
        
        # Deep copy the inputs manager using its built-in copy method
        if hasattr(self, 'inputs_manager'):
            new_instance.inputs_manager = self.inputs_manager.copy()

        # Deep copy the backup manager
        if hasattr(self, 'backup_manager'):
            new_instance.backup_manager = self.backup_manager.copy()
        
        # Copy spline control points if using splines
        if self.use_splines and hasattr(self, 'λ_ctrl'):
            new_instance.λ_ctrl = self.λ_ctrl.clone()
            if hasattr(self, 'norm_wvl'):
                new_instance.norm_wvl = copy.deepcopy(self.norm_wvl)
        
        # Copy LO basis if it exists
        if hasattr(self, 'LO_basis'):
            # The basis is already recreated in __init__, but we need to ensure
            # it has the same state if there are any learned parameters
            if hasattr(self.LO_basis, 'basis'):
                new_instance.LO_basis.basis = self.LO_basis.basis.clone()
            if hasattr(self.LO_basis, 'OPD_map'):
                new_instance.LO_basis.OPD_map = self.LO_basis.OPD_map.clone()
        
        return new_instance


    def _init_configs(self, config):
        if len(config) > 1: # Multiple sources
            if self.multiple_obs:
                model_config = MultipleTargetsInDifferentObservations(config, device=self.device)
            else:
                raise NotImplementedError("Multiple sources in one observation case is not implemented yet.")
        else:
            model_config = MultipleTargetsInDifferentObservations(config, device=self.device)


        return model_config


    def _init_model(self, config):
        pupil_angle = config['telescope']['PupilAngle']
        
        if hasattr(pupil_angle, '__len__'):
            if len(pupil_angle) > 1:
                if not torch.allclose(pupil_angle, pupil_angle[0]):
                    warn('Different pupil angles for different observations are not supported yet.')
            pupil_angle = pupil_angle[0]
            
            config['telescope']['PupilAngle'] = pupil_angle
       
        pupil = torch.tensor( PupilVLT(samples=320, rotation_angle=pupil_angle), device=self.device )
        PSD_include = {
            'fitting':         True,
            'WFS noise':       True,
            'spatio-temporal': True,
            'aliasing':        False,
            'chromatism':      True,
            'diff. refract':   True,
            'Moffat':          self.Moffat_absorber
        }
        
        self.model = TipTorch(
            AO_config = config,
            AO_type = 'LTAO',
            pupil = pupil,
            PSD_include = PSD_include,
            norm_regime = 'sum',
            device = self.device,
            oversampling = 1
        )


    def _init_NCPAs(self): 
        # LO_basis = PixelmapBasis(model, ignore_pupil=False)
        Z_basis = ZernikeBasis(self.model, N_modes=self.Z_mode_max, ignore_pupil=False)
        sausage_basis = MUSEPhaseBump(self.model, ignore_pupil=False)

        # LO NCPAs + phase bump are optimized jointly
        composite_basis = torch.concat([
            (sausage_basis.OPD_map).unsqueeze(0).flip(-2)*5e6*self.model.pupil.unsqueeze(0),
            Z_basis.basis[2:self.Z_mode_max,...]
        ], dim=0)

        self.LO_basis = ArbitraryBasis(self.model, composite_basis, ignore_pupil=False)
        self.LO_N_params = self.LO_basis.N_modes
        
        
    def _init_model_inputs(self):
        
        if self.multiple_obs:
            # This configuration assumes that no inputs are shared between different sources
            self.inputs_manager = InputsManager()
            
            def add_input(name, values, norm=Identity(), optimizable=True, is_shared=False):
                self.inputs_manager.add(name, values, norm, optimizable=optimizable)
            
        else:
            # Meanwhile, this configuration assumes that some inputs are shared between different
            # sources since they are all observed simultaneosuly
            self.inputs_manager = InputsManagersUnion({
                'shared':  InputsManager(),
                'per_src': InputsManager()
            })
         
            def add_input(name, values, norm, optimizable=True, is_shared=False):
                if is_shared:
                    self.inputs_manager.managers['shared'].add(name, values, norm, optimizable=optimizable)
                else:
                    self.inputs_manager.managers['per_src'].add(name, values, norm, optimizable=optimizable)

        N_wvl = len(self.wavelengths)
        N_src = self.model.N_src # number of simulated sources
        N_obs = self.model.N_obs # number of observations simulated (can be >1 only if multiple_obs=True)

        assert N_obs == 1 or self.multiple_obs, "Multiple observations >1 is only supported when multiple_obs=True"

        # Initialize normalizing transforms
        # The values are chosen in such a way so that on average, normalized parameters values
        # are distributed as Gauss(mu=0, std=1) 
        norm_F           = Uniform(a=0.0,   b=1.0)
        norm_bg          = Uniform(a=-5e-6, b=5e-6)
        norm_r0          = Uniform(a=0.08,  b=0.15)
        norm_L0          = Uniform(a=6,     b=34)
        norm_dxy         = Uniform(a=-1,    b=1)
        norm_J           = Uniform(a=0,     b=30)
        norm_Jxy         = Uniform(a=-180,  b=180)
        norm_dn          = Uniform(a=0,     b=5)
        norm_amp         = Uniform(a=0,     b=10)
        norm_b           = Uniform(a=0,     b=0.1)
        norm_alpha       = Uniform(a=-1,    b=10)
        norm_beta        = Uniform(a=0,     b=2)
        norm_ratio       = Uniform(a=0,     b=2)
        norm_theta       = Uniform(a=-np.pi/2, b=np.pi/2)
        norm_wind_speed  = Uniform(a=0, b=20)
        norm_LO          = Uniform(a=-20, b=50)
        norm_src_coords  = Uniform(None, -1.8e-5, 1.8e-5)
        norm_Cn2_profile = SoftmaxInv()
                
        # Add main PSF model parameters
        if self.use_splines:
            add_input('F_ctrl',  torch.tensor([[1.0,]*self.N_wvl_ctrl]*N_obs),  norm_F,   is_shared=True)  # Chromatic flux normalization correction shared by all PSFs
            add_input('dx_ctrl', torch.tensor([[0.0,]*self.N_wvl_ctrl]*N_src),  norm_dxy, is_shared=False) # Per-source precise chromatic astrometry correction
            add_input('dy_ctrl', torch.tensor([[0.0,]*self.N_wvl_ctrl]*N_src),  norm_dxy, is_shared=False) # -//-
            add_input('bg_ctrl', torch.tensor([[0.0,]*self.N_wvl_ctrl]*N_src),  norm_bg,  is_shared=False) # Per-source chromatic photomentry bias correction
        else:
            add_input('F',  torch.tensor([[1.0,]*N_wvl]*N_obs),  norm_F,   is_shared=True)
            add_input('dx', torch.tensor([[0.0,]*N_wvl]*N_src),  norm_dxy, is_shared=False)
            add_input('dy', torch.tensor([[0.0,]*N_wvl]*N_src),  norm_dxy, is_shared=False)
            add_input('bg', torch.tensor([[0.0,]*N_wvl]*N_src),  norm_bg,  is_shared=False)

        # Enabling chromatic defocus means that chromatic PSF bluriness is attributed to chromatic defocus aberration only.
        # Thus, jitter is considered monochromatic in this case.
        if self.chrom_defocus:
            add_input('J', torch.tensor([[25.0]]*N_obs), norm_J, is_shared=True) # If chromatic defocus is enabled, jitter is monochromatic
        else:
            if self.use_splines:
                add_input('J_ctrl', torch.tensor([[25.0,]*self.N_wvl_ctrl]*N_obs), norm_J, is_shared=True)
            else:
                add_input('J', torch.tensor([[25.0]*N_wvl]*N_obs), norm_J, is_shared=True)

        add_input('r0', self.model.r0.detach().clone(), norm_r0, is_shared=True)
        add_input('L0', self.model.L0.detach().clone(), norm_L0, is_shared=True)
        # add_input('Jxy', torch.tensor([[0.0]]*N_obs), norm_Jxy, optimizable=False, is_shared=True) # No fitter anisotropy
        add_input('dn',  torch.tensor([0.25]*N_obs),  norm_dn, is_shared=True) # HO WFSing error correction factor

        # Wind speed and direction are only accounted for the ground layer in this wrapper
        add_input('wind_speed_single', self.model.wind_speed[:,0].detach().clone().unsqueeze(-1), norm_wind_speed, is_shared=True)
        add_input('wind_dir_single', self.model.wind_dir[:,0].detach().clone().unsqueeze(-1), norm_wind_speed, optimizable=False, is_shared=True)
        add_input('Cn2_weights', self.model.Cn2_weights.detach().clone(), norm_Cn2_profile, optimizable=False, is_shared=True)
        
        # Auxiliary parameter to account for flux cropping due to finite PSF image size (computed on demand)
        add_input('flux_crop_ctrl', torch.tensor([[1.0,]*self.N_wvl_ctrl]*N_src), optimizable=False, is_shared=False)
        
        # Overall per-source flux scaling factor
        add_input('F_norm', torch.tensor([[1.0,]]*N_src), optimizable=(not self.multiple_obs), is_shared=False)
        # Sources direction within the field of view
        add_input('src_dirs_x', self.model.src_dirs_x.detach().clone(), norm=norm_src_coords, optimizable=False, is_shared=False)
        add_input('src_dirs_y', self.model.src_dirs_y.detach().clone(), norm=norm_src_coords, optimizable=False, is_shared=False)

        # Add Moffat PSD absorber's parameters
        if self.Moffat_absorber:
            add_input('amp',   torch.tensor([1e-4]*N_obs), norm_amp, is_shared=True)
            add_input('b',     torch.tensor([0.0]*N_obs), norm_b, is_shared=True)
            # add_input('alpha', torch.tensor([4.5]*N_obs), norm_alpha, is_shared=True)
            add_input('alpha', torch.tensor([2.0]*N_obs), norm_alpha, is_shared=True)
            add_input('beta',  torch.tensor([2.5]*N_obs), norm_beta, is_shared=True)
            add_input('ratio', torch.tensor([1.0]*N_obs), norm_ratio, is_shared=True)
            add_input('theta', torch.tensor([0.0]*N_obs), norm_theta, is_shared=True)

        if self.LO_NCPAs:
            if isinstance(self.LO_basis, PixelmapBasis):
                # add_input('LO_coefs', torch.zeros([self.model.N_obs, self.LO_N_params**2]), norm_LO, is_shared=True)
                # self.phase_func = lambda x: self.LO_basis(x.view(self.model.N_obs, self.LO_N_params, self.LO_N_params))
                # self.OPD_func   = lambda x: x.view(self.model.N_obs, self.LO_N_params, self.LO_N_params)
                raise NotImplementedError('Pixelmap LO basis is not properly tested yet yet.')


            elif isinstance(self.LO_basis, ZernikeBasis) or isinstance(self.LO_basis, ArbitraryBasis):
                add_input('LO_coefs', torch.zeros([self.model.N_obs, self.LO_N_params]), norm_LO, is_shared=True)
                self.OPD_func = lambda x: self.LO_basis.compute_OPD(x.view(self.model.N_obs, self.LO_N_params))

                if self.chrom_defocus:
                    if self.use_splines:
                        add_input('chrom_defocus_ctrl', torch.tensor([[0.0,]*self.N_wvl_ctrl]*self.model.N_obs), norm_LO, optimizable=self.chrom_defocus, is_shared=True)
                    else:
                        add_input('chrom_defocus', torch.tensor([[0.0,]*N_wvl]*self.model.N_obs), norm_LO, optimizable=self.chrom_defocus, is_shared=True)

                    def phase_func(x, y):
                        defocus_mode_id = 1 # the index of defocus mode given that 0 is the "phase bump" mode coefficient
                        coefs_chromatic = x.view(self.model.N_obs, self.LO_N_params).unsqueeze(1).repeat(1, N_wvl, 1) # Add a λ dimension
                        coefs_chromatic[:, :, defocus_mode_id] += y.view(self.model.N_obs, N_wvl) # add chromatic defocus
                        return self.LO_basis(coefs_chromatic)
                    
                    self.phase_func = phase_func
                else:
                    self.phase_func = lambda x, y: self.LO_basis(x.view(self.model.N_obs, self.LO_N_params))
            else:
                raise ValueError('Wrong LO type specified.')
        else:
            self.phase_func = None

        self.inputs_manager.to(self.device)
        self.inputs_manager.to_float()
        
        _ = self.inputs_manager.stack()
        self.backup_manager = self.inputs_manager.copy()
        
        if self.multiple_obs:
            self.per_src_inputs_list = [p for p in self.inputs_manager.parameters]
        else:
            self.per_src_inputs_list = [p for p in self.inputs_manager.managers['per_src'].parameters]


    def reset_parameters(self):
        ''' Reset model input parameters to their initial values'''
        self.inputs_manager = self.backup_manager.copy()
    

    def evaluate_splines(self, y_points, λ_grid):
        spline = NaturalCubicSpline(natural_cubic_spline_coeffs(t=self.λ_ctrl, x=y_points.T))
        return spline.evaluate(λ_grid).T


    def forward(self, x_dict=None, src_ids=None, include_list=None):
        # TODO: add logic to limit the maximum number of simulated sources
        # TODO: pad individual inputs when simulating less sources than model.N_src
        
        if x_dict is None:
            x_dict = self.inputs_manager.to_dict()
    
        if src_ids is not None:
            if self.multiple_obs:
                raise warn("Source selection is not implemented yet for multiple observations case. All sources will be simulated.")
            else:
                for key in self.per_src_inputs_list:
                    x_dict[key] = x_dict[key][src_ids, :]
    
        if self.use_splines:
            for entry in self.polychromatic_params:
                if entry+'_ctrl' in x_dict:
                    x_dict[entry] = self.evaluate_splines(x_dict[entry+'_ctrl'], self.norm_wvl(self.wavelengths))

        # Clone J entry to Jx and Jy
        x_dict['Jx'] = x_dict['J']
        x_dict['Jy'] = x_dict['J']
        
        if 'wind_speed_single' in x_dict:
            x_dict['wind_speed'] = torch.nn.functional.pad(x_dict['wind_speed_single'].view(-1, 1), (0, self.model.N_L - 1))
        
        if 'wind_dir_single' in x_dict:
            x_dict['wind_dir'] = torch.nn.functional.pad(x_dict['wind_dir_single'].view(-1, 1), (0, self.model.N_L - 1))

        x_ = { key: x_dict[key] for key in include_list } if include_list is not None else x_dict
        
        chrom_defocus = x_dict['chrom_defocus'] if self.chrom_defocus else None

        phase_ = (lambda: self.phase_func(x_dict['LO_coefs'], chrom_defocus)) if self.LO_NCPAs else None

        return self.model(x_, None, phase_generator=phase_)


    def forward_full_spectrum(self, x_dict=None, src_ids=None, include_list=None):
        raise NotImplementedError("This function is not fully tested yet.")
        return
    
        if not self.use_splines:
            raise ValueError("Full spectrum evaluation is only available when using the spline representation of the polychromatic parameters.")
        
        # Split λ array into batches
        max_λ_batch_size = 100
        λ_batches = [self.λ_full[i:i + max_λ_batch_size] for i in range(0, len(self.λ_full), max_λ_batch_size)]

        _initial_wvl = self.wavelengths.clone()

        with torch.no_grad():
            PSFs_all_sources = torch.zeros((self.model.N_src, self.num_λ_slices, self.model.N_pix, self.model.N_pix), device='cpu') # Force CPU to avoid memory overuse
            
            for i, λ_batch in enumerate(λ_batches):
                current_batch_size = len(λ_batch)
                λ_ids = slice(i*current_batch_size, (i+1)*current_batch_size)
                # Update simulated wavelengths
                self.SetWavelengths( torch.tensor(λ_batch, device=self.device) )
                # Need to simulate per-source due to memory limitations when simulating large wavelengths batches
                for src_id in src_ids if src_ids is not None else range(self.model.N_src):
                    # Evaluate PSF for the current λ batch
                    PSF_batch = self.forward(x_dict, src_ids=[src_id], include_list=include_list) #TODO (!): implement src_id in forward()
                    PSFs_all_sources[src_id, λ_ids, ...] = PSF_batch # Keep only the current source

        self.SetWavelengths(_initial_wvl)
        
        return PSFs_all_sources
    

    def SetWavelengths(self, wavelengths):
        self.model.SetWavelengths(wavelengths)
        self.wavelengths = wavelengths


    def SetImageSize(self, img_size):
        self.model.SetImageSize(img_size)
    
    
    def EvaluateFluxCropFactor(self):
        ''' Computes how much flux is cropped by the finite PSF image size in comparison to a quasi-infinite PSF'''
        quasi_inf_PSF_size = 511
        
        with torch.no_grad():
            if self.use_splines:
                wvl_current = self.wavelengths.clone()
                self.SetWavelengths(self.λ_ctrl_denorm)

            # The actual size of the simulated PSFs
            current_PSF_size = self.model.N_pix
            PSF_small = self.forward()
            # Quasi-infinite PSF image to compute how much flux is lost while cropping
            self.SetImageSize(quasi_inf_PSF_size)
            PSF_big = self.forward()
            self.SetImageSize(current_PSF_size)
            
            if self.use_splines:
                self.SetWavelengths(wvl_current)
                
        flux_correction = (PSF_big.amax(dim=(-2,-1)) / PSF_small.amax(dim=(-2,-1)))

        self.inputs_manager['flux_crop_ctrl'] = flux_correction.detach().clone()

        return self.evaluate_splines(self.inputs_manager['flux_crop_ctrl'], self.norm_wvl(self.wavelengths))            


    __call__ = forward

