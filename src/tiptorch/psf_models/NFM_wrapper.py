import torch
import numpy as np
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline

from tiptorch.psf_models.TipTorch import TipTorch
from tiptorch.tools.static_phase import ArbitraryBasis, PixelmapBasis, ZernikeBasis, MUSEPhaseBump
from tiptorch.tools.utils import PupilVLT
from tiptorch.managers.config_manager import MultipleTargetsInDifferentObservations
from tiptorch.managers.input_manager  import InputsManager, InputsManagersUnion
from tiptorch.tools.normalizers import DataTransform, Uniform, Uniform0_1, SoftmaxInv, Identity, SafeLog10

from warnings import warn
from tiptorch._config import default_device
import gc


class PSFModelNFM:
    def __init__(
        self,
        config,
        multiple_obs    = False,
        LO_NCPAs        = True,
        chrom_defocus   = False,
        Moffat_absorber = False,
        Z_mode_max      = 9,
        N_spline_nodes  = 5,
        device          = default_device
    ):
        self.Z_mode_max  = Z_mode_max
        self.LO_N_params = None
        
        self.LO_NCPAs        = LO_NCPAs
        self.device          = device
        self.Moffat_absorber = Moffat_absorber
        self.use_splines     = not (N_spline_nodes is None)
        self.chrom_defocus   = chrom_defocus and LO_NCPAs
        self.__config_raw    = config # Input config dictionary, defined in TipTop-style
                
        if N_spline_nodes:
            if N_spline_nodes < 2 or N_spline_nodes > 3681:
                raise ValueError(f"Number of control wavelengths for spline representation must be between 2 and 3681. Got {N_spline_nodes}.")
                
        # The PSF model can be used in two configurations:
        #  1) Simulating different sources in multiple observations with different conditions -- used mostly in ML training and PSF model calibration
        #  2) Simulating multiple sources in the same observation -- the practical use-case when multiple sources are simulated in one science exposure.
        #                                                            In this case, atmospheric conditions and AO correction is shared by all sources
        self.multiple_obs    = multiple_obs
        
        _config = self._init_configs(self.__config_raw) # Processed and converted config
        self.λ_sim = _config['sources_science']['Wavelength'].squeeze()

        # Full spectrum span of MUSE NFM
        self.λ_min = 475.e-9
        self.λ_max = 935.e-9
        self.Δλ = 0.125e-9
        self.num_λ_slices = 3681
        self.λ_full = torch.linspace(self.λ_min, self.λ_max, self.num_λ_slices, device=self.device, dtype=self.λ_sim.dtype)
        
        # Initialize the TipTorch model
        self._init_model(_config)
        # Initialize LO NCPAs
        if LO_NCPAs:
            self._init_NCPAs()
        
        self.polychromatic_params = ['F', 'dx', 'dy', 'bg', 'F_norm_λ'] + (['chrom_defocus'] if self.chrom_defocus else ['J'])

        if self.use_splines:
            # If splines are used, the chromatic parameters are represented by their values at a few control wavelengths and interpolated at the simulated
            # wavelengths using natural cubic splines. If so, all parameters must be defined only at the control wavelengths. Unlike objects spectra,
            # the chromatic behavior of the PSF is generally smooth and doesn't have high-frequency features, so a small number of control λs is fine.
            # The same control wavelengths must be used for all chromatic parameters to ensure consistency.
            self.N_wvl_ctrl = N_spline_nodes
            self.norm_wvl = Uniform0_1(a=self.λ_min, b=self.λ_max) # [nm], MUSE NFM wavelength range
            self.λ_ctrl_norm   = torch.linspace(0, 1, self.N_wvl_ctrl, device=self.device) # control λ nodes normalized to [0...1]
            self.λ_ctrl        = self.norm_wvl.inverse(self.λ_ctrl_norm) # [nm], control λs re-normalized back to the physical range
            self.λ_sim_normed  = self.norm_wvl(self.λ_sim) # [0...1], simulated wavelengths normalized to [0...1] range for spline evaluation
            self.λ_full_normed = self.norm_wvl(self.λ_full) # normalized full spectrum wavelengths for spline evaluation (if splines are used)


        self._init_model_inputs()
        
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            
        if self.device.type == 'mps':
            torch.mps.empty_cache()
    
    
    def cleanup(self):
        """Explicitly clean up GPU memory"""       
        # Clean up model (which will trigger TipTorch cleanup)
        del self.inputs_manager, self.λ_sim
        
        if self.LO_NCPAs:
            del self.LO_basis
            
        if self.use_splines:
            del self.λ_ctrl_norm,  self.λ_ctrl, self.norm_wvl, self.λ_sim_normed, self.λ_full_normed

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
            pass # Avoid errors during interpreter shutdown
    
 
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
            N_spline_nodes  = self.N_wvl_ctrl,
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
        
        # Copy wavelengths (handled by config as well)
        new_instance.λ_sim = self.λ_sim.clone()
        
        # Deep copy the inputs manager using its built-in copy method
        if hasattr(self, 'inputs_manager'):
            new_instance.inputs_manager = self.inputs_manager.copy()

        # Deep copy the backup manager
        if hasattr(self, 'backup_manager'):
            new_instance.backup_manager = self.backup_manager.copy()
        
        # Copy spline control points if using splines
        if self.use_splines and hasattr(self, 'λ_ctrl'):
            new_instance.λ_ctrl_norm = self.λ_ctrl_norm.clone()
            if hasattr(self, 'norm_wvl'):
                new_instance.norm_wvl = copy.deepcopy(self.norm_wvl)
        
        # Copy LO basis for NCPAs (if it exists)
        if hasattr(self, 'LO_basis'):
            # The basis is already recreated in __init__, but we need to ensure
            # it has the same state if there are any learned parameters
            if hasattr(self.LO_basis, 'basis'):
                new_instance.LO_basis.basis = self.LO_basis.basis.clone()
            
            if hasattr(self.LO_basis, 'OPD_map'):
                raise NotImplementedError("LO basis with PixelmapBasis is not supported yet.")
                # new_instance.LO_basis.OPD_map = self.LO_basis.OPD_map.clone()
        
        return new_instance


    def _init_configs(self, config):
        if type(config) is dict and 'NumberSources' in config:
            return config # Already pre-processed config, no need to merge or convert
        
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
        
        if hasattr(pupil_angle, '__len__') and pupil_angle.ndim > 0:
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
        # We need a composite basis because of the presence of phase bump mode, which is not orthogonal to the Zernike modes
        self.LO_basis = ArbitraryBasis(self.model, composite_basis, ignore_pupil=False)
        self.LO_N_params = self.LO_basis.N_modes
        
    
    def update_manager_params(self, x_dict: dict, src_ids: bool | None = None) -> None:
        if self.multiple_obs:
            if src_ids is not None:
                raise warn("Source selection is not implemented yet for multiple observations case. All sources will be updated.")
            
            self.inputs_manager.update(x_dict)
        else:
            # Update only the values related to the simulated sources if src_ids is specified
            self.inputs_manager.input_managers['per_src'].update(x_dict, selected_ids=src_ids)
            # Shared parameters are updated for all sources regardless of src_ids selection
            self.inputs_manager.input_managers['shared'].update(x_dict)
    
    
    def _init_model_inputs(self):
        if self.multiple_obs:
            # This configuration assumes that no inputs are shared between different sources
            self.inputs_manager = InputsManager()
            
            def add_input(name: str, values: torch.Tensor, norm: DataTransform=Identity(), optimizable: bool=True, is_shared: bool=False):
                # is_shared flag is ignored in this configuration since all inputs are per-source anyway
                self.inputs_manager.add(name, values, norm, optimizable=optimizable)   
        else:
            # Meanwhile, this configuration assumes that most of the model inputs are shared between different sources # since they are all share
            # the same atmospheric conditions, and only a few of them are source-specific (e.g. astrometry and photometry corrections)
            self.inputs_manager = InputsManagersUnion({
                'shared':  InputsManager(),
                'per_src': InputsManager()
            })
         
            def add_input(name: str, values: torch.Tensor, norm: DataTransform=Identity(), optimizable: bool=True, is_shared: bool=False):
                if is_shared:
                    self.inputs_manager.input_managers['shared'].add(name, values, norm, optimizable=optimizable)
                else:
                    self.inputs_manager.input_managers['per_src'].add(name, values, norm, optimizable=optimizable)

        N_wvl = len(self.λ_sim)
        N_src = self.model.N_src # number of simulated sources
        N_obs = self.model.N_obs # number of observations simulated (can be >1 only if multiple_obs=True)

        assert N_obs == 1 or self.multiple_obs, "Multiple observations >1 is only supported when multiple_obs=True"

        # Initialize normalizing transforms
        # Normalized parameters values are (approximately) distributed in the range between -1 and 1.
        # It helps with the convergence by avoiding extreme parameter values while PSF fitting or NNs training.
        # Normalization is ONLY applied when parameters are stacked in the single vector
        norm_F           = Uniform(a=0.0,   b=1.0)
        norm_bg          = Uniform(a=-5e-6, b=5e-6)
        norm_r0          = Uniform(a=0.08,  b=0.15)
        norm_L0          = Uniform(a=6,     b=34)
        norm_dxy         = Uniform(a=-1,    b=1)
        norm_J           = Uniform(a=0,     b=30)
        norm_dn          = Uniform(a=0,     b=5)
        norm_amp         = Uniform(a=0,     b=10)
        norm_b           = Uniform(a=0,     b=0.1)
        norm_alpha       = Uniform(a=-1,    b=10)
        norm_beta        = Uniform(a=0,     b=2)
        norm_ratio       = Uniform(a=0,     b=2)
        norm_theta       = Uniform(a=-np.pi/2, b=np.pi/2)
        norm_wind_speed  = Uniform(a=0, b=20)
        norm_F_λ_norm    = Uniform(a=0.5, b=1.0)
        norm_LO          = Uniform(a=-20, b=50)
        norm_src_coords  = Uniform(None, -1.8e-5, 1.8e-5)
        norm_Cn2_profile = SoftmaxInv()
        norm_F_norm      = SafeLog10()
        
        # Add main PSF model parameters
        if self.use_splines:
            # Chromatic flux normalization correction shared by all PSFs
            add_input('F_ctrl',  torch.tensor([[1.0,]*self.N_wvl_ctrl]*N_obs),  norm_F,   is_shared=True)
            # Shared chromatic photometry bias correction
            add_input('bg_ctrl', torch.tensor([[0.0,]*self.N_wvl_ctrl]*N_obs),  norm_bg,  is_shared=True)
            # Precise per-source chromatic astrometry correction
            add_input('dx_ctrl', torch.tensor([[0.0,]*self.N_wvl_ctrl]*N_src),  norm_dxy, is_shared=False)
            add_input('dy_ctrl', torch.tensor([[0.0,]*self.N_wvl_ctrl]*N_src),  norm_dxy, is_shared=False) 
            # An auxiliary chromatic normalization factor. It's different from F and used to store chromatic flux crop factor due to finite PSF size and
            # chromatic core vs. wings flux ratio, Meanwhile F, is used to normalize the overall flux level of the PSF across wavelengths due to its morphology
            add_input('F_norm_λ_ctrl', torch.tensor([[1.0,]*self.N_wvl_ctrl]*N_obs), norm=norm_F_λ_norm, optimizable=False, is_shared=True)  
        else:
            add_input('F',  torch.tensor([[1.0,]*N_wvl]*N_obs),  norm_F,   is_shared=True)
            add_input('bg', torch.tensor([[0.0,]*N_wvl]*N_obs),  norm_bg,  is_shared=True)
            add_input('dx', torch.tensor([[0.0,]*N_wvl]*N_src),  norm_dxy, is_shared=False)
            add_input('dy', torch.tensor([[0.0,]*N_wvl]*N_src),  norm_dxy, is_shared=False)
            add_input('F_norm_λ', torch.tensor([[1.0,]*self.N_wvl_ctrl]*N_obs), norm=norm_F_λ_norm, optimizable=False, is_shared=True)

        # Enabling chromatic defocus means that chromatic PSF bluriness is attributed to chromatic defocus aberration only.
        # Thus, TT jitter is considered monochromatic in this case. Otherwise, this bluriness is "absorbed" by the chromatic jitter parameters Jx and Jy
        if self.chrom_defocus:
            add_input('J', torch.tensor([[25.0]]*N_obs), norm_J, is_shared=True) # If chromatic defocus is enabled, jitter is monochromatic
        else:
            if self.use_splines:
                add_input('J_ctrl', torch.tensor([[25.0,]*self.N_wvl_ctrl]*N_obs), norm_J, is_shared=True)
            else:
                add_input('J', torch.tensor([[25.0]*N_wvl]*N_obs), norm_J, is_shared=True)

        add_input('Jxy', torch.tensor([0.0]), norm_J, optimizable=False, is_shared=True) # essentially, this disables the TT jitter anisotropy

        add_input('r0', self.model.r0.detach().clone(), norm_r0, is_shared=True)
        add_input('L0', self.model.L0.detach().clone(), norm_L0, is_shared=True)
        add_input('dn',  torch.tensor([0.25]*N_obs),    norm_dn, is_shared=True) # HO WFSing error correction factor

        # Wind speed and direction are only accounted for the ground layer
        add_input('wind_speed_single', self.model.wind_speed[:,0].detach().clone().unsqueeze(-1), norm_wind_speed, is_shared=True)
        add_input('wind_dir_single', self.model.wind_dir[:,0].detach().clone().unsqueeze(-1), norm_wind_speed, optimizable=False, is_shared=True)
        add_input('Cn2_weights', self.model.Cn2_weights.detach().clone(), norm_Cn2_profile, optimizable=False, is_shared=True)
        
        # Overall per-source flux scaling factor, can be used per-source for photometry fine tuning
        add_input('F_norm', torch.tensor([[1.0,]]*N_src), norm=norm_F_norm, optimizable=(not self.multiple_obs), is_shared=False)
        
        # Sources directions within the FoV
        add_input('src_dirs_x', self.model.src_dirs_x.detach().clone(), norm=norm_src_coords, optimizable=False, is_shared=False)
        add_input('src_dirs_y', self.model.src_dirs_y.detach().clone(), norm=norm_src_coords, optimizable=False, is_shared=False)

        # Add Moffat PSD absorber's parameters (if one is enabled)
        if self.Moffat_absorber:
            add_input('amp',   torch.tensor([1e-4]*N_obs), norm_amp,   is_shared=True)
            add_input('b',     torch.tensor([0.0]*N_obs),  norm_b,     is_shared=True)
            add_input('alpha', torch.tensor([2.0]*N_obs),  norm_alpha, is_shared=True)
            add_input('beta',  torch.tensor([2.5]*N_obs),  norm_beta,  is_shared=True)
            add_input('ratio', torch.tensor([1.0]*N_obs),  norm_ratio, is_shared=True)
            add_input('theta', torch.tensor([0.0]*N_obs),  norm_theta, is_shared=True)

        if self.LO_NCPAs:
            if isinstance(self.LO_basis, PixelmapBasis):
                # add_input('LO_coefs', torch.zeros([self.model.N_obs, self.LO_N_params**2]), norm_LO, is_shared=True)
                # self.phase_func = lambda x: self.LO_basis(x.view(self.model.N_obs, self.LO_N_params, self.LO_N_params))
                # self.OPD_func   = lambda x: x.view(self.model.N_obs, self.LO_N_params, self.LO_N_params)
                raise NotImplementedError('Pixelmap LO basis is not properly tested yet yet.')

            elif isinstance(self.LO_basis, ZernikeBasis) or isinstance(self.LO_basis, ArbitraryBasis):
                # The phase generation function is defined differently based on the different settings
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
            self.phase_func = None # No LO NCPAs simualted, so no need to generate phase maps

        self.inputs_manager.to(self.device)
        self.inputs_manager.to_float()
        
        _ = self.inputs_manager.stack()
        self.backup_manager = self.inputs_manager.copy()
        
        if self.multiple_obs:
            self.per_src_inputs_list = [p for p in self.inputs_manager.parameters]
        else:
            self.per_src_inputs_list = [p for p in self.inputs_manager.input_managers['per_src'].parameters]


    def reset_parameters(self):
        ''' Reset model input parameters to their initial values'''
        self.inputs_manager = self.backup_manager.copy()
    
    
    def get_optimizable_param_names(self):
        return self.inputs_manager.get_names(optimizable_only=True, flattened=False)


    def get_param_names(self):
        return self.inputs_manager.get_names(optimizable_only=False, flattened=False)


    def get_fixed_param_names(self):
        return list(set(self.get_param_names()) - set(self.get_optimizable_param_names()))


    def evaluate_splines(self, y_points, λ_grid):
        if y_points.shape[-1] != self.N_wvl_ctrl:
            raise ValueError(f"Number of control λs must match the last dimension of the input parameter values. Expected {self.N_wvl_ctrl}, got {y_points.shape[-1]}")
        
        if y_points.ndim <= 1:
            y_points = y_points.unsqueeze(0) # Ensure there's a batch dimension even when only one source is simulated
        
        spline = NaturalCubicSpline(natural_cubic_spline_coeffs(t=self.λ_ctrl_norm, x=y_points.T))
        return spline.evaluate(λ_grid).T


    def forward(self, x_dict=None, src_ids=None, include_list=None, update_params=True):
        # TODO: add logic to limit the maximum number of simulated sources
        # TODO: pad individual inputs when simulating less sources than model.N_src
        
        need_update = True and update_params
        
        if x_dict is None:
            x_dict = self.inputs_manager.to_dict()
            need_update = False
    
        # Select a subset of sources to simulate
        if src_ids is not None:
            if self.multiple_obs:
                raise warn("Source selection is not implemented yet for multiple observations case. All sources will be simulated.")
                #TODO: why isn't it possible? -> because then, the entire TipTorch's config must be changed. Doable but meh, maybe later
            else:
                for key in self.per_src_inputs_list:
                    x_dict[key] = x_dict[key][src_ids]
                    if x_dict[key].dim() <= 1: # When only one src is simulated, keep the batch dimension for consistency
                        x_dict[key] = x_dict[key].unsqueeze(0)

        # Evaluate all polychromatic quantities at the simulated wavelengths using the spline representation
        if self.use_splines:
            for entry in self.polychromatic_params:
                if entry+'_ctrl' in x_dict:
                    x_dict[entry] = self.evaluate_splines(x_dict[entry+'_ctrl'], self.λ_sim_normed)

        # Clone J entry to Jx and Jy
        x_dict['Jx'] = x_dict['J']
        x_dict['Jy'] = x_dict['J']
        
        # Create a per-layer wind profile assuming there is no wind outside the ground layer
        if 'wind_speed_single' in x_dict:
            x_dict['wind_speed'] = torch.nn.functional.pad(x_dict['wind_speed_single'].view(-1, 1), (0, self.model.N_L - 1))
            
        if 'wind_dir_single' in x_dict:
            x_dict['wind_dir'] = torch.nn.functional.pad(x_dict['wind_dir_single'].view(-1, 1), (0, self.model.N_L - 1))

        x_ = { key: x_dict[key] for key in include_list } if include_list is not None else x_dict
        
        chrom_defocus = x_dict['chrom_defocus'] if self.chrom_defocus else None

        phase_ = self.phase_func(x_dict['LO_coefs'], chrom_defocus) if self.LO_NCPAs else None

        if need_update:
            # Update inputs manager based of x_dict inputs
            self.update_manager_params(x_dict, src_ids=src_ids)
        
        return self.model(x_, None, phase=phase_)


    __call__ = forward


    def __getitem__(self, item):
        return self.inputs_manager[item]
    
    
    def __setitem__(self, key, value):
        self.inputs_manager[key] = value
    
    
    def SetWavelengths(self, wavelengths):
        # Cheap pointer check first (same tensor object → same values, no GPU sync needed)
        if self.λ_sim.data_ptr() == wavelengths.data_ptr():
            return
        # Fallback value check — forces a CUDA sync, but only reached for distinct tensor objects
        if self.λ_sim.shape == wavelengths.shape and \
            torch.allclose(wavelengths.flatten(), self.λ_sim.flatten(), atol=1e-12):
            return

        self.model.SetWavelengths(wavelengths)
        self.λ_sim = wavelengths
        if self.use_splines:
            self.λ_sim_normed = self.norm_wvl(self.λ_sim) # [0...1], simulated wavelengths normalized to [0...1] range for spline evaluation


    def SetImageSize(self, img_size):
        # Compare values to avoid unnecessary updates
        if self.model.N_pix == img_size:
            return
        self.model.SetImageSize(img_size)
    
    
    @torch.no_grad()
    def SimulateFullSpectrum(self, verbose=False):
        ''' The function to simulate PSFs across the full MUSE NFM spectral range at once. It simulates the PSF chromatic cube for each source
            sequentially parallelizing over wavelengths batches to avoid GPU memory overflow. '''

        # ----- Manage model inputs
        x_dict = self.inputs_manager.to_dict()
        x_dict_λ_full = {}

        # Evaluate all spline parameters at the full λs range beforehand to avoid redundant computations during the PSF simulation loop
        for entry in self.polychromatic_params:
            if entry + '_ctrl' in x_dict:
                x_dict_λ_full[entry] = self.evaluate_splines(x_dict[entry + '_ctrl'], self.λ_full_normed)
                x_dict[entry + '_ctrl']
                x_dict.pop(entry + '_ctrl') # Remove the control λs from x_dict to avoid confusion, since we now have the full λs values for these parameters

        x_dict_λ_full['Jx'] = x_dict_λ_full['J']
        x_dict_λ_full['Jy'] = x_dict_λ_full['J']
        x_dict_λ_full.pop('J') # Remove the joint jitter parameter

        # Extend to all atmospheric layers assuming the wind is only in ground layer
        if 'wind_speed_single' in x_dict:
            x_dict['wind_speed'] = torch.nn.functional.pad(x_dict['wind_speed_single'].view(-1, 1), (0, self.model.N_L - 1))
            x_dict.pop('wind_speed_single')

        if 'wind_dir_single' in x_dict:
            x_dict['wind_dir'] = torch.nn.functional.pad(x_dict['wind_dir_single'].view(-1, 1), (0, self.model.N_L - 1))
            x_dict.pop('wind_dir_single')

        chrom_defocus = x_dict['chrom_defocus'] if self.chrom_defocus else None
        phase_ = self.phase_func(x_dict['LO_coefs'], chrom_defocus) if self.LO_NCPAs else None

        # The final dictionary with all necessary values stored inside
        x_dict.update(x_dict_λ_full)

        per_src_params = [p.replace('_ctrl', '') for p in self.inputs_manager.input_managers['per_src'].to_dict().keys()]
        polychromatic_params = list(x_dict_λ_full.keys())

        # ----- Manage wavelengths batches and PSF simulation
        # Full spectral cubes for each source
        PSFs_combined = torch.zeros((self.model.N_src, self.num_λ_slices, self.model.N_pix, self.model.N_pix), device='cpu')

        max_λ_batch_size = 100 # TODO: adjust based on memory available and the number of sources simulated
        λ_batches = [torch.tensor(self.λ_full[i:i + max_λ_batch_size], device=self.device) for i in range(0, len(self.λ_full), max_λ_batch_size)]

        _initial_wvl = self.λ_sim.clone()
        _N_src = self.model.N_src

        self.model.N_src = 1 # Temporarily set to 1 to compute PSFs for each source separately

        if verbose:
            from tqdm import tqdm
            pbar = tqdm(total=self.num_λ_slices, desc='Simulating PSF across wavelengths')

        # Copy all shared non-chromatic entries, they stay the same for every λ and every source
        x_ = { key: x_dict[key] for key in x_dict.keys() if (key not in per_src_params) and (key not in polychromatic_params) }
        
        for i, λ_batch in enumerate(λ_batches):
            if verbose:
                pbar.update(len(λ_batch))
            
            current_batch_size = len(λ_batch)
            start = i * max_λ_batch_size
            λ_ids = slice(start, start + current_batch_size)

            self.model.SetWavelengths(λ_batch)
            
            # Select current λ batch for all chromatic parameters
            for entry in polychromatic_params:
                x_[entry] = x_dict[entry][..., λ_ids]
            
            # Compute the static OTF once per λ batch - it's source-independent, assuming it's constant over the field
            if self.LO_NCPAs:
                chrom_defocus_batch = x_dict['chrom_defocus'][..., λ_ids] if self.chrom_defocus else None
                self.model.ComputeStaticOTF(self.phase_func(x_dict['LO_coefs'], chrom_defocus_batch))

            # Iterate over all sources
            for src_id in range(_N_src):
                # Select current object simulated
                for entry in per_src_params:
                    if entry in polychromatic_params:
                        # Both per-source AND chromatic (e.g. dx, dy): select source AND λ batch
                        x_[entry] = x_dict[entry][src_id, λ_ids].unsqueeze(0)
                    else:
                        x_[entry] = x_dict[entry][src_id, ...].unsqueeze(0)

                PSFs_combined[src_id, λ_ids, ...] = self.model(x_, None, None).cpu()

        self.model.N_src = _N_src  # Restore the original number of sources
        self.SetWavelengths(_initial_wvl)  # recomputes grids + tomographic projectors for correct N_src
        # Restore OTF_static for the original wavelengths and LO coefficients
        if self.LO_NCPAs:
            self.model.ComputeStaticOTF(phase_)

        torch.cuda.empty_cache()
        
        return PSFs_combined # [N_src, N_λ_full, N_pix, N_pix]


    @torch.no_grad()
    def ComputeStrehl(self):
        ''' Computes Strehl ratio (SR) across the simulated wavelengths for the on-axis PSF '''
        # backup the original coordinates
        coords_backup = ( self.inputs_manager['src_dirs_x'].clone(), self.inputs_manager['src_dirs_y'].clone() )
        # Assume on-axis PSF for computing the ratio
        self.inputs_manager['src_dirs_x'] *= 0.0
        self.inputs_manager['src_dirs_y'] *= 0.0
        # compute only for the first source ignoring the field variability (just for speed's sake)  
        PSFs_pred = self.forward(src_ids=0)
        # Restore the original coordinates
        self.inputs_manager['src_dirs_x'], self.inputs_manager['src_dirs_y'] = coords_backup

        PSF_DL = self.model.DLPSF() # compute diffraction limited PSF
        
        Strehl_vs_λ = PSFs_pred[0,...].amax(dim=(-2,-1)) / PSF_DL[0,...].amax(dim=(-2,-1))
        return Strehl_vs_λ
