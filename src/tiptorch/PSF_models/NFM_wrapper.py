import torch
import numpy as np
from tiptorch.tools.cubic_splines import natural_cubic_spline_coeffs, NaturalCubicSpline

from tiptorch.PSF_models.TipTorch import TipTorch
from tiptorch.tools.static_phase import ArbitraryBasis, PixelmapBasis, ZernikeBasis, MUSEPhaseBump
from tiptorch.tools.utils import PupilVLT
from tiptorch.managers.config_manager import MultipleTargetsInDifferentObservations
from tiptorch.managers.input_manager  import InputsManager, InputsManagersUnion
from tiptorch.tools.normalizers import DataTransform, Uniform, Uniform0_1, SoftmaxInv, Identity, SafeLog10

from warnings import warn
from tiptorch._config import default_device
import gc


class PSFModelNFM:
    # Valid PSD regime identifiers
    _VALID_REGIMES = ('physics-based', 'psfao', 'hybrid')

    def __init__(
        self,
        config,
        multiple_obs   = False,
        LO_NCPAs       = True,
        chrom_defocus  = False,
        use_Moffat     = False,   # Deprecated: use model_type='hybrid' instead
        retain_PSDs    = False,
        Z_mode_max     = 9,
        N_spline_nodes = 5,
        device         = default_device,
        dtype          = torch.float32,
        model_type     = None,    # 'physics-based' | 'psfao' | 'hybrid'
        *, # MUSE NFM spectral data
        λ_min        = 475.e-9,
        λ_max        = 935.e-9,
        num_λ_slices = 3681,
    ):
        # Resolve model_type: explicit argument wins; fall back to use_Moffat for backward compat
        if model_type is None:
            model_type = 'hybrid' if use_Moffat else 'physics-based'
        if model_type not in self._VALID_REGIMES:
            raise ValueError(f"model_type must be one of {self._VALID_REGIMES}, got '{model_type}'")

        self.Z_mode_max    = Z_mode_max
        self.LO_N_params   = None
        self.retain_PSDs   = retain_PSDs
        self.LO_NCPAs      = LO_NCPAs
        self.model_type    = model_type
        self.use_Moffat    = model_type in ('hybrid',)   # kept for backward compat
        self.chrom_defocus = chrom_defocus and LO_NCPAs
        self.__config_raw  = config # Input config dictionary, defined in TipTop-style
        self.device        = torch.device(device)
        self.dtype         = dtype

        if N_spline_nodes:
            if N_spline_nodes < 2 or N_spline_nodes > num_λ_slices:
                raise ValueError(f"Number of control wavelengths for spline representation must be between 2 and {num_λ_slices}. Got {N_spline_nodes}.")
            self.use_splines = True # enable the use of splines for chromatic parameters representation
                
        # The PSF model can be used in two configurations:
        #  1) Simulating different sources in multiple observations with different conditions -- used mostly in ML training and PSF model calibration
        #  2) Simulating multiple sources in the same observation -- the practical use-case when multiple sources are simulated in one science exposure.
        #                                                            In this case, atmospheric conditions and AO correction is shared by all sources
        self.multiple_obs = multiple_obs
        
        _config = self._init_configs(self.__config_raw) # Processed and converted config
        self.λ_sim = _config['sources_science']['Wavelength'].squeeze()

        # Full spectrum span of MUSE NFM
        self.λ_min = λ_min
        self.λ_max = λ_max
        self.Δλ    = (self.λ_max - self.λ_min) / (num_λ_slices - 1)
        self.num_λ_slices = num_λ_slices
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
            self.N_wvl_ctrl    = N_spline_nodes
            self.norm_wvl      = Uniform0_1(a=self.λ_min, b=self.λ_max) # [nm], MUSE NFM wavelength range
            self.λ_ctrl_norm   = torch.linspace(0, 1, self.N_wvl_ctrl, device=self.device) # control λ nodes normalized to [0...1]
            self.λ_ctrl        = self.norm_wvl.inverse(self.λ_ctrl_norm) # [nm], control λs re-normalized back to the physical range
            self.λ_sim_normed  = self.norm_wvl(self.λ_sim) # [0...1], simulated wavelengths normalized to [0...1] range for spline evaluation
            self.λ_full_normed = self.norm_wvl(self.λ_full) # normalized full spectrum wavelengths for spline evaluation (if splines are used)

        self._init_model_inputs()
        
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            
        if self.device.type == 'mps':
            torch.mps.empty_cache()
    
    
    @staticmethod
    def _tree_to_cpu(obj):
        """Recursively detach tensors and move them to CPU for portable serialization."""
        if torch.is_tensor(obj):
            return obj.detach().cpu().clone()
        
        if isinstance(obj, dict):
            return {k: PSFModelNFM._tree_to_cpu(v) for k, v in obj.items()}
        
        if isinstance(obj, (list, tuple)):
            return type(obj)(PSFModelNFM._tree_to_cpu(v) for v in obj)
        
        return obj


    def save(self, *, cpu: bool = True):
        """
        Serialize the wrapper state.

        The returned object is intentionally a plain Python dictionary suitable for
        torch.save(...).  It stores both:
          1. constructor arguments needed to rebuild the wrapper; and
          2. mutable state that may have changed after construction.
        """
        dtype_name = str(self.model.dtype).replace('torch.', '') if hasattr(self.model, 'dtype') else str(self.dtype).replace('torch.', '')

        store_data = {
            # Constructor state.
            'config':         self.__config_raw,
            'multiple_obs':   self.multiple_obs,
            'LO_NCPAs':       self.LO_NCPAs,
            'chrom_defocus':  self.chrom_defocus,
            'model_type':     self.model_type,
            'use_Moffat':     self.use_Moffat,   # kept for backward compat readers
            'retain_PSDs':    self.retain_PSDs,
            'Z_mode_max':     self.Z_mode_max,
            'N_spline_nodes': self.N_wvl_ctrl if self.use_splines else None,

            # Full MUSE-NFM spectral grid definition.
            'λ_min':         self.λ_min,
            'λ_max':         self.λ_max,
            'num_λ_slices':  self.num_λ_slices,

            # Mutable runtime state.
            'inputs': self.inputs_manager.save(),
            
            'device':         str(self.device),
            'dtype':          dtype_name,
        }

        return self._tree_to_cpu(store_data) if cpu else store_data


    @classmethod
    def load(cls, store_data: dict, *, device=None, config=None):
        """
        Reconstruct a PSFModelNFM instance from save() output.

        Parameters
        ----------
        store_data : dict
            Dictionary produced by PSFModelNFM.save().
        device : str | torch.device | None
            Optional device override.  If None, uses the device stored in the save.
        config : dict | None
            Fallback config to use when the store was created before config
            serialization was added (backward compatibility).
        """
        
        def _dtype_from_name(dtype):
            """Accept torch.dtype objects and names such as 'float32' / 'torch.float32'."""
            if isinstance(dtype, torch.dtype):
                return dtype
            if dtype is None:
                return torch.float32
            name = str(dtype).replace('torch.', '')
            return getattr(torch, name)
        
        
        def _restore_manager(manager, payload, device):
            out = manager.load(payload)
            out.to_float()
            out.to(device)
            return out
        
        def _restore_config(config, device):
            sample_value = config['atmosphere']['Seeing']
            if sample_value.device != device:
                from tiptorch.managers.config_manager import ConfigManager
                config_manager = ConfigManager()
                config_manager.Convert(config, framework='pytorch', device=device)
                
            return config
        
        # Backward compatibility with the old save() key.
        if 'λ_min, λmax, N_λ' in store_data:
            λ_min, λ_max, num_λ_slices = store_data['λ_min, λmax, N_λ']
        else:
            λ_min        = store_data.get('λ_min', 475.e-9)
            λ_max        = store_data.get('λ_max', 935.e-9)
            num_λ_slices = store_data.get('num_λ_slices', 3681)

        device = torch.device(device if device is not None else store_data.get('device', default_device))
        dtype  = _dtype_from_name(store_data.get('dtype', torch.float32))

        _raw_config = store_data.get('config', config)
        if _raw_config is None:
            raise ValueError(
                "The saved file does not contain a 'config' entry (old format). "
                "Pass the config explicitly via PSFModelNFM.load(..., config=ob.model_config)."
            )

        new_instance = cls(
            config         = _restore_config(_raw_config, device),
            multiple_obs   = store_data.get('multiple_obs', store_data.get('multiple_OBs', False)),
            LO_NCPAs       = store_data.get('LO_NCPAs', True),
            chrom_defocus  = store_data.get('chrom_defocus', False),
            model_type     = store_data.get('model_type', 'hybrid' if store_data.get('use_Moffat', False) else 'physics-based'),
            retain_PSDs    = store_data.get('retain_PSDs', False),
            Z_mode_max     = store_data.get('Z_mode_max', 9),
            N_spline_nodes = store_data.get('N_spline_nodes', None),
            device         = device,
            dtype          = dtype,
            λ_min          = λ_min,
            λ_max          = λ_max,
            num_λ_slices   = num_λ_slices,
        )

        # Restore inputs and backup inputs.
        new_instance.inputs_manager = _restore_manager(new_instance.inputs_manager, store_data.get('inputs'), device)

        return new_instance


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
            model_type      = self.model_type,
            Z_mode_max      = self.Z_mode_max,
            device          = self.device,
            dtype           = self.model.dtype
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

        physics_on = self.model_type != 'psfao'
        PSD_include = {
            'fitting':         True,
            'WFS noise':       physics_on,
            'spatio-temporal': physics_on,
            'aliasing':        False,
            'chromatism':      physics_on,
            'diff. refract':   physics_on,
            'Moffat':          self.model_type in ('hybrid', 'psfao'),
        }
        
        self.model = TipTorch(
            AO_config = config,
            AO_type = 'LTAO' if self.model_type in ('physics-based', 'hybrid') else 'PSFAO',
            pupil = pupil,
            PSD_include = PSD_include,
            norm_regime = 'sum',
            device = self.device,
            oversampling = 1,
            retain_PSDs = self.retain_PSDs,
            dtype = self.dtype
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
            # An auxiliary chromatic normalization factor. It's different from F and used to store chromatic flux crop factor due to finite
            # PSF size and chromatic core vs. wings flux ratio, Meanwhile F, is used to normalize the overall flux level of the PSF
            # across wavelengths due to its morphology
            add_input('F_norm_λ_ctrl', torch.tensor([[1.0,]*self.N_wvl_ctrl]*N_obs), norm=norm_F_λ_norm, optimizable=False, is_shared=True)  
        else:
            # Same stuff here, but without splines
            add_input('F',  torch.tensor([[1.0,]*N_wvl]*N_obs),  norm_F,   is_shared=True)
            add_input('bg', torch.tensor([[0.0,]*N_wvl]*N_obs),  norm_bg,  is_shared=True)
            add_input('dx', torch.tensor([[0.0,]*N_wvl]*N_src),  norm_dxy, is_shared=False)
            add_input('dy', torch.tensor([[0.0,]*N_wvl]*N_src),  norm_dxy, is_shared=False)
            add_input('F_norm_λ', torch.tensor([[1.0,]*N_wvl]*N_obs), norm=norm_F_λ_norm, optimizable=False, is_shared=True)

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
        
        # Von Kármán turbulence parameters
        add_input('r0', self.model.r0.detach().clone(), norm_r0, is_shared=True)
        add_input('L0', self.model.L0.detach().clone(), norm_L0, is_shared=True)

        if self.model_type != 'psfao':
            # HO WFSing error correction factor — only needed when physics PSDs are active
            add_input('dn',  torch.tensor([0.25]*N_obs),    norm_dn, is_shared=True)
            
            # Wind speed and direction are only accounted for the ground layer
            add_input('wind_speed_single', self.model.wind_speed[:,0].detach().clone().unsqueeze(-1), norm_wind_speed, is_shared=True)
            add_input('wind_dir_single', self.model.wind_dir[:,0].detach().clone().unsqueeze(-1), norm_wind_speed, optimizable=False, is_shared=True)
            add_input('Cn2_weights', self.model.Cn2_weights.detach().clone(), norm_Cn2_profile, optimizable=False, is_shared=True)
            
            # Sources directions within the FoV
            add_input('src_dirs_x', self.model.src_dirs_x.detach().clone(), norm=norm_src_coords, optimizable=False, is_shared=False)
            add_input('src_dirs_y', self.model.src_dirs_y.detach().clone(), norm=norm_src_coords, optimizable=False, is_shared=False)

        # Add Moffat PSD absorber's parameters (if Moffat regime is active)
        if self.model_type in ('hybrid', 'psfao'):
            add_input('b',     torch.tensor([0.0]*N_obs),  norm_b,     optimizable=True,  is_shared=True)
            add_input('ratio', torch.tensor([1.0]*N_obs),  norm_ratio, optimizable=False, is_shared=True)
            add_input('theta', torch.tensor([0.0]*N_obs),  norm_theta, optimizable=False, is_shared=True)
            
            if self.model_type == 'psfao':
                # Stronger Moffat component to fully replace the physical PSDs
                add_input('amp',   torch.tensor([1e-1]*N_obs),  norm_amp,   optimizable=True,  is_shared=True)
                add_input('alpha', torch.tensor([1.0]*N_obs),  norm_alpha, optimizable=True,  is_shared=True)
                add_input('beta',  torch.tensor([1.5]*N_obs),  norm_beta,  optimizable=True,  is_shared=True)
            else:
                # small Moffat component
                add_input('amp',   torch.tensor([1e-4]*N_obs), norm_amp,   optimizable=True,  is_shared=True)
                add_input('beta',  torch.tensor([2.5]*N_obs),  norm_beta,  optimizable=True,  is_shared=True)
                add_input('alpha', torch.tensor([2.0]*N_obs),  norm_alpha, optimizable=True,  is_shared=True)

        # Overall per-source flux scaling factor, can be used per-source for photometry fine tuning
        add_input('F_norm', torch.tensor([[1.0,]]*N_src), norm=norm_F_norm, optimizable=(not self.multiple_obs), is_shared=False)
        
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
    
    
    def SetWavelengths(self, wavelengths: torch.Tensor, suppress_pupil_update=False):
        # Cheap pointer check first (same tensor object → same values, no GPU sync needed)
        if self.λ_sim.data_ptr() == wavelengths.data_ptr():
            return
        # Fallback value check - forces a CUDA sync, but only reached for distinct tensor objects
        if self.λ_sim.shape == wavelengths.shape and \
            torch.allclose(wavelengths.flatten(), self.λ_sim.flatten(), atol=1e-12):
            return

        self.model.SetWavelengths(wavelengths, suppress_pupil_update=suppress_pupil_update)
        self.λ_sim = wavelengths
        if self.use_splines:
            self.λ_sim_normed = self.norm_wvl(self.λ_sim) # [0...1], simulated wavelengths normalized to [0...1] range for spline evaluation


    def SetImageSize(self, img_size: int):
        # Compare values to avoid unnecessary updates
        if self.model.N_pix == img_size:
            return
        self.model.SetImageSize(img_size)
    
    
    @torch.no_grad()
    def SimulateSpectralRange(
        self,
        λ_min        = None,
        λ_max        = None,
        src_ids      = None,
        λ_batch_size = 100,
        sequential   = True,
        verbose      = False,
        force_cpu    = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        Simulate PSFs over a sub-range (or the full range) of the MUSE NFM spectrum.

        Parameters
        ----------
        λ_min, λ_max : float or None
            Wavelength bounds (same units as 'self.λ_full', i.e. metres); 'None' falls back to the respective end of the full spectrum.
        src_ids : int | list[int] | None
            Sources to simulate.  'None' simulates all sources.
        λ_batch_size : int
            Number of wavelength slices per forward pass.
        sequential : bool
            If 'True' (default) simulate one source at a time - lower VRAM, slower for large source counts.
            If 'False' simulate all selected sources in a single forward call per λ-batch - faster but VRAM scales with N_src_sim.
        verbose : bool
            Show a tqdm progress bar over wavelength slices.
        force_cpu : bool
            Store the output tensor on CPU to avoid VRAM overflow.

        Returns
        -------
        PSFs_out : torch.Tensor  [N_src_sim, N_λ_range, N_pix, N_pix]
        λ_range  : torch.Tensor  [N_λ_range]  - the wavelengths that were simulated
        '''
        
        # ── 1. Resolve the spectral sub-range ──────────────────────────────────
        λ_lo = self.λ_full.min().item() if λ_min is None else λ_min
        λ_hi = self.λ_full.max().item() if λ_max is None else λ_max

        mask = (self.λ_full >= λ_lo) & (self.λ_full <= λ_hi)
        if mask.sum() == 0:
            raise ValueError(f"No wavelength slices found in [{λ_lo}, {λ_hi}]. Check units.")

        λ_range        = self.λ_full[mask]               # [N_λ_range]
        global_indices = mask.nonzero(as_tuple=True)[0]  # positions in λ_full

        # ── 2. Resolve source list ──────────────────────────────────────────────
        _N_src = self.model.N_src
        
        if src_ids is None:
            src_ids_list = list(range(_N_src))
        elif isinstance(src_ids, int):
            src_ids_list = [src_ids]
        else:
            src_ids_list = list(src_ids)
            
        N_src_sim = len(src_ids_list)

        # ── 3. Build the fully-evaluated model inputs dict ─────────────────────────────
        x_dict = self.inputs_manager.to_dict()
        x_dict_λ_full = {}

        for entry in self.polychromatic_params:
            if entry + '_ctrl' in x_dict:
                x_dict_λ_full[entry] = self.evaluate_splines(x_dict[entry + '_ctrl'], self.λ_full_normed)
                x_dict.pop(entry + '_ctrl')

        x_dict_λ_full['Jx'] = x_dict_λ_full['J']
        x_dict_λ_full['Jy'] = x_dict_λ_full['J']
        x_dict_λ_full.pop('J')

        if 'wind_speed_single' in x_dict:
            x_dict['wind_speed'] = torch.nn.functional.pad(x_dict['wind_speed_single'].view(-1, 1), (0, self.model.N_L - 1))
            x_dict.pop('wind_speed_single')

        if 'wind_dir_single' in x_dict:
            x_dict['wind_dir'] = torch.nn.functional.pad(x_dict['wind_dir_single'].view(-1, 1), (0, self.model.N_L - 1))
            x_dict.pop('wind_dir_single')

        chrom_defocus = x_dict['chrom_defocus'] if self.chrom_defocus else None
        phase_        = self.phase_func(x_dict['LO_coefs'], chrom_defocus) if self.LO_NCPAs else None

        x_dict.update(x_dict_λ_full)

        per_src_params       = [p.replace('_ctrl', '') for p in self.inputs_manager.input_managers['per_src'].to_dict().keys()]
        polychromatic_params = list(x_dict_λ_full.keys())
        
        # Shared non-chromatic entries - constant across λ and sources
        x_ = { key: x_dict[key] for key in x_dict if key not in per_src_params and key not in polychromatic_params }

        # ── 4. Allocate output ──────────────────────────────────────────────────
        device    = torch.device('cpu') if force_cpu else self.device
        N_λ_range = len(λ_range)
        PSFs_out  = torch.zeros((N_src_sim, N_λ_range, self.model.N_pix, self.model.N_pix), device=device)

        out_batches  = [slice(i, min(i + λ_batch_size, N_λ_range)) for i in range(0, N_λ_range, λ_batch_size)]
        _initial_wvl = self.λ_sim.clone()

        if verbose:
            from tqdm import tqdm
            pbar = tqdm(total=N_λ_range, desc='Simulating spectral range')
            
        # ── 5. PSF simulation loop ──────────────────────────────────────────────────
        self.model.N_src = 1 if sequential else N_src_sim

        for out_sl in out_batches:
            global_sl = global_indices[out_sl]
            λ_batch   = λ_range[out_sl]

            if verbose:
                pbar.update(len(λ_batch))

            self.model.SetWavelengths(λ_batch)

            for entry in polychromatic_params:
                x_[entry] = x_dict[entry][..., global_sl]

            # LO NCPAs induced phase error is shared between all sources, but varies with λ
            if self.LO_NCPAs:
                cd_batch = x_dict['chrom_defocus'][..., global_sl] if self.chrom_defocus else None
                self.model.ComputeStaticOTF(self.phase_func(x_dict['LO_coefs'], cd_batch))

            if sequential:
                for out_idx, src_id in enumerate(src_ids_list):
                    for entry in per_src_params:
                        x_[entry] = (x_dict[entry][src_id, global_sl].unsqueeze(0) if entry in polychromatic_params else x_dict[entry][src_id, ...].unsqueeze(0))
                        
                    result = self.model(x_, None, None)
                    PSFs_out[out_idx, out_sl] = result.cpu() if force_cpu else result
            else:
                for entry in per_src_params:
                    x_[entry] = (x_dict[entry][src_ids_list, :][:, global_sl] if entry in polychromatic_params else x_dict[entry][src_ids_list, ...])
                
                result = self.model(x_, None, None)
                PSFs_out[:, out_sl] = result.cpu() if force_cpu else result

        # ── 6. Restore sparse model state ──────────────────────────────────────────────
        self.model.N_src = _N_src
        self.SetWavelengths(_initial_wvl)
        if self.LO_NCPAs:
            self.model.ComputeStaticOTF(phase_)

        if verbose:
            pbar.close()

        torch.cuda.empty_cache()

        return PSFs_out, λ_range   # [N_src_sim, N_λ_range, N_pix, N_pix], [N_λ_range]


    def SimulateFullSpectrum(self, src_ids=None, λ_batch_size=100, verbose=False, force_cpu=True) -> torch.Tensor:
        '''Backward-compatible wrapper - calls SimulateSpectralRange over the full spectrum.'''
        PSFs, _ = self.SimulateSpectralRange(
            src_ids=src_ids, λ_batch_size=λ_batch_size,
            sequential=True, verbose=verbose, force_cpu=force_cpu,
        )
        return PSFs


    @torch.no_grad()
    def ComputeStrehl(self):
        ''' Computes Strehl ratio (SR) across the simulated wavelengths for the on-axis PSF '''
        # Back-up the original coordinates
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
