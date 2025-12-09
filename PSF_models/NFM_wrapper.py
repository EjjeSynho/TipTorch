# %reload_ext autoreload
# %autoreload 2

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
from managers.input_manager  import InputsManager
from tools.normalizers import Uniform, Uniform0_1
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
        
        self.LO_N_params = 75
        self.Z_mode_max  = Z_mode_max
        
        self.LO_NCPAs        = LO_NCPAs
        self.multiple_obs    = multiple_obs
        self.device          = device
        self.Moffat_absorber = Moffat_absorber
        self.use_splines     = use_splines
        self.chrom_defocus   = chrom_defocus and LO_NCPAs
        
        config = self.init_configs(config)
        self.wavelengths = config['sources_science']['Wavelength'].squeeze()
        
        self.init_model(config)
        if LO_NCPAs:
            self.init_NCPAs()
        # self.polychromatic_params = ['F', 'dx', 'dy', 'Jx', 'Jy']
        self.polychromatic_params = ['F', 'dx', 'dy'] + (['chrom_defocus'] if self.chrom_defocus else ['J'])

        if self.use_splines:
            self.N_spline_ctrl = 5
            # TODO: must be scaling of spline values, too
            self.norm_wvl = Uniform0_1(a=self.wavelengths.min().item(), b=self.wavelengths.max().item())
            self.x_ctrl = torch.linspace(0, 1, self.N_spline_ctrl, device=self.device)

        self.init_model_inputs()
        
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        if self.device.type == 'mps':
            torch.mps.empty_cache()
            torch.mps.synchronize()
    

    def _cleanup_dict_recursive(self, obj):
        """Recursively clean up tensors in nested dictionaries"""
        if isinstance(obj, dict):
            for key in list(obj.keys()):
                self._cleanup_dict_recursive(obj[key])
                del obj[key]
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                self._cleanup_dict_recursive(item)
        elif isinstance(obj, torch.Tensor):
            del obj
    
    
    def cleanup(self):
        """Explicitly clean up GPU memory"""
        # Clean up wavelengths tensor first (it's a reference to data inside model_config)
        if hasattr(self, 'wavelengths'):
            del self.wavelengths
        
        # Clean up model (which will trigger TipTorch cleanup)
        if hasattr(self, 'model'):
            if hasattr(self.model, 'cleanup'):
                self.model.cleanup()
            del self.model
        
        # if hasattr(self, 'model_config'):
            # Just delete the reference, don't recursively clean
            # (already done by self.model.cleanup())
            # del self.model_config
        
        # Clean up basis
        if hasattr(self, 'LO_basis'):
            del self.LO_basis
        
        # Clean up inputs manager
        if hasattr(self, 'inputs_manager'):
            del self.inputs_manager
        
        # Clean up other tensors
        if hasattr(self, 'x_ctrl'):
            del self.x_ctrl
        
        if hasattr(self, 'norm_wvl'):
            del self.norm_wvl

        gc.collect()
        
        # Clear cache
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        if self.device.type == 'mps':
            torch.mps.empty_cache()
            torch.mps.synchronize()
    
    '''
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
            config=self.model.config,
            multiple_obs=self.multiple_obs,
            LO_NCPAs=self.LO_NCPAs,
            chrom_defocus=self.chrom_defocus,
            use_splines=self.use_splines,
            Moffat_absorber=self.Moffat_absorber,
            Z_mode_max=self.Z_mode_max,
            device=self.device
        )
        
        # Deep copy the TipTorch model state
        if hasattr(self.model, 'state_dict'):
            new_instance.model.load_state_dict(copy.deepcopy(self.model.state_dict()))
        
        # Copy wavelengths (already handled by config, but ensure reference is correct)
        new_instance.wavelengths = self.wavelengths.clone()
        
        # Deep copy the inputs manager using its built-in copy method
        if hasattr(self, 'inputs_manager'):
            new_instance.inputs_manager = self.inputs_manager.copy()
        
        # Copy spline control points if using splines
        if self.use_splines and hasattr(self, 'x_ctrl'):
            new_instance.x_ctrl = self.x_ctrl.clone()
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
    '''

    def __del__(self):
        """Destructor to ensure GPU memory is freed"""
        self.cleanup()
        

    def init_configs(self, config):
        if len(config) > 1: # Multiple sources
            if self.multiple_obs:
                model_config = MultipleTargetsInDifferentObservations(config, device=self.device)
            else:
                raise NotImplementedError("Multiple sources in one observation case is not implemented yet.")
        else:
            model_config = MultipleTargetsInDifferentObservations(config, device=self.device)


        return model_config


    def init_model(self, config):
        # model_config['DM']['DmPitchs'] = torch.tensor([0.245], dtype=torch.float32, device=device)  # [m]
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
        self.model = TipTorch(config, 'LTAO', pupil, PSD_include, 'sum', self.device, oversampling=1)
        # _ = self.model()


    def init_NCPAs(self): 
        # LO_basis = PixelmapBasis(model, ignore_pupil=False)
        Z_basis = ZernikeBasis(self.model, N_modes=self.LO_N_params, ignore_pupil=False)
        sausage_basis = MUSEPhaseBump(self.model, ignore_pupil=False)

        # LO NCPAs + phase bump optimized jointly
        composite_basis = torch.concat([
            (sausage_basis.OPD_map).unsqueeze(0).flip(-2)*5e6*self.model.pupil.unsqueeze(0),
            Z_basis.basis[2:self.Z_mode_max,...]
        ], dim=0)

        self.LO_basis = ArbitraryBasis(self.model, composite_basis, ignore_pupil=False)
        self.LO_N_params = self.LO_basis.N_modes
        
        
    def init_model_inputs(self):
        self.inputs_manager = InputsManager()
        
        N_wvl = len(self.wavelengths)
        N_src = self.model.N_src
        
        # Initialize normalizers/transforms
        norm_F           = Uniform(a=0.0,   b=1.0)
        norm_bg          = Uniform(a=-5e-6, b=5e-6)
        norm_r0          = Uniform(a=0,     b=1)
        norm_L0          = Uniform(a=0,     b=10)
        norm_dxy         = Uniform(a=-1,    b=1)
        norm_J           = Uniform(a=0,     b=50)
        norm_Jxy         = Uniform(a=-180,  b=180)
        norm_dn          = Uniform(a=0,     b=5)
        norm_amp         = Uniform(a=0,     b=10)
        norm_b           = Uniform(a=0,     b=0.1)
        norm_alpha       = Uniform(a=-1,    b=10)
        norm_beta        = Uniform(a=0,     b=2)
        norm_ratio       = Uniform(a=0,     b=2)
        norm_theta       = Uniform(a=-np.pi/2, b=np.pi/2)
        norm_wind_speed  = Uniform(a=0, b=10)
        # norm_wind_dir    = Uniform(a=0, b=360)
        # norm_sausage_pow = Uniform(a=0, b=1)
        norm_LO          = Uniform(a=-100, b=100)
        # norm_GL_h        = Uniform(a=0.0, b=2000.0)
        # norm_GL_frac     = Atanh()

        # Add base parameters
        if self.use_splines:

            self.inputs_manager.add('F_ctrl',  torch.tensor([[1.0,]*self.N_spline_ctrl]*N_src),  norm_F)
            self.inputs_manager.add('dx_ctrl', torch.tensor([[0.0,]*self.N_spline_ctrl]*N_src),  norm_dxy)
            self.inputs_manager.add('dy_ctrl', torch.tensor([[0.0,]*self.N_spline_ctrl]*N_src),  norm_dxy)
            self.inputs_manager.add('bg_ctrl', torch.tensor([[0.0,]*self.N_spline_ctrl]*N_src),  norm_bg)
            
            # self.inputs_manager.add('F_x_ctrl',  torch.linspace(0, 1, 5, device=self.device).unsqueeze(0))
            # self.inputs_manager.add('dx_x_ctrl', torch.linspace(0, 1, 5, device=self.device).unsqueeze(0))
            # self.inputs_manager.add('dy_x_ctrl', torch.linspace(0, 1, 5, device=self.device).unsqueeze(0))
            # self.inputs_manager.add('bg_x_ctrl', torch.linspace(0, 1, 5, device=self.device).unsqueeze(0))
                    
        else:
            self.inputs_manager.add('F',  torch.tensor([[1.0,]*N_wvl]*N_src),  norm_F)
            self.inputs_manager.add('dx', torch.tensor([[0.0,]*N_wvl]*N_src),  norm_dxy)
            self.inputs_manager.add('dy', torch.tensor([[0.0,]*N_wvl]*N_src),  norm_dxy)
            self.inputs_manager.add('bg', torch.tensor([[0.0,]*N_wvl]*N_src),  norm_bg)

        if self.chrom_defocus:
            self.inputs_manager.add('J', torch.tensor([[25.0]]*N_src), norm_J)
        else:
            if self.use_splines:
                self.inputs_manager.add('J_ctrl', torch.tensor([[25.0,]*self.N_spline_ctrl]*N_src), norm_J)
                # self.inputs_manager.add('J_x_ctrl', torch.linspace(0, 1, 5, device=self.device).unsqueeze(0))
                
                # self.inputs_manager.add('Jx_ctrl', torch.tensor([[10.0,]*self.N_spline_ctrl]*N_src), norm_dxy)
                # self.inputs_manager.add('Jy_ctrl', torch.tensor([[10.0,]*self.N_spline_ctrl]*N_src), norm_dxy)
            else:
                self.inputs_manager.add('J', torch.tensor([[25.0]*N_wvl]*N_src), norm_J)
                # self.inputs_manager.add('Jx', torch.tensor([[25.0]*N_wvl]*N_src),  norm_J)
                # self.inputs_manager.add('Jy', torch.tensor([[25.0]*N_wvl]*N_src),  norm_J)
                
        self.inputs_manager.add('r0', self.model.r0.clone(), norm_r0)
        self.inputs_manager.add('L0', self.model.L0.clone(), norm_L0)
        self.inputs_manager.add('wind_speed_single', self.model.wind_speed[:,0].clone().unsqueeze(-1), norm_wind_speed)
        self.inputs_manager.add('Jxy', torch.tensor([[0.0]]*N_src), norm_Jxy, optimizable=False)
        self.inputs_manager.add('dn',  torch.tensor([0.25]*N_src),  norm_dn)

        # GL_frac = np.maximum(model.Cn2_weights[0,-1].detach().cpu().numpy().item(), 0.9)
        # GL_h    = model.h[0,-1].detach().cpu().numpy().item()
        # self.inputs_manager.add('GL_frac', torch.tensor([GL_frac]), norm_GL_frac)
        # self.inputs_manager.add('GL_h',    torch.tensor([GL_h]), norm_GL_h)

        # Add Moffat parameters if needed
        if self.Moffat_absorber:
            self.inputs_manager.add('amp',   torch.tensor([1e-4]*N_src), norm_amp)
            self.inputs_manager.add('b',     torch.tensor([0.0]*N_src), norm_b)
            # self.inputs_manager.add('alpha', torch.tensor([4.5]*N_src), norm_alpha)
            self.inputs_manager.add('alpha', torch.tensor([2.0]*N_src), norm_alpha)
            self.inputs_manager.add('beta',  torch.tensor([2.5]*N_src), norm_beta)
            self.inputs_manager.add('ratio', torch.tensor([1.0]*N_src), norm_ratio)
            self.inputs_manager.add('theta', torch.tensor([0.0]*N_src), norm_theta)

        if self.LO_NCPAs:
            if isinstance(self.LO_basis, PixelmapBasis):
                self.inputs_manager.add('LO_coefs', torch.zeros([self.model.N_src, self.LO_N_params**2]), norm_LO)
                self.phase_func = lambda x: self.LO_basis(x.view(1, self.LO_N_params, self.LO_N_params))
                self.OPD_func   = lambda x: x.view(self.model.N_src, self.LO_N_params, self.LO_N_params)


            elif isinstance(self.LO_basis, ZernikeBasis) or isinstance(self.LO_basis, ArbitraryBasis):
                self.inputs_manager.add('LO_coefs', torch.zeros([self.model.N_src, self.LO_N_params]), norm_LO)
                self.OPD_func = lambda x: self.LO_basis.compute_OPD(x.view(self.model.N_src, self.LO_N_params))

                if self.chrom_defocus:
                    self.inputs_manager.add('chrom_defocus', torch.tensor([[0.0,]*N_wvl]*self.model.N_src), norm_LO, optimizable=self.chrom_defocus)
                    defocus_mode_id = 1 # the index of defocus mode

                    def phase_func(x):
                        # coefs_chromatic = self.inputs_manager["LO_coefs"].view(self.model.N_src, self.LO_N_params).unsqueeze(1).repeat(1, N_wvl, 1)
                        # coefs_chromatic[:, :, defocus_mode_id] += self.inputs_manager["chrom_defocus"].view(self.model.N_src, N_wvl) # add chromatic defocus
                        # return self.LO_basis(coefs_chromatic)
                        raise NotImplementedError("Chromatic defocus with callable phase function is not implemented yet.")
                    
                    self.phase_func = phase_func
                else:
                    # self.phase_func = lambda: self.LO_basis(self.inputs_manager["LO_coefs"].view(self.model.N_src, self.LO_N_params))
                    self.phase_func = lambda x: self.LO_basis(x.view(self.model.N_src, self.LO_N_params))
            else:
                raise ValueError('Wrong LO type specified.')
        else:
            self.phase_func = None


        self.inputs_manager.to(self.device)
        self.inputs_manager.to_float()
        
        _ = self.inputs_manager.stack()


    def evaluate_splines(self, y_points, x_grid):
        spline = NaturalCubicSpline(natural_cubic_spline_coeffs(t=self.x_ctrl, x=y_points.T))
        return spline.evaluate(x_grid).T


    def forward(self, x_dict=None, include_list=None):
        if x_dict is None:
            x_dict = self.inputs_manager.to_dict()
    
        if self.use_splines:
            for entry in self.polychromatic_params:
                if entry+'_ctrl' in x_dict:
                    x_dict[entry] = self.evaluate_splines(x_dict[entry+'_ctrl'], self.norm_wvl(self.wavelengths))

        # Clone J entry to Jx and Jy
        x_dict['Jx'] = x_dict['J']
        x_dict['Jy'] = x_dict['J']
        
            # x_dict['wind_dir']   = x_dict['wind_dir_single'].unsqueeze(-1).repeat(1, self.model.N_L)
        if 'wind_speed_single' in x_dict:
            x_dict['wind_speed'] = x_dict['wind_speed_single'].view(-1, 1).repeat(1, self.model.N_L)

        # x_dict['Cn2_weights'] = torch.hstack([x_dict['GL_frac'], 1.0 - x_dict['GL_frac']]).unsqueeze(0)
        # x_dict['h']           = torch.hstack([torch.tensor([0.0], device=self.device), x_dict['GL_h'].abs()]).unsqueeze(0)

        x_ = { key: x_dict[key] for key in include_list } if include_list is not None else x_dict

        phase_ = lambda: self.phase_func(x_dict['LO_coefs']) if self.LO_NCPAs else None

        return self.model(x_, None, phase_generator=phase_)


    def SetWavelengths(self, wavelengths):
        self.model.config['sources_science']['Wavelength'] = wavelengths.view(1,-1) # [nm]
        self.model.Update(init_grids=True, init_pupils=True, init_tomography=True)
        self.wavelengths = wavelengths # [nm]

    def SetImageSize(self, img_size):
        self.model.config['sensor_science']['FieldOfView'] = img_size
        self.model.Update(init_grids=True, init_pupils=True, init_tomography=True)

    __call__ = forward

