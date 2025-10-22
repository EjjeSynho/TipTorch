# %reload_ext autoreload
# %autoreload 2

import sys
sys.path.insert(0, '..')

import torch
import numpy as np
from warnings import warn
import gc

from PSF_models.TipTorch import TipTorch
from tools.static_phase import LWEBasis, PixelmapBasis, ZernikeBasis
from managers.input_manager import InputsManager
from tools.normalizers import Uniform
from tools.utils import GradientLoss, rad2mas
from project_settings import device


class PSFModelIRDIS:
    def __init__(
        self,
        config,
        LWE_flag        = True,
        LO_NCPAs        = False,
        fit_wind        = True,
        use_Zernike     = True,
        N_modes         = 9,
        LO_map_size     = 31,
        device          = device
    ):
        
        self.LWE_flag        = LWE_flag
        self.LO_NCPAs        = LO_NCPAs
        self.fit_wind        = fit_wind
        self.use_Zernike     = use_Zernike & LO_NCPAs
        self.N_modes         = N_modes
        self.LO_map_size     = LO_map_size
        self.device          = device
        
        self.init_model(config)
        # Initialize NCPAs (including LWE basis if needed)
        self.init_NCPAs()
        self.init_model_inputs()
        
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        if self.device.type == 'mps':
            torch.mps.empty_cache()
            torch.mps.synchronize()
    

    def cleanup(self):
        """Explicitly clean up GPU memory"""
        # Clean up model (which will trigger TipTorch cleanup)
        if hasattr(self, 'model'):
            if hasattr(self.model, 'cleanup'):
                self.model.cleanup()
            del self.model
        
        # Clean up basis
        if hasattr(self, 'LWE_basis'):
            del self.LWE_basis
            
        if hasattr(self, 'NCPAs_basis'):
            del self.NCPAs_basis
        
        # Clean up inputs manager
        if hasattr(self, 'inputs_manager'):
            del self.inputs_manager

        gc.collect()
        
        # Clear cache
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        if self.device.type == 'mps':
            torch.mps.empty_cache()
            torch.mps.synchronize()
    

    def __del__(self):
        """Destructor to ensure GPU memory is freed"""
        self.cleanup()
        

    def init_model(self, config):
        PSD_include = {
            'fitting':         True,
            'WFS noise':       True,
            'spatio-temporal': True,
            'aliasing':        True,
            'chromatism':      True,
            'diff. refract':   True,
            'Moffat':          False
        }
        self.model = TipTorch(config, 'SCAO', None, PSD_include, 'sum', self.device, oversampling=1)
        self.model.to_float()


    def init_NCPAs(self):
        # Always initialize LWE basis if LWE_flag is True
        if self.LWE_flag:
            self.LWE_basis = LWEBasis(self.model, ignore_pupil=False)
        
        # Initialize NCPAs basis only if LO_NCPAs flag is True
        if self.LO_NCPAs:
            if self.use_Zernike:
                self.NCPAs_basis = ZernikeBasis(self.model, self.N_modes+2, ignore_pupil=False)
                self.NCPAs_basis.basis = self.NCPAs_basis.basis[2:,...]  # remove tip/tilt
            else:
                self.NCPAs_basis = PixelmapBasis(self.model, ignore_pupil=False)
            
    def store_transforms(self):
        from tools.normalizers import TransformSequence, Uniform
        from project_settings import DATA_FOLDER
        import pickle
        
        norm_FWHM = TransformSequence(transforms=[Uniform(a=0,     b=5)])

        norm_F        = TransformSequence(transforms=[Uniform(a=0.0,   b=1.0)])
        norm_bg       = TransformSequence(transforms=[Uniform(a=-5e-6, b=5e-6)])
        norm_r0       = TransformSequence(transforms=[Uniform(a=0.05,  b=0.5)])
        norm_dxy      = TransformSequence(transforms=[Uniform(a=-1,    b=1)])
        norm_J        = TransformSequence(transforms=[Uniform(a=0,     b=40)])
        norm_Jxy      = TransformSequence(transforms=[Uniform(a=-180,  b=180)])
        norm_LWE      = TransformSequence(transforms=[Uniform(a=-20,   b=20)])
        norm_dn       = TransformSequence(transforms=[Uniform(a=-0.02, b=0.02)])
        norm_wind_spd = TransformSequence(transforms=[Uniform(a=0,     b=20)])
        norm_wind_dir = TransformSequence(transforms=[Uniform(a=0,     b=360)])
        norm_LO       = TransformSequence(transforms=[Uniform(a=-10,   b=10)])

        # Dump transforms (preserve original functionality)
        transforms_dump = {
            'F R':         norm_F,
            'F L':         norm_F,
            'bg R':        norm_bg,
            'bg L':        norm_bg,
            'r0':          norm_r0,
            'dx L':        norm_dxy,
            'dx R':        norm_dxy,
            'dy L':        norm_dxy,
            'dy R':        norm_dxy,
            'dn':          norm_dn,
            'Jx':          norm_J,
            'Jy':          norm_J,
            'Jxy':         norm_Jxy,
            'Wind dir':    norm_wind_dir,
            'Wind speed':  norm_wind_spd,
            'LWE coefs':   norm_LWE,
            'FWHM fit L':  norm_FWHM,
            'FWHM fit R':  norm_FWHM,
            'FWHM data L': norm_FWHM,
            'FWHM data R': norm_FWHM
        }

        path_transforms = DATA_FOLDER / 'reduced_telemetry/SPHERE/IRDIS_model_norm_transforms.pickle'

        with open(path_transforms, 'wb') as handle:
            df_transforms_store = {}
            for entry in transforms_dump:
                df_transforms_store[entry] = transforms_dump[entry].store()
            pickle.dump(df_transforms_store, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
                
    def init_model_inputs(self):
        self.inputs_manager = InputsManager()
        
        # Initialize normalizers
        norm_F        = Uniform(a=0.0,   b=1.0)
        norm_bg       = Uniform(a=-5e-6, b=5e-6)
        norm_r0       = Uniform(a=0.05,  b=0.5)
        norm_dxy      = Uniform(a=-1,    b=1)
        norm_J        = Uniform(a=0,     b=40)
        norm_Jxy      = Uniform(a=-180,  b=180)
        norm_LWE      = Uniform(a=-20,   b=20)
        norm_dn       = Uniform(a=-0.02, b=0.02)
        norm_wind_spd = Uniform(a=0,     b=20)
        norm_wind_dir = Uniform(a=0,     b=360)
        norm_LO       = Uniform(a=-10,   b=10)


        # Add parameters to InputsManager with their normalizers
        self.inputs_manager.add('r0',  self.model.r0.clone().cpu(), norm_r0)
        self.inputs_manager.add('F',   torch.tensor([[1.0,]*2]), norm_F)
        self.inputs_manager.add('dx',  torch.tensor([[0.0,]*2]), norm_dxy)
        self.inputs_manager.add('dy',  torch.tensor([[0.0,]*2]), norm_dxy)
        self.inputs_manager.add('bg',  torch.tensor([[0.0,]*2]), norm_bg)
        self.inputs_manager.add('dn',  torch.tensor([0.0]),      norm_dn)
        self.inputs_manager.add('Jx',  torch.tensor([[7.5]]),    norm_J)
        self.inputs_manager.add('Jy',  torch.tensor([[7.5]]),    norm_J)
        self.inputs_manager.add('Jxy', torch.tensor([[0]]),      norm_Jxy)

        if self.fit_wind:
            self.inputs_manager.add('wind_dir', self.model.wind_dir.clone().cpu(), norm_wind_dir)
            
        if self.LWE_flag:
            self.inputs_manager.add('LWE_coefs', torch.zeros([1,12]), norm_LWE)

        if self.LO_NCPAs:
            if self.use_Zernike:
                if self.N_modes is not None:
                    self.inputs_manager.add('LO_coefs', torch.zeros([1, self.N_modes]), norm_LO)
                    self.inputs_manager.set_optimizable('LO_coefs', True)
            else:
                if self.LO_map_size is not None:
                    self.inputs_manager.add('LO_coefs', torch.zeros([1, self.LO_map_size**2]), norm_LO)
                    self.inputs_manager.set_optimizable('LO_coefs', True)

        self.inputs_manager.to_float()
        self.inputs_manager.to(self.device)


    def phase_func(self, x_dict):
        """Create phase function based on configuration and coefficients"""
        
        if self.LWE_flag or self.LO_NCPAs:
        
            OPD = 0.0
            
            # Add LWE component if available
            if 'LWE_coefs' in x_dict:
                OPD += self.LWE_basis.compute_OPD(x_dict['LWE_coefs'])
            
            # Add NCPAs component if available
            if self.LO_NCPAs and hasattr(self, 'NCPAs_basis') and 'LO_coefs' in x_dict:
                if self.use_Zernike:
                    NCPAs_OPD = self.NCPAs_basis.compute_OPD(x_dict['LO_coefs'])
                else:
                    NCPAs_OPD = self.NCPAs_basis.compute_OPD(x_dict['LO_coefs'].view(1, self.LO_map_size, self.LO_map_size))
                OPD += NCPAs_OPD

            OPD = OPD.unsqueeze(1) if OPD.ndim == 3 else OPD # (N_src, 1, H, W) or (N_src, N_wvl, H, W)

            if hasattr(self, 'LWE_basis'):
                return self.LWE_basis.OPD2Phase(OPD)
            else:
                return self.NCPAs_basis.OPD2Phase(OPD)

        return None
    

    def forward(self, x_dict=None):
        if x_dict is None:
            x_dict = self.inputs_manager.to_dict()
            
        return self.model(x_dict, None, lambda: self.phase_func(x_dict))


    def compensate_PTT_coupling(self):
        """Compensate for PTT coupling in LWE modes"""
        from tools.static_phase import BuildPTTBasis, decompose_WF, project_WF
        
        if not self.LWE_flag or not hasattr(self, 'LWE_basis'):
            return None, None, None
            
        PTT_basis = BuildPTTBasis(self.model.pupil.cpu().numpy(), True).to(self.device).float()
        TT_max = PTT_basis.abs()[1,...].max().item()
        pixel_shift = lambda coef: 2 * TT_max * rad2mas * 1e-9 * coef / self.model.psInMas / self.model.D / (1-7/self.model.pupil.shape[-1])

        # Compute LWE OPD using compute_OPD method
        LWE_OPD = self.LWE_basis.compute_OPD(self.inputs_manager['LWE_coefs']) * 1e9  # [nm]
        PPT_OPD = project_WF(LWE_OPD, PTT_basis, self.model.pupil)
        PTT_coefs = decompose_WF(LWE_OPD, PTT_basis, self.model.pupil)

        # Update LWE coefficients and pixel shifts
        self.inputs_manager['LWE_coefs'] = decompose_WF(LWE_OPD-PPT_OPD, self.LWE_basis.modal_basis, self.model.pupil)
        self.inputs_manager['dx'] -= pixel_shift(PTT_coefs[:, 2])
        self.inputs_manager['dy'] -= pixel_shift(PTT_coefs[:, 1])

        # Compute NCPA OPD if available
        NCPA_OPD = None
        if self.LO_NCPAs and hasattr(self, 'NCPAs_basis') and 'LO_coefs' in self.inputs_manager.to_dict():
            if self.use_Zernike:
                NCPA_OPD = self.NCPAs_basis.compute_OPD(self.inputs_manager['LO_coefs']) * 1e9  # [nm]
            else:
                # Use compute_OPD method instead of direct interpolation
                NCPA_OPD = self.NCPAs_basis.compute_OPD(self.inputs_manager['LO_coefs'].view(1, self.LO_map_size, self.LO_map_size)) * 1e9  # [nm]

        LWE_OPD -= PPT_OPD

        return LWE_OPD, PPT_OPD, NCPA_OPD


    def get_OPD_map(self):
        """Get the combined OPD map"""
        combined_OPD = None
        
        # Get LWE OPD if available
        if self.LWE_flag and hasattr(self, 'LWE_basis'):
            LWE_OPD = self.LWE_basis.compute_OPD(self.inputs_manager['LWE_coefs']) * 1e9  # [nm]
            
            # Compensate for PTT if needed
            from tools.static_phase import BuildPTTBasis, project_WF
            PTT_basis = BuildPTTBasis(self.model.pupil.cpu().numpy(), True).to(self.device).float()
            PPT_OPD = project_WF(LWE_OPD, PTT_basis, self.model.pupil)
            combined_OPD = LWE_OPD - PPT_OPD
        
        # Get NCPA OPD if available
        if self.LO_NCPAs and hasattr(self, 'NCPAs_basis'):
            if self.use_Zernike:
                NCPA_OPD = self.NCPAs_basis.compute_OPD(self.inputs_manager['LO_coefs']) * 1e9  # [nm]
            else:
                NCPA_OPD = self.NCPAs_basis.compute_OPD(self.inputs_manager['LO_coefs'].view(1, self.LO_map_size, self.LO_map_size)) * 1e9  # [nm]
            
            if combined_OPD is not None:
                combined_OPD = combined_OPD + NCPA_OPD
            else:
                combined_OPD = NCPA_OPD
        
        return combined_OPD


    def GetNewPhotons(self):
        """Calculate new photon count based on current noise parameters"""
        from torchmin import minimize
        
        WFS_noise_var = self.model.dn + self.model.NoiseVariance(self.model.r0.abs())
        N_ph_0 = self.model.WFS_Nph.clone()

        def func_Nph(x):
            self.model.WFS_Nph = x
            var = self.model.NoiseVariance(self.model.r0.abs())
            return (WFS_noise_var-var).flatten().abs().sum()

        result_photons = minimize(func_Nph, N_ph_0, method='bfgs', disp=0)
        self.model.WFS_Nph = N_ph_0.clone()

        return result_photons.x

    __call__ = forward