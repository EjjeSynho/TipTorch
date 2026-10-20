#%%
import warnings
import torch
import numpy as np
import scipy.special as spc
import torchvision.transforms as transforms
from torch import fft
from torch.nn.functional import interpolate
from astropy.io import fits
from tiptorch.tools.utils import pdims, min_2d, to_little_endian, rotate_images
from tiptorch.tools.air_refraction import AirRefractiveIndexCalculator
from pathlib import Path


class TipTorch(torch.nn.Module):
    '''
    Frames and axes conventions.

    The image frame is the frame of the simulated PSF [N_src, N_wvl, N_pix, N_pix]. The x axis of TipTorch is the image row axis (dim -2)
    and the y axis is the column axis (dim -1): this holds for the PSF shifts dx, dy [pix], the jitter Jx, Jy, the science source and guide
    star directions src_dirs_x/y, GS_dirs_x/y, the wind velocity vx, vy and the grids kx, ky, U, V (kx and U vary along the rows). An
    azimuth of 0 deg in the config points along the rows, 90 deg along the columns.

    [telescope] PupilAngle [deg] is the rotation of the pupil, and of the DM / WFS frame attached to it, with respect to the image frame:
    counter-clockwise on the image for positive values, i.e. the PSF of a model with angle a is scipy.ndimage.rotate(PSF(0), a) (the sense
    of torchvision's rotate and of PupilVLT). It may hold one value per observation ([N_obs]); a single value is shared by all of them.
    The pupil, the apodizer, the static OPD maps (see tools.static_phase), the WFS sampling, the aliasing combs, the DM optimization
    directions, the LGS asterism of tomographic systems and the Moffat PSD are expressed in the pupil frame and rotate with it, through
    the pupil-frame grids returned by kxy_AO_(). The image-plane shifts and jitter, the science source directions, the wind and the single
    guide star of SCAO / SLAO stay in the image frame. Without rotation (all angles 0) the stored grids and masks are used as they are.
    '''

    def _load_pupil_and_apodizer(self):
        # Reference (unrotated) pupil and apodizer [H, W]: loaded from the config files unless given to the constructor (a tensor in the
        # reference orientation, or a generator of the pupil at a given angle). The phase bases are built on these reference masks
        if self.pupil is None:
            if self._pupil_generator is not None:
                self.pupil = torch.as_tensor(self._pupil_generator(0.0), device=self.device, dtype=self.dtype)
            else:
                self.pupil = self.make_tensor(to_little_endian(fits.getdata(Path(self.config['telescope']['PathPupil']))))

        if self.apodizer is None and self.config['telescope']['PathApodizer'] is not None:
            self.apodizer = self.make_tensor(to_little_endian(fits.getdata(Path(self.config['telescope']['PathApodizer']))))

        if self.apodizer is not None:
            assert self.pupil.shape[-1] == self.apodizer.shape[-1], "Pupil and apodizer must have the same size"

        self._orient_pupils()


    def _orient_pupils(self):
        '''
        Pupil and apodizer at the pupil angle(s): pupil_rot / apodizer_rot are [N_obs, H, W] with per-observation angles, [H, W] otherwise
        (the reference masks themselves without rotation). The pupil is regenerated per angle by the generator when there is one, else the
        binary mask is rotated with nearest-neighbour interpolation (no fractional edge values); the apodizer is rotated bilinearly.
        '''
        if self.rotate_pupil:
            angles = self.pupil_angle.tolist()
            if self._pupil_generator is not None:
                self.pupil_rot = torch.stack([torch.as_tensor(self._pupil_generator(angle), device=self.device, dtype=self.dtype) for angle in angles])
            else:
                self.pupil_rot = rotate_images(self.pupil, self.pupil_angle, mode='nearest')

            self.apodizer_rot = rotate_images(self.apodizer, self.pupil_angle, mode='bilinear') if self.apodizer is not None else None

            if len(angles) == 1:
                self.pupil_rot = self.pupil_rot.squeeze(0)
                self.apodizer_rot = self.apodizer_rot.squeeze(0) if self.apodizer_rot is not None else None
        else:
            self.pupil_rot, self.apodizer_rot = self.pupil, self.apodizer

        self._oriented_angles = self.pupil_angle.tolist() # the angles the masks are oriented at (a list, to survive device / dtype changes)


    def UpdateStaticOTF(self):
        """Recompute diffraction-limited static OTF for the current grid sampling."""
        if self.pupil is None:
            raise RuntimeError('Cannot update static OTF: pupil is not initialized. Call InitPupils() first.')

        if not hasattr(self, 'sampling_min') or not hasattr(self, 'nOtf'):
            raise RuntimeError('Cannot update static OTF: grids are not initialized. Call InitGrids() first.')

        phase_size = self.pupil.shape[-1]
        self.pupil_padder = torch.nn.ZeroPad2d(int(round(phase_size * self.sampling_min / 2 - phase_size / 2)))

        # Compute the diffraction-limited OTF from the oriented pupil(s) (and apodizer if present), [1 or N_obs, 1, nOtf, nOtf]
        pupil_phase = self.pupil_rot * self.apodizer_rot if self.apodizer_rot is not None else self.pupil_rot
        self.OTF_static_default = self.Phase2OTF(pupil_phase)
        
        # Live OTF is reset to the diffraction-limited reference whenever sampling changes
        self.OTF_static = self.OTF_static_default.clone()
        return self.OTF_static
    
    
    def _load_vibration_PSD(self):
        # As the pupil and apodizer, the wind-shake temporal PSD [3, N_f] (frequencies [Hz], tip and tilt PSDs) is read from the file given
        # in the config only when it was not provided externally, and only once
        if self.vibration_PSD is None and self.wind_PSD_file is not None:
            self.vibration_PSD = self.make_tensor(to_little_endian(fits.getdata(self.wind_PSD_file)))


    def InitPupils(self):
        self._load_pupil_and_apodizer()
        self._load_vibration_PSD()
        self.UpdateStaticOTF()
        

    def InitValues(self):
        # Reading parameters from the config file
        self.N_src = self.config['NumberSources'] # Make sure it is type(int) already in the config file

        self.wvl = self.config['sources_science']['Wavelength'].view(1, -1) # must be [1, N_wvl]
        self.N_wvl = self.wvl.shape[-1]
        self.wvl_atm = self.config['atmosphere']['Wavelength']

        # Science sources positions relative to the center of FOV, input in [arc]
        src_zenith  = self.config['sources_science']['Zenith'].flatten() / self.rad2arc  # [N_src]
        src_azimuth = torch.deg2rad(self.config['sources_science']['Azimuth']).flatten() # [N_src]
        self.src_dirs_x = torch.tan(src_zenith) * torch.cos(src_azimuth) # [N_src]
        self.src_dirs_y = torch.tan(src_zenith) * torch.sin(src_azimuth) # [N_src]
        
        self.on_axis = False # can be assigned externally to simplify tomographic/anisoplanatic computations
        
        self.psInMas = self.config['sensor_science']['PixelScale']
        self.D       = self.config['telescope']['TelescopeDiameter'] # [m]
        self.N_pix   = self.config['sensor_science']['FieldOfView']  # [pix]
        
        # Telescope pointing
        self.zenith_angle = torch.deg2rad(min_2d(self.config['telescope']['ZenithAngle'])) # [N_obs, 1]
        self.airmass      = 1.0 / torch.cos(self.zenith_angle) # [N_obs, 1]

        # Guide stars parameters
        self.GS_wvl    = self.config['sources_HO']['Wavelength'] #[m]
        self.GS_height = min_2d(self.config['sources_HO']['Height']) * self.airmass #[m]
        self.is_LGS    = (self.GS_height > 0) & self.GS_height.isfinite() # a height of 0 (or inf) means an NGS at an infinite distance
        # Height of the LGS constellation (center of mass of the LGS heights), infinite for NGS: stretches the layer heights and sets the cone effects
        self.LGS_height = torch.where(self.is_LGS.all(dim=-1, keepdim=True), self.GS_height.mean(dim=-1, keepdim=True), torch.inf) # [N_obs, 1]
        GS_angles   = self.config['sources_HO']['Zenith'] / self.rad2arc # [arcsec] from on-axis
        GS_azimuths = torch.deg2rad(self.config['sources_HO']['Azimuth']) # [deg] from on-axis
        self.GS_dirs_x = torch.tan(GS_angles) * torch.cos(GS_azimuths) # [N_obs, N_GS]
        self.GS_dirs_y = torch.tan(GS_angles) * torch.sin(GS_azimuths) # [N_obs, N_GS]
        
        self.N_GS = self.GS_dirs_y.size(-1)
     
        # Atmospheric parameters
        self.wind_speed  = self.config['atmosphere']['WindSpeed']
        self.wind_dir    = self.config['atmosphere']['WindDirection']
        self.Cn2_weights = self.config['atmosphere']['Cn2Weights']
        self.Cn2_heights = self.config['atmosphere']['Cn2Heights'] * self.airmass # [m]
        
        self.N_L = self.Cn2_heights.shape[-1] # number of atmospheric layers simulated in the model
        
        # N_obs is very important! It must = 1 if all simulated targets share the same observational conditions.
        # Doing so can dramatically reduce the computational cost. Otherwise, for multiple targets in multiple conditions,
        # it must be equal to the number of simulated targets / pointings, since each target can be assigned to its own observational conditions

        self.N_obs = self.Cn2_heights.shape[0]

        assert self.N_obs == 1 or self.N_obs == self.N_src

        # Rotation of the pupil (and of the DM / WFS frame) on the image [deg], counter-clockwise for positive values (see the class
        # docstring): one value shared by all observations [1], or one per observation [N_obs]
        self.pupil_angle = torch.as_tensor(self.config['telescope'].get('PupilAngle', 0.0), device=self.device, dtype=self.dtype).flatten()

        if self.pupil_angle.numel() > 1 and bool((self.pupil_angle == self.pupil_angle[0]).all()):
            self.pupil_angle = self.pupil_angle[:1] # identical angles: a single rotated pupil and grid serve all observations

        if self.pupil_angle.numel() not in (1, self.N_obs):
            raise ValueError(f'PupilAngle must hold 1 or N_obs = {self.N_obs} values, got {self.pupil_angle.numel()}')

        self.rotate_pupil = bool((self.pupil_angle != 0).any()) # without rotation, the stored grids and masks are used as they are
        self._pupil_angle_key = tuple(self.pupil_angle.tolist()) # cache key of the pupil-frame grids (rebuilt only when the angles change)

        # Deformable mirror(s) parameters
        self.pitch = self.config['DM']['DmPitchs'].min() #[m], select the DM with the finest resolution 
        self.kc    = 1.0 / (2.0 * self.pitch)
        
        self.h_DM  = self.config['DM']['DmHeights'] # [m]
        self.N_DM  = self.h_DM.shape[0]
        self.DM_opt_weight = self.config['DM']['OptimizationWeight'].view(self.N_obs, -1)  # [N_obs, N_optdir]
        self.N_optdir = self.DM_opt_weight.shape[-1]
        
        self.DM_opt_angle   = self.config['DM']['OptimizationZenith' ].view(self.N_obs, self.N_optdir) / self.rad2arc # [N_obs, N_optdir]
        self.DM_opt_azimuth = torch.deg2rad(self.config['DM']['OptimizationAzimuth'].view(self.N_obs, self.N_optdir)) # [N_obs, N_optdir]

        self.DM_opt_dir_x  = torch.tan(self.DM_opt_angle) * torch.cos(self.DM_opt_azimuth) # [N_obs, N_optdir]
        self.DM_opt_dir_y  = torch.tan(self.DM_opt_angle) * torch.sin(self.DM_opt_azimuth) # [N_obs, N_optdir]
        self.DM_rec_layers = self.config['DM']['NumberReconstructedLayers'] # [N_rec_layers]

        # HO WFS(s) parameters
        self.WFS_d_sub = self.config['sensor_HO']['SizeLenslets']
        self.WFS_n_sub = self.config['sensor_HO']['NumberLenslets']

        self.WFS_det_clock_rate = self.config['sensor_HO']['ClockRate'] # TODO: what is exactly the clock rate is?
        self.WFS_FOV = self.config['sensor_HO']['FieldOfView']
        self.WFS_RON = self.config['sensor_HO']['SigmaRON']
        self.WFS_wvl = self.GS_wvl
        self.WFS_psInMas   = self.config['sensor_HO']['PixelScale']
        self.WFS_spot_FWHM = self.config['sensor_HO']['SpotFWHM']
        self.WFS_excessive_factor = self.config['sensor_HO']['ExcessNoiseFactor']
        self.WFS_Nph = self.config['sensor_HO']['NumberPhotons']

        self.WFS_algorithm = self.config['sensor_HO']['Algorithm'].lower()
        self.WFS_algo_settings = self.config['sensor_HO']['WindowRadiusWCoG']
        self.WFS_type = self.config['sensor_HO']['WfsType'].upper()

        # Check WFS type
        self.is_SH      = (self.WFS_type == 'SHACK-HARTMANN')
        self.is_pyramid = (self.WFS_type == 'PYRAMID')

        # HO WFSs parameters stored in a dictionary       
        self.HOloop_rate  = self.config['RTC']['SensorFrameRate_HO'].flatten() # [Hz]
        self.HOloop_delay = self.config['RTC']['LoopDelaySteps_HO'].flatten() # [ms] (?)
        self.HOloop_gain  = self.config['RTC']['LoopGain_HO'].flatten()

        # Initialiaing the main optimizable parameters
        self.r0 = self.rad2arc * 0.976 * self.config['atmosphere']['Wavelength'] / self.config['atmosphere']['Seeing'] # [m]
        self.L0 = self.config['atmosphere']['L0'].flatten() # [m]
        
        self.r0_ = lambda : self.r0.abs().flatten() * self.airmass.flatten()**(-3.0/5.0)# [N_obs]
        # Layer heights stretched by the LGS cone, as P3's strechFactor; NGS heights leave them unchanged
        self.h_  = lambda: self.Cn2_heights / (1.0 - self.Cn2_heights/self.LGS_height) # [N_obs, N_layers]

        self.F   = torch.ones (self.N_src, self.N_wvl, device=self.device)
        self.bg  = torch.zeros(self.N_src, self.N_wvl, device=self.device)
        self.dx  = torch.zeros(self.N_src, self.N_wvl, device=self.device)
        self.dy  = torch.zeros(self.N_src, self.N_wvl, device=self.device)
        self.Jx  = torch.ones (self.N_src, device=self.device)
        self.Jy  = torch.ones (self.N_src, device=self.device)
        self.Jxy = torch.zeros(self.N_src, device=self.device)

        # One-dimensional wing of the PSF in the IMAGE frame (see WingKernel): applied by PSD2PSF only when use_wing is set. This block is
        # re-run on grid / wavelength updates, so the switch and the exponent are kept once set (the values are re-set by every forward call)
        self.use_wing   = getattr(self, 'use_wing', False)
        self.wing_power = getattr(self, 'wing_power', 1.0)                        # exponent of the wing MTF: 1 = Lorentzian wing
        self.wing_frac  = torch.zeros(self.N_src, self.N_wvl, device=self.device) # flux fraction in the wing, per wavelength
        self.wing_width = torch.ones (self.N_src, device=self.device) * 5.0       # half-width of the wing [detector px], achromatic
        self.wing_angle = torch.zeros(self.N_src, device=self.device)             # [deg], rotation of the wing axis from the image columns
        
        if self.PSD_include['Moffat']:
            self.amp   = torch.ones (self.N_src, device=self.device)*0.01 # Moffat amplitude [rad²]
            self.alpha = torch.ones (self.N_src, device=self.device)*0.1  # Moffat alpha [1/m]
            self.beta  = torch.ones (self.N_src, device=self.device)*2    # Moffat beta power law
            self.ratio = torch.ones (self.N_src, device=self.device)      # Moffat ellipticity
            self.b     = torch.zeros(self.N_src, device=self.device)      # background [rad²/m²]
            self.theta = torch.zeros(self.N_src, device=self.device)      # Moffat angle

        # if self.PSD_include['WFS noise'] or self.PSD_include['spatio-temporal'] or self.PSD_include ['aliasing']:
        self.dn  = torch.zeros(self.N_obs, device=self.device)
        # Half-width [deg] of the wind direction drift during the acquisition, wind_dir ± wind_dir_drift, [N_obs] or [N_obs, N_L].
        # It only acts when nTh > 1. The PSF is even in it, so its gradient vanishes at 0: start a fit from a non-zero value
        self.wind_dir_drift = torch.zeros(self.N_obs, device=self.device)

        self._init_error_terms()

        if self.AO_type is None:
            self._select_AO_correction()
        
        self.tomography = self.AO_type in ['LTAO', 'MCAO', 'GLAO']
        self.approx_noise_gain = self.tomography # Enable simplified noise gain computation for tomographic systems

        self.NoiseGain() # Compute the noise gain
              
        # Initialize index of refraction values
        self.IOR_src_wvl = self.n_air(self.wvl)
        self.IOR_wvl_atm = self.n_air(self.wvl_atm)
        self.IOR_GS_wvl  = self.n_air(self.GS_wvl) # GS_wvl may depend on the filter for SCAO or on LGS wavelength
 
 
    def _init_error_terms(self):
        '''
        Settings of the config-driven error terms, read from the TIPTOP entries [telescope] extraErrorNm / Exp / Min / Max (HO extra error,
        a generic error absorber for all sources like the Moffat term), extraErrorLoNm / LoExp / LoMin / LoMax and TechnicalFoV (LO extra
        error of the NGS directions), windPsdFile and [sensor_HO] addMcaoWFsensConeError. Only the entries present in the config are
        initialized and nothing is computed unless the corresponding PSD_include entries are enabled. As P3's getPSDatNGSpositions, when the
        config has a [sources_LO] section its N_LO directions are assumed to be the trailing N_LO sources.
        '''
        get    = lambda section, key, default=None: self.config[section][key] if self.config[section].get(key) is not None else default
        scalar = lambda x: float(torch.as_tensor(x).flatten()[0])
        full   = lambda n, x: torch.full((n,), x, device=self.device, dtype=self.dtype)

        rms_HO, exp_HO = scalar(get('telescope', 'extraErrorNm', 0.0)), scalar(get('telescope', 'extraErrorExp', -2.0))
        self.extra_error_shape    = (exp_HO, scalar(get('telescope', 'extraErrorMin', 0.0)), scalar(get('telescope', 'extraErrorMax', 0.0)))
        self.extra_error_LO_shape = (scalar(get('telescope', 'extraErrorLoExp', exp_HO)), scalar(get('telescope', 'extraErrorLoMin', 0.0)), scalar(get('telescope', 'extraErrorLoMax', 0.0)))
        self.extra_error_nm = full(self.N_src, rms_HO) if rms_HO > 0 or self.PSD_include['extra error'] else None # [N_src], an optimizable parameter

        rms_LO = self.make_tensor(get('telescope', 'extraErrorLoNm', rms_HO)).flatten() # one value, or [center, edge] of the technical field
        N_LO = min(self.config['sources_LO']['Zenith'].numel(), self.N_src) if 'sources_LO' in self.config and rms_LO.sum().item() >= 0 else 0
        self.extra_error_LO_nm = None
        
        if N_LO > 0 and rms_LO.sum().item() > 0:
            if rms_LO.numel() == 2: # linear interpolation between the field center and the edge of the technical field (as P3's extraErrorLoPSD)
                FoV = scalar(get('telescope', 'TechnicalFoV', 0.0))
                LO_zenith = self.make_tensor(self.config['sources_LO']['Zenith']).flatten()[:N_LO] # [arcsec]
                rms_LO = rms_LO[0] + (rms_LO[1]-rms_LO[0]) * ((LO_zenith / (0.5*FoV)).clamp(0, 1) if FoV > 0 else torch.zeros_like(LO_zenith))
            
            elif rms_LO.numel() != 1:
                raise ValueError('extraErrorLoNm must be a scalar or [center, edge] values')
            
            self.extra_error_LO_nm = torch.cat([full(self.N_src-N_LO, 0.0), rms_LO.expand(N_LO).to(device=self.device, dtype=self.dtype)]) # [N_src]
            
            if self.extra_error_nm is not None: # the NGS directions get the LO extra error instead of the HO one (as P3)
                self.extra_error_nm = torch.cat([self.extra_error_nm[:self.N_src-N_LO], full(N_LO, 0.0)])

        self.add_MCAO_cone = bool(get('sensor_HO', 'addMcaoWFsensConeError', False))
        wind_file = get('telescope', 'windPsdFile')
        self.wind_PSD_file = str(wind_file) if isinstance(wind_file, (str, Path)) and str(wind_file) not in ('', '0') else None


    def _enable_configured_terms(self):
        ''' Enable the error terms the config asks for, as P3 adds them automatically; the tilt and focus filters are left to the caller '''
        self.PSD_include['extra error']    = self.extra_error_nm is not None
        self.PSD_include['LO extra error'] = self.extra_error_LO_nm is not None
        self.PSD_include['wind shake']     = self.wind_PSD_file is not None
        self.PSD_include['cone effect']    = bool(self.is_LGS.all() and (self.N_GS == 1 or (self.tomography and self.add_MCAO_cone)))
    
    
    def _select_AO_correction(self):
        use_LGS = self.is_LGS.all() # LGS stars must be at finite altitude
        multiple_GS = self.GS_dirs_x.size(-1) > 1
        multiple_DM = self.N_DM > 1
        
        if not use_LGS:
            self.AO_type = 'SCAO'
        elif multiple_GS:
            self.AO_type = 'MCAO' if multiple_DM else ('LTAO' if self.DM_rec_layers > 1 else 'GLAO')
        else:
            self.AO_type = 'SLAO'
            
        if not multiple_DM:
            self.N_DM = 1 # Just to make sure
            

    def _FFTAutoCorr(self, x: torch.Tensor) -> torch.Tensor:
        return fft.fftshift( fft.ifft2(fft.fft2(x, dim=(-2,-1)).abs()**2, dim=(-2,-1)), dim=(-2,-1) ) / x.shape[-2] / x.shape[-1]

    
    def _gen_grid(self, N: int) -> tuple[torch.Tensor, torch.Tensor]:
        factor = 0.5*(1-N%2)
        return torch.meshgrid(*[torch.linspace(-N//2+N%2+factor, N//2-factor, N, device=self.device)]*2, indexing = 'ij')
    

    def _stabilize(self, tensor: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
        ''' Increases the numerical stability by replacing near-zero values with eps '''
        tensor[tensor.abs() < eps] = eps
        return tensor


    @staticmethod
    def _map_tensors(obj, fn):
        ''' Applies fn to every tensor in obj, rebuilding containers only if something has changed (keeps aliasing otherwise) '''
        if torch.is_tensor(obj):
            return fn(obj)
        if isinstance(obj, dict):
            new = {k: TipTorch._map_tensors(v, fn) for k, v in obj.items()}
            changed = any(new[k] is not v for k, v in obj.items())
        elif type(obj) in (list, tuple):
            new = type(obj)(TipTorch._map_tensors(v, fn) for v in obj)
            changed = any(a is not b for a, b in zip(new, obj))
        else:
            return obj
        return new if changed else obj


    def _convert_tensors(self, fn) -> None:
        ''' Applies fn to all tensors held by the model: attributes (incl. nested ones, e.g. in config), parameters, buffers, and submodules '''
        self._spectra_cache = {}
        for name, attr in list(vars(self).items()):
            if not name.startswith('_'):
                new_attr = self._map_tensors(attr, fn)
                if new_attr is not attr:
                    setattr(self, name, new_attr)
        self._apply(fn)


    def _to_double(self) -> None:
        self.to(torch.float64)


    def _to_float(self) -> None:
        self.to(torch.float32)


    def PistonFilter(self, f: torch.Tensor) -> torch.Tensor:
        ''' Piston mode filter for PSD '''
        x = torch.pi * self.D * f
        R = torch.special.bessel_j1(x) / x
        piston_filter = 1.0 - 4.0 * R.pow(2)
        piston_filter[..., self.nOtf_AO//2, self.nOtf_AO//2] *= 1-self.nOtf_AO % 2
        return self._stabilize(piston_filter)


    def _bessel_j3(self, x: torch.Tensor) -> torch.Tensor:
        ''' Compute 2*J₃(x)/x using Bessel recurrence relations '''
        # Use recurrence: J_{n+1}(x) = (2n/x)*J_n(x) - J_{n-1}(x)
        # For numerical stability near x=0, use analytical limits
        
        # Protect against division by zero
        x_safe = self._stabilize(x.clone(), eps=1e-6)        

        j0 = torch.special.bessel_j0(x_safe)
        j1 = torch.special.bessel_j1(x_safe)
        j2 = (2.0 / x_safe) * j1 - j0 # Compute J₂(x) using recurrence: J₂(x) = (2/x)*J₁(x) - J₀(x)
        j3 = (4.0 / x_safe) * j2 - j1 # Compute J₃(x) using recurrence: J₃(x) = (4/x)*J₂(x) - J₁(x)
        
        # Compute 2*J₃(x)/x
        result = 2.0 * j3 / x_safe
        
        # Apply analytical limit: 2*J₃(x)/x → 0 as x → 0
        result = torch.where(x < 1e-6, torch.zeros_like(result), result)
        return result


    def _spatial_filters(self, k: torch.Tensor, D: torch.Tensor | float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        ''' Piston and tilt rejection filters [Sasiela 93] on an arbitrary spatial frequency grid k [1/m] for an aperture of diameter D [m] '''
        x  = torch.pi * (self.D if D is None else D) * k
        x_ = torch.where(x < 1e-6, torch.ones_like(x), x) # protect the origin, analytical limits are restored below
        J1 = torch.special.bessel_j1(x_)
        J2 = 2.0 * J1 / x_ - torch.special.bessel_j0(x_)

        j1_term = torch.where(x < 1e-6, torch.ones_like(x),  2.0 * J1 / x_)
        j2_term = torch.where(x < 1e-6, torch.zeros_like(x), 4.0 * J2 / x_)

        piston_filter = (1.0 - j1_term.pow(2)).clamp_min(0.0)
        tilt_filter   = (piston_filter - j2_term.pow(2)).clamp_min(0.0)
        return piston_filter, tilt_filter


    def _half_sum(self, PSD_half: torch.Tensor) -> torch.Tensor:
        ''' Sum over the full grid of a half-grid PSD (the last column of the half grid is the DC column, counted once), [..., 1, 1] '''
        return 2*PSD_half.sum(dim=(-2,-1), keepdim=True) - PSD_half[..., :, -1:].sum(dim=-2, keepdim=True)


    def _cached(self, key, compute):
        ''' Grid-dependent constant tensors, computed once per grid (the cache is reset by InitGrids and on dtype conversion) '''
        if key not in self._spectra_cache:
            self._spectra_cache[key] = compute()
            
        return self._spectra_cache[key]


    def _unit_spectrum(self, name: str) -> torch.Tensor:
        '''
        Unit-power spectral shape of an error term set by an RMS value, on the half grid ([1, nOtf_y, nOtf_x], or the AO grid for the wind shake):
        'extra error' / 'LO extra error' are piston-filtered power laws k^exponent restricted to [k_min, k_max] (k_max <= 0: no upper cut), as
        P3's extraErrorPSD; 'focus error' is the global focus spectrum 1 - 3 (2 J3(pi D k) / (pi D k))² (P3's FocusFilter); 'wind shake' is the
        tilt-shaped, piston-filtered spectrum of the AO-corrected area (P3's windShakePSD). The RMS² [nm²] times the shape is the PSD in nm².
        '''
        def compute():
            if name in ('extra error', 'LO extra error'):
                exponent, k_min, k_max = self.extra_error_shape if name == 'extra error' else self.extra_error_LO_shape
                k = self.k
                in_band = k >= k_min
                
                if k_max > 0:
                    in_band &= k <= k_max
                    
                PSD = k.pow(exponent) * self._spatial_filters(k)[0] * in_band
                PSD[..., self.nOtf_y//2, self.nOtf_x-1] = 0.0 # remove the DC component
                
            elif name == 'focus error':
                PSD = 1.0 - 3.0 * self._bessel_j3(torch.pi * self.D * self.k).pow(2)
                
            elif name == 'wind shake':
                piston_filter, tilt_filter = self._spatial_filters(self.k_AO)
                PSD = (1.0 - tilt_filter) * piston_filter * self.mask_corrected_AO
                
            return PSD / self._half_sum(PSD)
        
        return self._cached(name, compute)


    def _wind_shake_power(self) -> torch.Tensor:
        ''' Tip + tilt vibration power [nm²] left by the HO loop, [N_obs]: the temporal PSDs vibration_PSD [3, N_f] filtered by the rejection transfer function and integrated (as P3's windShakePSD) '''
        rate, gain, delay = self.HOloop_rate.view(-1,1), self.HOloop_gain.view(-1,1), self.HOloop_delay.view(-1,1) # [N_obs, 1]
        f = 0.1 + (0.5*rate - 0.1) * torch.linspace(0, 1, int(5*rate.max().item()), device=self.device, dtype=rate.dtype) # [N_obs, N_f]
        
        PSD_t = self._interp1d(f, self.vibration_PSD[0], self.vibration_PSD[1]) + self._interp1d(f, self.vibration_PSD[0], self.vibration_PSD[2])
        _, rtfInt, _, _ = self.TransferFunctions(f, 1.0/rate, delay, gain)
        
        return ( (rtfInt.pow(2) * PSD_t).sum(dim=-1) * (f[:,1]-f[:,0]) ).abs()


    def _PSD_norm(self) -> torch.Tensor:
        ''' PSDs are computed in [rad²/m²] at the atmosphere wavelength; this factor converts them to [nm²] of OPD per PSD pixel '''
        return (self.dk * self.wvl_atm * 1e9 / (2*torch.pi))**2


    @staticmethod
    def _interp1d(x_new: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        ''' Batched linear interpolation of the (x, y) samples at x_new, zero outside of the sampled range (as np.interp with left=right=0) '''
        ids = torch.searchsorted(x, x_new.contiguous()).clamp(1, x.numel()-1)
        x0, x1, y0, y1 = x[ids-1], x[ids], y[ids-1], y[ids]
        y_new = y0 + (y1-y0) * (x_new-x0) / (x1-x0)
        return torch.where((x_new < x[0]) | (x_new > x[-1]), torch.zeros_like(y_new), y_new)


    def _to_odd(self, x: float) -> int:
        odd = int(np.round(x))
        if odd % 2 == 0:
            odd += 1 if x > odd else -1

        return odd
        
    
    def InitGrids(self) -> None:
        # Initialize grids and frequency masks for PSD and OTF computations.
        # for all generated PSDs and OTFs within one PSF batch the sampling must be the same
        pixels_per_l_D = self.wvl * self.rad2mas / (self.psInMas * self.D)
        self.sampling_factor = 2./pixels_per_l_D * self.oversampling
        self.sampling_factor /= torch.clamp(self.sampling_factor, 0, 1).min()
        
        self.sampling = self.sampling_factor * pixels_per_l_D # must be = 2*oversampling
        self.sampling_min = self.sampling.min().item()

        self.nOtfs = self._to_odd_arr(self.N_pix * self.sampling_factor.cpu().numpy().flatten())
        self.nOtf  = self.nOtfs.max().item()
        self.dk    = 1.0 / self.D / self.sampling.min() # PSD spatial frequency step
                
        # Initialize PSD spatial frequencies in the image frame (the pupil-frame grids of the rotated pupil are derived from them on demand
        # by kxy_AO_(), see the class docstring)
        self.kx, self.ky = self._gen_grid(self.nOtf)
        self.kx, self.ky = self.kx * self.dk, self.ky * self.dk
        self.k2 = self.kx**2 + self.ky**2
        self.k = torch.sqrt(self.k2)

        self._stabilize(self.kx) #TODO: is it necessary, after all?
        self._stabilize(self.ky)
        self._stabilize(self.k2)
        self._stabilize(self.k )

        if self.PSD_include['Moffat']:
            self.kxy = self.kx * self.ky
            self.kx2 = self.kx.pow(2)
            self.ky2 = self.ky.pow(2)
            
            self._stabilize(self.kxy)
            self._stabilize(self.kx2)
            self._stabilize(self.ky2)

        # Compute the frequency mask for the AO corrected and uncorrected frequency regions
        # rim_width = self.make_tensor(0.1)
        self.mask_corrected = torch.zeros_like(self.k2).int()
        self.mask_corrected [self.k2 <= self.kc**2] = 1.0
        self.mask = 1.0 - self.mask_corrected
        
        mask_slice = self.mask_corrected[self.mask_corrected.shape[0]//2, :].tolist()
        first_one = mask_slice.index(1)
        last_one = len(mask_slice)-mask_slice[::-1].index(1)-1 # detect the borders of the correction area
        self.nOtf_AO = last_one-first_one+1

        self.dk = 2*self.kc / self.nOtf_AO # Correct dk to account for the actual sampling

        corrected_ROI = (slice(first_one, last_one+1), slice(first_one, last_one+1))
        self.mask_corrected_AO = pdims(self.mask_corrected[corrected_ROI], -1)
        self.mask = pdims( self.mask, -1 )

        # Computing frequency grids for the AO corrected and uncorrected regions
        self.kx_AO = pdims( self.kx[corrected_ROI], -1 )
        self.ky_AO = pdims( self.ky[corrected_ROI], -1 )
        self.k_AO  = pdims( self.k [corrected_ROI], -1 )
        self.k2_AO = pdims( self.k2[corrected_ROI], -1 )

        self.kx = pdims( self.kx, -1 )
        self.ky = pdims( self.ky, -1 )
        self.k  = pdims( self.k,  -1 )
        self.k2 = pdims( self.k2, -1 )

        # Cut dimensions by half to optimize the computations by utilizing the PSD symmetry
        self.nOtf_AO_x = self.nOtf_AO // 2 + self.nOtf_AO % 2
        self.nOtf_AO_y = self.nOtf_AO
    
        self.nOtf_x = self.nOtf // 2 + self.nOtf % 2
        self.nOtf_y = self.nOtf

        self.kx_AO = self.kx_AO[..., :self.nOtf_AO_y, :self.nOtf_AO_x]
        self.ky_AO = self.ky_AO[..., :self.nOtf_AO_y, :self.nOtf_AO_x]
        self.k_AO  = self.k_AO [..., :self.nOtf_AO_y, :self.nOtf_AO_x]
        self.k2_AO = self.k2_AO[..., :self.nOtf_AO_y, :self.nOtf_AO_x]
        self.mask_corrected_AO = self.mask_corrected_AO[..., :self.nOtf_AO_y, :self.nOtf_AO_x]
        
        self.kx = self.kx[..., :self.nOtf_y, :self.nOtf_x]
        self.ky = self.ky[..., :self.nOtf_y, :self.nOtf_x]
        self.k  = self.k [..., :self.nOtf_y, :self.nOtf_x]
        self.k2 = self.k2[..., :self.nOtf_y, :self.nOtf_x]
        self.mask = self.mask[..., :self.nOtf_y, :self.nOtf_x]

        if self.PSD_include['Moffat']:
            self.kx2_AO = pdims( self.kx2[corrected_ROI], -1 )[..., :self.nOtf_AO_y, :self.nOtf_AO_x]
            self.ky2_AO = pdims( self.ky2[corrected_ROI], -1 )[..., :self.nOtf_AO_y, :self.nOtf_AO_x]
            self.kxy_AO = pdims( self.kxy[corrected_ROI], -1 )[..., :self.nOtf_AO_y, :self.nOtf_AO_x]

            self.kxy = pdims( self.kxy, -1 )[..., :self.nOtf_y, :self.nOtf_x]
            self.kx2 = pdims( self.kx2, -1 )[..., :self.nOtf_y, :self.nOtf_x]
            self.ky2 = pdims( self.ky2, -1 )[..., :self.nOtf_y, :self.nOtf_x]

        if self.PSD_include['aliasing']:
            # Spatial freq comb samples count involved in aliasing PSD calculation
            n_times_y = int(np.ceil(self.nOtf/self.nOtf_AO/2))
            n_times_x = int(np.ceil(self.nOtf/self.nOtf_AO/2)) - 1
            # -1 since when using a halfed frequency grid, the last sample is truncated (from both sides)
            
            # Limit aliasing combs count to save on memory and computation time
            n_times_limit = 4
            n_times_x, n_times_y = min(max(2, n_times_y), n_times_limit), min(max(2, n_times_x), n_times_limit)
            ids = np.array( [[i, j] for i in range(-n_times_y+1, n_times_y) for j in range(-n_times_x+1, n_times_x) if i != 0 or j != 0] )
            
            # The 0-th dimension is used to store the shifted spatial frequency
            # This is thing is 4D: [N_combs, N_obs or 1, kx, ky]
            m = self.make_tensor(ids[:,0])
            n = self.make_tensor(ids[:,1])
            self.N_combs = m.shape[0]

            # Comb shifts along the WFS axes [N_combs, 1, 1, 1], applied to the pupil-frame grids by _aliasing_combs
            self.comb_m = pdims(m/self.WFS_d_sub, 3)
            self.comb_n = pdims(n/self.WFS_d_sub, 3)
            self.km = self.kx_AO.unsqueeze(0) - self.comb_m
            self.kn = self.ky_AO.unsqueeze(0) - self.comb_n

        # Initialize OTF frequencies
        UV_range = torch.linspace(-1, 1, self.nOtf, device=self.device)
        self.U, self.V = torch.meshgrid(UV_range, UV_range, indexing = 'ij')
        self.U, self.V = pdims(self.U, -2), pdims(self.V, -2)
        
        # U, V span [-1, 1] over the OTF grid, i.e. up to the Nyquist frequency sampling*D/(2*wvl) [cycles/rad]; converted to [cycles/mas]²
        self.u_max = (self.sampling * self.D / self.wvl / self.rad2mas / 2)**2
        
        # self.center_aligner = torch.exp( 1j * torch.pi * (self.U + self.V) * (1 - self.N_pix%2))

        # Since only half of PSD is used, the padding of AO-corrected PSD component is done only on the left
        self.PSD_padder = torch.nn.ZeroPad2d( (a:=((self.nOtf-self.nOtf_AO)//2), 0, a, a) ) # pad_left, pad_right, pad_top, pad_bottom

        self.piston_filter = self.PistonFilter(self.k_AO)
        self._spectra_cache = {} # grid-dependent spectral shapes are rebuilt on demand

        # To avoid initializing it without a need
        if self.PSD_include['aliasing']:
            self.PR = self.PistonFilter(torch.hypot(self.km, self.kn)) # piston filter for aliased spatial frequencies


    def _pupil_rotation(self) -> tuple[torch.Tensor, torch.Tensor]:
        ''' Cosine and sine of the pupil angle(s), [1 or N_obs, 1, 1] '''
        angle = torch.deg2rad(self.pupil_angle).view(-1, 1, 1)
        return torch.cos(angle), torch.sin(angle)


    def kxy_AO_(self) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        Spatial frequency components (kx, ky) of the AO half grid in the pupil / DM / WFS frame, [1 or N_obs, nOtf_AO_y, nOtf_AO_x]: the
        image-frame grid rotated by -pupil_angle, so that the functions of the AO system defined in its own frame (WFS sampling, aliasing
        combs, DM optimization directions, Moffat PSD) appear rotated by +pupil_angle (counter-clockwise) in the image. Without rotation
        the stored image-frame grids are returned as they are; otherwise the rotated AO grids are built once per grid and set of angles
        (the grid cache, reset by InitGrids and by the angles of a new config), only these small half grids are kept per observation.
        k and k2 are rotation invariant and need no counterpart.
        '''
        if not self.rotate_pupil:
            return self.kx_AO, self.ky_AO

        def compute():
            cos, sin = self._pupil_rotation()
            return self.kx_AO*cos + self.ky_AO*sin, self.ky_AO*cos - self.kx_AO*sin

        return self._cached(('kxy_AO', self._pupil_angle_key), compute)


    def _Moffat_grids(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ''' (kx², kx·ky, ky²) of the AO half grid in the pupil frame for the Moffat PSD '''
        if not self.rotate_pupil:
            return self.kx2_AO, self.kxy_AO, self.ky2_AO

        def compute():
            kx, ky = self.kxy_AO_()
            return kx*kx, kx*ky, ky*ky

        return self._cached(('Moffat grids', self._pupil_angle_key), compute)


    def _aliasing_combs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ''' Aliased spatial frequencies (km, kn) [N_combs, 1 or N_obs, nOtf_AO_y, nOtf_AO_x], shifted along the WFS axes, and their piston filter '''
        if not self.rotate_pupil:
            return self.km, self.kn, self.PR

        def compute():
            kx, ky = self.kxy_AO_()
            km, kn = kx.unsqueeze(0) - self.comb_m, ky.unsqueeze(0) - self.comb_n
            return km, kn, self.PistonFilter(torch.hypot(km, kn))

        return self._cached(('aliasing combs', self._pupil_angle_key), compute)


    def Phase2OTF(self, phase: torch.Tensor | None) -> torch.Tensor:
        '''
        Compute OTF from a phase screen [H, W], [N_obs, H, W] (one pupil per observation) or [N_src, N_wvl, H, W].
        All phase screens are sampled equally for all wavelengths (re-scaling happens later when PSFs are computed)
        However, phase screens might be different for each wavelength.
        '''
        phase_padded = self.pupil_padder(phase)

        fl_even = self.nOtf % 2 == 0 and phase_padded.shape[-1] % 2 == 0 # TODO: what's the hell is this?
        phase_padded = phase_padded[..., :-1, :-1] if fl_even else phase_padded # to center-align if number of pixels is even

        OTF = self._FFTAutoCorr(phase_padded)
        OTF = OTF.view(1, 1, *OTF.shape) if OTF.ndim == 2 else OTF.unsqueeze(1) if OTF.ndim == 3 else OTF
        
        # PyTorch doesn't support interpolation of complex tensors yet
        OTF_ = self.interp(OTF.real, self.sampling_min) + self.interp(OTF.imag, self.sampling_min)*1j
        return OTF_ / OTF_.abs().amax(dim=(-2,-1), keepdim=True) # normalize OTF
    
    
    def ComputeStaticOTF(self, phase: torch.Tensor | None = None) -> torch.Tensor:
        if phase is not None:
            self.OTF_static = self.Phase2OTF(phase)
        # When phase is None, just keep the current OTF_static as-is
        return self.OTF_static


    def Update(self, config: dict | None = None, grids: bool = False, pupils: bool = False, tomography: bool = False, update_static_OTF: bool = True) -> None:
        # Update the model with a new configuration. To ensure optimal performance, different
        # components can be updated independently when needed. By default, only values are updated
        
        if config is not None:
            self.config = config
        
        # Set the model parameters to the ones defined in the config file
        self.InitValues()
        
        # Update grids used for computation of PSDs and OTFs. This is necessary when the sampling has changed.
        if grids:
            self.InitGrids()
        
        # Update pupil and apodizer masks
        if pupils:
            self.InitPupils()
        elif self.pupil is not None:
            # The pupil angles of a new config re-orient the already loaded masks (e.g. a new batch of observations)
            pupils_stale = self.pupil_angle.tolist() != self._oriented_angles
            if pupils_stale:
                self._orient_pupils()

            if grids or pupils_stale:
                if update_static_OTF:
                    # Grid-dependent static OTF must be refreshed when pupil data did not change.
                    self.UpdateStaticOTF()
                else:
                    # Caller will set OTF_static via ComputeStaticOTF(phase); skip the expensive
                    # pupil FFT but still refresh the pupil_padder which depends on sampling_min.
                    phase_size = self.pupil.shape[-1]
                    self.pupil_padder = torch.nn.ZeroPad2d( int(round(phase_size * self.sampling_min / 2 - phase_size / 2)) )
        
        # If the number of sources have changed, reinitialize the tomography projector
        if (self.tomography and tomography) or (self.tomography and grids):
            if self.on_axis:
                # Tomographic src to DM projector for on-axis is just an identity matrix
                self.P_beta_DM = torch.ones([self.N_src, self.nOtf_AO_y, self.nOtf_AO_x, 1, self.N_DM], dtype=torch.complex64, device=self.device) * pdims(self.mask_corrected_AO, 2)
                # TODO: comment
                self.P_opt = torch.ones([self.N_obs, self.nOtf_AO_y, self.nOtf_AO_x, 1, self.N_L], dtype=torch.complex64, device=self.device)
            else:
                self.OptimalDMProjector()
                self.DMProjector()

        # Some tensors are created in single precision regardless of the model precision
        self.to(self.dtype)


    def _initialize_PSDs_settings(self, PSD_include: dict | None = None) -> None:
        # The full list of all PSD components supported by the model
        PSD_entries_all = [
            'fitting',
            'WFS noise',
            'spatio-temporal',
            'aliasing',
            'chromatism',
            'Moffat',
            'diff. refract',
            'cone effect',    # cone effect of the LGS(s): focal anisoplanatism (single LGS) or the volume not sensed by the LGS WFSs (tomography)
            'extra error',    # [telescope] extraErrorNm power-law PSD, a generic error absorber with the RMS extra_error_nm (like the Moffat term)
            # LO terms, applied after the core PSD and updatable on their own (ComputePSD(update_LO_only=True)):
            'LO extra error', # [telescope] extraErrorLoNm power-law PSD of the NGS directions
            'wind shake',     # [telescope] windPsdFile temporal PSD filtered by the HO loop (no separate LO loop)
            'tilt filter',    # tip/tilt rejection filter applied to the PSD when a separate LO loop corrects tip/tilt
            'focus error'     # residual global focus error with the RMS focus_error_nm
        ]
        # Excluded unless enabled explicitly or by the config entries (see _enable_configured_terms)
        PSD_entries_off = ['Moffat', 'cone effect', 'extra error', 'LO extra error', 'wind shake', 'tilt filter', 'focus error']

        if PSD_include is not None:
            # One can select which error sources to include in the simulation
            self.PSD_include = PSD_include
            # Fill missing ones
            for key in PSD_entries_all:
                if key not in self.PSD_include: self.PSD_include[key] = False
        else:
            # Otherwise, all error contributors are simulated except for the Moffat PSD and the add-ons
            self.PSD_include = { key: key not in PSD_entries_off for key in PSD_entries_all }
        
        if not self.PSD_include['fitting']:
            warnings.warn('The fitting PSD must be always be enabled. Setting it on')
        
        # Otherwise, the model won't work at all
        self.PSD_include['fitting'] = True
        
        #TODO: an open-loop case


    def __init__(self,
                 AO_config: dict,
                 AO_type: str | None = None,
                 pupil = None,
                 PSD_include: dict | None = None,
                 retain_PSDs: bool = False,
                 norm_regime: str = 'sum',
                 device: torch.device = torch.device('cpu'),
                 oversampling: int = 1,
                 dtype: torch.dtype = torch.float32,
                 nTh: int = 0):
        
        super().__init__()

        self.device = device
        self.oversampling = oversampling
        # The pupil is a [H, W] mask in the reference orientation (rotated by TipTorch when the pupil angle is not 0), or a generator
        # angle_deg -> [H, W] mask called per observation angle (e.g. an analytic pupil, see PupilVLT)
        self._pupil_generator = pupil if callable(pupil) else None
        self.pupil = None if callable(pupil) else pupil
        self._oriented_angles = None
        self.retain_PSDs = retain_PSDs
        self.dtype = dtype
        self.is_float = dtype in (torch.float32, torch.complex64)
               
        # TODO: automatic AO correction type selection
        self.AO_type = AO_type

        # Useful lambda functions
        self.r0_new = lambda r0, lmbd, lmbd0: r0*(lmbd/lmbd0).pow(6/5)
        self.interp = lambda x, sampling: interpolate(x, size=(self.nOtf,self.nOtf), mode='bilinear', align_corners=False) * sampling**2
        self.make_tensor = lambda x: torch.as_tensor(x, device=self.device, dtype=self.dtype) if type(x) is not torch.Tensor else x
        self._to_odd_arr = lambda arr: np.vectorize(self._to_odd)(arr)

        self._initialize_PSDs_settings(PSD_include)

        # Initialize constants
        self.mas2arc  = self.make_tensor(1e-3) #TODO: do we need it, though?
        self.rad2mas  = self.make_tensor(3600 * 180 * 1000 / torch.pi)
        self.rad2arc  = self.make_tensor(self.rad2mas / 1000)
        self.cte = self.make_tensor( (24*spc.gamma(6/5)/5)**(5/6)*(spc.gamma(11/6)**2/(2*np.pi**(11/3))) )
        # The characteristic function of Gaussian image motion is
        # exp[-(2*pi)^2 * sigma^2 * pupil_separation^2 / 2].
        self.jitter_norm_fact = self.make_tensor(2 * torch.pi)**2

        self.n_air = AirRefractiveIndexCalculator(device=self.device, dtype=self.dtype)

        if self.device.type == 'cuda':
            self.start = torch.cuda.Event(enable_timing=True)
            self.end   = torch.cuda.Event(enable_timing=True)

        # PSF normalization regimes
        self.norm_regime = norm_regime
        self.norm_scale  = self.make_tensor(1.0)
        
        if    self.norm_regime == 'sum': self.normalizer = torch.sum
        elif  self.norm_regime == 'max': self.normalizer = torch.amax
        else: self.normalizer = lambda x, dim, keepdim: self.norm_scale # no normalization

        # Number of wind directions sampled within the wind direction drift by the controller (0 = fixed wind direction)
        self.nTh = nTh
        if self.nTh > 0:
            # Gauss-Legendre nodes on [-1, 1] and weights summing to 1: the average over the drift is the weighted sum over its nodes
            nodes, weights = np.polynomial.legendre.leggauss(self.nTh)
            self.wind_drift_nodes, self.wind_drift_weights = self.make_tensor(nodes), self.make_tensor(weights/2) # [nTh]
        
        # Piston filters
        self.piston_filter = None # piston mode filter in the AO-corrected freqs domain
        self.apodizer = None
        self.PR = None # piston mode filter in aliased freqs domain

        # LO terms: the wind-shake temporal PSD is loaded with the pupils, the residual focus RMS is set by the caller (e.g. an LO loop)
        self.vibration_PSD = None
        self.focus_error_nm = None
        self._spectra_cache = {}

        # Read the config data and initialize the AO system
        self.Update(
            config = AO_config,
            grids  = True,
            pupils = True,
            tomography = True # Try updating the tomographic reconstructors. If AO is not tomographic, the model will figure it out and skip the update
        )

        if PSD_include is None: # unless chosen explicitly, the config-driven error terms follow the config (P3 behaviour)
            self._enable_configured_terms()
        

    def DMProjector(self):
        """ Projects correction in the direction of science target(s). Must be updated only when target coordinates were changed """
        
        kx = pdims(self.kx_AO, 1) # [1, nOtf_AO, nOtf_AO, 1]
        ky = pdims(self.ky_AO, 1) # [1, nOtf_AO, nOtf_AO, 1]
        h_DM = self.h_DM.view(1, 1, 1, self.N_DM) # [N_obs, 1, 1, N_DM]
        
        N_src_ = len(self.src_dirs_x.flatten())
        
        beta_x = self.src_dirs_x.view(N_src_, 1, 1, 1)
        beta_y = self.src_dirs_y.view(N_src_, 1, 1, 1)
    
        f = (beta_x*kx + beta_y*ky) * self.mask_corrected_AO.unsqueeze(-1)

        self.P_beta_DM = torch.exp( 2j*torch.pi*h_DM * f ).unsqueeze(-2) # [N_src, nOtf_AO, nOtf_AO, 1, N_L]
        # TODO: support multiple DMs?


    def OptimalDMProjector(self):
        h_dm = self.h_DM.view(1, 1, 1, self.N_DM)
        h = self.h_().view(self.N_obs, 1, 1, 1, self.N_L)
        
        opt_w = self.DM_opt_weight.view(self.N_obs, 1, 1, self.N_optdir, 1, 1)

        theta_x = self.DM_opt_dir_x.view(self.N_obs, 1, 1, self.N_optdir)
        theta_y = self.DM_opt_dir_y.view(self.N_obs, 1, 1, self.N_optdir)

        kx, ky = self.kxy_AO_() # the optimization directions are in the DM frame
        f = theta_x * pdims(kx, 1) + theta_y * pdims(ky, 1)
        P_L = torch.exp( 2j*torch.pi*h * pdims(f,1) ).unsqueeze(-2) # [N_obs, nOtf_AO, nOtf_AO, N_optdir, 1, N_L]
        
        if self.AO_type == 'LTAO':
            self.P_opt = (P_L * opt_w).sum(dim=3) # [N_obs, nOtf_AO, nOtf_AO, 1, N_L]
            return
        
        mask = self.mask_corrected_AO.view(1, self.nOtf_AO_y, self.nOtf_AO_x, 1)
        P_DM   = torch.exp( 2j*torch.pi*h_dm * pdims(f*mask,1) ).unsqueeze(-2) # [N_obs, nOtf_AO, nOtf_AO, N_optdir, 1, N_DM]
        P_DM_t = torch.conj( P_DM.permute(0, 1, 2, 3, 5, 4) )    # [N_obs, nOtf_AO, nOtf_AO, N_optdir, N_DM, 1]

        mat1   = ((P_DM_t @ P_L)  * opt_w).sum(dim=3)  # [N_obs, nOtf_AO, nOtf_AO, N_DM, N_L]
        to_inv = ((P_DM_t @ P_DM) * opt_w).sum(dim=3)  # [N_obs, nOtf_AO, nOtf_AO, N_DM, N_DM]
        
        # to_inv is rank-deficient (e.g. more DMs than optimization directions resolve at low frequencies), so it must be
        # regularized by truncation. lstsq is avoided: its rank cut-off is driver-dependent (gelsy on CPU, none at all on CUDA)
        self.P_opt = torch.linalg.pinv(to_inv, rcond=1e-2) @ mat1 # [N_obs, nOtf_AO, nOtf_AO, N_DM, N_L]


    def TransferFunctions(self, freq: torch.Tensor, Ts: torch.Tensor, delay: torch.Tensor, loop_gain: torch.Tensor):
        # z = torch.exp(2j*torch.pi*freq*Ts) # no minus according to Sanchit
        # hInt = loopGain / self._stabilize(1.0 - 1.0/z, 1e-12)
        # rtfInt = 1. / self._stabilize(1+hInt*z**(-delay), 1e-12) # Rejection transfer function
        # atfInt = self._stabilize(hInt * z**(-delay) * rtfInt)    # Aliasing transfer function
        # ntfInt = self._stabilize(atfInt / z, 1e-12)              # Noise transfer function
        # ntfInt = self._stabilize(hInt * z**(-delay-1)) # according to Sanchit, but it maybe wrong

        z = torch.exp(2j*torch.pi*freq*Ts) # no minus according to Sanchit
        # z == 1 at freq == 0, which would make this denominator vanish; stabilize to avoid a div-by-zero / NaN gradient
        hInt = loop_gain / self._stabilize(1.0 - 1.0/z, 1e-12)
        z_pow_delay = z.pow(-delay) # cached: reused below instead of being recomputed
        rtfInt = 1. / self._stabilize(1 + hInt*z_pow_delay, 1e-12)  # Rejection transfer function
        atfInt = hInt * z_pow_delay * rtfInt                        # Aliasing transfer function
        ntfInt = atfInt / z                                         # Noise transfer function
        # ntfInt = hInt * z**(-delay-1) # according to Sanchit, but it maybe wrong
                    
        return hInt, rtfInt, atfInt, ntfInt


    def NoiseGain(self, nF: int = 1000):
        if not self.approx_noise_gain:
            Ts = min_2d(1.0 / self.HOloop_rate) # sampling time, [N_obs, 1]
            delay    = min_2d(self.HOloop_delay) # latency between the measurement and the correction
            loopGain = min_2d(self.HOloop_gain)

            # Log-spaced frequencies from 1e-3 Hz to Nyquist for all observations at once,
            # without a per-observation loop and the host-device syncs caused by .item()
            steps = torch.linspace(0, 1, nF, device=self.device, dtype=Ts.dtype).unsqueeze(0) # [1, nF]
            f = 10 ** (-3 + (torch.log10(0.5/Ts) + 3) * steps) # [N_obs, nF]

            _, _, _, ntfInt = self.TransferFunctions(f, Ts, delay, loopGain)
            self.noise_gain = (torch.trapz(ntfInt.abs().pow(2), f, dim=1) * 2*Ts.squeeze(1)).view(self.N_obs, 1, 1)
        else:
            # An average value of the noise transfer function as a function of the delay in frames for an integrator gain of 0.5
            self.noise_gain = torch.min(torch.tensor(0.8, device=self.device), 0.4 + 0.1333 * self.HOloop_delay).pow(2).view(self.N_obs, 1, 1)
    
     
    def Controller(self):
        '''
        Spatialized AO loop transfer functions under the frozen-flow hypothesis, f = -k·v (Correia et al. 2017, JOSA A 34, 1877, Eqs. 13-15).
        The wind direction may drift within wind_dir ± wind_dir_drift [deg] during the acquisition: the transfer functions are then averaged
        uniformly over these directions at a fixed wind speed (ibid., Eqs. 23-24, with the same direction bounds for H1, H2 and Hn), using an
        nTh-node Gauss-Legendre quadrature. With nTh = 0, the wind direction is fixed and the averaging is skipped.
        '''
        Ts =  1.0 / self.HOloop_rate  # sampling time
        delay     = self.HOloop_delay # latency between the measurement and the correction
        loop_gain = self.HOloop_gain

        if self.nTh == 0:
            idim = lambda x: x.view(self.N_obs, 1, 1, self.N_L)
            vx, vy = idim(self.vx), idim(self.vy)
            kx, ky = self.kx_AO.unsqueeze(-1), self.ky_AO.unsqueeze(-1) # add atmo. layers dimension: [N_src, nOtf_AO, nOtf_AO, 1]
            w = idim(self.Cn2_weights)
            
        else:
            idim = lambda x: x.view(self.N_obs, 1, 1, -1, 1)
            # Wind velocities rotated by the drift angles sampled within ± wind_dir_drift
            d_theta = torch.deg2rad(idim(self.wind_dir_drift)) * self.wind_drift_nodes # [N_obs, 1, 1, nL or 1, nTh]
            cos_dth, sin_dth = torch.cos(d_theta), torch.sin(d_theta)
            vx = idim(self.vx)*cos_dth - idim(self.vy)*sin_dth # [N_obs, 1, 1, nL, nTh]
            vy = idim(self.vx)*sin_dth + idim(self.vy)*cos_dth
           
            kx, ky = pdims(self.kx_AO, 2), pdims(self.ky_AO, 2) # add atmo. layers and wind directions dimensions: [N_src, nOtf_AO, nOtf_AO, 1, 1]
            w = idim(self.Cn2_weights) * self.wind_drift_weights

        fi = -(vx*kx + vy*ky) # [N_src, nOtf_AO, nOtf_AO, nL] or [N_src, nOtf_AO, nOtf_AO, nL, nTh]
        n_ext = fi.dim() - 1 # trailing dimensions to add to the per-observation loop parameters

        _, _, atfInt, ntfInt = self.TransferFunctions(fi, pdims(Ts, n_ext), pdims(delay, n_ext), pdims(loop_gain, n_ext))

        # AO transfer functions, averaged over the wind directions and summed over the atmospheric layers
        self.h1 = (w * atfInt).flatten(3).sum(dim=-1)
        self.h2 = (w * atfInt.abs().pow(2)).flatten(3).sum(dim=-1)
        self.hn = (w * ntfInt.abs().pow(2)).flatten(3).sum(dim=-1)


    def ReconstructionFilter(self, WFS_noise_var: torch.Tensor, MV: int = 0): # TODO: make MV switchable
        kx, ky = self.kxy_AO_() # the WFS samples the pupil frame
        Av = torch.sinc(self.WFS_d_sub*kx) * torch.sinc(self.WFS_d_sub*ky) * torch.exp(1j*torch.pi*self.WFS_d_sub*(kx+ky))
        self.SxAv = ( 2j*torch.pi*kx*self.WFS_d_sub*Av ).view(-1, self.nOtf_AO_y, self.nOtf_AO_x) # Same for a given AO regime
        self.SyAv = ( 2j*torch.pi*ky*self.WFS_d_sub*Av ).view(-1, self.nOtf_AO_y, self.nOtf_AO_x)
        
        noise_variance = WFS_noise_var.view(self.N_obs, -1).mean(dim=-1) # last dimension is for GSs, 0th dim is reserved for the observations
        
        W_n = pdims(noise_variance / (2*self.kc)**2, 2)

        self.W_atm = self.VonKarmanSpectrum(self.r0_(), self.L0.abs(), self.k2_AO) * self.piston_filter # TODO: at what λ must it be?

        gPSD = torch.abs(self.SxAv)**2 + torch.abs(self.SyAv)**2 + MV*W_n/self.W_atm / pdims(self.wvl_atm/self.GS_wvl, 2)**2
        self.Rx = torch.conj(self.SxAv) / gPSD
        self.Ry = torch.conj(self.SyAv) / gPSD
        
        self.Rx[..., self.nOtf_AO//2, self.nOtf_AO//2] = 1e-12 # For numerical stability TODO: is it really needed?
        self.Ry[..., self.nOtf_AO//2, self.nOtf_AO//2] = 1e-12

    
    def _anisoplanatism_phasor(self):
        '''
        Layer-weighted phasor between the science directions and the single guide star of a non-tomographic system (as P3's SCAO / SLAO
        spatio-temporal PSD), [N_src, nOtf_AO_y, nOtf_AO_x]. The layer heights are stretched by the LGS cone. The sign of the exponent is
        the conjugate of P3's because TipTorch's TransferFunctions use the conjugate temporal convention (z = exp(+2j pi f Ts)); with it the
        spatio-temporal PSD reproduces P3's. TipTorch pairs the x offset with kx (the first grid axis), as its tomographic projectors and
        its wind do; P3's SCAO branch pairs it with ky instead.
        '''
        theta_x = (self.src_dirs_x.view(-1) - self.GS_dirs_x[:, 0]).view(-1, 1, 1, 1) # [N_src, 1, 1, 1]
        theta_y = (self.src_dirs_y.view(-1) - self.GS_dirs_y[:, 0]).view(-1, 1, 1, 1)
        
        if not (theta_x.any() or theta_y.any()): # all sources on the guide star axis
            return torch.ones(1, 1, 1, device=self.device, dtype=self.dtype)
        
        h = self.h_().view(self.N_obs, 1, 1, self.N_L)
        w = self.Cn2_weights.view(self.N_obs, 1, 1, self.N_L)
        
        return (w * torch.exp(-2j*torch.pi * h * (theta_x*self.kx_AO.unsqueeze(-1) + theta_y*self.ky_AO.unsqueeze(-1)))).sum(dim=-1)


    def SpatioTemporalPSD(self):
        if not self.tomography:
            A = self._anisoplanatism_phasor() # ones on-axis
            Ff = self.Rx*self.SxAv + self.Ry*self.SyAv
            PSD_ST = (1 + Ff.abs()**2 * self.h2 - 2*torch.real(Ff*self.h1*A)) * self.W_atm * self.mask_corrected_AO
            
        else:
            kx = pdims(self.kx_AO, 1)
            ky = pdims(self.ky_AO, 1)
            h = self.h_().view(self.N_obs, 1, 1, self.N_L)
        
            N_src_ = len(self.src_dirs_x.flatten())
            beta_x = self.src_dirs_x.view(N_src_, 1, 1, 1)
            beta_y = self.src_dirs_y.view(N_src_, 1, 1, 1)
            
            # delta_T = ((1 + self.HOloop_delay) / self.HOloop_rate).view(self.N_obs, 1, 1, 1) #TODO: cross-check it
            delta_T = (self.HOloop_delay / self.HOloop_rate).view(self.N_obs, 1, 1, 1)
            
            self.P_beta_L = torch.exp( 2j*torch.pi * (h*(beta_x*kx + beta_y*ky) - delta_T*self.freq_t) ).unsqueeze(-2)
            proj = self.P_beta_L - self.P_beta_DM @ self.W_alpha
            proj_t = torch.conj(torch.permute(proj, (0,1,2,4,3)))
            PSD_ST = (proj @ self.C_phi @ proj_t).abs().squeeze() * self.piston_filter * self.mask_corrected_AO

        return PSD_ST


    def NoisePSD(self, WFS_noise_var: torch.Tensor):
        if not self.tomography:
            noisePSD = torch.abs(self.Rx**2 + self.Ry**2) / (2*self.kc)**2
            noisePSD = noisePSD * self.piston_filter * self.noise_gain * WFS_noise_var.view(self.N_obs, 1, 1) * self.mask_corrected_AO
            
        else:
            PW = self.P_beta_DM @ self.W
            PW_t = torch.conj(torch.permute(PW, (0,1,2,4,3)))
            noisePSD = (PW @ self.C_b @ PW_t).squeeze(-1).squeeze(-1)            
            noisePSD = noisePSD * self.noise_gain * self.mask_corrected_AO * self.piston_filter
            
        return noisePSD


    def AliasingPSD(self):
        T  = self.WFS_det_clock_rate / self.HOloop_rate
        td = pdims(T * self.HOloop_delay, [-1,3]) # [N_combs, N_src, nOtf_AO, n_Otf_AO, nL]
        T  = pdims(T, [-1,3])

        # Adding 0th dimension for shifted grid pieces
        Rx1 = (2j*torch.pi*self.WFS_d_sub * self.Rx).unsqueeze(0)
        Ry1 = (2j*torch.pi*self.WFS_d_sub * self.Ry).unsqueeze(0)

        km, kn, PR = self._aliasing_combs() # aliased frequencies in the pupil (WFS) frame

        # Compute von Karman spectrum for aliased spatial frequencies
        W_mn = self.VonKarmanSpectrum(self.r0_().view(1, self.N_obs), self.L0.abs().view(1, self.N_obs), km**2 + kn**2) * PR

        Q = (Rx1*km + Ry1*kn) * torch.sinc(self.WFS_d_sub*km) * torch.sinc(self.WFS_d_sub*kn)
        tf = self.h1.unsqueeze(0).unsqueeze(-1) # [N_combs, N_src, nOtf_AO, n_Otf_AO, nL]

        # Add aliasing dimension and more
        vx = self.vx.view(1, self.N_obs, 1, 1, self.N_L)
        vy = self.vy.view(1, self.N_obs, 1, 1, self.N_L)
        Cn2_weights = self.Cn2_weights.view(1, self.N_obs, 1, 1, self.N_L)

        # Adds  atmospheric layers dimension
        km, kn = km.unsqueeze(-1), kn.unsqueeze(-1)
        # TODO: do we really need an additional Cn2_weights multiplication here since it's already in h1 term?
        avr = (Cn2_weights * tf * torch.sinc(km*vx*T) * torch.sinc(kn*vy*T) * \
            torch.exp( 2j*torch.pi*td*(km*vx + kn*vy) )).sum(dim=-1) # sum along atmospheric layers

        # Sum along aliasing samples axis      
        aliasing_PSD = torch.sum( W_mn*(Q*avr).abs()**2, dim=0 ) * self.mask_corrected_AO    
        return aliasing_PSD


    def VonKarmanSpectrum(self, r0: torch.Tensor, L0: torch.Tensor, freq2: torch.Tensor):
        return self.cte*pdims(r0,2)**(-5/3) * (freq2 + 1/pdims(L0,2)**2)**(-11/6)


    def VonKarmanPSD(self):
        return self.VonKarmanSpectrum(self.r0_(), self.L0.abs(), self.k2) * self.mask


    def ChromatismPSD(self):
        N2 = self.IOR_GS_wvl - 1.0 # air refractivity at WFSing wavelength (N ≡ n-1)
        N1 = self.IOR_src_wvl.view(1, self.N_wvl, 1, 1) - 1.0 # air refractivity at science source wavelength
        
        chromatic_PSD = (1.0 - N1/N2)**2 * self.W_atm.unsqueeze(1)
        return chromatic_PSD


    def DifferentialRefractionPSD(self):
        # TODO: account for the pupil angle?
        h = self.h_().view(self.N_obs, 1, 1, 1, self.N_L)
        
        w = self.Cn2_weights.view(self.N_obs, 1, 1, 1, self.N_L)
        k = self.k_AO.view(1, 1, self.nOtf_AO_y, self.nOtf_AO_x, 1)

        src_azimuth = pdims(torch.arctan2(self.src_dirs_y, self.src_dirs_x).flatten(), 2) # [1, 1, N_src]

        cos_ang   = torch.cos(torch.arctan2(self.ky_AO, self.kx_AO) - src_azimuth).unsqueeze(1) # [N_src, 1, nOtf_AO_y, nOtf_AO_x]
        tan_theta = torch.tan((self.IOR_src_wvl - pdims(self.IOR_GS_wvl, 1)) * torch.tan(self.zenith_angle)) # [N_obs, N_wvl]
        
        return self.W_atm.unsqueeze(1) * ( 2*w*(1.0-torch.cos(2*torch.pi*h*k * pdims(tan_theta, 3) * pdims(cos_ang, 1))) ).sum(dim=-1) * self.mask_corrected_AO


    def ConeEffectPSD(self, PSD_AO: torch.Tensor, n_phases: int = 5) -> torch.Tensor:
        '''
        Cone effect of the LGS(s) on the AO half grid [N_src or N_obs, N_wvl or 1, nOtf_AO_y, nOtf_AO_x], restricted to the AO-corrected area
        (P3 adds its single-LGS term on the full grid, i.e. also in the ring kc < k < kc/g where the DM cannot act and the fitting error
        already holds the full atmospheric power).
        - Single LGS (SLAO, as P3's focalAnisoplanatismPSD): through the cone a layer sinusoid of frequency f is sensed at f*(H-h)/H; the RMS
          fraction left after subtracting the best-scaled sensed sinusoid over the pupil, averaged over n_phases phase offsets, is squared and
          filters the atmospheric spectrum of every layer. The pupil means of the sinusoids are analytical (mean of cos(2 pi f x) = sinc(f D)).
        - Tomographic LGS system (as P3's mcaoWFsensConePSD): for every layer and source, the part of the atmosphere corrected in the AO area
          (atmosphere minus the residual PSD_AO) is low-pass filtered above the frequency set by the angle between the source and the
          sensed volume, and scaled by the equivalent sensed aperture.
        '''
        H, h, w = self.LGS_height, self.Cn2_heights, self.Cn2_weights.view(self.N_obs, self.N_L, 1, 1) # [N_obs, 1], [N_obs, N_L]
        W_atm = self.VonKarmanSpectrum(self.r0_(), self.L0.abs(), self.k2_AO) # [N_obs, ny, nx]

        if not self.tomography:
            
            g = ((H - h) / H).view(self.N_obs, 1, self.N_L, 1) # frequency compression through the cone
            N_f = int(self.k_AO.max().item() / self.dk) + 2
            f = (torch.arange(N_f, device=self.device, dtype=self.dtype) * self.dk).view(1, -1, 1, 1) # [1, N_f, 1, 1] radial frequencies
            f_safe = torch.where(f > 1e-5, f, torch.full_like(f, self.dk)) # keeps the pupil variances positive (finite gradients); the DC is masked
            phi = (2*torch.pi * torch.arange(n_phases, device=self.device, dtype=self.dtype) / n_phases).view(1, 1, 1, -1)
            m = lambda freq: torch.sinc(freq * self.D) # pupil mean of cos(2*pi*freq*x) for x in [-D/2, D/2]

            f_cone = f_safe * g
            sin2_phi, cos_2phi = torch.sin(phi)**2, torch.cos(2*phi)
            var_ref  = 0.5 - 0.5*cos_2phi*m(2*f_safe) - sin2_phi*m(f_safe)**2 # pupil variance of sin(2*pi*f*x + phi)
            var_cone = 0.5 - 0.5*cos_2phi*m(2*f_cone) - sin2_phi*m(f_cone)**2
            cov = 0.5*(m(f_safe - f_cone) - cos_2phi*m(f_safe + f_cone)) - sin2_phi*m(f_safe)*m(f_cone)
            correlation = cov / torch.sqrt((var_ref * var_cone).clamp_min(1e-24))
            coeff = torch.sqrt((2.0 - 2.0*correlation).clamp_min(1e-12)).mean(dim=-1) * (f[..., 0] > 1e-5) # [N_obs, N_f, N_L] residual RMS fraction

            def radial_indices(): # linear interpolation of the radial coefficients on the 2D grid (clamped at the edge, as np.interp)
                t  = (self.k_AO[0] / self.dk).clamp(0, N_f - 1).flatten()
                i0 = t.floor().long().clamp(max=N_f - 2)
                return i0, t - i0
           
            i0, t = self._cached('cone effect indices', radial_indices)
            coeff = coeff.permute(0, 2, 1) # [N_obs, N_L, N_f]
            c = (coeff[..., i0] + (coeff[..., i0+1] - coeff[..., i0]) * t).view(self.N_obs, self.N_L, self.nOtf_AO_y, self.nOtf_AO_x)
            
            return ((w * c.pow(2)).sum(dim=1) * W_atm * self.mask_corrected_AO).unsqueeze(1)

        src_zenith = self.config['sources_science']['Zenith'].flatten().view(-1, 1) # [N_src, 1] in [arcsec]
        GS_zenith  = self.config['sources_HO']['Zenith'].view(self.N_obs, -1).amax(dim=-1, keepdim=True) # [N_obs, 1]
        LGS_FoV = 2*GS_zenith
        effective_FoV = (LGS_FoV/self.rad2arc - self.D/H) * self.rad2arc
        angle_E = torch.minimum(src_zenith, GS_zenith) - torch.where(effective_FoV > 0, effective_FoV/2, effective_FoV) # [N_src, 1]
        angle_L = (src_zenith - LGS_FoV/2).clamp_min(0)

        f_cut = self.rad2arc / (angle_E * h) # [N_src, N_L]
        D_eq  = (self.D - angle_L * h / self.rad2arc).clamp_max(self.D)
        valid = (h > 0) & (angle_E > 0) & (f_cut < self.kc) & (D_eq > 0)
        f_cut = torch.where(valid, f_cut, torch.full_like(f_cut, self.kc)) # invalid cut-off frequencies would overflow the pole below (masked at the end)

        # First-order digital low-pass filter z*(1-z_pole)/(z-z_pole) with z = exp(i*pi*k/(f_s/2)), normalized to the corner frequency of the full grid:
        # |z| = 1, so 1 - |filter|² = 1 - (1-z_pole)² / (1 + z_pole² - 2*z_pole*cos(pi*k/(f_s/2))) in real arithmetic
        f_s = 2*self.k.max()
        cos_k = self._cached('cone effect cosine', lambda: torch.cos(torch.pi * self.k_AO / (f_s/2))) # [1, nOtf_AO_y, nOtf_AO_x]
        z_pole = torch.exp(2*torch.pi*f_cut / f_s).view(*f_cut.shape, 1, 1)
        high_pass = 1 - (1 - z_pole)**2 / (1 + z_pole**2 - 2*z_pole*cos_k)
        gain = (high_pass * (D_eq/self.D).view(*D_eq.shape, 1, 1)**2).clamp_min(0) * valid.view(*valid.shape, 1, 1) # [N_src, N_L, ny, nx]
        gain = (w * gain).sum(dim=1).unsqueeze(1) # [N_src, 1, ny, nx]
        
        return gain * (W_atm.unsqueeze(1) - PSD_AO.real).clamp_min(0) * self.piston_filter * self.mask_corrected_AO # the tomographic noise term is complex-valued
    
    
    def JitterKernel(self, Jx: torch.Tensor, Jy: torch.Tensor, Jxy: torch.Tensor):
        # Assuming Jxy is in [deg], convert it to [rad]
        cos_theta = torch.cos( torch.deg2rad(Jxy) )
        sin_theta = torch.sin( torch.deg2rad(Jxy) )

        U_prime = self.U * cos_theta + self.V * sin_theta
        V_prime = -self.U * sin_theta + self.V * cos_theta

        Djitter = pdims(self.u_max * self.jitter_norm_fact, 2) * ( (Jx*U_prime)**2 + (Jy*V_prime)**2 )
        return torch.exp(-0.5 * Djitter) #TODO: cover the Nyquist sampled case? But check maybe it is automatic, already?
    

    def WingKernel(self, frac: torch.Tensor, width: torch.Tensor, angle: torch.Tensor):
        """
        MTF of a one-dimensional wing of the PSF along one image axis: (1 - f) + f * exp(-(2 pi w |nu|)^p), with nu the spatial frequency
        [cycles / detector px] along the wing axis, f the flux fraction in the wing (per wavelength), w its half-width [px] and p the exponent
        (p = 1: Lorentzian wing of HWHM w; p = 2: Gaussian). The axis is the image columns (the second image axis, V), rotated by `angle` [deg]
        in the sense of JitterKernel. Like the jitter, it is built on the image-frame grid, so it does not turn with the pupil angle.
        It models an instrument-frame spread (slicer / spectrograph) which an isotropic halo or a jitter cannot represent.
        """
        cos_theta = torch.cos( torch.deg2rad(angle) )
        sin_theta = torch.sin( torch.deg2rad(angle) )
        V_prime = -self.U * sin_theta + self.V * cos_theta
        # U, V span [-1, 1] up to the Nyquist frequency sqrt(u_max) [cycles/mas]; x psInMas gives [cycles / detector px]
        nu_px = V_prime.abs() * pdims(torch.sqrt(self.u_max) * self.psInMas, 2)
        f = pdims(min_2d(frac), 2)
        w = pdims(min_2d(width), 2)
        return (1.0 - f) + f * torch.exp( -(2 * torch.pi * w * nu_px) ** self.wing_power )


    def NoiseVariance(self):
        """
        Compute WFS noise variance with fully vectorized implementation.
        For now, it assumes the same type of WFS for all GSs for greater parallelism.
        
        For SH, it supports multiple centroiding algorithms:
        - 'cog':  Center of Gravity (standard)
        - 'tcog': Thresholded CoG
        - 'wcog': Weighted CoG
        - 'qc':   Quad Cell
        Also support pyramid WFS.
        
        Returns:
            sigma_noise_sqr: [N_obs, N_GS] noise variance tensor
        """
        
        # Parse WFS parameters
        WFS_wvl  = self.WFS_wvl
        WFS_RON  = self.WFS_RON
        WFS_Nph  = self.WFS_Nph.abs().view(self.N_obs, self.N_GS)#.clamp(min=1.0)  # prevent 1/0=inf -> 0*inf=NaN in backward
        r0_WFS   = self.r0_new(self.r0_(), WFS_wvl, self.wvl_atm).view(-1,1)
        WFS_nPix = self.WFS_FOV / self.WFS_n_sub
        WFS_pixelScale = self.WFS_psInMas * self.mas2arc  # [arcsec]

        # Shack-Hartmann WFS calculations
        if self.is_SH:
            # nD: spot FWHM in pixels without turbulence (diffraction-limited), for DL: nT = nD = 2
            nD = torch.maximum(
                self.rad2arc * WFS_wvl / self.WFS_d_sub / WFS_pixelScale, 
                torch.tensor(1.0, device=self.device)
            ).view(-1,1)
            
            spot_FWHM_turb = 0.98 * self.rad2arc * WFS_wvl / r0_WFS / torch.sqrt(torch.tensor(2.0, device=self.device))  # [arcsec]
            
            # nT: spot FWHM in pixels with turbulence and TT removed according to [Thomas et al. 2006]
            nT = torch.maximum(
                torch.hypot(self.WFS_spot_FWHM.max() * self.mas2arc, spot_FWHM_turb) / WFS_pixelScale,
                torch.tensor(1.0, device=self.device)
            ).view(-1,1)
            
            # Center of gravity
            if self.WFS_algorithm == 'cog':
                sigma_sqr_RON  = (torch.pi**2 / 3) * (WFS_RON / WFS_Nph)**2 * (WFS_nPix**2 / nD).unsqueeze(-1)**2
                sigma_sqr_shot = (torch.pi**2 / (2 * torch.log(torch.tensor(2.0, device=self.device)))) / WFS_Nph * (nT / nD)
            
            # Truncated CoG algorithm
            elif self.WFS_algorithm == 'tcog':
                nPix_eff = torch.ceil(nT**2 * torch.pi / 4)
                sigma_sqr_RON  = (torch.pi**2 / 3) * (WFS_RON / WFS_Nph)**2 * (nPix_eff / nD).unsqueeze(-1)**2
                sigma_sqr_shot = (torch.pi**2 / (2 * torch.log(torch.tensor(2.0, device=self.device)))) / WFS_Nph * (nT / nD)
            
            # Weighted CoG algorithm
            elif self.WFS_algorithm == 'wcog':
                nW = torch.maximum(self.WFS_algo_settings, nT)  # Ensure nW >= nT
                
                nT_nW_sum = nT**2 + nW**2
                sigma_sqr_RON = (torch.pi**3 / (32 * torch.log(torch.tensor(2.0))**2)) * \
                                (WFS_RON / WFS_Nph)**2 * \
                                (nT_nW_sum**4 / (nD**2 * nW**4))
                        
                sigma_sqr_shot = (torch.pi**2 / (2 * torch.log(torch.tensor(2.0)))) / WFS_Nph * \
                                 (nT / nD)**2 * \
                                 (nT_nW_sum**4 / ((2 * nT**2 + nW**2)**2 * nW**4))
            
            # Quad Cell algorithm
            elif self.WFS_algorithm == 'qc':
                k = torch.where(nT > nD,
                    torch.sqrt(2.0 * torch.pi) * nT / (2.0 * torch.sqrt(2.0 * torch.log(torch.tensor(2.0, device=self.device)))) / nD,
                    torch.tensor(1.0, device=self.device)
                ).unsqueeze(-1)
                
                sigma_sqr_RON  = k * 4 * torch.pi**2 * (WFS_RON / WFS_Nph)**2
                sigma_sqr_shot = k * torch.pi**2 / WFS_Nph
            
            # Limit unreasonably high variances
            sigma_sqr_RON  = torch.clamp(sigma_sqr_RON,  max=6.0)
            sigma_sqr_shot = torch.clamp(sigma_sqr_shot, max=6.0)
            
            # TODO: maybe use soft clamp for better differentiability?
            # sigma_sqr_RON  = 3.0 * torch.tanh(sigma_sqr_RON / 3.0)
            # sigma_sqr_shot = 3.0 * torch.tanh(sigma_sqr_shot / 3.0)
        
        # Pyramid WFS calculations
        elif self.is_pyramid:
            sigma_sqr_RON  = 4 * WFS_RON**2 / WFS_Nph**2
            sigma_sqr_shot = WFS_Nph / WFS_Nph**2
        
        # Total noise variance
        sigma_noise_sqr = (self.WFS_excessive_factor * sigma_sqr_shot + sigma_sqr_RON) * (self.WFS_wvl / self.wvl_atm).unsqueeze(-1)**2
        
        return sigma_noise_sqr


    def TomographicReconstructors(self, WFS_noise_var):
        '''        
        Note that if all simulated sources use the same atmospheric profile, r0, L0, and noise variance,
        then it's possible to compute one tomographic reconstructor for all simulated sources.
        For example, this is the case when all objects are within one FoV and belong to one observation
        '''
        h = self.h_().view(self.N_obs, 1, 1, 1, self.N_L)

        # The LGS asterism is launched from the telescope and the WFSs sample the pupil: both live in the pupil frame
        kx_p, ky_p = self.kxy_AO_()
        kx = pdims(kx_p, 2)
        ky = pdims(ky_p, 2)
        GS_dirs_x = self.GS_dirs_x.view(self.N_obs, 1, 1, self.N_GS, 1)
        GS_dirs_y = self.GS_dirs_y.view(self.N_obs, 1, 1, self.N_GS, 1)

        diag_mask = lambda N: torch.eye(N, device=self.device).view(1,1,1,N,N).expand(self.N_obs, self.nOtf_AO_y, self.nOtf_AO_x, -1, -1)

        M    = 2j*torch.pi*self.k_AO * torch.sinc(self.WFS_d_sub*kx_p) * torch.sinc(self.WFS_d_sub*ky_p)
        M    = pdims(M, 2).expand(self.N_obs, self.nOtf_AO_y, self.nOtf_AO_x, self.N_GS, self.N_GS) * diag_mask(self.N_GS) # [N_obs, nOtf_y, nOtf_x, nGS, nGS]
        P    = torch.exp( 2j*torch.pi*h * (kx*GS_dirs_x + ky*GS_dirs_y) ) # [N_obs, nOtf_y, nOtf_x, nGS, nL]
        MP   = torch.einsum('nwhik,nwhkj->nwhij', M, P)  # [N_obs, nOtf_y, nOtf_x, nGS, nL]
        MP_t = torch.conj(MP.permute(0,1,2,4,3))

        expand_dims = lambda x_, N: torch.diag_embed(x_).view(self.N_obs, 1, 1, N, N).expand(self.N_obs, self.nOtf_AO_y, self.nOtf_AO_x, N, N)
        
        # Note, that size of WFS_noise_var == N_obs
        WFS_noise_variance = WFS_noise_var.to(dtype=MP.dtype) # Convert to complex
        # As previosuly mentioned, if the same tomo reconstructor is used for all targets, then WFS_noise_variance also must be the same for all targets
        self.C_b   = expand_dims( WFS_noise_variance[:self.N_obs], self.N_GS )
        kernel     = self.VonKarmanSpectrum(self.r0_().to(dtype=MP.dtype), self.L0.abs(), self.k2_AO) * self.piston_filter
        self.C_phi = pdims(kernel, 2) * expand_dims(self.Cn2_weights, self.N_L)

        # MP·C_phi·MPᴴ + C_b is Hermitian positive definite (C_b > 0 due to shot noise), so solve W·A = B exactly,
        # without truncation. Inversion happens relative to the last two dimensions of these tensors
        A = MP @ self.C_phi @ MP_t + self.C_b
        self.W_tomo = torch.linalg.solve(A, self.C_phi @ MP_t, left=False)
         
        self.W = self.P_opt @ self.W_tomo # [N_obs, nOtf_AO_y, nOtf_AO_x, N_GS, N_L]
        
        # NOTE that size of HOloop_rate == N_obs (wait, why?)
        samp_time = 1.0 / self.HOloop_rate
        www = 2j * torch.pi * pdims(self.k_AO, 1) * torch.sinc((samp_time * self.WFS_det_clock_rate).view(self.N_obs,1,1,1) * self.freq_t)
        MP_alpha_L = www.unsqueeze(-2) * P * (torch.sinc(self.WFS_d_sub*kx) * torch.sinc(self.WFS_d_sub*ky)) # [N_obs, nOtf_AO_y, nOtf_AO_x, 1, N_L]
        self.W_alpha = self.W @ MP_alpha_L # [N_obs, nOtf_AO_y, nOtf_AO_x, 1, N_L]


    def MoffatPSD(self, amp: torch.Tensor, b: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, ratio: torch.Tensor, theta: torch.Tensor):
        """ Computes the PSF AO-style Moffat PSD as described in [Fétick et al. 2019] """
        ax = alpha * ratio
        ay = alpha / ratio

        uxx, uxy, uyy = self._Moffat_grids() # the Moffat absorbs AO residuals, so it is oriented in the pupil frame

        c  = torch.cos(theta)
        s  = torch.sin(theta)
        s2 = torch.sin(2.0 * theta)

        rxx = (c/ax)**2 + (s/ay)**2
        rxy =  s2/ay**2 -  s2/ax**2
        ryy = (c/ay)**2 + (s/ax)**2

        uu = rxx*uxx + rxy*uxy + ryy*uyy

        V = (1.0+uu)**(-beta)  # defines the shape of the Moffat

        removeInside = 0.0
        E = (beta-1.) / (torch.pi*ax*ay)
        F_out = (1. +      (self.kc**2)/(ax*ay))**(1.-beta)
        F_in  = (1. + (removeInside**2)/(ax*ay))**(1.-beta)
        F = 1. / (F_in-F_out)

        MoffatPSD = (amp * V*E*F + b) * self.mask_corrected_AO * self.piston_filter

        return MoffatPSD


    def DLPSF(self):
        ''' Computes a diffraction-limited PSF '''
        self.PSF_DL = self.OTF2PSF(self.OTF_static_default)
        DL_norm_scale = self.normalizer(self.PSF_DL, dim=(-2,-1), keepdim=True)
        self.PSF_DL /= DL_norm_scale
        
        return self.PSF_DL


    def OLPSD(self):
        ''' Compute open-loop PSD, i.e. the atmospheric PSD without any AO correction. '''
        PSD_half = self.VonKarmanSpectrum(self.r0_(), self.L0.abs(), self.k2)  # [N_obs, nOtf_y, nOtf_x]

        # Remove DC explicitly. Piston cancels in the structure function anyway, but this avoids a huge useless covariance offset.
        PSD_half[..., self.nOtf_y // 2, self.nOtf_x - 1] = 0.0
        PSD_half = PSD_half.unsqueeze(1) #[N_obs, 1, nOtf_y, nOtf_x]
        # Match the convention used by ComputePSD(), PSD is converted from rad^2 to nm^2 OPD at λ_atm
        # Recover full centered PSD from the stored half-plane representation.
        self.PSD_open_loop = self.half_PSD_to_full(PSD_half * self._PSD_norm())
        
        return self.PSD_open_loop # [N_obs, 1, nOtf, nOtf]


    def OLPSF(self, include_static: bool = True, include_jitter: bool = False):
        '''
        Computes the open-loop / seeing-limited PSF.

        include_static: If True, use static telescope aberrations.
        include_jitter: If False, disables the jitter kernel. AO-only jitter must be zero because the von Karman PSD already contains atmospheric tip/tilt.
        '''

        PSD = self.OLPSD()

        if include_static:
            OTF_static = self.OTF_static
        else:
            OTF_static = torch.ones( (1, 1, self.nOtf, self.nOtf), device=self.device, dtype=torch.complex64 if self.is_float else torch.complex128 )

        if include_jitter:
            self.PSF_open_loop = self.PSD2PSF(PSD, OTF_static)
            return self.PSF_open_loop
        
        # Otherwise, jitter must be muted since PSD2PSF always applies JitterKernel() (the image-frame wing is muted with it)
        Jx_old, Jy_old, Jxy_old, use_wing_old = self.Jx.clone(), self.Jy.clone(), self.Jxy.clone(), self.use_wing

        self.Jx  = torch.zeros_like(Jx_old)
        self.Jy  = torch.zeros_like(Jy_old)
        self.Jxy = torch.zeros_like(Jxy_old)
        self.use_wing = False

        self.PSF_open_loop = self.PSD2PSF(PSD, OTF_static)

        self.Jx, self.Jy, self.Jxy, self.use_wing = Jx_old, Jy_old, Jxy_old, use_wing_old

        return self.PSF_open_loop # [N_obs, N_wvl, N_pix, N_pix]


    def ComputePSD(self, update_LO_only: bool = False) -> torch.Tensor:
        '''
        Residual PSD [N_src, N_wvl, nOtf, nOtf] in [nm²] per pixel. All terms are computed on the half grids in [rad²/m²] at the atmosphere
        wavelength and expanded to the full grid only at the end. The core PSD (AO-corrected terms with the cone effect, fitting, extra error)
        is kept in PSD_core; with update_LO_only it is reused and only the LO terms (wind shake, tilt filter, LO extra error, focus
        error) are re-applied, e.g. after setting focus_error_nm.
        '''
        include = self.PSD_include
        if not any(include.values()):
            self.PSD = torch.zeros([self.N_src, self.N_wvl, self.nOtf, self.nOtf], device=self.device)
            return self.PSD

        if update_LO_only:
            return self._apply_LO(self.PSD_core, self.PSDs if self.retain_PSDs and hasattr(self, 'PSDs') else {})

        if include['WFS noise'] or include['spatio-temporal'] or include['aliasing']:
            WFS_noise_var = (self.dn.view(self.N_obs,-1) + self.NoiseVariance()).abs() # [rad^2] at atmo wvl
            # TODO: check the wind direction sign conventions! sin and cos might be swapped and - should b in front of wind speed
            self.vx = self.wind_speed * torch.cos( torch.deg2rad(self.wind_dir) )
            self.vy = self.wind_speed * torch.sin( torch.deg2rad(self.wind_dir) )
            
            self.freq_t = self.vx.view(self.N_obs, 1, 1, self.N_L) * pdims(self.kx_AO, 1) + \
                          self.vy.view(self.N_obs, 1, 1, self.N_L) * pdims(self.ky_AO, 1) # [N_src, nOtf_AO, nOtf_AO, nL]
                          
            self.Controller()
            self.ReconstructionFilter(WFS_noise_var)
            
            if self.tomography:
                self.TomographicReconstructors(WFS_noise_var)
                
                if not self.on_axis:
                    self.DMProjector()

        # Terms of the AO-corrected area, [N_src or N_obs, N_wvl or 1, nOtf_AO_y, nOtf_AO_x]
        AO_terms = {
            'WFS noise':       lambda: self.NoisePSD(WFS_noise_var).unsqueeze(1),
            'spatio-temporal': lambda: self.SpatioTemporalPSD().unsqueeze(1),
            'aliasing':        lambda: self.AliasingPSD().unsqueeze(1),
            'chromatism':      self.ChromatismPSD, # polychromatic already
            'diff. refract':   self.DifferentialRefractionPSD,
            'Moffat':          lambda: self.MoffatPSD(*[pdims(x, 2) for x in (self.amp.abs(), self.b, self.alpha, self.beta, self.ratio, self.theta)]).unsqueeze(1),
        }
        PSDs = {name: term() for name, term in AO_terms.items() if include[name]}
        PSD_AO = sum(PSDs.values(), torch.zeros(1, device=self.device))
        
        if include['cone effect']:
            PSDs['cone effect'] = self.ConeEffectPSD(PSD_AO)
            PSD_AO = PSD_AO + PSDs['cone effect']

        # Full half grid [N_src, N_wvl, nOtf_y, nOtf_x]
        PSD = self.PSD_padder(PSD_AO) if PSD_AO.dim() == 4 else PSD_AO
        if include['fitting']:
            PSDs['fitting'] = self.VonKarmanPSD().unsqueeze(1)
            # if getattr(self, 'fitting_excess', None) is not None: # diagnostic multiplicative excess (1 + eps) [N_obs] of the fitting error, None by default
            #     PSDs['fitting'] = PSDs['fitting'] * (1.0 + self.fitting_excess.view(-1, 1, 1, 1))
            PSD = PSD + PSDs['fitting']
                
        if include['extra error'] and self.extra_error_nm is not None:
            PSDs['extra error'] = self.extra_error_nm.view(-1, 1, 1, 1).pow(2) * self._unit_spectrum('extra error') / self._PSD_norm()
            PSD = PSD + PSDs['extra error']

        self.PSD_core = PSD
        return self._apply_LO(PSD, PSDs)


    def _apply_LO(self, PSD: torch.Tensor, PSDs: dict) -> torch.Tensor:
        ''' Apply the LO terms to the half-grid core PSD (P3's order: wind shake, tilt filter, extra error, then the focus error) and expand to the full grid '''
        include = self.PSD_include
        rms2 = lambda rms: self.make_tensor(rms).view(-1, 1, 1, 1).pow(2) / self._PSD_norm()

        if include['wind shake'] and self.vibration_PSD is not None:
            PSDs['wind shake'] = rms2(self._wind_shake_power().sqrt()) * self._unit_spectrum('wind shake')
            PSD = PSD + self.PSD_padder(PSDs['wind shake'])
        
        if include['tilt filter']: # the (dimensionless) filter itself: tip/tilt is left to a separate LO loop
            PSDs['tilt filter'] = self._cached('tilt filter', lambda: self._spatial_filters(self.k)[1])
            PSD = PSD * PSDs['tilt filter']
        
        if include['LO extra error'] and self.extra_error_LO_nm is not None:
            PSDs['LO extra error'] = rms2(self.extra_error_LO_nm) * self._unit_spectrum('LO extra error')
            PSD = PSD + PSDs['LO extra error']
       
        if include['focus error'] and self.focus_error_nm is not None:
            PSDs['focus error'] = rms2(self.focus_error_nm) * self._unit_spectrum('focus error')
            PSD = PSD + PSDs['focus error']

        PSD[..., self.nOtf_y//2, self.nOtf_x-1] = 0.0 # remove the DC component
        self.PSD = self.half_PSD_to_full(PSD * self._PSD_norm()) # [nm²] on the full grid
        if self.retain_PSDs:
            self.PSDs = PSDs # all computed terms, for the error budget and diagnostics
        return self.PSD
    
    
    def half_PSD_to_full(self, half_PSD):
        return torch.cat([
            half_PSD,
            torch.flip(half_PSD[..., :, :-1], dims=(-2,-1)) # Works only for odd num. of pixels
        ], dim=-1)
    
    
    def _rfft2_to_full(self, matrix_rfft2):
        return torch.cat([
            torch.flip(matrix_rfft2[..., :, 1:].conj(), dims=[-2,-1]),
            matrix_rfft2
        ], dim=-1) # Works only for odd number of pixels
    

    def OTF2PSF(self, OTF): 
        PSF_big = fft.fftshift(fft.ifft2(fft.ifftshift(OTF, dim=(-2,-1))), dim=(-2,-1)).abs()
        
        PSF = []
        for i in range(self.wvl.shape[-1]):
            transform = transforms.CenterCrop((self.nOtfs[i], self.nOtfs[i]))
            n = 0 if PSF_big.shape[1] == 1 else i # when there is no chromatic OTF
            
            if OTF.shape[-1] > self.N_pix:
                PSF_interp = interpolate(
                    transform(PSF_big[:,n,...]).unsqueeze(1),
                    size = (self.N_pix, self.N_pix),
                    mode = 'bilinear'
                ) * (PSF_big.shape[-1] / self.N_pix)**2 # preserve energy
                PSF.append(PSF_interp)
            
            else: # Keep the requested detector size even when the OTF grid is odd.
                PSF_cropped = transforms.CenterCrop((self.N_pix, self.N_pix))(
                    PSF_big[:, n, ...]).unsqueeze(1)
                PSF.append( PSF_cropped )

        return torch.hstack(PSF)


    def PSD2PSF(self, PSD: torch.Tensor, OTF_static: torch.Tensor):
        # Ensure that wavelength dimension is present
        F   = pdims(min_2d(self.F),  2)
        bg  = pdims(min_2d(self.bg), 2)
        dx  = pdims(min_2d(self.dx), 2)
        dy  = pdims(min_2d(self.dy), 2)
        Jx  = pdims(min_2d(self.Jx ), 2)
        Jy  = pdims(min_2d(self.Jy ), 2)
        Jxy = pdims(min_2d(self.Jxy), 2)
                
        # Computing OTF from PSD, real is to remove the imaginary part that appears due to numerical errors
        cov = self._rfft2_to_full(torch.fft.fftshift(torch.fft.rfft2(torch.fft.ifftshift(PSD.abs(), dim=(-2,-1)), dim=(-2,-1)), dim=-2).real)

        # Computing the Structure Function from the covariance
        SF = 2*(cov.abs().amax(dim=(-2,-1), keepdim=True) - cov).real

        # Phasor to shift the PSF with the subpixel accuracy
        fftPhasor = torch.exp( -torch.pi*1j * pdims(self.sampling_factor,2) * (self.U*dx + self.V*dy) )
        OTF_turb  = torch.exp( -0.5 * SF * pdims(2*torch.pi*1e-9/self.wvl,2)**2 )
        
        # Compute the residual tip/tilt kernel
        OTF_jitter = self.JitterKernel(Jx.abs(), Jy.abs(), Jxy)
        
        # Resulting combined OTF
        self.OTF = OTF_turb * OTF_static * fftPhasor * OTF_jitter
        if self.use_wing: # one-dimensional image-frame wing (unity at zero frequency, so the normalization below is unaffected)
            self.OTF = self.OTF * self.WingKernel(self.wing_frac.abs(), self.wing_width.abs(), pdims(self.wing_angle, 2))
        # self.OTF = OTF_static
        # Index before abs(): the gradient of abs() is z/|z|, NaN where the OTF underflows to exactly zero (far wings at low r0),
        # and 0 * NaN from the indexing then poisons the whole batch in the backward pass
        self.OTF_norm = pdims(self.OTF[..., self.nOtf//2, self.nOtf//2].abs(), 2)
        self.OTF = self.OTF / self.OTF_norm
        
        # Computing final PSF
        # PSF_out = self.OTF2PSF(self.OTF)
        PSF_out = self.OTF2PSF(self.OTF)
        self.norm_scale = self.normalizer(PSF_out, dim=(-2,-1), keepdim=True)
        
        return (PSF_out / self.norm_scale) * F + bg


    def ErrorBudget(self, verbose=False):
        '''' Computes error budget over simulated PSD contributors in [nm RMS] assuming their statistical indipendence '''
        error_budget = {}

        if not self.retain_PSDs:
            print("No PSDs stored. Recomputing PSDs with 'retain_PSDs = True'")
            self.retain_PSDs = True
            self.ComputePSD()
            
        PSD_norm = (self.wvl_atm*1e9/2/torch.pi)**2

        for entry, PSD in self.PSDs.items():
            if entry == 'tilt filter': # a filter, not a PSD
                continue

            if len(PSD.shape) > 1:            
                PSD = self.half_PSD_to_full(PSD * PSD_norm).real # [nm^2 m^-2]
                
                error_budget[entry] = (PSD * self.dk**2).sum().sqrt().item()
                
                if verbose:
                    print(f"{entry:15s}: {error_budget[entry]:.2f} [nm RMS]")
        
        error_budget['Total HO'] = self.PSD.real.sum().sqrt().item() # self.PSD is already normalized
        
        if verbose:
            print(f"{'Total HO':15s}: {error_budget['Total HO']:.2f} [nm RMS]")
            
        #TODO: TT jitter error in [nm RMS]
            
        return error_budget


    def SetWavelengths(self, wavelengths: torch.Tensor, refresh_static_OTF: bool = True):
        ''' Set new simulated wavelengths in [nm].

        Args:
            wavelengths: New simulated wavelengths in [nm].
            refresh_static_OTF: If True (default), the diffraction-limited static OTF
                is recomputed from the pupil for the new grid. Set to False when the
                caller will immediately overwrite ``OTF_static`` via
                ``ComputeStaticOTF(phase)`` — this skips a wasted pupil FFT per batch.
        '''
        self.config['sources_science']['Wavelength'] = wavelengths.view(1,-1) # [nm]
        
        if refresh_static_OTF:
            # Full update: grids + diffraction-limited static OTF + tomography projector.
            self.Update(grids=True, pupils=False, tomography=True) # Avoid an expensive update of pupil masks
        else:
            # Skip the static-OTF FFT — the caller will set OTF_static via ComputeStaticOTF(phase).
            self.Update(grids=True, pupils=False, tomography=True, update_static_OTF=False)
            
        self.wavelengths = wavelengths # [nm]


    def SetImageSize(self, img_size: int):
        ''' Set new image size in pixels '''
        self.config['sensor_science']['FieldOfView'] = img_size
        self.Update(grids=True, pupils=False, tomography=True)


    def to(self, *args, **kwargs):
        '''
        Moves and/or casts all tensors of the model, including the ones nested in containers such as config (PyTorch-style).

        Signature variations:
            to(device=None, dtype=None, non_blocking=False)
            to(dtype, non_blocking=False)
            to(tensor, non_blocking=False)

        Floating and complex tensors are cast to the matching precision, i.e. float32 <-> complex64 and float64 <-> complex128,
        so either a real or a complex dtype can be passed. Integer and boolean tensors only change device.

        Returns:
            self
        '''
        device, dtype, non_blocking, _ = torch._C._nn._parse_to(*args, **kwargs)

        float_dtype, complex_dtype = None, None
        if dtype is not None:
            if   dtype in (torch.float32, torch.complex64):  float_dtype, complex_dtype = torch.float32, torch.complex64
            elif dtype in (torch.float64, torch.complex128): float_dtype, complex_dtype = torch.float64, torch.complex128
            else:
                raise TypeError(f'Unsupported dtype: {dtype}. Use float32, float64, complex64, or complex128.')
            self.dtype, self.is_float = float_dtype, float_dtype == torch.float32

        if device is not None:
            if device.type == 'cuda' and device.index is None:
                device = torch.device('cuda', torch.cuda.current_device())
            self.device = device

        def convert(t):
            target = float_dtype if t.is_floating_point() else complex_dtype if t.is_complex() else None
            return t.to(device=device, dtype=target, non_blocking=non_blocking)

        self._convert_tensors(convert)
        return self


    def forward(self, x=None, PSD=None, phase=None):
        if x is not None:
            for name, value in x.items():
                if hasattr(self, name):
                    setattr(self, name, value)

        return self.PSD2PSF(
            self.ComputePSD() if PSD is None else PSD,
            self.ComputeStaticOTF(phase)
        )


    @staticmethod
    def _holds_tensor(obj) -> bool:
        if torch.is_tensor(obj):
            return True
        if isinstance(obj, dict):
            obj = obj.values()
        elif not isinstance(obj, (list, tuple)):
            return False
        return any(TipTorch._holds_tensor(x) for x in obj)


    def cleanup(self):
        """Explicitly clean up GPU memory and release resources"""
        # Drop every public attribute holding tensors (directly or inside containers, e.g. config).
        # References are dropped rather than cleared in-place, so caller-owned dicts stay intact
        for name, value in list(vars(self).items()):
            if not name.startswith('_') and self._holds_tensor(value):
                delattr(self, name)

        self._spectra_cache = {}
        self._parameters.clear()
        self._buffers.clear()
        self._modules.clear() # pupil_padder, PSD_padder

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif self.device.type == 'mps':
            torch.mps.empty_cache()
            torch.mps.synchronize()
    

    def __del__(self):
        """Destructor to ensure GPU memory is freed"""
        try:
            self.cleanup()
        except:
            # Silently fail if cleanup has issues during destruction
            pass
        
