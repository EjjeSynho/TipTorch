#%%
import warnings
import torch
import numpy as np
import scipy.special as spc
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch import fft, nn
from torch.nn.functional import interpolate
from astropy.io import fits
from tiptorch.tools.utils import pdims, min_2d, to_little_endian
from tiptorch.tools.air_refraction import AirRefractiveIndexCalculator
from pathlib import Path


class TipTorch(torch.nn.Module):

    def _load_pupil_and_apodizer(self):
        # If not provided externally, TipTorch tries to load pupil and apodizer from the config file
        pupil_loaded = self.pupil is None
        if pupil_loaded:
            pupil_path = Path(self.config['telescope']['PathPupil'])
            self.pupil = self.make_tensor(to_little_endian(fits.getdata(pupil_path)))

        apodizer_loaded = self.apodizer is None and self.config['telescope']['PathApodizer'] is not None
        if apodizer_loaded:
            apodizer_path = Path(self.config['telescope']['PathApodizer'])
            self.apodizer = self.make_tensor(to_little_endian(fits.getdata(apodizer_path)))

        if self.apodizer is not None:
            assert self.pupil.shape[-1] == self.apodizer.shape[-1], "Pupil and apodizer must have the same size"

        # Externally provided pupil/apodizer are assumed to already be correctly oriented, so rotate only what was just loaded
        if self.pupil_angle != 0.0 and pupil_loaded:
            angle = self.pupil_angle.item() if torch.is_tensor(self.pupil_angle) else self.pupil_angle
            # Pupil is a binary mask, so use nearest-neighbor to avoid introducing fractional edge values
            self.pupil = TF.rotate(self.pupil.unsqueeze(0), -angle, interpolation=TF.InterpolationMode.NEAREST).squeeze(0)

            if apodizer_loaded:
                self.apodizer = TF.rotate(self.apodizer.unsqueeze(0), -angle, interpolation=TF.InterpolationMode.BILINEAR).squeeze(0)



    def UpdateStaticOTF(self):
        """Recompute diffraction-limited static OTF for the current grid sampling."""
        if self.pupil is None:
            raise RuntimeError('Cannot update static OTF: pupil is not initialized. Call InitPupils() first.')
        
        if not hasattr(self, 'sampling_min') or not hasattr(self, 'nOtf'):
            raise RuntimeError('Cannot update static OTF: grids are not initialized. Call InitGrids() first.')

        phase_size = self.pupil.shape[-1]
        self.pupil_padder = torch.nn.ZeroPad2d(int(round(phase_size * self.sampling_min / 2 - phase_size / 2)))

        # Compute the diffraction-limited OTF from the pupil (and apodizer if present)
        pupil_phase = self.pupil * self.apodizer if self.apodizer is not None else self.pupil
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
        
        self.pupil_angle = self.config['telescope']['PupilAngle'] if 'PupilAngle' in self.config['telescope'] else 0.0

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
        
        if self.PSD_include['Moffat']:
            self.amp   = torch.ones (self.N_src, device=self.device)*0.01 # Moffat amplitude [rad²]
            self.alpha = torch.ones (self.N_src, device=self.device)*0.1  # Moffat alpha [1/m]
            self.beta  = torch.ones (self.N_src, device=self.device)*2    # Moffat beta power law
            self.ratio = torch.ones (self.N_src, device=self.device)      # Moffat ellipticity
            self.b     = torch.zeros(self.N_src, device=self.device)      # background [rad²/m²]
            self.theta = torch.zeros(self.N_src, device=self.device)      # Moffat angle

        # if self.PSD_include['WFS noise'] or self.PSD_include['spatio-temporal'] or self.PSD_include ['aliasing']:
        self.dn  = torch.zeros(self.N_obs, device=self.device)

        self._init_addon_values()

        if self.AO_type is None:
            self._select_AO_correction()
        
        self.tomography = self.AO_type in ['LTAO', 'MCAO', 'GLAO']
        self.approx_noise_gain = self.tomography # Enable simplified noise gain computation for tomographic systems

        self.NoiseGain() # Compute the noise gain
              
        # Initialize index of refraction values
        self.IOR_src_wvl = self.n_air(self.wvl)
        self.IOR_wvl_atm = self.n_air(self.wvl_atm)
        self.IOR_GS_wvl  = self.n_air(self.GS_wvl) # GS_wvl may depend on the filter for SCAO or on LGS wavelength
 
 
    def _init_addon_values(self):
        '''
        Settings of the P3-style add-on PSDs from the TIPTOP entries [telescope] extraErrorNm / Exp / Min / Max, extraErrorLoNm / LoExp / LoMin /
        LoMax, TechnicalFoV, windPsdFile and [sensor_HO] addMcaoWFsensConeError. Only the entries present in the config are initialized and
        nothing is computed unless the corresponding PSD_include entries are enabled. As P3's getPSDatNGSpositions, when the config has a
        [sources_LO] section its N_LO directions are assumed to be the trailing N_LO sources: they get the LO extra error, the others the HO one.
        '''
        get    = lambda section, key, default=None: self.config[section][key] if self.config[section].get(key) is not None else default
        scalar = lambda x: float(torch.as_tensor(x).flatten()[0])

        self.extra_error_nm = self.extra_error_LO_nm = None
        rms_HO = scalar(get('telescope', 'extraErrorNm', 0.0))
        rms_LO = self.make_tensor(get('telescope', 'extraErrorLoNm', rms_HO)).flatten() # one value, or [center, edge] of the technical field
        N_LO = min(self.config['sources_LO']['Zenith'].numel(), self.N_src) if 'sources_LO' in self.config and rms_LO.sum().item() >= 0 else 0

        if rms_HO > 0 or (N_LO > 0 and rms_LO.sum().item() > 0):
            exp_HO = scalar(get('telescope', 'extraErrorExp', -2.0))
            self.extra_error_shape    = (exp_HO, scalar(get('telescope', 'extraErrorMin', 0.0)), scalar(get('telescope', 'extraErrorMax', 0.0)))
            self.extra_error_LO_shape = (scalar(get('telescope', 'extraErrorLoExp', exp_HO)), scalar(get('telescope', 'extraErrorLoMin', 0.0)), scalar(get('telescope', 'extraErrorLoMax', 0.0)))
            if N_LO > 0:
                LO_zenith = self.make_tensor(self.config['sources_LO']['Zenith']).flatten()[:N_LO] # [arcsec]
                if rms_LO.numel() == 2: # linear interpolation between the field center and the edge of the technical field (as P3's extraErrorLoPSD)
                    FoV = scalar(get('telescope', 'TechnicalFoV', 0.0))
                    t = (LO_zenith / (0.5*FoV)).clamp(0, 1) if FoV > 0 else torch.zeros_like(LO_zenith)
                    rms_LO = rms_LO[0] + (rms_LO[1]-rms_LO[0]) * t
                elif rms_LO.numel() == 1:
                    rms_LO = rms_LO.expand(N_LO)
                else:
                    raise ValueError('extraErrorLoNm must be a scalar or [center, edge] values')
            else:
                rms_LO = rms_LO[:0] # no LO directions
            science = lambda x: torch.full((self.N_src-N_LO,), x, device=self.device, dtype=self.dtype)
            self.extra_error_nm    = torch.cat([science(rms_HO), torch.zeros(N_LO, device=self.device, dtype=self.dtype)])  # [N_src]
            self.extra_error_LO_nm = torch.cat([science(0.0), rms_LO.to(device=self.device, dtype=self.dtype)]) # [N_src]

        self.add_MCAO_cone = bool(get('sensor_HO', 'addMcaoWFsensConeError', False))
        wind_file = get('telescope', 'windPsdFile')
        self.wind_PSD_file = str(wind_file) if isinstance(wind_file, (str, Path)) and str(wind_file) not in ('', '0') else None


    def _enable_configured_addons(self):
        ''' Enable the add-on terms the config asks for, as P3 adds them automatically; the tilt and focus filters are left to the caller '''
        self.PSD_include['extra error']      = self.extra_error_nm is not None
        self.PSD_include['wind shake']       = self.wind_PSD_file is not None
        self.PSD_include['cone effect']      = bool(self.N_GS == 1 and self.is_LGS.all())
        self.PSD_include['MCAO cone effect'] = bool(self.add_MCAO_cone and self.tomography and self.is_LGS.all())


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


    def _convert_tensors_dtype(self, target_float: torch.dtype, target_complex: torch.dtype) -> None:
        ''' Helper to convert all tensor attributes to specified dtypes '''
        self._addon_cache = {}
        for attr_name, attr in self.__dict__.items():
            if torch.is_tensor(attr):
                if torch.is_floating_point(attr):
                    setattr(self, attr_name, attr.to(target_float))
                elif torch.is_complex(attr):
                    setattr(self, attr_name, attr.to(target_complex))


    def _to_double(self) -> None:
        self.is_float = False
        self._convert_tensors_dtype(torch.float64, torch.complex128)


    def _to_float(self) -> None:
        self.is_float = True
        self._convert_tensors_dtype(torch.float32, torch.complex64)


    def PistonFilter(self, f: torch.Tensor) -> torch.Tensor:
        ''' Piston mode filter for PSD '''
        x = torch.pi * self.D * f
        R = torch.special.bessel_j1(x) / x
        piston_filter = 1.0 - 4.0 * R.pow(2)
        piston_filter[..., self.nOtf_AO//2, self.nOtf_AO//2] *= 1-self.nOtf_AO % 2
        return self._stabilize(piston_filter)


    def FocusFilter(self, f: torch.Tensor) -> torch.Tensor:
        ''' Spatial filter to remove focus related errors '''
        # Compute x = π * D * √(k²)
        x = torch.pi * self.D * torch.sqrt(f)
        
        # Compute j3_term = 2*J₃(x)/x using PyTorch Bessel functions
        j3_term = self._bessel_j3(x)
        
        # Focus filter: 1 - 3 * (2*J₃(x)/x)²
        focus_filter = 1.0 - 3.0 * j3_term.pow(2)
        focus_filter[..., self.nOtf_AO//2, self.nOtf_AO//2] *= 1 - self.nOtf_AO % 2
        
        return self._stabilize(focus_filter)
        

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
        ''' Grid-dependent constant tensors of the add-on terms, computed once per grid (the cache is reset by InitGrids) '''
        if key not in self._addon_cache:
            self._addon_cache[key] = compute()
        return self._addon_cache[key]


    def _normalized_PSD_half(self, PSD_half: torch.Tensor, rms_nm: torch.Tensor | float) -> torch.Tensor:
        ''' Scale a half-grid PSD shape so that it integrates over the full grid to the requested RMS² [nm²], given per source or observation '''
        return self.make_tensor(rms_nm).view(-1, 1, 1, 1).pow(2) * PSD_half / self._half_sum(PSD_half)


    def _normalized_PSD(self, PSD_half: torch.Tensor, rms_nm: torch.Tensor | float) -> torch.Tensor:
        ''' Expand a half-grid PSD shape to the full grid and scale it to integrate to the requested RMS² [nm²], given per source or observation '''
        return self.half_PSD_to_full(self._normalized_PSD_half(PSD_half, rms_nm))


    def _PSD_norm(self) -> torch.Tensor:
        ''' PSDs are computed in [rad²/m²] at the atmosphere wavelength; this factor converts them to [nm²] of OPD per PSD pixel '''
        return (self.dk * self.wvl_atm * 1e9 / (2*torch.pi))**2


    def _full_grid_filters(self) -> tuple[torch.Tensor, torch.Tensor]:
        ''' Piston and tilt rejection filters on the full half grid [1, nOtf_y, nOtf_x], computed once per grid '''
        return self._cached('filters', lambda: self._spatial_filters(self.k))


    def TiltFilter(self) -> torch.Tensor:
        ''' Tilt rejection filter on the full PSD grid, [1, nOtf, nOtf]. The PSD is multiplied by it when tip/tilt is corrected by a separate LO loop '''
        return self.half_PSD_to_full(self._full_grid_filters()[1])


    def _extra_error_shape(self, exponent: float, k_min: float, k_max: float) -> torch.Tensor:
        ''' Unit-power piston-filtered power-law spectrum on the half grid [1, nOtf_y, nOtf_x]; k_min/k_max [1/m] restrict it (k_max <= 0 disables the upper cut) '''
        def compute():
            k = self.k
            in_band = k >= k_min if k_max <= 0 else (k >= k_min) & (k <= k_max)
            PSD = k.pow(exponent) * self._full_grid_filters()[0] * in_band
            PSD[..., self.nOtf_y//2, self.nOtf_x-1] = 0.0 # remove the DC component
            return PSD / self._half_sum(PSD) # unit total power
        return self._cached(('extra error', exponent, k_min, k_max), compute)


    def ExtraErrorPSD(self, rms_nm: torch.Tensor | float, exponent: float = -2.0, k_min: float = 0.0, k_max: float = 0.0) -> torch.Tensor:
        '''
        Piston-filtered power-law PSD [nm²] with the requested RMS WFE (as P3's extraErrorPSD).
        rms_nm is a scalar or [N_src] tensor, k_min/k_max [1/m] restrict the spectrum (k_max <= 0 disables the upper cut).
        Returns [N_src or 1, 1, nOtf, nOtf]
        '''
        return self._normalized_PSD(self._extra_error_shape(exponent, k_min, k_max), rms_nm)


    def _extra_error_PSD_half(self) -> torch.Tensor:
        ''' Configured HO + LO extra-error PSDs on the half grid in [rad²/m²], [N_src, 1, nOtf_y, nOtf_x]; only the RMS values vary between calls '''
        rms2 = lambda rms: rms.view(-1, 1, 1, 1).pow(2)
        PSD = rms2(self.extra_error_nm) * self._extra_error_shape(*self.extra_error_shape)
        if self.extra_error_LO_nm.sum() > 0:
            PSD = PSD + rms2(self.extra_error_LO_nm) * self._extra_error_shape(*self.extra_error_LO_shape)
        return PSD / self._PSD_norm()


    def _focus_error_shape(self) -> torch.Tensor:
        ''' Unit-power global focus spectrum on the half grid [1, nOtf_y, nOtf_x], computed once per grid '''
        def compute():
            PSD = 1.0 - 3.0 * self._bessel_j3(torch.pi * self.D * self.k).pow(2)
            return PSD / self._half_sum(PSD)
        return self._cached('focus error', compute)


    def FocusErrorPSD(self, rms_nm: torch.Tensor | float) -> torch.Tensor:
        ''' Full-grid PSD [nm²] of a residual global focus error with the requested RMS, [N_src or 1, 1, nOtf, nOtf] '''
        return self._normalized_PSD(self._focus_error_shape(), rms_nm)


    def _wind_shake_PSD_half(self, vibration_PSD: torch.Tensor) -> torch.Tensor:
        ''' Wind shake PSD on the AO-corrected half grid in [nm²] per PSD pixel, [N_obs, 1, nOtf_AO_y, nOtf_AO_x] '''
        data = self.make_tensor(to_little_endian(np.asarray(vibration_PSD)) if not torch.is_tensor(vibration_PSD) else vibration_PSD)
        rate, gain, delay = self.HOloop_rate.view(-1,1), self.HOloop_gain.view(-1,1), self.HOloop_delay.view(-1,1) # [N_obs, 1]

        f = 0.1 + (0.5*rate - 0.1) * torch.linspace(0, 1, int(5*rate.max().item()), device=self.device, dtype=rate.dtype) # [N_obs, N_f]
        PSD_t = self._interp1d(f, data[0], data[1]) + self._interp1d(f, data[0], data[2]) # tip + tilt
        _, rtfInt, _, _ = self.TransferFunctions(f, 1.0/rate, delay, gain)
        power = ( (rtfInt.pow(2) * PSD_t).sum(dim=-1) * (f[:,1]-f[:,0]) ).abs() # [N_obs]

        def shape(): # unit-power tilt-shaped, piston-filtered spectrum of the AO-corrected area
            piston_filter, tilt_filter = self._spatial_filters(self.k_AO)
            PSD = (1.0 - tilt_filter) * piston_filter * self.mask_corrected_AO
            return PSD / self._half_sum(PSD)
        return power.view(-1, 1, 1, 1) * self._cached('wind shake', shape)


    def WindShakePSD(self, vibration_PSD: torch.Tensor) -> torch.Tensor:
        '''
        Residual wind shake / vibration PSD [nm²] (as P3's windShakePSD).
        vibration_PSD: [3, N_f] with temporal frequencies [Hz], tip and tilt temporal PSDs [nm²/Hz].
        The temporal PSDs are filtered by the HO loop rejection transfer function and integrated. The resulting
        power is spread over the tilt-shaped, piston-filtered part of the AO-corrected area. Returns [N_obs, 1, nOtf, nOtf]
        '''
        return self.half_PSD_to_full(self.PSD_padder(self._wind_shake_PSD_half(vibration_PSD)))


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
                
        # Initialize PSD spatial frequencies
        # This assumes that pupil rotation is the same for all simulated PSFs in the batch
        if self.pupil_angle != 0.0:
            kx, ky = self._gen_grid(self.nOtf)
            rot_ang = torch.deg2rad(self.pupil_angle)
            self.kx = kx * torch.cos(rot_ang) - ky * torch.sin(rot_ang)
            self.ky = kx * torch.sin(rot_ang) + ky * torch.cos(rot_ang)
        else:
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
            # This is thing is 4D: [N_combs, N_L, kx, ky]
            m = self.make_tensor(ids[:,0])
            n = self.make_tensor(ids[:,1])
            self.N_combs = m.shape[0]
            
            self.km = self.kx_AO.repeat([self.N_combs,1,1,1]) - pdims(m/self.WFS_d_sub, 3)
            self.kn = self.ky_AO.repeat([self.N_combs,1,1,1]) - pdims(n/self.WFS_d_sub, 3)

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
        self._addon_cache = {} # grid-dependent shapes of the add-on terms are rebuilt on demand

        # To avoid initializing it without a need
        if self.PSD_include['aliasing']:
            self.PR = self.PistonFilter(torch.hypot(self.km, self.kn)) # piston filter for aliased spatial frequencies


    def Phase2OTF(self, phase: torch.Tensor | None) -> torch.Tensor:
        '''
        Compute OTF from a phase screen.
        All phase screens are sampled equally for all wavelengths (re-scaling happens later when PSFs are computed)
        However, phase screens might be different for each wavelength.
        '''
        phase_padded = self.pupil_padder(phase)
            
        fl_even = self.nOtf % 2 == 0 and phase_padded.shape[-1] % 2 == 0 # TODO: what's the hell is this?
        phase_padded = phase_padded[:-1, :-1] if fl_even else phase_padded # to center-align if number of pixels is even

        OTF = self._FFTAutoCorr(phase_padded)
        OTF = OTF.view(1, 1, *OTF.shape) if OTF.ndim == 2 else OTF
        
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

        if self.is_float:
            self._to_float()
        # Update grids used for computation of PSDs and OTFs. This is necessary when the sampling has changed.
        if grids:
            self.InitGrids()
        # Update pupil and apodizer masks
        if pupils:
            self.InitPupils()
        elif grids and self.pupil is not None:
            if update_static_OTF:
                # Grid-dependent static OTF must be refreshed when pupil data did not change.
                self.UpdateStaticOTF()
            else:
                # Caller will set OTF_static via ComputeStaticOTF(phase); skip the expensive
                # pupil FFT but still refresh the pupil_padder which depends on sampling_min.
                phase_size = self.pupil.shape[-1]
                self.pupil_padder = torch.nn.ZeroPad2d(
                    int(round(phase_size * self.sampling_min / 2 - phase_size / 2))
                )
        
        # If the number of sources have changed, reinitialize the tomography projector
        if (self.tomography and tomography) or (self.tomography and grids):
            if self.on_axis:
                # Tomographic src to DM projector for on-axis is just an identity matrix
                self.P_beta_DM = torch.ones([self.N_src, self.nOtf_AO_y, self.nOtf_AO_x, 1, self.N_DM], dtype=torch.complex64, device=self.device) * pdims(self.mask_corrected_AO, 2)
                # TODO: comment
                self.P_opt = torch.ones([self.N_obs, self.nOtf_AO_y, self.nOtf_AO_x, 1, self.N_L], dtype=torch.complex64, device=self.device)
            else:
                self.OptimalDMProjector(inv_method=self.inversion_method)
                self.DMProjector()


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
            'cone effect',      # focal anisoplanatism of a single LGS (SLAO), computed only for a single LGS
            'MCAO cone effect', # volume not sensed by the LGS WFSs of a tomographic system, computed only when [sensor_HO] addMcaoWFsensConeError is set
            'extra error',      # [telescope] extraErrorNm / extraErrorLoNm power-law PSDs
            'wind shake',       # [telescope] windPsdFile temporal PSD filtered by the HO loop
            'tilt filter',      # tip/tilt rejection filter applied to the PSD when a separate LO loop corrects tip/tilt
            'focus error'       # residual global focus error with the RMS given by focus_error_nm
        ]
        # The P3-style add-ons are excluded unless enabled explicitly or by the config entries (see _enable_configured_addons)
        PSD_entries_off = ['Moffat', 'cone effect', 'MCAO cone effect', 'extra error', 'wind shake', 'tilt filter', 'focus error']

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
                 dtype: torch.dtype = torch.float32):
        
        super().__init__()
        
        self.device = device
        self.oversampling = oversampling
        self.pupil = pupil
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
        
        # Default inversion method for tomographic reconstruction
        self.inversion_method = 'lstsq'
        
        # Piston filters
        self.piston_filter = None # piston mode filter in the AO-corrected freqs domain
        self.apodizer = None
        self.PR = None # piston mode filter in aliased freqs domain

        # P3-style add-ons: the wind-shake temporal PSD is loaded with the pupils, the residual focus RMS is set by the caller (e.g. an LO loop)
        self.vibration_PSD = None
        self.focus_error_nm = None
        self._addon_cache = {}

        # Read the config data and initialize the AO system
        self.Update(
            config = AO_config,
            grids  = True,
            pupils = True,
            tomography = True # Try updating the tomographic reconstructors. If AO is not tomographic, the model will figure it out and skip the update
        )

        if PSD_include is None: # unless chosen explicitly, the add-ons follow the config (P3 behaviour)
            self._enable_configured_addons()
        

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


    def OptimalDMProjector(self, inv_method='lstsq'):
        h_dm = self.h_DM.view(1, 1, 1, self.N_DM)
        h = self.h_().view(self.N_obs, 1, 1, 1, self.N_L)
        
        opt_w = self.DM_opt_weight.view(self.N_obs, 1, 1, self.N_optdir, 1, 1)

        theta_x = self.DM_opt_dir_x.view(self.N_obs, 1, 1, self.N_optdir)
        theta_y = self.DM_opt_dir_y.view(self.N_obs, 1, 1, self.N_optdir)

        f = theta_x * pdims(self.kx_AO, 1) + theta_y * pdims(self.ky_AO, 1)
        P_L = torch.exp( 2j*torch.pi*h * pdims(f,1) ).unsqueeze(-2) # [N_obs, nOtf_AO, nOtf_AO, N_optdir, 1, N_L]
        
        if self.AO_type == 'LTAO':
            self.P_opt = (P_L * opt_w).sum(dim=3) # [N_obs, nOtf_AO, nOtf_AO, 1, N_L]
            return
        
        mask = self.mask_corrected_AO.view(1, self.nOtf_AO_y, self.nOtf_AO_x, 1)
        P_DM   = torch.exp( 2j*torch.pi*h_dm * pdims(f*mask,1) ).unsqueeze(-2) # [N_obs, nOtf_AO, nOtf_AO, N_optdir, 1, N_DM]
        P_DM_t = torch.conj( P_DM.permute(0, 1, 2, 3, 5, 4) )    # [N_obs, nOtf_AO, nOtf_AO, N_optdir, N_DM, 1]

        mat1   = ((P_DM_t @ P_L)  * opt_w).sum(dim=3)  # [N_obs, nOtf_AO, nOtf_AO, N_DM, N_L]
        to_inv = ((P_DM_t @ P_DM) * opt_w).sum(dim=3)  # [N_obs, nOtf_AO, nOtf_AO, N_DM, N_DM]
        
        if inv_method == 'lstsq':
            # Solve to_inv @ P_opt = mat1 directly for P_opt (equivalent to P_opt = pinv(to_inv) @ mat1),
            # since to_inv is square [N_DM, N_DM] and mat1 is [N_DM, N_L], their sizes at dim -2 already match
            self.P_opt = torch.linalg.lstsq(to_inv, mat1, rcond=1e-2).solution # [N_obs, nOtf_AO, nOtf_AO, N_DM, N_L]

        elif inv_method == 'pinv':
            mat2 = torch.linalg.pinv(to_inv, rcond=1e-2) # Last 2 dimensions are inverted
            self.P_opt = mat2 @ mat1 # [N_obs, nOtf_AO, nOtf_AO, N_DM, N_L]


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


    # TODO: accelerate this function
    def NoiseGain(self, nF: int = 1000):
        if not self.approx_noise_gain:
            Ts = 1.0 / self.HOloop_rate  # sampling time
            delay    = self.HOloop_delay # latency between the measurement and the correction
            loopGain = self.HOloop_gain
            
            f = torch.zeros([self.N_obs, nF], device=self.device)
            
            # TODO: this is slow
            for i in range(self.N_obs):
                f[i,:] = torch.logspace(-3, torch.log10(0.5/Ts[i]).item(), nF)

            # NOTE: faster version: Create end values for logspace for all sources at once
            # log_end = torch.log10(0.5/Ts.flatten())  # [N_obs]
            # log_start = torch.full((self.N_obs,), -3, device=self.device)
            # steps = torch.linspace(0, 1, nF, device=self.device).unsqueeze(0)  # [1, nF]
            # f = 10 ** (log_start.unsqueeze(1) + (log_end.unsqueeze(1) - log_start.unsqueeze(1)) * steps)

            _, _, _, ntfInt = self.TransferFunctions(f, min_2d(Ts), min_2d(delay), min_2d(loopGain))
            self.noise_gain = torch.trapz(ntfInt.abs().pow(2), f, dim=1).view(self.N_obs, 1, 1) * 2*Ts
        else:
            # An average value of the noise transfer function as a function of the delay in frames for an integrator gain of 0.5
            self.noise_gain = torch.min(torch.tensor(0.8, device=self.device), 0.4 + 0.1333 * self.HOloop_delay).pow(2).view(self.N_obs, 1, 1)
    
     
    def Controller(self):
        #nTh = 1
        idim = lambda x: x.view(self.N_obs, 1, 1, self.N_L)
        
        vy = idim(self.vy)
        vx = idim(self.vx)
        kx = self.kx_AO.unsqueeze(-1) # add atmo. layers dimension: [N_src, nOtf_AO, nOtf_AO, nL]
        ky = self.ky_AO.unsqueeze(-1) # add atmo. layers dimension: [N_src, nOtf_AO, nOtf_AO, nL]
        
        Ts =  1.0 / self.HOloop_rate  # sampling time
        delay     = self.HOloop_delay # latency between the measurement and the correction
        loop_gain = self.HOloop_gain
        
        #TODO: implement nTh to incorparate the uncertainty in wind direction
        thetaWind = torch.zeros(1, device=self.device) #torch.linspace(0, 2*torch.pi-2*torch.pi/nTh, nTh) TODO: remove GPU initialization of 0
        costh = torch.cos(thetaWind) # stays for the uncertainty in the wind diirection

        fi = -(vx*kx + vy*ky)*costh # [N_src, nOtf_AO, nOtf_AO, nL]

        _, _, atfInt, ntfInt = self.TransferFunctions(fi, pdims(Ts,3), pdims(delay,3), pdims(loop_gain,3))
        
        # AO transfer function
        self.h1 = idim(self.Cn2_weights) * atfInt #/nTh
        self.h2 = idim(self.Cn2_weights) * atfInt.abs().pow(2) #/nTh
        self.hn = idim(self.Cn2_weights) * ntfInt.abs().pow(2) #/nTh

        self.h1 = self.h1.sum(dim=-1) # sum over the atmospheric layers
        self.h2 = self.h2.sum(dim=-1) 
        self.hn = self.hn.sum(dim=-1) 


    def ReconstructionFilter(self, WFS_noise_var: torch.Tensor, MV: int = 0): # TODO: make MV switchable
        Av = torch.sinc(self.WFS_d_sub*self.kx_AO) * torch.sinc(self.WFS_d_sub*self.ky_AO) * torch.exp(1j*torch.pi*self.WFS_d_sub*(self.kx_AO+self.ky_AO))
        self.SxAv = ( 2j*torch.pi*self.kx_AO*self.WFS_d_sub*Av ).view(1, self.nOtf_AO_y, self.nOtf_AO_x) # Same for a given AO regime
        self.SyAv = ( 2j*torch.pi*self.ky_AO*self.WFS_d_sub*Av ).view(1, self.nOtf_AO_y, self.nOtf_AO_x)
        
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


    def AnisoplanatismPSD(self):
        ''' Angular anisoplanatism PSD of a non-tomographic system on the AO half grid [N_src, nOtf_AO_y, nOtf_AO_x] (as P3's anisoplanatismPSD, diagnostic only) '''
        return 2*(self.Cn2_weights.sum(dim=-1).view(-1, 1, 1) - self._anisoplanatism_phasor().real) * self.W_atm * self.mask_corrected_AO


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

        # Compute von Karman spectrum for aliased spatial frequencies
        W_mn = self.VonKarmanSpectrum(self.r0_().view(1, self.N_obs), self.L0.abs().view(1, self.N_obs), self.km**2 + self.kn**2) * self.PR
        
        Q = (Rx1*self.km + Ry1*self.kn) * torch.sinc(self.WFS_d_sub*self.km) * torch.sinc(self.WFS_d_sub*self.kn)
        tf = self.h1.unsqueeze(0).unsqueeze(-1) # [N_combs, N_src, nOtf_AO, n_Otf_AO, nL]

        # Add aliasing dimension and more
        vx = self.vx.view(1, self.N_obs, 1, 1, self.N_L)
        vy = self.vy.view(1, self.N_obs, 1, 1, self.N_L)
        Cn2_weights = self.Cn2_weights.view(1, self.N_obs, 1, 1, self.N_L)
        
        # Adds  atmospheric layers dimension
        km, kn = self.km.unsqueeze(-1), self.kn.unsqueeze(-1)
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


    def ConeEffectPSD(self, n_phases: int = 5) -> torch.Tensor:
        '''
        Focal anisoplanatism (cone effect) PSD of a single LGS on the full half grid [N_obs, nOtf_y, nOtf_x] (as P3's focalAnisoplanatismPSD).
        Through the cone, a layer sinusoid of frequency f is sensed as a sinusoid of frequency f*(H-h)/H. The RMS fraction left after
        subtracting the best-scaled sensed sinusoid over the pupil diameter, averaged over n_phases phase offsets, is squared and filters
        the atmospheric spectrum of every layer. Frequencies that fall beyond the correction band once compressed are left to the fitting
        error. P3 samples the pupil with 1001 points; here the pupil means of the sinusoids are analytical (mean of cos(2 pi f x) = sinc(f D)).
        '''
        H = self.LGS_height # [N_obs, 1]
        g = ((H - self.Cn2_heights) / H).view(self.N_obs, 1, self.N_L, 1) # frequency compression through the cone
        f = (torch.arange(self.nOtf_x, device=self.device, dtype=self.dtype) * self.dk).view(1, -1, 1, 1) # [1, N_f, 1, 1] radial frequencies
        phi = (2*torch.pi * torch.arange(n_phases, device=self.device, dtype=self.dtype) / n_phases).view(1, 1, 1, -1)
        m = lambda freq: torch.sinc(freq * self.D) # pupil mean of cos(2*pi*freq*x) for x in [-D/2, D/2]

        in_band = (f > 1e-5) & (f * g <= self.kc) # the DC and the frequencies beyond the correction band once compressed are masked at the end
        f_safe = torch.where(f > 1e-5, f, torch.full_like(f, self.dk)) # keeps the variances below positive, i.e. the gradients finite
        f_cone = f_safe * g
        sin2_phi, cos_2phi = torch.sin(phi)**2, torch.cos(2*phi)
        var_ref  = 0.5 - 0.5*cos_2phi*m(2*f_safe) - sin2_phi*m(f_safe)**2 # pupil variance of sin(2*pi*f*x + phi)
        var_cone = 0.5 - 0.5*cos_2phi*m(2*f_cone) - sin2_phi*m(f_cone)**2
        cov = 0.5*(m(f_safe - f_cone) - cos_2phi*m(f_safe + f_cone)) - sin2_phi*m(f_safe)*m(f_cone)
        correlation = cov / torch.sqrt((var_ref * var_cone).clamp_min(1e-24))
        coeff = torch.sqrt((2.0 - 2.0*correlation).clamp_min(1e-12)).mean(dim=-1) # [N_obs, N_f, N_L] residual RMS relative to the input sinusoid
        coeff = coeff * in_band[..., 0]

        # Linear interpolation of the radial coefficients on the 2D grid (clamped at the edge, as np.interp); the grid indices are constant
        def radial_indices():
            t  = (self.k[0] / self.dk).clamp(0, self.nOtf_x - 1).flatten()
            i0 = t.floor().long().clamp(max=self.nOtf_x - 2)
            return i0, t - i0
        i0, w = self._cached('cone effect', radial_indices)
        coeff = coeff.permute(0, 2, 1) # [N_obs, N_L, N_f]
        c0 = coeff[..., i0]
        c = (c0 + (coeff[..., i0+1] - c0) * w).view(self.N_obs, self.N_L, self.nOtf_y, self.nOtf_x)

        W_atm = self.VonKarmanSpectrum(self.r0_(), self.L0.abs(), self.k2) # [N_obs, nOtf_y, nOtf_x]
        return (self.Cn2_weights.view(self.N_obs, self.N_L, 1, 1) * c.pow(2)).sum(dim=1) * W_atm


    def MCAOConePSD(self, PSD_residual: torch.Tensor) -> torch.Tensor:
        '''
        PSD of the turbulence that the LGS WFSs of a tomographic system do not sense because of the cone effect, on the AO half grid
        [N_src, N_wvl, nOtf_AO_y, nOtf_AO_x] (as P3's mcaoWFsensConePSD). For every layer and source, the part of the atmospheric spectrum
        corrected in the AO area (atmosphere minus the residual PSD) is low-pass filtered above the frequency set by the angle between the
        source and the sensed volume, and scaled by the equivalent sensed aperture.
        '''
        src_zenith = self.config['sources_science']['Zenith'].flatten().view(-1, 1) # [N_src, 1] in [arcsec]
        GS_zenith  = self.config['sources_HO']['Zenith'].view(self.N_obs, -1).amax(dim=-1, keepdim=True) # [N_obs, 1]
        H = self.LGS_height
        h = self.Cn2_heights # [N_obs, N_L]

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
        cos_k = self._cached('MCAO cone effect', lambda: torch.cos(torch.pi * self.k_AO / (f_s/2))) # [1, nOtf_AO_y, nOtf_AO_x]
        z_pole = torch.exp(2*torch.pi*f_cut / f_s).view(*f_cut.shape, 1, 1)
        high_pass = 1 - (1 - z_pole)**2 / (1 + z_pole**2 - 2*z_pole*cos_k)
        gain = (high_pass * (D_eq/self.D).view(*D_eq.shape, 1, 1)**2).clamp_min(0) * valid.view(*valid.shape, 1, 1) # [N_src, N_L, ny, nx]
        gain = (self.Cn2_weights.view(self.N_obs, self.N_L, 1, 1) * gain).sum(dim=1).unsqueeze(1) # [N_src, 1, ny, nx]

        W_atm = self.VonKarmanSpectrum(self.r0_(), self.L0.abs(), self.k2_AO).unsqueeze(1) # [N_obs, 1, ny, nx]
        return gain * (W_atm - PSD_residual.real).clamp_min(0) * self.piston_filter * self.mask_corrected_AO # the tomographic noise term is complex-valued


    def JitterKernel(self, Jx: torch.Tensor, Jy: torch.Tensor, Jxy: torch.Tensor):
        # Assuming Jxy is in [deg], convert it to [rad]
        cos_theta = torch.cos( torch.deg2rad(Jxy) )
        sin_theta = torch.sin( torch.deg2rad(Jxy) )

        U_prime = self.U * cos_theta + self.V * sin_theta
        V_prime = -self.U * sin_theta + self.V * cos_theta

        Djitter = pdims(self.u_max * self.jitter_norm_fact, 2) * ( (Jx*U_prime)**2 + (Jy*V_prime)**2 )
        return torch.exp(-0.5 * Djitter) #TODO: cover the Nyquist sampled case? But check maybe it is automatic, already?
    

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


    def TomographicReconstructors(self, WFS_noise_var, inv_method='lstsq'):
        '''        
        Note that if all simulated sources use the same atmospheric profile, r0, L0, and noise variance,
        then it's possible to compute one tomographic reconstructor for all simulated sources.
        For example, this is the case when all objects are within one FoV and belong to one observation
        '''
        h = self.h_().view(self.N_obs, 1, 1, 1, self.N_L)
        
        kx = pdims(self.kx_AO, 2)
        ky = pdims(self.ky_AO, 2)
        GS_dirs_x = self.GS_dirs_x.view(self.N_obs, 1, 1, self.N_GS, 1)
        GS_dirs_y = self.GS_dirs_y.view(self.N_obs, 1, 1, self.N_GS, 1)
        
        diag_mask = lambda N: torch.eye(N, device=self.device).view(1,1,1,N,N).expand(self.N_obs, self.nOtf_AO_y, self.nOtf_AO_x, -1, -1)
        
        M = 2j*torch.pi*self.k_AO * torch.sinc(self.WFS_d_sub*self.kx_AO) * torch.sinc(self.WFS_d_sub*self.ky_AO)
        M = pdims(M, 2).expand(self.N_obs, self.nOtf_AO_y, self.nOtf_AO_x, self.N_GS, self.N_GS) * diag_mask(self.N_GS) # [N_obs, nOtf_y, nOtf_x, nGS, nGS]
        P = torch.exp( 2j*torch.pi*h * (kx*GS_dirs_x + ky*GS_dirs_y) ) # [N_obs, nOtf_y, nOtf_x, nGS, nL]
        MP   = torch.einsum('nwhik,nwhkj->nwhij', M, P)  # [N_obs, nOtf_y, nOtf_x, nGS, nL]
        MP_t = torch.conj(MP.permute(0,1,2,4,3))

        expand_dims = lambda x_, N: torch.diag_embed(x_).view(self.N_obs, 1, 1, N, N).expand(self.N_obs, self.nOtf_AO_y, self.nOtf_AO_x, N, N)
        
        # Note, that size of WFS_noise_var == N_obs
        WFS_noise_variance = WFS_noise_var.to(dtype=MP.dtype) # Convert to complex
        # As previosuly mentioned, if the same tomo reconstructor is used for all targets, then WFS_noise_variance also must be the same for all targets
        self.C_b = expand_dims( WFS_noise_variance[:self.N_obs], self.N_GS )
        kernel = self.VonKarmanSpectrum(self.r0_().to(dtype=MP.dtype), self.L0.abs(), self.k2_AO) * self.piston_filter
        self.C_phi = pdims(kernel, 2) * expand_dims(self.Cn2_weights, self.N_L)

        # Inversion happens relative to the last two dimensions of these tensors
        if inv_method == 'pinv':
            self.W_tomo = (self.C_phi @ MP_t) @ torch.linalg.pinv(MP @ self.C_phi @ MP_t + self.C_b, rcond=1e-3) #1e-4

        elif inv_method == 'lstsq':
            A = (MP @ self.C_phi @ MP_t + self.C_b).transpose(-2, -1)
            B = (self.C_phi @ MP_t).transpose(-2, -1)
            # Solve the least squares problem for W^T, since we deal with W*A = B
            self.W_tomo = torch.linalg.lstsq(A, B, rcond=1e-2).solution.transpose(-2, -1)           
        else:
            raise ValueError('Unknown inversion method specified.') 
         
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

        uxx = self.kx2_AO
        uxy = self.kxy_AO
        uyy = self.ky2_AO

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
        
        # Otherwise, jitter must be muted since PSD2PSF always applies JitterKernel()
        Jx_old, Jy_old, Jxy_old = self.Jx.clone(), self.Jy.clone(), self.Jxy.clone()

        self.Jx  = torch.zeros_like(Jx_old)
        self.Jy  = torch.zeros_like(Jy_old)
        self.Jxy = torch.zeros_like(Jxy_old)

        self.PSF_open_loop = self.PSD2PSF(PSD, OTF_static)

        self.Jx, self.Jy, self.Jxy = Jx_old, Jy_old, Jxy_old

        return self.PSF_open_loop # [N_obs, N_wvl, N_pix, N_pix]


    def ComputePSD(self, update_addons_only: bool = False):
        '''
        Residual PSD [N_src, N_wvl, nOtf, nOtf] in [nm²] per pixel. All terms are computed on the half grid, in [rad²/m²] at the atmosphere
        wavelength, and expanded to the full grid only at the end. With update_addons_only the core PSD of the previous call (fitting,
        AO-corrected terms, cone effects, kept in PSD_core) is reused and only the add-ons (wind shake, tilt filter, extra error, focus error)
        are re-applied, e.g. after setting focus_error_nm.
        '''
        if all(not value for value in self.PSD_include.values()):
            self.PSD = torch.zeros([self.N_src, self.N_wvl, self.nOtf, self.nOtf], device=self.device)
            return self.PSD

        if update_addons_only:
            PSDs = self.PSDs if self.retain_PSDs and hasattr(self, 'PSDs') else {entry: torch.zeros(1, device=self.device) for entry in self.PSD_include}
            return self._apply_addons(self.PSD_core, PSDs)

        if self.PSD_include['Moffat']:
            amp   = pdims(self.amp,   2)
            b     = pdims(self.b,     2)
            alpha = pdims(self.alpha, 2)
            beta  = pdims(self.beta,  2)
            ratio = pdims(self.ratio, 2)
            theta = pdims(self.theta, 2)

        if self.PSD_include['WFS noise'] or self.PSD_include['spatio-temporal'] or self.PSD_include ['aliasing']:

            WFS_noise_var = (self.dn.view(self.N_obs,-1) + self.NoiseVariance()).abs() # [rad^2] at atmo wvl
            # TODO: check the wind direction sign conventions! sin and cos might be swapped and - should b in front of wind speed
            self.vx = self.wind_speed * torch.cos( torch.deg2rad(self.wind_dir) )
            self.vy = self.wind_speed * torch.sin( torch.deg2rad(self.wind_dir) )

            self.freq_t = self.vx.view(self.N_obs, 1, 1, self.N_L) * pdims(self.kx_AO, 1) + \
                          self.vy.view(self.N_obs, 1, 1, self.N_L) * pdims(self.ky_AO, 1) # [N_src, nOtf_AO, nOtf_AO, nL]

            self.Controller()
            self.ReconstructionFilter(WFS_noise_var)
                        
            if self.tomography:
                self.TomographicReconstructors(WFS_noise_var, inv_method=self.inversion_method)
                if not self.on_axis:
                    self.DMProjector()

        # Put all contributiors together and sum up the resulting PSD
        PSDs = {entry: torch.zeros(1, device=self.device) for entry in self.PSD_include}

        if self.PSD_include['fitting']:
            PSDs['fitting'] = self.VonKarmanPSD().unsqueeze(1)
    
        if self.PSD_include['WFS noise']:
            PSDs['WFS noise'] = self.NoisePSD(WFS_noise_var).unsqueeze(1)
        
        if self.PSD_include['spatio-temporal']:
            PSDs['spatio-temporal'] = self.SpatioTemporalPSD().unsqueeze(1)
        
        if self.PSD_include['aliasing']:
            PSDs['aliasing'] = self.AliasingPSD().unsqueeze(1)
        
        if self.PSD_include['chromatism']:
            PSDs['chromatism'] = self.ChromatismPSD() # no need to add dimension since it's polychromatic already

        if self.PSD_include['Moffat']:
            PSDs['Moffat'] = self.MoffatPSD(amp.abs(), b, alpha, beta, ratio, theta).unsqueeze(1)

        if self.PSD_include['diff. refract']:
            PSDs['diff. refract'] = self.DifferentialRefractionPSD()

        # Resulting dimensions are: [N_scr, N_wvl, nOtf_AO, nOtf_AO]
        PSD_AO = PSDs['WFS noise'] + PSDs['spatio-temporal'] + PSDs['aliasing'] + PSDs['chromatism'] + PSDs['Moffat'] + PSDs['diff. refract']

        # The cone effects are part of the core PSD, as in P3's powerSpectrumDensity (the MCAO one depends on the residual PSD in the AO area)
        if self.PSD_include['MCAO cone effect'] and self.add_MCAO_cone and self.tomography and self.is_LGS.all():
            PSDs['MCAO cone effect'] = self.MCAOConePSD(PSD_AO)
            PSD_AO = PSD_AO + PSDs['MCAO cone effect']

        PSD = PSDs['fitting'] + self.PSD_padder(PSD_AO)

        if self.PSD_include['cone effect'] and self.N_GS == 1 and self.is_LGS.all():
            PSDs['cone effect'] = self.ConeEffectPSD().unsqueeze(1) # full grid, as P3's SLAO case
            PSD = PSD + PSDs['cone effect']

        self.PSD_core = PSD # half grid [N_src, N_wvl, nOtf_y, nOtf_x] in [rad²/m²], reused by ComputePSD(update_addons_only=True)
        return self._apply_addons(PSD, PSDs)


    def _apply_addons(self, PSD: torch.Tensor, PSDs: dict) -> torch.Tensor:
        ''' Add the P3-style add-ons to the half-grid core PSD (P3's order: wind shake, tilt filter, extra error; then the focus error) and expand to the full grid '''
        if self.PSD_include['wind shake'] and self.vibration_PSD is not None:
            PSDs['wind shake'] = self._wind_shake_PSD_half(self.vibration_PSD) / self._PSD_norm()
            PSD = PSD + self.PSD_padder(PSDs['wind shake'])

        if self.PSD_include['tilt filter']:
            PSDs['tilt filter'] = self._full_grid_filters()[1] # the (dimensionless) filter itself: tip/tilt is left to a separate LO loop
            PSD = PSD * PSDs['tilt filter']

        if self.PSD_include['extra error'] and self.extra_error_nm is not None:
            PSDs['extra error'] = self._extra_error_PSD_half()
            PSD = PSD + PSDs['extra error']

        if self.PSD_include['focus error'] and self.focus_error_nm is not None:
            PSDs['focus error'] = self.make_tensor(self.focus_error_nm).view(-1, 1, 1, 1).pow(2) * self._focus_error_shape() / self._PSD_norm()
            PSD = PSD + PSDs['focus error']

        # Removing the DC component from half-PSD
        PSD[..., self.nOtf_y//2, self.nOtf_x-1] = 0.0

        # All PSDs are computed in [rad^2] at the atmospheric wvls and then normalized to [nm^2] OPD at science wvl
        # Recover the full-size PSD from the half-sized one
        self.PSD = self.half_PSD_to_full(PSD * self._PSD_norm()) # [nm^2]

        if self.retain_PSDs:
            self.PSDs = PSDs  # store all generated PSDs for debugging and visualization purposes

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
    
    
    # def complex_grid_sample(self, x, grid, mode='bicubic'):
    #     """
    #     x    : [B, 1, H, W] complex, centered
    #     grid : [B, Hout, Wout, 2] in normalized source coordinates
    #         last dim = (x, y)
    #     """
    #     xr = grid_sample(x.real, grid, mode=mode, padding_mode='zeros', align_corners=True)
    #     xi = grid_sample(x.imag, grid, mode=mode, padding_mode='zeros', align_corners=True)
    #     return torch.complex(xr, xi)


    # def detector_OTF_grid(self, N):
    #     """
    #     Centered detector OTF coordinates consistent with fftshift/ifftshift.
    #     Returned coordinates are normalized so that they live roughly in [-1, 1).
    #     """
    #     u = 2.0 * torch.fft.fftshift(torch.fft.fftfreq(N, d=1.0, device=self.device, dtype=self.dtype))
    #     V_det, U_det = torch.meshgrid(u, u, indexing='ij')
    #     return U_det, V_det


    # def pixel_MTF_detector_grid(self, U_det, V_det):
    #     """
    #     Square-pixel detector MTF on the detector OTF grid.
    #     torch.sinc(x) = sin(pi x)/(pi x)
    #     """
    #     mtf = torch.sinc(0.5 * U_det) * torch.sinc(0.5 * V_det)
    #     return mtf.unsqueeze(0).unsqueeze(0)  # [1,1,N,N]


    # def zoom_OTF_complex(self, OTF, scale, out_size, mode='bicubic'):
    #     """
    #     OTF     : [N_src, 1, H, W] complex, centered, sampled on the source normalized grid
    #             used in this codebase (self.U/self.V ~ linspace(-1,1,self.nOtf))
    #     scale   : >1 compresses PSF in image space
    #     out_size: detector-grid size

    #     Returns : [N_src, 1, out_size, out_size] complex, centered
    #     """

    #     # Detector OTF coordinates on the OUTPUT Fourier grid
    #     U_det, V_det = self.detector_OTF_grid(out_size)

    #     # Image compression by scale s <=> OTF(u,v) evaluated at (u/s, v/s)
    #     # Since the source OTF grid is normalized to [-1,1] in this codebase,
    #     # these are directly the grid_sample coordinates when align_corners=True.
    #     grid = torch.stack((U_det/scale, V_det/scale), dim=-1)  # [N, N, 2]
    #     grid = grid.unsqueeze(0).expand(self.N_src, -1, -1, -1) # [N_src, N, N, 2]

    #     return self.complex_grid_sample(OTF, grid, mode=mode)


    # def OTF2PSF(self, OTF):
    #     """
    #     OTF: [B, Nwvl or 1, H, W] complex, centered
    #     Returns detector-sampled PSFs [B, Nwvl, N_pix, N_pix]
    #     """
    #     out = []
    #     eps = torch.finfo(OTF.real.dtype).eps

    #     for i in range(self.wvl.shape[-1]):
    #         n = 0 if OTF.shape[1] == 1 else i

    #         # effective image-plane compression factor for this wavelength
    #         s = float(self.nOtfs[i]) / float(self.N_pix)

    #         OTF_i = OTF[:, n:n+1, :, :]  # [B,1,H,W], centered

    #         # 1) resample OTF onto the detector Fourier grid
    #         OTF_det = self.zoom_OTF_complex(OTF_i, scale=s, out_size=self.N_pix, mode='bicubic')

    #         # 2) apply square-pixel MTF on THAT detector grid
    #         U_det, V_det = self.detector_OTF_grid(self.N_pix)
    #         pix_mtf = self.pixel_MTF_detector_grid(U_det, V_det)
    #         OTF_det = OTF_det * pix_mtf

    #         # 3) preserve DC exactly (helps keep total flux stable after interpolation)
    #         dc_src  = OTF_i[..., OTF_i.shape[-2] // 2, OTF_i.shape[-1] // 2].real
    #         dc_det  = OTF_det[..., self.N_pix // 2, self.N_pix // 2].real.clamp_min(eps)
    #         OTF_det = OTF_det * (dc_src / dc_det).unsqueeze(-1).unsqueeze(-1)

    #         # 4) detector-sampled PSF
    #         PSF_i = torch.fft.fftshift( torch.fft.ifft2( torch.fft.ifftshift(OTF_det, dim=(-2, -1)), dim=(-2, -1) ), dim=(-2, -1) ).real

    #         # numerical cleanup
    #         PSF_i = PSF_i.clamp_min(0.0)

    #         out.append(PSF_i)

    #     return torch.cat(out, dim=1)


    # def OTF2PSF_no_interp(self, OTF):
    #     ''' Debug version of OTF2PSF that only center-crops without interpolation '''
    #     PSF_big = fft.fftshift(fft.ifft2(fft.ifftshift(OTF, dim=(-2,-1))), dim=(-2,-1)).abs()
        
    #     PSF = []
    #     for i in range(self.wvl.shape[-1]):
    #         transform = transforms.CenterCrop((self.nOtfs[i], self.nOtfs[i]))
    #         n = 0 if PSF_big.shape[1] == 1 else i # when there is no chromatic OTF
            
    #         PSF_cropped = transforms.CenterCrop((self.N_pix, self.N_pix))(
    #             transform(PSF_big[:,n,...])
    #         ).unsqueeze(1)
    #         PSF.append(PSF_cropped)

    #     return torch.hstack(PSF)


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
        # self.OTF = OTF_static
        self.OTF_norm = pdims(self.OTF.abs()[..., self.nOtf//2, self.nOtf//2], 2)
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

        for entry in self.PSD_include:
            PSD = self.PSDs[entry]

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


    def _to_device_recursive(self, obj, device):
        if isinstance(obj, torch.Tensor):
            if obj.device != device:
                if isinstance(obj, nn.Parameter):
                    obj.data = obj.data.to(device)
                    if obj.grad is not None:
                        obj.grad = obj.grad.to(device)
                else:
                    obj = obj.to(device)
                    
        elif isinstance(obj, nn.Module):
            obj.to(device)
            
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                self._to_device_recursive(item, device)
                
        elif isinstance(obj, dict):
            for item in obj.values():
                self._to_device_recursive(item, device)
                
        return obj


    def to(self, *args, **kwargs):
        '''
        Moves and/or casts the parameters and buffers (PyTorch-style).
        
        Signature variations:
            to(device=None, dtype=None, non_blocking=False)
            to(dtype, non_blocking=False)
            to(tensor, non_blocking=False)
            
        Args:
            device (torch.device or str): the desired device
            dtype (torch.dtype): the desired floating point or complex dtype
            tensor (torch.Tensor): Tensor whose dtype and device are the desired dtype and device
            non_blocking (bool): if True and source is in pinned memory, the copy will be asynchronous
            
        Returns:
            self
        '''
        device = None
        dtype = None
        non_blocking = False
        
        # Parse positional arguments
        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, torch.Tensor):
                # Extract device and dtype from tensor
                device = arg.device
                dtype = arg.dtype
            elif isinstance(arg, (torch.device, str)):
                device = arg
            elif isinstance(arg, torch.dtype):
                dtype = arg
            elif isinstance(arg, dict):
                # Handle dict with device/dtype keys
                device = arg.get('device', device)
                dtype = arg.get('dtype', dtype)
                non_blocking = arg.get('non_blocking', non_blocking)
        elif len(args) == 2:
            device, dtype = args
        elif len(args) > 2:
            raise TypeError(f'to() received too many arguments ({len(args)})')
        
        # Parse keyword arguments (override positional if both provided)
        device = kwargs.get('device', device)
        dtype = kwargs.get('dtype', dtype)
        non_blocking = kwargs.get('non_blocking', non_blocking)
        
        # Handle dtype conversion using our optimized methods
        if dtype is not None:
            if dtype in (torch.float32, torch.complex64):
                self._to_float()
            elif dtype in (torch.float64, torch.complex128):
                self._to_double()
            else:
                raise TypeError(f'Unsupported dtype: {dtype}. Use float32, float64, complex64, or complex128.')
        
        # Handle device conversion
        if device is not None:
            if isinstance(device, str):
                device = torch.device(device)
            if self.device != device:
                self.device = device
                for name, attr in self.__dict__.items():
                    new_attr = self._to_device_recursive(attr, device)
                    if new_attr is not attr:
                        setattr(self, name, new_attr)
        
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
        """Explicitly clean up GPU memory and release resources"""
        # Clear cached tensors and computations
        if hasattr(self, 'OTF_static'):
            del self.OTF_static
        if hasattr(self, 'OTF_static_default'):
            del self.OTF_static_default
        if hasattr(self, 'OTF'):
            del self.OTF
        if hasattr(self, 'PSDs'):
            del self.PSDs
        
        # Clear pupil-related tensors
        if hasattr(self, 'pupil'):
            del self.pupil
        if hasattr(self, 'apodizer'):
            del self.apodizer
        if hasattr(self, 'pupil_padder'):
            del self.pupil_padder
        
        # Clear large attribute tensors
        attrs_to_clear = [
            'wvl', 'F', 'bg', 'dx', 'dy', 'Jx', 'Jy', 'Jxy','r0', 'L0', 'wind_speed', 'wind_dir',
            'Cn2_weights', 'Cn2_heights', 'h', 'GS_dirs_x', 'GS_dirs_y', 'src_dirs_x', 'src_dirs_y',
            'kx', 'ky', 'kx_AO', 'ky_AO', 'U', 'V', 'mask_corrected', 'mask_corrected_AO',
            'P_beta_DM', 'piston_filter', 'PR'
        ]
        
        for attr_name in attrs_to_clear:
            if hasattr(self, attr_name):
                delattr(self, attr_name)
        
        # Clean up config dictionary (contains nested tensors)
        if hasattr(self, 'config'):
            self._cleanup_dict_recursive(self.config)
            del self.config
        
        # Clear any remaining nn.Module parameters and buffers
        for name in list(self._parameters.keys()):
            del self._parameters[name]
        
        for name in list(self._buffers.keys()):
            del self._buffers[name]
        
        # Clear CUDA cache if using GPU
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    

    def __del__(self):
        """Destructor to ensure GPU memory is freed"""
        try:
            self.cleanup()
        except:
            # Silently fail if cleanup has issues during destruction
            pass
        
