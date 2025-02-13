#%%
import sys
sys.path.insert(0, '..')

import time
import numpy as np
import torch
from torch import fft, nn
from torch.nn.functional import interpolate
# import torch.autograd.profiler as profiler
import scipy.special as spc
from astropy.io import fits
import torchvision.transforms as transforms
from tools.utils import pdims, min_2d, to_little_endian, PupilVLT
from tools.air_refraction import AirRefractiveIndexCalculator


class TipTorch_new_half(torch.nn.Module): 
    def InitPupils(self):
        # with profiler.record_function("INIT PUPLAS"):
        
        try:
            # Manage pupil      
            pupil_path = self.config['telescope']['PathPupil']
            if 'PupilAngle' in self.config['telescope']:
                self.pupil_angle = self.config['telescope']['PupilAngle']
            else:
                self.pupil_angle = 0.0
            # TODO: pupil generator
            self.pupil =  self.make_tensor( PupilVLT( #TODO: accept arbitrary pupil
                samples = 320,
                vangle = [0,0],
                rotation_angle = self.pupil_angle
            ) )
            
            # Manage apodizer
            apodizer_path = self.config['telescope']['PathApodizer']
            
            if apodizer_path is not None and self.apodizer is None:
                self.apodizer = self.make_tensor( to_little_endian(fits.getdata(apodizer_path)) )
    
        except FileNotFoundError:
            pupil_path = pupil_path.replace('\\', '/')
            apodizer_path = apodizer_path.replace('\\', '/')
               
        
    def InitValues(self):
        # with profiler.record_function("INIT PARAMETERS"):
        # Reading parameters from the file
        self.N_src = self.config['NumberSources']#.int().item()
                
        self.wvl = self.config['sources_science']['Wavelength']
        # self.wvl = self.wvl if self.wvl.ndim == 2 else self.wvl.unsqueeze(0).T
        self.wvl = min_2d(self.wvl)
        self.N_wvl = self.wvl.shape[-1]
        self.wvl_atm = self.config['atmosphere']['Wavelength']

        # Science sources positions relative to the center of FOV
        self.src_zenith  = self.config['sources_science']['Zenith'].flatten() / self.rad2arc  # [N_src]
        self.src_azimuth = torch.deg2rad(self.config['sources_science']['Azimuth']).flatten() # [N_src]
        self.src_dirs_x  = torch.tan(self.src_zenith) * torch.cos(self.src_azimuth).flatten() # [N_src]
        self.src_dirs_y  = torch.tan(self.src_zenith) * torch.sin(self.src_azimuth).flatten() # [N_src]
        
        self.on_axis = (self.src_dirs_x.abs().sum() + self.src_dirs_y.abs().sum() == 0).item() # all angles are zeros
        
        self.psInMas = self.config['sensor_science']['PixelScale']
        self.D       = self.config['telescope']['TelescopeDiameter'] # [m]
        self.N_pix   = self.config['sensor_science']['FieldOfView'].int().item() # [pix]
        
        # Deformable mirror(s) parameters
        self.pitch = self.config['DM']['DmPitchs'] #[m]
        self.kc    = self.one_gpu / (self.two_gpu * self.pitch) # TODO: multiple DMs
        
        self.h_DM  = self.config['DM']['DmHeights'].flatten() # [m]
        self.N_DM  = self.h_DM.shape[0]
        
        self.DM_opt_angle   = self.config['DM']['OptimizationZenith' ].view(self.N_src, -1, self.N_DM) / self.rad2arc # [N_src x N_optdir]
        self.DM_opt_azimuth = torch.deg2rad(self.config['DM']['OptimizationAzimuth'].view(self.N_src, -1, self.N_DM)) # [N_src x N_optdir]

        self.DM_opt_dir_x  = torch.tan(self.DM_opt_angle) * torch.cos(self.DM_opt_azimuth) # [N_src x N_optdir]
        self.DM_opt_dir_y  = torch.tan(self.DM_opt_angle) * torch.sin(self.DM_opt_azimuth) # [N_src x N_optdir]
        self.DM_opt_weight = self.config['DM']['OptimizationWeight'].flatten() # [N_src, N_optdir]
        self.N_optdir = self.DM_opt_weight.shape[-1]
        
        # Telescope pointing
        self.zenith_angle = torch.deg2rad(pdims(self.config['telescope']['ZenithAngle'], 1)) # [N_src, 1]
        self.airmass      = 1.0 / torch.cos(self.zenith_angle) # [N_src, 1]

        # Guidestars parameters
        self.GS_wvl      = self.config['sources_HO']['Wavelength'] #[m]
        self.GS_height   = self.config['sources_HO']['Height'] * self.airmass #[m]
        self.GS_angles   = self.config['sources_HO']['Zenith'] / self.rad2arc # defined in [arcsec] from on-axis
        self.GS_azimuths = torch.deg2rad(self.config['sources_HO']['Azimuth']) # defined in [deg] from on-axis
        self.GS_dirs_x   = torch.tan(self.GS_angles) * torch.cos(self.GS_azimuths) # [N_src, N_GS]
        self.GS_dirs_y   = torch.tan(self.GS_angles) * torch.sin(self.GS_azimuths) # [N_src, N_GS]
        
        self.N_GS = self.GS_dirs_y.size(-1)
     
        # Atmospheric parameters
        self.wind_speed  = self.config['atmosphere']['WindSpeed']
        self.wind_dir    = self.config['atmosphere']['WindDirection']
        self.Cn2_weights = min_2d(self.config['atmosphere']['Cn2Weights'])
        self.Cn2_heights = min_2d(self.config['atmosphere']['Cn2Heights']) * self.airmass # [m]
        
        self.stretch  = self.one_gpu / (self.one_gpu - self.Cn2_heights/self.GS_height)
        self.h  = self.Cn2_heights * self.stretch
        self.nL = self.Cn2_heights.shape[-1]

        # WFS parameters
        self.WFS_d_sub = self.config['sensor_HO']['SizeLenslets']
        self.WFS_n_sub = self.config['sensor_HO']['NumberLenslets']

        self.WFS_det_clock_rate = self.config['sensor_HO']['ClockRate'].flatten() # TODO: what is exactly the clock rate?
        self.WFS_FOV = self.config['sensor_HO']['FieldOfView']
        self.WFS_RON = self.config['sensor_HO']['SigmaRON']
        self.WFS_wvl = self.make_tensor(self.GS_wvl)
        self.WFS_psInMas   = self.config['sensor_HO']['PixelScale']
        self.WFS_spot_FWHM = self.make_tensor(self.config['sensor_HO']['SpotFWHM'][0])
        self.WFS_excessive_factor = self.config['sensor_HO']['ExcessNoiseFactor']
        self.WFS_Nph = self.config['sensor_HO']['NumberPhotons']

        # HO WFSs parameters stored in a dictionary
        '''
        self.WFS_HO = {
            'd_sub': self.config['sensor_HO']['SizeLenslets'],
            'n_sub': self.config['sensor_HO']['NumberLenslets'],
            'det_clock_rate': self.config['sensor_HO']['ClockRate'].flatten(),  # TODO: what is exactly the clock rate?
            'FOV': self.config['sensor_HO']['FieldOfView'],
            'RON': self.config['sensor_HO']['SigmaRON'],
            'wvl': torch.tensor(self.GS_wvl, device=self.device) if not isinstance(self.GS_wvl, torch.Tensor) else self.GS_wvl,
            'psInMas': self.config['sensor_HO']['PixelScale'],
            'spot_FWHM': torch.tensor(self.config['sensor_HO']['SpotFWHM'][0], device=self.device) if not isinstance(self.config['sensor_HO']['SpotFWHM'][0], torch.Tensor) else self.config['sensor_HO']['SpotFWHM'][0],
            'excessive_factor': self.config['sensor_HO']['ExcessNoiseFactor'],
            'Nph': self.config['sensor_HO']['NumberPhotons']
        }
        '''

        self.HOloop_rate  = self.config['RTC']['SensorFrameRate_HO'] # [Hz]
        self.HOloop_delay = self.config['RTC']['LoopDelaySteps_HO'] # [ms] (?)
        self.HOloop_gain  = self.config['RTC']['LoopGain_HO']

        # Initialiaing the main optimizable parameters
        self.r0 = self.rad2arc * 0.976 * self.config['atmosphere']['Wavelength'] / self.config['atmosphere']['Seeing'] # [m]
        
        self.L0  = self.config['atmosphere']['L0'] # [m]
        self.F   = torch.ones (self.N_src, self.N_wvl, device=self.device)
        self.bg  = torch.zeros(self.N_src, self.N_wvl, device=self.device)
        self.dx  = torch.zeros(self.N_src, self.N_wvl, device=self.device)
        self.dy  = torch.zeros(self.N_src, self.N_wvl, device=self.device)
        self.Jx  = torch.ones (self.N_src, device=self.device)
        self.Jy  = torch.ones (self.N_src, device=self.device)
        self.Jxy = torch.ones (self.N_src, device=self.device)*0.1

        if self.PSD_include['Moffat']:
            self.amp   = torch.ones (self.N_src, device=self.device)*4.0  # Phase PSD Moffat amplitude [rad²]
            self.b     = torch.ones (self.N_src, device=self.device)*0.01 # Phase PSD background [rad² m²]
            self.alpha = torch.ones (self.N_src, device=self.device)*0.1  # Phase PSD Moffat alpha [1/m]
            self.beta  = torch.ones (self.N_src, device=self.device)*2    # Phase PSD Moffat beta power law
            self.ratio = torch.ones (self.N_src, device=self.device)      # Phase PSD Moffat ellipticity
            self.theta = torch.zeros(self.N_src, device=self.device)      # Phase PSD Moffat angle

        # if self.PSD_include['WFS noise'] or self.PSD_include['spatio-temporal'] or self.PSD_include ['aliasing']:
        self.dn  = torch.zeros(self.N_src, device=self.device)

        # An easier initialization for the MUSE NFM
        if self.AO_type == 'LTAO':
            self.N_DM = 1
            
        self.NoiseGain()
        
        
        self.IOR_src_wvl = self.n_air(self.wvl)
        self.IOR_wvl_atm = self.n_air(self.wvl_atm)
        self.IOR_GS_wvl = self.n_air(self.GS_wvl)


    def _FFTAutoCorr(self, x):
        return fft.fftshift( fft.ifft2(fft.fft2(x, dim=(-2,-1)).abs()**2, dim=(-2,-1)), dim=(-2,-1) ) / x.shape[-2] / x.shape[-1]

    
    def _gen_grid(self, N):
        factor = 0.5*(1-N%2)
        return torch.meshgrid(*[torch.linspace(-N//2+N%2+factor, N//2-factor, N, device=self.device)]*2, indexing = 'ij')
    

    def _stabilize(self, tensor, eps=1e-9):
        ''' Increases the numerical stability by replacing near-zero values with eps '''
        tensor[tensor.abs() < eps] = eps
        return tensor


    def to_double(self):
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if torch.is_tensor(attr):
                current_dtype = attr.dtype
                if   'float'   in str(current_dtype): new_dtype = torch.float64
                elif 'complex' in str(current_dtype): new_dtype = torch.complex128
                setattr(self, attr_name, attr.to(new_dtype))


    def to_float(self):
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if torch.is_tensor(attr):
                current_dtype = attr.dtype
                if   'float'   in str(current_dtype): new_dtype = torch.float32
                elif 'complex' in str(current_dtype): new_dtype = torch.complex64
                setattr(self, attr_name, attr.to(new_dtype))


    def PistonFilter(self, f):
        ''' Piston mode filter for PSD '''
        x = torch.pi * self.D * f
        R = torch.special.bessel_j1(x) / x
        piston_filter = self.one_gpu - 4*R.pow(2)
        piston_filter[..., self.nOtf_AO//2, self.nOtf_AO//2] *= 1-self.nOtf_AO % 2
        return self._stabilize(piston_filter)

        
    def _to_odd(self, x):
        odd = int(np.round(x))
        if odd % 2 == 0:
            odd += 1 if x > odd else -1
        return odd
        
    def InitGrids(self):
        # with profiler.record_function("INIT GRIDS"):
        # Initialize grids
        # for all generated PSDs within one batch sampling must be the same
       
        pixels_per_l_D = self.wvl * self.rad2mas / (self.psInMas * self.D)
        self.sampling_factor = self.two_gpu / pixels_per_l_D * self.oversampling 
        self.sampling = self.sampling_factor * pixels_per_l_D # must be = 2*oversampling
        self.sampling_min = self.sampling.min().item()

        self.nOtfs = self._to_odd_arr(self.N_pix * self.sampling_factor.cpu().numpy().flatten())
        self.nOtf  = self.nOtfs.max().item()
        self.dk    = self.one_gpu / self.D / self.sampling.min() # PSD spatial frequency step
                
        # Initialize PSD spatial frequencies
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

        self._stabilize(self.kx)
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
        rim_width = self.make_tensor(0.1)
        self.mask_corrected = torch.zeros_like(self.k2).int()
        self.mask_rim_in    = torch.zeros_like(self.k2).int() # masks to select the regions close to the corrected/uncorrected freqs split
        self.mask_rim_out   = torch.zeros_like(self.k2).int()
        self.mask_corrected [self.k2 <= self.kc**2] = self.one_gpu.int()
        self.mask_rim_in    [self.k2 <= (1-rim_width)*self.kc**2] = self.one_gpu.int()
        self.mask_rim_out   [self.k2 <= (1+rim_width)*self.kc**2] = self.one_gpu.int()
        self.mask_rim_in    = self.mask_corrected - self.mask_rim_in
        self.mask_rim_out   = self.mask_rim_out - self.mask_corrected
        self.mask           = self.one_gpu - self.mask_corrected
        
        mask_slice = self.mask_corrected[self.mask_corrected.shape[0]//2, :].tolist()
        first_one = mask_slice.index(1)
        last_one = len(mask_slice)-mask_slice[::-1].index(1)-1 # detect the borders of the correction area
        self.nOtf_AO = last_one-first_one+1

        corrected_ROI = (slice(first_one, last_one+1), slice(first_one, last_one+1))
        self.mask_corrected_AO = pdims(self.mask_corrected[corrected_ROI], -1)
        self.mask         = pdims( self.mask, -1 )
        self.mask_rim_out = pdims( self.mask_rim_out, -1 )
        self.mask_rim_in  = pdims( self.mask_rim_in, -1 )

        # TODO: delete rim masks?

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

        self.kx_AO = self.kx_AO[..., :self.nOtf_AO_x]
        self.ky_AO = self.ky_AO[..., :self.nOtf_AO_x]
        self.k_AO  = self.k_AO [..., :self.nOtf_AO_x]
        self.k2_AO = self.k2_AO[..., :self.nOtf_AO_x]
        self.mask_corrected_AO = self.mask_corrected_AO[..., :self.nOtf_AO_x]
        
        self.kx = self.kx[..., :self.nOtf_x]
        self.ky = self.ky[..., :self.nOtf_x]
        self.k  = self.k [..., :self.nOtf_x]
        self.k2 = self.k2[..., :self.nOtf_x]
        self.mask = self.mask[..., :self.nOtf_x]


        if self.PSD_include['Moffat']:
            self.kx2_AO = pdims( self.kx2[corrected_ROI], -1 )
            self.ky2_AO = pdims( self.ky2[corrected_ROI], -1 )
            self.kxy_AO = pdims( self.kxy[corrected_ROI], -1 )

            self.kxy = pdims( self.kxy, -1 )
            self.kx2 = pdims( self.kx2, -1 )
            self.ky2 = pdims( self.ky2, -1 )

        if self.PSD_include['aliasing']:
            # Comb samples count involved in alising PSD calculation
            n_times_y = int(np.ceil(self.nOtf/self.nOtf_AO/2))
            n_times_x = int(np.ceil(self.nOtf/self.nOtf_AO/2)) - 1
            # -1 since when using a halfed frequency grid, the last sample is truncated (from both sides)
            
            # Limit aliasing combs count to save on memory and computation time
            n_times_limit = 4
            n_times_x, n_times_y = min(max(2, n_times_y), n_times_limit), min(max(2, n_times_x), n_times_limit)
            ids = np.array( [[i, j] for i in range(-n_times_y+1, n_times_y) for j in range(-n_times_x+1, n_times_x) if i != 0 or j != 0] )
            
            # The 0-th dimension is used to store shifted spatial frequency
            # This is thing is 5D: (aliasing samples) x (atmospheric layers) x (kx) x (ky) TODO: NO, it's 4D
            m = self.make_tensor(ids[:,0])
            n = self.make_tensor(ids[:,1])
            self.N_combs = m.shape[0]
            
            self.km = self.kx_AO.repeat([self.N_combs,1,1,1]) - pdims(m/self.WFS_d_sub, 3)
            self.kn = self.ky_AO.repeat([self.N_combs,1,1,1]) - pdims(n/self.WFS_d_sub, 3)

        # Initialize OTF frequencies
        UV_range = torch.linspace(-1, 1, self.nOtf, device=self.device)
        self.U, self.V = torch.meshgrid(UV_range, UV_range)
        self.U, self.V = pdims(self.U, -2), pdims(self.V, -2)
        
        self.u_max = self.sampling * self.D / self.wvl / self.rad2mas
        
        self.center_aligner = torch.exp( 1j * torch.pi * (self.U + self.V) * (1 - self.N_pix%2))

        # Compute pupil OTFs
        phase_size = self.pupil.shape[-1]
        self.pupil_padder = torch.nn.ZeroPad2d( int(round(phase_size*self.sampling_min/2-phase_size/2)) )
        
        self.OTF_static_standart = self.StandartStaticOTF().real # Includes only apodizer and pupil
        self.OTF_static = self.OTF_static_standart.clone()
        
        # self.PSD_padder = torch.nn.ZeroPad2d((self.nOtf-self.nOtf_AO)//2)
        
        # since only half of PSD is used, the padding of AO-corrected PSD component is done only on the left
        # pad_left, pad_right, pad_top, pad_bottom
        self.PSD_padder = torch.nn.ZeroPad2d( (a:=((self.nOtf-self.nOtf_AO)//2), 0, a, a) )

        # if self.piston_filter is None:
        self.piston_filter = self.PistonFilter(self.k_AO)
        
        # To avoid reinitializing it without a need
        if self.PSD_include['aliasing']:
            self.PR = self.PistonFilter(torch.hypot(self.km, self.kn))

        # Diffraction-limited PSF
        self.PSF_DL = self.OTF2PSF(self.OTF_static_standart)
        # torch.cuda.empty_cache()
    
        if self.tomography:
            self.DMProjector()
            self.OptimalDMProjector()
              
    
    def Phase2OTF(self, phase):    
        # TODO: correct the description
        '''
        Compute OTF from a phase screen.
        All phase screens are sampled equally for all wavelengths.
        However, phase screens might be different for each wavelength.
        '''
        phase_padded = self.pupil_padder(phase)
            
        fl_even = self.nOtf % 2 == 0 and phase_padded.shape[-1] % 2 == 0 # TODO: what's the hell is this?
        phase_padded = phase_padded[:-1, :-1] if fl_even else phase_padded # to center-align if number of pixels is even

        OTF = self._FFTAutoCorr(phase_padded)
        OTF = OTF.view(1, 1, *OTF.shape) if OTF.ndim == 2 else OTF
        
        # PyTorch doesn't support interpolation of complex tensors yet
        OTF_ = self.interp(OTF.real, self.sampling_min) + self.interp(OTF.imag, self.sampling_min)*1j
        return OTF_ / OTF_.abs().amax(dim=(-2,-1), keepdim=True)
    
    
    def StandartStaticOTF(self):
        pupil_phase = self.pupil * self.apodizer if self.apodizer is not None else self.pupil
        return self.Phase2OTF(pupil_phase)
        

    def ComputeStaticOTF(self, phase_generator):
        if phase_generator is not None:
            self.OTF_static = self.Phase2OTF(phase_generator())

        return self.OTF_static


    def Update(self, reinit_grids=True, reinit_pupils=False):
        self.InitValues()

        if self.N_src != self.sampling.shape[0] or reinit_grids:
            self.InitGrids()
            
        if reinit_pupils:
            self.InitPupils()
            self.OTF_static_standart = self.StandartStaticOTF().real # Includes only apodizer and pupil
            self.PSF_DL = self.OTF2PSF(self.OTF_static_standart)
            

    def _initialize_PSDs_settings(self, TipTop_flag, PSFAO_flag):
        # The full list of the supported PSD components
        self.PSD_entries = [
            'fitting',
            'WFS noise',
            'spatio-temporal',
            'aliasing',
            'chromatism',
            'Moffat',
            'diff. refract'
        ]
        
        # The list of the PSD components that are included in the current simulation
        self.PSD_include = {key:False for key in self.PSD_entries}

        if TipTop_flag:
            self.PSD_include = {key:True for key in self.PSD_entries if key != 'Moffat'}
        
        self.PSD_include['Moffat']  = PSFAO_flag
        self.PSD_include['fitting'] = True #TODO: an open-loop case

        if not TipTop_flag and not PSFAO_flag:
            raise ValueError('At least one of the following must be True: TipTop or PSFAO')


    def __init__(self, AO_config, norm_regime='sum', device=torch.device('cpu'), TipTop=True, PSFAO=False, oversampling=1):
        super().__init__()
        
        self.device = device
        self.oversampling = oversampling

        self.AO_type = 'LTAO'
        self.tomography = True if self.AO_type in ['LTAO', 'MCAO', 'GLAO'] else False

        # Useful lambda functions
        self.r0_new = lambda r0, lmbd, lmbd0: r0*(lmbd/lmbd0).pow(6/5)
        self.make_tensor = lambda x: torch.as_tensor(x, device=self.device) if type(x) is not torch.Tensor else x
        self.interp = lambda x, sampling: interpolate(x, size=(self.nOtf,self.nOtf), mode='bilinear', align_corners=False) * sampling**2
        self._to_odd_arr = lambda arr: np.vectorize(self._to_odd)(arr)

        self._initialize_PSDs_settings(TipTop, PSFAO)

        # Initialize constants
        self.var_RON_const  = self.make_tensor(np.pi**2/3)
        self.var_Shot_const = self.make_tensor(np.pi**2)

        self.mas2arc  = self.make_tensor(1e-3)
        self.zero_gpu = self.make_tensor(0.)
        self.one_gpu  = self.make_tensor(1.)
        self.two_gpu  = self.make_tensor(2.)
        self.rad2mas  = self.make_tensor(3600 * 180 * 1000) / torch.pi
        self.rad2arc  = self.rad2mas / self.make_tensor(1000)
        self.cte = self.make_tensor( (24*spc.gamma(6/5)/5)**(5/6)*(spc.gamma(11/6)**2/(2*np.pi**(11/3))) )
        self.jitter_norm_fact = self.make_tensor( 2*np.sqrt(2*np.log(2)) ).pow(2)

        self.n_air = AirRefractiveIndexCalculator(device=self.device, dtype=torch.float32)

        if self.device.type != 'cpu':
            self.start = torch.cuda.Event(enable_timing=True)
            self.end   = torch.cuda.Event(enable_timing=True)

        self.norm_regime = norm_regime
        self.norm_scale  = self.make_tensor(1.0) # TODO: num obj x num wvl
        
        if    self.norm_regime == 'sum': self.normalizer = torch.sum
        elif  self.norm_regime == 'max': self.normalizer = torch.amax
        else: self.normalizer = lambda x, dim, keepdim: self.make_tensor(1.0)
                    
        self.piston_filter = None
        self.PR = None
        self.apodizer = None
        
        # Read data and initialize AO system
        self.config = AO_config
        self.InitValues()
        self.InitPupils()
        self.InitGrids()
        self.PSDs = {}


    def DMProjector(self):
        if self.on_axis:
            self.P_beta_DM = torch.ones([self.N_src, self.nOtf_AO_y, self.nOtf_AO_x, 1, self.N_DM], dtype=torch.complex64, device=self.device) * pdims(self.mask_corrected_AO, 2)
            return
            
        kx = pdims(self.kx_AO, 1) # [1 x nOtf_AO x nOtf_AO x 1]
        ky = pdims(self.ky_AO, 1) # [1 x nOtf_AO x nOtf_AO x 1]
        h_DM = self.h_DM.view(self.N_src, 1, 1, self.N_DM) # [N_src x 1 x 1 x N_DM]
        
        beta_x = self.src_dirs_x.view(self.N_src, 1, 1, 1)
        beta_y = self.src_dirs_y.view(self.N_src, 1, 1, 1)
    
        f = (beta_x*kx + beta_y*ky) * self.mask_corrected_AO.unsqueeze(-1)

        self.P_beta_DM = torch.exp( 2j*torch.pi*h_DM * f ).unsqueeze(-2) # [N_src x nOtf_AO x nOtf_AO x 1 x N_L]


    # TODO: input atmospheric params externaly to account for the modified atmosphere
    def OptimalDMProjector(self):  
        if self.on_axis:
            self.P_opt = torch.ones([self.N_src, self.nOtf_AO_y, self.nOtf_AO_x, 1, self.nL], dtype=torch.complex64, device=self.device)
            return
        
        h_dm = self.h_DM.view(1, 1, 1, self.N_DM)
        h    = self.h.view(1, 1, 1, 1, self.nL)
        opt_w = self.DM_opt_weight.view(self.N_src, 1, 1, self.N_optdir, 1, 1) 

        theta_x = self.DM_opt_dir_x.view(self.N_src, 1, 1, self.N_optdir)
        theta_y = self.DM_opt_dir_y.view(self.N_src, 1, 1, self.N_optdir)

        f = theta_x * pdims(self.kx_AO, 1) + theta_y * pdims(self.ky_AO, 1)
        P_L = torch.exp( 2j*torch.pi*h * pdims(f,1) ).unsqueeze(-2) # [N_src x nOtf_AO x nOtf_AO x N_optdir x 1 x N_L]
        
        if self.AO_type == 'LTAO':
            self.P_opt = (P_L * opt_w).sum(dim=3) # [N_src x nOtf_AO x nOtf_AO x 1 x N_L]
            return
        
        mask = self.mask_corrected_AO.view(1, self.nOtf_AO_y, self.nOtf_AO_x, 1)
        P_DM   = torch.exp( 2j*torch.pi*h_dm * pdims(f*mask,1) ).unsqueeze(-2) # [N_src x nOtf_AO x nOtf_AO x N_optdir x 1 x N_DM]
        P_DM_t = torch.conj( P_DM.permute(0, 1, 2, 3, 4, 5) )    # [N_src x nOtf_AO x nOtf_AO x N_optdir x N_DM x 1]

        mat1   = ((P_DM_t @ P_L)  * opt_w).sum(dim=3)  # [N_src x nOtf_AO x nOtf_AO x N_DM x N_L]
        to_inv = ((P_DM_t @ P_DM) * opt_w).sum(dim=3)  # [N_src x nOtf_AO x nOtf_AO x N_DM x N_DM]
        # TODO: lstsq solver
        mat2 = torch.linalg.pinv(to_inv, rcond=1e-2) # Last 2 dimensions are inverted
        
        self.P_opt = mat2 @ mat1 # [N_src x nOtf_AO x nOtf_AO x 1 x N_L]


    def TransferFunctions(self, freq, Ts, delay, loopGain):
        # z = torch.exp(2j*torch.pi*freq*Ts) # no minus according to Sanchit
        # hInt = loopGain / self._stabilize(1.0 - 1.0/z, 1e-12)
        # rtfInt = 1. / self._stabilize(1+hInt*z**(-delay), 1e-12) # Rejection transfer function
        # atfInt = self._stabilize(hInt * z**(-delay) * rtfInt)    # Aliasing transfer function
        # ntfInt = self._stabilize(atfInt / z, 1e-12)              # Noise transfer function
        # ntfInt = self._stabilize(hInt * z**(-delay-1)) # according to Sanchit, but it maybe wrong
        
        z = torch.exp(2j*torch.pi*freq*Ts) # no minus according to Sanchit
        hInt = loopGain / (1.0 - 1.0/z)
        rtfInt = 1. / (1 + hInt*z.pow(-delay)) # Rejection transfer function
        atfInt = hInt * z.pow(-delay) * rtfInt    # Aliasing transfer function
        ntfInt = atfInt / z              # Noise transfer function
        # ntfInt = hInt * z**(-delay-1) # according to Sanchit, but it maybe wrong
                    
        return hInt, rtfInt, atfInt, ntfInt


    def NoiseGain(self, nF=1000):
        Ts = 1.0 / self.HOloop_rate  # sampling time
        delay    = self.HOloop_delay # latency between the measurement and the correction
        loopGain = self.HOloop_gain
        
        f = torch.zeros([self.N_src, nF], device=self.device)
        
        for i in range(self.N_src):
            f[i,:] = torch.logspace(-3, torch.log10(0.5/Ts[i]), nF)

        _, _, _, ntfInt = self.TransferFunctions(f, Ts.unsqueeze(-1), delay.unsqueeze(-1), loopGain.unsqueeze(-1))
        self.noise_gain = pdims( torch.trapz(ntfInt.abs().pow(2), f, dim=1)*2*Ts, 2 )
    

    def Controller(self):
        #nTh = 1
        idim = lambda x: x.unsqueeze(1).unsqueeze(1)
        
        vy = idim(self.vy)
        vx = idim(self.vx)
        kx = self.kx_AO.unsqueeze(-1) # add atmo layers dimension: [N_src x nOtf_AO x nOtf_AO x nL]
        ky = self.ky_AO.unsqueeze(-1) # add atmo layers dimension: [N_src x nOtf_AO x nOtf_AO x nL]
        
        # TODO: expand dimensions from the beginning        
        Ts = 1.0 / self.HOloop_rate  # sampling time
        delay    = self.HOloop_delay # latency between the measurement and the correction
        loopGain = self.HOloop_gain
        
        #TODO: implement nTh
        thetaWind = self.zero_gpu #torch.linspace(0, 2*torch.pi-2*torch.pi/nTh, nTh)
        costh = torch.cos(thetaWind) # stays for the uncertainty in the wind diirection

        fi = -(vx*kx + vy*ky)*costh # [N_src x nOtf_AO x nOtf_AO x nL]

        _, _, atfInt, ntfInt = self.TransferFunctions(fi, pdims(Ts,3), pdims(delay,3), pdims(loopGain,3))
        # AO transfer function
        self.h1 = idim(self.Cn2_weights) * atfInt #/nTh
        self.h2 = idim(self.Cn2_weights) * atfInt.abs().pow(2) #/nTh
        self.hn = idim(self.Cn2_weights) * ntfInt.abs().pow(2) #/nTh

        self.h1 = self.h1.sum(dim=-1) # sum over the atmospheric layers
        self.h2 = self.h2.sum(dim=-1) 
        self.hn = self.hn.sum(dim=-1) 


    def ReconstructionFilter(self, WFS_noise_var):
        Av = torch.sinc(self.WFS_d_sub*self.kx_AO)*torch.sinc(self.WFS_d_sub*self.ky_AO) * torch.exp(1j*torch.pi*self.WFS_d_sub*(self.kx_AO+self.ky_AO))
        self.SxAv = ( 2j*torch.pi*self.kx_AO*self.WFS_d_sub*Av ).repeat([self.N_src,1,1] )
        self.SyAv = ( 2j*torch.pi*self.ky_AO*self.WFS_d_sub*Av ).repeat([self.N_src,1,1] )

        WFS_wvl = self.GS_wvl

        MV = 0
        # TODO: leave a singleton dimension for nGs in any case
        if WFS_noise_var.shape[1] > 1: # it means that there are several LGS, 0th dim is reserved for the targets
            varNoise = WFS_noise_var.mean(dim=1) # averaging the varience over the LGSs
        else:
            varNoise = WFS_noise_var
        
        W_n = pdims(varNoise / (2*self.kc)**2, 2)

        # TODO: isn't this one should be computed for the WFSing wvl?
        self.W_atm = self.VonKarmanSpectrum(self.r0.abs(), self.L0.abs(), self.k2_AO) * self.piston_filter

        gPSD = torch.abs(self.SxAv)**2 + torch.abs(self.SyAv)**2 + MV*W_n/self.W_atm / pdims(self.wvl_atm/WFS_wvl, 2)**2
        self.Rx = torch.conj(self.SxAv) / gPSD
        self.Ry = torch.conj(self.SyAv) / gPSD
        self.Rx[..., self.nOtf_AO//2, self.nOtf_AO//2] = 1e-9 # For numerical stability
        self.Ry[..., self.nOtf_AO//2, self.nOtf_AO//2] = 1e-9

    '''
    def ReconstructionFilter_new(self, WFS):
        WFS_d_sub = WFS['d_sub']
        WFS_wvl = WFS['wvl']
        MV = 0 # TODO: test the MV != 0  case
        
        Av = torch.sinc(WFS_d_sub*self.kx_AO)*torch.sinc(WFS_d_sub*self.ky_AO) * torch.exp(1j*np.pi*WFS_d_sub*(self.kx_AO+self.ky_AO))
        self.SxAv = ( 2j*np.pi*self.kx_AO*WFS_d_sub*Av ).repeat([self.N_src,1,1] )
        self.SyAv = ( 2j*np.pi*self.ky_AO*WFS_d_sub*Av ).repeat([self.N_src,1,1] )

        # TODO: leave a singleton dimension for nGs in any case
        if WFS['noise_var'].shape[1] > 1: # it means that there are several LGS, 0th dim is reserved for the targets
            WFS_noise_var = WFS['noise_var'].mean(dim=1) # averaging the varience over the LGSs
        else:
            WFS_noise_var = WFS['noise_var']
        
        W_n = pdims(WFS_noise_var / (2*self.kc)**2, 2)

        # TODO: isn't this one should be computed for the WFSing wvl?
        self.W_atm = self.VonKarmanSpectrum(self.r0.abs(), self.L0.abs(), self.k2_AO) * self.piston_filter

        gPSD = self.SxAv.abs()**2 + self.SyAv.abs()**2 + MV*W_n/self.W_atm / pdims(self.wvl_atm/WFS_wvl, 2)**2
        self.Rx = torch.conj(self.SxAv) / gPSD
        self.Ry = torch.conj(self.SyAv) / gPSD
        self.Rx[..., self.nOtf_AO//2, self.nOtf_AO//2] = 1e-9 # For numerical stability
        self.Ry[..., self.nOtf_AO//2, self.nOtf_AO//2] = 1e-9
    '''

    
    def SpatioTemporalPSD(self):
        if not self.tomography:
            A = torch.ones([self.W_atm.shape[0], self.nOtf_AO_y, self.nOtf_AO_x], device=self.device) #TODO: fix it. A should be initialized differently
            Ff = self.Rx*self.SxAv + self.Ry*self.SyAv
            psd_ST = (1 + Ff.abs()**2 * self.h2 - 2*torch.real(Ff*self.h1*A)) * self.W_atm * self.mask_corrected_AO
            
        else:
            kx = pdims(self.kx_AO, 1)
            ky = pdims(self.ky_AO, 1)
            h  = self.h.view(self.N_src, 1, 1, self.nL)
            
            beta_x = self.src_dirs_x.view(self.N_src, 1, 1, 1)
            beta_y = self.src_dirs_y.view(self.N_src, 1, 1, 1)
            
            delta_T = pdims( (1 + self.HOloop_delay) / self.HOloop_rate, 3)
            
            P_beta_L = torch.exp( 2j*torch.pi * (h*(beta_x*kx + beta_y*ky) - delta_T*self.freq_t) ).unsqueeze(-2)
            proj = P_beta_L - self.P_beta_DM @ self.W_alpha
            proj_t = torch.conj(torch.permute(proj, (0,1,2,4,3)))
            psd_ST = torch.squeeze(torch.squeeze(torch.abs((proj @ self.C_phi @ proj_t)))) * self.piston_filter * self.mask_corrected_AO
                
        return psd_ST
        

    def NoisePSD(self, WFS_noise_var):
        if not self.tomography:
            noisePSD = torch.abs(self.Rx**2 + self.Ry**2) / (2*self.kc)**2
            noisePSD = noisePSD * self.piston_filter * self.noise_gain * WFS_noise_var * self.mask_corrected_AO
            
        else:
            PW = self.P_beta_DM @ self.W
            PW_t = torch.conj(torch.permute(PW, (0,1,2,4,3)))
            noisePSD = (PW @ self.C_b @ PW_t).squeeze(-1).squeeze(-1)
            
            if WFS_noise_var.shape[1] > 1:
                varNoise = WFS_noise_var.mean(dim=1) # averaging the varience over all LGSs
            else:
                varNoise = WFS_noise_var
            
            noisePSD = noisePSD * self.noise_gain * pdims(varNoise,2) * self.mask_corrected_AO * self.piston_filter
            
        return noisePSD


    def AliasingPSD(self):
        T  = self.WFS_det_clock_rate / self.HOloop_rate
        td = pdims(T * self.HOloop_delay, [-1,3]) # [N_combs x N_src x nOtf_AO x n_Otf_AO x nL]
        T  = pdims(T, [-1,3])

        # Adding 0th dimension for shifted grid pieces
        Rx1 = (2j*torch.pi*self.WFS_d_sub * self.Rx).unsqueeze(0)
        Ry1 = (2j*torch.pi*self.WFS_d_sub * self.Ry).unsqueeze(0)

        W_mn = self.VonKarmanSpectrum(self.r0.abs().unsqueeze(0), self.L0.abs().unsqueeze(0), self.km**2 + self.kn**2) * self.PR
        
        Q = (Rx1*self.km + Ry1*self.kn) * torch.sinc(self.WFS_d_sub*self.km) * torch.sinc(self.WFS_d_sub*self.kn)
        tf = self.h1.unsqueeze(0).unsqueeze(-1) # [N_combs x N_src x nOtf_AO x n_Otf_AO x nL]

        # Add aliasing dimension and more
        vx = self.vx.view(1, self.N_src, 1, 1, self.nL)
        vy = self.vy.view(1, self.N_src, 1, 1, self.nL)
        Cn2_weights = self.Cn2_weights.view(1, self.N_src, 1, 1, self.nL)
        
        # Add atmo layers dimension
        km, kn = self.km.unsqueeze(-1), self.kn.unsqueeze(-1)

        avr = (Cn2_weights * tf * torch.sinc(km*vx*T) * torch.sinc(kn*vy*T) * \
            torch.exp( 2j*torch.pi*td*(km*vx + kn*vy) )).sum(dim=-1) # sum along atmolayers axis

        # Sum along aliasing samples axis      
        aliasing_PSD = torch.sum( W_mn*(Q*avr).abs()**2, dim=0 ) * self.mask_corrected_AO    
        return aliasing_PSD


    def VonKarmanSpectrum(self, r0, L0, freq2):
        return self.cte*pdims(r0,2)**(-5/3) * (freq2 + 1/pdims(L0,2)**2)**(-11/6)


    def VonKarmanPSD(self):
        return self.VonKarmanSpectrum(self.r0.abs(), self.L0.abs(), self.k2) * self.mask


    def ChromatismPSD(self):
        n2 = self.IOR_GS_wvl.view(self.N_src, 1, 1, 1)
        n1 = self.IOR_src_wvl.view(1, self.N_wvl, 1, 1)
        
        chromatic_PSD = ((n2-n1)/n2)**2 * self.W_atm.unsqueeze(1)
        return chromatic_PSD


    def DifferentialRefractionPSD(self):
        # TODO: account for the derotator angle
        h = self.h.view(self.N_src, 1, 1, 1, self.nL)
        w = self.Cn2_weights.view(self.N_src, 1, 1, 1, self.nL)
        k = self.k_AO.view(1, 1, self.nOtf_AO_y, self.nOtf_AO_x, 1)
        # [N_src x 1 x nOtf_AO_y x nOtf_AO_x]
        cos_ang   = torch.cos(torch.arctan2(self.ky_AO, self.kx_AO) - pdims(self.src_azimuth, 2)).unsqueeze(1)
        # [N_src x N_wvl]
        tan_theta = torch.tan((self.IOR_src_wvl - pdims(self.IOR_GS_wvl, 1)) * torch.tan(self.zenith_angle))
        
        return self.W_atm.unsqueeze(1) * ( 2*w*(1-torch.cos(2*torch.pi*h*k * pdims(tan_theta,3) * pdims(cos_ang,1))) ).sum(dim=-1)
    

    def JitterKernel(self, Jx, Jy, Jxy):
        # Assuming Jxy is in [deg], convert it to [rad]
        cos_theta = torch.cos( torch.deg2rad(Jxy) )
        sin_theta = torch.sin( torch.deg2rad(Jxy) )

        U_prime = self.U * cos_theta - self.V * sin_theta
        V_prime = self.U * sin_theta + self.V * cos_theta

        Djitter = pdims(self.u_max.pow(2) * self.jitter_norm_fact, 2) * ( (Jx*U_prime)**2 + (Jy*V_prime)**2 )
        return torch.exp(-0.5 * Djitter) #TODO: cover Nyquist sampled case
    

    def NoiseVariance(self):       
        WFS_wvl = self.WFS_wvl
        WFS_Nph = self.WFS_Nph.abs().view(self.N_src, self.N_GS)
        r0_WFS  = self.r0_new(self.r0.view(self.N_src), WFS_wvl, self.wvl_atm).abs()
        WFS_nPix = self.WFS_FOV / self.WFS_n_sub
        WFS_pixelScale = self.WFS_psInMas * self.mas2arc # [arcsec]
       
        # Read-out noise calculation
        nD = torch.maximum( self.rad2arc*WFS_wvl/self.WFS_d_sub/WFS_pixelScale, self.one_gpu ) #spot FWHM in pixels and without turbulence
        # Photon-noise calculation
        nT = torch.maximum( torch.hypot(self.WFS_spot_FWHM.max()*self.mas2arc, self.rad2arc*WFS_wvl/r0_WFS) / WFS_pixelScale, self.one_gpu )

        varRON  =  self.var_RON_const  * (self.WFS_RON / WFS_Nph).pow(2) * (WFS_nPix.pow(2)/nD).pow(2).unsqueeze(-1)
        varShot =  self.var_Shot_const / (2*WFS_Nph)  * (nT/nD).pow(2).unsqueeze(-1)
        
        # Noise variance calculation
        varNoise = self.WFS_excessive_factor * (varRON+varShot) * (self.wvl_atm/WFS_wvl).pow(2).unsqueeze(-1)

        return varNoise

    '''
    def NoiseVariance_new(self, WFS): #TODO: fix excessive CPU->GPU copying
        WFS_wvl = WFS['wvl']
        WFS_Nph = WFS['Nph'].abs().view(self.N_src, self.N_GS)
        r0_WFS = r0_new(self.r0.view(self.N_src), WFS_wvl, self.wvl_atm).abs()
        WFS_nPix = WFS['FOV'] / WFS['n_sub']
        WFS_pixelScale = WFS['psInMas'] / 1e3  # [arcsec]

        # Read-out noise calculation
        nD = torch.maximum(
            self.rad2arc * WFS_wvl / WFS['d_sub'] / WFS_pixelScale, 
            self.make_tensor(1.0)
        )  # spot FWHM in pixels without turbulence

        # Photon-noise calculation
        nT = torch.maximum(
            torch.hypot(self.WFS_spot_FWHM.max() / 1e3, self.rad2arc * WFS_wvl / r0_WFS) / WFS_pixelScale,
            self.make_tensor(1.0)
        )

        varRON  = np.pi**2 / 3 * (WFS['RON']**2 / WFS_Nph**2) * (WFS_nPix**2 / nD).unsqueeze(-1)**2
        varShot = np.pi**2 / (2*WFS_Nph) * (nT/nD).unsqueeze(-1)**2

        # Noise variance calculation
        varNoise = WFS['excessive_factor'] * (varRON + varShot) * (self.wvl_atm / WFS_wvl).unsqueeze(-1)**2

        return varNoise
    '''

    def TomographicReconstructors(self, WFS_noise_var, inv_method='lstsq'):
        h = self.h.view(self.N_src, 1, 1, 1, self.nL)
        kx = pdims(self.kx_AO, 2)
        ky = pdims(self.ky_AO, 2)
        GS_dirs_x = self.GS_dirs_x.view(self.N_src, 1, 1, self.N_GS, 1)
        GS_dirs_y = self.GS_dirs_y.view(self.N_src, 1, 1, self.N_GS, 1)
        
        diag_mask = lambda N: torch.eye(N, device=self.device).view(1,1,1,N,N).expand(self.N_src, self.nOtf_AO_y, self.nOtf_AO_x, -1, -1)
        
        M = 2j*torch.pi*self.k_AO * torch.sinc(self.WFS_d_sub*self.kx_AO) * torch.sinc(self.WFS_d_sub*self.ky_AO)
        M = pdims(M, 2).expand(self.N_src, self.nOtf_AO_y, self.nOtf_AO_x, self.N_GS, self.N_GS) * diag_mask(self.N_GS) # [N_src x nOtf_y x nOtf_x x nGS x nGS]
        
        P = torch.exp( 2j*torch.pi*h * (kx*GS_dirs_x + ky*GS_dirs_y) ) # N_src x nOtf_y x nOtf_x x nGS x nL

        MP   = torch.einsum('nwhik,nwhkj->nwhij', M, P)  # N_src x nOtf_y x nOtf_x x nL x nGS
        MP_t = torch.conj(MP.permute(0,1,2,4,3))

        fix_dims = lambda x_, N: torch.diag_embed(x_).view(self.N_src, 1, 1, N, N).expand(self.N_src, self.nOtf_AO_y, self.nOtf_AO_x, N, N)

        WFS_noise_variance = WFS_noise_var.to(dtype=MP.dtype)

        self.C_b = fix_dims( WFS_noise_variance, self.N_GS )

        kernel = self.VonKarmanSpectrum(self.r0.abs().to(dtype=MP.dtype), self.L0.abs(), self.k2_AO) * self.piston_filter
        # self.kernel = kernel
        self.C_phi = pdims(kernel, 2) * fix_dims(self.Cn2_weights, self.nL)

        # Tikhonov_reg = False

        # Inversion happens relative to the last two dimensions of the these tensors
        if inv_method == 'standart':
            self.W_tomo = (self.C_phi @ MP_t) @ torch.linalg.pinv(MP @ self.C_phi @ MP_t + self.C_b, rcond=1e-2)

        elif inv_method == 'lstsq':
            # if Tikhonov_reg:
            #     lambda_reg = 1e-6  # Ridge Regression regularization parameter
            #     A = (MP @ self.C_phi @ MP_t + self.C_b).transpose(-2, -1)
            #     I_mat = torch.eye(A.size(-1), device=A.device, dtype=A.dtype)
            #     A_reg = A.transpose(-2, -1) @ A + lambda_reg * I_mat # Regularized A matrix
            #     B = (self.C_phi @ MP_t).transpose(-2, -1)
            #     W_tomo = torch.linalg.lstsq(A_reg, A.transpose(-2, -1) @ B).solution.transpose(-2, -1)
                
            # else:
            A = (MP @ self.C_phi @ MP_t + self.C_b).transpose(-2, -1)
            B = (self.C_phi @ MP_t).transpose(-2, -1)
            # Solve the least squares problem for W^T, since we deal with W*A = B
            self.W_tomo = torch.linalg.lstsq(A, B, rcond=1e-2).solution.transpose(-2, -1)           
        else:
            raise ValueError('Unknown inversion method specified.') 
         

        self.W = self.P_opt @ self.W_tomo
        
        samp_time = 1.0 / self.HOloop_rate

        www = 2j * torch.pi * pdims(self.k_AO, 1) * torch.sinc(pdims(samp_time * self.WFS_det_clock_rate, 3) * self.freq_t)

        self.MP_alpha_L = www.unsqueeze(-2) * P * (torch.sinc(self.WFS_d_sub*kx) * torch.sinc(self.WFS_d_sub*ky))

        self.W_alpha = self.W @ self.MP_alpha_L


    def MoffatPSD(self, amp, b, alpha, beta, ratio, theta):
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

        V = (1.0+uu)**(-beta) # Moffat shape

        removeInside = 0.0
        E = (beta-1) / (torch.pi*ax*ay)
        Fout = (1 +      (self.kc**2)/(ax*ay))**(1-beta)
        Fin  = (1 + (removeInside**2)/(ax*ay))**(1-beta)
        F = 1 / (Fin-Fout)

        MoffatPSD = (amp * V*E*F + b) * self.mask_corrected_AO * self.piston_filter
              
        return MoffatPSD

    
    def DLPSF(self):
        return self.PSF_DL / self.norm_scale


    def ComputePSD(self):
        if all(not value for value in self.PSD_include.values()):
            self.PSD = torch.zeros([self.N_src, self.N_wvl, self.nOtf, self.nOtf], device=self.device)
            return self.PSD
        
        if self.PSD_include['Moffat']:
            amp   = pdims(self.amp,   2)
            b     = pdims(self.b,     2)
            alpha = pdims(self.alpha, 2)
            beta  = pdims(self.beta,  2)
            ratio = pdims(self.ratio, 2)
            theta = pdims(self.theta, 2)

        if self.PSD_include['WFS noise'] or self.PSD_include['spatio-temporal'] or self.PSD_include ['aliasing']:

            WFS_noise_var = (self.dn.unsqueeze(-1) + self.NoiseVariance()).abs() # [rad^2] at atmo wvl
                        
            self.vx = self.wind_speed * torch.cos( torch.deg2rad(self.wind_dir) )
            self.vy = self.wind_speed * torch.sin( torch.deg2rad(self.wind_dir) )

            self.freq_t = self.vx.view(self.N_src, 1, 1, self.nL) * pdims(self.kx_AO, 1) + \
                          self.vy.view(self.N_src, 1, 1, self.nL) * pdims(self.ky_AO, 1) # [N_src x nOtf_AO x nOtf_AO x nL]

            self.Controller()
            self.ReconstructionFilter(WFS_noise_var)
                        
            if self.AO_type == 'LTAO':
                self.TomographicReconstructors(WFS_noise_var, inv_method='lstsq')
        

        # All PSD components are computed in [rad^2] at the atmospheric wavelength
        dk = 2*self.kc / self.nOtf_AO
        PSD_norm = lambda wvl: (dk*wvl*1e9/2/torch.pi)**2

        # Most contributors are computed for the atmospheric wavelength, wavelengths-dependant inputs as well are assumed for the atmospheric wavelength
        # But WFS operates on another wavelength, so this PSD components is normalized for WFSing wvl 
        # Put all contributiors together and sum up the resulting PSD

        self.PSDs = {entry: self.zero_gpu for entry in self.PSD_entries}

        if self.PSD_include['fitting']:
            self.PSDs['fitting'] = self.VonKarmanPSD().unsqueeze(1)
    
        if self.PSD_include['WFS noise']:
            self.PSDs['WFS noise'] = self.NoisePSD(WFS_noise_var).unsqueeze(1)
        
        if self.PSD_include['spatio-temporal']:
            self.PSDs['spatio-temporal'] = self.SpatioTemporalPSD().unsqueeze(1)
        
        if self.PSD_include['aliasing']:
            self.PSDs['aliasing'] = self.AliasingPSD().unsqueeze(1)
        
        if self.PSD_include['chromatism']:
            self.PSDs['chromatism'] = self.ChromatismPSD() # no need to add dimension since it's polychromatic already

        if self.PSD_include['Moffat']:
            self.PSDs['Moffat'] = self.MoffatPSD(amp, b, alpha, beta, ratio, theta).unsqueeze(1)

        if self.PSD_include['diff. refract']:
            self.PSDs['diff. refract'] = self.DifferentialRefractionPSD()

        #TODO: anisoplanatism for non-tomography!
        #TODO: SLAO support!
        
        PSD = self.PSDs['fitting'] + self.PSD_padder(
              self.PSDs['WFS noise'] + \
              self.PSDs['spatio-temporal'] + \
              self.PSDs['aliasing'] + \
              self.PSDs['chromatism'] + \
              self.PSDs['Moffat'] + \
              self.PSDs['diff. refract']) # [N_scr x N_wvl x nOtf_AO x nOtf_AO]
                
        self.PSD = PSD * PSD_norm(self.wvl_atm) # [nm^2]
        return self._half_PSD_to_full(self.PSD)
    
    
    def _half_PSD_to_full(self, half_PSD):
        n_cols_half = half_PSD.size(-1) 
        return torch.cat([
            half_PSD,
            torch.flip(half_PSD[..., :, :n_cols_half-n_cols_half % 2], dims=(-2,-1))
        ], dim=-1)
    
    
    def _rfft2_to_full(self, matrix_rfft2):
        width = matrix_rfft2.size(-1)
        return torch.cat([
            torch.flip(matrix_rfft2[..., :, width % 2:].conj(), dims=[-2,-1]),
            matrix_rfft2
        ], dim=-1)
    

    def OTF2PSF(self, OTF): 
        #TODO: check this function for OTF.shape[-1] < self.N_pix case
        PSF_big = fft.fftshift(fft.ifft2(fft.ifftshift(OTF*self.center_aligner, dim=(-2,-1))), dim=(-2,-1)).abs()
        
        PSF = []
        for i in range(self.wvl.shape[-1]): # TODO: check if "if" statement is needed here
            n = 0 if PSF_big.shape[1] == 1 else i # In the case OTF is only computed for minimal sampling
            if OTF.shape[-1] > self.N_pix:

                PSF_interp = interpolate(
                    PSF_big[:,n,...].unsqueeze(1),
                    size = (self.N_pix, self.N_pix),
                    mode = 'bilinear'
                ) * (PSF_big.shape[-1] / self.N_pix)**2 # preserve energy
                
                PSF.append(PSF_interp)
            
            else: #TODO: check flux attenuation when cropping
                transform = transforms.CenterCrop((self.nOtfs[i], self.nOtfs[i]))
                PSF_cropped = transform(PSF_big[:,n,...]).unsqueeze(1)
                PSF.append( PSF_cropped )

        return torch.hstack(PSF)


    def PSD2PSF(self, PSD, OTF_static):
        F   = pdims(self.F,   2) # TODO: move it from here to down
        bg  = pdims(self.bg,  2) # TODO: move it from here to down
        dx  = pdims(self.dx,  2)
        dy  = pdims(self.dy,  2)
        Jx  = pdims(self.Jx,  3) if self.Jx.ndim  == 1 else pdims(self.Jx,  2)
        Jy  = pdims(self.Jy,  3) if self.Jy.ndim  == 1 else pdims(self.Jy,  2)
        Jxy = pdims(self.Jxy, 3) if self.Jxy.ndim == 1 else pdims(self.Jxy, 2)

        # Removing the DC component
        PSD[..., self.nOtf//2, self.nOtf//2] = 0.0
        # Computing OTF from PSD, real is to remove the imaginary part that appears due to numerical errors
        # cov = fft.fftshift(fft.fft2(fft.ifftshift(PSD, dim=(-2,-1))), dim=(-2,-1)) # FFT axes are -2,-1 #TODO: real FFT?
        cov = self._rfft2_to_full(torch.fft.fftshift(torch.fft.rfft2(PSD.abs(), dim=(-2,-1)), dim=-2).abs())
     
        self.cov = cov
  
        # Computing the Structure Function from the covariance
        SF = 2*(cov.abs().amax(dim=(-2,-1), keepdim=True) - cov).real
        # self.SF = SF
        # Phasor to shift the PSF with the subpixel accuracy
        # TODO: replace self.U by unsqueezing it earlier
        fftPhasor = torch.exp( -torch.pi*1j * pdims(self.sampling_factor,2) * (self.U*dx + self.V*dy) )
        OTF_turb  = torch.exp( -0.5 * SF * pdims(2*torch.pi*1e-9/self.wvl,2)**2 )
        self.OTF_turb = OTF_turb
        
        # Compute the residual tip/tilt kernel
        OTF_jitter = self.JitterKernel(Jx.abs(), Jy.abs(), Jxy.abs())
        # Resulting combined OTF
        self.OTF = OTF_turb * OTF_static * fftPhasor * OTF_jitter
        self.fftPhasor = fftPhasor
        # self.OTF_jitter = OTF_jitter
        
        self.OTF_norm = self.OTF.abs()[..., self.nOtf//2, self.nOtf//2].unsqueeze(-1).unsqueeze(-1) #TODO: replace by pdims(..., 2)
        self.OTF = self.OTF / self.OTF_norm
        
        PSF_out = self.OTF2PSF(self.OTF)
        
        self.norm_scale = self.normalizer(PSF_out, dim=(-2,-1), keepdim=True)
        
        # torch.cuda.empty_cache()
        
        return (PSF_out / self.norm_scale) * F + bg


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


    def to(self, device):
        if isinstance(device, str):
            device = torch.device(device)
        if self.device == device:
            return self
        self.device = device
        
        for name, attr in self.__dict__.items():
            new_attr = self._to_device_recursive(attr, device)
            if new_attr is not attr:
                setattr(self, name, new_attr)
        return self


    def forward(self, x=None, PSD=None, phase_generator=None):
        if x is not None:
            for name, value in x.items():
                if hasattr(self, name):
                    if isinstance(getattr(self, name), nn.Parameter):
                        setattr(self, name, nn.Parameter(value))
                    else:
                        setattr(self, name, value)
            
        return self.PSD2PSF(\
            self.ComputePSD() if PSD is None else PSD, \
            self.ComputeStaticOTF(phase_generator)
        )

    # TODO: timer as a separate class
    def StartTimer(self):
        if self.device.type == 'cpu':
            self.start = time.time()
        else:
            self.start.record()


    def EndTimer(self):
        if self.device.type == 'cpu':
            self.end = time.time()
            return (self.end-self.start)*1000.0 # in [ms]
        else:
            self.end.record()
            torch.cuda.synchronize()
            return self.start.elapsed_time(self.end)

# %%
