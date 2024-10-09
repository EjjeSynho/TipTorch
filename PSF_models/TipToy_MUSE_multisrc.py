#%%
import sys
sys.path.insert(0, '..')

import numpy as np
import torch
from torch import fft, nn
from torch.nn.functional import interpolate
import scipy.special as spc
from astropy.io import fits
import torchvision.transforms as transforms

import time

import matplotlib.pyplot as plt
from tools.utils import rad2mas, rad2arc, deg2rad, r0_new, r0, pdims, min_2d, to_little_endian, PupilVLT

# import torch.autograd.profiler as profiler

class TipTorch(torch.nn.Module): 
    def InitPupils(self):
        # with profiler.record_function("INIT PUPLAS"):
        
        try:
            # Manage pupil      
            pupil_path = self.config['telescope']['PathPupil']
            if 'PupilAngle' in self.config['telescope']:
                self.pupil_angle = self.config['telescope']['PupilAngle']
            else:
                self.pupil_angle = 0.0
            # TODOL pupil generator
            self.pupil =  self.make_tensor( PupilVLT(
                samples = 320,
                vangle = [0,0],
                rotation_angle = self.pupil_angle,
                secondary_diameter=1.12,
                spider_width=0.039*2
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
        num_src = self.config['NumberSources']
        self.N_src = int(num_src)
        self.AO_type = 'LTAO'
        
        self.wvl = self.config['sources_science']['Wavelength']
        # self.wvl = self.wvl if self.wvl.ndim == 2 else self.wvl.unsqueeze(0).T
        self.wvl = min_2d(self.wvl)
        self.N_wvl = self.wvl.shape[-1]

        # Setting internal parameters
        self.psInMas = self.config['sensor_science']['PixelScale'] #[mas]
        if isinstance(self.psInMas, torch.Tensor):
            if self.psInMas.dim() > 0:
                if not torch.all(self.psInMas == self.psInMas[0]).item(): raise ValueError('All pixel scales must be the same for all samples')
                self.psInMas = self.psInMas[0]

        self.D     = self.config['telescope']['TelescopeDiameter']
        self.nPix  = int(self.config['sensor_science']['FieldOfView'])
        self.pitch = self.config['DM']['DmPitchs'] #[m]
        #self.h_DM  = self.AO_config['DM']['DmHeights'] # ????? what is h_DM?
        #self.nDM   = 1
        self.kc    = 1/(2*self.pitch)
        # #TODO: telescope zenith != sample zenith?
        self.zenith_angle  = self.config['telescope']['ZenithAngle']
        self.airmass       = (1.0 / torch.cos(self.zenith_angle * deg2rad)).unsqueeze(-1) # 1st dimension is for atmo layers, 0th for objects

        self.GS_wvl     = self.config['sources_HO']['Wavelength'] #[m]
        self.GS_height  = self.config['sources_HO']['Height'] * self.airmass #[m]
        self.GS_angle   = self.config['sources_HO']['Zenith']  / rad2arc
        self.GS_azimuth = self.config['sources_HO']['Azimuth'] * deg2rad
        self.GS_dirs_x  = torch.tan(self.GS_angle) * torch.cos(self.GS_azimuth)
        self.GS_dirs_y  = torch.tan(self.GS_angle) * torch.sin(self.GS_azimuth)
        self.nGS = self.GS_dirs_y.size(-1)
     
        self.wind_speed  = self.config['atmosphere']['WindSpeed']
        self.wind_dir    = self.config['atmosphere']['WindDirection']
        self.Cn2_weights = min_2d(self.config['atmosphere']['Cn2Weights'])
        self.Cn2_heights = min_2d(self.config['atmosphere']['Cn2Heights']) * self.airmass # [m]
        
        self.stretch  = 1.0 / (1.0 - self.Cn2_heights/self.GS_height)
        self.h  = self.Cn2_heights * self.stretch
        self.nL = self.Cn2_heights.shape[-1]

        self.WFS_d_sub = self.config['sensor_HO']['SizeLenslets']
        self.WFS_n_sub = self.config['sensor_HO']['NumberLenslets']

        self.WFS_det_clock_rate = self.config['sensor_HO']['ClockRate'].flatten() # [(?)]
        self.WFS_FOV = self.config['sensor_HO']['FieldOfView']
        self.WFS_RON = self.config['sensor_HO']['SigmaRON']
        self.WFS_wvl = self.make_tensor(self.GS_wvl) #TODO: clarify this
        self.WFS_psInMas   = self.config['sensor_HO']['PixelScale']
        self.WFS_spot_FWHM = self.make_tensor(self.config['sensor_HO']['SpotFWHM'][0])
        self.WFS_excessive_factor = self.config['sensor_HO']['ExcessNoiseFactor']
        self.WFS_Nph = self.config['sensor_HO']['NumberPhotons']

        self.HOloop_rate  = self.config['RTC']['SensorFrameRate_HO'] # [Hz] (?)
        self.HOloop_delay = self.config['RTC']['LoopDelaySteps_HO'] # [ms] (?)
        self.HOloop_gain  = self.config['RTC']['LoopGain_HO']

        # Initialiaing the main optimizable parameters
        self.r0  = r0(self.config['atmosphere']['Seeing'], self.config['atmosphere']['Wavelength']) # [m]
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

        if self.AO_type == 'LTAO':
        # An easier initialization for the MUSE NFM
            self.nDM = 1


    def _fftAutoCorr(self, x):
        # return fft.fftshift( fft.ifft2( torch.abs(fft.fft2(x, norm='forward')**2) ) )
        # return fft.fftshift( fft.ifft2(fft.fft2(x).abs()**2) ) / x.shape[-2] / x.shape[-1]
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

    # Piston mode filter, attenuates the center of the PSD
    def PistonFilter(self, f):
        x = (np.pi*self.D*f).cpu().numpy()
        R = spc.j1(x)/x
        piston_filter = self.make_tensor(1.0-4*R**2)
        piston_filter[..., self.nOtf_AO//2, self.nOtf_AO//2] *= 1-self.nOtf_AO % 2
        return self._stabilize(piston_filter)


    def InitGrids(self):
        # with profiler.record_function("INIT GRIDDIES"):
        # Initialize grids
        # for all generated PSDs within one batch sampling must be the same
        '''
        to_odd = lambda f: int( np.ceil(f)//2 * 2 - 1) if not f % 2 else int(f)
        to_odd_arr = lambda arr: np.array([to_odd(f) for f in arr])

        # to_odd = lambda f: int( np.ceil(f)//2 * 2 - 1) if not f % 2 else np.ceil(f).astype(int)
        # to_even = lambda f: int( np.ceil(f)//2 * 2 )

        # Manage sampling
        pixels_per_l_D = self.wvl*rad2mas / (self.psInMas*self.D)
        self.sampling_factor = torch.ceil(2.0/pixels_per_l_D) * self.oversampling # to avoid aliasing or to provide oversampling
        self.sampling = self.sampling_factor * pixels_per_l_D
        self.nOtf = to_odd(self.nPix * self.sampling_factor.max().item()) + 2
        '''
        
        def to_odd(x):
            odd = int(np.round(x))
            if odd % 2 == 0:
                odd += 1 if x > odd else -1
            return odd

        to_odd_arr = lambda arr: np.vectorize(to_odd)(arr)
        
        pixels_per_l_D = self.wvl*rad2mas / (self.psInMas*self.D)
        self.sampling_factor = 2.0/pixels_per_l_D * self.oversampling 
        self.sampling = self.sampling_factor * pixels_per_l_D # must be = 2*oversamplig

        self.nOtfs = to_odd_arr(self.nPix * self.sampling_factor.cpu().numpy().flatten())
        self.nOtf  = self.nOtfs.max().item()

        # sampling_factor_ = self.nOtfs / self.nPix
        # sampling_ = sampling_factor_ * pixels_per_l_D.cpu().numpy()    
        
        self.dk = 1/self.D/self.sampling.min() # PSD spatial frequency step
        self.cte = (24*spc.gamma(6/5)/5)**(5/6)*(spc.gamma(11/6)**2/(2*np.pi**(11/3)))
        
        # Initialize PSD spatial frequencies
        if self.pupil_angle != 0.0:  
            kx, ky = self._gen_grid(self.nOtf)
            rot_ang = self.make_tensor(self.pupil_angle*deg2rad)
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
            self.kx2 = self.kx**2
            self.ky2 = self.ky**2
            
            self._stabilize(self.kxy)
            self._stabilize(self.kx2)
            self._stabilize(self.ky2)

        # Compute the frequency mask for the AO corrected and uncorrected frequency regions
        rim_width = 0.1
        self.mask_corrected = torch.zeros_like(self.k2).int()
        self.mask_rim_in    = torch.zeros_like(self.k2).int() # masks to select the regions close to the corrected/uncorrected freqs split
        self.mask_rim_out   = torch.zeros_like(self.k2).int()
        self.mask_corrected [self.k2 <= self.kc**2] = 1
        self.mask_rim_in    [self.k2 <= (1-rim_width)*self.kc**2] = 1
        self.mask_rim_out   [self.k2 <= (1+rim_width)*self.kc**2] = 1
        self.mask_rim_in    = self.mask_corrected - self.mask_rim_in
        self.mask_rim_out   = self.mask_rim_out - self.mask_corrected
        self.mask           = 1 - self.mask_corrected
        
        mask_slice = self.mask_corrected[self.mask_corrected.shape[0]//2, :].tolist()
        first_one = mask_slice.index(1)
        last_one = len(mask_slice)-mask_slice[::-1].index(1)-1 # detect the borders of the correction area
        self.nOtf_AO = last_one-first_one+1

        corrected_ROI = (slice(first_one, last_one+1), slice(first_one, last_one+1))
        self.mask_corrected_AO = pdims(self.mask_corrected[corrected_ROI], -1)
        self.mask         = pdims( self.mask, -1 )
        self.mask_rim_out = pdims( self.mask_rim_out, -1 )
        self.mask_rim_in  = pdims( self.mask_rim_in, -1 )

        # Computing frequency grids for the AO corrected and uncorrected regions
        self.kx_AO = pdims( self.kx[corrected_ROI], -1 )
        self.ky_AO = pdims( self.ky[corrected_ROI], -1 )
        self.k_AO  = pdims( self.k [corrected_ROI], -1 )
        self.k2_AO = pdims( self.k2[corrected_ROI], -1 )

        self.kx = pdims( self.kx, -1 )
        self.ky = pdims( self.ky, -1 )
        self.k  = pdims( self.k,  -1 )
        self.k2 = pdims( self.k2, -1 )

        if self.PSD_include['Moffat']:
            self.kx2_AO = pdims( self.kx2[corrected_ROI], -1 )
            self.ky2_AO = pdims( self.ky2[corrected_ROI], -1 )
            self.kxy_AO = pdims( self.kxy[corrected_ROI], -1 )

            self.kxy = pdims( self.kxy, -1 )
            self.kx2 = pdims( self.kx2, -1 )
            self.ky2 = pdims( self.ky2, -1 )

        if self.PSD_include['aliasing']:
            # Comb samples involved in alising PSD calculation
            n_times = min(4, max(2, int(np.ceil(self.nOtf/self.nOtf_AO/2)) ) )
            # ids = np.array( [[i, j] for i in range(-n_times, n_times+1) for j in range(-n_times, n_times+1) if i != 0 or j != 0] )
            ids = np.array( [[i, j] for i in range(-n_times+1, n_times) for j in range(-n_times+1, n_times) if i != 0 or j != 0] )
            
            # For NGS-like alising 0th dimension is used to store shifted spatial frequency
            # This is thing is 5D: (aliasing samples) x (N src) x (Atmospheric layers) x (kx) x (ky) TODO: NO, it's 4D
            m = self.make_tensor(ids[:,0])
            n = self.make_tensor(ids[:,1])
            self.N_combs = m.shape[0]
            
            self.km = self.kx_AO.repeat([self.N_combs,1,1,1]) - pdims(m/self.WFS_d_sub, 3)
            self.kn = self.ky_AO.repeat([self.N_combs,1,1,1]) - pdims(n/self.WFS_d_sub, 3)

        # Initialize OTF frequencies
        self.U,  self.V   = self._gen_grid(self.nOtf)
        self.U,  self.V   = pdims(self.U/self.nOtf, -1)*2, pdims(self.V/self.nOtf, -1)*2
        self.U2, self.V2  = self.U**2, self.V**2
        self.UV, self.UV2 = self.U*self.V, self.U**2 + self.V**2
        
        self._stabilize(self.U2 )
        self._stabilize(self.V2 )
        self._stabilize(self.UV )
        self._stabilize(self.UV2)
         
        self.center_aligner = torch.exp( 1j*np.pi*(self.U+self.V) * (1-self.nPix%2))

        # Compute pupil OTFs
        self.OTF_static_standart = self.StandartStaticOTF().real # Includes only apodizer and pupil
        self.OTF_static = self.OTF_static_standart.clone()
        
        # PSD kingdome
        self.PSD_padder = torch.nn.ZeroPad2d((self.nOtf-self.nOtf_AO)//2)

        # if self.piston_filter is None:
        self.piston_filter = self.PistonFilter(self.k_AO)
        
        # To avoid reinitializing it without a need
        if self.PSD_include['aliasing']:
            self.PR = self.PistonFilter(torch.hypot(self.km, self.kn))

        # Diffraction-limited PSF
        self.PSF_DL = self.OTF2PSF(self.OTF_static_standart)
        torch.cuda.empty_cache()
    
        if self.AO_type == 'LTAO':
            self.P_beta_DM = torch.ones([self.N_src, self.nOtf_AO, self.nOtf_AO, 1, self.nDM], dtype=torch.complex64, device=self.device) * pdims(self.mask_corrected_AO, 2)
            self.P_opt     = torch.ones([self.N_src, self.nOtf_AO, self.nOtf_AO, 1, self.nL],  dtype=torch.complex64, device=self.device) * pdims(self.mask_corrected_AO, 2)

    
    def Phase2OTF(self, phase, sampling):    
        '''
        Compute OTF from a phase screen.
        All phase screens are sampled equally for all wavelengths.
        However, phase screens might be different for each wavelength.
        '''
        pupil_size   = phase.shape[-1]
        pupil_padder = torch.nn.ZeroPad2d( int(pupil_size*sampling/2-pupil_size/2) )
        phase_padded = pupil_padder(phase)
            
        fl_even = self.nOtf % 2 == 0 and phase_padded.shape[0] % 2 == 0
        phase_padded = phase_padded[:-1, :-1] if fl_even else phase_padded # to center-align if number of pixels is even

        OTF = self._fftAutoCorr(phase_padded)
        OTF = OTF.view(1, 1, *OTF.shape) if OTF.ndim == 2 else OTF

        interp = lambda x: \
            interpolate(x, size=(self.nOtf,self.nOtf), mode='bilinear', align_corners=False) * sampling**2
            
        # PyTorch doesn't support interpolation of complex tensors yet
        OTF_ = interp(OTF.real) + interp(OTF.imag)*1j
        return OTF_ / OTF_.abs().amax(dim=(-2,-1), keepdim=True)
    
    
    def StandartStaticOTF(self):
        '''
        # Compute static OTF for each wavelength
        wvl_cpu = self.wvl.cpu().numpy()
        ids = np.unravel_index(np.unique(wvl_cpu, return_index=True)[1], self.wvl.shape)
        
        if self.apodizer is not None:
            pupil = self.pupil * self.apodizer #* torch.exp(1j*2*np.pi/wvl)
        else:
            pupil = self.pupil
    
        # Iterates over wavelengths only if some wavelengths are repeated, this saves computation time
        OTF_static_dict = {}
        for i, wvl in enumerate(wvl_cpu[ids].tolist()):
            OTF_static_dict.setdefault(wvl, self.Phase2OTF(pupil, self.sampling[ids][i]))
        
        OTF_static_ = torch.stack(
            [torch.stack([OTF_static_dict[wvl_cpu[obj_id, wvl_id].item()] 
                for wvl_id in range(self.wvl.shape[1])])
                for obj_id in range(self.N_src)]
            )
        del OTF_static_dict
        
        return OTF_static_
        '''

        if self.apodizer is not None:
            pupil_phase = self.pupil * self.apodizer
        else:
            pupil_phase = self.pupil

        return self.Phase2OTF(pupil_phase, self.sampling.min().item())
        

    def ComputeStaticOTF(self, phase_generator):
        if phase_generator is not None:
            # raise NotImplementedError('Phase generator is not implemented yet')
            '''            
            OTF_static = torch.zeros([self.N_src, self.wvl.shape[1], self.nOtf, self.nOtf], device=self.device, dtype=torch.complex64)
            phase = phase_generator()

            for i in range(self.N_src):
                for j in range(len(self.wvl[i])):
                    OTF_static[i,j,...] = self.Phase2OTF(phase[i,j,...], self.sampling[i][j])

            self.OTF_static = OTF_static 
            '''
            self.OTF_static = self.Phase2OTF(phase_generator(), self.sampling.min().item())

        return self.OTF_static


    def Update(self, reinit_grids=True, reinit_pupils=False):
        # optimizables_copy = [i for i in self.optimizables]
        # self.optimizables = []
        self.InitValues()
        # self.optimizables = optimizables_copy

        if self.N_src != self.sampling.shape[0] or reinit_grids:
            self.InitGrids()
            
        if reinit_pupils:
            self.InitPupils()            
            self.OTF_static_standart = self.StandartStaticOTF().real # Includes only apodizer and pupil
            self.PSF_DL = self.OTF2PSF(self.OTF_static_standart)
            

    def _initialize_PSDs_settings(self, TipTop_flag, PSFAO_flag):
        # The full list of the supported PSD components
        self.PSD_entries = ['fitting', 'WFS noise', 'spatio-temporal', 'aliasing', 'chromatism', 'Moffat']
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
        self.make_tensor = lambda x: torch.tensor(x, device=self.device) if type(x) is not torch.Tensor else x

        self._initialize_PSDs_settings(TipTop, PSFAO)

        if self.device.type != 'cpu':
            self.start = torch.cuda.Event(enable_timing=True)
            self.end   = torch.cuda.Event(enable_timing=True)

        self.norm_regime = norm_regime
        self.norm_scale  = self.make_tensor(1.0) # TODO: num obj x num wvl
        
        if    self.norm_regime == 'sum': self.normalizer = torch.sum
        elif  self.norm_regime == 'max': self.normalizer = torch.amax
        else: self.normalizer = lambda x, dim, keepdim: self.make_tensor(1.0)
        
        self.oversampling = oversampling
    
        # Read data and initialize AO system
        self.piston_filter = None
        self.PR = None
        self.apodizer = None
        
        self.config = AO_config
        self.InitValues()
        self.InitPupils()
        self.InitGrids()
        self.PSDs = {}


    def Controller(self, nF=1000):
        #nTh = 1
        idim = lambda x: x.unsqueeze(1).unsqueeze(1)
        
        vy = idim(self.vy)
        vx = idim(self.vx)
        kx = self.kx_AO.unsqueeze(-1) # add atmo layers dimension: [N_src x nOtf_AO x nOtf_AO x nL]
        ky = self.ky_AO.unsqueeze(-1) # add atmo layers dimension: [N_src x nOtf_AO x nOtf_AO x nL]
        
        Ts = 1.0 / self.HOloop_rate  # sampling time
        delay    = self.HOloop_delay # latency between the measurement and the correction
        loopGain = self.HOloop_gain

        def TransferFunctions(freq, Ts, delay, loopGain):
            z = torch.exp(2j*np.pi*freq*Ts) # no minus according to Sanchit
            hInt = loopGain / self._stabilize(1.0 - 1.0/z, 1e-12)
            rtfInt = 1. / self._stabilize(1+hInt*z**(-delay), 1e-12) # Rejection transfer function
            atfInt = self._stabilize(hInt * z**(-delay) * rtfInt)    # Aliasing transfer function
            ntfInt = self._stabilize(atfInt / z, 1e-12)              # Noise transfer function
            # ntfInt = self._stabilize(hInt * z**(-delay-1)) # according to Sanchit, but it maybe wrong
            return hInt, rtfInt, atfInt, ntfInt

        #f = torch.logspace(-3, torch.log10(torch.tensor([0.5/Ts])).item(), nF)
        #TODO: vectorize it
        f = torch.zeros([self.N_src, nF], device=self.device)
        for i in range(self.N_src):
            f[i,:] = torch.logspace(-3, torch.log10(0.5/Ts[i]), nF)

        _, _, _, ntfInt = TransferFunctions(f, Ts.unsqueeze(-1), delay.unsqueeze(-1), loopGain.unsqueeze(-1))
        self.noise_gain = pdims( torch.trapz(torch.abs(ntfInt)**2, f, dim=1)*2*Ts, 2 )
        
        thetaWind = self.make_tensor(0.0) #torch.linspace(0, 2*np.pi-2*np.pi/nTh, nTh)
        costh = torch.cos(thetaWind) # stays for the uncertainty in the wind diirection

        fi = (-vx*kx - vy*ky)*costh # [N_src x nOtf_AO x nOtf_AO x nL]

        _, _, atfInt, ntfInt = TransferFunctions(fi, pdims(Ts,3),
                                                     pdims(delay,3),
                                                     pdims(loopGain,3))
        # AO transfer function
        self.h1 = idim(self.Cn2_weights) * atfInt #/nTh
        self.h2 = idim(self.Cn2_weights) * atfInt.abs()**2 #/nTh
        self.hn = idim(self.Cn2_weights) * ntfInt.abs()**2 #/nTh

        self.h1 = self.h1.sum(axis=-1) # sum over the atmospheric layers
        self.h2 = self.h2.sum(axis=-1) 
        self.hn = self.hn.sum(axis=-1) 


    def ReconstructionFilter(self, r0, L0, WFS_noise_var):
        Av = torch.sinc(self.WFS_d_sub*self.kx_AO)*torch.sinc(self.WFS_d_sub*self.ky_AO) * torch.exp(1j*np.pi*self.WFS_d_sub*(self.kx_AO+self.ky_AO))
        self.SxAv = ( 2j*np.pi*self.kx_AO*self.WFS_d_sub*Av ).repeat([self.N_src,1,1])
        self.SyAv = ( 2j*np.pi*self.ky_AO*self.WFS_d_sub*Av ).repeat([self.N_src,1,1])

        WFS_wvl = self.GS_wvl

        MV = 0
        # TODO: leave a singleton dimension for nGs in any case
        if WFS_noise_var.shape[1] > 1: # it means that there are several LGS, 0th dim is reserved for the targets
            varNoise = WFS_noise_var.mean(dim=1) # averaging the varience over the LGSs
        else:
            varNoise = WFS_noise_var
        
        W_n = pdims(varNoise / (2*self.kc)**2, 2) #TODO: should it be kc for 500 nm?

        # TODO: isn't this one should be computed for the WFSing wvl?

        self.W_atm = self.VonKarmanSpectrum(r0, L0, self.k2_AO) * self.piston_filter   #TODO: clarify this V

        gPSD = torch.abs(self.SxAv)**2 + torch.abs(self.SyAv)**2 + MV*W_n/self.W_atm / pdims(500e-9/WFS_wvl, 2)**2
        self.Rx = torch.conj(self.SxAv) / gPSD
        self.Ry = torch.conj(self.SyAv) / gPSD
        self.Rx[..., self.nOtf_AO//2, self.nOtf_AO//2] = 1e-9 # For numerical stability
        self.Ry[..., self.nOtf_AO//2, self.nOtf_AO//2] = 1e-9

    
    def SpatioTemporalPSD(self):
        if self.AO_type == 'SCAO':
            A = torch.ones([self.W_atm.shape[0], self.nOtf_AO, self.nOtf_AO], device=self.device) #TODO: fix it. A should be initialized differently
            Ff = self.Rx*self.SxAv + self.Ry*self.SyAv
            psd_ST = (1 + abs(Ff)**2 * self.h2 - 2*torch.real(Ff*self.h1*A)) * self.W_atm * self.mask_corrected_AO
            
        elif self.AO_type == 'LTAO':
            '''
            TODO: unblock for off-axis correction
            Beta = [self.ao.src.direction[0,s], self.ao.src.direction[1,s]]
            fx = Beta[0]*kx_AO
            fy = Beta[1]*ky_AO

            delta_h = h*(fx+fy) - delta_T * wind_speed_nGs_nL * freq_t
            '''
            delta_T  = pdims( (1 + self.HOloop_delay) / self.HOloop_rate, 4)
            delta_h  = -delta_T * self.freq_t * self.wind_speed.view(self.N_src, 1, 1, 1, self.nL)
            P_beta_L = torch.exp(2j*np.pi*delta_h)

            proj = P_beta_L - self.P_beta_DM @ self.W_alpha
            proj_t = torch.conj(torch.permute(proj, (0,1,2,4,3)))
            psd_ST = torch.squeeze(torch.squeeze(torch.abs((proj @ self.C_phi @ proj_t)))) * self.piston_filter * self.mask_corrected_AO
                
        return psd_ST
        

    def NoisePSD(self, WFS_noise_var):
        if self.AO_type == 'SCAO':
            noisePSD = abs(self.Rx**2 + self.Ry**2) / (2*self.kc)**2
            noisePSD = noisePSD * self.piston_filter * self.noise_gain * WFS_noise_var * self.mask_corrected_AO
            
        elif self.AO_type == 'LTAO':
            PW = self.P_beta_DM @ self.W
            PW_t = torch.conj(torch.permute(PW, (0,1,2,4,3)))
            noisePSD = (PW @ self.C_b @ PW_t).squeeze(-1).squeeze(-1)
            
            if WFS_noise_var.shape[1] > 1:
                varNoise = WFS_noise_var.mean(dim=1) # averaging the varience over the LGSs
            else:
                varNoise = WFS_noise_var
            
            noisePSD = noisePSD * self.noise_gain * pdims(varNoise,2) * self.mask_corrected_AO * self.piston_filter
            
        return noisePSD


    def AliasingPSD(self, r0, L0):
        T  = self.WFS_det_clock_rate / self.HOloop_rate
        td = pdims(T * self.HOloop_delay, [-1,3]) # [N_combs x N_src x nOtf_AO x n_Otf_AO x nL]
        T  = pdims(T, [-1,3])

        # Adding 0th dimension for shifted grid pieces
        Rx1 = (2j*np.pi*self.WFS_d_sub * self.Rx).unsqueeze(0)
        Ry1 = (2j*np.pi*self.WFS_d_sub * self.Ry).unsqueeze(0)

        W_mn = self.VonKarmanSpectrum(r0.unsqueeze(0), L0.unsqueeze(0), self.km**2 + self.kn**2) * self.PR
        
        Q = (Rx1*self.km + Ry1*self.kn) * torch.sinc(self.WFS_d_sub*self.km) * torch.sinc(self.WFS_d_sub*self.kn)
        tf = self.h1.unsqueeze(0).unsqueeze(-1) # [N_combs x N_src x nOtf_AO x n_Otf_AO x nL]

        # Add aliasing dimension and more
        vx = self.vx.view(1, self.N_src, 1, 1, self.nL)
        vy = self.vy.view(1, self.N_src, 1, 1, self.nL)
        Cn2_weights = self.Cn2_weights.view(1, self.N_src, 1, 1, self.nL)
        
        # Add atmo layers dimension
        km = self.km.unsqueeze(-1)
        kn = self.kn.unsqueeze(-1)

        avr = (Cn2_weights * tf * torch.sinc(km*vx*T) * torch.sinc(kn*vy*T) * \
            torch.exp( 2j*np.pi*td*(km*vx + kn*vy) )).sum(dim=-1) # sum along atmolayers axis

        # sum along aliasing samples axis      
        aliasing_PSD = torch.sum( W_mn*(Q*avr).abs()**2, dim=0 ) * self.mask_corrected_AO    
        return aliasing_PSD


    def VonKarmanSpectrum(self, r0, L0, freq2):
        return self.cte*r0**(-5/3) * (freq2 + 1/L0**2)**(-11/6)


    def VonKarmanPSD(self, r0, L0):
        return self.VonKarmanSpectrum(r0, L0, self.k2) * self.mask


    #TODO: polychromatic support
    def ChromatismPSD(self, r0, L0):
        # wvlRef = self.wvl[:,0].flatten() if self.wvl.ndim > 1 else self.wvl #TODO: wavelength dependency!
        wvlRef = self.wvl
        # W_atm = self.VonKarmanSpectrum(r0, L0, self.k2_AO) * self.piston_filter
        IOR = lambda lmbd: 23.7+6839.4/(130-(lmbd*1.e6)**(-2))+45.47/(38.9-(lmbd*1.e6)**(-2)) #TODO: proper IOR for wavelength
        n2 = pdims(IOR(self.GS_wvl), 3)
        n1 = pdims(IOR(wvlRef), 2)
        chromatic_PSD = ((n2-n1)/n2)**2 * self.W_atm.unsqueeze(1)
        return chromatic_PSD

    '''
    def JitterCore(self, Jx, Jy, Jxy): #TODO: wavelength dependency!
        u_max = self.sampling * self.D / self.wvl /  (3600*180*1e3/np.pi)
        norm_fact = u_max**2 * (2*np.sqrt(2*np.log(2)))**2
        
        U2 = self.U2.unsqueeze(1) # add wavelengths dimension
        V2 = self.V2.unsqueeze(1) # add wavelengths dimension
        UV = self.UV.unsqueeze(1) # add wavelengths dimension
        
        Djitter = pdims(norm_fact, 2) * (Jx**2 * U2 + Jy**2 * V2 + 2*Jxy*UV)
        return torch.exp(-0.5*Djitter) #TODO: cover Nyquist sampled case
    '''
    
    
    def JitterCore(self, Jx, Jy, Jxy): #TODO: wavelength dependency!
        # Assuming self.Jxy is in degrees, convert it to radians
        cos_theta = torch.cos(torch.deg2rad(Jxy))
        sin_theta = torch.sin(torch.deg2rad(Jxy))

        U_prime = self.U.unsqueeze(1) * cos_theta - self.V.unsqueeze(1) * sin_theta
        V_prime = self.U.unsqueeze(1) * sin_theta + self.V.unsqueeze(1) * cos_theta

        U2_prime, V2_prime = U_prime**2, V_prime**2

        u_max = self.sampling * self.D / self.wvl / (3600*180*1e3/np.pi)
        norm_fact = u_max**2 * (2*np.sqrt(2*np.log(2)))**2

        Djitter = pdims(norm_fact, 2) * (Jx**2 * U2_prime + Jy**2 * V2_prime)
        return torch.exp(-0.5*Djitter) #TODO: cover Nyquist sampled case
    

    def NoiseVariance(self, r0):
        WFS_wvl = self.WFS_wvl
        WFS_Nph = self.WFS_Nph.abs().view(self.N_src, self.nGS)
        r0_WFS =  r0_new(r0.view(self.N_src), WFS_wvl, 0.5e-6).abs()
        WFS_nPix = self.WFS_FOV / self.WFS_n_sub#.repeat(self.N_src, 1)#.view(WFS_Nph.size())
        WFS_pixelScale = self.WFS_psInMas / 1e3 # [arcsec]
       
        # Read-out noise calculation
        nD = torch.maximum( rad2arc*WFS_wvl/self.WFS_d_sub/WFS_pixelScale, self.make_tensor(1.0) ) #spot FWHM in pixels and without turbulence
        # Photon-noise calculation
        nT = torch.maximum( torch.hypot(self.WFS_spot_FWHM.max()/1e3, rad2arc*WFS_wvl/r0_WFS) / WFS_pixelScale, self.make_tensor(1.0) )

        varRON  = np.pi**2/3 * (self.WFS_RON**2/WFS_Nph**2) * (WFS_nPix**2/nD).unsqueeze(-1)**2
        varShot = np.pi**2 / (2*WFS_Nph) * (nT/nD).unsqueeze(-1)**2
        
        # Noise variance calculation
        varNoise = self.WFS_excessive_factor * (varRON+varShot) * (500e-9/WFS_wvl).unsqueeze(-1)**2

        return varNoise


    def TomographicReconstructors(self, r0, L0, WFS_noise_var, inv_method='lstsq'):
        h = self.h.view(self.N_src, 1, 1, 1, self.nL)
        kx = pdims(self.kx_AO, 2)
        ky = pdims(self.ky_AO, 2)
        GS_dirs_x = self.GS_dirs_x.view(self.N_src, 1, 1, self.nGS, 1)
        GS_dirs_y = self.GS_dirs_y.view(self.N_src, 1, 1, self.nGS, 1)
        
        diag_mask = lambda N: torch.eye(N, device=self.device).view(1,1,1,N,N).expand(self.N_src, self.nOtf_AO, self.nOtf_AO, -1, -1)
        
        M = 2j*np.pi*self.k_AO * torch.sinc(self.WFS_d_sub*self.kx_AO) * torch.sinc(self.WFS_d_sub*self.ky_AO)
        M = pdims(M, 2).expand(self.N_src, self.nOtf_AO, self.nOtf_AO, self.nGS, self.nGS) * diag_mask(self.nGS) # [N_src x nOtf x nOtf x nGS x nGS]
        
        P = torch.exp( 2j*np.pi*h * (kx*GS_dirs_x + ky*GS_dirs_y) ) # N_src x nOtf x nOtf x nGS x nL

        #MP = M @ P # should be the same as below
        MP   = torch.einsum('nwhik,nwhkj->nwhij', M, P)  # N_src x nOtf x nOtf x nL x nGS
        MP_t = torch.conj(MP.permute(0,1,2,4,3))

        # self.MP = MP
        # self.M = M
        # self.P = P

        fix_dims = lambda x_, N:  torch.diag_embed(x_).view(self.N_src, 1, 1, N, N).expand(self.N_src, self.nOtf_AO, self.nOtf_AO, N, N)

        varNoise = WFS_noise_var.to(dtype=MP.dtype)

        self.C_b = fix_dims( varNoise, self.nGS )

        kernel = self.VonKarmanSpectrum(r0.to(dtype=MP.dtype), L0, self.k2_AO) * self.piston_filter
        # self.kernel = kernel
        self.C_phi = pdims(kernel, 2) * fix_dims(self.Cn2_weights, self.nL)

        # Tikhonov_reg = False

        # Inversion happens relative to the last two dimensions of the these tensors
        if inv_method == 'standart':
            W_tomo = (self.C_phi @ MP_t) @ torch.linalg.pinv(MP @ self.C_phi @ MP_t + self.C_b, rcond=1e-2)

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
            W_tomo = torch.linalg.lstsq(A, B).solution.transpose(-2, -1)           
        else:
            raise ValueError('Unknown inversion method specified.') 
        
        '''
        DMS_opt_dir = torch.tensor([0.0, 0.0])
        opt_weights = 1.0

        theta_x = torch.tensor([DMS_opt_dir[0]/206264.8 * np.cos(DMS_opt_dir[1]*np.pi/180)])
        theta_y = torch.tensor([DMS_opt_dir[0]/206264.8 * np.sin(DMS_opt_dir[1]*np.pi/180)])

        P_L  = torch.zeros([nOtf, nOtf, 1, nL], dtype=torch.complex64)
        fx = theta_x*kx
        fy = theta_y*ky
        P_DM = torch.unsqueeze(torch.unsqueeze(torch.exp(2j*np.pi*h_DM*(fx+fy))*mask_corrected,2),3)
        P_DM_t = torch.conj(P_DM.permute(0,1,3,2))
        for l in range(nL):
            P_L[:,:,0,l] = torch.exp(2j*np.pi*h[l]*(fx+fy))*mask_corrected

        P_opt = torch.linalg.pinv((P_DM_t @ P_DM)*opt_weights, rcond=1e-2) @ ((P_DM_t @ P_L)*opt_weights)

        src_direction = torch.tensor([0,0])
        fx = src_direction[0]*kx
        fy = src_direction[1]*ky
        P_beta_DM = torch.unsqueeze(torch.unsqueeze(torch.exp(2j*np.pi*h_DM*(fx+fy))*mask_corrected,2),3)
        '''

        self.W_tomo = W_tomo

        self.W = self.P_opt @ W_tomo

        wDir_x = torch.cos(self.wind_dir * np.pi / 180.0).view(self.N_src, 1, 1, 1, self.nL)
        wDir_y = torch.sin(self.wind_dir * np.pi / 180.0).view(self.N_src, 1, 1, 1, self.nL)

        wind_speed = self.wind_speed.view(self.N_src, 1, 1, 1, self.nL)

        self.freq_t = wDir_x*kx + wDir_y*ky

        samp_time = 1.0 / self.HOloop_rate
        www = 2j*torch.pi*pdims(self.k_AO,2) * torch.sinc(pdims(samp_time * self.WFS_det_clock_rate,4) * wind_speed * self.freq_t)

        # self.www = www

        #MP_alpha_L = www*torch.sinc(WFS_d_sub*kx_1_1)*torch.sinc(WFS_d_sub*ky_1_1)\
        #                                *torch.exp(2j*np.pi*h*(kx_nGs_nL*GS_dirs_x_nGs_nL + ky_nGs_nL*GS_dirs_y_nGs_nL))
        
        MP_alpha_L = www * P * (torch.sinc(self.WFS_d_sub*kx) * torch.sinc(self.WFS_d_sub*ky))
        
        self.MP_alpha_L = MP_alpha_L
        
        self.W_alpha = self.W @ MP_alpha_L


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
        E = (beta-1) / (np.pi*ax*ay)
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
        
        r0  = pdims(self.r0, 2)
        L0  = pdims(self.L0, 2)

        if self.PSD_include['Moffat']:
            amp   = pdims(self.amp,   2)
            b     = pdims(self.b,     2)
            alpha = pdims(self.alpha, 2)
            beta  = pdims(self.beta,  2)
            ratio = pdims(self.ratio, 2)
            theta = pdims(self.theta, 2)

        if self.PSD_include['WFS noise'] or self.PSD_include['spatio-temporal'] or self.PSD_include ['aliasing']:
            # WFS_wvl = pdims(self.WFS_wvl, 2)
            WFS_noise_var = (self.dn.unsqueeze(-1) + self.NoiseVariance(r0)).abs() # [rad^2] at WFSing wavelength TODO: really at this wvl?
                        
            self.vx = self.wind_speed * torch.cos( torch.deg2rad(self.wind_dir) )
            self.vy = self.wind_speed * torch.sin( torch.deg2rad(self.wind_dir) )

            self.Controller()
            self.ReconstructionFilter(r0.abs(), L0.abs(), WFS_noise_var)
            
            if self.AO_type == 'LTAO':
                self.TomographicReconstructors(r0.abs(), L0.abs(), WFS_noise_var, inv_method='lstsq')
        

        # All PSD components are computed in radians and here normalized to [nm^2]
        dk = 2*self.kc/self.nOtf_AO
        PSD_norm = lambda wvl: (dk*wvl*1e9/2/np.pi)**2

        # self.dk = dk
        
        # Most contributors are computed for 500 [nm], wavelengths-dependant inputs as well are assumed for 500 [nm]
        # But WFS operates on another wavelength, so this PSD components is normalized for WFSing wvl 
        # Put all contributiors together and sum up the resulting PSD

        self.PSDs = {entry: self.make_tensor(0.0) for entry in self.PSD_entries}

        if self.PSD_include['fitting']:
            self.PSDs['fitting'] = self.VonKarmanPSD(r0.abs(), L0.abs()).unsqueeze(1)
    
        if self.PSD_include['WFS noise']:
            self.PSDs['WFS noise'] = self.NoisePSD(WFS_noise_var).unsqueeze(1) #TODO: to which wavelength to normalize?
        
        if self.PSD_include['spatio-temporal']:
            self.PSDs['spatio-temporal'] = self.SpatioTemporalPSD().unsqueeze(1)
        
        if self.PSD_include['aliasing']:
            self.PSDs['aliasing'] = self.AliasingPSD(r0.abs(), L0).unsqueeze(1)
        
        if self.PSD_include['chromatism']:
            self.PSDs['chromatism'] = self.ChromatismPSD(r0.abs(), L0.abs()) # no need to add dimension since it's polychromatic already

        if self.PSD_include['Moffat']:
            self.PSDs['Moffat'] = self.MoffatPSD(amp, b, alpha, beta, ratio, theta).unsqueeze(1)

        #TODO: anisoplanatism!
        #TODO: SLAO support!
        #TODO: differential refraction!
        
        PSD = self.PSDs['fitting'] + self.PSD_padder(
              self.PSDs['WFS noise'] + \
              self.PSDs['spatio-temporal'] + \
              self.PSDs['aliasing'] + \
              self.PSDs['chromatism'] + \
              self.PSDs['Moffat']) # [N_scr x N_wvl x nOtf_AO x nOtf_AO], [rad^2] at 500 nm
        
        # self.PSD = PSD.abs() * PSD_norm(500e-9) # [nm^2] at 500 nm
        
        # self.psd = PSD.clone()
        
        self.PSD = PSD * PSD_norm(500e-9) # [nm^2] at 500 nm #TODO: change to atmo wavelength from config
        # self.PSD = PSD * pdims(PSD_norm(self.wvl), 2) # [nm^2] at 500 nm #TODO: change to atmo wavelength from config

        return self.PSD
    

    def OTF2PSF(self, OTF):
        '''
        # s = tuple([OTF.shape[-2] + 1-self.nPix%2]*2)
        # PSF = fft.fftshift(fft.ifft2(fft.ifftshift(OTF*self.center_aligner, dim=(-2,-1)), s=s), dim=(-2,-1)).abs()
        PSF = fft.fftshift(fft.ifft2(fft.ifftshift(OTF*self.center_aligner, dim=(-2,-1))), dim=(-2,-1)).abs()
        return interpolate(PSF, size=(self.nPix, self.nPix), mode='bilinear') if OTF.shape[-1] != self.nPix else PSF    
        '''
        # PSF_big = fft.fftshift(fft.ifft2(fft.ifftshift(OTF*self.center_aligner, dim=(-2,-1)), s=s), dim=(-2,-1)).abs()
        PSF_big = fft.fftshift(fft.ifft2(fft.ifftshift(OTF*self.center_aligner, dim=(-2,-1))), dim=(-2,-1)).abs()

        PSF = []
        for i in range(self.wvl.shape[-1]):
            n = 0 if PSF_big.shape[1] == 1 else i # In the case OTF is only computed for minimal sampling
            transform = transforms.CenterCrop((self.nOtfs[i], self.nOtfs[i]))
            PSF.append(interpolate(transform(PSF_big[:,n,...]).unsqueeze(1), size=(self.nPix, self.nPix), mode='bilinear') if OTF.shape[-1] != self.nPix else transform(PSF_big[:,n,...]).unsqueeze(1))
            
        return torch.hstack(PSF)


    def PSD2PSF(self, PSD, OTF_static):
        F   = pdims(self.F,   2)
        dx  = pdims(self.dx,  2)
        dy  = pdims(self.dy,  2)
        bg  = pdims(self.bg,  2)
        Jx  = pdims(self.Jx,  3) if self.Jx.ndim  == 1 else pdims(self.Jx,  2)
        Jy  = pdims(self.Jy,  3) if self.Jy.ndim  == 1 else pdims(self.Jy,  2)
        Jxy = pdims(self.Jxy, 3) if self.Jxy.ndim == 1 else pdims(self.Jxy, 2)

        # Removing the DC component
        PSD[..., self.nOtf//2, self.nOtf//2] = 0.0
        # Computing OTF from PSD, real is to remove the imaginary part that appears due to numerical errors
        cov = fft.fftshift(fft.fft2(fft.ifftshift(PSD, dim=(-2,-1))), dim=(-2,-1)) # FFT axes are -2,-1 #TODO: real FFT?
     
        self.cov = cov
  
        # Computing the Structure Function from the covariance
        SF = 2*(cov.abs().amax(dim=(-2,-1), keepdim=True) - cov).real
        # self.SF = SF
        # Phasor to shift the PSF with the subpixel accuracy
        fftPhasor = torch.exp( -np.pi*1j * pdims(self.sampling_factor,2) * (self.U.unsqueeze(1)*dx + self.V.unsqueeze(1)*dy) )
        OTF_turb  = torch.exp( -0.5 * SF * pdims(2*np.pi*1e-9/self.wvl,2)**2 )
        self.OTF_turb = OTF_turb
        
        # Compute the residual tip/tilt kernel
        OTF_jitter = self.JitterCore(Jx.abs(), Jy.abs(), Jxy.abs())
        # Resulting combined OTF
        self.OTF = OTF_turb * OTF_static * fftPhasor * OTF_jitter
        self.fftPhasor = fftPhasor
        self.OTF_jitter = OTF_jitter

        PSF_out = self.OTF2PSF(self.OTF)
        self.norm_scale = self.normalizer(PSF_out, dim=(-2,-1), keepdim=True)
        
        torch.cuda.empty_cache()
        
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
                # print(f"Transferring '{name}' to device '{device}'")
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
