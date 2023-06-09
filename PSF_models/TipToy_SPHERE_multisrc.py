#%%
import sys
sys.path.insert(0, '..')

import numpy as np
import torch
from torch import fft, nn
from torch.nn.functional import interpolate
import scipy.special as spc
from astropy.io import fits
import time

import matplotlib.pyplot as plt

from tools.utils import rad2mas, rad2arc, deg2rad, r0_new, pdims, min_2d

# import torch.autograd.profiler as profiler

class TipTorch(torch.nn.Module):
    
    def InitPupils(self):
        # with profiler.record_function("INIT PUPLAS"):
        pupil_path = self.config['telescope']['PathPupil']
        pupil_apodizer = self.config['telescope']['PathApodizer']
        
        self.pupil    = self.make_tensor(fits.getdata(pupil_path).astype('float'))
        self.apodizer = self.make_tensor(fits.getdata(pupil_apodizer).astype('float'))
        
        
    def InitValues(self):
        # with profiler.record_function("INIT PARAMETERS"):
        # Reading parameters from the file
        num_src = self.config['NumberSources']
        self.N_src = int(num_src)
        
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
        #self.zenith_angle  = torch.tensor(self.AO_config['telescope']['ZenithAngle'], device=self.device) # [deg] #TODO: telescope zenith != sample zenith?
        self.zenith_angle  = self.config['telescope']['ZenithAngle']
        self.airmass       = 1.0 / torch.cos(self.zenith_angle * deg2rad)

        self.GS_wvl     = self.config['sources_HO']['Wavelength'][0].item() #[m]
        #self.GS_height  = self.AO_config['sources_HO']['Height'] * self.airmass #[m]
        self.wind_speed  = self.config['atmosphere']['WindSpeed']
        self.wind_dir    = self.config['atmosphere']['WindDirection']
        self.Cn2_weights = min_2d(self.config['atmosphere']['Cn2Weights'])
        self.Cn2_heights = min_2d(self.config['atmosphere']['Cn2Heights']) * self.airmass.unsqueeze(1) # [m]
        
        #self.stretch     = 1.0 / (1.0-self.Cn2_heights/self.GS_height)
        self.h  = self.Cn2_heights #* self.stretch
        self.nL = self.Cn2_heights.size(0)

        self.WFS_d_sub = self.config['sensor_HO']['SizeLenslets'] #TODO: seems like it's absent
        self.WFS_n_sub = self.config['sensor_HO']['NumberLenslets']

        self.WFS_det_clock_rate = self.config['sensor_HO']['ClockRate'].flatten() # [(?)]
        self.WFS_FOV = self.config['sensor_HO']['FieldOfView']
        self.WFS_RON = self.config['sensor_HO']['SigmaRON']
        self.WFS_psInMas = self.config['sensor_HO']['PixelScale']
        self.WFS_wvl = self.make_tensor(self.GS_wvl) #TODO: clarify this
        self.WFS_spot_FWHM = self.make_tensor(self.config['sensor_HO']['SpotFWHM'][0])
        self.WFS_excessive_factor = self.config['sensor_HO']['ExcessNoiseFactor']
        self.WFS_Nph = self.config['sensor_HO']['NumberPhotons']

        self.HOloop_rate  = self.config['RTC']['SensorFrameRate_HO'] # [Hz] (?)
        self.HOloop_delay = self.config['RTC']['LoopDelaySteps_HO'] # [ms] (?)
        self.HOloop_gain  = self.config['RTC']['LoopGain_HO']

        # Initialiaing the main optimizable parameters
        self.r0  = rad2arc*0.976*self.config['atmosphere']['Wavelength'] / self.config['atmosphere']['Seeing']
        self.L0  = self.config['atmosphere']['L0'] # [m]
        self.F   = torch.ones (self.N_src, self.N_wvl, device=self.device)
        self.bg  = torch.zeros(self.N_src, self.N_wvl, device=self.device)
        self.dx  = torch.zeros(self.N_src, device=self.device)
        self.dy  = torch.zeros(self.N_src, device=self.device)
        self.Jx  = torch.ones (self.N_src, device=self.device)*0.1
        self.Jy  = torch.ones (self.N_src, device=self.device)*0.1
        self.Jxy = torch.ones (self.N_src, device=self.device)*0.1
        self._optimizables = ['r0', 'F', 'dx', 'dy', 'bg', 'Jx', 'Jy', 'Jxy']

        if self.PSD_include['Moffat']:
            self.amp   = torch.ones (self.N_src, device=self.device)*4.0  # Phase PSD Moffat amplitude [rad²]
            self.b     = torch.ones (self.N_src, device=self.device)*0.01 # Phase PSD background [rad² m²]
            self.alpha = torch.ones (self.N_src, device=self.device)*0.1  # Phase PSD Moffat alpha [1/m]
            self.beta  = torch.ones (self.N_src, device=self.device)*2  # Phase PSD Moffat beta power law
            self.ratio = torch.ones (self.N_src, device=self.device)      # Phase PSD Moffat ellipticity
            self.theta = torch.zeros(self.N_src, device=self.device)      # Phase PSD Moffat angle
            self._optimizables += ['amp', 'b', 'alpha', 'beta', 'ratio', 'theta']

        if self.PSD_include['WFS noise'] or self.PSD_include['spatio-temporal'] or self.PSD_include ['aliasing']:
            self.dn  = torch.zeros(self.N_src, device=self.device)
            self._optimizables += ['dn']

        for name in self._optimizables:
            setattr(self, name, nn.Parameter(getattr(self, name)))


    @property
    def optimizables(self):
        return self._optimizables
    

    @optimizables.setter
    def optimizables(self, new_optimizables):
        self._make_optimizable(new_optimizables)
        self._optimizables = new_optimizables
    

    def _make_optimizable(self, args):
        for name in args:
            if name not in self._optimizables and not isinstance(getattr(self, name), nn.Parameter):
                setattr(self, name, nn.Parameter(getattr(self, name)))
        for name in self.optimizables:
            if name not in args and isinstance(getattr(self, name), nn.Parameter):
                buffer = getattr(self, name).detach().clone()
                delattr(self, name)
                setattr(self, name, buffer)


    def _fftAutoCorr(self, x):
        x_fft = fft.fft2(x)
        return fft.fftshift( fft.ifft2(x_fft*torch.conj(x_fft))/x.size(0)*x.size(1) )


    def InitGrids(self):
        # with profiler.record_function("INIT GRIDDIES"):
        # Initialize grids
        # for all generated PSDs within one batch sampling musat be the same

        to_odd = lambda f: int( np.ceil(f)//2 * 2 - 1) if not f % 2 else int(f)
        # to_even = lambda f: int( np.ceil(f)//2 * 2 )

        # Manage sampling
        pixels_per_l_D = self.wvl*rad2mas / (self.psInMas*self.D)
        self.sampling_factor = torch.ceil(2.0/pixels_per_l_D) * self.oversampling # to avoid aliasing or to provide oversampling
        self.sampling = self.sampling_factor * pixels_per_l_D
        self.nOtf = self.nPix * self.sampling_factor.max().item()
        self.nOtf = to_odd(self.nOtf) #Forces nOtf to be odd

        self.dk = 1/self.D/self.sampling.max() # PSD spatial frequency step
        self.cte = (24*spc.gamma(6/5)/5)**(5/6)*(spc.gamma(11/6)**2/(2*np.pi**(11/3)))

        # print('Sampling factor: ', *self.sampling_factor.detach().squeeze().cpu().numpy().tolist())
        # print('Sampling: ', *self.sampling.detach().squeeze().cpu().numpy().tolist())
        # print('Wavelengths: ', *(self.wvl*1e9).detach().squeeze().cpu().numpy().tolist())
        # print('dk: ', self.dk)
        
        def gen_grid(N):
            factor = 0.5*(1-N%2)
            return torch.meshgrid(*[torch.linspace(-N//2+N%2+factor, N//2-factor, N, device=self.device)]*2, indexing = 'ij')

        def stabilize(tensor, eps=1e-9): # Add to increase numerical stability by replacing near-zero values with eps
            tensor[torch.abs(tensor) < eps] = eps
            return tensor

        # Initialize spatial frequencies
        self.kx, self.ky = gen_grid(self.nOtf) # PSD spatial frequencies
        self.kx, self.ky = self.kx * self.dk, self.ky * self.dk

        self.k2 = self.kx**2 + self.ky**2
        self.k = torch.sqrt(self.k2)

        stabilize(self.kx)
        stabilize(self.ky)
        stabilize(self.k2)
        stabilize(self.k )

        if self.PSD_include['Moffat']:
            self.kxy = self.kx * self.ky
            self.kx2 = self.kx**2
            self.ky2 = self.ky**2
            
            stabilize(self.kxy)
            stabilize(self.kx2)
            stabilize(self.ky2)

        # Compute the frequency mask for the AO corrected and uncorrected regions
        rim_width = 0.1
        self.mask_corrected = torch.zeros_like(self.k2).int()
        self.mask_rim_in    = torch.zeros_like(self.k2).int() # masks to select the regions close to the corrected/uncorrected freqs split
        self.mask_rim_out   = torch.zeros_like(self.k2).int()
        self.mask_corrected[self.k2 <= self.kc**2] = 1
        self.mask_rim_in   [self.k2 <= (1-rim_width)*self.kc**2] = 1
        self.mask_rim_out  [self.k2 <= (1+rim_width)*self.kc**2] = 1
        self.mask_rim_in  = self.mask_corrected - self.mask_rim_in
        self.mask_rim_out = self.mask_rim_out - self.mask_corrected
        
        self.mask = 1 - self.mask_corrected
        
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
            # This is thing is 5D: (aliased samples) x (N src) x (Atmospheric layers) x (kx) x (ky) TODO: NO, it's 4D
            m = self.make_tensor(ids[:,0])
            n = self.make_tensor(ids[:,1])
            self.N_combs = m.shape[0]
            
            self.km = self.kx_AO.repeat([self.N_combs,1,1,1]) - pdims(m/self.WFS_d_sub, 3)
            self.kn = self.ky_AO.repeat([self.N_combs,1,1,1]) - pdims(n/self.WFS_d_sub, 3)

        # Initialize OTF frequencies
        self.U, self.V = gen_grid(self.nOtf)
        self.U, self.V = pdims(self.U/self.nOtf, -1), pdims(self.V/self.nOtf, -1)

        self.U2  = self.U**2
        self.V2  = self.V**2
        self.UV  = self.U*self.V
        self.UV2 = self.U**2 + self.V**2
        
        stabilize(self.U2 )
        stabilize(self.V2 )
        stabilize(self.UV )
        stabilize(self.UV2)
        
        self.center_aligner = torch.exp( 1j*np.pi*(self.U+self.V) * (1-self.nPix%2))

        # with profiler.record_function("COMPUTE PUPLAS"):
        # Compute pupil polychromatic OTFs
        ids = np.unravel_index(np.unique(self.wvl.cpu().numpy(), return_index=True)[1], self.wvl.shape)
        all_wvls = self.wvl[ids].cpu().numpy().tolist()
        all_samplings = self.sampling[ids]

        pupil_pixels = self.pupil.shape[0]
        for i, wvl in enumerate(all_wvls):
            if wvl not in self.OTF_static_dict.keys():
                padding_size = int(pupil_pixels*all_samplings[i]/2 - pupil_pixels/2)
                pupil_padder = torch.nn.ZeroPad2d(padding_size).to(self.device)
                
                allocated_memory = torch.cuda.memory_allocated() + torch.cuda.memory_reserved()
                total_memory     = torch.cuda.get_device_properties(self.device).total_memory
                padded_tensor_size = self.pupil.element_size() * (self.pupil.shape[0] + 2*padding_size)**2
                
                # Decide where to perform the computation depending on the available memory
                if allocated_memory + padded_tensor_size > total_memory:
                    torch.cuda.empty_cache()
                    pupil_padded = pupil_padder(self.pupil.cpu()*self.apodizer.cpu())
                else:
                    pupil_padded = pupil_padder(self.pupil*self.apodizer)
                    
                fl_even = self.nOtf % 2 == 0 and pupil_padded.shape[0] % 2 == 0
                pupil_padded = pupil_padded[:-1, :-1] if fl_even else pupil_padded # to center-align if number of pixels is even

                OTF_static_ = interpolate(\
                    torch.real(self._fftAutoCorr(pupil_padded))[None,None,...],
                    size=(self.nOtf,self.nOtf), mode='bilinear', align_corners=False).squeeze()
                
                self.OTF_static_dict[wvl] = ( OTF_static_ / OTF_static_.max() ).to(self.device)
                torch.cuda.empty_cache()

        buf_obj = []
        for obj_id in range(self.N_src): # iterate over all sources
            buf_wvl = []
            for wvl_id in range(self.wvl.shape[1]): # iterate over the lambdas of the current object
                buf_wvl.append(self.OTF_static_dict[self.wvl[obj_id,wvl_id].cpu().item()]) # get the OTF for the current wavelength
            buf_obj.append(torch.stack(buf_wvl))
        self.OTF_static = torch.stack(buf_obj) # stack all sources into the 4D tensor: N_obj x N_wvl x nOtf x nOtf

        self.PSD_padder = torch.nn.ZeroPad2d((self.nOtf-self.nOtf_AO)//2)

        # Piston filter, filters centering pixel
        def PistonFilter(f):
            x = (np.pi*self.D*f).cpu().numpy()
            R = spc.j1(x)/x
            piston_filter = self.make_tensor(1.0-4*R**2)
            piston_filter[..., self.nOtf_AO//2, self.nOtf_AO//2] *= 1-self.nOtf_AO%2
            return stabilize(piston_filter)

        # Detects oddity to determine if piston needed
        self.piston_filter = PistonFilter(self.k_AO)
        
        if self.PSD_include['aliasing']:
            self.PR = PistonFilter(torch.hypot(self.km, self.kn))

        # Diffraction-limited PSF
        self.PSF_DL = self.OTF2PSF(self.OTF_static)
        torch.cuda.empty_cache()
    

    def Update(self, reinit_grids=True, reinit_pupils=False):
        optimizables_copy = [i for i in self.optimizables]
        self.optimizables = []
        self.InitValues()
        self.optimizables = optimizables_copy
        if reinit_pupils: self.InitPupils()
        if reinit_grids: self.InitGrids()


    def _initialize_PSDs_settings(self, TipTop_flag, PSFAO_flag):
        # The full list of the supported PSD components
        self.PSD_entries = ['fitting', 'WFS noise', 'spatio-temporal', 'aliasing', 'chromatism', 'Moffat']
        # The list of the PSD components that are included in the current simulation
        self.PSD_include = {key:False for key in self.PSD_entries}

        if TipTop_flag:
            self.PSD_include = {key:True for key in self.PSD_entries if key != 'Moffat'}
        
        self.PSD_include['Moffat']  = PSFAO_flag
        self.PSD_include['fitting'] = True #TODO: open-loop case

        if not TipTop_flag and not PSFAO_flag:
            raise ValueError('At least one of the following must be True: TipTop or PSFAO')


    def __init__(self, AO_config, norm_regime='sum', device=torch.device('cpu'), TipTop=True, PSFAO=False, oversampling=1):
        super().__init__()
        
        self.device = device
        self.make_tensor = lambda x: torch.tensor(x, device=self.device) if type(x) is not torch.Tensor else x

        self._initialize_PSDs_settings(TipTop, PSFAO)
        self.OTF_static_dict = {}

        if self.device.type != 'cpu':
            self.start = torch.cuda.Event(enable_timing=True)
            self.end   = torch.cuda.Event(enable_timing=True)

        self.norm_regime = norm_regime
        self.norm_scale  = self.make_tensor(1.0) # TODO: num obj x num wvl
        
        if self.norm_regime == 'sum':
            self.normalizer = torch.sum
        elif self.norm_regime == 'max':
            self.normalizer = torch.amax
        else:
            self.normalizer = lambda x, dim, keepdim: self.make_tensor(1.0)
        
        self.oversampling = oversampling
    
        # Read data and initialize AO system
        self.config = AO_config
        self.InitValues()
        self.InitPupils()
        self.InitGrids()
        self.PSDs = {}


    def Controller(self, nF=1000):
        #nTh = 1
        Ts = 1.0 / self.HOloop_rate # samplingTime
        delay = self.HOloop_delay #latency
        loopGain = self.HOloop_gain

        def TransferFunctions(freq, Ts, delay, loopGain):
            z = torch.exp(-2j*np.pi*freq*Ts)
            hInt = loopGain/(1.0 - z**(-1.0))
            rtfInt = 1.0 / (1+hInt*z**(-delay))
            atfInt = hInt * z**(-delay)*rtfInt
            ntfInt = atfInt / z
            return hInt, rtfInt, atfInt, ntfInt

        #f = torch.logspace(-3, torch.log10(torch.tensor([0.5/Ts])).item(), nF)
        f = torch.zeros([self.N_src, nF], device=self.device)
        for i in range(self.N_src): 
            f[i,:] = torch.logspace(-3, torch.log10(0.5/Ts[i]), nF)

        _, _, _, ntfInt = TransferFunctions(f, Ts.unsqueeze(1), delay.unsqueeze(1), loopGain.unsqueeze(1))
        self.noise_gain = pdims( torch.trapz(torch.abs(ntfInt)**2, f, dim=1)*2*Ts, 2 )

        thetaWind = self.make_tensor(0.0) #torch.linspace(0, 2*np.pi-2*np.pi/nTh, nTh)
        costh = torch.cos(thetaWind) #TODO: what is thetaWind?

        fi = -self.vx*self.kx_AO*costh - self.vy*self.ky_AO*costh

        _, _, atfInt, ntfInt = TransferFunctions(fi, pdims(Ts,3),
                                                     pdims(delay,3),
                                                     pdims(loopGain,3))
        # AO transfer function
        self.h1 = pdims(self.Cn2_weights,2) * atfInt #/nTh
        self.h2 = pdims(self.Cn2_weights,2) * abs(atfInt)**2 #/nTh
        self.hn = pdims(self.Cn2_weights,2) * abs(ntfInt)**2 #/nTh

        self.h1 = torch.sum(self.h1, axis=1) #sum over the atmospheric layers
        self.h2 = torch.sum(self.h2, axis=1) 
        self.hn = torch.sum(self.hn, axis=1) 


    def ReconstructionFilter(self, r0, L0, WFS_noise_var):
        Av = torch.sinc(self.WFS_d_sub*self.kx_AO)*torch.sinc(self.WFS_d_sub*self.ky_AO) * torch.exp(1j*np.pi*self.WFS_d_sub*(self.kx_AO+self.ky_AO))
        self.SxAv = ( 2j*np.pi*self.kx_AO*self.WFS_d_sub*Av ).repeat([self.N_src,1,1])
        self.SyAv = ( 2j*np.pi*self.ky_AO*self.WFS_d_sub*Av ).repeat([self.N_src,1,1])

        MV = 0
        Wn = WFS_noise_var / (2*self.kc)**2 #TODO: should it be kc for 500 nm?
        # TODO: isn't this one computed for the WFSing wvl?

        self.W_atm = self.VonKarmanSpectrum(r0, L0, self.k2_AO) * self.piston_filter   #TODO: clarify this V

        gPSD = torch.abs(self.SxAv)**2 + torch.abs(self.SyAv)**2 + MV*Wn/self.W_atm / (500e-9/self.GS_wvl)**2
        self.Rx = torch.conj(self.SxAv) / gPSD
        self.Ry = torch.conj(self.SyAv) / gPSD
        self.Rx[..., self.nOtf_AO//2, self.nOtf_AO//2] = 1e-9 # For numerical stability
        self.Ry[..., self.nOtf_AO//2, self.nOtf_AO//2] = 1e-9

    
    def SpatioTemporalPSD(self):
        A = torch.ones([self.W_atm.shape[0], self.nOtf_AO, self.nOtf_AO], device=self.device) #TODO: fix it. A should be initialized differently
        Ff = self.Rx*self.SxAv + self.Ry*self.SyAv
        psd_ST = (1 + abs(Ff)**2 * self.h2 - 2*torch.real(Ff*self.h1*A)) * self.W_atm * self.mask_corrected_AO
        return psd_ST


    def NoisePSD(self, WFS_noise_var):
        noisePSD = abs(self.Rx**2 + self.Ry**2) / (2*self.kc)**2
        noisePSD = noisePSD * self.piston_filter * self.noise_gain * WFS_noise_var * self.mask_corrected_AO
        return noisePSD


    def AliasingPSD(self, r0, L0):
        T = pdims(self.WFS_det_clock_rate / self.HOloop_rate, [-1,3])
        td = T * pdims(self.HOloop_delay, [-1,3])

        Rx1 = pdims(2j*np.pi*self.WFS_d_sub * self.Rx, -1)
        Ry1 = pdims(2j*np.pi*self.WFS_d_sub * self.Ry, -1)

        W_mn = (self.km**2 + self.kn**2 + 1/L0.unsqueeze(0)**2)**(-11/6)

        Q = (Rx1*self.km + Ry1*self.kn) * torch.sinc(self.WFS_d_sub*self.km) * torch.sinc(self.WFS_d_sub*self.kn)
        tf = pdims(self.h1,-1).unsqueeze(2)

        # Reference to the variables but with diamensions expansion
        vx = pdims(self.vx,-1)
        vy = pdims(self.vy,-1)
        km = self.km.unsqueeze(2)
        kn = self.kn.unsqueeze(2)

        avr = (pdims(self.Cn2_weights, [-1,2]) * tf * 
               torch.sinc(km*vx*T) * torch.sinc(kn*self.vy*T) * \
               torch.exp(2j*np.pi*km*vx*td) * torch.exp(2j*np.pi*kn*vy*td)).sum(dim=2) # sum along layers axis as well as along aliasing samples axis      

        aliasing_PSD = torch.sum(self.PR*W_mn*abs(Q*avr)**2, dim=0) * self.cte * r0**(-5/3) * self.mask_corrected_AO    
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
        IOR = lambda lmbd: 23.7+6839.4/(130-(lmbd*1.e6)**(-2))+45.47/(38.9-(lmbd*1.e6)**(-2))
        n2 = IOR(self.GS_wvl)
        n1 = IOR(wvlRef)
        chromatic_PSD = pdims(((n2-n1)/n2)**2, 2) * self.W_atm.unsqueeze(1)
        return chromatic_PSD


    def JitterCore(self, Jx, Jy, Jxy): #TODO: wavelength dependency!
        u_max = self.sampling*self.D/self.wvl/(3600*180*1e3/np.pi)
        norm_fact = u_max**2 * (2*np.sqrt(2*np.log(2)))**2
        Djitter = pdims(norm_fact, 2) * (Jx**2 * self.U2 + Jy**2 * self.V2 + 2*Jxy*self.UV).unsqueeze(1)
        return torch.exp(-0.5*Djitter) #TODO: cover Nyquist sampled case


    def NoiseVariance(self, r0):
        r0_WFS = r0_new(r0.abs(), self.WFS_wvl, 0.5e-6).flatten() #from (Nsrc x 1 x 1) to (Nsrc)
        WFS_nPix = self.WFS_FOV / self.WFS_n_sub
        WFS_pixelScale = self.WFS_psInMas / 1e3 # [arcsec]
       
        # Read-out noise calculation
        nD = torch.maximum( rad2arc*self.WFS_wvl/self.WFS_d_sub/WFS_pixelScale, self.make_tensor(1.0) ) #spot FWHM in pixels and without turbulence
        # Photon-noise calculation
        nT = torch.maximum( torch.hypot(self.WFS_spot_FWHM.max()/1e3, rad2arc*self.WFS_wvl/r0_WFS) / WFS_pixelScale, self.make_tensor(1.0) )

        varRON  = np.pi**2/3 * (self.WFS_RON**2/self.WFS_Nph**2) * (WFS_nPix**2/nD)**2
        varShot = np.pi**2/(2*self.WFS_Nph) * (nT/nD)**2
        
        # Noise variance calculation
        varNoise = self.WFS_excessive_factor * (varRON+varShot) # TODO: clarify with Benoit and Thierry
        return varNoise * (500e-9/self.GS_wvl)**2


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
            WFS_noise_var = torch.abs( pdims(self.dn,2) + pdims(self.NoiseVariance(r0.abs()),2) ) # [rad^2] at WFSing wavelength TODO: really?
            self.vx = pdims( min_2d(self.wind_speed) * torch.cos(min_2d(self.wind_dir) * np.pi/180.0), 2 )
            self.vy = pdims( min_2d(self.wind_speed) * torch.sin(min_2d(self.wind_dir) * np.pi/180.0), 2 )

            self.Controller()
            self.ReconstructionFilter(r0.abs(), L0.abs(), WFS_noise_var)

        # All PSD components are computed in radians and here normalized to [nm^2]
        dk = 2*self.kc/self.nOtf_AO
        PSD_norm = lambda wvl: (dk*wvl*1e9/2/np.pi)**2

        # Most contributors are computed for 500 [nm], wavelengths-dependant inputs as well are assumed for 500 [nm]
        # But WFS operates on another wavelength, so this PSD components is normalized for WFSing wvl 
        # Put all contributiors together and sum up the resulting PSD
        self.PSDs = {entry: self.make_tensor(0.0) for entry in self.PSD_entries}

        if self.PSD_include['fitting']:
            self.PSDs['fitting'] = self.VonKarmanPSD(r0.abs(), L0.abs()).unsqueeze(1) * PSD_norm(500e-9)
    
        if self.PSD_include['WFS noise']:
            self.PSDs['WFS noise'] = self.NoisePSD(WFS_noise_var).unsqueeze(1) * PSD_norm(500e-9) #TODO: to what to normalize?
        
        if self.PSD_include['spatio-temporal']:
            self.PSDs['spatio-temporal'] = self.SpatioTemporalPSD().unsqueeze(1) * PSD_norm(500e-9)
        
        if self.PSD_include['aliasing']:
            self.PSDs['aliasing'] = self.AliasingPSD(r0.abs(), L0).unsqueeze(1) * PSD_norm(500e-9)
        
        if self.PSD_include['chromatism']:
            self.PSDs['chromatism'] = self.ChromatismPSD(r0.abs(), L0.abs()) * PSD_norm(500e-9) # no need to add dimension since it's polychromatic already

        if self.PSD_include['Moffat']:
            self.PSDs['Moffat'] = self.MoffatPSD(amp, b.abs(), alpha, beta, ratio, theta).unsqueeze(1) * PSD_norm(500e-9)

        #TODO: anisoplanatism!
        #TODO: SLAO support!
        #TODO: differential refraction!
        
        PSD = self.PSDs['fitting'] + self.PSD_padder(
              self.PSDs['WFS noise'] + \
              self.PSDs['spatio-temporal'] + \
              self.PSDs['aliasing'] + \
              self.PSDs['chromatism'] + \
              self.PSDs['Moffat'])
        
        self.PSD = PSD
        
        return PSD
    

    def OTF2PSF(self, OTF):
        s = tuple([OTF.shape[-2] + 1-self.nPix%2]*2)
        PSF = fft.fftshift(fft.ifft2(fft.ifftshift(OTF*self.center_aligner, dim=(-2,-1)), s=s), dim=(-2,-1)).abs()
        return interpolate(PSF, size=(self.nPix, self.nPix), mode='bilinear') if OTF.shape[-1] != self.nPix else PSF           


    def PSD2PSF(self, PSD):
        F   = pdims(self.F,   2)
        dx  = pdims(self.dx,  2)
        dy  = pdims(self.dy,  2)
        bg  = pdims(self.bg,  2)
        Jx  = pdims(self.Jx,  2)
        Jy  = pdims(self.Jy,  2)
        Jxy = pdims(self.Jxy, 2)

        # Removing the DC component
        PSD[..., self.nOtf_AO//2, self.nOtf_AO//2] = 0.0
        # Computing OTF from PSD
        cov = fft.fftshift(fft.fft2(fft.fftshift(PSD, dim=(-2,-1))), dim=(-2,-1)) # FFT axes are -2,-1 #TODO: real FFT?
        # Computing the Structure Function from the covariance
        SF = 2*(cov.abs().amax(dim=(-2,-1), keepdim=True) - cov.abs())
        # Phasor to shift the PSF with the subpixel accuracy
        fftPhasor = torch.exp( -np.pi*1j * pdims(self.sampling_factor,2) * (self.U*dx + self.V*dy).unsqueeze(1) )
        OTF_turb  = torch.exp( -0.5 * SF * pdims(2*np.pi*1e-9/self.wvl,2)**2 )
        # Compute the residual tip/tilt kernel
        OTF_jitter = self.JitterCore(Jx.abs(), Jy.abs(), Jxy.abs())
        # Resulting OTF
        OTF = OTF_turb * self.OTF_static * fftPhasor * OTF_jitter

        self.OTF = OTF

        PSF_out = self.OTF2PSF(OTF)

        self.norm_scale = self.normalizer(PSF_out, dim=(-2,-1), keepdim=True)

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


    def forward(self, x=None, PSD=None):
        if x is not None:
            for name, value in x.items():
                if isinstance(getattr(self, name), nn.Parameter):
                    setattr(self, name, nn.Parameter(value))
                else:
                    setattr(self, name, value)
        return self.PSD2PSF(self.ComputePSD() if PSD is None else PSD)


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
