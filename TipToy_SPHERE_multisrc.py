#%%
import torch
from torch import nn, optim, fft
from torch.nn import ParameterDict, ParameterList, Parameter
from torch.nn.functional import interpolate

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as spc
from astropy.io import fits
import pickle
import time
import re
import os
from os import path
from copy import deepcopy

from parameterParser import parameterParser
from SPHERE_data import SPHERE_database,LoadSPHEREsampleByID
from utils import rad2mas, rad2arc, deg2rad, asec2rad, seeing, r0, r0_new
from utils import Center, BackgroundEstimate, CircularMask
from utils import register_hooks, iter_graph
from utils import OptimizeTRF, OptimizeLBFGS
from utils import radial_profile, plot_radial_profile
# end of import list

#%%
data_samples = []
data_samples.append( LoadSPHEREsampleByID('C:/Users/akuznets/Data/SPHERE/test/', 347)[1] )
data_samples.append( LoadSPHEREsampleByID('C:/Users/akuznets/Data/SPHERE/test/', 351)[1] )
#data_samples.append( LoadSPHEREsampleByID('C:/Users/akuznets/Data/SPHERE/test/', 340)[1] )

path_root = path.normpath('C:/Users/akuznets/Projects/TIPTOP_old/P3')
path_ini = path.join(path_root, path.normpath('aoSystem/parFiles/irdis.ini'))

config_file = parameterParser(path_root, path_ini).params
config_file['atmosphere']['Cn2Weights'] = [0.95, 0.05]
config_file['atmosphere']['Cn2Heights'] = [0, 10000]

def show_row(row, log=True):
    if log: buf = torch.log( torch.hstack(row) )
    else: buf = torch.hstack(row)
    plt.imshow(buf.abs().detach().cpu())
    plt.show()

#%% ------------------------ Managing paths ------------------------
class TipToy(torch.nn.Module):
    def SetDataSample(self, data_samples):
        # Reading parameters from the file
        num_src = len(data_samples)
        self.Nsrc = num_src
        self.wvl = np.array([data_sample['spectrum']['lambda'] for data_sample in data_samples])
        if not np.all(self.wvl  == self.wvl[0]):
            raise ValueError('All wavelength must be the same for now')
        self.wvl = self.wvl[0]
        
        self.configs = deepcopy(self.AO_config)
        self.configs['sources_science']['Wavelength'] = self.wvl
        self.configs['sensor_science']['FieldOfView'] = data_samples[0]['image'].shape[0] #TODO: image size check
        self.configs['atmosphere']['Seeing']        = torch.tensor([data_sample['seeing']['SPARTA'] for data_sample in data_samples], device=self.device)
        self.configs['atmosphere']['WindSpeed']     = torch.tensor([data_sample['Wind speed']['header'] for data_sample in data_samples], device=self.device)
        self.configs['atmosphere']['WindDirection'] = torch.tensor([data_sample['Wind direction']['header'] for data_sample in data_samples], device=self.device)
        self.configs['sensor_science']['Zenith']    = torch.tensor([90.0-data_sample['telescope']['altitude'] for data_sample in data_samples], device=self.device) # TODO: aren't they the same?
        self.configs['telescope']['ZenithAngle']    = torch.tensor([90.0-data_sample['telescope']['altitude'] for data_sample in data_samples], device=self.device) # TODO: aren't they the same?
        self.configs['sensor_science']['Azimuth']   = torch.tensor([data_sample['telescope']['azimuth'] for data_sample in data_samples], device=self.device)
        self.configs['sensor_science']['SigmaRON']  = torch.tensor([data_sample['Detector']['ron']  for data_sample in data_samples], device=self.device)
        self.configs['sensor_science']['Gain']      = torch.tensor([data_sample['Detector']['gain'] for data_sample in data_samples], device=self.device)
        self.configs['sensor_HO']['ClockRate']      = torch.tensor([self.AO_config['sensor_HO']['ClockRate']  for _ in range(num_src)], device=self.device)
        self.configs['sensor_HO']['NumberPhotons']  = torch.tensor([data_sample['WFS']['Nph vis'] for data_sample in data_samples], device=self.device)
        self.configs['sources_HO']['Wavelength']    = torch.tensor([self.AO_config['sources_HO']['Wavelength'][0] for _ in range(num_src)], device=self.device)
        self.configs['sources_HO']['Height']        = torch.tensor([self.AO_config['sources_HO']['Height']  for _ in range(num_src)], device=self.device)
        self.configs['sources_HO']['Zenith']        = torch.tensor([self.AO_config['sources_HO']['Zenith']  for _ in range(num_src)], device=self.device)
        self.configs['sources_HO']['Azimuth']       = torch.tensor([self.AO_config['sources_HO']['Azimuth'] for _ in range(num_src)], device=self.device)
        self.configs['atmosphere']['Cn2Weights']    = torch.tensor([self.AO_config['atmosphere']['Cn2Weights'] for _ in range(num_src)], device=self.device)
        self.configs['atmosphere']['Cn2Heights']    = torch.tensor([self.AO_config['atmosphere']['Cn2Heights'] for _ in range(num_src)], device=self.device)
        self.configs['RTC']['LoopDelaySteps_HO']    = torch.tensor([self.AO_config['RTC']['LoopDelaySteps_HO'] for _ in range(num_src)], device=self.device)
        self.configs['RTC']['LoopGain_HO']          = torch.tensor([self.AO_config['RTC']['LoopGain_HO']   for _ in range(num_src)], device=self.device)
        self.configs['RTC']['SensorFrameRate_HO']   = torch.tensor([data_sample['WFS']['rate'] for data_sample in data_samples], device=self.device)

        # Setting internal parameters
        self.D       = self.AO_config['telescope']['TelescopeDiameter']
        self.psInMas = self.AO_config['sensor_science']['PixelScale'] #[mas]
        self.nPix    = self.AO_config['sensor_science']['FieldOfView']
        self.pitch   = self.AO_config['DM']['DmPitchs'][0] #[m]
        #self.h_DM    = self.AO_config['DM']['DmHeights'][0] # ????? what is h_DM?
        #self.nDM     = 1
        self.kc      = 1/(2*self.pitch) #TODO: kc is not consistent with vanilla TIPTOP

        #self.zenith_angle  = torch.tensor(self.AO_config['telescope']['ZenithAngle'], device=self.device) # [deg] #TODO: telescope zenith != sample zenith?
        self.zenith_angle  = self.configs['telescope']['ZenithAngle']
        self.airmass       = 1.0 / torch.cos(self.zenith_angle * deg2rad)

        self.GS_wvl     = self.AO_config['sources_HO']['Wavelength'][0] #[m]
        #self.GS_height  = self.AO_config['sources_HO']['Height'] * self.airmass #[m]

        self.wind_speed  = self.configs['atmosphere']['WindSpeed']
        self.wind_dir    = self.configs['atmosphere']['WindDirection']
        self.Cn2_weights = self.configs['atmosphere']['Cn2Weights']
        self.Cn2_heights = self.configs['atmosphere']['Cn2Heights'] * self.airmass.unsqueeze(1) #[m]
        #self.stretch     = 1.0 / (1.0-self.Cn2_heights/self.GS_height)
        self.h           = self.Cn2_heights #* self.stretch
        self.nL          = self.Cn2_heights.size(0)

        self.WFS_d_sub = np.mean(self.AO_config['sensor_HO']['SizeLenslets'])
        self.WFS_n_sub = np.mean(self.AO_config['sensor_HO']['NumberLenslets'])

        self.WFS_det_clock_rate = self.configs['sensor_HO']['ClockRate'].flatten() # [(?)]
        self.WFS_FOV = self.configs['sensor_HO']['FieldOfView']
        self.WFS_RON = self.configs['sensor_HO']['SigmaRON']
        self.WFS_psInMas = self.configs['sensor_HO']['PixelScale']
        self.WFS_wvl = torch.tensor(self.GS_wvl, device=self.device) #TODO: clarify this
        self.WFS_spot_FWHM = torch.tensor(self.configs['sensor_HO']['SpotFWHM'][0], device=self.device)
        self.WFS_excessive_factor = self.configs['sensor_HO']['ExcessNoiseFactor']
        self.WFS_Nph = self.configs['sensor_HO']['NumberPhotons']

        self.HOloop_rate  = self.configs['RTC']['SensorFrameRate_HO'] # [Hz] (?)
        self.HOloop_delay = self.configs['RTC']['LoopDelaySteps_HO'] # [ms] (?)
        self.HOloop_gain  = self.configs['RTC']['LoopGain_HO']


    def InitGrids(self):

        self.add2d = lambda x: x.unsqueeze(1).unsqueeze(2)
        self.add1d = lambda x: x.unsqueeze(0)

        if self.pixels_per_l_D is None:
            self.pixels_per_l_D = self.wvl*rad2mas / (self.psInMas*self.D)

        self.sampling_factor = int(np.ceil(2.0/self.pixels_per_l_D)) # check how much it is less than Nyquist
        self.sampling = self.sampling_factor * self.pixels_per_l_D
        self.nOtf = self.nPix * self.sampling_factor

        self.dk = 1/self.D/self.sampling # PSD spatial frequency step
        self.cte = (24*spc.gamma(6/5)/5)**(5/6)*(spc.gamma(11/6)**2/(2*np.pi**(11/3)))

        # Initialize spatial frequencies
        self.kx, self.ky = torch.meshgrid(
            torch.linspace(-self.nOtf/2, self.nOtf/2-1, self.nOtf, device=self.device)*self.dk + 1e-10,
            torch.linspace(-self.nOtf/2, self.nOtf/2-1, self.nOtf, device=self.device)*self.dk + 1e-10,
            indexing = 'ij')

        self.k2 = self.kx**2 + self.ky**2
        self.k = torch.sqrt(self.k2)

        self.mask = torch.ones_like(self.k2, device=self.device)
        self.mask[self.k2 <= self.kc**2] = 0
        self.mask_corrected = 1.0-self.mask

        self.nOtf_AO = int(2*self.kc/self.dk)
        self.nOtf_AO += self.nOtf_AO % 2

        # Comb samples involved in antialising
        n_times = min(4,max(2,int(np.ceil(self.nOtf/self.nOtf_AO/2))))
        ids = []
        for mi in range(-n_times,n_times):
            for ni in range(-n_times,n_times):
                if mi or ni: #exclude (0,0)
                    ids.append([mi,ni])
        ids = np.array(ids)

        m = torch.tensor(ids[:,0], device=self.device)
        n = torch.tensor(ids[:,1], device=self.device)
        self.N_combs = m.shape[0]

        corrected_ROI = slice(self.nOtf//2-self.nOtf_AO//2, self.nOtf//2+self.nOtf_AO//2)
        corrected_ROI = (corrected_ROI,corrected_ROI)

        self.mask_AO = self.mask[corrected_ROI]
        self.mask_corrected_AO = self.add1d( self.mask_corrected[corrected_ROI] )
        self.mask = self.add1d( self.mask )

        self.kx_AO = self.add1d( self.kx[corrected_ROI] )
        self.ky_AO = self.add1d( self.ky[corrected_ROI] )
        self.k_AO  = self.add1d( self.k [corrected_ROI] )
        self.k2_AO = self.add1d( self.k2[corrected_ROI] )

        self.kx = self.add1d( self.kx )
        self.ky = self.add1d( self.ky )
        self.k  = self.add1d( self.k  )
        self.k2 = self.add1d( self.k2 )

        # For NGS-like alising 0th dimension is used to store shifted spatial frequency
        self.km = self.kx_AO.repeat([self.N_combs,1,1,1]) - self.add2d(m/self.WFS_d_sub).unsqueeze(3)
        self.kn = self.ky_AO.repeat([self.N_combs,1,1,1]) - self.add2d(n/self.WFS_d_sub).unsqueeze(3)
        
        # Initialize OTF frequencines
        self.U, self.V = torch.meshgrid(
            torch.linspace(0, self.nOtf-1, self.nOtf, device=self.device),
            torch.linspace(0, self.nOtf-1, self.nOtf, device=self.device),
            indexing = 'ij')

        self.U = self.add1d( (self.U-self.nOtf/2) * 2/self.nOtf )
        self.V = self.add1d( (self.V-self.nOtf/2) * 2/self.nOtf )

        self.U2  = self.U**2
        self.V2  = self.V**2
        self.UV  = self.U*self.V
        self.UV2 = self.U**2 + self.V**2

        pupil_path = self.AO_config['telescope']['PathPupil']
        pupil_apodizer = self.AO_config['telescope']['PathApodizer']

        pupil    = torch.tensor(fits.getdata(pupil_path).astype('float'), device=self.device)
        apodizer = torch.tensor(fits.getdata(pupil_apodizer).astype('float'), device=self.device)

        pupil_pix  = pupil.shape[0]
        #padded_pix = nOtf
        padded_pix = int(pupil_pix*self.sampling)

        pupil_padded = torch.zeros([padded_pix, padded_pix], device=self.device)
        pupil_padded[
            padded_pix//2-pupil_pix//2 : padded_pix//2+pupil_pix//2,
            padded_pix//2-pupil_pix//2 : padded_pix//2+pupil_pix//2
        ] = pupil*apodizer

        def fftAutoCorr(x):
            x_fft = fft.fft2(x)
            return fft.fftshift( fft.ifft2(x_fft*torch.conj(x_fft))/x.size(0)*x.size(1) )

        self.OTF_static = torch.real( fftAutoCorr(pupil_padded) ).unsqueeze(0).unsqueeze(0)
        self.OTF_static = interpolate(self.OTF_static, size=(self.nOtf,self.nOtf), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
        self.OTF_static = self.add1d( self.OTF_static / self.OTF_static.max() )

        self.PSD_padder = torch.nn.ZeroPad2d((self.nOtf-self.nOtf_AO)//2)

        # Piston filter
        def PistonFilter(f):
            x = (np.pi*self.D*f).cpu().numpy() #TODO: find Bessel analog for pytorch
            R = spc.j1(x)/x
            piston_filter = torch.tensor(1.0-4*R**2, device=self.device)
            piston_filter[..., self.nOtf_AO//2, self.nOtf_AO//2] *= 0.0
            return piston_filter

        self.piston_filter = PistonFilter(self.k_AO)
        self.PR = PistonFilter(torch.hypot(self.km,self.kn))


    def Update(self, data_sample, reinit_grids=True):
        self.SetDataSample(data_sample)
        if reinit_grids: self.InitGrids()


    def __init__(self, AO_config, data_sample, device=None, pixels_per_l_D=None):
        if device is None or device == 'cpu' or device == 'CPU':
            self.device = torch.device('cpu')
            self.is_gpu = False

        elif device == 'cuda' or device == 'CUDA':
            self.device  = torch.device('cuda') # Will use the default CUDA device
            self.start = torch.cuda.Event(enable_timing=True)
            self.end   = torch.cuda.Event(enable_timing=True)
            self.is_gpu = True

        super().__init__()

        self.norm_regime = 'sum'

        # Read data and initialize AO system
        self.pixels_per_l_D = pixels_per_l_D
        self.AO_config = AO_config
        self.SetDataSample(data_sample)
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
            rtfInt = 1.0/(1 + hInt*z**(-delay))
            atfInt = hInt * z**(-delay)*rtfInt
            ntfInt = atfInt / z
            return hInt, rtfInt, atfInt, ntfInt

        #f = torch.logspace(-3, torch.log10(torch.tensor([0.5/Ts])).item(), nF)
        f = torch.zeros([self.Nsrc,nF], device=self.device)
        for i in range(self.Nsrc): 
            f[i,:] = torch.logspace(-3, torch.log10(0.5/Ts[i]), nF)

        _, _, _, ntfInt = TransferFunctions(f, Ts.unsqueeze(1), delay.unsqueeze(1), loopGain.unsqueeze(1))
        self.noise_gain = self.add2d( torch.trapz(torch.abs(ntfInt)**2, f, dim=1)*2*Ts )

        thetaWind = torch.tensor(0.0) #torch.linspace(0, 2*np.pi-2*np.pi/nTh, nTh)
        costh = torch.cos(thetaWind) #TODO: what is thetaWind?
        
        fi = -self.vx*self.kx_AO*costh - self.vy*self.ky_AO*costh
        _, _, atfInt, ntfInt = TransferFunctions(fi, self.add2d(Ts),
                                                     self.add2d(delay),
                                                     self.add2d(loopGain))

        # AO transfer function
        self.h1 = self.Cn2_weights.T.unsqueeze(2).unsqueeze(3) * atfInt.unsqueeze(0) #/nTh
        self.h2 = self.Cn2_weights.T.unsqueeze(2).unsqueeze(3) * abs(atfInt.unsqueeze(0))**2 #/nTh
        self.hn = self.Cn2_weights.T.unsqueeze(2).unsqueeze(3) * abs(ntfInt.unsqueeze(0))**2 #/nTh

        self.h1 = torch.sum(self.h1, axis=0) #summing along the weights dimension (4D->3D)
        self.h2 = torch.sum(self.h2, axis=0) 
        self.hn = torch.sum(self.hn, axis=0) 


    def ReconstructionFilter(self, r0, L0, WFS_noise_var):
        Av = torch.sinc(self.WFS_d_sub*self.kx_AO)*torch.sinc(self.WFS_d_sub*self.ky_AO) * torch.exp(1j*np.pi*self.WFS_d_sub*(self.kx_AO+self.ky_AO))
        self.SxAv = ( 2j*np.pi*self.kx_AO*self.WFS_d_sub*Av ).repeat([self.Nsrc,1,1])
        self.SyAv = ( 2j*np.pi*self.ky_AO*self.WFS_d_sub*Av ).repeat([self.Nsrc,1,1])

        MV = 0
        Wn = WFS_noise_var / (2*self.kc)**2
        # TODO: isn't this one computed for the WFSing wvl?
        #self.W_atm = self.cte*r0.**(-5/3) * (self.k2_AO + 1/L0**2)**(-11/6) * \
        #    (self.wvl/self.GS_wvl)**2 #TODO: check for SPHERE

        self.W_atm = self.VonKarmanSpectrum(r0, L0, self.k2_AO) * (self.wvl/self.GS_wvl)**2
        
        gPSD = torch.abs(self.SxAv)**2 + torch.abs(self.SyAv)**2 + MV*Wn/self.W_atm
        self.Rx = torch.conj(self.SxAv) / gPSD
        self.Ry = torch.conj(self.SyAv) / gPSD
        self.Rx[..., self.nOtf_AO//2, self.nOtf_AO//2] *= 0
        self.Ry[..., self.nOtf_AO//2, self.nOtf_AO//2] *= 0

    
    def SpatioTemporalPSD(self):
        A = torch.ones([self.W_atm.shape[0], self.nOtf_AO, self.nOtf_AO], device=self.device) #TODO: fix it. A should be initialized differently
        Ff = self.Rx*self.SxAv + self.Ry*self.SyAv
        psd_ST = (1+abs(Ff)**2 * self.h2 - 2*torch.real(Ff*self.h1*A)) * self.W_atm * self.mask_corrected_AO
        return psd_ST

    def NoisePSD(self, WFS_noise_var):
        noisePSD = abs(self.Rx**2 + self.Ry**2) / (2*self.kc)**2
        noisePSD = noisePSD * self.piston_filter * self.noise_gain * WFS_noise_var * self.mask_corrected_AO
        return noisePSD


    def AliasingPSD(self, r0, L0):
        T = self.add2d(self.WFS_det_clock_rate / self.HOloop_rate)
        td = T * self.add2d(self.HOloop_delay)

        Rx1 = self.add1d(2j*np.pi*self.WFS_d_sub * self.Rx)
        Ry1 = self.add1d(2j*np.pi*self.WFS_d_sub * self.Ry)

        W_mn = (self.km**2 + self.kn**2 + 1/self.add1d(L0)**2)**(-11/6)
        Q = (Rx1*self.km + Ry1*self.kn) * torch.sinc(self.WFS_d_sub*self.km) * torch.sinc(self.WFS_d_sub*self.kn)

        tf = self.add1d(self.h1)

        avr = ( (self.Cn2_weights.T).unsqueeze(1).unsqueeze(3).unsqueeze(4) * \
            self.add1d( torch.sinc(self.km*self.vx*T) * torch.sinc(self.kn*self.vy*T) * \
            torch.exp(2j*np.pi*self.km*self.vx*td) * \
            torch.exp(2j*np.pi*self.kn*self.vy*td) * tf) ).sum(axis=0)

        aliasing_PSD = torch.sum(self.PR*W_mn*abs(Q*avr)**2, axis=0) * self.cte*r0**(-5/3) * self.mask_corrected_AO
        return aliasing_PSD


    def VonKarmanSpectrum(self, r0, L0, freq2):
        return self.cte*r0**(-5/3) * (freq2 + 1/L0**2)**(-11/6)

    def VonKarmanPSD(self, r0, L0):
        return self.VonKarmanSpectrum(r0, L0, self.k2) * self.mask


    def ChromatismPSD(self, r0, L0):
        wvlRef = self.wvl #TODO: polychromatic support
        W_atm = self.VonKarmanSpectrum(r0, L0, self.k2_AO) * self.piston_filter
        IOR = lambda lmbd: 23.7+6839.4/(130-(lmbd*1.e6)**(-2))+45.47/(38.9-(lmbd*1.e6)**(-2))
        n2 = IOR(self.GS_wvl)
        n1 = IOR(wvlRef)
        chromatic_PSD = ((n2-n1)/n2)**2 * W_atm
        return chromatic_PSD

    def JitterCore(self, Jx, Jy, Jxy):
        u_max = self.sampling*self.D/self.wvl/(3600*180*1e3/np.pi)
        norm_fact = u_max**2 * (2*np.sqrt(2*np.log(2)))**2
        Djitter = norm_fact * (Jx**2 * self.U2 + Jy**2 * self.V2 + 2*Jxy*self.UV)
        return torch.exp(-0.5*Djitter) #TODO: cover Nyquist sampled case


    def NoiseVariance(self, r0):
        r0_WFS = r0_new(r0.abs(), self.GS_wvl, 0.5e-6).unsqueeze(1).unsqueeze(2) #from (Nsrc x 1 x 1) to (Nsrc)
        WFS_nPix = self.WFS_FOV / self.WFS_n_sub
        WFS_pixelScale = self.WFS_psInMas / 1e3 # [arcsec]
        # Read-out noise calculation
        nD = torch.tensor([1.0, rad2arc*self.wvl/self.WFS_d_sub/WFS_pixelScale]).max() #spot FWHM in pixels and without turbulence
        varRON = np.pi**2/3 * (self.WFS_RON**2/self.WFS_Nph**2) * (WFS_nPix**2/nD)**2
        # Photon-noise calculation
        nT = torch.tensor(
            [[1.0]*self.Nsrc,
            torch.hypot(self.WFS_spot_FWHM.max()/1e3, rad2arc*self.WFS_wvl/r0_WFS) / WFS_pixelScale], 
            device=self.device).amax(dim=0)
        varShot = np.pi**2/(2*self.WFS_Nph) * (nT/nD)**2
        # Noise variance calculation
        varNoise = self.WFS_excessive_factor * (varRON+varShot)
        return self.add2d( varNoise )

    '''
    def DLPSF(self):
        PSF = torch.abs( fft.fftshift(fft.ifft2(fft.fftshift(self.OTF_static))) ).unsqueeze(0).unsqueeze(0)
        PSF_out = interpolate(PSF, size=(self.nPix,self.nPix), mode='area').squeeze(0).squeeze(0)
        return (PSF_out/PSF_out.sum())
    '''


    def PSD2PSF(self, r0, L0, F, dx, dy, bg, dn, Jx, Jy, Jxy):
        r0  = self.add2d(r0)
        L0  = self.add2d(L0)
        F   = self.add2d(F)
        dx  = self.add2d(dx)
        dy  = self.add2d(dy)
        bg  = self.add2d(bg)
        dn  = self.add2d(dn)
        Jx  = self.add2d(Jx)
        Jy  = self.add2d(Jy)
        Jxy = self.add2d(Jxy)

        WFS_noise_var = torch.abs(dn + self.NoiseVariance(r0))
        self.vx = self.add2d(self.wind_speed*torch.cos(self.wind_dir*np.pi/180))
        self.vy = self.add2d(self.wind_speed*torch.sin(self.wind_dir*np.pi/180))

        self.Controller()
        self.ReconstructionFilter(r0.abs(), L0.abs(), WFS_noise_var)

        dk = 2*self.kc/self.nOtf_AO

        # All PSD components are normalized to [nm] assuming that all cpntributors
        # are computed for 500 [nm]
        PSD_norm = (dk*500/2/np.pi)**2

        # WFS operates on another wavelength -> this PSD components is normalized for WFSing wvl 
        noise_PSD_norm = (dk*self.GS_wvl*1e9/2/np.pi)**2

        # Put all contributiors together and sum up the resulting PSD
        '''
        self.PSDs = {
            'fitting':         self.VonKarmanPSD(r0.abs(),L0.abs()) * PSD_norm,
            'WFS noise':       self.NoisePSD(WFS_noise_var) * noise_PSD_norm,
            'spatio-temporal': self.SpatioTemporalPSD() * PSD_norm,
            'aliasing':        self.AliasingPSD(r0.abs(), L0) * PSD_norm,
            'chromatism':      self.ChromatismPSD(r0.abs(), L0.abs()) * PSD_norm
        }

        PSD = self.PSDs['fitting'] + self.PSD_padder(
            self.PSDs['WFS noise'] + \
            self.PSDs['spatio-temporal'] + \
            self.PSDs['aliasing'] + \
            self.PSDs['chromatism']
        )
        '''

        PSD = self.VonKarmanPSD(r0.abs(), L0.abs()) * PSD_norm + \
            self.PSD_padder(
                self.NoisePSD(WFS_noise_var) * noise_PSD_norm + \
                self.ChromatismPSD(r0.abs(), L0.abs()) * PSD_norm + \
                self.SpatioTemporalPSD() * PSD_norm + \
                self.AliasingPSD(r0.abs(), L0) * PSD_norm
            )
        #TODO: smth wrong with the noise PSD

        # Computing OTF from PSD
        cov = 2*fft.fftshift(fft.fft2(fft.fftshift(PSD))) # FFT axes are set to 1,2 by PyTorch by default
        # Computing the Structure Function from the covariance
        SF = torch.abs(cov).max()-cov
        # Phasor to shift the PSF with the subpixel accuracy
        fftPhasor  = torch.exp(-np.pi*1j*self.sampling_factor * (self.U*dx + self.V*dy))
        OTF_turb   = torch.exp(-0.5*SF*(2*np.pi*1e-9/self.wvl)**2)
        OTF_jitter = self.JitterCore(Jx.abs(), Jy.abs(), Jxy.abs())
        OTF = OTF_turb * self.OTF_static * fftPhasor * OTF_jitter

        #plt.imshow(OTF[0,:,:].abs().detach().cpu())
        #plt.show()
        #print(OTF.sum())

        PSF = torch.abs( fft.fftshift(fft.ifft2(fft.fftshift(OTF))) ).unsqueeze(0)
        PSF_out = interpolate(PSF, size=(self.nPix,self.nPix), mode='area').squeeze(0)

        if self.norm_regime == 'max':
            return PSF_out/torch.amax(PSF_out, dim=(1,2), keepdim=True) * F + bg
        elif self.norm_regime == 'sum':
            return PSF_out/PSF_out.sum(dim=(1,2), keepdim=True) * F + bg
        else:
            return PSF_out * F + bg


    def forward(self, x):
        inp = [x[:,i] for i in range(x.shape[1])]
        return self.PSD2PSF(*inp)


    def StartTimer(self):
        if self.is_gpu:
            self.start.record()
        else:
            self.start = time.time()


    def EndTimer(self):
        if self.is_gpu:
            self.end.record()
            torch.cuda.synchronize()
            return self.start.elapsed_time(self.end)
        else:
            self.end = time.time()
            return (self.end-self.start)*1000.0 # in [ms]

# end of class defenition

#%% -------------------------------------------------------------
toy = TipToy(config_file, data_samples, 'CUDA')
toy.norm_regime = 'max'

ims = []
for sample in data_samples:
    im = sample['image']
    if    toy.norm_regime == 'max': param = im.max()
    elif  toy.norm_regime == 'sum': param = im.sum()
    else: param = 1.0
    ims.append(torch.tensor(im/param, device=toy.device))


PSF_0 = torch.tensor(torch.dstack(ims)).permute([2,0,1])

N_src = len(data_samples)

r0  = torch.tensor([sample['r0'] for sample in data_samples], requires_grad=True, device=toy.device).flatten()
L0  = torch.tensor([25.0]*N_src, requires_grad=False, device=toy.device).flatten()
F   = torch.tensor([1.0 ]*N_src, requires_grad=True,  device=toy.device).flatten()
dx  = torch.tensor([0.0 ]*N_src, requires_grad=True,  device=toy.device).flatten()
dy  = torch.tensor([0.0 ]*N_src, requires_grad=True,  device=toy.device).flatten()
bg  = torch.tensor([0.0 ]*N_src, requires_grad=True,  device=toy.device).flatten()
dn  = torch.tensor([0.05]*N_src, requires_grad=True,  device=toy.device).flatten()
Jx  = torch.tensor([10.0]*N_src, requires_grad=True,  device=toy.device).flatten()
Jy  = torch.tensor([10.0]*N_src, requires_grad=True,  device=toy.device).flatten()
Jxy = torch.tensor([2.0 ]*N_src, requires_grad=True,  device=toy.device).flatten()


parameters = [r0, L0, F, dx, dy, bg, dn, Jx, Jy, Jxy]
x = torch.stack(parameters).T.unsqueeze(0)

#toy.StartTimer()
PSF_1 = toy.PSD2PSF(*parameters)
#print(toy.EndTimer())
#PSF_DL = toy.DLPSF()

def draw_result(PSF_in, PSF_out):
    for i in range(PSF_out.shape[0]):
        PSF_1buf = PSF_out[i,:,:]
        PSF_0buf = PSF_in[i,:,:]

        el_croppo = slice(PSF_in.shape[1]//2-32, PSF_in.shape[2]//2+32)
        el_croppo = (el_croppo, el_croppo)

        plt.imshow(torch.log(torch.hstack((
            PSF_0buf.abs()[el_croppo], PSF_1buf.abs()[el_croppo], \
            ((PSF_1buf-PSF_0buf).abs()[el_croppo])
        ) )).detach().cpu())
        plt.show()

draw_result(PSF_0, PSF_1)

for i in range(PSF_0.shape[0]):
    plot_radial_profile(PSF_0[i,:,:], PSF_1[i,:,:], 'TipToy', title='IRDIS PSF', dpi=100)

#%%
loss_fn = nn.L1Loss(reduction='sum')

# Confines a value between 0 and the specified value
#window_loss = lambda x, x_max: \
#    (x>0).float()*(0.01/x)**2 + \
#    (x<0).float()*100 + \
#    100*(x>x_max).float()*(x-x_max)**2
#
## TODO: specify loss weights
#def loss_fn(a,b):
#    z = loss(a,b) + \
#        window_loss(r0, 0.5) * 5.0 + \
#        window_loss(Jx, 50) + \
#        window_loss(Jy, 50) + \
#        window_loss(Jxy, 400) + \
#        window_loss(dn+toy.NoiseVariance(r0), 1.0)
#    return z

optimizer_lbfgs = OptimizeLBFGS(toy, parameters, loss_fn)

for i in range(20):
    optimizer_lbfgs.Optimize(PSF_0, [F, dx, dy, r0, dn], 5)
    optimizer_lbfgs.Optimize(PSF_0, [bg], 2)
    optimizer_lbfgs.Optimize(PSF_0, [Jx, Jy, Jxy], 3)

PSF_1 = toy.PSD2PSF(*parameters)
#SR = lambda PSF: (PSF.max()/PSF_DL.max() * PSF_DL.sum()/PSF.sum()).item()

draw_result(PSF_0, PSF_1)

for i in range(PSF_0.shape[0]):
    plot_radial_profile(PSF_0[i,:,:], PSF_1[i,:,:], 'TipToy', title='IRDIS PSF', dpi=100)

#%%
n_result = (dn + toy.NoiseVariance(r0)).abs().item()
n_init = toy.NoiseVariance(torch.tensor(data_test['r0'], device=toy.device)).item()

print("".join(['_']*52))
print('MAE + window value:', loss_fn(PSF_1, PSF_0).item())
print("r0,r0': ({:.3f}, {:.2f})".format(data_test['r0'], r0.item()))
print("F,bg:  ({:.3f}, {:.1E} )".format(F.data.item(), bg.item()))
print("dx,dy: ({:.2f}, {:.2f})".format(dx.data.item(), dy.item()))
print("Jx,Jy, Jxy: ({:.1f}, {:.1f}, {:.1f})".format(Jx.item(), Jy.item(), Jxy.item()))
print("n, n': ({:.2f},{:.2f})".format(n_init, n_result))

plot_radial_profile(PSF_0, PSF_1, 'TipToy', title='IRDIS PSF', dpi=100)
plt.show()
#la chignon et tarte
