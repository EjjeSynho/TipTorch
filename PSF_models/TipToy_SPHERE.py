#%%
import torch
from torch import nn, optim, fft
from torch.nn import ParameterDict, ParameterList, Parameter
from torch.nn.functional import interpolate

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as spc
from scipy import signal
from astropy.io import fits
import pickle
import time
import os
from os import path
import re

from tools.parameterParser import parameterParser
from SPHERE_data import SPHERE_database,LoadSPHEREsampleByID
from utils import rad2mas, rad2arc, deg2rad, asec2rad, seeing, r0, r0_new
from utils import Center, BackgroundEstimate, CircularMask
from utils import register_hooks, iter_graph
from utils import OptimizeTRF, OptimizeLBFGS
from utils import radial_profile, plot_radial_profile
# end of import list


#210
#13
#190
#26
#422
#102

#260 #450
#num = 351
num = 347

num_id, data_test = LoadSPHEREsampleByID('C:/Users/akuznets/Data/SPHERE/test/', num)

im = data_test['image']
path_root = path.normpath('C:/Users/akuznets/Projects/TIPTOP_old/P3')
path_ini = path.join(path_root, path.normpath('aoSystem/parFiles/irdis.ini'))

config_file = parameterParser(path_root, path_ini).params
#config_file['atmosphere']['Cn2Weights'] = [0.95, 0.05]
#config_file['atmosphere']['Cn2Heights'] = [0, 10000]

#%% ------------------------ Managing paths ------------------------
class TipToy(torch.nn.Module):
    def SetDataSample(self, data_sample):
        # Reading parameters from the file
        self.wvl = data_sample['spectrum']['lambda']

        self.AO_config['atmosphere']['Seeing']          = data_sample['seeing']
        self.AO_config['atmosphere']['WindSpeed']       = [data_sample['Wind speed']['header']]
        self.AO_config['atmosphere']['WindDirection']   = [data_sample['Wind direction']['header']]
        self.AO_config['sources_science']['Wavelength'] = [self.wvl]
        self.AO_config['sensor_science']['FieldOfView'] = data_sample['image'].shape[0]
        self.AO_config['sensor_science']['Zenith']      = [90.0-data_sample['telescope']['altitude']]
        self.AO_config['sensor_science']['Azimuth']     = [data_sample['telescope']['azimuth']]
        self.AO_config['sensor_science']['SigmaRON']    = data_sample['Detector']['ron']
        self.AO_config['sensor_science']['Gain']        = data_sample['Detector']['gain']

        self.AO_config['RTC']['SensorFrameRate_HO']  = data_sample['WFS']['rate']
        self.AO_config['sensor_HO']['NumberPhotons'] = data_sample['WFS']['Nph vis']

        # Setting internal parameters
        self.D       = self.AO_config['telescope']['TelescopeDiameter']
        self.psInMas = self.AO_config['sensor_science']['PixelScale'] #[mas]
        self.nPix    = self.AO_config['sensor_science']['FieldOfView']
        self.pitch   = self.AO_config['DM']['DmPitchs'][0] #[m]
        self.h_DM    = self.AO_config['DM']['DmHeights'][0] # ????? what is h_DM?
        self.nDM     = 1
        self.kc      = 1/(2*self.pitch) #TODO: kc is not consistent with vanilla TIPTOP

        self.zenith_angle  = torch.tensor(self.AO_config['telescope']['ZenithAngle'], device=self.device) # [deg]
        self.airmass       = 1.0 / torch.cos(self.zenith_angle * deg2rad)

        self.GS_wvl     = self.AO_config['sources_HO']['Wavelength'][0] #[m]
        self.GS_height  = self.AO_config['sources_HO']['Height'] * self.airmass #[m]
        self.GS_angle   = torch.tensor(self.AO_config['sources_HO']['Zenith'],  device=self.device) / rad2arc
        self.GS_azimuth = torch.tensor(self.AO_config['sources_HO']['Azimuth'], device=self.device) * deg2rad
        self.GS_dirs_x  = torch.tan(self.GS_angle) * torch.cos(self.GS_azimuth)
        self.GS_dirs_y  = torch.tan(self.GS_angle) * torch.sin(self.GS_azimuth)
        self.nGS = self.GS_dirs_y.size(0)

        self.wind_speed  = torch.tensor(self.AO_config['atmosphere']['WindSpeed'], device=self.device)
        self.wind_dir    = torch.tensor(self.AO_config['atmosphere']['WindDirection'], device=self.device)
        self.Cn2_weights = torch.tensor(self.AO_config['atmosphere']['Cn2Weights'], device=self.device)
        self.Cn2_heights = torch.tensor(self.AO_config['atmosphere']['Cn2Heights'], device=self.device) * self.airmass #[m]
        #self.stretch     = 1.0 / (1.0-self.Cn2_heights/self.GS_height)
        self.h           = self.Cn2_heights #* self.stretch
        self.nL          = self.Cn2_heights.size(0)

        self.WFS_d_sub = np.mean(self.AO_config['sensor_HO']['SizeLenslets'])
        self.WFS_n_sub = np.mean(self.AO_config['sensor_HO']['NumberLenslets'])
        self.WFS_det_clock_rate = np.mean(self.AO_config['sensor_HO']['ClockRate']) # [(?)]
        self.WFS_FOV = self.AO_config['sensor_HO']['FieldOfView']
        self.WFS_RON = self.AO_config['sensor_HO']['SigmaRON']
        self.WFS_psInMas = self.AO_config['sensor_HO']['PixelScale']
        self.WFS_wvl = torch.tensor(self.GS_wvl, device=self.device) #TODO: clarify this
        self.WFS_spot_FWHM = torch.tensor(self.AO_config['sensor_HO']['SpotFWHM'][0], device=self.device)
        self.WFS_excessive_factor = self.AO_config['sensor_HO']['ExcessNoiseFactor']
        self.WFS_Nph = torch.tensor(self.AO_config['sensor_HO']['NumberPhotons'], device=self.device)

        self.HOloop_rate  = np.mean(self.AO_config['RTC']['SensorFrameRate_HO']) # [Hz] (?)
        self.HOloop_delay = self.AO_config['RTC']['LoopDelaySteps_HO'] # [ms] (?)
        self.HOloop_gain  = self.AO_config['RTC']['LoopGain_HO']


    def InitGrids(self):
        if self.pixels_per_l_D is None:
            self.pixels_per_l_D = self.wvl*rad2mas / (self.psInMas*self.D)

        self.sampling_factor = int(np.ceil(2.0/self.pixels_per_l_D)) # check how much it is less than Nyquist
        self.sampling = self.sampling_factor * self.pixels_per_l_D
        self.nOtf = self.nPix * self.sampling_factor

        self.dk = 1/self.D/self.sampling # PSD spatial frequency step
        self.cte = (24*spc.gamma(6/5)/5)**(5/6)*(spc.gamma(11/6)**2/(2*np.pi**(11/3)))

        #with torch.no_grad():
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
        self.mask_corrected_AO = self.mask_corrected[corrected_ROI]
        self.mask_corrected_AO_1_1  = torch.unsqueeze(torch.unsqueeze(self.mask_corrected_AO,2),3)

        self.kx_AO = self.kx[corrected_ROI]
        self.ky_AO = self.ky[corrected_ROI]
        self.k_AO  = self.k [corrected_ROI]
        self.k2_AO = self.k2[corrected_ROI]

        #M_mask = torch.zeros([self.nOtf_AO,self.nOtf_AO,self.nGS,self.nGS], device=self.device)
        #for j in range(self.nGS):
        #    M_mask[:,:,j,j] += 1.0

        # Matrix repetitions and dimensions expansion to avoid in runtime
        self.kx_1_1 = torch.unsqueeze(torch.unsqueeze(self.kx_AO,2),3)
        self.ky_1_1 = torch.unsqueeze(torch.unsqueeze(self.ky_AO,2),3)
        self.k_1_1  = torch.unsqueeze(torch.unsqueeze(self.k_AO, 2),3)

        #self.kx_nGs_nGs = self.kx_1_1.repeat([1,1,self.nGS,self.nGS]) * M_mask
        #self.ky_nGs_nGs = self.ky_1_1.repeat([1,1,self.nGS,self.nGS]) * M_mask
        #self.k_nGs_nGs  = self.k_1_1.repeat([1,1,self.nGS,self.nGS])  * M_mask
        #self.kx_nGs_nL  = self.kx_1_1.repeat([1,1,self.nGS,self.nL])
        #self.ky_nGs_nL  = self.ky_1_1.repeat([1,1,self.nGS,self.nL])
        #self.k_nGs_nL   = self.k_1_1.repeat([1,1,self.nGS,self.nL])
        #self.kx_1_nL    = self.kx_1_1.repeat([1,1,1,self.nL])
        #self.ky_1_nL    = self.ky_1_1.repeat([1,1,1,self.nL])

        # For NGS-like alising 2nd dimension is used to store combs information
        self.km = self.kx_1_1.repeat([1,1,self.N_combs,1]) - torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(m/self.WFS_d_sub,0),0),3)
        self.kn = self.ky_1_1.repeat([1,1,self.N_combs,1]) - torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(n/self.WFS_d_sub,0),0),3)

        #self.GS_dirs_x_nGs_nL = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(self.GS_dirs_x,0),0),3).repeat([self.nOtf_AO,self.nOtf_AO,1,self.nL])# * dim_N_N_nGS_nL
        #self.GS_dirs_y_nGs_nL = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(self.GS_dirs_y,0),0),3).repeat([self.nOtf_AO,self.nOtf_AO,1,self.nL])# * dim_N_N_nGS_nL

        # Initialize OTF frequencines
        self.U,self.V = torch.meshgrid(
            torch.linspace(0, self.nOtf-1, self.nOtf, device=self.device),
            torch.linspace(0, self.nOtf-1, self.nOtf, device=self.device),
            indexing = 'ij')

        self.U = (self.U-self.nOtf/2) * 2/self.nOtf
        self.V = (self.V-self.nOtf/2) * 2/self.nOtf

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
        self.OTF_static = self.OTF_static / self.OTF_static.max()

        self.PSD_padder = torch.nn.ZeroPad2d((self.nOtf-self.nOtf_AO)//2)

        # Piston filter
        def PistonFilter(f):
            x = (np.pi*self.D*f).cpu().numpy() #TODO: find Bessel analog for pytorch
            R = spc.j1(x)/x
            piston_filter = torch.tensor(1.0-4*R**2, device=self.device)
            piston_filter[self.nOtf_AO//2,self.nOtf_AO//2,...] *= 0.0
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
        self.include_PSD = {}
        self.include_PSD['vK'] = True
        self.include_PSD['noise'] = True
        self.include_PSD['ST'] = True
        self.include_PSD['aliasing'] = True
        self.include_PSD['chromatism'] = True


    def Controller(self, nF=1000):
        #nTh = 1
        Ts = 1.0 / self.HOloop_rate # samplingTime
        delay = self.HOloop_delay #latency
        loopGain = self.HOloop_gain

        def TransferFunctions(freq):
            z = torch.exp(-2j*np.pi*freq*Ts)
            hInt = loopGain/(1.0 - z**(-1.0))
            rtfInt = 1.0/(1 + hInt*z**(-delay))
            atfInt = hInt * z**(-delay)*rtfInt
            ntfInt = atfInt / z
            return hInt, rtfInt, atfInt, ntfInt

        f = torch.logspace(-3, torch.log10(torch.tensor([0.5/Ts])).item(), nF)
        _, _, _, ntfInt = TransferFunctions(f)
        self.noise_gain = torch.trapz(torch.abs(ntfInt)**2, f)*2*Ts

        thetaWind = torch.tensor(0.0) #torch.linspace(0, 2*np.pi-2*np.pi/nTh, nTh)
        costh = torch.cos(thetaWind)
        
        fi = -self.vx*self.kx_1_1*costh - self.vy*self.ky_1_1*costh
        _, _, atfInt, ntfInt = TransferFunctions(fi)

        # AO transfer function
        self.h1 = self.Cn2_weights * atfInt #/nTh
        self.h2 = self.Cn2_weights * abs(atfInt)**2 #/nTh
        self.hn = self.Cn2_weights * abs(ntfInt)**2 #/nTh

        self.h1 = torch.sum(self.h1,axis=(2,3))
        self.h2 = torch.sum(self.h2,axis=(2,3))
        self.hn = torch.sum(self.hn,axis=(2,3))


    def ReconstructionFilter(self, r0, L0, WFS_noise_var):
        Av = torch.sinc(self.WFS_d_sub*self.kx_AO)*torch.sinc(self.WFS_d_sub*self.ky_AO) * torch.exp(1j*np.pi*self.WFS_d_sub*(self.kx_AO+self.ky_AO))
        self.SxAv = 2j*np.pi*self.kx_AO*self.WFS_d_sub*Av
        self.SyAv = 2j*np.pi*self.ky_AO*self.WFS_d_sub*Av

        MV = 0
        Wn = WFS_noise_var/(2*self.kc)**2
        # TODO: isn't this one computed for the WFSing wvl?
        self.W_atm = self.cte*r0**(-5/3)*(self.k2_AO + 1/L0**2)**(-11/6)*(self.wvl/self.GS_wvl)**2 #TODO: check for SPHERE
        gPSD = torch.abs(self.SxAv)**2 + torch.abs(self.SyAv)**2 + MV*Wn/self.W_atm
        self.Rx = torch.conj(self.SxAv) / gPSD
        self.Ry = torch.conj(self.SyAv) / gPSD
        self.Rx[self.nOtf_AO//2, self.nOtf_AO//2] *= 0
        self.Ry[self.nOtf_AO//2, self.nOtf_AO//2] *= 0


    def SpatioTemporalPSD(self):
        A = torch.ones([self.nOtf_AO, self.nOtf_AO], device=self.device) #TODO: fix it. A should be initialized differently
        Ff = self.Rx*self.SxAv + self.Ry*self.SyAv
        psd_ST = (1+abs(Ff)**2 * self.h2 - 2*torch.real(Ff*self.h1*A)) * self.W_atm * self.mask_corrected_AO
        return psd_ST


    def NoisePSD(self, WFS_noise_var):
        noisePSD = abs(self.Rx**2 + self.Ry**2) / (2*self.kc)**2
        #print(noisePSD.sum().item())
        noisePSD = noisePSD * self.piston_filter * self.noise_gain * WFS_noise_var * self.mask_corrected_AO
        #print(noisePSD.sum().item())
        return noisePSD

    def AliasingPSD(self, r0, L0):
        T = self.WFS_det_clock_rate / self.HOloop_rate
        td = T * self.HOloop_delay

        Rx1 = torch.unsqueeze(torch.unsqueeze(2j*np.pi*self.WFS_d_sub * self.Rx,2),3)
        Ry1 = torch.unsqueeze(torch.unsqueeze(2j*np.pi*self.WFS_d_sub * self.Ry,2),3)

        W_mn = (self.km**2 + self.kn**2 + 1/L0**2)**(-11/6)
        Q = (Rx1*self.km + Ry1*self.kn) * torch.sinc(self.WFS_d_sub*self.km) * torch.sinc(self.WFS_d_sub*self.kn)

        tf = torch.unsqueeze(torch.unsqueeze(self.h1,2),3)

        avr = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(self.Cn2_weights,0),0),0) * \
            (torch.sinc(self.km*self.vx*T) * torch.sinc(self.kn*self.vy*T) * \
            torch.exp(2j*np.pi*self.km*self.vx*td) * torch.exp(2j*np.pi*self.kn*self.vy*td) * tf.repeat([1,1,self.N_combs,self.nL]))

        aliasing_PSD = torch.sum(self.PR*W_mn*abs(Q*avr.sum(axis=3,keepdim=True))**2, axis=(2,3))*self.cte*r0**(-5/3) * self.mask_corrected_AO
        return aliasing_PSD


    def VonKarmanPSD(self, r0, L0):
        return self.cte*r0**(-5/3)*(self.k2 + 1/L0**2)**(-11/6) * self.mask


    def ChromatismPSD(self, r0, L0):
        wvlRef = self.wvl #TODO: polychromatic support
        W_atm = r0**(-5/3)*self.cte*(self.k2_AO + 1/L0**2)**(-11/6) * self.piston_filter #TODO: W_phi and vK spectrum
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
        r0_WFS = r0_new(r0.abs(), self.GS_wvl, 0.5e-6)
        WFS_nPix = self.WFS_FOV / self.WFS_n_sub
        WFS_pixelScale = self.WFS_psInMas / 1e3 # [arcsec]
        # Read-out noise calculation
        nD = torch.tensor([1.0, rad2arc*self.wvl/self.WFS_d_sub/WFS_pixelScale]).max() #spot FWHM in pixels and without turbulence
        varRON = np.pi**2/3 * (self.WFS_RON**2/self.WFS_Nph**2) * (WFS_nPix**2/nD)**2
        # Photon-noise calculation
        nT = torch.tensor([1.0, torch.hypot(self.WFS_spot_FWHM.max()/1e3, rad2arc*self.WFS_wvl/r0_WFS) / WFS_pixelScale], device=self.device).max()
        varShot = np.pi**2/(2*self.WFS_Nph) * (nT/nD)**2
        # Noise variance calculation
        varNoise = self.WFS_excessive_factor * (varRON+varShot)
        return varNoise


    def DLPSF(self):
        PSF = torch.abs( fft.fftshift(fft.ifft2(fft.fftshift(self.OTF_static))) ).unsqueeze(0).unsqueeze(0)
        PSF_out = interpolate(PSF, size=(self.nPix,self.nPix), mode='area').squeeze(0).squeeze(0)
        return (PSF_out/PSF_out.sum())


    def PSD2PSF(self, r0, L0, F, dx, dy, bg, dn, Jx, Jy, Jxy):
        WFS_noise_var = torch.abs(dn + self.NoiseVariance(r0))
        self.vx = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(self.wind_speed*torch.cos(self.wind_dir*np.pi/180.),0),0),0)
        self.vy = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(self.wind_speed*torch.sin(self.wind_dir*np.pi/180.),0),0),0)

        self.Controller()
        self.ReconstructionFilter(r0.abs(), L0.abs(), WFS_noise_var)

        dk = 2*self.kc/self.nOtf_AO

        # All PSD components are normalized to [nm] assuming that all cpntributors
        # are computed for 500 [nm]
        PSD_norm = (dk*500/2/np.pi)**2

        # WFS operates on another wavelength -> this PSD components is normalized for WFSing wvl 
        noise_PSD_norm = (dk*self.GS_wvl*1e9/2/np.pi)**2

        # Put all contributiors together and sum up the resulting PSD

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

        #PSD = self.VonKarmanPSD(r0.abs(),L0.abs()) * PSD_norm + \
        #self.PSD_padder(
        #    self.NoisePSD(WFS_noise_var) * noise_PSD_norm + \
        #    self.SpatioTemporalPSD() * PSD_norm + \
        #    self.AliasingPSD(r0.abs(), L0) * PSD_norm + \
        #    self.ChromatismPSD(r0.abs(), L0.abs()) * PSD_norm
        #)

        # Computing OTF from PSD
        cov = 2*fft.fftshift(fft.fft2(fft.fftshift(PSD))) # FFT axes are set to 1,2 by PyTorch by default
        # Computing the Structure Function from the covariance
        SF = torch.abs(cov).max()-cov
        # Phasor to shift the PSF with the subpixel accuracy
        fftPhasor  = torch.exp(-np.pi*1j*self.sampling_factor*(self.U*dx+self.V*dy))
        OTF_turb   = torch.exp(-0.5*SF*(2*np.pi*1e-9/self.wvl)**2)
        OTF_jitter = self.JitterCore(Jx.abs(),Jy.abs(),Jxy.abs())
        OTF = OTF_turb * self.OTF_static * fftPhasor * OTF_jitter

        #plt.imshow(OTF.abs().detach().cpu())
        #plt.show()
        #print(OTF.sum())

        PSF = torch.abs( fft.fftshift(fft.ifft2(fft.fftshift(OTF))) ).unsqueeze(0).unsqueeze(0)
        PSF_out = interpolate(PSF, size=(self.nPix,self.nPix), mode='area').squeeze(0).squeeze(0) #TODO: proper flux scaling
        
        if self.norm_regime == 'max':
            return PSF_out/PSF_out.max() * F + bg
        elif self.norm_regime == 'sum':
            return PSF_out/PSF_out.sum() * F + bg
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
toy = TipToy(config_file, data_test, 'CUDA')
toy.norm_regime = 'max'

el_croppo = slice(im.shape[0]//2-32, im.shape[1]//2+32)
el_croppo = (el_croppo, el_croppo)

if    toy.norm_regime == 'max': param = im.max()
elif  toy.norm_regime == 'sum': param = im.sum()
else: param = 1.0
PSF_0 = torch.tensor(im/param, device=toy.device)

dx_0, dy_0 = Center(PSF_0)

bg_0 = BackgroundEstimate(PSF_0, radius=90)

r0  = torch.tensor(data_test['r0'], requires_grad=True, device=toy.device)
L0  = torch.tensor(25.0, requires_grad=False, device=toy.device)
F   = torch.tensor(1.0,  requires_grad=True,  device=toy.device)
dx  = torch.tensor(0.0,  requires_grad=True,  device=toy.device)
dy  = torch.tensor(0.0,  requires_grad=True,  device=toy.device)
bg  = torch.tensor(0.0,  requires_grad=True,  device=toy.device)
dn  = torch.tensor(0.05, requires_grad=True,  device=toy.device)
Jx  = torch.tensor(10.0, requires_grad=True,  device=toy.device)
Jy  = torch.tensor(10.0, requires_grad=True,  device=toy.device)
Jxy = torch.tensor(2.0,  requires_grad=True,  device=toy.device)

parameters = [r0, L0, F, dx, dy, bg, dn, Jx, Jy, Jxy]
x = torch.stack(parameters).T.unsqueeze(0)

#toy.StartTimer()
PSF_1 = toy(x)
#print(toy.EndTimer())
PSF_DL = toy.DLPSF()

plt.imshow(torch.log( torch.hstack((PSF_0.abs()[el_croppo], PSF_1.abs()[el_croppo], ((PSF_1-PSF_0).abs()[el_croppo])) )).detach().cpu())
plot_radial_profile(PSF_0, PSF_1, 'TipToy', title='IRDIS PSF', dpi=100)

#%%
loss = nn.L1Loss(reduction='sum')

# Confines a value between 0 and the specified value
window_loss = lambda x, x_max: \
    (x>0).float()*(0.01/x)**2 + \
    (x<0).float()*100 + \
    100*(x>x_max).float()*(x-x_max)**2

# TODO: specify loss weights
def loss_fn(a,b):
    z = loss(a,b) + \
        window_loss(r0, 0.5) * 5.0 + \
        window_loss(Jx, 50) + \
        window_loss(Jy, 50) + \
        window_loss(Jxy, 400) + \
        window_loss(dn+toy.NoiseVariance(r0), 1.0)
    return z

#%
optimizer_lbfgs = OptimizeLBFGS(toy, parameters, loss_fn)

for i in range(20):
    optimizer_lbfgs.Optimize(PSF_0, [F, dx, dy, r0, dn], 5)
    optimizer_lbfgs.Optimize(PSF_0, [bg], 2)
    optimizer_lbfgs.Optimize(PSF_0, [Jx, Jy, Jxy], 3)

PSF_1 = toy.PSD2PSF(*parameters)
SR = lambda PSF: (PSF.max()/PSF_DL.max() * PSF_DL.sum()/PSF.sum()).item()

n_result = (dn + toy.NoiseVariance(r0)).abs().item()
n_init = toy.NoiseVariance(torch.tensor(data_test['r0'], device=toy.device)).item()

print("".join(['_']*52))
print('MAE + window value:', loss_fn(PSF_1, PSF_0).item())
print("r0,r0': ({:.3f}, {:.2f})".format(data_test['r0'], r0.item()))
print("F,bg:  ({:.3f}, {:.1E} )".format(F.data.item(), bg.item()))
print("dx,dy: ({:.2f}, {:.2f})".format(dx.data.item(), dy.item()))
print("Jx,Jy, Jxy: ({:.1f}, {:.1f}, {:.1f})".format(Jx.item(), Jy.item(), Jxy.item()))
print("n, n': ({:.2f},{:.2f})".format(n_init, n_result))

plt.imshow(torch.log( torch.hstack((PSF_0.abs()[el_croppo], PSF_1.abs()[el_croppo], ((PSF_1-PSF_0).abs()[el_croppo])) )).detach().cpu())
plt.show()

plot_radial_profile(PSF_0, PSF_1, 'TipToy', title='IRDIS PSF', dpi=100)
plt.show()
#la chignon et tarte


#%%
params = [p.clone().detach() for p in parameters]
PSF_ref = toy.PSD2PSF(*params)

dp = torch.tensor(1e-2, device=toy.device)

PSF_diff = []
for i in range(len(params)):
    params[i] += dp
    PSF_mod = toy.PSD2PSF(*params)
    PSF_diff.append( (PSF_mod-PSF_ref)/dp )
    params[i] -= dp

loss_fn = nn.L1Loss(reduction='sum')
def f(*params):
    return loss_fn(toy.PSD2PSF(*params), toy.PSD2PSF(*[p+dp for p in params]))

sensetivity = torch.autograd.functional.jacobian(f, tuple(params))

#%%
from matplotlib.colors import SymLogNorm

names = [r'r$_0$', r'L$_0$', 'F', 'dx', 'dy', 'bg', 'dn', r'J$_x$', r'J$_y$', r'J$_{xy}$']
scales = []

for name, diff_map in zip(names, PSF_diff):
    scales.append(diff_map.abs().sum())
    z_lims = max([diff_map.min().abs().item(), diff_map.max().abs().item()])
    plt.imshow( (diff_map).cpu(), cmap=plt.get_cmap('Spectral'), norm=SymLogNorm(z_lims*1e-3, vmin=-z_lims, vmax=z_lims) )
    plt.title(name)
    plt.colorbar()
    plt.show()


#%% =============================== MAKE DATASET ==========================================
### =======================================================================================
### =======================================================================================

# Load the SPHERE PSF database
path_fitted = 'C:/Users/akuznets/Data/SPHERE/fitted_TipToy_maxnorm/'
path_input  = 'C:/Users/akuznets/Data/SPHERE/test/'

database = SPHERE_database(path_input, path_fitted)

# Filter bad samples
bad_samples = []
for sample in database:
    buf = np.array([sample['fitted']['r0'],
                    sample['fitted']['F'],
                    sample['fitted']['n'],
                    sample['fitted']['dn'],
                    sample['fitted']['Jx'],
                    sample['fitted']['Jy'],
                    sample['fitted']['Jxy'],
                    sample['fitted']['dx'],
                    sample['fitted']['dy'],
                    sample['fitted']['bg']])

    wvl = sample['input']['spectrum']['lambda']
    r0_500 = r0_new(np.abs(sample['fitted']['r0']), 0.5e-6, wvl)
    
    n = sample['fitted']['n'] + sample['fitted']['dn']

    if np.any(np.isnan(buf)) or \
       np.isnan(sample['input']['WFS']['Nph vis']) or \
       np.abs(r0_500) > 3*sample['input']['r0'] or n > 2:
       bad_samples.append(sample['file_id']) 

for bad_sample in bad_samples:
    database.remove(bad_sample)

print(str(len(bad_samples))+' samples were filtered, '+str(len(database.data))+' samples remained')

'''
r0_500 = []
N_ph   = []
tau_0  = []
air    = []
wspd   = []
wdir   = []
seeing = []

for sample in database:
    wvl = sample['input']['spectrum']['lambda']
    r0_500.append( r0_new(np.abs(sample['fitted']['r0']), 0.5e-6, wvl) )
    seeing.append( sample['input']['seeing']['SPARTA'] )
    N_ph.append( np.log10(sample['input']['WFS']['Nph vis'] * sample['input']['WFS']['rate']*1240) )
    tau_0 .append( sample['input']['tau0']['SPARTA'] )
    air.append( sample['input']['telescope']['airmass'] )
    wspd.append( sample['input']['Wind speed']['MASSDIMM'] )
    wdir.append( sample['input']['Wind direction']['MASSDIMM'] )

r0_500 = np.array(r0_500)
N_ph = np.array(N_ph)
tau_0 = np.array(tau_0)
air = np.array(air)
wspd = np.array(wspd)
wdir = np.array(wdir)
seeing = np.array(seeing)

counts = plt.hist(N_ph, bins=20)
'''

def GetInputs(data_sample):
    #wvl     = data_sample['input']['spectrum']['lambda']
    r_0     = 3600*180/np.pi*0.976*0.5e-6 / data_sample['input']['seeing']['SPARTA'] # [m]
    tau0    = data_sample['input']['tau0']['SPARTA']
    wspeed  = data_sample['input']['Wind speed']['MASSDIMM']
    wdir    = data_sample['input']['Wind direction']['MASSDIMM']
    airmass = data_sample['input']['telescope']['airmass']
    Nph = np.log10(
        data_sample['input']['WFS']['Nph vis'] * data_sample['input']['WFS']['rate']*1240)
    input = np.array([
        r_0, tau0, wspeed, wdir, airmass, Nph,
        data_sample['fitted']['dx'],
        data_sample['fitted']['dy'],
        data_sample['fitted']['bg']])
    return input


def GetLabels(data_sample):
    toy.Update(data_sample['input'], reinit_grids=False)
    r0_500 = r0_new(np.abs(data_sample['fitted']['r0']), 0.5e-6, toy.wvl)
    #WFS_noise_var = data_sample['fitted']['n'] + data_sample['fitted']['dn']
    WFS_noise_var = data_sample['fitted']['dn']
    buf =  np.array([r0_500,
                     25.0,
                     data_sample['fitted']['F'],
                     data_sample['fitted']['dx'],
                     data_sample['fitted']['dy'],
                     data_sample['fitted']['bg'],
                     WFS_noise_var,
                     data_sample['fitted']['Jx'],
                     data_sample['fitted']['Jy'],
                     data_sample['fitted']['Jxy']])
    return buf


# Filter samples with the same (most presented) wavelength
wvls = []
for sample in database:
    wvl = sample['input']['spectrum']['lambda']
    wvls.append(wvl)
wvls = np.array(wvls)

sample_ids = np.arange(len(database))
wvl_unique, _, unique_indices, counts = np.unique(wvls, return_index=True, return_inverse=True, return_counts=True)
database_wvl = database.subset(sample_ids[unique_indices==np.argmax(counts)])

# Filter bad samples manually by their file ids
bad_file_ids = [
    90,  860, 840, 839, 832, 860, 846, 844, 836, 79,  78,  77,  76,  769,
    757, 754, 752, 738, 723, 696, 681, 676, 653, 642, 63,  636, 62,  620, 
    623, 616, 615, 599, 594, 58,  57,  584, 52,  54,  521, 51,  495, 494, 468, 
    456, 433, 415, 414, 373, 368, 364, 352, 342, 338, 336, 315, 296, 297, 
    298, 291, 290, 289, 276, 264, 253, 252, 236, 234, 233, 227, 221, 220, 
    215, 214, 213, 212, 211, 209, 208, 207, 206, 204, 203, 202, 201, 200, 786,
    193, 192, 191, 190, 189, 188, 174, 172, 171, 170, 169, 166, 165, 159, 
    158, 156, 155, 143, 139, 135, 132, 130, 128, 126, 96,  92,  787, 750,
    53,  513, 490, 369, 299, 270, 263, 255, 98,  88,  87,  86,  862, 796, 781]

bad_ids = [database_wvl.find(file_id)['index'] for file_id in bad_file_ids if database_wvl.find(file_id) is not None]
good_ids = list(set(np.arange(len(database_wvl))) - set(bad_ids))
database_wvl_good = database_wvl.subset(good_ids)


def GenerateDataset(dataset, with_PSF=False):
    x = [] # inputs
    y = [] # labels

    for sample in dataset:
        input = GetInputs(sample)
        if with_PSF:
            pred = sample['input']['image']
            pred = pred / pred.max()
        else:
            pred = GetLabels(sample)
        x.append(torch.tensor(input, device=toy.device).float())
        y.append(torch.tensor(pred,  device=toy.device).float())
    
    if with_PSF:
        return torch.vstack(x), torch.dstack(y).permute([2,0,1])
    else:
        return torch.vstack(x), torch.vstack(y)


#%%
validation_ids = np.unique(np.random.randint(0, high=len(database_wvl_good), size=30, dtype=int)).tolist()
database_train, database_val = database_wvl_good.split(validation_ids)

X_train, y_train = GenerateDataset(database_train, with_PSF=False)
X_val, y_val = GenerateDataset(database_val, with_PSF=False)

print(str(X_train.shape[0])+' samples in train dataset, '+str(X_val.shape[0])+' in validation')

#%%
class Gnosis(torch.nn.Module):
    def __init__(self, input_size, hidden_size, psf_model=None, tranform_fun=lambda x:x, device='cpu'):
        self.device = device
        super(Gnosis, self).__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size

        self.fc1  = torch.nn.Linear(self.input_size, self.hidden_size*2, device=self.device)
        self.act1 = torch.nn.Tanh()
        self.fc2  = torch.nn.Linear(self.hidden_size*2, hidden_size, device=self.device)
        self.act2 = torch.nn.Tanh()
        self.fc3  = torch.nn.Linear(self.hidden_size, 10, device=self.device)

        self.inp_normalizer = torch.ones(self.input_size, device=self.device)
        self.out_normalizer = torch.ones(10, device=self.device)
        self.inp_bias = torch.zeros(self.input_size, device=self.device)
        self.out_bias = torch.zeros(10, device=self.device)

        self.psf_model = psf_model
        self.tranform_fun = tranform_fun


    def forward(self, x):
        hidden1 = self.fc1(x * self.inp_normalizer + self.inp_bias)
        act1 = self.act1(hidden1)
        hidden2 = self.fc2(act1)
        act2 = self.act2(hidden2)
        model_inp = self.fc3(act2) * self.out_normalizer + self.out_bias
        if self.psf_model is None:
            return model_inp
        else:
            return self.psf_model(self.tranform_fun(model_inp)) # to rescale r0 mostly


gnosis = Gnosis(input_size=9, hidden_size=200, device=toy.device)
gnosis.inp_normalizer = torch.tensor([5, 50, 1/50, 1/360, 1, 0.5, 2, 2, 1e6], device=toy.device).unsqueeze(0)
gnosis.inp_bias = torch.tensor([0, 0, 0, 0, -1, -3, 0, 0, 0],   device=toy.device).unsqueeze(0)
gnosis.out_normalizer = 1.0/torch.tensor([1, 1, 1, 10, 10, 2e6, 1, .1, .1, .1], device=toy.device).unsqueeze(0)

#%%
loss_fn = nn.L1Loss() #reduction='sum')
optimizer = optim.SGD([{'params': gnosis.fc1.parameters()},
                       {'params': gnosis.act1.parameters()},
                       {'params': gnosis.fc2.parameters()},
                       {'params': gnosis.act2.parameters()},
                       {'params': gnosis.fc3.parameters()}], lr=1e-3, momentum=0.9)

for i in range(6000):
    optimizer.zero_grad()
    loss = loss_fn(gnosis(X_train), y_train)
    loss.backward()
    if not i % 1000: print(loss.item())
    optimizer.step()

print('Validation accuracy: '+str(loss_fn(gnosis(X_val), y_val).item()))

#torch.save(gnosis.state_dict(), 'gnosis_weights_psfao.dict')
#gnosis.load_state_dict(torch.load('gnosis_weights_psfao.dict'))
#gnosis.eval()


#%%
def r0_transform(pred):
    pred[0][0] = r0_new(pred[0][0], wvl_unique[np.argmax(counts)], 0.5e-6)
    return pred

def PSFcomparator(data_sample):
    toy2 = TipToy(config_file, data_sample['input'], 'CUDA')
    toy2.norm_regime = 'max'
    gnosis.psf_model = toy2
    #gnosis.tranform_fun = r0_transform

    x_test = torch.tensor(GetInputs(data_sample), device=toy2.device).float()
    PSF_2 = gnosis(x_test)
    A = torch.tensor(data_sample['input']['image'], device=toy2.device)
    C = torch.tensor(data_sample['fitted']['Img. fit'], device=toy2.device)
    norm = A.max()
    PSF_0 = A / norm
    PSF_1 = C / norm

    r0  = torch.tensor(data_sample['input']['r0'], device=toy2.device)
    L0  = torch.tensor(25.0, device=toy2.device)
    F   = torch.tensor(1.0,  device=toy2.device)
    dx  = torch.tensor(0.0,  device=toy2.device)
    dy  = torch.tensor(0.0,  device=toy2.device)
    bg  = torch.tensor(0.0,  device=toy2.device)
    dn  = torch.tensor(0.0,  device=toy2.device)
    Jx  = torch.tensor(10.0, device=toy2.device)
    Jy  = torch.tensor(10.0, device=toy2.device)
    Jxy = torch.tensor(2.0,  device=toy2.device)

    PSF_3 = toy2.PSD2PSF(r0, L0, F, dx, dy, bg, dn, Jx, Jy, Jxy)
    return PSF_0, PSF_1, PSF_2, PSF_3


loss_fn = nn.L1Loss()

fit_diff = []
gnosis_diff = []
direct_diff = []

PSF_0s = []
PSF_1s = []
PSF_2s = []
PSF_3s = []

profile_0s = []
profile_1s = []
profile_2s = []
profile_3s = []

#for data_sample in database_train:   #TODO: fix memory issue
for data_sample in database_val:

    PSF_0, PSF_1, PSF_2, PSF_3 = PSFcomparator(data_sample)
    fit_diff.append(loss_fn(PSF_0, PSF_1).item())
    gnosis_diff.append(loss_fn(PSF_0, PSF_2).item())
    direct_diff.append(loss_fn(PSF_0, PSF_3).item())

    PSF_0s.append(PSF_0.detach().cpu().numpy())
    PSF_1s.append(PSF_1.detach().cpu().numpy())
    PSF_2s.append(PSF_2.detach().cpu().numpy())
    PSF_3s.append(PSF_3.detach().cpu().numpy())

    profile_0s.append( radial_profile(PSF_0.detach().cpu().numpy())[:32] )
    profile_1s.append( radial_profile(PSF_1.detach().cpu().numpy())[:32] )
    profile_2s.append( radial_profile(PSF_2.detach().cpu().numpy())[:32] )
    profile_3s.append( radial_profile(PSF_3.detach().cpu().numpy())[:32] )

fit_diff = np.array(fit_diff)
gnosis_diff = np.array(gnosis_diff)
direct_diff = np.array(direct_diff)

PSF_0s = np.dstack(PSF_0s)
PSF_1s = np.dstack(PSF_1s)
PSF_2s = np.dstack(PSF_2s)
PSF_3s = np.dstack(PSF_3s)

profile_0s = np.vstack(profile_0s)
profile_1s = np.vstack(profile_1s)
profile_2s = np.vstack(profile_2s)
profile_3s = np.vstack(profile_3s)

c = profile_0s.mean(axis=0).max()

profile_0s /= c * 0.01
profile_1s /= c * 0.01
profile_2s /= c * 0.01
profile_3s /= c * 0.01

#%%
fig = plt.figure(figsize=(6,4), dpi=150)
plt.grid()

def plot_std(x,y, label, color, style):
    y_m = np.nanmean(y, axis=0)
    y_s = np.nanstd(y, axis=0)
    lower_bound = y_m-y_s
    upper_bound = y_m+y_s

    plt.fill_between(x, lower_bound, upper_bound, color=color, alpha=0.3)
    plt.plot(x, y_m, label=label, color=color, linestyle=style)

x = np.arange(32)
plot_std(x, profile_0s, 'Input PSF', 'darkslategray', '-')
plot_std(x, profile_0s-profile_1s, '$\Delta$ Fit', 'royalblue', '--')
plot_std(x, profile_0s-profile_2s, '$\Delta$ Gnosis', 'darkgreen', ':')
plot_std(x, profile_0s-profile_3s, '$\Delta$ Direct', 'orchid', 'dashdot')

plt.title('Accuracy comparison (avg. for validation dataset)')
plt.yscale('symlog')
plt.xlim([x.min(), x.max()])
plt.legend()
plt.ylabel('Abs. relative diff., [%]')
plt.xlabel('Pixels')

#%%
fd = np.nanmean(fit_diff)
gd = np.nanmean(gnosis_diff)
dd = np.nanmean(direct_diff)

print('Fitting: '+str(np.round(dd/fd*100-100).astype('int'))+'% improvement compared to direct prediction')
print('Gnosis: ' +str(np.round(dd/gd*100-100).astype('int'))+'% improvement compared to direct prediction')

#%% ==========================================================================================
loss_fn = nn.L1Loss() #reduction='sum')
optimizer = optim.SGD([{'params': gnosis.fc1.parameters()},
                       {'params': gnosis.act1.parameters()},
                       {'params': gnosis.fc2.parameters()},
                       {'params': gnosis.act2.parameters()},
                       {'params': gnosis.fc3.parameters()}], lr=1e-9, momentum=0.9)

sample = database_train[0]
toy = TipToy(config_file, sample['input'], 'CUDA')
toy.norm_regime = 'max'
gnosis.psf_model = toy

for i in range(20):
    for sample in database_train:
        toy.Update(data_sample['input'], reinit_grids=False)
        input = torch.tensor(GetInputs(sample), device=toy.device).unsqueeze(0).float()
        label = torch.tensor(sample['input']['image'] / sample['input']['image'].max(), device=toy.device).float()
        optimizer.zero_grad()
        pred = gnosis(input)
        loss = loss_fn(pred, label)
        loss.backward()
        print(loss.item())
        optimizer.step()
# %%
