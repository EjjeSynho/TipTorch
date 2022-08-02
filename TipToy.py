#%%
import matplotlib
import torch
from torch import nn, optim, fft
from torch.nn import ParameterDict, ParameterList, Parameter


import numpy as np
import matplotlib.pyplot as plt
import scipy.special as spc
import scipy
from scipy.optimize import least_squares
from scipy.ndimage import center_of_mass
from torch.nn.functional import interpolate
from astropy.io import fits
from skimage.transform import resize
from graphviz import Digraph
from VLT_pupil import PupilVLT, CircPupil
import torch
import pickle
import os
from os import path
from parameterParser import parameterParser
import re


#cuda  = torch.device('cuda') # Default CUDA device
##cuda  = torch.device('cpu')
#start = torch.cuda.Event(enable_timing=True)
#end   = torch.cuda.Event(enable_timing=True)

rad2mas  = 3600 * 180 * 1000 / np.pi
rad2arc  = rad2mas / 1000
deg2rad  = np.pi / 180
asec2rad = np.pi / 180 / 3600

seeing = lambda r0, lmbd: rad2arc*0.976*lmbd/r0 # [arcs]
r0 = lambda seeing, lmbd: rad2arc*0.976*lmbd/seeing # [m]
r0_new = lambda r0, lmbd, lmbd0: r0*(lmbd/lmbd0)**1.2 # [m]

#path_test = 'C:\\Users\\akuznets\\Data\\SPHERE\\test\\210_SPHER.2017-09-19T00.38.31.896IRD_FLUX_CALIB_CORO_RAW_left.pickle'
#path_test = 'C:\\Users\\akuznets\\Data\\SPHERE\\test\\13_SPHER.2016-09-28T06.29.29.592IRD_FLUX_CALIB_CORO_RAW_left.pickle'
#path_test = 'C:\\Users\\akuznets\\Data\\SPHERE\\test\\190_SPHER.2017-03-07T06.09.15.212IRD_FLUX_CALIB_CORO_RAW_left.pickle'
#path_test = 'C:\\Users\\akuznets\\Data\\SPHERE\\test\\26_SPHER.2017-09-01T07.45.22.723IRD_FLUX_CALIB_CORO_RAW_left.pickle'
#path_test = 'C:\\Users\\akuznets\\Data\\SPHERE\\test\\422_SPHER.2017-05-20T10.28.56.559IRD_FLUX_CALIB_CORO_RAW_left.pickle'
#path_test = 'C:\\Users\\akuznets\\Data\\SPHERE\\test\\102_SPHER.2017-06-17T00.24.09.582IRD_FLUX_CALIB_CORO_RAW_left.pickle'

num = 444

files = os.listdir('C:/Users/akuznets/Data/SPHERE/test/')
for file in files:
    sample_num = re.findall(r'[0-9]+', file)[0]
    if int(sample_num) == num:
        path_test = 'C:/Users/akuznets/Data/SPHERE/test/' + file
        break



with open(path_test, 'rb') as handle:
    data_test = pickle.load(handle)

im  = data_test['image']

path_root = path.normpath('C:/Users/akuznets/Projects/TIPTOP/P3')
path_ini = path.join(path_root, path.normpath('aoSystem/parFiles/irdis.ini'))

config_file =  parameterParser(path_root, path_ini).params


#%% ------------------------ Managing paths ------------------------
class TipToy(torch.nn.Module):

    def SetDataSample(self, data_sample):
        self.wvl = data_test['spectrum']['lambda']

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


    def InitAOsystemConfigs(self):
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
        self.stretch     = 1.0 / (1.0-self.Cn2_heights/self.GS_height)
        self.h           = self.Cn2_heights * self.stretch
        self.nL          = self.Cn2_heights.size(0)

        self.WFS_d_sub = np.mean(self.AO_config['sensor_HO']['SizeLenslets'])
        self.WFS_n_sub = np.mean(self.AO_config['sensor_HO']['NumberLenslets'])
        self.WFS_det_clock_rate = np.mean(self.AO_config['sensor_HO']['ClockRate']) # [(?)]
        self.WFS_FOV = self.AO_config['sensor_HO']['FieldOfView']
        self.WFS_RON = self.AO_config['sensor_HO']['SigmaRON']
        self.WFS_psInMas = self.AO_config['sensor_HO']['PixelScale']
        self.WFS_wvl = torch.tensor(640e-9, device=self.device) #TODO: clarify this
        self.WFS_spot_FWHM = torch.tensor(self.AO_config['sensor_HO']['SpotFWHM'][0], device=self.device)
        self.WFS_excessive_factor = self.AO_config['sensor_HO']['ExcessNoiseFactor']
        self.WFS_Nph = torch.tensor(self.AO_config['sensor_HO']['NumberPhotons'], device=self.device)

        self.HOloop_rate  = np.mean(self.AO_config['RTC']['SensorFrameRate_HO']) # [Hz] (?)
        self.HOloop_delay = self.AO_config['RTC']['LoopDelaySteps_HO'] # [ms] (?)
        self.HOloop_gain  = self.AO_config['RTC']['LoopGain_HO']


    def InitGrids(self):
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

        M_mask = torch.zeros([self.nOtf_AO,self.nOtf_AO,self.nGS,self.nGS], device=self.device)
        for j in range(self.nGS):
            M_mask[:,:,j,j] += 1.0

        # Matrix repetitions and dimensions expansion to avoid in runtime
        self.kx_1_1 = torch.unsqueeze(torch.unsqueeze(self.kx_AO,2),3)
        self.ky_1_1 = torch.unsqueeze(torch.unsqueeze(self.ky_AO,2),3)
        self.k_1_1  = torch.unsqueeze(torch.unsqueeze(self.k_AO, 2),3)

        self.kx_nGs_nGs = self.kx_1_1.repeat([1,1,self.nGS,self.nGS]) * M_mask
        self.ky_nGs_nGs = self.ky_1_1.repeat([1,1,self.nGS,self.nGS]) * M_mask
        self.k_nGs_nGs  = self.k_1_1.repeat([1,1,self.nGS,self.nGS])  * M_mask
        self.kx_nGs_nL  = self.kx_1_1.repeat([1,1,self.nGS,self.nL])
        self.ky_nGs_nL  = self.ky_1_1.repeat([1,1,self.nGS,self.nL])
        self.k_nGs_nL   = self.k_1_1.repeat([1,1,self.nGS,self.nL])
        self.kx_1_nL    = self.kx_1_1.repeat([1,1,1,self.nL])
        self.ky_1_nL    = self.ky_1_1.repeat([1,1,1,self.nL])

        # For NGS-like alising 2nd dimension is used to store combs information
        self.km = self.kx_1_1.repeat([1,1,self.N_combs,1]) - torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(m/self.WFS_d_sub,0),0),3)
        self.kn = self.ky_1_1.repeat([1,1,self.N_combs,1]) - torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(n/self.WFS_d_sub,0),0),3)

        self.GS_dirs_x_nGs_nL = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(self.GS_dirs_x,0),0),3).repeat([self.nOtf_AO,self.nOtf_AO,1,self.nL])# * dim_N_N_nGS_nL
        self.GS_dirs_y_nGs_nL = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(self.GS_dirs_y,0),0),3).repeat([self.nOtf_AO,self.nOtf_AO,1,self.nL])# * dim_N_N_nGS_nL

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


    def __init__(self, AO_config, data_sample, device=None):
        #factory_kwargs = {'device': device, 'dtype': dtype}
        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device  = torch.device('cuda') # Will use the default CUDA device

        super().__init__()

        # Read data and initialize AO system
        self.AO_config = AO_config
        self.SetDataSample(data_sample)
        self.InitAOsystemConfigs()
        self.InitGrids()


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
        noise_gain = torch.trapz(torch.abs(ntfInt)**2, f)*2*Ts

        thetaWind = torch.tensor(0.0) #torch.linspace(0, 2*np.pi-2*np.pi/nTh, nTh)
        costh = torch.cos(thetaWind)
        
        vx = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(self.wind_speed*torch.cos(self.wind_dir*np.pi/180.),0),0),0)
        vy = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(self.wind_speed*torch.sin(self.wind_dir*np.pi/180.),0),0),0)
        fi = -vx*self.kx_1_1*costh - vy*self.ky_1_1*costh
        _, _, atfInt, ntfInt = TransferFunctions(fi)

        # AO transfer function
        h1 = self.Cn2_weights * atfInt #/nTh
        h2 = self.Cn2_weights * abs(atfInt)**2 #/nTh
        hn = self.Cn2_weights * abs(ntfInt)**2 #/nTh

        h1 = torch.sum(h1,axis=(2,3))
        h2 = torch.sum(h2,axis=(2,3))
        hn = torch.sum(hn,axis=(2,3))
        return h1, h2, hn, noise_gain


    def ReconstructionFilter(self, r0, L0, WFS_noise_var):
        Av = torch.sinc(self.WFS_d_sub*self.kx_AO)*torch.sinc(self.WFS_d_sub*self.ky_AO) * torch.exp(1j*np.pi*self.WFS_d_sub*(self.kx_AO+self.ky_AO))
        SxAv = 2j*np.pi*self.kx_AO*self.WFS_d_sub*Av
        SyAv = 2j*np.pi*self.ky_AO*self.WFS_d_sub*Av

        MV = 0
        Wn = WFS_noise_var/(2*self.kc)**2

        W_atm = self.cte*r0**(-5/3)*(self.k2_AO + 1/L0**2)**(-11/6)*(self.wvl/self.GS_wvl)**2
        gPSD = torch.abs(SxAv)**2 + torch.abs(SyAv)**2 + MV*Wn/W_atm
        Rx = torch.conj(SxAv) / gPSD
        Ry = torch.conj(SyAv) / gPSD
        Rx[self.nOtf_AO//2, self.nOtf_AO//2] *= 0
        Ry[self.nOtf_AO//2, self.nOtf_AO//2] *= 0
        return Rx, Ry, SxAv, SyAv, W_atm


    def SpatioTemporalPSD(self, Rx, Ry, SxAv, SyAv, h1, h2, W_atm):
        A = torch.ones([self.nOtf_AO, self.nOtf_AO], device=self.device) #TODO: fix it. A should be initialized differently
        Ff = Rx*SxAv + Ry*SyAv
        psd_ST = (1+abs(Ff)**2 * h2 - 2*torch.real(Ff*h1*A)) * W_atm * self.mask_corrected_AO
        return psd_ST


    def NoisePSD(self, Rx, Ry, noise_gain, WFS_noise_var):
        noisePSD = abs(Rx**2 + Ry**2) /(2*self.kc)**2
        noisePSD = noisePSD * self.piston_filter * noise_gain * WFS_noise_var * self.mask_corrected_AO
        return noisePSD


    def AliasingPSD(self, Rx, Ry, h1, r0, L0):
        T = self.WFS_det_clock_rate / self.HOloop_rate
        td = T * self.HOloop_delay
        vx = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(self.wind_speed*torch.cos(self.wind_dir*np.pi/180.),0),0),0)
        vy = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(self.wind_speed*torch.sin(self.wind_dir*np.pi/180.),0),0),0)

        Rx1 = torch.unsqueeze(torch.unsqueeze(2j*np.pi*self.WFS_d_sub * Rx,2),3)
        Ry1 = torch.unsqueeze(torch.unsqueeze(2j*np.pi*self.WFS_d_sub * Ry,2),3)

        W_mn = (self.km**2 + self.kn**2 + 1/L0**2)**(-11/6)
        Q = (Rx1*self.km + Ry1*self.kn) * torch.sinc(self.WFS_d_sub*self.km) * torch.sinc(self.WFS_d_sub*self.kn)

        tf = torch.unsqueeze(torch.unsqueeze(h1,2),3)

        avr = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(self.Cn2_weights,0),0),0) * \
            (torch.sinc(self.km*vx*T) * torch.sinc(self.kn*vy*T) * \
            torch.exp(2j*np.pi*self.km*vx*td) * torch.exp(2j*np.pi*self.kn*vy*td) * tf.repeat([1,1,self.N_combs,self.nL]))

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


    def NoiseVariance(self, r0): #TODO: do input of actual r0 and rescale it inside
        WFS_nPix = self.WFS_FOV / self.WFS_n_sub
        WFS_pixelScale = self.WFS_psInMas / 1e3 # [arcsec]
        # Read-out noise calculation
        nD = torch.tensor([1.0, rad2arc*self.wvl/self.WFS_d_sub/WFS_pixelScale]).max() #spot FWHM in pixels and without turbulence
        varRON = np.pi**2/3 * (self.WFS_RON**2/self.WFS_Nph**2) * (WFS_nPix**2/nD)**2
        # Photon-noise calculation
        nT = torch.tensor([1.0, torch.hypot(self.WFS_spot_FWHM.max()/1e3, rad2arc*self.WFS_wvl/r0) / WFS_pixelScale], device=self.device).max()
        varShot = np.pi**2/(2*self.WFS_Nph) * (nT/nD)**2
        # Noise variance calculation
        varNoise = self.WFS_excessive_factor * (varRON+varShot)
        return varNoise


    def DLPSF(self):
        PSF = torch.abs( fft.fftshift(fft.ifft2(fft.fftshift(self.OTF_static))) ).unsqueeze(0).unsqueeze(0)
        PSF_out = interpolate(PSF, size=(self.nPix,self.nPix), mode='area').squeeze(0).squeeze(0)
        return (PSF_out/PSF_out.sum())


    def PSD2PSF(self, r0, L0, F, dx, dy, bg, WFS_noise_var, Jx, Jy, Jxy):
        # non-negative reparametrization
        r0  = torch.abs(r0)
        L0  = torch.abs(L0)
        Jx  = torch.abs(Jx)
        Jy  = torch.abs(Jy)
        Jxy = torch.abs(Jxy)

        #WFS_noise_var = torch.abs(WFS_noise_var)
        WFS_noise_var2 = ( WFS_noise_var + self.NoiseVariance(r0_new(r0, 0.64e-6, self.wvl)) ).abs()

        h1, h2, _, noise_gain = self.Controller()
        Rx, Ry, SxAv, SyAv, W_atm = self.ReconstructionFilter(r0, L0, WFS_noise_var2)

        PSD =  self.VonKarmanPSD(r0,L0) + \
        self.PSD_padder(
            self.NoisePSD(Rx, Ry, noise_gain, WFS_noise_var2) + \
            self.SpatioTemporalPSD(Rx, Ry, SxAv, SyAv, h1, h2, W_atm) + \
            self.AliasingPSD(Rx, Ry, h1, r0, L0) + \
            self.ChromatismPSD(r0, L0)
        )

        dk = 2*self.kc/self.nOtf_AO
        PSD *= (dk*self.wvl*1e9/2/np.pi)**2
        cov = 2*fft.fftshift(fft.fft2(fft.fftshift(PSD)))
        SF  = torch.abs(cov).max()-cov
        fftPhasor = torch.exp(-np.pi*1j*self.sampling_factor*(self.U*dx+self.V*dy))
        OTF_turb  = torch.exp(-0.5*SF*(2*np.pi*1e-9/self.wvl)**2)

        OTF = OTF_turb * self.OTF_static * fftPhasor * self.JitterCore(Jx,Jy,Jxy)
        PSF = torch.abs( fft.fftshift(fft.ifft2(fft.fftshift(OTF))) ).unsqueeze(0).unsqueeze(0)
        PSF_out = interpolate(PSF, size=(self.nPix,self.nPix), mode='area').squeeze(0).squeeze(0)
        return (PSF_out/PSF_out.sum() * F + bg)



#%% -------------------------------------------------------------

toy = TipToy(config_file, data_test)

el_croppo = slice(256//2-32, 256//2+32)
el_croppo = (el_croppo,el_croppo)

#def BackgroundEstimate(im, radius=90):
#    buf_x, buf_y = torch.meshgrid(
#        torch.linspace(-nPix//2, nPix//2, nPix, device=self.device),
#        torch.linspace(-nPix//2, nPix//2, nPix, device=self.device),
#        indexing = 'ij'
#    )
#    mask_noise = buf_x**2 + buf_y**2
#    mask_noise[mask_noise < radius**2] = 0.0
#    mask_noise[mask_noise > 0.0] = 1.0
#    return torch.median(im[mask_noise>0.]).data

param = im.sum()
PSF_0 = torch.tensor(im/param, device=toy.device)
#Cut
plt.imshow(torch.log(PSF_0[el_croppo]).detach().cpu())
plt.show()
#Cut

#def Center(im):
#    WoG_ROI = 16
#    center = np.array(np.unravel_index(im.argmax().item(), im.shape))
#    crop = slice(center[0]-WoG_ROI//2, center[1]+WoG_ROI//2)
#    crop = (crop, crop)
#    buf = PSF_0[crop].detach().cpu().numpy()
#    WoG = np.array(scipy.ndimage.center_of_mass(buf)) + im.shape[0]//2-WoG_ROI//2
#    return WoG-np.array(im.shape)//2

r0_scaled = r0_new(data_test['r0'], toy.wvl, 0.5e-6) #* (1+0.25*(np.random.random_sample()-0.5))
r0_scaled_WFS = r0_new(data_test['r0'], 0.64e-6, 0.5e-6)
noise_var = toy.NoiseVariance(torch.tensor(r0_scaled_WFS, device=toy.device)).item()#.clip(0.05, 1.0).item()

#dx_0, dy_0 = Center(PSF_0)
r0  = torch.tensor(r0_scaled, requires_grad=True,  device=toy.device)
L0  = torch.tensor(25.0, requires_grad=False, device=toy.device)
F   = torch.tensor(1.0,  requires_grad=True,  device=toy.device)
dx  = torch.tensor(0.0, requires_grad=True,  device=toy.device)
dy  = torch.tensor(0.0, requires_grad=True,  device=toy.device)
bg  = torch.tensor(0.0,  requires_grad=True,  device=toy.device)
n   = torch.tensor(0.0,  requires_grad=True, device=toy.device)
Jx  = torch.tensor(10.0, requires_grad=True,  device=toy.device)
Jy  = torch.tensor(10.0, requires_grad=True,  device=toy.device)
Jxy = torch.tensor(2.0,  requires_grad=True,  device=toy.device)
PSF_1 = toy.PSD2PSF(r0, L0, F, dx, dy, bg, n, Jx, Jy, Jxy)
#Cut
plt.imshow(torch.log( torch.hstack((PSF_0[el_croppo], PSF_1[el_croppo], ((PSF_1-PSF_0).abs()[el_croppo])) )).detach().cpu())
#Cut

PSF_DL = toy.DLPSF()

#%%
'''
X1 = torch.stack([r0,F,dx,dy,bg,n,Jx,Jy,Jxy]).detach().cpu().numpy()
def wrapper(X):
    r0, F, dx, dy, bg, n, Jx, Jy, Jxy = torch.tensor(X, dtype=torch.float32, device=cuda)
    L0 = torch.tensor(25.0, dtype=torch.float32, device=cuda)
    return PSD2PSF(r0, L0, F, dx, dy, bg, n, Jx, Jy, Jxy)

func = lambda x: (PSF_0-wrapper(x)).detach().cpu().numpy().reshape(-1)
result = least_squares(func, X1, method = 'trf',
                       ftol=1e-8, xtol=1e-8, gtol=1e-8,
                       max_nfev=2000, verbose=1, loss="linear")
X1 = result.x
r0, F, dx, dy, bg, n, Jx, Jy, Jxy = torch.tensor(X1, dtype=torch.float32, device=cuda)
L0 = torch.tensor(25.0, dtype=torch.float32, device=cuda)
'''
#%%
def OptimParams(loss_fun, params, iterations, method='LBFGS', verbous=True):
    last_loss = 1e16
    trigger_times = 0

    if method == 'LBFGS':
        optimizer = optim.LBFGS(params, lr=10, history_size=20, max_iter=4, line_search_fn="strong_wolfe")
    elif method == 'Adam':
        optimizer = optim.Adam(params, lr=1e-3)

    #history = []
    for i in range(iterations):
        optimizer.zero_grad()
        loss = loss_fun( PSD2PSF(r0, L0, F, dx, dy, bg, n, Jx, Jy, Jxy), PSF_0 )
        loss.backward()

        #history.append(loss.item())
        #if len(history) > 2:
        #    if np.abs(loss.item()-history[-1]) < 1e-4 and np.abs(loss.item()-history[-2]) < 1e-4:
        #        break

        if method == 'LBFGS':
            optimizer.step( lambda: loss_fun( PSD2PSF(r0, L0, F, dx, dy, bg, n, Jx, Jy, Jxy), PSF_0 ) )
        elif method == 'Adam':
            optimizer.step()

        if verbous:
            if method == 'LBFGS': print('Loss:', loss.item(), end="\r", flush=True)
            elif method == 'Adam':
                if not i % 10: print('Loss:', loss.item(), end="\r", flush=True)
        # Early stop check
        if np.round(loss.item(),3) >= np.round(last_loss,3):
            trigger_times += 1
            if trigger_times >=3:
                return
        last_loss = loss.item()    

loss_fn = nn.L1Loss(reduction='sum')
for i in range(20):
    OptimParams(loss_fn, [F, dx, dy, r0, n], 5)
    #OptimParams(loss_fn, [n], 5)
    OptimParams(loss_fn, [bg], 2)
    OptimParams(loss_fn, [Jx, Jy, Jxy], 3)

PSF_1 = PSD2PSF(r0, L0, F, dx, dy, bg, n, Jx, Jy, Jxy)
SR = lambda PSF: (PSF.max()/PSF_DL.max() * PSF_DL.sum()/PSF.sum()).item()

#%%
#Cut
n_result = (n + NoiseVariance(r0_new(r0, 0.64e-6, wvl)) ).abs().data.item()
n_init = NoiseVariance(torch.tensor(r0_new(data_test['r0'], 0.64e-6, 0.5e-6), device=self.device)).item()

print("".join(['_']*52))
print('Loss:', loss_fn(PSF_1, PSF_0).item())
print("r0,r0': ({:.3f}, {:.2f})".format(data_test['r0'], r0_new(r0.data.item(), 0.5e-6, wvl)))
print("I,bg:  ({:.3f}, {:.1E} )".format(F.data.item(), bg.data.item()))
print("dx,dy: ({:.2f}, {:.2f})".format(dx.data.item(), dy.data.item()))
print("Jx,Jy, Jxy: ({:.1f}, {:.1f}, {:.1f})".format(Jx.data.item(), Jy.data.item(), Jxy.data.item()))
print("n, n': ({:.2f},{:.2f})".format(n_init, n_result))


plt.imshow(torch.log( torch.hstack((PSF_0[el_croppo], PSF_1[el_croppo], ((PSF_1-PSF_0).abs()[el_croppo])) )).detach().cpu())
plt.show()
#Cut

#%%
def FitGauss2D(PSF):
    nPix_crop = 16
    crop = slice(nPix//2-nPix_crop//2, nPix//2+nPix_crop//2)
    PSF_cropped = torch.tensor(PSF[crop,crop], requires_grad=False, device=self.device)
    PSF_cropped = PSF_cropped / PSF_cropped.max()

    px, py = torch.meshgrid(
        torch.linspace(-nPix_crop/2, nPix_crop/2-1, nPix_crop, device=self.device),
        torch.linspace(-nPix_crop/2, nPix_crop/2-1, nPix_crop, device=self.device),
        indexing = 'ij')

    def Gauss2D(X):
        return X[0]*torch.exp( -((px-X[1])/(2*X[3]))**2 - ((py-X[2])/(2*X[4]))**2 )

    X0 = torch.tensor([1.0, 0.0, 0.0, 1.1, 1.1], requires_grad=True, device=self.device)

    loss_fn = nn.MSELoss()
    optimizer = optim.LBFGS([X0], history_size=10, max_iter=4, line_search_fn="strong_wolfe")

    for _ in range(20):
        optimizer.zero_grad()
        loss = loss_fn(Gauss2D(X0), PSF_cropped)
        loss.backward()
        optimizer.step(lambda: loss_fn(Gauss2D(X0), PSF_cropped))

    FWHM = lambda x: 2*np.sqrt(2*np.log(2)) * np.abs(x)

    #plt.imshow(Gauss2D(X0).detach().cpu())
    #plt.show()
    #plt.imshow(PSF_cropped.detach().cpu())
    #plt.show()

    return FWHM(X0[3].detach().cpu().numpy()), FWHM(X0[4].detach().cpu().numpy())


#%%
#Cut
def radial_profile(data, center=None):
    if center is None:
        center = (data.shape[0]//2, data.shape[1]//2)
    y, x = np.indices((data.shape))
    r = np.sqrt( (x-center[0])**2 + (y-center[1])**2 )
    r = r.astype('int')

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile[0:data.shape[0]//2]

PSF_1 = PSD2PSF(r0, L0, F, dx, dy, bg, n, Jx, Jy, Jxy)

profile_0 = radial_profile(PSF_0.clone().detach().cpu().numpy())[:32]
profile_1 = radial_profile(PSF_1.clone().detach().cpu().numpy())[:32]
profile_diff = np.abs(profile_1-profile_0) / PSF_0.max().cpu().numpy() * 100 #[%]

fig = plt.figure(figsize=(6,4), dpi=150)
ax = fig.add_subplot(111)
ax.set_title('TipToy fitting')
l2 = ax.plot(profile_0, label='Data')
l1 = ax.plot(profile_1, label='TipToy')
ax.set_xlabel('Pixels')
ax.set_ylabel('Relative intensity')
ax.set_yscale('log')
ax.set_xlim([0, len(profile_1)])
ax.grid()

ax2 = ax.twinx()
l3 = ax2.plot(profile_diff, label='Difference', color='green')
ax2.set_ylim([0, profile_diff.max()*1.5])
ax2.set_ylabel('Difference [%]')

ls = l1+l2+l3
labs = [l.get_label() for l in ls]
ax2.legend(ls, labs, loc=0)

plt.show()
#Cut

#la chignon et tarte
