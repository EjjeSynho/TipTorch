#%%
import sys
sys.path.insert(0, '..')

#TC10%: seeing better than 0.6 and tau0 larger than 5.2 ms
#TC20%: seeing < 0.70” and tau0 > 4.4 ms
#TC30%: seeing < 0.80” and tau0 > 4.1 ms
#TC50%: seeing < 1.00“ and tau0 > 3.2ms

import torch
import numpy as np
from torch import nn, optim, fft
import matplotlib.pyplot as plt
import scipy.special as spc
import time
from scipy.optimize import least_squares
from scipy.ndimage import center_of_mass
from torch.nn.functional import interpolate
from astropy.io import fits
from skimage.transform import resize
from graphviz import Digraph
import pickle
import os
from os import path
from tools.utils import plot_radial_profiles_new, SR, draw_PSF_stack, rad2mas

from tools.parameter_parser import ParameterParser
from data_processing.MUSE_preproc_utils_old import MUSEcube
from tools.utils import rad2mas, rad2arc, deg2rad, asec2rad, seeing, r0, r0_new


path_ini = '../data/parameter_files/muse_ltao.ini'

# Load image
data_dir = path.normpath('C:/Users/akuznets/Data/MUSE/DATA_Fernando/')
listData = os.listdir(data_dir)
sample_id = 5
sample_name = listData[sample_id]
path_im = path.join(data_dir, sample_name)
angle = np.zeros([len(listData)])
angle[0] = -46
angle[5] = -44
angle = angle[sample_id]

data_cube = MUSEcube(path_im, crop_size=200, angle=angle)
im, _, wvl = data_cube.Layer(5)
obs_info = dict( data_cube.obs_info )

# Load and correct AO system parameters
config_file = ParameterParser(path_ini).params
config_file['sources_science']['Wavelength'] = wvl
config_file['sensor_science']['FieldOfView'] = im.shape[0]


#%%
class TipToyMUSE(torch.nn.Module):

    def SetDataSample(self, obs_info):
        self.wvl = config_file['sources_science']['Wavelength']
        self.airmass_0 = obs_info['AIRMASS']

        self.AO_config['atmosphere']['Seeing']        = obs_info['SPTSEEIN']
        self.AO_config['atmosphere']['L0']            = obs_info['SPTL0']
        self.AO_config['atmosphere']['WindSpeed']     = [obs_info['WINDSP']] * len(self.AO_config['atmosphere']['Cn2Heights'])
        self.AO_config['atmosphere']['WindDirection'] = [obs_info['WINDIR']] * len(self.AO_config['atmosphere']['Cn2Heights'])
        self.AO_config['sensor_science']['Zenith']    = [90.0-obs_info['TELALT']]
        self.AO_config['sensor_science']['Azimuth']   = [obs_info['TELAZ']]
        self.AO_config['sensor_HO']['NoiseVariance']  = 4.5

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
        self.stretch     = 1.0 / (1.0-self.Cn2_heights/self.GS_height)
        self.h           = self.Cn2_heights * self.stretch
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

        pupil = torch.tensor(fits.getdata(pupil_path).astype('float'), device=self.device)
        if pupil_apodizer is None: apodizer = 1.0
        else:
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


    def __init__(self, AO_config, obs_info, device=None, pixels_per_l_D=None):
        if device is None or device == 'cpu' or device == 'CPU':
            self.device = torch.device('cpu')
            self.is_gpu = False

        elif device == 'cuda' or device == 'CUDA':
            self.device  = torch.device('cuda') # Will use the default CUDA device
            self.start = torch.cuda.Event(enable_timing=True)
            self.end   = torch.cuda.Event(enable_timing=True)
            self.is_gpu = True

        super().__init__()

        # Read data and initialize AO system
        self.pixels_per_l_D = pixels_per_l_D
        self.AO_config = AO_config
        self.SetDataSample(obs_info)
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

        self.W_atm = self.cte*r0**(-5/3)*(self.k2_AO + 1/L0**2)**(-11/6)*(self.wvl/self.GS_wvl)**2 #TODO: check for SPHERE
        gPSD = torch.abs(self.SxAv)**2 + torch.abs(self.SyAv)**2 + MV*Wn/self.W_atm
        self.Rx = torch.conj(self.SxAv) / gPSD
        self.Ry = torch.conj(self.SyAv) / gPSD
        self.Rx[self.nOtf_AO//2, self.nOtf_AO//2] *= 0
        self.Ry[self.nOtf_AO//2, self.nOtf_AO//2] *= 0


    def TomographicReconstructors(self, r0, L0, WFS_noise_var, inv_method='standart'):
        M = 2j*np.pi*self.k_nGs_nGs * \
            torch.sinc(self.WFS_d_sub*self.kx_nGs_nGs) * torch.sinc(self.WFS_d_sub*self.ky_nGs_nGs)
        P = torch.exp(2j*np.pi*self.h * \
            (self.kx_nGs_nL*self.GS_dirs_x_nGs_nL + self.ky_nGs_nL*self.GS_dirs_y_nGs_nL))

        MP = M @ P
        MP_t = torch.conj(torch.permute(MP, (0,1,3,2)))

        self.C_b = torch.ones((self.nOtf_AO, self.nOtf_AO, self.nGS, self.nGS), \
            dtype=torch.complex64, device=self.device) * torch.eye(4, device=self.device) * WFS_noise_var #torch.diag(WFS_noise_var)
        C_b_inv = torch.ones((self.nOtf_AO, self.nOtf_AO, self.nGS, self.nGS), \
            dtype=torch.complex64, device=self.device) * torch.eye(4, device=self.device) * 1./WFS_noise_var #torch.diag(WFS_noise_var)
        #TODO: ro at WFS wvl!
        #kernel = torch.unsqueeze(torch.unsqueeze(r0**(-5/3)*cte*(k2_AO + 1/L0**2)**(-11/6) * piston_filter, 2), 3)
        kernel = torch.unsqueeze(torch.unsqueeze(r0_new(r0, 589e-9, 500e-9)**(-5/3)* \
            self.cte*(self.k2_AO + 1/L0**2)**(-11/6) * self.piston_filter, 2), 3)
        self.C_phi  = kernel.repeat(1, 1, self.nL, self.nL) * torch.diag(self.Cn2_weights) + 0j
        
        if inv_method == 'fast' or inv_method == 'lstsq':
            #kernel_inv = torch.unsqueeze(torch.unsqueeze(1.0/ (r0**(-5/3)*cte*(k2_AO + 1/L0**2)**(-11/6)), 2), 3)
            kernel_inv = torch.unsqueeze(torch.unsqueeze(1.0/(r0_new(r0, 589e-9, 500e-9)**(-5/3) * \
                self.cte*(self.k2_AO + 1/L0**2)**(-11/6)), 2), 3)
            C_phi_inv = kernel_inv.repeat(1, 1, self.nL, self.nL) * torch.diag(1.0/self.Cn2_weights) + 0j

        if inv_method == 'standart':
            W_tomo = (self.C_phi @ MP_t) @ torch.linalg.pinv(MP @ self.C_phi @ MP_t + \
                self.C_b, rcond=1e-2)

        # elif inv_method == 'fast':
        #     W_tomo = torch.linalg.pinv(MP_t @ C_b_inv @ MP + C_phi_inv, rcond=1e-2) @ (MP_t @ C_b_inv) * \
        #         torch.unsqueeze(torch.unsqueeze(self.piston_filter,2),3).repeat(1,1,self.nL,self.nGS)
                
        # elif inv_method == 'lstsq':
        #     W_tomo = torch.linalg.lstsq(MP_t @ C_b_inv @ MP + C_phi_inv, MP_t @ C_b_inv).solution * \
        #         torch.unsqueeze(torch.unsqueeze(self.piston_filter,2),3).repeat(1,1,2,4)

        #TODO: in vanilla TIPTOP windspeeds are interpolated linearly if number of mod layers is changed!!!!!

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

        with torch.no_grad(): # an easier initialization for MUSE NFM
            self.P_beta_DM = torch.ones([W_tomo.shape[0], W_tomo.shape[1],1,1], dtype=torch.complex64, device=self.device) * self.mask_corrected_AO_1_1
            self.P_opt     = torch.ones([W_tomo.shape[0], W_tomo.shape[1],1,2], dtype=torch.complex64, device=self.device) * (self.mask_corrected_AO_1_1.repeat([1,1,1,self.nL])) #*dim_1_1_to_1_nL)

        self.W = self.P_opt @ W_tomo

        wDir_x = torch.cos(self.wind_dir*np.pi/180.0)
        wDir_y = torch.sin(self.wind_dir*np.pi/180.0)

        self.freq_t = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(wDir_x,0),0),0)*self.kx_1_nL + \
                      torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(wDir_y,0),0),0)*self.ky_1_nL

        samp_time = 1.0 / self.HOloop_rate
        www = 2j*torch.pi*self.k_nGs_nL * \
            torch.sinc(samp_time*self.WFS_det_clock_rate*self.wind_speed*self.freq_t).repeat([1,1,self.nGS,1]) #* dim_1_nL_to_nGS_nL

        #MP_alpha_L = www*torch.sinc(WFS_d_sub*kx_1_1)*torch.sinc(WFS_d_sub*ky_1_1)\
        #                                *torch.exp(2j*np.pi*h*(kx_nGs_nL*GS_dirs_x_nGs_nL + ky_nGs_nL*GS_dirs_y_nGs_nL))
        MP_alpha_L = www * P * \
            ( torch.sinc(self.WFS_d_sub*self.kx_1_1)*torch.sinc(self.WFS_d_sub*self.ky_1_1) )
        self.W_alpha = (self.W @ MP_alpha_L)


    def SpatioTemporalPSD(self):
        '''
        Beta = [self.ao.src.direction[0,s], self.ao.src.direction[1,s]]
        fx = Beta[0]*kx_AO
        fy = Beta[1]*ky_AO

        delta_h = h*(fx+fy) - delta_T * wind_speed_nGs_nL * freq_t
        '''
        delta_T  = (1 + self.HOloop_delay) / self.HOloop_rate
        delta_h  = -delta_T * self.freq_t * self.wind_speed
        P_beta_L = torch.exp(2j*np.pi*delta_h)

        proj = P_beta_L - self.P_beta_DM @ self.W_alpha
        proj_t = torch.conj(torch.permute(proj,(0,1,3,2)))
        psd_ST = torch.squeeze(torch.squeeze(torch.abs((proj @ self.C_phi @ proj_t)))) * \
            self.piston_filter * self.mask_corrected_AO
        return psd_ST


    def NoisePSD(self, WFS_noise_var):
        PW = self.P_beta_DM @ self.W
        noisePSD = PW @ self.C_b @ torch.conj(PW.permute(0,1,3,2))
        noisePSD = torch.squeeze(torch.squeeze(torch.abs(noisePSD))) * \
            self.piston_filter * self.noise_gain * WFS_noise_var * self.mask_corrected_AO #torch.mean(WFS_noise_var) 
        return noisePSD


    def AliasingPSD(self, r0, L0):
        T = self.WFS_det_clock_rate / self.HOloop_rate
        td = T * self.HOloop_delay

        Rx1 = torch.unsqueeze(torch.unsqueeze(2j*np.pi*self.WFS_d_sub * self.Rx,2),3)
        Ry1 = torch.unsqueeze(torch.unsqueeze(2j*np.pi*self.WFS_d_sub * self.Ry,2),3)

        W_mn = (self.km**2 + self.kn**2 + 1/L0**2)**(-11/6)
        Q = (Rx1*self.km+Ry1*self.kn) * torch.sinc(self.WFS_d_sub*self.km) * torch.sinc(self.WFS_d_sub*self.kn)

        tf = torch.unsqueeze(torch.unsqueeze(self.h1,2),3)

        avr = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(self.Cn2_weights,0),0),0) * \
            (torch.sinc(self.km*self.vx*T) * torch.sinc(self.kn*self.vx*T) * \
            torch.exp(2j*np.pi*self.km*self.vx*td) * torch.exp(2j*np.pi*self.kn*self.vx*td) * \
                tf.repeat([1,1,self.N_combs,self.nL]))

        aliasing_PSD = torch.sum(self.PR*W_mn*abs(Q*avr.sum(axis=3,keepdim=True))**2, axis=(2,3)) * \
            self.cte*r0**(-5/3) * self.mask_corrected_AO
        return aliasing_PSD


    def VonKarmanPSD(self, r0, L0):
        return self.cte*r0**(-5/3)*(self.k2 + 1/L0**2)**(-11/6) * self.mask


    def ChromatismPSD(self, r0, L0):
        wvlRef = self.wvl #TODO: polychromatic support
        W_atm = r0**(-5/3)*self.cte*(self.k2_AO + 1/L0**2)**(-11/6) * \
            self.piston_filter #TODO: W_phi and vK spectrum
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


    def PSD2PSF(self, r0, L0, F, dx, dy, bg, WFS_noise_var, Jx, Jy, Jxy):
        WFS_noise_var = torch.abs(WFS_noise_var)

        self.vx = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(
            self.wind_speed*torch.cos(self.wind_dir*np.pi/180.),0),0),0)
        self.vy = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(
            self.wind_speed*torch.sin(self.wind_dir*np.pi/180.),0),0),0)

        self.TomographicReconstructors(r0.abs(), L0.abs(), WFS_noise_var)
        self.Controller()
        self.ReconstructionFilter(r0.abs(), L0.abs(), WFS_noise_var)

        PSD = self.VonKarmanPSD(r0.abs(), L0.abs()) + \
        self.PSD_padder(
            self.NoisePSD(WFS_noise_var) + \
            self.SpatioTemporalPSD() + \
            self.AliasingPSD(r0.abs(), L0.abs()) + \
            self.ChromatismPSD(r0.abs(), L0.abs()))

        dk = 2*self.kc/self.nOtf_AO
        cov = 2*fft.fftshift(fft.fft2(fft.fftshift(PSD)))
        SF  = torch.abs(cov).max()-cov
        #SF *= (dk*wvl*1e9/2/np.pi)**2
        SF *= (dk*500/2/np.pi)**2 # PSD is computed for 500 [nm]

        fftPhasor = torch.exp(-np.pi*1j*self.sampling_factor*(self.U*dx+self.V*dy))
        OTF_turb  = torch.exp(-0.5*SF*(2*np.pi*1e-9/self.wvl)**2)
        OTF = OTF_turb * self.OTF_static * fftPhasor * self.JitterCore(Jx.abs(),Jy.abs(),Jxy.abs())
        #OTF = OTF[1:,1:]

        PSF = torch.abs( fft.fftshift(fft.ifft2(fft.fftshift(OTF))) ).unsqueeze(0).unsqueeze(0)
        PSF_out = interpolate(PSF, size=(self.nPix,self.nPix), mode='bilinear').squeeze(0).squeeze(0)
        return (PSF_out/PSF_out.sum() * F + bg)


    def forward(self, r0, L0, F, dx, dy, bg, n, Jx, Jy, Jxy):
        return self.PSD2PSF(r0, L0, F, dx, dy, bg, n, Jx, Jy, Jxy)


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


#%%
toy_MUSE = TipToyMUSE(config_file, obs_info, 'CUDA')

r0  = torch.tensor(0.1,   device=toy_MUSE.device)
L0  = torch.tensor(47.93, device=toy_MUSE.device)
F   = torch.tensor(1.0,   device=toy_MUSE.device)
dx  = torch.tensor(0.0,   device=toy_MUSE.device)
dy  = torch.tensor(0.0,   device=toy_MUSE.device)
bg  = torch.tensor(0.0,   device=toy_MUSE.device)
n   = torch.tensor(3.5,   device=toy_MUSE.device)
Jx  = torch.tensor(5.0,   device=toy_MUSE.device)
Jy  = torch.tensor(5.0,   device=toy_MUSE.device)
Jxy = torch.tensor(2.0,   device=toy_MUSE.device)

parameters = [r0, L0, F, dx, dy, bg, n, Jx, Jy, Jxy]

PSF_1 = toy_MUSE.PSD2PSF(*parameters)
PSF_0 = torch.tensor(im/im.sum(), device=toy_MUSE.device) #* 1e2
plt.imshow(torch.log10(PSF_0).detach().cpu())
plt.show()
plt.imshow(torch.log10(PSF_1).detach().cpu())
plt.show()

#%%
from torchmin import minimize

# Join tensors to a single tensor
def join_tensors(tensors):
    return torch.cat([t.view(-1) for t in tensors])

#Split a single tensor to a list of tensors
def split_tensor(tensor_list):
    return [tensor_list[i] for i in range(len(tensor_list))]

x0 = join_tensors([r0, F, dx, dy, n, Jx, Jy, Jxy])

def func(x_):
    inputs = split_tensor(x_)
    inputs.insert(1, L0)
    inputs.insert(4, bg)
    return toy_MUSE.PSD2PSF(*inputs)
    
def loss_fn(x_):
    loss = func(x_)- PSF_0
    return loss.flatten().abs().sum()

#%%
result = minimize(loss_fn, x0, max_iter=200, tol=1e-4, method='l-bfgs', disp=2)

x0 = result.x

#%%
with torch.no_grad():
    PSF_1 = func(x0)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    plot_radial_profiles_new( PSF_0[None,...].cpu().numpy(),  PSF_1[None,...].cpu().numpy(), 'Data', 'TipTorch', title='Left PSF',  ax=ax[0] )
    # plot_radial_profiles_new( PSF_0[:,-1,...].cpu().numpy(), PSF_1[:,-1,...].cpu().numpy(), 'Data', 'TipTorch', title='Right PSF', ax=ax[1] )
    plt.show()
  
    # wvl_select = np.s_[0, -N_wvl//2, -1]
  
    draw_PSF_stack(
        # PSF_0[:,wvl_select,...],
        # PSF_1[:,wvl_select,...],
        PSF_0[None,...],
        PSF_1[None,...],
        average=True, crop=80)#, scale=None)
        


#%%
optimizer_trf = OptimizeTRF(toy_MUSE, parameters)
optimizer_trf.Optimize(PSF_0)

PSF_1 = toy_MUSE(*parameters)

r0, L0, F, dx, dy, bg, n, Jx, Jy, Jxy = parameters

print("r0,L0: ({:.3f}, {:.2f})".format(r0.abs().item(), L0.abs().item()))
print("I,bg:  ({:.2f}, {:.1E})".format(F.item(), bg.item()))
print("dx,dy: ({:.2f}, {:.2f})".format(dx.item(), dy.item()))
print("Jx,Jy: ({:.1f}, {:.1f}, {:.1f})".format(Jx.abs().item(), Jy.abs().item(), Jxy.abs().item()))
print("WFS noise: {:.2f}".format(n.data.item()))

#%%
loss_fn1 = nn.L1Loss(reduction='sum')
def loss_fn(A,B):
    return loss_fn1(A,B) + torch.max(torch.abs(A-B)) + torch.abs(torch.max(A)-torch.max(B))


optimizer_lbfgs = OptimizeLBFGS(toy_MUSE, parameters, loss_fn)

for i in range(10):
    optimizer_lbfgs.Optimize(PSF_0, [r0, F, dx, dy], 5)
    optimizer_lbfgs.Optimize(PSF_0, [bg], 3)
    optimizer_lbfgs.Optimize(PSF_0, [n], 5)
    optimizer_lbfgs.Optimize(PSF_0, [Jx, Jy, Jxy], 3)

PSF_1 = toy_MUSE(*parameters)

print("r0,L0: ({:.3f}, {:.2f})".format(r0.data.item(), L0.data.item()))
print("I,bg:  ({:.2f}, {:.1E})".format(F.data.item(), bg.data.item()))
print("dx,dy: ({:.2f}, {:.2f})".format(dx.data.item(), dy.data.item()))
print("Jx,Jy: ({:.1f}, {:.1f}, {:.1f})".format(Jx.data.item(), Jy.data.item(), Jxy.data.item()))
print("WFS noise: {:.2f}".format(n.data.item()))


#%%
PSF_1 = toy_MUSE(*parameters)

plot_radial_profile(PSF_0, PSF_1, 'TipToy', title='MUSE NFM PSF')

plt.show()

#la chignon et tarte
