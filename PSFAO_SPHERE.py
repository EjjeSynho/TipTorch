#%%
import torch
from torch import nn, optim, fft
from torch.nn import ParameterDict, ParameterList, Parameter
from torch.nn.functional import interpolate

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as spc
from scipy.optimize import least_squares
from scipy import signal
from astropy.io import fits
import pickle
import time
import os
from os import path
import re
from tqdm import tqdm

from parameterParser import parameterParser
from utils import rad2mas, rad2arc, deg2rad, asec2rad, seeing, r0, r0_new
from utils import Center, BackgroundEstimate, CircularMask
from utils import register_hooks, iter_graph
from utils import OptimizeTRF, OptimizeLBFGS
from utils import radial_profile, plot_radial_profile
from SPHERE_data import SPHERE_database, LoadSPHEREsampleByID
# end of import list

#210
#13
#190
#26
#422
#102

num = 260

num_id, data_test = LoadSPHEREsampleByID('C:/Users/akuznets/Data/SPHERE/test/', num)

im = data_test['image']
path_root = path.normpath('C:/Users/akuznets/Projects/TIPTOP/P3')
path_ini = path.join(path_root, path.normpath('aoSystem/parFiles/irdis.ini'))

config_file = parameterParser(path_root, path_ini).params


#%% ------------------------ Managing paths ------------------------
class PSFAO(torch.nn.Module):

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

        self.k2  = self.kx**2 + self.ky**2
        self.k   = torch.sqrt(self.k2)
        self.kxy = self.kx * self.ky
        self.kx2 = self.kx**2
        self.ky2 = self.ky**2

        self.mask = torch.ones_like(self.k2, device=self.device)
        self.mask[self.k2 <= self.kc**2] = 0
        self.mask_corrected = 1.0-self.mask

        self.nOtf_AO = int(2*self.kc/self.dk)
        self.nOtf_AO += self.nOtf_AO % 2

        corrected_ROI = slice(self.nOtf//2-self.nOtf_AO//2, self.nOtf//2+self.nOtf_AO//2)
        corrected_ROI = (corrected_ROI,corrected_ROI)

        self.mask_AO = self.mask[corrected_ROI]
        self.mask_corrected_AO = self.mask_corrected[corrected_ROI]
        self.mask_corrected_AO_1_1  = torch.unsqueeze(torch.unsqueeze(self.mask_corrected_AO,2),3)

        self.kx_AO  = self.kx  [corrected_ROI]
        self.ky_AO  = self.ky  [corrected_ROI]
        self.kx2_AO = self.kx2 [corrected_ROI]
        self.ky2_AO = self.ky2 [corrected_ROI]
        self.kxy_AO = self.kxy [corrected_ROI]
        self.k_AO   = self.k   [corrected_ROI]
        self.k2_AO  = self.k2  [corrected_ROI]

        # Matrix repetitions and dimensions expansion to avoid in runtime
        self.kx_1_1 = torch.unsqueeze(torch.unsqueeze(self.kx_AO,2),3)
        self.ky_1_1 = torch.unsqueeze(torch.unsqueeze(self.ky_AO,2),3)
        self.k_1_1  = torch.unsqueeze(torch.unsqueeze(self.k_AO, 2),3)

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


    def VonKarmanPSD(self, r0, L0):
        #return self.cte*r0**(-5/3)*(self.k2 + 1/L0**2)**(-11/6) * self.mask
        return self.cte*r0.unsqueeze(1).unsqueeze(2)**(-5/3)*(self.k2.unsqueeze(0) + \
            1/L0.unsqueeze(1).unsqueeze(2)**2)**(-11/6) * self.mask.unsqueeze(0)


    def JitterCore(self, Jx, Jy, Jxy):
        u_max = self.sampling*self.D/self.wvl/(3600*180*1e3/np.pi)
        norm_fact = u_max**2 * (2*np.sqrt(2*np.log(2)))**2
        Djitter = norm_fact * (Jx**2 * self.U2 + Jy**2 * self.V2 + 2*Jxy*self.UV)
        return torch.exp(-0.5*Djitter) #TODO: cover Nyquist sampled case


    def DLPSF(self):
        PSF = torch.abs( fft.fftshift(fft.ifft2(fft.fftshift(self.OTF_static))) ).unsqueeze(0).unsqueeze(0)
        PSF_out = interpolate(PSF, size=(self.nPix,self.nPix), mode='area').squeeze(0).squeeze(0)
        return (PSF_out/PSF_out.sum())


    def MoffatPSD(self, amp, b, alpha, beta, ratio, theta):
        ax = (alpha * ratio).unsqueeze(1).unsqueeze(2)
        ay = (alpha / ratio).unsqueeze(1).unsqueeze(2)

        uxx = self.kx2_AO.unsqueeze(0)
        uxy = self.kxy_AO.unsqueeze(0)
        uyy = self.ky2_AO.unsqueeze(0)

        #def reduced_center_coord(uxx, uxy, uyy, ax, ay, theta):
        c  = torch.cos(theta).unsqueeze(1).unsqueeze(2)
        s  = torch.sin(theta).unsqueeze(1).unsqueeze(2)
        s2 = torch.sin(2.0 * theta).unsqueeze(1).unsqueeze(2)

        rxx = (c/ax)**2 + (s/ay)**2
        rxy =  s2/ay**2 -  s2/ax**2
        ryy = (c/ay)**2 + (s/ax)**2

        uu = rxx*uxx + rxy*uxy + ryy*uyy

        V = (1.0+uu)**(-beta.unsqueeze(1).unsqueeze(2)) # Moffat shape

        removeInside = 0.0
        E = (beta.unsqueeze(1).unsqueeze(2)-1) / (np.pi*ax*ay)
        Fout = (1 +      (self.kc**2)/(ax*ay))**(1-beta.unsqueeze(1).unsqueeze(2))
        Fin  = (1 + (removeInside**2)/(ax*ay))**(1-beta.unsqueeze(1).unsqueeze(2))
        F = 1/(Fin-Fout)

        MoffatPSD = (amp.unsqueeze(1).unsqueeze(2) * V*E*F + b.unsqueeze(1).unsqueeze(2)) * \
            self.mask_corrected_AO.unsqueeze(0)
        MoffatPSD[..., self.nOtf_AO//2, self.nOtf_AO//2] *= 0.0

        return MoffatPSD

        '''        
        ax = alpha * ratio
        ay = alpha / ratio

        def reduced_center_coord(uxx, uxy, uyy, ax, ay, theta):
            c  = torch.cos(theta)
            s  = torch.sin(theta)
            s2 = torch.sin(2.0 * theta)

            rxx = (c/ax)**2 + (s/ay)**2
            rxy =  s2/ay**2 -  s2/ax**2
            ryy = (c/ay)**2 + (s/ax)**2
            
            uu = rxx*uxx + rxy*uxy + ryy*uyy
            return uu

        uu = reduced_center_coord(self.kx2_AO, self.kxy_AO, self.ky2_AO, ax, ay, theta)
        V = (1.0+uu)**(-beta) # Moffat shape

        removeInside = 0.0
        E = (beta-1) / (np.pi*ax*ay)
        Fout = (1 +      (self.kc**2)/(ax*ay))**(1-beta)
        Fin  = (1 + (removeInside**2)/(ax*ay))**(1-beta)
        F = 1/(Fin-Fout)

        MoffatPSD = (amp * V*E*F + b) * self.mask_corrected_AO
        MoffatPSD[self.nOtf_AO//2, self.nOtf_AO//2] *= 0.0
        return MoffatPSD
        '''


    def PSD2PSF(self, r0, L0, F, dx, dy, bg, amp, b, alpha, beta, ratio, theta):
    
        PSD = self.VonKarmanPSD(r0.abs(), L0.abs()) + \
            self.PSD_padder(self.MoffatPSD(amp, b.abs(), alpha, beta, ratio, theta))

        dk = 2*self.kc/self.nOtf_AO
        PSD *= (dk*self.wvl*1e9/2/np.pi)**2
        cov = 2*fft.fftshift(fft.fft2(fft.fftshift(PSD)))
        SF  = torch.abs(cov).max()-cov
        fftPhasor = torch.exp(-np.pi*1j*self.sampling_factor*(
            self.U.unsqueeze(0)*dx.unsqueeze(1).unsqueeze(2) +
            self.V.unsqueeze(0)*dy.unsqueeze(1).unsqueeze(2)))

        OTF_turb  = torch.exp(-0.5*SF*(2*np.pi*1e-9/self.wvl)**2)

        OTF = OTF_turb * self.OTF_static.unsqueeze(0) * fftPhasor
        PSF = torch.abs( fft.fftshift(fft.ifft2(fft.fftshift(OTF))) ).unsqueeze(0)
        PSF_out = interpolate(PSF, size=(self.nPix,self.nPix), mode='area').squeeze(0)

        if self.norm_regime == 'max':
            return PSF_out/torch.amax(PSF_out, dim=(1,2), keepdim=True) * F.unsqueeze(1).unsqueeze(2) + bg.unsqueeze(1).unsqueeze(2)
        elif self.norm_regime == 'sum':
            return PSF_out/PSF_out.sum(dim=(1,2), keepdim=True) * F.unsqueeze(1).unsqueeze(2) + bg.unsqueeze(1).unsqueeze(2)
        else:
            return PSF_out * F.unsqueeze(1).unsqueeze(2) + bg.unsqueeze(1).unsqueeze(2)

        '''        
        PSD = self.VonKarmanPSD(r0.abs(), L0.abs()) + \
            self.PSD_padder(self.MoffatPSD(amp, b.abs(), alpha, beta, ratio, theta))

        # Computing OTF from PSD
        dk = 2*self.kc/self.nOtf_AO
        PSD *= (dk*self.wvl*1e9/2/np.pi)**2
        cov = 2*fft.fftshift(fft.fft2(fft.fftshift(PSD)))
        SF  = torch.abs(cov).max()-cov
        fftPhasor = torch.exp(-np.pi*1j*self.sampling_factor*(self.U*dx+self.V*dy))
        OTF_turb  = torch.exp(-0.5*SF*(2*np.pi*1e-9/self.wvl)**2)

        OTF = OTF_turb * self.OTF_static * fftPhasor
        PSF = torch.abs( fft.fftshift(fft.ifft2(fft.fftshift(OTF))) ).unsqueeze(0).unsqueeze(0)
        PSF_out = interpolate(PSF, size=(self.nPix,self.nPix), mode='area').squeeze(0).squeeze(0)

        return PSF_out/PSF_out.sum() * F + bg
        '''

    def forward(self, x):
        inp = [x[:,i] for i in range(x.shape[1])]
        return self.PSD2PSF(*inp)
    #def forward(self, r0, L0, F, dx, dy, bg, amp, b, alpha, beta, ratio, theta):
        #return self.PSD2PSF(r0, L0, F, dx, dy, bg, amp, b, alpha, beta, ratio, theta)


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

#%%
psfao = PSFAO(config_file, data_test, 'CUDA')
psfao.norm_regime = 'max'
#psfao.norm_regime = 'sum'

r0    = torch.tensor([r0_new(data_test['r0'], psfao.wvl, 0.5e-6)], requires_grad=True, device=psfao.device)
L0    = torch.tensor([25.0], requires_grad=False, device=psfao.device) # Outer scale [m]
F     = torch.tensor([1.0],  requires_grad=True,  device=psfao.device)
dx    = torch.tensor([0.0],  requires_grad=True,  device=psfao.device)
dy    = torch.tensor([0.0],  requires_grad=True,  device=psfao.device)
bg    = torch.tensor([0.0],  requires_grad=True,  device=psfao.device)
amp   = torch.tensor([3.0],  requires_grad=True,  device=psfao.device) # Phase PSD Moffat amplitude [rad²]
b     = torch.tensor([1e-4], requires_grad=True,  device=psfao.device) # Phase PSD background [rad² m²]
alpha = torch.tensor([0.1],  requires_grad=True,  device=psfao.device) # Phase PSD Moffat alpha [1/m]
beta  = torch.tensor([1.6],  requires_grad=True,  device=psfao.device) # Phase PSD Moffat beta power law
ratio = torch.tensor([1.0],  requires_grad=True,  device=psfao.device) # Phase PSD Moffat ellipticity
theta = torch.tensor([0.0],  requires_grad=True,  device=psfao.device) # Phase PSD Moffat angle


rep = [2]
r0    = r0.repeat(rep)
L0    = L0.repeat(rep)
F     = F.repeat(rep)
dx    = dx.repeat(rep)
dy    = dy.repeat(rep)
bg    = bg.repeat(rep)
amp   = amp.repeat(rep)
b     = b.repeat(rep)
alpha = alpha.repeat(rep)
beta  = beta.repeat(rep)
ratio = ratio.repeat(rep)
theta = theta.repeat(rep)

parameters = [r0, L0, F, dx, dy, bg, amp, b, alpha, beta, ratio, theta]
x = torch.stack(parameters).T#.repeat([4,1])

el_croppo = slice(256//2-32, 256//2+32)
el_croppo = (0, el_croppo, el_croppo)

if    psfao.norm_regime == 'max': param = im.max()
elif  psfao.norm_regime == 'sum': param = im.sum()
else: param = 1.0
PSF_0 = torch.tensor(im/param, device=psfao.device).unsqueeze(0)

psfao.StartTimer()
#PSF_1 = psfao(x)
PSF_1 = psfao.PSD2PSF(*parameters)
print(psfao.EndTimer())

plt.imshow(torch.log( torch.hstack((PSF_0.abs()[el_croppo], PSF_1.abs()[el_croppo], ((PSF_1-PSF_0).abs()[el_croppo])) )).detach().cpu())
#%% ---------------------------------------------------------------------------
regime_opt = 'TRF'
#regime_opt = 'LBFGS'

#if regime_opt = 'LBFGS':
loss_fn = nn.L1Loss(reduction='sum')
optimizer_lbfgs = OptimizeLBFGS(psfao, parameters, loss_fn)

for i in range(20):
    optimizer_lbfgs.Optimize(PSF_0, [r0, F, dx, dy], 2)
    optimizer_lbfgs.Optimize(PSF_0, [bg], 2)
    optimizer_lbfgs.Optimize(PSF_0, [amp, alpha, beta], 5)
    optimizer_lbfgs.Optimize(PSF_0, [ratio, theta], 5)
    optimizer_lbfgs.Optimize(PSF_0, [b], 2)
PSF_1 = psfao.PSD2PSF(*parameters)

#%%
optimizer_trf = OptimizeTRF(psfao, parameters)
parameters = optimizer_trf.Optimize(PSF_0)
PSF_1 = psfao.PSD2PSF(*parameters)

#%%

plt.imshow(torch.log( torch.hstack((PSF_0.abs()[el_croppo], PSF_1.abs()[el_croppo], ((PSF_1-PSF_0).abs()[el_croppo])) )).detach().cpu())
plt.show()

plot_radial_profile(PSF_0.squeeze(0), PSF_1.squeeze(0), 'PSF AO', title='IRDIS PSF')
plt.show()

#la chignon et tarte

#%% =============================== MAKE DATASET ==========================================
### =======================================================================================
### =======================================================================================

# Load the SPHERE PSF database
path_fitted = 'C:/Users/akuznets/Data/SPHERE/fitted 4/'
path_input  = 'C:/Users/akuznets/Data/SPHERE/test/'

database = SPHERE_database(path_input, path_fitted)

# Filter bad samples
bad_samples = []
for sample in database:
    buf = np.array([sample['fitted']['r0'], sample['fitted']['F'],
                    sample['fitted']['dx'], sample['fitted']['dy'],
                    sample['fitted']['bg'],
                    sample['fitted']['amp'], sample['fitted']['b'],
                    sample['fitted']['alpha'], sample['fitted']['beta'],
                    sample['fitted']['ratio'], sample['fitted']['theta']])
    
    if np.any(np.isnan(buf)) or np.isnan(sample['input']['WFS']['Nph vis']):
       bad_samples.append(sample['file_id']) 

for bad_sample in bad_samples:
    database.remove(bad_sample)

print(str(len(bad_samples))+' samples were filtered, '+str(len(database.data))+' samples remained')


# %%
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


def GetLabels(sample):
    #r0_500 = r0_new(np.abs(data_sample['fitted']['r0']), 0.5e-6, data_sample['input']['spectrum']['lambda'])
    buf =  np.array([sample['fitted']['r0'],
                     25.0,
                     sample['fitted']['F'],
                     sample['fitted']['dx'],
                     sample['fitted']['dy'],
                     sample['fitted']['bg'],
                     sample['fitted']['amp'],
                     sample['fitted']['b'],
                     sample['fitted']['alpha'],
                     sample['fitted']['beta'],
                     sample['fitted']['ratio'],
                     sample['fitted']['theta']])
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
        x.append(torch.tensor(input, device=psfao.device).float())
        y.append(torch.tensor(pred,  device=psfao.device).float())
    
    if with_PSF:
        return torch.vstack(x), torch.dstack(y).permute([2,0,1])
    else:
        return torch.vstack(x), torch.vstack(y)

#%%

X,Y = GenerateDataset(database_wvl, with_PSF=True)

for i in range(Y.shape[0]):
    plt.plot(Y[i,128,:].detach().cpu())
plt.show()


#%%

testo = Y.sum(dim=0, keepdim=True)
plt.imshow(torch.log(testo).detach().cpu()[el_croppo])

#%%

validation_ids = np.unique(np.random.randint(0, high=len(database_wvl), size=30, dtype=int)).tolist()
database_train, database_val = database.split(validation_ids)

X_train, y_train = GenerateDataset(database_train, with_PSF=False)
X_val, y_val = GenerateDataset(database_val, with_PSF=False)

print(str(X_train.shape[0])+' samples in train dataset, '+str(X_val.shape[0])+' in validation')

# %%
class Gnosis(torch.nn.Module):
        def __init__(self, input_size, hidden_size, psf_model=None, device='cpu'):
            self.device = device
            super(Gnosis, self).__init__()
            self.input_size  = input_size
            self.hidden_size = hidden_size

            self.fc1  = torch.nn.Linear(self.input_size, self.hidden_size*2, device=self.device)
            self.act1 = torch.nn.Tanh()
            self.fc2  = torch.nn.Linear(self.hidden_size*2, hidden_size, device=self.device)
            self.act2 = torch.nn.Tanh()
            self.fc3  = torch.nn.Linear(self.hidden_size, 12, device=self.device)
  
            self.inp_normalizer = torch.ones(self.input_size, device=self.device)
            self.out_normalizer = torch.ones(12, device=self.device)
            self.inp_bias = torch.zeros(self.input_size, device=self.device)
            self.out_bias = torch.zeros(12, device=self.device)

            self.psf_model = psf_model


        def forward(self, x):
            hidden1 = self.fc1(x * self.inp_normalizer + self.inp_bias)
            act1 = self.act1(hidden1)
            hidden2 = self.fc2(act1)
            act2 = self.act2(hidden2)
            model_inp = self.fc3(act2) * self.out_normalizer + self.out_bias
            #model_inp = (self.fc3(act2)*0.0+1.0) * self.out_normalizer + self.out_bias
            if self.psf_model is None:
                return model_inp
            else:
                return self.psf_model(model_inp)


gnosis = Gnosis(input_size=9, hidden_size=200, device=psfao.device)
gnosis.inp_normalizer = torch.tensor([5, 50, 1/50, 1/360, 1, 0.5, 2, 2, 1e6], device=psfao.device).unsqueeze(0)
gnosis.inp_bias = torch.tensor([0, 0, 0, 0, -1, -3, 0, 0, 0],   device=psfao.device).unsqueeze(0)
gnosis.out_normalizer = 0.25/torch.tensor([0.5, 0.01, 0.25, 1, 1, 2e6, 0.125, 1e3, 1e2, 0.2, 0.5, 0.5], device=psfao.device).unsqueeze(0)


#%%
loss_fn = nn.L1Loss() #reduction='sum')
optimizer = optim.SGD([{'params': gnosis.fc1.parameters()},
                       {'params': gnosis.act1.parameters()},
                       {'params': gnosis.fc2.parameters()},
                       {'params': gnosis.act2.parameters()},
                       {'params': gnosis.fc3.parameters()}], lr=1e-4, momentum=0.9)

for i in range(40000):
    optimizer.zero_grad()
    #x,c = GetInputs(data, i)
    loss = loss_fn(gnosis(X_train), y_train)
    loss.backward()
    if i % 1000: print(loss.item())
    optimizer.step() # lambda: loss_fn(gnosis(X,C), y))

print('Validation accuracy: '+str(loss_fn(gnosis(X_val), y_val).item()))

#%%

def PSFcomparator(data_sample):
    psfao2 = PSFAO(config_file, data_sample['input'], 'CUDA')

    x_test, c_test = GetInputs(data_sample)
    x_test = torch.tensor(x_test, device=psfao2.device).float()
    c_test = torch.tensor(c_test, device=psfao2.device).float()
    y2 = gnosis(x_test)

    def ReturnPSFfromPred(y,c):
        r0_,L0_,F_,amp_,b_,alpha_,beta_,ratio_,theta_ = y.detach()
        dx_, dy_, bg_ = c.detach()
        r0_ = r0_new(r0_, psfao2.wvl, 0.5e-6)
        return psfao2(r0_,L0_,F_,dx_,dy_,bg_,amp_,b_,alpha_,beta_,ratio_,theta_)

    #PSF_1 = ReturnPSFfromPred(y1, c_test)
    PSF_2 = ReturnPSFfromPred(y2, c_test)

    A = torch.tensor(data_sample['input']['image'], device=psfao2.device)
    C = torch.tensor(data_sample['fitted']['Img. fit'], device=psfao2.device)
    PSF_0 = A/A.sum()
    PSF_1 = C/C.sum()


    return PSF_0, PSF_1, PSF_2

#%%

loss_fn = nn.L1Loss()

fit_diff = []
gnosis_diff = []

PSF_0s = []
PSF_1s = []
PSF_2s = []

profile_0s = []
profile_1s = []
profile_2s = []

for i in range(len(database_val)):
    data_sample = database_val[i]
    PSF_0, PSF_1, PSF_2 = PSFcomparator(data_sample)
    fit_diff.append(loss_fn(PSF_0, PSF_1).item())
    gnosis_diff.append(loss_fn(PSF_0, PSF_2).item())

    PSF_0s.append(PSF_0)
    PSF_1s.append(PSF_1)
    PSF_2s.append(PSF_2)

    profile_0s.append( radial_profile(PSF_0.detach().cpu().numpy())[:32] )
    profile_1s.append( radial_profile(PSF_1.detach().cpu().numpy())[:32] )
    profile_2s.append( radial_profile(PSF_2.detach().cpu().numpy())[:32] )

fit_diff = np.array(fit_diff)
gnosis_diff = np.array(gnosis_diff)

PSF_0s = torch.dstack(PSF_0s)
PSF_1s = torch.dstack(PSF_1s)
PSF_2s = torch.dstack(PSF_2s)

profile_0s = np.vstack(profile_0s)
profile_1s = np.vstack(profile_1s)
profile_2s = np.vstack(profile_2s)

c = profile_0s.mean(axis=0).max()

profile_0s /= c*0.01
profile_1s /= c*0.01
profile_2s /= c*0.01

# %%
fig = plt.figure(figsize=(6,4), dpi=150)
plt.grid()

def plot_std(x,y, label, color, style):
    y_m = y.mean(axis=0)
    y_s = y.std(axis=0)
    lower_bound = y_m-y_s
    upper_bound = y_m+y_s

    plt.fill_between(x, lower_bound, upper_bound, color=color, alpha=0.3)
    plt.plot(x, y_m, label=label, color=color, linestyle=style)

x = np.arange(32)
plot_std(x, np.abs(profile_0s-profile_1s), '$\Delta$ Fit', 'royalblue', '--')
plot_std(x, np.abs(profile_0s-profile_2s), '$\Delta$ Gnosis', 'darkgreen', ':')

plt.title('Accuracy comparison (avg. for validation dataset)')
plt.yscale('symlog')
plt.xlim([x.min(), x.max()])
plt.legend()
plt.ylabel('Abs. relative diff., [%]')
plt.xlabel('Pixels')

#%%
data_sample = database_val[i]
PSF_0, PSF_1, PSF_2 = PSFcomparator(data_sample)

profile_0 = radial_profile(PSF_0.detach().cpu().numpy())[:32]
profile_1 = radial_profile(PSF_1.detach().cpu().numpy())[:32]
profile_2 = radial_profile(PSF_2.detach().cpu().numpy())[:32]

profile_diff1 = np.abs(profile_1-profile_0) / profile_0.max() * 100 #[%]
profile_diff2 = np.abs(profile_2-profile_0) / profile_0.max() * 100 #[%]

fig = plt.figure(figsize=(6,4), dpi=150)
ax = fig.add_subplot(111)
ax.set_title('Fitting vs. Gnosis vs. Direct pred.')
l0 = ax.plot(profile_0, label='Data')
l1 = ax.plot(profile_1, label='Fit')
l2 = ax.plot(profile_2, label='Gnosis', color='g')
ax.set_xlabel('Pixels')
ax.set_ylabel('Relative intensity')
ax.set_yscale('log')
ax.set_xlim([0, len(profile_1)])
ax.grid()

ax2 = ax.twinx()
l3 = ax2.plot(profile_diff1, label='$\Delta$ Fit', color='tab:orange', linestyle='dashdot')
l4 = ax2.plot(profile_diff2, label='$\Delta$ Gnosis', color='g', linestyle=':')
ax2.set_ylim([0, max([profile_diff1.max(), profile_diff2.max()])*1.25])
ax2.set_ylabel('Difference [%]')

ls = l0+l1+l2+l3+l4
labels = [l.get_label() for l in ls]
ax2.legend(ls, labels, loc=0)

plt.show()