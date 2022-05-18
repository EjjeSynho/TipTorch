#%%
import torch
import numpy as np
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
from torch import fft
import numpy as np
import scipy.special as spc
from torch import fft
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

#%% ------------------------ Managing paths ------------------------
#path_test = 'C:\\Users\\akuznets\\Data\\SPHERE\\test\\33_SPHER.2017-03-05T05.00.21.009IRD_FLUX_CALIB_CORO_RAW_left.pickle'
#path_test = 'C:\\Users\\akuznets\\Data\\SPHERE\\test\\46_SPHER.2017-04-18T04.37.07.151IRD_FLUX_CALIB_CORO_RAW_left.pickle'
path_test = 'C:\\Users\\akuznets\\Data\\SPHERE\\test\\44_SPHER.2017-04-02T04.00.12.314IRD_FLUX_CALIB_CORO_RAW_left.pickle'

with open(path_test, 'rb') as handle:
    data_test = pickle.load(handle)

path_root = path.normpath('C:/Users/akuznets/Projects/TIPTOP/P3')
path_ini = path.join(path_root, path.normpath('aoSystem/parFiles/irdis.ini'))

parser = parameterParser(path_root, path_ini)
params = parser.params

im  = data_test['image']
wvl = data_test['spectrum']['lambda']

params['atmosphere']['Seeing']          = data_test['seeing']
params['atmosphere']['WindSpeed']       = [data_test['Wind speed']['header']]
params['atmosphere']['WindDirection']   = [data_test['Wind direction']['header']]
params['sources_science']['Wavelength'] = [wvl]
params['sensor_science']['FieldOfView'] = data_test['image'].shape[0]
params['sensor_science']['Zenith']      = [90.0-data_test['telescope']['altitude']]
params['sensor_science']['Azimuth']     = [data_test['telescope']['azimuth']]
params['sensor_science']['SigmaRON']    = data_test['Detector']['ron']
params['sensor_science']['Gain']        = data_test['Detector']['gain']

params['RTC']['SensorFrameRate_HO']  = data_test['WFS']['rate']
params['sensor_HO']['NumberPhotons'] = data_test['WFS']['Nph vis']


#%%
cuda  = torch.device('cuda') # Default CUDA device
#cuda  = torch.device('cpu')
start = torch.cuda.Event(enable_timing=True)
end   = torch.cuda.Event(enable_timing=True)

rad2mas = 3600 * 180 * 1000 / np.pi
rad2arc = rad2mas / 1000
deg2rad = np.pi / 180
asec2rad = np.pi / 180 / 3600

seeing = lambda rₒ, λ: rad2arc*0.976*λ/rₒ # [arcs]
rₒ = lambda seeing, λ: rad2arc*0.976*λ/seeing # [m]
rₒ_new = lambda rₒ, λ, λₒ: rₒ*(λ/λₒ)**1.2 # [m]

D       = params['telescope']['TelescopeDiameter']
wvl     = wvl
psInMas = params['sensor_science']['PixelScale'] #[mas]
nPix    = params['sensor_science']['FieldOfView']
pitch   = params['DM']['DmPitchs'][0] #[m]
h_DM    = params['DM']['DmHeights'][0] # ????? what is h_DM?
nDM     = 1
kc      = 1/(2*pitch) #TODO: kc is not consistent with vanilla TIPTOP

zenith_angle  = torch.tensor(params['telescope']['ZenithAngle'], device=cuda) # [deg]
airmass       = 1.0 / torch.cos(zenith_angle * deg2rad)

GS_wvl     = params['sources_HO']['Wavelength'][0] #[m]
GS_height  = params['sources_HO']['Height'] * airmass #[m]
GS_angle   = torch.tensor(params['sources_HO']['Zenith'],  device=cuda) / rad2arc
GS_azimuth = torch.tensor(params['sources_HO']['Azimuth'], device=cuda) * deg2rad
GS_dirs_x  = torch.tan(GS_angle) * torch.cos(GS_azimuth)
GS_dirs_y  = torch.tan(GS_angle) * torch.sin(GS_azimuth)
nGS = GS_dirs_y.size(0)

wind_speed  = torch.tensor(params['atmosphere']['WindSpeed'], device=cuda)
wind_dir    = torch.tensor(params['atmosphere']['WindDirection'], device=cuda)
Cn2_weights = torch.tensor(params['atmosphere']['Cn2Weights'], device=cuda)
Cn2_heights = torch.tensor(params['atmosphere']['Cn2Heights'], device=cuda) * airmass #[m]
stretch     = 1.0 / (1.0-Cn2_heights/GS_height)
h           = Cn2_heights * stretch
nL          = Cn2_heights.size(0)

WFS_d_sub = np.mean(params['sensor_HO']['SizeLenslets'])
WFS_n_sub = np.mean(params['sensor_HO']['NumberLenslets'])
WFS_det_clock_rate = np.mean(params['sensor_HO']['ClockRate']) # [(?)]
WFS_FOV = params['sensor_HO']['FieldOfView']
WFS_RON = params['sensor_HO']['SigmaRON']
WFS_psInMas = params['sensor_HO']['PixelScale']
WFS_Nph = torch.tensor(params['sensor_HO']['NumberPhotons'], device=cuda)
WFS_wvl = torch.tensor(640e-9, device=cuda) #TODO: clarify this
WFS_spot_FWHM = torch.tensor(params['sensor_HO']['SpotFWHM'][0], device=cuda)
WFS_excessive_factor = params['sensor_HO']['ExcessNoiseFactor']

HOloop_rate  = np.mean(params['RTC']['SensorFrameRate_HO']) # [Hz] (?)
HOloop_delay = params['RTC']['LoopDelaySteps_HO'] # [ms] (?)
HOloop_gain  = params['RTC']['LoopGain_HO']

#-------------------------------------------------------
pixels_per_l_D = wvl*rad2mas / (psInMas*D)
sampling_factor = int(np.ceil(2.0/pixels_per_l_D)) # check how much it is less than Nyquist
sampling = sampling_factor * pixels_per_l_D
nOtf = nPix * sampling_factor

dk = 1/D/sampling # PSD spatial frequency step
cte = (24*spc.gamma(6/5)/5)**(5/6)*(spc.gamma(11/6)**2/(2*np.pi**(11/3)))

#with torch.no_grad():
# Initialize spatial frequencies
kx, ky = torch.meshgrid(
    torch.linspace(-nOtf/2, nOtf/2-1, nOtf, device=cuda)*dk + 1e-10,
    torch.linspace(-nOtf/2, nOtf/2-1, nOtf, device=cuda)*dk + 1e-10,
    indexing = 'ij')

k2 = kx**2 + ky**2
k = torch.sqrt(k2)

mask = torch.ones_like(k2, device=cuda)
mask[k2 <= kc**2] = 0
mask_corrected = 1.0-mask

#with open('C:\\Users\\akuznets\\Desktop\\buf\\masks.pickle', 'rb') as handle:
#    masks = pickle.load(handle)

#mask = torch.tensor(masks['out'], device=cuda)
#mask_corrected = torch.tensor(masks['in'], device=cuda)
nOtf_AO = int(2*kc/dk)
nOtf_AO += nOtf_AO % 2

# Comb samples involved in antialising
n_times = min(4,max(2,int(np.ceil(nOtf/nOtf_AO/2))))
ids = []
for mi in range(-n_times,n_times):
    for ni in range(-n_times,n_times):
        if mi or ni: #exclude (0,0)
            ids.append([mi,ni])
ids = np.array(ids)

m = torch.tensor(ids[:,0], device=cuda)
n = torch.tensor(ids[:,1], device=cuda)
N_combs = m.shape[0]

corrected_ROI = slice(nOtf//2-nOtf_AO//2, nOtf//2+nOtf_AO//2)
corrected_ROI = (corrected_ROI,corrected_ROI)

#mask_AO = torch.tensor(masks['out_ao'], device=cuda)
#mask_corrected_AO = torch.tensor(masks['in_ao'], device=cuda)
mask_AO = mask[corrected_ROI]
mask_corrected_AO = mask_corrected[corrected_ROI]
mask_corrected_AO_1_1  = torch.unsqueeze(torch.unsqueeze(mask_corrected_AO,2),3)

kx_AO = kx[corrected_ROI]
ky_AO = ky[corrected_ROI]
k_AO  = k [corrected_ROI]
k2_AO = k2[corrected_ROI]

M_mask = torch.zeros([nOtf_AO,nOtf_AO,nGS,nGS], device=cuda)
for j in range(nGS):
    M_mask[:,:,j,j] += 1.0

# Matrix repetitions and dimensions expansion to avoid in in runtime
kx_1_1 = torch.unsqueeze(torch.unsqueeze(kx_AO,2),3)
ky_1_1 = torch.unsqueeze(torch.unsqueeze(ky_AO,2),3)
k_1_1  = torch.unsqueeze(torch.unsqueeze(k_AO, 2),3)

kx_nGs_nGs = kx_1_1.repeat([1,1,nGS,nGS]) * M_mask
ky_nGs_nGs = ky_1_1.repeat([1,1,nGS,nGS]) * M_mask
k_nGs_nGs  = k_1_1.repeat([1,1,nGS,nGS])  * M_mask
kx_nGs_nL  = kx_1_1.repeat([1,1,nGS,nL])
ky_nGs_nL  = ky_1_1.repeat([1,1,nGS,nL])
k_nGs_nL   = k_1_1.repeat([1,1,nGS,nL])
kx_1_nL    = kx_1_1.repeat([1,1,1,nL])
ky_1_nL    = ky_1_1.repeat([1,1,1,nL])

# For NGS-like alising 2nd dimension is used to store combs information
km = kx_1_1.repeat([1,1,N_combs,1]) - torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(m/WFS_d_sub,0),0),3)
kn = ky_1_1.repeat([1,1,N_combs,1]) - torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(n/WFS_d_sub,0),0),3)

noise_nGs_nGs = torch.normal(mean=torch.zeros([nOtf_AO,nOtf_AO,nGS,nGS]), std=torch.ones([nOtf_AO,nOtf_AO,nGS,nGS])*0.001) #TODO: remove noise, do proper matrix inversion
noise_nGs_nGs = noise_nGs_nGs.to(cuda)

GS_dirs_x_nGs_nL = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(GS_dirs_x,0),0),3).repeat([nOtf_AO,nOtf_AO,1,nL])# * dim_N_N_nGS_nL
GS_dirs_y_nGs_nL = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(GS_dirs_y,0),0),3).repeat([nOtf_AO,nOtf_AO,1,nL])# * dim_N_N_nGS_nL

# Initialize OTF frequencines
U,V = torch.meshgrid(
    torch.linspace(0, nOtf-1, nOtf, device=cuda),
    torch.linspace(0, nOtf-1, nOtf, device=cuda),
    indexing = 'ij')

U = (U-nOtf/2) * 2/nOtf
V = (V-nOtf/2) * 2/nOtf

U2  = U**2
V2  = V**2
UV  = U*V
UV2 = U**2 + V**2

pupil_path = params['telescope']['PathPupil']
pupil_apodizer = params['telescope']['PathApodizer']

pupil    = torch.tensor(fits.getdata(pupil_path).astype('float'), device=cuda)
apodizer = torch.tensor(fits.getdata(pupil_apodizer).astype('float'), device=cuda)

pupil_pix  = pupil.shape[0]
#padded_pix = nOtf
padded_pix = int(pupil_pix*sampling)

pupil_padded = torch.zeros([padded_pix, padded_pix], device=cuda)
pupil_padded[
    padded_pix//2-pupil_pix//2 : padded_pix//2+pupil_pix//2,
    padded_pix//2-pupil_pix//2 : padded_pix//2+pupil_pix//2
] = pupil*apodizer

def fftAutoCorr(x):
    x_fft = fft.fft2(x)
    return fft.fftshift( fft.ifft2(x_fft*torch.conj(x_fft))/x.size(0)*x.size(1) )

OTF_static = torch.real( fftAutoCorr(pupil_padded) ).unsqueeze(0).unsqueeze(0)
OTF_static = interpolate(OTF_static, size=(nOtf,nOtf), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
OTF_static = OTF_static / OTF_static.max()

PSD_padder = torch.nn.ZeroPad2d((nOtf-nOtf_AO)//2)

# Piston filter
def PistonFilter(f):
    x = (np.pi*D*f).cpu().numpy() #TODO: find Bessel analog for pytorch
    R = spc.j1(x)/x
    piston_filter = torch.tensor(1.0-4*R**2, device=cuda)
    piston_filter[nOtf_AO//2,nOtf_AO//2,...] *= 0.0
    return piston_filter

piston_filter = PistonFilter(k_AO)
PR = PistonFilter(torch.hypot(km,kn))


#%%
def Controller(nF=1000):
    #nTh = 1
    Ts = 1.0 / HOloop_rate # samplingTime
    delay = HOloop_delay #latency
    loopGain = HOloop_gain

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
    
    vx = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(wind_speed*torch.cos(wind_dir*np.pi/180.),0),0),0)
    vy = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(wind_speed*torch.sin(wind_dir*np.pi/180.),0),0),0)
    fi = -vx*kx_1_1*costh - vy*ky_1_1*costh
    _, _, atfInt, ntfInt = TransferFunctions(fi)

    # AO transfer function
    h1 = Cn2_weights * atfInt #/nTh
    h2 = Cn2_weights * abs(atfInt)**2 #/nTh
    hn = Cn2_weights * abs(ntfInt)**2 #/nTh

    h1 = torch.sum(h1,axis=(2,3))
    h2 = torch.sum(h2,axis=(2,3))
    hn = torch.sum(hn,axis=(2,3))
    return h1, h2, hn, noise_gain


def ReconstructionFilter(r0, L0, WFS_noise_var):
    Av = torch.sinc(WFS_d_sub*kx_AO)*torch.sinc(WFS_d_sub*ky_AO) * torch.exp(1j*np.pi*WFS_d_sub*(kx_AO+ky_AO))
    SxAv = 2j*np.pi*kx_AO*WFS_d_sub*Av
    SyAv = 2j*np.pi*ky_AO*WFS_d_sub*Av

    MV = 0
    Wn = WFS_noise_var/(2*kc)**2

    W_atm = cte*r0**(-5/3)*(k2_AO + 1/L0**2)**(-11/6)*(wvl/GS_wvl)**2
    gPSD = torch.abs(SxAv)**2 + torch.abs(SyAv)**2 + MV*Wn/W_atm
    Rx = torch.conj(SxAv) / gPSD
    Ry = torch.conj(SyAv) / gPSD
    Rx[nOtf_AO//2, nOtf_AO//2] *= 0
    Ry[nOtf_AO//2, nOtf_AO//2] *= 0
    return Rx, Ry, SxAv, SyAv, W_atm


def SpatioTemporalPSD(Rx, Ry, SxAv, SyAv, h1, h2, W_atm):
    A = torch.ones([nOtf_AO, nOtf_AO], device=cuda) #TODO: fix it. A should be initialized differently
    Ff = Rx*SxAv + Ry*SyAv
    psd_ST = (1+abs(Ff)**2 * h2 - 2*torch.real(Ff*h1*A)) * W_atm * mask_corrected_AO
    return psd_ST


def NoisePSD(Rx, Ry, noise_gain, WFS_noise_var):
    noisePSD = abs(Rx**2 + Ry**2) /(2*kc)**2
    noisePSD = noisePSD* piston_filter * noise_gain * WFS_noise_var * mask_corrected_AO
    return noisePSD


def AliasingPSD(Rx, Ry, h1, r0, L0):
    T = WFS_det_clock_rate / HOloop_rate
    td = T * HOloop_delay
    vx = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(wind_speed*torch.cos(wind_dir*np.pi/180.),0),0),0)
    vy = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(wind_speed*torch.sin(wind_dir*np.pi/180.),0),0),0)

    Rx1 = torch.unsqueeze(torch.unsqueeze(2j*np.pi*WFS_d_sub * Rx,2),3)
    Ry1 = torch.unsqueeze(torch.unsqueeze(2j*np.pi*WFS_d_sub * Ry,2),3)

    W_mn = (km**2 + kn**2 + 1/L0**2)**(-11/6)
    Q = (Rx1*km + Ry1*kn) * torch.sinc(WFS_d_sub*km) * torch.sinc(WFS_d_sub*kn)

    tf = torch.unsqueeze(torch.unsqueeze(h1,2),3)

    avr = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(Cn2_weights,0),0),0) * \
        (torch.sinc(km*vx*T) * torch.sinc(kn*vy*T) * \
        torch.exp(2j*np.pi*km*vx*td) * torch.exp(2j*np.pi*kn*vy*td) * tf.repeat([1,1,N_combs,nL]))

    aliasing_PSD = torch.sum(PR*W_mn*abs(Q*avr.sum(axis=3,keepdim=True))**2, axis=(2,3))*cte*r0**(-5/3) * mask_corrected_AO
    return aliasing_PSD


def VonKarmanPSD(r0, L0):
    return cte*r0**(-5/3)*(k2 + 1/L0**2)**(-11/6) * mask


def ChromatismPSD(r0, L0):
    wvlRef = wvl #TODO: polychromatic support
    W_atm = r0**(-5/3)*cte*(k2_AO + 1/L0**2)**(-11/6) * piston_filter #TODO: W_phi and vK spectrum
    IOR = lambda lmbd: 23.7+6839.4/(130-(lmbd*1.e6)**(-2))+45.47/(38.9-(lmbd*1.e6)**(-2))
    n2 = IOR(GS_wvl)
    n1 = IOR(wvlRef)
    chromatic_PSD = ((n2-n1)/n2)**2 * W_atm
    return chromatic_PSD


def JitterCore(Jx, Jy, Jxy):
    u_max = sampling*D/wvl/(3600*180*1e3/np.pi)
    norm_fact = u_max**2 * (2*np.sqrt(2*np.log(2)))**2
    Djitter = norm_fact * (Jx**2 * U2 + Jy**2 * V2 + 2*Jxy*UV)
    return torch.exp(-0.5*Djitter) #TODO: cover Nyquist sampled case


def NoiseVariance(r0): #TODO: do input of actual r0 and rescale it inside
    WFS_nPix = WFS_FOV / WFS_n_sub
    WFS_pixelScale = WFS_psInMas / 1e3 # [asec]

    # Read-out noise calculation
    nD = torch.tensor([1.0, rad2arc*wvl/WFS_d_sub/WFS_pixelScale]).max() #spot FWHM in pixels and without turbulence
    varRON = np.pi**2/3 * (WFS_RON**2/WFS_Nph**2) * (WFS_nPix**2/nD)**2
    # Photon-noise calculation
    nT = torch.tensor([1.0, torch.hypot(WFS_spot_FWHM.max()/1e3, rad2arc*WFS_wvl/r0) / WFS_pixelScale], device=cuda).max()
    varShot = np.pi**2/(2*WFS_Nph) * (nT/nD)**2
    # Noise variance calculation
    varNoise = WFS_excessive_factor * (varRON+varShot)
    return varNoise


#%% -------------------------------------------------------------

def DLPSF():
    PSF = torch.abs( fft.fftshift(fft.ifft2(fft.fftshift(OTF_static))) ).unsqueeze(0).unsqueeze(0)
    PSF_out = interpolate(PSF, size=(nPix,nPix), mode='area').squeeze(0).squeeze(0)
    return (PSF_out/PSF_out.sum()) #* 1e2


def PSD2PSF(r0, L0, F, dx, dy, bg, WFS_noise_var, Jx, Jy, Jxy):
    # non-negative reparametrization
    r0  = torch.abs(r0)
    L0  = torch.abs(L0)
    Jx  = torch.abs(Jx)
    Jy  = torch.abs(Jy)
    Jxy = torch.abs(Jxy)
    WFS_noise_var = torch.abs(WFS_noise_var)

    h1, h2, _, noise_gain = Controller()
    Rx, Ry, SxAv, SyAv, W_atm = ReconstructionFilter(r0, L0, WFS_noise_var)

    PSD =  VonKarmanPSD(r0,L0) + \
    PSD_padder(
        NoisePSD(Rx, Ry, noise_gain, WFS_noise_var) + \
        SpatioTemporalPSD(Rx, Ry, SxAv, SyAv, h1, h2, W_atm) + \
        AliasingPSD(Rx, Ry, h1, r0, L0) + \
        ChromatismPSD(r0, L0)
    )

    dk = 2*kc/nOtf_AO
    PSD *= (dk*wvl*1e9/2/np.pi)**2
    cov = 2*fft.fftshift(fft.fft2(fft.fftshift(PSD)))
    SF  = torch.abs(cov).max()-cov
    fftPhasor = torch.exp(-np.pi*1j*sampling_factor*(U*dx+V*dy))
    OTF_turb  = torch.exp(-0.5*SF*(2*np.pi*1e-9/wvl)**2)

    OTF = OTF_turb * OTF_static * fftPhasor * JitterCore(Jx,Jy,Jxy)
    PSF = torch.abs( fft.fftshift(fft.ifft2(fft.fftshift(OTF))) ).unsqueeze(0).unsqueeze(0)
    PSF_out = interpolate(PSF, size=(nPix,nPix), mode='area').squeeze(0).squeeze(0)
    return (PSF_out/PSF_out.sum() * F + bg) #* 1e2

el_croppo = slice(256//2-32, 256//2+32)
el_croppo = (el_croppo,el_croppo)

def BackgroundEstimate(im, radius=90):
    buf_x, buf_y = torch.meshgrid(
        torch.linspace(-nPix//2, nPix//2, nPix, device=cuda),
        torch.linspace(-nPix//2, nPix//2, nPix, device=cuda),
        indexing = 'ij'
    )
    mask_noise = buf_x**2 + buf_y**2
    mask_noise[mask_noise < radius**2] = 0.0
    mask_noise[mask_noise > 0.0] = 1.0
    return torch.median(im[mask_noise>0.]).data

PSF_0 = torch.tensor(im/im.sum(), device=cuda)
plt.imshow(torch.log(PSF_0[el_croppo]).detach().cpu())
plt.show()

def Center(im):
    center = np.array(np.unravel_index(im.argmax().item(), im.shape))-np.array(im.shape)//2
    return center

#im = PSF_0
#WoG_ROI = 16
#center = np.array(np.unravel_index(im.argmax().item(), im.shape))
#crop = slice(im.shape[0]//2-WoG_ROI//2, im.shape[0]//2+WoG_ROI//2)
#crop = (crop, crop)
#buf = PSF_0[crop].detach().cpu().numpy()
#WoG = center_of_mass(PSF_0[crop].detach().cpu())
#plt.imshow(PSF_0[crop].detach().cpu())

r0_scaled = rₒ_new(data_test['r0'], wvl, 0.5e-6)
r0_scaled_WFS = rₒ_new(data_test['r0'], 0.64e-6, 0.5e-6)
noise_var = NoiseVariance(torch.tensor(r0_scaled_WFS, device=cuda)).clip(0.1, 0.3).item()

r0  = torch.tensor(r0_scaled,  requires_grad=True,  device=cuda)
L0  = torch.tensor(25.0, requires_grad=False, device=cuda)
F   = torch.tensor(1.0,  requires_grad=True,  device=cuda)
dx  = torch.tensor(0.0,  requires_grad=True,  device=cuda)
dy  = torch.tensor(0.0,  requires_grad=True,  device=cuda)
bg  = torch.tensor(0.0,  requires_grad=True,  device=cuda)
n   = torch.tensor(noise_var, requires_grad=True, device=cuda)
Jx  = torch.tensor(10.0, requires_grad=True,  device=cuda)
Jy  = torch.tensor(10.0, requires_grad=True,  device=cuda)
Jxy = torch.tensor(2.0, requires_grad=True,  device=cuda)

PSF_DL = DLPSF()

#%%
def OptimParams(loss_fun, params, iterations, method='LBFGS', verbous=True):
    if method == 'LBFGS':
        optimizer = optim.LBFGS(params, lr=10, history_size=20, max_iter=4, line_search_fn="strong_wolfe")
    elif method == 'Adam':
        optimizer = optim.Adam(params, lr=1e-2)

    history = []
    for i in range(iterations):
        optimizer.zero_grad()
        loss = loss_fun( PSD2PSF(r0, L0, F, dx, dy, bg, n, Jx, Jy, Jxy), PSF_0 )
        loss.backward()
        if verbous:
            if method == 'LBFGS':
                print('Loss:', loss.item(), end="\r")
            elif method == 'Adam':
                if not i % 10: print('Loss:', loss.item(), end="\r")

        history.append(loss.item())
        if len(history) > 2:
            if np.abs(loss.item()-history[-1]) < 1e-4 and np.abs(loss.item()-history[-2]) < 1e-4:
                break
        if method == 'LBFGS':
            optimizer.step( lambda: loss_fun( PSD2PSF(r0, L0, F, dx, dy, bg, n, Jx, Jy, Jxy), PSF_0 ) )
        elif method == 'Adam':
            optimizer.step()

loss_fn = nn.L1Loss(reduction='sum')
for i in range(20):
    OptimParams(loss_fn, [F, dx, dy], 5)
    OptimParams(loss_fn, [r0], 5)
    OptimParams(loss_fn, [n], 5)
    OptimParams(loss_fn, [bg], 2)
    OptimParams(loss_fn, [Jx, Jy, Jxy], 3)

print("r0,L0: ({:.3f}, {:.2f})".format(r0.data.item(), L0.data.item()))
print("I,bg:  ({:.3f}, {:.1E} )".format(F.data.item(), bg.data.item()))
print("dx,dy: ({:.2f}, {:.2f})".format(dx.data.item(), dy.data.item()))
print("Jx,Jy, Jxy: ({:.1f}, {:.1f}, {:.1f})".format(Jx.data.item(), Jy.data.item(), Jxy.data.item()))
print("WFS noise: {:.2f}".format(n.data.item()))

PSF_1 = PSD2PSF(r0, L0, F, dx, dy, bg, n, Jx, Jy, Jxy)
SR = PSF_1.max()/PSF_DL.max() * PSF_DL.sum()/PSF_1.sum()

#%%

def FitGauss2D(PSF):
    nPix_crop = 16
    crop = slice(nPix//2-nPix_crop//2, nPix//2+nPix_crop//2)
    PSF_cropped = torch.tensor(PSF[crop,crop], requires_grad=False, device=cuda)
    PSF_cropped = PSF_cropped / PSF_cropped.max()

    px, py = torch.meshgrid(
        torch.linspace(-nPix_crop/2, nPix_crop/2-1, nPix_crop, device=cuda),
        torch.linspace(-nPix_crop/2, nPix_crop/2-1, nPix_crop, device=cuda),
        indexing = 'ij')

    def Gauss2D(X):
        return X[0]*torch.exp( -((px-X[1])/(2*X[3]))**2 - ((py-X[2])/(2*X[4]))**2 )

    X0 = torch.tensor([1.0, 0.0, 0.0, 1.1, 1.1], requires_grad=True, device=cuda)

    loss_fn = nn.MSELoss()
    optimizer = optim.LBFGS([X0], history_size=10, max_iter=4, line_search_fn="strong_wolfe")

    for _ in range(20):
        optimizer.zero_grad()
        loss = loss_fn(Gauss2D(X0), PSF_cropped)
        loss.backward()
        optimizer.step(lambda: loss_fn(Gauss2D(X0), PSF_cropped))

    FWHM = lambda x: 2*np.sqrt(2*np.log(2)) * np.abs(x)

    plt.imshow(Gauss2D(X0).detach().cpu())
    plt.show()
    plt.imshow(PSF_cropped.detach().cpu())
    plt.show()

    return FWHM(X0[3].cpu().data), FWHM(X0[4].cpu().data)


#%%
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

#%% ==================================================================================================================================
#=====================================================================================================================================
#=====================================================================================================================================
#=====================================================================================================================================
#=====================================================================================================================================
#=====================================================================================================================================
#=====================================================================================================================================
#=====================================================================================================================================
#=====================================================================================================================================
#=====================================================================================================================================
#=====================================================================================================================================
#%% ==================================================================================================================================
import torch.nn.functional as F

X = torch.tensor([.5, 30], requires_grad=True)
PSD = cte*X[0]**(-5/3)*(k2 + 1/X[1]**2)**(-11/6)*mask

#Q = fft.fftshift(fft.fft2(fft.fftshift(PSD)))
PSD = torch.unsqueeze(torch.unsqueeze(PSD, 0), 0)
Q1 = F.conv2d(PSD, PSD, bias=None, stride=1, padding='same') / spc.factorial(0)
Q2 = F.conv2d(Q1,  PSD, bias=None, stride=1, padding='same') / spc.factorial(1)
Q3 = F.conv2d(Q2,  PSD, bias=None, stride=1, padding='same') / spc.factorial(2)
Q4 = F.conv2d(Q3,  PSD, bias=None, stride=1, padding='same') / spc.factorial(3)
Q  = F.conv2d(Q4,  PSD, bias=None, stride=1, padding='same') / spc.factorial(4)

external_grad = torch.ones([1,1,256,256])*1e-3 #, dtype=torch.complex128)
Q.backward(gradient=external_grad)

print(X.grad)

#%%
'''
nput = torch.randn(16, 16, dtype=torch.double,requires_grad=True)
test = gradcheck(fft.fft2(input), input, eps=1e-6, atol=1e-4)

print(test)
'''

'''
x = torch.randn((1, 1), requires_grad=True)
with torch.autograd.profiler.profile() as prof:
    for _ in range(100):  # any normal python code, really!
        y = x ** 2
        y.backward()
# NOTE: some columns were removed for brevity
print(prof.key_averages().table(sort_by="self_cpu_time_total"))
'''

#%%

'''
PSD = x[0]**(-5/3)*cte*((k2_new)**2 + 1/x[1]**2)**(-11/6)
PSD = PSD.detach().cpu().numpy()

if PSD.shape[0] % 2 == 0:
    PSD = np.vstack([PSD, np.zeros([1,PSD.shape[1]])])
    PSD = np.hstack([PSD, np.zeros([PSD.shape[0],1])])

norm1 = PSD.sum()
norm2 = 1./np.prod(1./spc.factorial(np.arange(N_ord)))

PSD = cp.array(PSD, dtype=cp.float64) / norm1
PSF = cp.copy(PSD)

#%
#start = time.time()
for n in range(0,N_ord):
    PSF = convolve2d(PSF, PSD, mode='same', boundary='symm') / spc.factorial(n)
PSF = PSF*norm1*norm2 / (nOtf**2)
#end = time.time()
#print(end - start)

print(PSF.max())

PSF_conv = cp.asnumpy( PSF / PSF.max() )

#PSF_ref = AtmoPSF(torch.tensor([r0, L0, 1., 0., 0.], device='cuda')).detach().cpu().numpy()

center1 = np.unravel_index(np.argmax(PSF_ref),  PSF_ref.shape)
center2 = np.unravel_index(np.argmax(PSF_conv), PSF_conv.shape)

plt.plot( PSF_ref [center1[0],center1[1]:], label='FFT' )
plt.plot( PSF_conv[center2[0],center2[1]:], label='Conv' )
plt.legend()
plt.grid()
'''

'''
import torch.nn.functional as F

PSD = x[0]**(-5/3)*cte*(k2_new/4.75 + 1/x[1]**2)**(-11/6)
PSD = PSD.to(device='cuda').double()

if PSD.size(0) % 2 == 0:
    PSD = PSD[1:,1:]

norm1 = PSD.sum()
norm2 = 1./np.prod(1./spc.factorial(np.arange(N_ord)))

PSD = torch.unsqueeze(torch.unsqueeze(PSD / norm1, 0), 0)
PSF = PSD.detach().clone()

#import time
#a = time.time()
for n in range(0,N_ord):
    PSF = F.conv2d(PSF, PSD, bias=None, stride=1, padding='same') / spc.factorial(N_ord)
PSF /= PSF.max()
PSF_conv = PSF[0,0,:,:].detach().cpu().numpy()
#b = time.time()
center1 = np.unravel_index(np.argmax(PSF_ref),  PSF_ref.shape)
center2 = np.unravel_index(np.argmax(PSF_conv), PSF_conv.shape)

plt.plot( PSF_ref [center1[0],center1[1]:], label='FFT' )
plt.plot( PSF_conv[center2[0],center2[1]:], label='Conv' )
plt.legend()
plt.grid()
'''

#%% --------------------------------------------

import torch
import torch.optim as optim
import matplotlib.pyplot as plt

def f(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

# Gradient descent
x_gd = 10*torch.ones(2, 1)
x_gd.requires_grad = True

optimizer = optim.SGD([x_gd], lr=1e-5)
h_gd = []
for i in range(100):
    optimizer.zero_grad()
    objective = f(x_gd)
    objective.backward()
    optimizer.step()
    h_gd.append(objective.item())

# L-BFGS
x_lbfgs = 10*torch.ones(2, 1)
x_lbfgs.requires_grad = True

optimizer = optim.LBFGS([x_lbfgs],
                        history_size=10,
                        max_iter=4,
                        line_search_fn="strong_wolfe")
h_lbfgs = []
for i in range(100):
    optimizer.zero_grad()
    objective = f(x_lbfgs)
    objective.backward()
    optimizer.step(lambda: f(x_lbfgs))
    h_lbfgs.append(objective.item())


# Plotting
plt.semilogy(h_gd, label='GD')
plt.semilogy(h_lbfgs, label='L-BFGS')
plt.legend()
plt.show()

#%%
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

cuda = torch.device('cuda')

def f(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

# L-BFGS
x_lbfgs = 10*torch.ones(2, 1, device=cuda)
x_lbfgs.requires_grad = True

optimizer = optim.LBFGS([x_lbfgs],
                        history_size=10,
                        max_iter=4,
                        line_search_fn="strong_wolfe")
h_lbfgs = []
for i in range(100):
    optimizer.zero_grad()
    objective = f(x_lbfgs)
    objective.backward()
    optimizer.step(lambda: f(x_lbfgs))
    #print(objective.item())
    h_lbfgs.append(objective.item())


# Plotting
plt.semilogy(h_lbfgs, label='L-BFGS')
plt.legend()
plt.show()

#%%

"""
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class Gauss(Module):
    params: Tensor

    def __init__(self)-> None: #, device=None, dtype=None) -> None:
        #factory_kwargs = {'device': device, 'dtype': dtype}
        super(Gauss, self).__init__()
        #if in_params is None:
        self.params = Parameter(torch.Tensor([1.0, 0.0, 0.0, 1.0]))
        #else:
        #    self.params = Parameter(in_params)

    def forward(self, xx, yy):
        return self.params[0]*torch.exp(-((xx-self.params[1]).pow(2)+(yy-self.params[2]).pow(2)) / (2*self.params[3].pow(2)))


device = torch.cuda.device(0)
N = 20
with torch.no_grad():
    xx, yy = torch.meshgrid(torch.linspace(-N, N-1, N*2), torch.linspace(-N, N-1, N*2))

model2 = Gauss()

def Gauss2D(X):
    return X[0]*torch.exp(-((xx-X[1]).pow(2) + (yy-X[2]).pow(2)) / (2*X[3].pow(2)))
X0 = torch.tensor([2.0, 1.0, -2.0, 2.0], requires_grad=False)

noise = AddGaussianNoise(0., 0.1)
b = noise(Gauss2D(X0))
#b = Gauss2D(X0)

plt.imshow(b)
plt.show()

niter = 10
loss_fn = nn.MSELoss()
optimizer = optim.LBFGS(model2.parameters(),
                        history_size=10,
                        max_iter=4,
                        line_search_fn="strong_wolfe")

for _ in range(0, niter):
    optimizer.zero_grad()
    loss = loss_fn(model2(xx, yy), b)
    loss.backward()
    optimizer.step(lambda: loss_fn(model2(xx, yy), b))

print(loss.data)

plt.imshow(model2(xx,yy).detach().numpy())
plt.show()

for i in model2.parameters():
    print(i)
"""


#%%

import numpy as np
from scipy.optimize import least_squares
from psfFitting.confidenceInterval import confidence_interval

# Initialize spatial frequencies
kx, ky = np.meshgrid(
    np.linspace(-nOtf/2-1, nOtf/2, nOtf)*dk,
    np.linspace(-nOtf/2-1, nOtf/2, nOtf)*dk)
k2 = kx**2 + ky**2

mask = np.ones_like(k2)
mask[nOtf//2, nOtf//2] = 0

#la chignon et tarte
U,V = np.meshgrid(
    np.linspace(-1, 1-2/nOtf, nOtf),
    np.linspace(-1, 1-2/nOtf, nOtf),
    indexing = 'ij')

U2 = U**2
V2 = V**2
UV = U*V
UV2 = U**2 + V**2

#print(V.min(), V.max())

#%%
#la chignon et tarte

from skimage.transform import resize

X0 = np.array([.15, 20., 1.0, 20.0, 0.0])
X1 = np.array([.50, 20., 1.0, 0.0, 0.0])
def PSD2PSF(X):
    PSD = cte*X[0]**(-5/3)*(k2 + 1/X[1]**2)**(-11/6)*mask
    cov = 2*np.fft.fftshift(np.fft.fft2(np.fft.fftshift(PSD)))
    SF  = np.real(cov).max()-cov
    
    fftPhasor = np.exp(-sampling_factor*np.pi*1j*(U*X[3]+V*X[4]))
    OTF_turb = np.exp(-0.5*SF*(2*np.pi*1e-9/wvl)**2) * fftPhasor   
    PSF = np.real( np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(OTF_turb))) )
    PSF = resize(PSF, (nPix,nPix), anti_aliasing=False)

    return PSF * 1e4

PSF_0 = PSD2PSF(X0) #reference
PSF_1 = PSD2PSF(X1)

#%%

im_norm = PSF_0

# Defining the cost functions
class CostClass(object):
    def __init__(self, alpha_positivity=None):
        self.iter = 0
        self.alpha_positivity = alpha_positivity


    def __call__(self, y):
        self.iter += 1
        im_est = PSD2PSF(y)
        return (im_norm - im_est).reshape(-1)

cost = CostClass()

result = least_squares(cost, X1,
                       method = 'trf', ftol=1e-8, xtol=1e-8, gtol=1e-8,
                       max_nfev = 1000, verbose = 1, loss = "linear")
                               
print(result.x)

#la chignon et tarte
