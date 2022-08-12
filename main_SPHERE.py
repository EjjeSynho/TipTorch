#%%
import matplotlib
import torch
from torch import nn, optim, fft
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

#%% ------------------------ Managing paths ------------------------
#Cut
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
#Cut

#%%
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

seeing = lambda r0, lmbd: rad2arc*0.976*lmbd/r0 # [arcs]
r0 = lambda seeing, lmbd: rad2arc*0.976*lmbd/seeing # [m]
r0_new = lambda r0, lmbd, lmbd0: r0*(lmbd/lmbd0)**1.2 # [m]

D       = params['telescope']['TelescopeDiameter']
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
WFS_wvl = torch.tensor(GS_wvl, device=cuda) #TODO: clarify this
WFS_spot_FWHM = torch.tensor(params['sensor_HO']['SpotFWHM'][0], device=cuda)
WFS_excessive_factor = params['sensor_HO']['ExcessNoiseFactor']

if np.isnan(params['sensor_HO']['NumberPhotons']):
    el_croppo = (slice(256//2-32, 256//2+32), slice(256//2-32, 256//2+32))
    WFS_Nph = torch.tensor(im.sum() / 40**2 / 10, device=cuda)
else:
    WFS_Nph = torch.tensor(params['sensor_HO']['NumberPhotons'], device=cuda)

HOloop_rate  = np.mean(params['RTC']['SensorFrameRate_HO']) # [Hz] (?)
HOloop_delay = params['RTC']['LoopDelaySteps_HO'] # [ms] (?)
HOloop_gain  = params['RTC']['LoopGain_HO']

if np.isnan( np.mean(params['RTC']['SensorFrameRate_HO']) ):
    HOloop_rate = 1380
if np.isnan( params['RTC']['LoopDelaySteps_HO'] ):
    HOloop_delay = 3
if np.isnan( params['RTC']['LoopGain_HO'] ):
    HOloop_gain = 0.5

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

# Matrix repetitions and dimensions expansion to avoid in runtime
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
    WFS_pixelScale = WFS_psInMas / 1e3 # [arcsec]
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
    return (PSF_out/PSF_out.sum())


def PSD2PSF(r0, L0, F, dx, dy, bg, WFS_noise_var, Jx, Jy, Jxy):
    # non-negative reparametrization
    r0  = torch.abs(r0)
    L0  = torch.abs(L0)
    Jx  = torch.abs(Jx)
    Jy  = torch.abs(Jy)
    Jxy = torch.abs(Jxy)

    #WFS_noise_var = torch.abs(WFS_noise_var)
    WFS_noise_var2 = ( WFS_noise_var + NoiseVariance(r0_new(r0, GS_wvl, wvl)) ).abs()

    h1, h2, _, noise_gain = Controller()
    Rx, Ry, SxAv, SyAv, W_atm = ReconstructionFilter(r0, L0, WFS_noise_var2)

    PSD =  VonKarmanPSD(r0,L0) + \
    PSD_padder(
        NoisePSD(Rx, Ry, noise_gain, WFS_noise_var2) + \
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

param = im.sum()
PSF_0 = torch.tensor(im/param, device=cuda)
#Cut
plt.imshow(torch.log(PSF_0[el_croppo]).detach().cpu())
plt.show()
#Cut

def Center(im):
    WoG_ROI = 16
    center = np.array(np.unravel_index(im.argmax().item(), im.shape))
    crop = slice(center[0]-WoG_ROI//2, center[1]+WoG_ROI//2)
    crop = (crop, crop)
    buf = PSF_0[crop].detach().cpu().numpy()
    WoG = np.array(scipy.ndimage.center_of_mass(buf)) + im.shape[0]//2-WoG_ROI//2
    return WoG-np.array(im.shape)//2

r0_scaled = r0_new(data_test['r0'], wvl, 0.5e-6) #* (1+0.25*(np.random.random_sample()-0.5))
noise_var = NoiseVariance(torch.tensor(r0_new(data_test['r0'], GS_wvl, 0.5e-6), device=cuda)).item()#.clip(0.05, 1.0).item()

dx_0, dy_0 = Center(PSF_0)
r0  = torch.tensor(r0_scaled, requires_grad=True,  device=cuda)
L0  = torch.tensor(25.0, requires_grad=False, device=cuda)
F   = torch.tensor(1.0,  requires_grad=True,  device=cuda)
dx  = torch.tensor(dx_0, requires_grad=True,  device=cuda)
dy  = torch.tensor(dy_0, requires_grad=True,  device=cuda)
bg  = torch.tensor(0.0,  requires_grad=True,  device=cuda)
n   = torch.tensor(0.0,  requires_grad=True, device=cuda)
Jx  = torch.tensor(10.0, requires_grad=True,  device=cuda)
Jy  = torch.tensor(10.0, requires_grad=True,  device=cuda)
Jxy = torch.tensor(2.0,  requires_grad=True,  device=cuda)
PSF_1 = PSD2PSF(r0, L0, F, dx, dy, bg, n, Jx, Jy, Jxy)
#Cut
plt.imshow(torch.log( torch.hstack((PSF_0[el_croppo], PSF_1[el_croppo], ((PSF_1-PSF_0).abs()[el_croppo])) )).detach().cpu())
#Cut

PSF_DL = DLPSF()

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
n_result = (n + NoiseVariance(r0_new(r0, GS_wvl, wvl)) ).abs().data.item()
n_init = NoiseVariance(torch.tensor(r0_new(data_test['r0'], GS_wvl, 0.5e-6), device=cuda)).item()

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
