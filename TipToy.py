#%%
import torch
from torch import nn, optim, fft
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.special as spc
from scipy.ndimage import center_of_mass
from torch.nn.functional import interpolate
from astropy.io import fits
from skimage.transform import resize
import torch
import pickle
from os import path
from parameterParser import parameterParser
import sys

#%% ------------------------ Managing paths ------------------------
path_test = sys.argv[1]
#path_test = 'C:\\Users\\akuznets\\Data\\SPHERE\\test\\0_SPHER.2016-08-27T23.59.07.572IRD_FLUX_CALIB_CORO_RAW_left.pickle'
#path_test = 'C:\\Users\\akuznets\\Data\\SPHERE\\test\\102_SPHER.2017-06-17T00.24.09.582IRD_FLUX_CALIB_CORO_RAW_left.pickle'

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

param = im.sum()
PSF_0 = torch.tensor(im/param, device=cuda, dtype=torch.float32)

#plt.imshow(torch.log(PSF_0[el_croppo]).detach().cpu())
#plt.show()

def Center(im):
    WoG_ROI = 16
    center = np.array(np.unravel_index(im.argmax().item(), im.shape))
    crop = slice(center[0]-WoG_ROI//2, center[1]+WoG_ROI//2)
    crop = (crop, crop)
    buf = im[crop].detach().cpu().numpy()
    WoG = np.array(center_of_mass(buf)) + im.shape[0]//2-WoG_ROI//2
    return (WoG-np.array(im.shape)//2).astype('float')

r0_scaled = r0_new(data_test['r0'], wvl, 0.5e-6)
r0_scaled_WFS = r0_new(data_test['r0'], 0.64e-6, 0.5e-6)
noise_var = NoiseVariance(torch.tensor(r0_scaled_WFS, device=cuda)).clip(0.1, 1.0).item()

dx_0, dy_0 = Center(PSF_0)
r0  = torch.tensor(r0_scaled, requires_grad=True,  device=cuda)
L0  = torch.tensor(25.0, requires_grad=False, device=cuda)
F   = torch.tensor(1.0,  requires_grad=True,  device=cuda)
dx  = torch.tensor(dx_0,   requires_grad=True,  device=cuda, dtype=torch.float32)
dy  = torch.tensor(dy_0,   requires_grad=True,  device=cuda, dtype=torch.float32)
bg  = torch.tensor(0.0,  requires_grad=True,  device=cuda)
n   = torch.tensor(noise_var, requires_grad=True, device=cuda)
Jx  = torch.tensor(10.0, requires_grad=True,  device=cuda)
Jy  = torch.tensor(10.0, requires_grad=True,  device=cuda)
Jxy = torch.tensor(2.0, requires_grad=True,  device=cuda)

PSF_DL = DLPSF()

#%%
def OptimParams(loss_fun, params, iterations, method='LBFGS', verbous=True):
    optimizer = optim.LBFGS(params, lr=10, history_size=20, max_iter=4, line_search_fn="strong_wolfe")

    history = []
    for i in range(iterations):
        optimizer.zero_grad()
        loss = loss_fun( PSD2PSF(r0, L0, F, dx, dy, bg, n, Jx, Jy, Jxy), PSF_0 )
        loss.backward()
        if verbous: print('Loss:', loss.item(), end="\r", flush=True)

        history.append(loss.item())
        if len(history) > 2:
            if np.abs(loss.item()-history[-1]) < 1e-4 and np.abs(loss.item()-history[-2]) < 1e-4:
                break
        optimizer.step( lambda: loss_fun( PSD2PSF(r0, L0, F, dx, dy, bg, n, Jx, Jy, Jxy), PSF_0 ) )


loss_fn = nn.L1Loss(reduction='sum')
for i in range(20):
    OptimParams(loss_fn, [F, dx, dy], 5)
    OptimParams(loss_fn, [r0, n], 5)
    OptimParams(loss_fn, [bg], 2)
    OptimParams(loss_fn, [Jx, Jy, Jxy], 3)

PSF_1 = PSD2PSF(r0, L0, F, dx, dy, bg, n, Jx, Jy, Jxy)
#SR = PSF_1.max()/PSF_DL.max() * PSF_DL.sum()/PSF_1.sum()
SR = lambda PSF: PSF.max()/PSF_DL.max() * PSF_DL.sum()/PSF.sum()

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

    return FWHM(X0[3].detach().cpu().numpy()), FWHM(X0[4].detach().cpu().numpy())


#la chignon et tarte

# %%
save_data = {
    'F':   F.item(),
    'dx':  dx.item(),
    'dy':  dy.item(),
    'r0':  r0.item(),
    'n':   n.item(),
    'bg':  bg.item(),
    'Jx':  Jx.item(),
    'Jy':  Jy.item(),
    'Jxy': Jxy.item(),
    'SR data': SR(PSF_0).item(), # data PSF
    'SR fit':  SR(PSF_1).item(), # fitted PSF
    'FWHM fit':  FitGauss2D(PSF_1), 
    'FWHM data': FitGauss2D(PSF_0),
    'Img. data': PSF_0.detach().cpu().numpy()*param,
    'Img. fit':  PSF_1.detach().cpu().numpy()*param
}

path_save = 'C:\\Users\\akuznets\\Data\\SPHERE\\fitted\\'
index = path.split(path_test)[-1].split('_')[0]

with open(path_save+str(index)+'.pickle', 'wb') as handle:
    pickle.dump(save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('Saved at:', path_save+str(index)+'.pickle')

#%%
'''
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


class TipToy(nn.Module):

    def PistonFilter(self, f):
        x = (np.pi*self.D*f).cpu().numpy() #TODO: find Bessel analog for pytorch
        R = spc.j1(x)/x
        piston_filter = torch.tensor(1.0-4*R**2, device=self.device)
        piston_filter[self.self.nOtf_AO//2,self.self.nOtf_AO//2,...] *= 0.0

        return piston_filter

    def __init__(self, params)-> None: #, device=None, dtype=None) -> None:
        super(TipToy, self).__init__()
        self.params = nn.ParameterList([nn.Parameter(torch.randn(10, 10))])

        rad2mas = 3600 * 180 * 1000 / np.pi
        rad2arc = rad2mas / 1000
        deg2rad = np.pi / 180
        asec2rad = np.pi / 180 / 3600

        seeing = lambda r0, lmbd: rad2arc*0.976*lmbd/r0 # [arcs]
        r0 = lambda seeing, lmbd: rad2arc*0.976*lmbd/seeing # [m]
        r0_new = lambda r0, lmbd, lmbdₒ: r0*(lmbd/lmbdₒ)**1.2 # [m]

        D          = params['telescope']['TelescopeDiameter']
        self.wvl   = wvl
        psInMas    = params['sensor_science']['PixelScale'] #[mas]
        self.nPix  = params['sensor_science']['FieldOfView']
        self.pitch = params['DM']['DmPitchs'][0] #[m]
        self.h_DM  = params['DM']['DmHeights'][0] # ????? what is h_DM?
        #self.nDM   = 1
        kc         = 1/(2*self.pitch) #TODO: kc is not consistent with vanilla TIPTOP

        zenith_angle  = torch.tensor(params['telescope']['ZenithAngle'], device=self.device) # [deg]
        airmass       = 1.0 / torch.cos(zenith_angle * deg2rad)

        self.GS_wvl    = params['sources_HO']['Wavelength'][0] #[m]
        self.GS_height = params['sources_HO']['Height'] * airmass #[m]
        GS_angle       = torch.tensor(params['sources_HO']['Zenith'],  device=self.device) / rad2arc
        GS_azimuth     = torch.tensor(params['sources_HO']['Azimuth'], device=self.device) * deg2rad
        self.GS_dirs_x = torch.tan(GS_angle) * torch.cos(GS_azimuth)
        self.GS_dirs_y = torch.tan(GS_angle) * torch.sin(GS_azimuth)
        self.nGS       = self.GS_dirs_y.size(0)

        self.wind_speed  = torch.tensor(params['atmosphere']['WindSpeed'], device=self.device)
        self.wind_dir    = torch.tensor(params['atmosphere']['WindDirection'], device=self.device)
        self.Cn2_weights = torch.tensor(params['atmosphere']['Cn2Weights'], device=self.device)
        self.Cn2_heights = torch.tensor(params['atmosphere']['Cn2Heights'], device=self.device) * airmass #[m]
        stretch          = 1.0 / (1.0-self.Cn2_heights/self.GS_height)
        self.h           = self.Cn2_heights * stretch
        self.nL          = self.Cn2_heights.size(0)

        self.WFS_d_sub = np.mean(params['sensor_HO']['SizeLenslets'])
        self.WFS_n_sub = np.mean(params['sensor_HO']['NumberLenslets'])
        self.WFS_det_clock_rate = np.mean(params['sensor_HO']['ClockRate']) # [(?)]
        self.WFS_FOV = params['sensor_HO']['FieldOfView']
        self.WFS_psInMas = params['sensor_HO']['PixelScale']
        self.WFS_RON = params['sensor_HO']['SigmaRON']
        self.WFS_Nph = torch.tensor(params['sensor_HO']['NumberPhotons'][0], device=self.device)
        self.WFS_spot_FWHM = torch.tensor(params['sensor_HO']['SpotFWHM'][0], device=self.device)
        self.WFS_excessive_factor = params['sensor_HO']['ExcessNoiseFactor']

        self.HOloop_rate  = np.mean(params['RTC']['SensorFrameRate_HO']) # [Hz] (?)
        self.HOloop_delay = params['RTC']['LoopDelaySteps_HO'] # [ms] (?)
        self.HOloop_gain  = params['RTC']['LoopGain_HO']

        #-------------------------------------------------------
        pixels_per_l_D = wvl*rad2mas / (psInMas*D)
        self.sampling_factor = int(np.ceil(2.0/pixels_per_l_D)) # check how much it is less than Nyquist
        self.sampling = self.sampling_factor * pixels_per_l_D
        self.nOtf = self.nPix * self.sampling_factor

        self.dk = 1/D/self.sampling # PSD spatial frequency step
        self.cte = (24*spc.gamma(6/5)/5)**(5/6)*(spc.gamma(11/6)**2/(2*np.pi**(11/3)))

        # Initialize spatial frequencies
        kx, ky = torch.meshgrid(
            torch.linspace(-self.nOtf/2, self.nOtf/2-1, self.nOtf, device=self.device)*dk + 1e-10,
            torch.linspace(-self.nOtf/2, self.nOtf/2-1, self.nOtf, device=self.device)*dk + 1e-10,
            indexing = 'ij')

        k2 = kx**2 + ky**2
        k = torch.sqrt(k2)

        mask = torch.ones_like(k2, device=self.device)
        mask[k2 <= kc**2] = 0
        mask_corrected = 1.0-mask

        self.nOtf_AO = int(2*kc/self.dk)

        # Comb samples involved in antialising
        n_times = min(4,max(2,int(np.ceil(self.nOtf/self.self.nOtf_AO/2))))
        ids = []
        for mi in range(-n_times,n_times):
            for ni in range(-n_times,n_times):
                if mi or ni: #exclude (0,0)
                    ids.append([mi,ni])
        ids = np.array(ids)

        m = torch.tensor(ids[:,0], device=self.device)
        n = torch.tensor(ids[:,1], device=self.device)
        N_combs = m.shape[0]

        corrected_ROI = slice(self.nOtf//2-self.self.nOtf_AO//2, self.nOtf//2+self.self.nOtf_AO//2)
        corrected_ROI = (corrected_ROI,corrected_ROI)

        self.mask_AO = mask[corrected_ROI]
        self.mask_corrected_AO = mask_corrected[corrected_ROI]
        self.mask_corrected_AO_1_1  = torch.unsqueeze(torch.unsqueeze(self.mask_corrected_AO,2),3)

        self.kx_AO = kx[corrected_ROI]
        self.ky_AO = ky[corrected_ROI]
        self.k_AO  = k [corrected_ROI]
        self.k2_AO = k2[corrected_ROI]

        M_mask = torch.zeros([self.self.nOtf_AO,self.self.nOtf_AO,self.nGS,self.nGS], device=self.device)
        for j in range(self.nGS):
            M_mask[:,:,j,j] += 1.0

        # Matrix repetitions and dimensions expansion to avoid in in runtime
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
        km = self.kx_1_1.repeat([1,1,N_combs,1]) - torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(m/self.WFS_d_sub,0),0),3)
        kn = self.ky_1_1.repeat([1,1,N_combs,1]) - torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(n/self.WFS_d_sub,0),0),3)

        self.noise_nGs_nGs = torch.normal(mean=torch.zeros([self.nOtf_AO,self.nOtf_AO,self.nGS,self.nGS]), std=torch.ones([self.nOtf_AO,self.nOtf_AO,self.nGS,self.nGS])*0.001) #TODO: remove noise, do proper matrix inversion
        self.noise_nGs_nGs = self.noise_nGs_nGs.to(self.device)

        self.GS_dirs_x_nGs_nL = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(self.GS_dirs_x,0),0),3).repeat([self.nOtf_AO,self.nOtf_AO,1,self.nL])# * dim_N_N_nGS_nL
        self.GS_dirs_y_nGs_nL = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(self.GS_dirs_y,0),0),3).repeat([self.nOtf_AO,self.nOtf_AO,1,self.nL])# * dim_N_N_nGS_nL

        # Initialize OTF frequencines
        U,V = torch.meshgrid(
            torch.linspace(0, self.nOtf-1, self.nOtf, device=self.device),
            torch.linspace(0, self.nOtf-1, self.nOtf, device=self.device),
            indexing = 'ij')

        U = (U-self.nOtf/2)*2 / self.nOtf
        V = (V-self.nOtf/2)*2 / self.nOtf

        U2  = U**2
        V2  = V**2
        UV  = U*V
        UV2 = U**2 + V**2

        pupil_path = params['telescope']['PathPupil']
        pupil_apodizer = params['telescope']['PathApodizer']

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

        OTF_static = torch.real( fftAutoCorr(pupil_padded) ).unsqueeze(0).unsqueeze(0)
        OTF_static = interpolate(OTF_static, size=(self.nOtf,self.nOtf), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
        OTF_static = OTF_static / OTF_static.max()

        PSD_padder = torch.nn.ZeroPad2d((self.nOtf-self.nOtf_AO)//2)

        piston_filter = self.PistonFilter(self.k_AO)
        PR = self.PistonFilter(torch.hypot(km,kn))



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
        
        vx = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(wind_speed*torch.cos(wind_dir*np.pi/180.),0),0),0)
        vy = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(wind_speed*torch.sin(wind_dir*np.pi/180.),0),0),0)
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


    def ReconstructionFilter(self, r0, L0):
        Av = torch.sinc(self.WFS_d_sub*self.kx_AO)*torch.sinc(self.WFS_d_sub*self.ky_AO) * torch.exp(1j*np.pi*self.WFS_d_sub*(self.kx_AO+self.ky_AO))
        self.SxAv = 2j*np.pi*self.kx_AO*self.WFS_d_sub*Av
        self.SyAv = 2j*np.pi*self.ky_AO*self.WFS_d_sub*Av

        MV = 0
        Wn = self.WFS_noise_var/(2*self.kc)**2

        self.W_atm = self.cte*r0**(-5/3)*(self.k2_AO + 1/L0**2)**(-11/6)*(wvl/self.GS_wvl)**2
        gPSD = torch.abs(self.SxAv)**2 + torch.abs(self.SyAv)**2 + MV*Wn/self.W_atm
        self.Rx = torch.conj(self.SxAv) / gPSD
        self.Ry = torch.conj(self.SyAv) / gPSD
        self.Rx[self.nOtf_AO//2, self.nOtf_AO//2] *= 0
        self.Ry[self.nOtf_AO//2, self.nOtf_AO//2] *= 0
        #return Rx, Ry, SxAv, SyAv, W_atm


    def SpatioTemporalPSD(self):
        A = torch.ones([self.nOtf_AO, self.nOtf_AO], device=self.device) #TODO: fix it. A should be initialized differently
        Ff = self.Rx*self.SxAv + self.Ry*self.SyAv
        psd_ST = (1+abs(Ff)**2 * self.h2 - 2*torch.real(Ff*self.h1*A)) * self.W_atm * self.mask_corrected_AO
        return psd_ST


    def NoisePSD(self):
        noisePSD = abs(self.Rx**2 + self.Ry**2) /(2*self.kc)**2
        noisePSD = noisePSD * self.piston_filter * noise_gain * WFS_noise_var * mask_corrected_AO
        return noisePSD


    def AliasingPSD(r0, L0):
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


    def NoiseVariance(r0):
        WFS_nPix = WFS_FOV / WFS_n_sub
        WFS_pixelScale = WFS_psInMas / 1e3 # [asec]

        # Read-out noise calculation
        nD = torch.tensor([1.0, rad2arc*wvl/WFS_d_sub/WFS_pixelScale]).max() #spot FWHM in pixels and without turbulence
        varRON = np.pi**2/3 * (WFS_RON**2/WFS_Nph**2) * (WFS_nPix**2/nD)**2
        # Photon-noise calculation
        nT = torch.tensor([ 1.0, torch.hypot(WFS_spot_FWHM.max()/1e3, rad2arc*wvl/r0) / WFS_pixelScale ], device=self.device).max()
        varShot = np.pi**2/(2*WFS_Nph) * (nT/nD)**2
        # Noise variance calculation
        varNoise = WFS_excessive_factor * (varRON+varShot)
        return varNoise


    #%% -------------------------------------------------------------
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

        dk = 2*kc/self.nOtf_AO
        PSD *= (dk*wvl*1e9/2/np.pi)**2
        cov = 2*fft.fftshift(fft.fft2(fft.fftshift(PSD)))
        SF  = torch.abs(cov).max()-cov
        fftPhasor = torch.exp(-np.pi*1j*sampling_factor*(U*dx+V*dy))
        OTF_turb  = torch.exp(-0.5*SF*(2*np.pi*1e-9/wvl)**2)

        OTF = OTF_turb * OTF_static * fftPhasor * JitterCore(Jx,Jy,Jxy)
        PSF = torch.abs( fft.fftshift(fft.ifft2(fft.fftshift(OTF))) ).unsqueeze(0).unsqueeze(0)
        PSF_out = interpolate(PSF, size=(nPix,nPix), mode='area').squeeze(0).squeeze(0)
        return (PSF_out/PSF_out.sum() * F + bg) #* 1e2


#%% ------------------------ Managing paths ------------------------
path_test = 'C:\\Users\\akuznets\\Data\\SPHERE\\test\\33_SPHER.2017-03-05T05.00.21.009IRD_FLUX_CALIB_CORO_RAW_left.pickle'
with open(path_test, 'rb') as handle:
    data_test = pickle.load(handle)

path_root = path.normpath('C:/Users/akuznets/Projects/TIPTOP/P3')
path_ini = path.join(path_root, path.normpath('aoSystem/parFiles/irdis.ini'))

parser = parameterParser(path_root, path_ini)
params = parser.params

im = data_test['image']
wvl = data_test['spectrum']['lambda']

params['atmosphere']['Seeing'] = data_test['seeing']
params['atmosphere']['WindSpeed'] = [data_test['Wind speed']['header']]
params['atmosphere']['WindDirection'] = [data_test['Wind direction']['header']]
params['sources_science']['Wavelength'] = [wvl]
params['sensor_science']['FieldOfView'] = data_test['image'].shape[0]
params['sensor_science']['Zenith'] = [90.0-data_test['telescope']['altitude']]
params['sensor_science']['Azimuth'] = [data_test['telescope']['azimuth']]
params['sensor_science']['SigmaRON'] = data_test['Detector']['ron']
params['sensor_science']['Gain'] = data_test['Detector']['gain']


cuda  = torch.device('cuda') # Default CUDA device

'''