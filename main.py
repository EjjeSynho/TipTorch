#%%
import torch
import numpy as np
from torch import nn, optim, fft
import matplotlib.pyplot as plt
import scipy.special as spc
from scipy.optimize import least_squares
from scipy.ndimage import center_of_mass
from torch.nn.functional import interpolate
from astropy.io import fits
from skimage.transform import resize
import pickle
import os
from os import path

from parameterParser import parameterParser
from MUSE import MUSEcube
from utils import rad2mas, rad2arc, deg2rad, asec2rad, seeing, r0, r0_new
from utils import iter_graph, register_hooks
from utils import Center, BackgroundEstimate
from utils import OptimizeTRF, OptimizeLBFGS
from utils import radial_profile, plot_radial_profile

path_root = path.normpath('C:/Users/akuznets/Projects/TIPTOP/P3')
path_ini = path.join(path_root, path.normpath('aoSystem/parFiles/muse_ltao.ini'))

# Load image
data_dir = path.normpath('C:/Users/akuznets/Data/MUSE/DATA/')
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
obs_info = data_cube.obs_info

# Load and correct AO system parameters
parser = parameterParser(path_root, path_ini)
params = parser.params

airmass = obs_info['AIRMASS']

params['atmosphere']['Seeing'] = obs_info['SPTSEEIN']
params['atmosphere']['L0']     = obs_info['SPTL0']
params['atmosphere']['WindSpeed']     = [obs_info['WINDSP']] * len(params['atmosphere']['Cn2Heights'])
params['atmosphere']['WindDirection'] = [obs_info['WINDIR']] * len(params['atmosphere']['Cn2Heights'])
params['sources_science']['Wavelength'] = [wvl]
params['sensor_science']['FieldOfView'] = im.shape[0]
params['sensor_science']['Zenith']      = [90.0-obs_info['TELALT']]
params['sensor_science']['Azimuth']     = [obs_info['TELAZ']]
params['sensor_HO']['NoiseVariance']    = 4.5

#%%
cuda  = torch.device('cuda') # Default CUDA device
#cuda  = torch.device('cpu')
start = torch.cuda.Event(enable_timing=True)
end   = torch.cuda.Event(enable_timing=True)

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
WFS_psInMas = params['sensor_HO']['PixelScale']
WFS_RON = params['sensor_HO']['SigmaRON']
WFS_Nph = params['sensor_HO']['NumberPhotons']
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
nOtf += nOtf%2
#sampling_factor = nOtf/nPix
#sampling = sampling_factor * pixels_per_l_D

#nOtf = 800
#sampling_factor = nOtf / nPix
#sampling = sampling_factor * pixels_per_l_D

#pixels_per_l_D = wvl*rad2mas / (psInMas*D)
#sampling_factor = 2.0/pixels_per_l_D # check how much it is less than Nyquist
#sampling = sampling_factor * pixels_per_l_D
#nOtf = np.round(nPix * sampling_factor, 0).astype('uint')

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

noise_nGs_nGs = torch.normal(mean=torch.zeros([nOtf_AO,nOtf_AO,nGS,nGS]), std=torch.ones([nOtf_AO,nOtf_AO,nGS,nGS])*0.01) #TODO: remove noise, do proper matrix inversion
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

pupil_path = "C:/Users/akuznets/Projects/TIPTOP/P3/aoSystem/data/VLT_CALIBRATION/VLT_PUPIL/ut4pupil320.fits"
pupil = torch.tensor(fits.getdata(pupil_path).astype('float'), device=cuda)

pupil_pix  = pupil.shape[0]
padded_pix = int(pupil_pix*sampling)

pupil_padded = torch.zeros([padded_pix, padded_pix], device=cuda)
pupil_padded[
    padded_pix//2-pupil_pix//2 : padded_pix//2+pupil_pix//2,
    padded_pix//2-pupil_pix//2 : padded_pix//2+pupil_pix//2
] = pupil

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
    #if len(f.shape) == 2:
    #    piston_filter[nOtf_AO//2,nOtf_AO//2] *= 0.0
    #elif len(f.shape) == 3:
    #    piston_filter[nOtf_AO//2,nOtf_AO//2,:] *= 0.0
    #elif len(f.shape) == 4:
    #    piston_filter[nOtf_AO//2,nOtf_AO//2,:,:] *= 0.0
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
    return Rx, Ry


def TomographicReconstructors(r0, L0, WFS_noise_var, inv_method='fast'):
    M = 2j*np.pi*k_nGs_nGs*torch.sinc(WFS_d_sub*kx_nGs_nGs) * torch.sinc(WFS_d_sub*ky_nGs_nGs)
    P = torch.exp(2j*np.pi*h*(kx_nGs_nL*GS_dirs_x_nGs_nL + ky_nGs_nL*GS_dirs_y_nGs_nL))

    MP = M @ P
    MP_t = torch.conj(torch.permute(MP, (0,1,3,2)))

    C_b = torch.ones((nOtf_AO,nOtf_AO,nGS,nGS), dtype=torch.complex64, device=cuda) * torch.eye(4, device=cuda) * WFS_noise_var #torch.diag(WFS_noise_var)
    C_b_inv = torch.ones((nOtf_AO,nOtf_AO,nGS,nGS), dtype=torch.complex64, device=cuda) * torch.eye(4, device=cuda) * 1./WFS_noise_var #torch.diag(WFS_noise_var)
    #TODO: ro at WFS wvl!
    #kernel = torch.unsqueeze(torch.unsqueeze(r0**(-5/3)*cte*(k2_AO + 1/L0**2)**(-11/6) * piston_filter, 2), 3)
    kernel = torch.unsqueeze(torch.unsqueeze(r0_new(r0, 589e-9, 500e-9)**(-5/3)*cte*(k2_AO + 1/L0**2)**(-11/6) * piston_filter, 2), 3)
    C_phi  = kernel.repeat(1, 1, nL, nL) * torch.diag(Cn2_weights) + 0j
    
    if inv_method == 'fast' or inv_method == 'lstsq':
        #kernel_inv = torch.unsqueeze(torch.unsqueeze(1.0/ (r0**(-5/3)*cte*(k2_AO + 1/L0**2)**(-11/6)), 2), 3)
        kernel_inv = torch.unsqueeze(torch.unsqueeze(1.0/(r0_new(r0, 589e-9, 500e-9)**(-5/3)*cte*(k2_AO + 1/L0**2)**(-11/6)), 2), 3)
        C_phi_inv = kernel_inv.repeat(1, 1, nL, nL) * torch.diag(1.0/Cn2_weights) + 0j

    if inv_method == 'standart':
        W_tomo = (C_phi @ MP_t) @ torch.linalg.pinv(MP @ C_phi @ MP_t + C_b + noise_nGs_nGs, rcond=1e-2)

    elif inv_method == 'fast':
        W_tomo = torch.linalg.pinv(MP_t @ C_b_inv @ MP + C_phi_inv, rcond=1e-2) @ (MP_t @ C_b_inv) * \
            torch.unsqueeze(torch.unsqueeze(piston_filter,2),3).repeat(1,1,nL,nGS)
            
    elif inv_method == 'lstsq':
        W_tomo = torch.linalg.lstsq(MP_t @ C_b_inv @ MP + C_phi_inv, MP_t @ C_b_inv).solution * \
            torch.unsqueeze(torch.unsqueeze(piston_filter,2),3).repeat(1,1,2,4)

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
        P_beta_DM = torch.ones([W_tomo.shape[0], W_tomo.shape[1],1,1], dtype=torch.complex64, device=cuda) * mask_corrected_AO_1_1
        P_opt     = torch.ones([W_tomo.shape[0], W_tomo.shape[1],1,2], dtype=torch.complex64, device=cuda) * (mask_corrected_AO_1_1.repeat([1,1,1,nL])) #*dim_1_1_to_1_nL)

    W = P_opt @ W_tomo

    wDir_x = torch.cos(wind_dir*np.pi/180.0)
    wDir_y = torch.sin(wind_dir*np.pi/180.0)

    freq_t = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(wDir_x,0),0),0)*kx_1_nL + \
             torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(wDir_y,0),0),0)*ky_1_nL

    samp_time = 1.0 / HOloop_rate
    www = 2j*torch.pi*k_nGs_nL * torch.sinc(samp_time*WFS_det_clock_rate*wind_speed*freq_t).repeat([1,1,nGS,1]) #* dim_1_nL_to_nGS_nL

    #MP_alpha_L = www*torch.sinc(WFS_d_sub*kx_1_1)*torch.sinc(WFS_d_sub*ky_1_1)\
    #                                *torch.exp(2j*np.pi*h*(kx_nGs_nL*GS_dirs_x_nGs_nL + ky_nGs_nL*GS_dirs_y_nGs_nL))
    MP_alpha_L = www * P * ( torch.sinc(WFS_d_sub*kx_1_1)*torch.sinc(WFS_d_sub*ky_1_1) )
    W_alpha = (W @ MP_alpha_L)

    return W, W_alpha, P_beta_DM, C_phi, C_b, freq_t


def SpatioTemporalPSD(W_alpha, P_beta_DM, C_phi, freq_t):
    '''
    Beta = [self.ao.src.direction[0,s], self.ao.src.direction[1,s]]
    fx = Beta[0]*kx_AO
    fy = Beta[1]*ky_AO

    delta_h = h*(fx+fy) - delta_T * wind_speed_nGs_nL * freq_t
    '''
    delta_T  = (1 + HOloop_delay) / HOloop_rate
    delta_h  = -delta_T * freq_t * wind_speed
    P_beta_L = torch.exp(2j*np.pi*delta_h)

    proj = P_beta_L - P_beta_DM @ W_alpha
    proj_t = torch.conj(torch.permute(proj,(0,1,3,2)))
    psd_ST = torch.squeeze(torch.squeeze(torch.abs((proj @ C_phi @ proj_t)))) * piston_filter * mask_corrected_AO
    return psd_ST


def NoisePSD(W, P_beta_DM, C_b, noise_gain, WFS_noise_var):
    PW = P_beta_DM @ W
    noisePSD = PW @ C_b @ torch.conj(PW.permute(0,1,3,2))
    noisePSD = torch.squeeze(torch.squeeze(torch.abs(noisePSD))) * piston_filter * noise_gain * WFS_noise_var * mask_corrected_AO #torch.mean(WFS_noise_var) 
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

#%%
def PSD2PSF(r0, L0, F, dx, dy, bg, WFS_noise_var, Jx, Jy, Jxy):
    # non-negative reparametrization
    Jx = torch.abs(Jx)
    Jy = torch.abs(Jy)
    WFS_noise_var = torch.abs(WFS_noise_var)

    W, W_alpha, P_beta_DM, C_phi, C_b, freq_t = TomographicReconstructors(r0.abs(), L0.abs(), WFS_noise_var)
    h1, _, _, noise_gain = Controller()
    Rx, Ry = ReconstructionFilter(r0.abs(), L0.abs(), WFS_noise_var)

    PSD =  VonKarmanPSD(r0.abs(), L0.abs()) + \
    PSD_padder(
        NoisePSD(W, P_beta_DM, C_b, noise_gain, WFS_noise_var) + \
        SpatioTemporalPSD(W_alpha, P_beta_DM, C_phi, freq_t) + \
        AliasingPSD(Rx, Ry, h1, r0.abs(), L0.abs()) + \
        ChromatismPSD(r0.abs(), L0.abs())
    )

    dk = 2*kc/nOtf_AO
    cov = 2*fft.fftshift(fft.fft2(fft.fftshift(PSD)))
    SF  = torch.abs(cov).max()-cov
    #SF *= (dk*wvl*1e9/2/np.pi)**2
    SF *= (dk*500/2/np.pi)**2 # PSD is computed for 500 [nm]

    fftPhasor = torch.exp(-np.pi*1j*sampling_factor*(U*dx+V*dy))
    OTF_turb  = torch.exp(-0.5*SF*(2*np.pi*1e-9/wvl)**2)
    OTF = OTF_turb * OTF_static * fftPhasor * JitterCore(Jx.abs(),Jy.abs(),Jxy.abs())
    #OTF = OTF[1:,1:]

    PSF = torch.abs( fft.fftshift(fft.ifft2(fft.fftshift(OTF))) ).unsqueeze(0).unsqueeze(0)
    PSF_out = interpolate(PSF, size=(nPix,nPix), mode='bilinear').squeeze(0).squeeze(0)
    return (PSF_out/PSF_out.sum() * F + bg)

'''
r0  = torch.tensor(0.064, requires_grad=True,  device=cuda)
L0  = torch.tensor(47.93, requires_grad=False, device=cuda)
F   = torch.tensor(1.0,   requires_grad=True,  device=cuda)
dx  = torch.tensor(0.0,   requires_grad=True,  device=cuda)
dy  = torch.tensor(0.0,   requires_grad=True,  device=cuda)
bg  = torch.tensor(0.0,   requires_grad=True, device=cuda)
n   = torch.tensor(4.5,   requires_grad=True, device=cuda)
Jx  = torch.tensor(10.0,  requires_grad=True, device=cuda)
Jy  = torch.tensor(10.0,  requires_grad=True, device=cuda)
Jxy = torch.tensor(5.0,   requires_grad=True, device=cuda)

#start.record()
PSF_0 = PSD2PSF(r0, L0, F, dx, dy, bg, n, Jx, Jy, Jxy)
'''


#%%
r0  = torch.tensor(0.1,   requires_grad=True,  device=cuda)
L0  = torch.tensor(47.93, requires_grad=False, device=cuda)
F   = torch.tensor(1.0,   requires_grad=True,  device=cuda)
dx  = torch.tensor(0.0,   requires_grad=True,  device=cuda)
dy  = torch.tensor(0.0,   requires_grad=True,  device=cuda)
bg  = torch.tensor(0.0,   requires_grad=True,  device=cuda)
n   = torch.tensor(3.5,   requires_grad=True,  device=cuda)
Jx  = torch.tensor(5.0,   requires_grad=True,  device=cuda)
Jy  = torch.tensor(5.0,   requires_grad=True,  device=cuda)
Jxy = torch.tensor(2.0,   requires_grad=True,  device=cuda)

parameters = [r0, L0, F, dx, dy, bg, n, Jx, Jy, Jxy]

#end.record()
#torch.cuda.synchronize()
PSF_1 = PSD2PSF(*parameters)
#print(start.elapsed_time(end))

PSF_0 = torch.tensor(im/im.sum(), device=cuda) #* 1e2
plt.imshow(torch.log(PSF_0).detach().cpu())


#%%
def wrapper(X):
    r0, F, dx, dy, bg, n, Jx, Jy, Jxy = torch.tensor(X, dtype=torch.float32, device=cuda)
    L0 = torch.tensor(47.93, dtype=torch.float32, device=cuda)
    return PSD2PSF(r0, L0, F, dx, dy, bg, n, Jx, Jy, Jxy)

X1 = torch.stack([r0,F,dx,dy,bg,n,Jx,Jy,Jxy]).detach().cpu().numpy()

func = lambda x: (PSF_0-wrapper(x)).detach().cpu().numpy().reshape(-1)
result = least_squares(func, X1, method = 'trf',
                       ftol=1e-9, xtol=1e-9, gtol=1e-9,
                       max_nfev=1000, verbose=1, loss="linear")
X1 = result.x
r0, F, dx, dy, bg, n, Jx, Jy, Jxy = torch.tensor(X1, dtype=torch.float32, device=cuda)
L0 = torch.tensor(47.93, dtype=torch.float32, device=cuda)

print("r0,L0: ({:.3f}, {:.2f})".format(r0.abs().item(), L0.abs().item()))
print("I,bg:  ({:.2f}, {:.1E})".format(F.item(), bg.item()))
print("dx,dy: ({:.2f}, {:.2f})".format(dx.item(), dy.item()))
print("Jx,Jy: ({:.1f}, {:.1f}, {:.1f})".format(Jx.abs().item(), Jy.abs().item(), Jxy.abs().item()))
print("WFS noise: {:.2f}".format(n.data.item()))

#%%
'''
#loss_fn = nn.L1Loss()
#external_grad = torch.tensor(1)
#loss_fn(PSF_0,PSF_1).backward(gradient=external_grad)

loss_fn = nn.L1Loss()
PSF_1 = PSD2PSF(r0, L0, F, dx, dy, bg, n, Jx, Jy, Jxy)
Q = loss_fn(PSF_0, PSF_1)
get_dot = register_hooks(Q)
Q.backward()
dot = get_dot()
#dot.save('tmp.dot') # to get .dot
#dot.render('tmp') # to get SVG
dot # in Jupyter, you can just render the variable
'''

#%
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
                print(loss.item())
            elif method == 'Adam':
                if not i % 10: print(loss.item())

        history.append(loss.item())
        if len(history) > 2:
            if np.abs(loss.item()-history[-1]) < 1e-4 and np.abs(loss.item()-history[-2]) < 1e-4:
                break
        if method == 'LBFGS':
            optimizer.step( lambda: loss_fun( PSD2PSF(r0, L0, F, dx, dy, bg, n, Jx, Jy, Jxy), PSF_0 ) )
        elif method == 'Adam':
            optimizer.step()

r0  = torch.tensor(0.1,   requires_grad=True,  device=cuda)
L0  = torch.tensor(47.93, requires_grad=False, device=cuda)
F   = torch.tensor(1.0,   requires_grad=True,  device=cuda)
dx  = torch.tensor(0.0,   requires_grad=True,  device=cuda)
dy  = torch.tensor(0.0,   requires_grad=True,  device=cuda)
bg  = torch.tensor(0.0,   requires_grad=True,  device=cuda)
n   = torch.tensor(3.5,   requires_grad=True,  device=cuda)
Jx  = torch.tensor(5.0,   requires_grad=True,  device=cuda)
Jy  = torch.tensor(5.0,   requires_grad=True,  device=cuda)
Jxy = torch.tensor(2.0,   requires_grad=True,  device=cuda)

loss_fn1 = nn.L1Loss(reduction='sum')
def loss_fn(A,B):
    return loss_fn1(A,B) + torch.max(torch.abs(A-B)) + torch.abs(torch.max(A)-torch.max(B))

for i in range(10):
    OptimParams(loss_fn, [r0, F, dx, dy], 5)
    OptimParams(loss_fn, [n], 5)
    OptimParams(loss_fn, [bg], 3)
    OptimParams(loss_fn, [Jx, Jy, Jxy], 3)

print("r0,L0: ({:.3f}, {:.2f})".format(r0.data.item(), L0.data.item()))
print("I,bg:  ({:.2f}, {:.1E})".format(F.data.item(), bg.data.item()))
print("dx,dy: ({:.2f}, {:.2f})".format(dx.data.item(), dy.data.item()))
print("Jx,Jy: ({:.1f}, {:.1f}, {:.1f})".format(Jx.data.item(), Jy.data.item(), Jxy.data.item()))
print("WFS noise: {:.2f}".format(n.data.item()))

#%%
'''
def f(r0, L0, F, dx, dy, bg, n, Jx, Jy):
    return loss_fn(PSD2PSF(r0, L0, F, dx, dy, bg, n, Jx, Jy), PSF_0)

#sensetivity = torch.autograd.functional.jacobian(f, (r0, L0, F, dx, dy, bg, n, Jx, Jy))
sensetivity = torch.autograd.functional.hessian(f, (r0, L0, F, dx, dy, bg, n, Jx, Jy))
sensetivity = torch.tensor(sensetivity)
sensetivity = np.array(sensetivity)

from matplotlib import colors
norm=colors.LogNorm(vmin=1e-7, vmax=sensetivity.max())

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xticklabels(['', '$r_0$','$L_0$','F','dx','dy','bg','n','$J_x$','$J_y$'])
ax.set_yticklabels(['', '$r_0$','$L_0$','F','dx','dy','bg','n','$J_x$','$J_y$'])
cax = ax.matshow(np.abs(sensetivity), norm=norm)
'''

PSF_1 = PSD2PSF(r0, L0, F, dx, dy, bg, n, Jx, Jy, Jxy)

plot_radial_profile(PSF_0, PSF_1, 'TipToy')

plt.show()

#%% ==================================================================================================================================
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
"""

#la chignon et tarte
