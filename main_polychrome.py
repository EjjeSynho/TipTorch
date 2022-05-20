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
from MUSE import MUSEcube


def iter_graph(root, callback):
    queue = [root]
    seen = set()
    while queue:
        fn = queue.pop()
        if fn in seen:
            continue
        seen.add(fn)
        for next_fn, _ in fn.next_functions:
            if next_fn is not None:
                queue.append(next_fn)
        callback(fn)

def register_hooks(var):
    fn_dict = {}
    def hook_c_b(fn):
        def register_grad(grad_input, grad_output):
            fn_dict[fn] = grad_input
        fn.register_hook(register_grad)
    iter_graph(var.grad_fn, hook_c_b)

    def is_bad_grad(grad_output):
        if grad_output is None:
            return False
        return grad_output.isnan().any() or (grad_output.abs() >= 1e6).any()

    def make_dot():
        node_attr = dict(style='filled',
                        shape='box',
                        align='left',
                        fontsize='12',
                        ranksep='0.1',
                        height='0.2')
        dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))

        def size_to_str(size):
            return '('+(', ').join(map(str, size))+')'

        def build_graph(fn):
            if hasattr(fn, 'variable'):  # if GradAccumulator
                u = fn.variable
                node_name = 'Variable\n ' + size_to_str(u.size())
                dot.node(str(id(u)), node_name, fillcolor='lightblue')
            else:
                def grad_ord(x):
                    mins = ""
                    maxs = ""
                    y = [buf for buf in x if buf is not None]
                    for buf in y:
                        min_buf = torch.abs(buf).min().cpu().numpy().item()
                        max_buf = torch.abs(buf).max().cpu().numpy().item()

                        if min_buf < 0.1 or min_buf > 99:
                            mins += "{:.1e}".format(min_buf) + ', '
                        else:
                            mins += str(np.round(min_buf,1)) + ', '
                        if max_buf < 0.1 or max_buf > 99:
                            maxs += "{:.1e}".format(max_buf) + ', '
                        else:
                            maxs += str(np.round(max_buf,1)) + ', '
                    return mins[:-2] + ' | ' + maxs[:-2]

                assert fn in fn_dict, fn
                fillcolor = 'white'
                if any(is_bad_grad(gi) for gi in fn_dict[fn]):
                    fillcolor = 'red'
                dot.node(str(id(fn)), str(type(fn).__name__)+'\n'+grad_ord(fn_dict[fn]), fillcolor=fillcolor)
            for next_fn, _ in fn.next_functions:
                if next_fn is not None:
                    next_id = id(getattr(next_fn, 'variable', next_fn))
                    dot.edge(str(next_id), str(id(fn)))
        iter_graph(var.grad_fn, build_graph)
        return dot

    return make_dot


#%%
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

data_cube = MUSEcube(path_im, angle)
im, _, wvl = data_cube.Layer(0)
obs_info = data_cube.obs_info

cuda  = torch.device('cuda') # Default CUDA device
#cuda  = torch.device('cpu')

poly_cube   = data_cube.cube_img
wavelengths = torch.tensor(data_cube.wavelengths, device=cuda)
#%%
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
params['sensor_HO']['NoiseVariance']    = 5.0

#%%

start = torch.cuda.Event(enable_timing=True)
end   = torch.cuda.Event(enable_timing=True)

rad2mas = 3600 * 180 * 1000 / np.pi
rad2arc = rad2mas / 1000
deg2rad = np.pi / 180
asec2rad = np.pi / 180 / 3600

seeing_from_r0 = lambda r0, λ: rad2arc*0.976*λ/r0 # [arcs]
r0_from_seeing = lambda seeing, λ: rad2arc*0.976*λ/seeing # [m]
r0_rescale     = lambda r0, λ, λₒ: r0*(λ/λₒ)**1.2 # [m] 

D       = params['telescope']['TelescopeDiameter']
#wvl     = wvl
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
#pixels_per_l_D = wavelengths*rad2mas / (psInMas*D)
#sampling_factor = torch.ceil(2.0/pixels_per_l_D).int() # check how much it is less than Nyquist
#sampling = sampling_factor * pixels_per_l_D
#nOtf = nPix * sampling_factor.max().item()

pixels_per_l_D = wavelengths*rad2mas / (psInMas*D)
sampling_factor = 2.0/pixels_per_l_D # check how much it is less than Nyquist
sampling = sampling_factor * pixels_per_l_D
nOtf = torch.round(nPix * sampling_factor.max()).int().item()
nOtf += nOtf%2

dk = 1/D/sampling.min() # PSD spatial frequency step
cte = (24*spc.gamma(6/5)/5)**(5/6)*(spc.gamma(11/6)**2/(2*np.pi**(11/3)))

#with torch.no_grad():
# Initialize spatial frequencies
kx, ky = torch.meshgrid(
    torch.linspace(-nOtf/2, nOtf/2-1, nOtf, device=cuda)*dk + 1e-10,
    torch.linspace(-nOtf/2, nOtf/2-1, nOtf, device=cuda)*dk + 1e-10)
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

pupil_path = "C:/Users/akuznets/Projects/TIPTOP/P3/aoSystem/data/VLT_CALIBRATION/VLT_PUPIL/ut4pupil320.fits"
pupil = torch.tensor(fits.getdata(pupil_path).astype('float'), device=cuda)
#pupil = torch.tensor( PupilVLT(int(nOtf//sampling)) )

pupil_pix  = pupil.shape[0]
#padded_pix = nOtf

OTFs_static = torch.zeros([len(wavelengths), nOtf, nOtf], device=cuda)

def fftAutoCorr(x):
    x_fft = fft.fft2(x)
    return fft.fftshift( fft.ifft2(x_fft*torch.conj(x_fft))/x.size(0)*x.size(1) )

for i,samp in enumerate(sampling):
    padded_pix = int(pupil_pix*sampling[i])

    pupil_padded = torch.zeros([padded_pix, padded_pix], device=cuda)
    pupil_padded[
        padded_pix//2-pupil_pix//2 : padded_pix//2+pupil_pix//2,
        padded_pix//2-pupil_pix//2 : padded_pix//2+pupil_pix//2
    ] = pupil

    OTF_static = torch.real( fftAutoCorr(pupil_padded) ).unsqueeze(0).unsqueeze(0)
    OTF_static = interpolate(OTF_static, size=(nOtf,nOtf), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
    OTFs_static[i,:,:] = OTF_static / OTF_static.max()


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
    return Rx, Ry


def TomographicReconstructors(r0, L0, WFS_noise_var):
    M = 2j*np.pi*k_nGs_nGs*torch.sinc(WFS_d_sub*kx_nGs_nGs) * torch.sinc(WFS_d_sub*ky_nGs_nGs)
    P = torch.exp(2j*np.pi*h*(kx_nGs_nL*GS_dirs_x_nGs_nL + ky_nGs_nL*GS_dirs_y_nGs_nL))

    MP = M @ P
    MP_t = torch.conj(torch.permute(MP, (0,1,3,2)))

    C_b = torch.ones((nOtf_AO,nOtf_AO,nGS,nGS), dtype=torch.complex64, device=cuda) * torch.eye(4, device=cuda) * WFS_noise_var #torch.diag(WFS_noise_var)
    C_b_inv = torch.ones((nOtf_AO,nOtf_AO,nGS,nGS), dtype=torch.complex64, device=cuda) * torch.eye(4, device=cuda) * 1./WFS_noise_var #torch.diag(WFS_noise_var)
    
    kernel = torch.unsqueeze(torch.unsqueeze(r0_rescale(r0, 589, 500)**(-5/3)*cte*(k2_AO + 1/L0**2)**(-11/6) * piston_filter, 2), 3)
    kernel_inv = torch.unsqueeze(torch.unsqueeze(1.0/(r0_rescale(r0, 589, 500)**(-5/3)*cte*(k2_AO + 1/L0**2)**(-11/6)), 2), 3)
    
    C_phi  = kernel.repeat(1, 1, nL, nL) * torch.diag(Cn2_weights) + 0j
    C_phi_inv = kernel_inv.repeat(1, 1, nL, nL) * torch.diag(1.0/Cn2_weights) + 0j

    W_tomo = (C_phi @ MP_t) @ torch.linalg.pinv(MP @ C_phi @ MP_t + C_b + noise_nGs_nGs, rcond=1e-2)
    
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
    #aliasing_PSD = torch.sum(PR*W_mn*abs(Q*avr.sum(axis=3,keepdim=True))**2, axis=(2,3))*0.0229*r0**(-5/3) * mask_corrected_AO
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


def JitterCore(Jx, Jy): #, Jxy):
    u = sampling*D/wavelengths/(3600*180*1e3/np.pi)
    norm_fact = torch.unsqueeze(torch.unsqueeze(u**2 * (2*np.sqrt(2*np.log(2)))**2, 1), 2)
    Djitter = norm_fact * (Jx**2 * U2 + Jy**2 * V2) #+ 2*Jxy*UV)
    return torch.exp(-0.5*Djitter) #TODO: cover Nyquist sampled case


#%%
#def PSD2PSF(r0, L0, F, dx, dy, bg, WFS_noise_var, Jx, Jy):
    # non-negative reparametrization
r0 = torch.tensor(0.1,   requires_grad=True,  device=cuda)
L0 = torch.tensor(47.93, requires_grad=False, device=cuda)
#F  = torch.tensor(1.0,   requires_grad=True,  device=cuda)
#dx = torch.tensor(0.0,   requires_grad=True,  device=cuda)
#dy = torch.tensor(0.0,   requires_grad=True,  device=cuda)
#bg = torch.tensor(0.0,   requires_grad=True,  device=cuda)
n  = torch.tensor(4.5,   requires_grad=True,  device=cuda)
Jx = torch.tensor(10.0,  requires_grad=True,  device=cuda)
Jy = torch.tensor(10.0,  requires_grad=True,  device=cuda)
WFS_noise_var = n


dx = torch.zeros([len(wavelengths)], requires_grad=True,  device=cuda).unsqueeze(1).unsqueeze(2)
dy = torch.zeros([len(wavelengths)], requires_grad=True,  device=cuda).unsqueeze(1).unsqueeze(2)
bg = torch.zeros([len(wavelengths)], requires_grad=True,  device=cuda).unsqueeze(1).unsqueeze(2)
F  = torch.ones ([len(wavelengths)], requires_grad=True,  device=cuda).unsqueeze(1).unsqueeze(2)


r0 = torch.abs(r0)
L0 = torch.abs(L0)
Jx = torch.abs(Jx)
Jy = torch.abs(Jy)
WFS_noise_var = torch.abs(WFS_noise_var)

W, W_alpha, P_beta_DM, C_phi, C_b, freq_t = TomographicReconstructors(r0, L0, WFS_noise_var)
h1, _, _, noise_gain = Controller()
Rx, Ry = ReconstructionFilter(r0, L0, WFS_noise_var)

PSD =  VonKarmanPSD(r0,L0) + \
PSD_padder(
    NoisePSD(W, P_beta_DM, C_b, noise_gain, WFS_noise_var) + \
    SpatioTemporalPSD(W_alpha, P_beta_DM, C_phi, freq_t) + \
    AliasingPSD(Rx, Ry, h1, r0, L0) + \
    ChromatismPSD(r0, L0)
)

dk = 2*kc/nOtf_AO
A = (dk*wavelengths[torch.argmax(sampling_factor).item()]*1e9/2/np.pi)**2

cov = 2*fft.fftshift(fft.fft2(fft.fftshift(PSD)))
SF = torch.abs(cov).max()-cov
SF_poly = SF.unsqueeze(0) * A
#print(SF_poly[2:,:].abs().max())

fftPhasor = torch.exp(-np.pi*1j*sampling_factor.unsqueeze(1).unsqueeze(2) * (U*dx+V*dy))
OTF_turb  = torch.exp(-0.5*SF_poly*(2*np.pi*1e-9/wavelengths.unsqueeze(1).unsqueeze(2))**2)
OTF = OTF_turb * OTFs_static * fftPhasor * JitterCore(Jx,Jy)

PSF = torch.abs( fft.fftshift(fft.ifft2(fft.fftshift(OTF))) ).unsqueeze(0)
PSF_out = interpolate(PSF, size=(nPix,nPix), mode='area').squeeze(0).squeeze(0)

#return 
PSF_out = (PSF_out/PSF_out.sum(dim=(1,2), keepdim=True) * F + bg) #* 1e2
#%%

for i in range(10):
    #plt.imshow(torch.log(OTF_turb[i,:,:].abs()).detach().cpu())
    #plt.colorbar()
    plt.plot(np.abs(PSF_out[i,100,90:110].abs().detach().cpu().numpy()))
plt.show()

#%%
start.record()

end.record()
torch.cuda.synchronize()
#print(start.elapsed_time(end))

def BackgroundEstimate(im, radius=90):
    buf_x, buf_y = torch.meshgrid(
        torch.linspace(-nPix//2, nPix//2, nPix, device=cuda),
        torch.linspace(-nPix//2, nPix//2, nPix, device=cuda)
    )
    mask_noise = buf_x**2 + buf_y**2
    mask_noise[mask_noise < radius**2] = 0.0
    mask_noise[mask_noise > 0.0] = 1.0
    return torch.median(im[mask_noise>0.]).data

PSF_0 = torch.tensor(im/im.sum(), device=cuda) #* 1e2
plt.imshow(torch.log(PSF_0).detach().cpu())

def Center(im):
    center = np.array(np.unravel_index(im.argmax().item(), im.shape))-np.array(im.shape)//2
    return center


#%%

PSF_0 = PSD2PSF(
    torch.tensor(0.105, device=cuda), torch.tensor(47.93, device=cuda),
    torch.tensor(1.0,   device=cuda),
    torch.tensor(1.0,   device=cuda), torch.tensor(-2.0, device=cuda),
    torch.tensor(1e-6,  device=cuda), #bg
    torch.tensor(5.0,   device=cuda),
    torch.tensor(20.0,  device=cuda), torch.tensor(20.0,  device=cuda)
)

#%%
def OptimParams(loss_fun, params, iterations, method='LBFGS', verbous=True):
    if method == 'LBFGS':
        optimizer = optim.LBFGS(params, lr=10, history_size=20, max_iter=4, line_search_fn="strong_wolfe")
    elif method == 'Adam':
        optimizer = optim.Adam(params, lr=1e-2)

    history = []
    for i in range(iterations):
        optimizer.zero_grad()
        loss = loss_fun( PSD2PSF(r0, L0, F, dx, dy, bg, n, Jx, Jy), PSF_0 )
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
            optimizer.step( lambda: loss_fun( PSD2PSF(r0, L0, F, dx, dy, bg, n, Jx, Jy), PSF_0 ) )
        elif method == 'Adam':
            optimizer.step()


loss_fn1 = nn.L1Loss(reduction='sum')
def loss_fn(A,B):
    return loss_fn1(A,B) + torch.max(torch.abs(A-B)) + torch.abs(torch.max(A)-torch.max(B))

for i in range(10):
    OptimParams(loss_fn, [r0, F, dx, dy], 5)
    OptimParams(loss_fn, [n], 5)
    OptimParams(loss_fn, [bg], 3)
    OptimParams(loss_fn, [Jx, Jy], 3)

print("r0,L0: ({:.3f}, {:.2f})".format(r0.data.item(), L0.data.item()))
print("I,bg:  ( ",F.data.item(), ' ', bg.data.item(), ')')
print("dx,dy: ({:.2f}, {:.2f})".format(dx.data.item(), dy.data.item()))
print("Jx,Jy: ({:.1f}, {:.1f})".format(Jx.data.item(), Jy.data.item()))
print("WFS noise: {:.2f}".format(n.data.item()))


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

PSF_1 = PSD2PSF(r0, L0, F, dx, dy, bg, n, Jx, Jy)

profile_0 = radial_profile(PSF_0.detach().cpu().numpy())
profile_1 = radial_profile(PSF_1.detach().cpu().numpy())
profile_diff = np.abs(profile_1-profile_0) / PSF_0.max().cpu().numpy() * 100 #[%]

fig = plt.figure(figsize=(6,4), dpi=300)
ax = fig.add_subplot(111)
ax.set_title('TipToy fitting')
l2 = ax.plot(profile_0, label='TIPTOP')
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

#la chignon et tarte

