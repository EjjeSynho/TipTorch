#%%
#from math import factorial
#from turtle import shape
#from importlib_metadata import requires
#from torch.nn import Module
#from torch.nn.parameter import Parameter
#from torchinfo import summary
#from torch.autograd import gradcheck
#import cupy as cp
#from cupyx.scipy.signal import convolve2d
#from scipy import signal as sg
#from tqdm import tqdm
#from torch import Tensor , complex128, real
#from torch.autograd import Variable, Function
import torch
import numpy as np
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
from torch import fft
import numpy as np
import scipy.special as spc
from torch import fft
from torch.nn.functional import interpolate

from graphviz import Digraph
from VLT_pupil import PupilVLT, CircPupil
import torch


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


'''
x = torch.randn(1, 4, requires_grad=True)
y = torch.randn(1, 4, requires_grad=True)
z = x * y
z = z.sum() * 2
get_dot = register_hooks(z)
z.backward()
dot = get_dot()
#dot.save('tmp.dot') # to get .dot
#dot.render('tmp') # to get SVG
dot # in Jupyter, you can just render the variable
'''

'''
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
z = x + y
end.record()

# Waits for everything to finish running
torch.cuda.synchronize()

print(start.elapsed_time(end))
'''
#with torch.autograd.profiler.profile(use_cuda=True) as prof:
#   # do something
#print(prof)


#%%

# CPU time = 0.1627281904220581 [s]

cuda  = torch.device('cuda') # Default CUDA device
#cuda  = torch.device('cpu')
start = torch.cuda.Event(enable_timing=True)
end   = torch.cuda.Event(enable_timing=True)

with torch.cuda.device(0):
    rad2mas = 3600 * 180 * 1000 / np.pi
    rad2arc = rad2mas / 1000
    deg2rad = np.pi / 180.0

    D       = 8.1 #[m] ????? 8.1 or 8
    wvl     = 728e-9 #[m]
    psInMas = 25.0 #[mas]
    nPix    = 200 #128
    pitch   = 0.2 #[m] 0.22?
    h_DM    = 0.0 # ????? what is h_DM?
    nDM     = 1
    kc      = 1/(2*pitch) #TODO: kc is not consistent with vanilla TIPTOP

    zenith_angle  = torch.tensor(42.4, device=cuda) # [deg]
    airmass       = 1.0 / torch.cos(zenith_angle * deg2rad)

    GS_wvl        = 589e-9 #[m]
    GS_angle      = 7.5 / rad2arc #[rad]
    GS_angle_delta = 2.0 / rad2arc #[rad]
    GS_height     = 90e3*airmass #[m]

    #GS_dirs_x     = torch.tensor([GS_angle+GS_angle_delta, 0, -GS_angle, GS_angle_delta], device=cuda) #[rad]
    #GS_dirs_y     = torch.tensor([GS_angle_delta, GS_angle-GS_angle_delta, -GS_angle_delta, -GS_angle+GS_angle_delta], device=cuda) #[rad]
    
    #GS_dirs_x     = torch.tensor([GS_angle+GS_angle_delta, 0, -GS_angle, 0], device=cuda) #[rad]
    #GS_dirs_y     = torch.tensor([0, GS_angle-GS_angle_delta, 0, -GS_angle+GS_angle_delta], device=cuda) #[rad]

    GS_dirs_x     = torch.tensor([GS_angle, 0, -GS_angle, 0], device=cuda) #[rad]
    GS_dirs_y     = torch.tensor([0, GS_angle, 0, -GS_angle], device=cuda) #[rad]
    nGS           = GS_dirs_y.size(0)

    wind_speed  = torch.tensor([4., 4.1],  device=cuda)
    wind_dir    = torch.tensor([4.5, 4.7], device=cuda)

    Cn2_weights = torch.tensor([0.91, 0.09], device=cuda)
    Cn2_heights = torch.tensor([0.0, 1e4],   device=cuda) * airmass #[m]
    stretch     = 1.0 / (1.0-Cn2_heights/GS_height)
    h           = Cn2_heights * stretch
    nL          = Cn2_heights.size(0)

    WFS_d_sub     = 0.2
    WFS_noise_var = 5.69594009 #TODO: calculate it from imput data
    WFS_det_clock_rate = 1. # [(?)]

    HOloop_rate = 500.0 # [Hz] (?)
    HOloop_delay = 2 # [ms] (?)
    HOloop_gain = 0.5

    pixels_per_l_D = wvl*rad2mas / (psInMas*D)
    sampling_factor = int(np.ceil(2.0/pixels_per_l_D)) # check how much it is less than Nyquist
    sampling = sampling_factor * pixels_per_l_D
    nOtf = nPix * sampling_factor

    dk = 1/D/sampling # PSD spatial frequency step
    cte = (24*spc.gamma(6/5)/5)**(5/6)*(spc.gamma(11/6)**2/(2*np.pi**(11/3)))

    #with torch.no_grad():
    # Initialize spatial frequencies
    kx, ky = torch.meshgrid(
        torch.linspace(-nOtf/2-1, nOtf/2, nOtf, device=cuda)*dk,
        torch.linspace(-nOtf/2-1, nOtf/2, nOtf, device=cuda)*dk)
    k2 = kx**2 + ky**2
    k = torch.sqrt(k2)

    mask = torch.ones_like(k2, device=cuda)
    mask[k2 <= kc**2] = 0
    mask_corrected = 1.0-mask

    nOtf_AO = int(2*kc/dk)
    
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

    noise_nGs_nGs = torch.normal(mean=torch.zeros([nOtf_AO,nOtf_AO,nGS,nGS]), std=torch.ones([nOtf_AO,nOtf_AO,nGS,nGS])*0.1) #TODO: remove noise, do proper matrix inversion
    noise_nGs_nGs = noise_nGs_nGs.to(cuda)

    GS_dirs_x_nGs_nL = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(GS_dirs_x,0),0),3).repeat([nOtf_AO,nOtf_AO,1,nL])# * dim_N_N_nGS_nL
    GS_dirs_y_nGs_nL = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(GS_dirs_y,0),0),3).repeat([nOtf_AO,nOtf_AO,1,nL])# * dim_N_N_nGS_nL

    # Initialize OTF frequencines
    U,V = torch.meshgrid(
        torch.linspace(-1, 1, nOtf, device=cuda),
        torch.linspace(-1, 1, nOtf, device=cuda) )
    U2  = U**2
    V2  = V**2
    UV  = U*V
    UV2 = U**2 + V**2

    #def Gauss2D(X):
    #    return X[0]*torch.exp(-((U-X[1]).pow(2) + (V-X[2]).pow(2)) / (2*X[3].pow(2)))
    #OTF_J = Gauss2D(torch.tensor([1.0, 0.0, 0.0, 0.1]))

    #pupil = torch.tensor(PupilVLT(nPix, vangle=[0,0], petal_modes=False))
    nPix_pupil = np.round(nOtf/sampling)
    nPix_pupil += nPix_pupil % 2
    nPix_pupil = int(nPix_pupil)

    pupil = torch.tensor(CircPupil(nPix_pupil)*1.0, device=cuda)
    pupil_padded = torch.zeros([nOtf, nOtf], device=cuda)
    pupil_padded[
        nOtf//2-nPix_pupil//2:nOtf//2+nPix_pupil//2,
        nOtf//2-nPix_pupil//2:nOtf//2+nPix_pupil//2
    ] = pupil

    fftAutoCorr = lambda x: fft.ifft2(fft.fft2(x)*torch.conj(fft.fft2(x)))/(x.size(0)*x.size(1))
    OTF_static = torch.real( fft.fftshift(fftAutoCorr(pupil_padded)) )
    OTF_static = OTF_static / OTF_static.max()

    PSD_padder = torch.nn.ZeroPad2d((nOtf-nOtf_AO)//2)

    # Piston filter
    def PistonFilter(f):
        x = (np.pi*D*f).cpu().numpy() #TODO: find Bessel analog for pytorch
        R = spc.j1(x)/x
        piston_filter = torch.tensor(1.0-4*R**2, device=cuda)
        if len(f.shape) == 2:
            piston_filter[nOtf_AO//2,nOtf_AO//2] *= 0.0
        elif len(f.shape) == 3:
            piston_filter[nOtf_AO//2,nOtf_AO//2,:] *= 0.0
        elif len(f.shape) == 4:
            piston_filter[nOtf_AO//2,nOtf_AO//2,:,:] *= 0.0
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


def ReconstructionFilter(r0, L0):
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


def TomographicReconstructors(r0, L0):
    M = 2j*np.pi*k_nGs_nGs*torch.sinc(WFS_d_sub*kx_nGs_nGs) * torch.sinc(WFS_d_sub*ky_nGs_nGs)
    P = torch.exp(2j*np.pi*h*(kx_nGs_nL*GS_dirs_x_nGs_nL + ky_nGs_nL*GS_dirs_y_nGs_nL))

    MP = M @ P
    MP_t = torch.conj(torch.permute(MP, (0,1,3,2)))

    C_b = torch.ones((nOtf_AO,nOtf_AO,nGS,nGS), dtype=torch.complex64, device=cuda) * torch.eye(4, device=cuda) * WFS_noise_var #torch.diag(WFS_noise_var)
    kernel = torch.unsqueeze(torch.unsqueeze(r0**(-5/3)*cte*(k2_AO + 1/L0**2)**(-11/6) * piston_filter, 2), 3)
    C_phi  = kernel.repeat(1, 1, nL, nL) * torch.diag(Cn2_weights) + 0j
    W_tomo = (C_phi @ MP_t) @ torch.linalg.pinv(MP @ C_phi @ MP_t + C_b + noise_nGs_nGs, rcond=1e-2)

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

    #MPalphaL = www*np.sinc(WFS_d_sub*kx_nGs_nL)*np.sinc(WFS_d_sub*kx_nGs_nL)\
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


def NoisePSD(W, P_beta_DM, C_b, noise_gain):
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

    aliasing_PSD = torch.sum(PR*W_mn*abs(Q*avr.sum(axis=3,keepdim=True))**2, axis=(2,3) )*cte*r0**(-5/3) * mask_corrected_AO
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

#%
'''
r0 = torch.tensor([0.10539785053590676])

WFS_psInMas = 1214
WFS_pixel_scale  = WFS_psInMas / 1e3 # [asec]
WFS_excess = 1.0
WFS_spotFWHM = torch.tensor([0.0, 0.0])

ron = 3
nph = 500
fovInPix = 592
n_lenslets = 40
nPix = fovInPix / n_lenslets

# read-out noise calculation
nD = torch.tensor([1, rad2arc * wvl/WFS_d_sub/WFS_pixel_scale]).max()  #spot FWHM in pixels and without turbulence
varRON = np.pi**2/3*(ron**2 /nph**2) * (nPix**2/nD)**2
#if varRON > 3:
#    print('The read-out noise variance is very high (%.1f >3 rd^2), there is certainly smth wrong with your inputs, set to 0'%(varRON))
#    varRON = 0
# photo-noise calculation
nT = torch.tensor( [1, torch.hypot(WFS_spotFWHM.max()/1e3, rad2arc*wvl/r0)/WFS_pixel_scale] ).max()
varShot = np.pi**2/(2*nph)*(nT/nD)**2
#if varShot > 3:
#    print('The shot noise variance is very high (%.1f >3 rd^2), there is certainly smth wrong with your inputs, set to 0'%(varShot))
#    varShot = 0
WFS_noise_var = WFS_excess * (varRON+varShot)

'''
#%%

#%------------------------------------------------------------------------------
#L0 = torch.tensor([47.93], requires_grad=True, device=cuda)
#r0 = torch.tensor([0.10539785053590676], requires_grad=False, device=cuda)

#h1, h2, hn, noise_gain = Controller()
#Rx, Ry = ReconstructionFilter()

#W, W_alpha, P_beta_DM, C_phi, C_b, freq_t = TomographicReconstructors(r0, L0)
#PSD = SpatioTemporalPSD(W_alpha, P_beta_DM, C_phi, freq_t)


#%%

def PSD2PSF(r0, L0, F, dx, dy, bg):
    r0 = torch.abs(r0) # non-negative reparametrization
    L0 = torch.abs(L0) # non-negative reparametrization

    W, W_alpha, P_beta_DM, C_phi, C_b, freq_t = TomographicReconstructors(r0, L0)
    h1, _, _, noise_gain = Controller()
    Rx, Ry = ReconstructionFilter(r0, L0)

    PSD = VonKarmanPSD(r0,L0) + PSD_padder(
            NoisePSD(W, P_beta_DM, C_b, noise_gain) + \
            SpatioTemporalPSD(W_alpha, P_beta_DM, C_phi, freq_t) + \
            AliasingPSD(Rx, Ry, h1, r0, L0) + \
            ChromatismPSD(r0, L0)
        )

    PSD *= (dk*wvl*1e9/2/np.pi)**2
    cov = 2*fft.fftshift(fft.fft2(fft.fftshift(PSD)))
    SF  = torch.abs(cov).max()-cov

    fftPhasor = torch.exp(-np.pi*1j*sampling_factor*(U*dx+V*dy))
    OTF_turb  = torch.exp(-0.5*SF*(2*np.pi*1e-9/wvl)**2)
    OTF = OTF_turb * fftPhasor * OTF_static
    PSF = torch.abs( fft.fftshift(fft.ifft2(fft.fftshift(OTF))) )
    PSF = torch.unsqueeze(torch.unsqueeze(PSF, dim=0), dim=0)
    PSF_out = interpolate(PSF, size=(nPix,nPix), mode='area')
    return (PSF_out.squeeze(0).squeeze(0)/PSF.sum() * F + bg) * 1e4

start.record()
PSF_0 = PSD2PSF(
    torch.tensor(0.155, device=cuda), torch.tensor(47.93, device=cuda),
    torch.tensor(1.0, device=cuda),
    torch.tensor(1.0, device=cuda), torch.tensor(-2.0, device=cuda),
    torch.tensor(0.0, device=cuda)
)
end.record()

def BackgroundEstimate(im, radius=75):
    buf_x, buf_y = torch.meshgrid(
        torch.linspace(-nPix//2, nPix//2, nPix, device=cuda),
        torch.linspace(-nPix//2, nPix//2, nPix, device=cuda)
    )
    mask_noise = buf_x**2 + buf_y**2
    mask_noise[mask_noise < radius**2] = 0.0
    mask_noise[mask_noise > 0.0] = 1.0
    return torch.median(im[mask_noise>0.]).data

noise = torch.abs(torch.normal(mean=torch.zeros_like(PSF_0), std=torch.ones_like(PSF_0) * 1e-2))
PSF_0 += noise #+ PSF_0.max()*noise_pow # add artificial noise
PSF_0 -= BackgroundEstimate(PSF_0)
plt.imshow(torch.log(PSF_0).detach().cpu())


#%
'''
import pickle
with open('C:\\Users\\akuznets\\Desktop\\buf\\im.pickle', 'rb') as handle:
    PSF_real = pickle.load(handle)
PSF_0 = torch.tensor(PSF_real / PSF_real.sum(), device=cuda) * 1e3
PSF_0 -= BackgroundEstimate(PSF_0)
'''
#%
# Optimized parameters
r0 = torch.tensor(0.105, requires_grad=True, device=cuda)
L0 = torch.tensor(47.93, requires_grad=False, device=cuda)
F  = torch.tensor(1.0,   requires_grad=True, device=cuda)
dx = torch.tensor(0.0,   requires_grad=True, device=cuda)
dy = torch.tensor(0.0,   requires_grad=True, device=cuda)
bg = torch.tensor(0.0,   requires_grad=False, device=cuda)

PSF_1 = PSD2PSF(r0, L0, F, dx, dy, bg)

torch.cuda.synchronize()
print(start.elapsed_time(end))

'''
crop = 32
zoomed = slice(nPix//2-crop, nPix//2+crop)
zoomed = (zoomed, zoomed)

plt.imshow(torch.log(torch.abs(PSF_0-PSF_1)).detach().cpu().numpy()[zoomed])
plt.show()
'''

'''
loss_fn = nn.MSELoss()
Q = loss_fn(PSF_0, PSF_1)
get_dot = register_hooks(Q)
Q.backward()
dot = get_dot()
#dot.save('tmp.dot') # to get .dot
#dot.render('tmp') # to get SVG
dot # in Jupyter, you can just render the variable
'''

#%%
niter = 26
#loss_fn = nn.MSELoss()
loss_fn = nn.L1Loss()
#loss_fn = nn.CrossEntropyLoss()

#optimizer = optim.SGD([r0, L0, I, dx, dy, bg], lr=1e-3) #, momentum=0.9)
optimizer = optim.LBFGS(
    [r0, L0, F, dx, dy, bg],
    history_size = 20,
    max_iter = 4,
    line_search_fn = "strong_wolfe"
)

for iteration in range(0, niter):
    optimizer.zero_grad()
    loss = loss_fn( PSD2PSF(r0, L0, F, dx, dy, bg), PSF_0 )
    loss.backward()
    #if not iteration % 10:
    print(loss.item())
    #optimizer.step()
    optimizer.step( lambda: loss_fn( PSD2PSF(r0, L0, F, dx, dy, bg), PSF_0 ) )

print('-------------------------------------------------------------')
print('r0: ', r0.data.item())
print('L0: ', L0.data.item())
print('I, bg: ', F.data.item(),',',bg.data.item())
print('dx/dy: (', dx.data.item(),',',dy.data.item(),')')

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
    return radialprofile 

PSF_1 = PSD2PSF(r0, L0, F, dx, dy, bg)

plt.plot(radial_profile(PSF_1.detach().cpu().numpy()), label='Estimate')
plt.plot(radial_profile(PSF_0.detach().cpu().numpy()), label='Data')
plt.legend()
plt.yscale('log')
plt.grid()
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
    np.linspace(-1, 1-2/nOtf, nOtf) )
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
