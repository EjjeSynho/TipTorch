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
from torch import nn, unsqueeze
from torch import optim
import matplotlib.pyplot as plt
from torch import fft
import numpy as np
import scipy.special as spc
from torch import fft
import torch.nn.functional as F
from torch.nn.functional import interpolate

from graphviz import Digraph
from VLT_pupil import PupilVLT, CircPupil
import torch


#%%

x = torch.tensor([1,2,3], dtype=torch.float32).T
#B0 = torch.tensor([[0,1,2], [3,4,5], [6,7,8]], dtype=torch.float32)
#B1 = torch.tensor([[0,1,2], [3,4,5], [6,1,8]], dtype=torch.float32, requires_grad=True)

M1 = torch.diag(torch.range(1,3))

def f(s,x):
    #B = torch.tensor([[0,s[1],2+s[0]], [3*s[0],4,5+s[1]], [6,1,8]])
    #return torch.linalg.pinv(B) @ x

    return torch.tensor([s[0]**2, s[1], 2.0]) * x

s0 = torch.tensor([1.0, 1.0])
s1 = torch.tensor([2.0, 0.5], requires_grad=True)

A = f(s0,x)

loss_fn = nn.MSELoss()
optimizer = optim.LBFGS([s1],
                        history_size=10,
                        max_iter=4,
                        line_search_fn="strong_wolfe")
h_lbfgs = []
for i in range(10):
    optimizer.zero_grad()
    objective = loss_fn(A, f(s1,x))
    objective.backward(retain_graph=True)
    optimizer.step(lambda: loss_fn(A, f(s1,x)))
    print(objective.item())

#print(B1)


#%%

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
    def hook_cb(fn):
        def register_grad(grad_input, grad_output):
            fn_dict[fn] = grad_input
        fn.register_hook(register_grad)
    iter_graph(var.grad_fn, hook_cb)

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
                        min_buf = torch.abs(buf).min().numpy().item()
                        max_buf = torch.abs(buf).max().numpy().item()

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

#%%
rad2mas = 3600 * 180 * 1000 / np.pi
rad2arc = rad2mas / 1000
deg2rad = np.pi / 180

D       = 8.1 #[m] ????? 8.1 or 8
wvl     = 728e-9 #[m]
psInMas = 25.0 #[mas]
nPix    = 200 #128
pitch   = 0.22 #[m]
h_DM    = 0.0 # ????? what is h_DM?
nDM     = 1
kc      = 1/(2*pitch) #TODO: kc is not consistent with vanilla TIPTOP

zenith_angle  = 42.4 # [deg]
airmass       = 1.0 / np.cos(zenith_angle * deg2rad)

GS_angle      = 7.5/rad2arc #[rad]
GS_height     = 90e3*airmass #[m]
GS_directions = torch.tensor([[GS_angle, 0, -GS_angle, 0], [0, GS_angle, 0, -GS_angle]]) #[rad]
GS_dirs_x     = torch.tensor([GS_angle, 0, -GS_angle, 0]) #[rad]
GS_dirs_y     = torch.tensor([0, GS_angle, 0, -GS_angle]) #[rad]
nGS           = GS_directions.size(1)

Cn2_weights = torch.tensor([0.91, 0.09])
Cn2_heights = torch.tensor([0.0, 1e4])*airmass #[m]
stretch     = 1.0 / (1.0-Cn2_heights/GS_height)
h           = Cn2_heights * stretch
nL          = Cn2_heights.size(0)

WFS_d_sub     = 0.2
WFS_noise_var = 5.69594009 #torch.tensor([5.69594009]*GS_directions.size(1))
noise_gain    = 0.6

pixels_per_l_D = wvl*rad2mas / (psInMas*D)
sampling_factor = int(np.ceil(2.0/pixels_per_l_D)) # check how much it is less than Nyquist
sampling = sampling_factor * pixels_per_l_D
nOtf = nPix * sampling_factor

dk = np.min(1/D/sampling) # PSD spatial frequency step
cte = (24*spc.gamma(6/5)/5)**(5/6)*(spc.gamma(11/6)**2/(2*np.pi**(11/3)))

with torch.no_grad():
    # Initialize spatial frequencies
    kx, ky = torch.meshgrid(
        torch.linspace(-nOtf/2-1, nOtf/2, nOtf)*dk,
        torch.linspace(-nOtf/2-1, nOtf/2, nOtf)*dk)
    k2 = kx**2 + ky**2
    k = np.sqrt(k2)

    M_mask = torch.zeros([600,600,nGS,nGS])
    for j in range(nGS):
        M_mask[:,:,j,j] += 1.

    kx_nGs_nGs = torch.ones([1,1,nGS,nGS]) * torch.unsqueeze(torch.unsqueeze(kx,2),3) * M_mask
    ky_nGs_nGs = torch.ones([1,1,nGS,nGS]) * torch.unsqueeze(torch.unsqueeze(ky,2),3) * M_mask
    k_nGs_nGs  = torch.ones([1,1,nGS,nGS]) * torch.unsqueeze(torch.unsqueeze(k,2),3)  * M_mask
    kx_nGs_nL  = torch.ones([1,1,nGS,nL])  * torch.unsqueeze(torch.unsqueeze(kx,2),3)
    ky_nGs_nL  = torch.ones([1,1,nGS,nL])  * torch.unsqueeze(torch.unsqueeze(ky,2),3)

    dim_expander_nGS_nL = torch.ones([nOtf,nOtf,nGS,nL])
    noise_nGs_nGs = torch.normal(mean=torch.zeros([nOtf,nOtf,nGS,nGS]), std=torch.ones([nOtf,nOtf,nGS,nGS])*0.1)

    # Piston filter
    x = (np.pi*D*k).numpy()
    R = spc.j1(x)/x
    piston_filter = torch.tensor(1.0-4*R**2)
    piston_filter[nOtf//2,nOtf//2] *= 0.0

    mask = torch.ones_like(k2)
    mask[k2 <= kc**2] = 0
    mask_AO = 1.0 - mask

    # Initialize OTF frequencines
    U,V = torch.meshgrid(
        torch.linspace(-1, 1-2/nOtf, nOtf),
        torch.linspace(-1, 1-2/nOtf, nOtf) )
    U2  = U**2
    V2  = V**2
    UV  = U*V
    UV2 = U**2 + V**2

    #def Gauss2D(X):
    #    return X[0]*torch.exp(-((U-X[1]).pow(2) + (V-X[2]).pow(2)) / (2*X[3].pow(2)))
    #OTF_J = Gauss2D(torch.tensor([1.0, 0.0, 0.0, 0.1]))

    #pupil = torch.tensor(PupilVLT(nPix, vangle=[0,0], petal_modes=False))
    pupil = torch.tensor(CircPupil(nPix)*1.0)

    pupil_padded = torch.zeros([nOtf, nOtf])
    pupil_padded[nOtf//2-nPix//2:nOtf//2+nPix//2,nOtf//2-nPix//2:nOtf//2+nPix//2] = pupil
    fftAutoCorr = lambda x: fft.ifft2(fft.fft2(x)*torch.conj(fft.fft2(x)))/(x.size(0)*x.size(1))
    OTF_static = torch.real( fft.fftshift(fftAutoCorr(pupil_padded)) )
    OTF_static = OTF_static / OTF_static.max()


#%
L0 = torch.tensor([47.93], requires_grad=True)
r0 = torch.tensor([0.10539785053590676], requires_grad=False)



h_buf = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(h,0),0),0) * dim_expander_nGS_nL
GS_dirs_x_buf = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(GS_dirs_x,0),0),3) * dim_expander_nGS_nL
GS_dirs_y_buf = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(GS_dirs_y,0),0),3) * dim_expander_nGS_nL

#M = torch.zeros([nOtf, nOtf, nGS, nGS], dtype=torch.complex64)
#P = torch.zeros([nOtf, nOtf, nGS, nL],  dtype=torch.complex64)
#for j in range(nGS):
#    M[:, :, j, j] = 2j*np.pi*k*torch.sinc(WFS_d_sub*kx) * torch.sinc(WFS_d_sub*ky)
#    for n in range(nL):
#        P[:, :, j, n] = torch.exp(2j*np.pi*h[n]*(kx*GS_directions[0,j] + ky*GS_directions[1,j]))

M = 2j*np.pi*k_nGs_nGs*torch.sinc(WFS_d_sub*kx_nGs_nGs) * torch.sinc(WFS_d_sub*ky_nGs_nGs)
P = torch.exp(2j*np.pi*h_buf*(kx_nGs_nL*GS_dirs_x_buf + ky_nGs_nL*GS_dirs_y_buf))

MP = M @ P
MP_t = torch.conj(torch.permute(MP, (0,1,3,2)))

Cb = torch.ones((nOtf,nOtf,nGS,nGS), dtype=torch.complex64) * torch.eye(4) * WFS_noise_var #torch.diag(WFS_noise_var)
kernel = torch.unsqueeze(torch.unsqueeze(r0**(-5/3)*cte*(k2 + 1/L0**2)**(-11/6) * piston_filter, 2), 3)
C_phi  = kernel.repeat(1, 1, nL, nL) * torch.diag(Cn2_weights) + 0j
Wtomo  = (C_phi@MP_t) @ torch.linalg.pinv(MP @ C_phi @ MP_t + Cb + noise_nGs_nGs, rcond=1e-2)

#TODO: in vanilla TIPTOP windspeeds are interpolated linearly if number of mod layers is changed!!!!!

#%-----------------------------------------------------------------------------
'''
DMS_opt_dir = torch.tensor([0.0, 0.0])
opt_weights = 1.0

theta_x = torch.tensor([DMS_opt_dir[0]/206264.8 * np.cos(DMS_opt_dir[1]*np.pi/180)])
theta_y = torch.tensor([DMS_opt_dir[0]/206264.8 * np.sin(DMS_opt_dir[1]*np.pi/180)])

P_L  = torch.zeros([nOtf, nOtf, 1, nL], dtype=torch.complex64)
fx = theta_x*kx
fy = theta_y*ky
P_DM = torch.unsqueeze(torch.unsqueeze(torch.exp(2j*np.pi*h_DM*(fx+fy))*mask_AO,2),3)
P_DM_t = torch.conj(P_DM.permute(0,1,3,2))
for l in range(nL):
    P_L[:,:,0,l] = torch.exp(2j*np.pi*h[l]*(fx+fy))*mask_AO

P_opt = torch.linalg.pinv((P_DM_t @ P_DM)*opt_weights, rcond=1e-2) @ ((P_DM_t @ P_L)*opt_weights)

src_direction = torch.tensor([0,0])
fx = src_direction[0]*kx
fy = src_direction[1]*ky
PbetaDM = torch.unsqueeze(torch.unsqueeze(torch.exp(2j*np.pi*h_DM*(fx+fy))*mask_AO,2),3)
'''
with torch.no_grad():
    PbetaDM = torch.ones([Wtomo.shape[0],Wtomo.shape[1],1,1], dtype=torch.complex64)
    P_opt   = torch.ones([Wtomo.shape[0],Wtomo.shape[1],1,2], dtype=torch.complex64)

PW = PbetaDM @ P_opt @ Wtomo

noisePSD = PW @ Cb @ torch.conj(PW.permute(0,1,3,2))
noisePSD = torch.squeeze(torch.squeeze(torch.abs(noisePSD))) * piston_filter * noise_gain * WFS_noise_var * mask_AO #torch.mean(WFS_noise_var) 


Q = noisePSD.sum()
'''
external_grad = torch.tensor(1)
Q.backward(gradient=external_grad)
print(r0.grad)
'''
get_dot = register_hooks(Q)
Q.backward()
dot = get_dot()
#dot.save('tmp.dot') # to get .dot
#dot.render('tmp') # to get SVG
dot # in Jupyter, you can just render the variable


#%%

plt.imshow(np.log(np.real(noisePSD[255:345,255:345])))

#%%
loss_fn = nn.L1Loss()

#X0 = torch.tensor([.15, 20.0, 1.0, 0.0, 0.0], requires_grad=False)
#X1 = torch.tensor([.50, 15.0, 1.0, 0.0, 0.0], requires_grad=True)

r0 = torch.tensor(0.5, requires_grad=True)
L0 = torch.tensor(50,  requires_grad=False)
I  = torch.tensor(1.0, requires_grad=False)
dx = torch.tensor(0.0, requires_grad=True)
dy = torch.tensor(0.0, requires_grad=True)
bg = torch.tensor(0.0, requires_grad=False)

#X0 = torch.tensor([.15, 20.0], requires_grad=False)
#X1 = torch.tensor([.50, 15.0], requires_grad=True)

def vonKarmanPSF(r0, L0, I, dx, dy, bg):
    PSD = cte*r0**(-5/3)*(k2 + 1/L0**2)**(-11/6)*mask
    cov = 2*fft.fftshift(fft.fft2(fft.fftshift(PSD)))
    SF  = torch.abs(cov).max()-cov
    
    fftPhasor = torch.exp(-sampling_factor*np.pi*1j*(U*dx+V*dy))
    OTF_turb = torch.exp(-0.5*SF*(2*np.pi*1e-9/wvl)**2)
    OTF = OTF_turb * OTF_static * fftPhasor
    #return OTF
    PSF = torch.abs( fft.fftshift(fft.ifft2(fft.fftshift(OTF))) )*I + bg
    PSF = torch.unsqueeze(torch.unsqueeze(PSF, dim=0), dim=0)
    PSF_out = interpolate(PSF, size=(nPix,nPix), mode='area')
    return PSF_out.squeeze(0).squeeze(0) * 1e4

PSF_0 = vonKarmanPSF(0.15, 20.0, 1.0, 0.0, 1.0, 0.0) #X0)
#PSF_0 = PSF_0/PSF_0.sum()
PSF_1 = vonKarmanPSF(r0, L0, I, dx, dy, bg) #X1)

'''
zoom = PSF_0.detach().cpu().numpy()
crop = 64
zoom = zoom[nPix//2-crop:nPix//2+crop,nPix//2-crop:nPix//2+crop]
#plt.imshow(np.log(zoom))
plt.imshow(zoom)
plt.show()
'''

#plt.imshow(torch.abs(PSF_0))

#%%
#plt.imshow(np.log((PSF_1).detach().cpu().numpy()))
#plt.show()

#%
'''
def GetCenter(img):
    buf = img.detach().cpu().numpy()
    max_id = np.unravel_index(np.argmax(buf), buf.shape)
    return max_id
print(GetCenter(PSF_1))
'''

'''
plt.imshow(np.log(PSF_0.detach().numpy()))
plt.show()
plt.imshow(np.log(PSF_1.detach().numpy()))
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

'''
Q = loss_fn(PSF_0, PSF_1)
external_grad = torch.tensor(1)
Q.backward(gradient=external_grad)
print(X1.grad)
'''

#%
niter = 500
loss_fn = nn.MSELoss()

#def loss_fn_2(A,B):
#    return loss_fn(A,B) + 0.1*torch.abs(A.max()-B.max())

optimizer = optim.SGD([r0, L0, I, dx, dy, bg], lr=1e-4) #, momentum=0.9)
#optimizer = optim.LBFGS([X1],
#                        history_size=10,
#                        max_iter=4,
#                        line_search_fn="strong_wolfe")

for _ in range(0, niter):
    optimizer.zero_grad()
    loss = loss_fn( vonKarmanPSF(r0, L0, I, dx, dy, bg), PSF_0 )
    loss.backward()
    print(loss)
    optimizer.step() # lambda: loss_fn(vonKarmanPSF(X1), PSF_0) )

print(loss.data)
print(r0, L0, I, dx, dy, bg)


#%% ==================================================================================================================================
#=====================================================================================================================================

'''
def Scan(X_inp, scan_range, id, loss):
    loss_val = []
    for i in scan_range:
        delta = torch.zeros_like(X_inp)
        delta[id] += i
        X_buf = X_inp + delta
        #PSF_buf = torch.abs( vonKarmanPSF(X_buf) )
        #loss_val.append(loss(PSF_buf, PSF_0))
        loss_val.append(lossPSF(vonKarmanPSF(X0), vonKarmanPSF(X_buf)))
    return np.array(loss_val)

loss_dy = Scan(X0,torch.linspace(-10,10,100),3, loss=loss_fn)

plt.plot(loss_dy)
'''

#%%

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

#%
#Npup = 550 ; Number of pixels for pupil array
#Npsd = 4*Npup ; Number of pixels for PSD array
#Nwcp = 2.2*Npup ; Number of pixels for W(C*p) array

# >>>> For OTF
#pix2freq = float(Npup)/(D*Npsd) ; pixel size for PSD array [m^-1 / pix]
#f2D = circarr(Npsd) ; Tableau 2D des fréquences [1/m]
#f2D *= pix2freq

# >>>> For Convo
#PUP = circarr(Npup) LT Npup/2; Tableau pupille (1 dans la pupille, 0 dehors)
#PIX2D = circarr(Nwcp) ; Tableau de distance CCD [pix]; Tableaux relatifs à W(C*p)
#SAMP = 2.2
#CC = (1./D/SAMP) ; facteur de conversion pixel CCD -> fréquences
#PSD_CC = PSDresCircSym(PIX2D*CC,r0=r0,Lext=L0,Foa=F_OA,moff=[errBCK,errAMP,errALPHA,errBETA])
#WCP[*,*,0]  = (CC^2.0) * PSD_CC
#%

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

# %%
from scipy import optimize

factor = 1./np.array([76,48,36,26,21,18,16,14])
N = np.array([3,5,7,10,13,15,17,20])

f = lambda x,A: A*(x+1)
test = optimize.curve_fit(f, N, factor, p0=[1])

plt.plot(N, factor)
plt.plot(N,f(N, test[0]))

#%%

N_ord = 10
rs = np.array([0.05, 0.1, 0.15, 0.2, 0.3, 0.35,  0.4])
factor = 1./np.array([1947.0008921332449, 581.1942961591775, 290.59714807958875, 181.62321754974298, 88.05974184229963, 67.58073211153227, 52.835845105379775])

f = lambda x,n,A: A*x**n
test = optimize.curve_fit(f, rs, factor, p0=[1,1])


plt.plot(rs,factor)
plt.plot(rs,f(rs, *test[0]))


#%% --------------------- Decomposed structure function ------------------------


#%% --------------------------------------------
'''
def AtmoPSD(x):
    PSD = x[0]**(-5/3)*cte*(k2**2 + 1/x[1]**2)**(-11/6)
    return PSD

params = torch.tensor([0.3, .3], device='cuda')
params.requires_grad = True

PSD_0 = AtmoPSD(torch.tensor([0.3, .3], device='cuda'))

#cov = 2* fft.fftshift(fft.fft2(fft.fftshift(PSD))) 
#SF = torch.real(cov).max()-cov

params = torch.tensor([0.15, 1.], device='cuda')
params.requires_grad = True

loss_fn = nn.MSELoss()

niter = 10
optimizer = optim.LBFGS([params],
                        history_size=10,
                        max_iter=4,
                        line_search_fn="strong_wolfe")

for _ in range(0, niter):
    optimizer.zero_grad()
    loss = loss_fn(AtmoPSD(params), PSD_0)
    loss.backward()
    optimizer.step(lambda: loss_fn(AtmoPSD(params), PSD_0))

print(loss)
print(params)
'''
#%%

#%%

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

def f(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

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
def vonKarmanPSF(X):
    PSD = cte*X[0]**(-5/3)*(k2 + 1/X[1]**2)**(-11/6)*mask
    cov = 2*np.fft.fftshift(np.fft.fft2(np.fft.fftshift(PSD)))
    SF  = np.real(cov).max()-cov
    
    fftPhasor = np.exp(-sampling_factor*np.pi*1j*(U*X[3]+V*X[4]))
    OTF_turb = np.exp(-0.5*SF*(2*np.pi*1e-9/wvl)**2) * fftPhasor   
    PSF = np.real( np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(OTF_turb))) )
    PSF = resize(PSF, (nPix,nPix), anti_aliasing=False)

    return PSF * 1e4

PSF_0 = vonKarmanPSF(X0) #reference
PSF_1 = vonKarmanPSF(X1)

#%%

im_norm = PSF_0

# Defining the cost functions
class CostClass(object):
    def __init__(self, alpha_positivity=None):
        self.iter = 0
        self.alpha_positivity = alpha_positivity


    def __call__(self, y):
        self.iter += 1
        im_est = vonKarmanPSF(y)
        return (im_norm - im_est).reshape(-1)

cost = CostClass()

result = least_squares(cost, X1,
                       method = 'trf', ftol=1e-8, xtol=1e-8, gtol=1e-8,
                       max_nfev = 1000, verbose = 1, loss = "linear")
                               
print(result.x)

#la chignon et tarte
