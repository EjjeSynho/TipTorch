#%%
%reload_ext autoreload
%autoreload 2

import sys
sys.path.insert(0, '..')

import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch import optim
from tools.utils import ParameterReshaper, plot_radial_profiles, SR, draw_PSF_stack
from PSF_models.TipToy_SPHERE_multisrc import TipTorch
from data_processing.SPHERE_preproc_utils import SPHERE_preprocess
from pprint import pprint
from tools.parameter_parser import ParameterParser
from tools.config_manager import ConfigManager, GetSPHEREonsky, GetSPHEREsynth
from tools.utils import rad2mas, SR

from project_globals import SPHERE_DATA_FOLDER, device

#%% Initialize data sample
with open(SPHERE_DATA_FOLDER+'sphere_df.pickle', 'rb') as handle:
    psf_df = pickle.load(handle)

psf_df = psf_df[psf_df['invalid'] == False]
# psf_df = psf_df[psf_df['mag R'] < 7]
# psf_df = psf_df[psf_df['Num. DITs'] < 50]
# psf_df = psf_df[psf_df['Class A'] == True]
# psf_df = psf_df[np.isfinite(psf_df['λ left (nm)']) < 1700]
# psf_df = psf_df[psf_df['Δλ left (nm)'] < 80]
good_ids = psf_df.index.values.tolist()

# psf_df['Jitter X'] = psf_df['Jitter X'].abs()
# psf_df['Jitter Y'] = psf_df['Jitter Y'].abs()

def compress(row):
    if row['Class A']: return 'A'
    elif row['Class B']: return 'B'
    elif row['Class C']: return 'C'
    else: return 'misc'

psf_df['Class'] = psf_df.apply(compress, axis=1)

#%%

import seaborn as sns

# sns.scatterplot(data=psf_df, x='Jitter X', y='Wind speed (header)', type='kde')

# sns.kdeplot(data=psf_df, x="Jitter X", y='r0 (SPARTA)', fill=False, alpha=.5)

sns.displot(psf_df, x='Jitter X', bins=20)

#%%
import numpy as np
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler

# suppose x is your gamma-distributed data
x = np.random.gamma(shape=1, scale=1, size=1000)

# compute the ranks of the data
x_ranks = rankdata(x)

# map the ranks to the quantiles of the standard normal distribution
x_norm = np.sort(np.random.normal(size=len(x)))
x_transformed = x_norm[x_ranks.argsort()]

# Now standardize the data to have 0 mean and standard deviation 1
scaler = StandardScaler()
x_standardized = scaler.fit_transform(x_transformed.reshape(-1, 1))

# Now, x_standardized should be approximately normally distributed with mean 0 and standard deviation 1.

_ = plt.hist(x)

#%%
# 448, 452, 465, 552, 554, 556, 564, 576, 578, 580, 581
# sample_ids = [578]
# sample_ids = [576]
# sample_ids = [992]
sample_ids = [1209] # high noise
# sample_ids = [456]
# sample_ids = [465]
# sample_ids = [1393] #50 DITs
# sample_ids = [1408]
# sample_ids = [898]

# regime = '1P2NI'
regime = '1P21I'
# regime = 'NP2NI'
norm_regime = 'sum'

PSF_0, bg, _, data_samples, merged_config = SPHERE_preprocess(sample_ids, regime, norm_regime, device)

# Jx = merged_config['sensor_HO']['Jitter X'].abs()
# Jy = merged_config['sensor_HO']['Jitter Y'].abs()
# J_msqr = torch.sqrt(Jx**2 + Jy**2)

#%% Initialize model
'''
path_ini = 'C:/Users/akuznets/Projects/TipToy/data/parameter_files/irdis.ini'
merged_config = ParameterParser(path_ini).params
merged_config['NumberSources'] = 1

config_manager = ConfigManager( GetSPHEREonsky() )
config_manager.Convert(merged_config, framework='pytorch', device=device)

merged_config['telescope']['ZenithAngle'] = torch.tensor([merged_config['telescope']['ZenithAngle']]).to(device)
merged_config['atmosphere']['Cn2Weights'] = merged_config['atmosphere']['Cn2Weights'].unsqueeze(0)
merged_config['atmosphere']['Cn2Heights'] = merged_config['atmosphere']['Cn2Heights'].unsqueeze(0)
merged_config['atmosphere']['Seeing'] = torch.tensor([merged_config['atmosphere']['Seeing']]).to(device)
merged_config['atmosphere']['L0'] = torch.tensor([merged_config['atmosphere']['L0']]).to(device)
merged_config['RTC']['SensorFrameRate_HO'] = torch.tensor([merged_config['RTC']['SensorFrameRate_HO']]).to(device)

merged_config['RTC']['LoopDelaySteps_HO'] = torch.tensor([merged_config['RTC']['LoopDelaySteps_HO']]).to(device)
merged_config['RTC']['LoopGain_HO'] = torch.tensor([merged_config['RTC']['LoopGain_HO']]).to(device)
'''
# merged_config['sources_science']['Wavelength'] = [merged_config['sources_science']['Wavelength'][0]]


from PSF_models.TipToy_SPHERE_multisrc import TipTorch
# toy = TipTorch(merged_config, norm_regime, device)
toy = TipTorch(merged_config, None, device)

# toy.PSD_include['aliasing'] = False
# toy.PSD_include['spatio-temporal'] = False
# toy.PSD_include['WFS noise'] = False
# toy.PSD_include['chromatism'] = False
# toy.PSD_include['Moffat'] = False
# toy.PSD_include['fitting'] = True

_ = toy()

# print(toy.PSDs['spatio-temporal'].max())

# plt.imshow(torch.log10(toy.PSD[0,0,...]).detach().cpu().numpy())
# plt.show()

#%
# toy.optimizables = ['r0', 'F', 'dx', 'dy', 'bg', 'dn', 'Jx', 'Jy', 'Jxy', 'wind_dir', 'wind_speed']
# toy.optimizables = ['r0', 'F', 'dx', 'dy', 'bg', 'dn', 'Jx', 'Jy', 'Jxy', 'wind_speed']
# toy.optimizables = ['r0', 'F', 'dx', 'dy', 'bg', 'dn', 'Jx', 'Jy', 'Jxy']
toy.optimizables = ['F', 'dx', 'dy', 'bg', 'Jx', 'Jy', 'Jxy']

_ = toy({
    # 'Jx':  Jx.flatten(),
    # 'Jy':  Jy.flatten(),
    'F':   torch.tensor([0.89, 0.91]*toy.N_src, device=toy.device).flatten(),
    'Jx':  torch.tensor([28.8]*toy.N_src, device=toy.device).flatten(),
    'Jy':  torch.tensor([28.8]*toy.N_src, device=toy.device).flatten(),
    'Jxy': torch.tensor([1.0]*toy.N_src, device=toy.device).flatten(),
    'bg':  bg.to(device)
})

PSF_1 = toy()
#print(toy.EndTimer())
PSF_DL = toy.DLPSF()

draw_PSF_stack(PSF_0, PSF_1, average=True)
mask_in  = toy.mask_rim_in.unsqueeze(1).float()
mask_out = toy.mask_rim_out.unsqueeze(1).float()

'''
test_1 = toy.PSD * toy.mask_rim_in.unsqueeze(1)
test_2 = toy.PSD * toy.mask_rim_out.unsqueeze(1)
test_3 = toy.PSD * (toy.mask_rim_in.unsqueeze(1) + toy.mask_rim_out.unsqueeze(1))

plt.imshow(test_1[0,...].mean(dim=0).squeeze().detach().cpu().numpy())
plt.show()
plt.imshow(test_2[0,...].mean(dim=0).squeeze().detach().cpu().numpy())
plt.show()
plt.imshow(test_3[0,...].mean(dim=0).squeeze().detach().cpu().numpy())
plt.show()

print(torch.mean(test_1))
print(torch.mean(test_2))
# print(torch.mean(test_3))

plt.imshow(torch.log10(toy.PSD[0,...].mean(dim=0)).squeeze().detach().cpu().numpy())
plt.show()
'''

#%%
bounds_dict = {
    'r0' :         (0.05, 0.5),
    'bg' :         (-1e-5, 1e-5),
    'F' :          (0.1, 2),
    'dn' :         (0, 0.05),
    'Jx' :         (-20, 20),
    'Jy' :         (-20, 20),
    'Jxy' :        (-200, 200),
    'dx' :         (-1.5, 1.5),
    'dy' :         (-1.5, 1.5),
    'wind_dir' :   (0, 360),
    'wind_speed' : (0, 35)
}

# bounds = [bounds_dict[name] for name, _ in toy.named_parameters()]

#%%
from scipy.optimize import least_squares

class ParameterReshaper():
    def __init__(self, model, bounds_dict):
        super().__init__()

        import functools
        def prod(iterable):
            return functools.reduce(lambda x, y: x * y, iterable)
        
        self.model  = model
        self.device = model.device
        self.parameter_names = [name for name, p in model.named_parameters() if p.requires_grad]
        self.parameters_list = [getattr(self.model, name) for name in self.parameter_names]
        self.p_shapes = [p.shape for p in self.parameters_list]
        self.p_shapes_flat = [prod(shp) for shp in self.p_shapes]
        self.N_p = len(self.p_shapes)
        self.ids = [slice(sum(self.p_shapes_flat[:i]), sum(self.p_shapes_flat[:i+1])) for i in range(self.N_p)]
        self.bounds_dict = bounds_dict

    def flatten(self):
        return torch.hstack([p.clone().detach().cpu().view(-1) for p in self.parameters_list])

    def unflatten(self, p):
        p_list = [ p[self.ids[i]].reshape(self.p_shapes[i]) for i in range(self.N_p) ]
        for name, new_p in zip(self.parameter_names, p_list):
            if type(new_p) != torch.tensor:
                new_p = new_p.clone().detach().to(self.device)
            param = getattr(self.model, name)
            param.data.copy_(new_p)

    def get_flattened_bounds(self):
        lower_bounds, upper_bounds = [], []
        for name, p_shape in zip(self.parameter_names, self.p_shapes):
            lower, upper = self.bounds_dict[name]
            lower_bound_tensor = torch.tensor(lower).to(self.device).expand(p_shape).view(-1).to(self.device)
            upper_bound_tensor = torch.tensor(upper).to(self.device).expand(p_shape).view(-1).to(self.device)
            lower_bounds.append(lower_bound_tensor)
            upper_bounds.append(upper_bound_tensor)
        return torch.hstack(lower_bounds), torch.hstack(upper_bounds)
    

reshaper = ParameterReshaper(toy, bounds_dict)
x0 = reshaper.flatten().to(device).detach().cpu().numpy()

x_l, x_h = reshaper.get_flattened_bounds()
x_l = x_l.detach().cpu().numpy()
x_h = x_h.detach().cpu().numpy()
#%%
# x0 = (x0 - x_l)/(x_h - x_l)

'''
from tools.utils import register_hooks
from torchmin.optim.scipy_minimizer import ScipyMinimizer

optimizer = ScipyMinimizer(toy.parameters(), method='bfgs', tol=1e-9)
loss_fn = nn.L1Loss(reduction='sum')

for i in range(20):
    optimizer.zero_grad()
    loss = loss_fn( toy(), PSF_0 )
    # if np.isnan(loss.item()): return
    # early_stopping(loss)
    optimizer.step( lambda: loss_fn(toy(), PSF_0) )
    loss.backward()
    print(loss.item())

'''
#%

def run(x):
    x_ = x * (x_h-x_l) + x_l
    reshaper.unflatten(torch.tensor(x_).to(device))
    PSF_1 = toy()
    return (PSF_1-PSF_0).detach().cpu().numpy().reshape(-1) * 100

# result = least_squares(run, x0, method='trf', ftol=1e-9, xtol=1e-9, gtol=1e-9, max_nfev=1000, verbose=2, loss="linear")
result = least_squares(run, x0, method='trf', bounds=(x_l, x_h), ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=1000, verbose=2, loss="linear")

# def run(x):
#     x_ = x * (x_h-x_l) + x_l
#     reshaper.unflatten(x_)
#     PSF_1 = toy()
#     return (PSF_1-PSF_0).reshape(-1).sum().abs()


# loss_fn = nn.L1Loss()

# Q = loss_fn(PSF_0, PSF_1)
# get_dot = register_hooks(Q)
# Q.backward()
# dot = get_dot()
# #dot.save('tmp.dot') # to get .dot
# #dot.render('tmp') # to get SVG
# dot # in Jupyter, you can just render the variable


# result = minimize(run, x0, method='dogleg')
# x1 = result.x


# reshaper.unflatten(torch.tensor(result.x).to(device))


#%% PSF fitting (no early-stopping)
from tools.utils import OptimizeLBFGS, mask_circle
loss = nn.L1Loss(reduction='sum')

'''
fft_DL = torch.fft.fftshift(torch.fft.fft2(PSF_DL)).detach()
R = np.ceil(140 * 0.8).astype(int)
FFT_ROI = slice(PSF_1.shape[-2]//2-R//2, PSF_1.shape[-2]//2+R//2)
mask = torch.tensor(mask_circle(PSF_1.shape[-1], R//2, center=(0,0), centered=True)).to(device).unsqueeze(0).unsqueeze(0) + 1e-4

def FFT_loss(PSF_0, PSF_1):
    fft_0  = torch.fft.fftshift(torch.fft.fft2(PSF_0))
    fft_1  = torch.fft.fftshift(torch.fft.fft2(PSF_1))
    # plt.imshow(torch.log10(fft_1/fft_DL*mask)[0,0,FFT_ROI,FFT_ROI].abs().detach().cpu().numpy())
    # plt.imshow(torch.abs(lossy[0,0,FFT_ROI,FFT_ROI]).detach().cpu().numpy())
    return torch.mean(torch.abs((fft_1-fft_0)/fft_DL*mask)[...,FFT_ROI,FFT_ROI]**0.3)

# Confines a value between 0 and the specified value
window_loss = lambda x, x_max: \
    torch.gt(x,0)*(0.01/x)**2 + torch.lt(x,0)*100 + 100*torch.gt(x,x_max)*(x-x_max)**2

# TODO: specify loss weights
def loss_fn1(a,b):
        # window_loss(toy.r0,  0.5).sum() * 1.0 + \
        # window_loss(toy.Jx,  50 ).sum() * 0.5 + \
        # window_loss(toy.Jy,  50 ).sum() * 0.5 + \
        # window_loss(toy.Jxy, 400).sum() * 0.5 + \
    z = loss(a,b) + \
        window_loss(toy.dn + toy.NoiseVariance(toy.r0), toy.NoiseVariance(toy.r0).max()*1.5).sum() * 0.1 + \
        torch.lt(torch.max(toy.PSD*mask_in), torch.max(toy.PSD*mask_out)).float() #+ \
        # FFT_loss(a,b)
    return z
'''
loss_fn1 = loss

optimizer_lbfgs = OptimizeLBFGS(toy, loss_fn1)
 
    
optimizer_lbfgs.Optimize(PSF_0, [toy.bg], 3)
for i in range(10):
    optimizer_lbfgs.Optimize(PSF_0, [toy.F], 4)
    optimizer_lbfgs.Optimize(PSF_0, [toy.dx, toy.dy], 4)
    # optimizer_lbfgs.Optimize(PSF_0, [toy.r0, toy.dn], 3)
    # optimizer_lbfgs.Optimize(PSF_0, [toy.dn], 3)
    # optimizer_lbfgs.Optimize(PSF_0, [toy.wind_dir, toy.wind_speed], 3)
    optimizer_lbfgs.Optimize(PSF_0, [toy.Jx, toy.Jy], 4)
    optimizer_lbfgs.Optimize(PSF_0, [toy.Jxy], 4)

PSF_1 = toy()
PSF_DL = toy.DLPSF()

S_0 = SR(PSF_0, PSF_DL).detach().cpu().numpy()

#%%
WFE = torch.mean(toy.PSD.sum(axis=(-2,-1))**0.5)
WFE_jitter = toy.D/4 * 1e9*(toy.Jx+toy.Jy)*0.5/rad2mas
WFE_total  = torch.sqrt(WFE**2 + WFE_jitter**2).item()

rads = 2*np.pi*WFE_total*1e-9/ toy.wvl.flatten()[0]

S = torch.exp(-rads**2)

# toy.wvl.flatten[0]

#%%
bounds_dict = {
    'r0' :         (0.05, 0.5),
    'bg' :         (-1e-5, 1e-5),
    'F' :          (0.1, 2),
    'dn' :         (0, 0.05),
    'Jx' :         (-20, 20),
    'Jy' :         (-20, 20),
    'Jxy' :        (-200, 200),
    'dx' :         (-1.5, 1.5),
    'dy' :         (-1.5, 1.5),
    'wind_dir' :   (0, 360),
    'wind_speed' : (0, 35)
}

bounds = [bounds_dict[name] for name, _ in toy.named_parameters()]

loss = nn.L1Loss(reduction='sum')

def normalize_parameters(parameters, bounds):
    with torch.no_grad():
        for param, (lower, upper) in zip(parameters, bounds):
            param /= upper-lower
            param -= lower/(upper - lower)

def denormalize_parameters(parameters, bounds):
    with torch.no_grad():
        for param, (lower, upper) in zip(parameters, bounds):
            param = (param + lower/(upper-lower)) * (upper-lower)


def project_parameters(parameters, bounds):
    with torch.no_grad():
        for param, (lower, upper) in zip(parameters, bounds):
            below_lower = param < lower
            above_upper = param > upper

            if torch.any(below_lower):
                param.data[below_lower] = 2 * lower - param.data[below_lower]

            if torch.any(above_upper):
                param.data[above_upper] = 2 * upper - param.data[above_upper]

            torch.clamp(param, lower, upper)


def optimize_with_boundary_conditions(model, loss_fn, bounds, lr=1, max_iter=20):

    normalize_parameters(model.parameters(), bounds)
    
    optimizer = optim.LBFGS(model.parameters(), lr=lr, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        loss = loss_fn(toy(), PSF_0)
        loss.backward()
        print('loss:', loss.item())
        return loss

    for _ in range(max_iter):
        optimizer.step(closure)
        project_parameters(model.parameters(), bounds)

    # Denormalize parameters
    denormalize_parameters(model.parameters(), bounds)
    return model

# Example usage:
# Replace "your_model" with your model instance and "your_loss_fn" with your loss function
# bounds = [(80, 100), (1e-6, 1e-4)]  # Example bounds for two parameters

optimized_model = optimize_with_boundary_conditions(toy, loss_fn, bounds)


#%%
# print('\nStrehl ratio: ', SR(PSF_1, PSF_DL))
draw_PSF_stack(PSF_0, PSF_1, average=True)

destack = lambda PSF_stack: [ x for x in torch.split(PSF_stack[:,0,...].cpu(), 1, dim=0) ]
plot_radial_profiles(destack(PSF_0), destack(PSF_1), 'Data', 'TipToy', title='IRDIS PSF', dpi=200)

#%% ================================= PSFAO fitting =================================
toy = TipTorch(merged_config, norm_regime, device, TipTop=False, PSFAO=True)

toy.optimizables = ['r0', 'F', 'dx', 'dy', 'bg', 'Jx', 'Jy', 'Jxy', 'amp', 'b', 'alpha', 'beta', 'ratio', 'theta']

_ = toy({
    'Jxy': torch.tensor([0.1]*toy.N_src, device=toy.device).flatten(),
    # 'Jx':  J_msqr.flatten(),
    # 'Jy':  J_msqr.flatten(),
    'Jx':  Jx.flatten(),
    'Jy':  Jy.flatten(),
    'bg':  bg.to(device)
})

#%%
loss = nn.L1Loss(reduction='sum')

# TODO: specify loss weights
def loss_fn(a,b):
    z = loss(a,b) + \
        torch.lt(torch.mean(toy.PSD*mask_in), torch.mean(toy.PSD*mask_out))
    return z

optimizer_lbfgs = OptimizeLBFGS(toy, loss_fn)


optimizer_lbfgs.Optimize(PSF_0, [toy.bg], 5)
for i in range(10):
    optimizer_lbfgs.Optimize(PSF_0, [toy.F], 3)
    optimizer_lbfgs.Optimize(PSF_0, [toy.dx, toy.dy], 3)
    optimizer_lbfgs.Optimize(PSF_0, [toy.b], 3)
    optimizer_lbfgs.Optimize(PSF_0, [toy.r0, toy.amp, toy.alpha, toy.beta], 3)
    optimizer_lbfgs.Optimize(PSF_0, [toy.ratio, toy.theta], 3)
    optimizer_lbfgs.Optimize(PSF_0, [toy.Jx, toy.Jy, toy.Jxy], 3)

PSF_1 = toy()


#%%
print('\nStrehl ratio: ', SR(PSF_1, PSF_DL))
draw_PSF_stack(PSF_0, PSF_1, average=True)

plot_radial_profiles(destack(PSF_0), destack(PSF_1), 'Data', 'TipToy', title='IRDIS PSF', dpi=200, cutoff=64)

#%% ================================= Read OOPAO sample =================================

# regime = 'NP2NI'
regime = '1P21I'

PSF_2, bg, norms, synth_samples, synth_config = SPHERE_preprocess(sample_ids, regime, norm_regime, synth=True)

#%
from tools.utils import rad2mas
TT_res = synth_samples[0]['WFS']['tip/tilt residuals']

D = synth_config['telescope']['TelescopeDiameter']
ang_pix = synth_samples[0]['Detector']['psInMas'] / rad2mas


jitter = lambda a: 2*2*a/D/ang_pix

TT_jitter = jitter(TT_res)

Jx = TT_jitter[:,0].std() * ang_pix * rad2mas * 2.355
Jy = TT_jitter[:,1].std() * ang_pix * rad2mas * 2.355

#% Initialize model
toy = TipToy(synth_config, norm_regime, device)

toy.optimizables = ['r0', 'F', 'dx', 'dy', 'bg', 'dn', 'Jx', 'Jy', 'Jxy', 'wind_dir', 'wind_speed']
_ = toy({
    'r0': torch.tensor([0.1]*toy.N_src, device=toy.device).flatten(),
    # 'wind_dir': torch.tensor([0.0, 0.0]*toy.N_src, device=toy.device).flatten(),
    # 'wind_speed': torch.tensor([0.0, 0.0]*toy.N_src, device=toy.device).flatten(),
    'Jxy': torch.tensor([0.1]*toy.N_src, device=toy.device).flatten(),
    'Jx':  torch.tensor([1.0]*toy.N_src, device=toy.device).flatten(),
    'Jy':  torch.tensor([1.0]*toy.N_src, device=toy.device).flatten(),
    'dx':  torch.tensor([0.0]*toy.N_src, device=toy.device).flatten(),
    'dy':  torch.tensor([0.0]*toy.N_src, device=toy.device).flatten(),
    'bg':  bg.to(device)
})

# toy.optimizables = ['r0', 'F', 'dx', 'dy', 'bg', 'dn', 'Jx', 'Jy', 'Jxy', 'wind_dir', 'wind_speed']
# _ = toy({
#     'Jxy': torch.tensor([0.1]*toy.N_src, device=toy.device).flatten(),
#     'Jx':  torch.tensor([Jx]*toy.N_src, device=toy.device).flatten(),
#     'Jy':  torch.tensor([Jy]*toy.N_src, device=toy.device).flatten(),
#     'dx':  torch.tensor([-0.5]*toy.N_src, device=toy.device).flatten(),
#     'dy':  torch.tensor([-0.5]*toy.N_src, device=toy.device).flatten(),
#     'bg':  bg.to(device)
# })


from tools.utils import rad2arc

PSF_3 = toy()
PSF_DL = toy.DLPSF()

wind_dir   = synth_config['atmosphere']['WindSpeed'].clone().detach()
wind_speed = synth_config['atmosphere']['WindDirection'].clone().detach()
r0 = rad2arc*0.976*synth_config['atmosphere']['Wavelength'] / synth_config['atmosphere']['Seeing'].clone().detach()


#%%
'''
def plt2PIL(fig=None):
    from PIL import Image
    # Render the figure on a canvas
    if fig is None:
        # fig = plt.gcf()
        canvas = plt.get_current_fig_manager().canvas
    else:
        canvas = fig.canvas

    canvas.draw()
    rgba = canvas.buffer_rgba()

    # Create a numpy array from the bytes
    buffer = np.array(rgba).tobytes()
    # Create a PIL image from the bytes
    pil_image = Image.frombuffer('RGBA', (canvas.get_width_height()), buffer, 'raw', 'RGBA', 0, 1)

    return pil_image
'''

'''
def draw_result_PIL(PSF_in, PSF_out):
    ROI_size = 128
    ROI = slice(PSF_in.shape[-2]//2-ROI_size//2, PSF_in.shape[-1]//2+ROI_size//2)
    dPSF = (PSF_out - PSF_in).abs()

    cut = lambda x: np.log(x.abs().detach().cpu().numpy()[..., ROI, ROI])

    if regime == '1P2NI':
        row = []
        for wvl in range(PSF_in.shape[1]):
            row.append(
                np.hstack([cut(PSF_in[:, wvl,...].mean(dim=0)),
                           cut(PSF_out[:, wvl,...].mean(dim=0)),
                           cut(dPSF[:, wvl,...].mean(dim=0))]) )
        plt.imshow(np.vstack(row))
        plt.title('Sources average')
        plt.show()

    else:
        for src in range(toy.N_src):
            row = []
            for wvl in range(PSF_in.shape[1]):
                row.append( np.hstack([cut(PSF_in[src, wvl,...]), cut(PSF_out[src, wvl,...]), cut(dPSF[src, wvl,...])]) )
            plt.imshow(np.vstack(row))
            plt.title('Source %d' % src)
            plt.show()


draw_result_PIL(PSF_2, PSF_3)
'''

#%%
# reconstruction_result = np.save('C:/Users/akuznets/Data/SPHERE/PSF_TipToy.npy', PSF_1.detach().cpu().numpy())
# initial_PSF = np.save('C:/Users/akuznets/Data/SPHERE/PSF_init.npy', PSF_0.detach().cpu().numpy())

#%%
import torch
from torchmin import minimize
from scipy.optimize import least_squares
from tools.utils import ParameterReshaper

reshaper = ParameterReshaper()

x0 = reshaper.flatten(parameters).numpy()#.to(device)
# x0.requires_grad = True
x1 = reshaper.unflatten(x0)


#%%
def run(x):
    p = reshaper.unflatten(x)
    PSF_1 = toy(*p)
    # return loss_fn(PSF_1, PSF_0)
    return (PSF_1-PSF_0).detach().cpu().numpy().reshape(-1)

# run(x0)

result = least_squares(run, x0, method='trf', ftol=1e-9, xtol=1e-9, gtol=1e-9, max_nfev=1000, verbose=2, loss="linear")

#%%
x1 = reshaper.unflatten(result.x)

PSF_1 = toy(*x1)
print('\nStrehl ratio: ', SR(PSF_1, PSF_DL))

draw_PSF_stack(PSF_0, PSF_1)

#%%


# Select from the following methods:
#  ['bfgs', 'l-bfgs', 'cg', 'newton-cg', 'newton-exact', 
#   'trust-ncg', 'trust-krylov', 'trust-exact', 'dogleg']

# BFGS
result = minimize(run, x0, method='dogleg')
x1 = result.x

print(x1-x0)


#%%
# def rosen(x):
#     return torch.sum(100*(x[..., 1:] - x[..., :-1]**2)**2 + (1 - x[..., :-1])**2)

# initial point
# x0 = torch.tensor([1., 8.])

# Select from the following methods:
#  ['bfgs', 'l-bfgs', 'cg', 'newton-cg', 'newton-exact', 
#   'trust-ncg', 'trust-krylov', 'trust-exact', 'dogleg']

# BFGS
result = minimize(run, x0, method='dogleg')
x1 = result.x

print(x1-x0)

# Newton Conjugate Gradient
# result = minimize(rosen, x0, method='newton-cg')

# Newton Exact
# result = minimize(rosen, x0, method='newton-exact')


#%%
'''
from data_processing.SPHERE_data import SPHERE_database, SPHERE_dataset, LoadSPHEREsampleByID

#la chignon et tarte
dataset = SPHERE_dataset()
database_wvl = dataset.FilterWavelength()
x, y, i0, i1 = dataset.GenerateDataset(database_wvl)

#%% =============================== MAKE DATASET ==========================================

### =======================================================================================
### =======================================================================================

# Load the SPHERE PSF database
path_fitted = 'C:/Users/akuznets/Data/SPHERE/fitted_TipToy_maxnorm/'
path_input  = 'C:/Users/akuznets/Data/SPHERE/test/'

database = SPHERE_database(path_input, path_fitted)

# Filter bad samples
bad_samples = []
for sample in database:
    buf = np.array([sample['fitted']['r0'],
                    sample['fitted']['F'],
                    sample['fitted']['n'],
                    sample['fitted']['dn'],
                    sample['fitted']['Jx'],
                    sample['fitted']['Jy'],
                    sample['fitted']['Jxy'],
                    sample['fitted']['dx'],
                    sample['fitted']['dy'],
                    sample['fitted']['bg']])

    wvl = sample['input']['spectrum']['lambda']
    r0_500 = r0_new(np.abs(sample['fitted']['r0']), 0.5e-6, wvl)
    
    n = sample['fitted']['n'] + sample['fitted']['dn']

    if np.any(np.isnan(buf)) or \
       np.isnan(sample['input']['WFS']['Nph vis']) or \
       np.abs(r0_500) > 3*sample['input']['r0'] or n > 2:
       bad_samples.append(sample['file_id']) 

for bad_sample in bad_samples:
    database.remove(bad_sample)

print(str(len(bad_samples))+' samples were filtered, '+str(len(database.data))+' samples remained')

#%%

# r0_fit  = []
# r0_data = []
# N_ph    = []
# tau_0   = []
# air     = []
# wspd    = []
# wdir    = []
# seeing  = []

# for sample in database:
#     wvl = sample['input']['spectrum']['lambda']
#     r0_fit.append( np.abs(sample['fitted']['r0']) )
#     r0_data.append( np.abs(sample['input']['r0']) )
#     seeing.append( sample['input']['seeing']['SPARTA'] )
#     N_ph.append( np.log10(sample['input']['WFS']['Nph vis'] * sample['input']['WFS']['rate']*1240) )
#     tau_0 .append( sample['input']['tau0']['SPARTA'] )
#     air.append( sample['input']['telescope']['airmass'] )
#     wspd.append( sample['input']['Wind speed']['MASSDIMM'] )
#     wdir.append( sample['input']['Wind direction']['MASSDIMM'] )

# r0_data = np.array(r0_data)
# r0_fit = np.array(r0_fit)
# N_ph = np.array(N_ph)
# tau_0 = np.array(tau_0)
# air = np.array(air)
# wspd = np.array(wspd)
# wdir = np.array(wdir)
# seeing = np.array(seeing)

# counts = plt.hist(N_ph, bins=20)
# plt.show()
# counts = plt.hist(r0_fit, bins=20)
# plt.show()
# counts = plt.hist(seeing, bins=20)
# plt.show()


#%%
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


def GetLabels(data_sample):
    r0_500 = r0_new(np.abs(data_sample['fitted']['r0']), 0.5e-6, toy.wvl)
    #WFS_noise_var = data_sample['fitted']['n'] + data_sample['fitted']['dn']
    WFS_noise_var = data_sample['fitted']['dn']
    buf =  np.array([r0_500,
                     25.0,
                     data_sample['fitted']['F'],
                     data_sample['fitted']['dx'],
                     data_sample['fitted']['dy'],
                     data_sample['fitted']['bg'],
                     WFS_noise_var,
                     data_sample['fitted']['Jx'],
                     data_sample['fitted']['Jy'],
                     data_sample['fitted']['Jxy']])
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

# Filter bad samples manually by their file ids
bad_file_ids = [
    90,  860, 840, 839, 832, 860, 846, 844, 836, 79,  78,  77,  76,  769,
    757, 754, 752, 738, 723, 696, 681, 676, 653, 642, 63,  636, 62,  620, 
    623, 616, 615, 599, 594, 58,  57,  584, 52,  54,  521, 51,  495, 494, 468, 
    456, 433, 415, 414, 373, 368, 364, 352, 342, 338, 336, 315, 296, 297, 
    298, 291, 290, 289, 276, 264, 253, 252, 236, 234, 233, 227, 221, 220, 
    215, 214, 213, 212, 211, 209, 208, 207, 206, 204, 203, 202, 201, 200, 786,
    193, 192, 191, 190, 189, 188, 174, 172, 171, 170, 169, 166, 165, 159, 
    158, 156, 155, 143, 139, 135, 132, 130, 128, 126, 96,  92,  787, 750,
    53,  513, 490, 369, 299, 270, 263, 255, 98,  88,  87,  86,  862, 796, 781]

bad_ids = [database_wvl.find(file_id)['index'] for file_id in bad_file_ids if database_wvl.find(file_id) != None]
good_ids = list(set(np.arange(len(database_wvl))) - set(bad_ids))
database_wvl_good = database_wvl.subset(good_ids)


def GenerateDataset(dataset, with_PSF=False):
    x = [] # inputs
    y = [] # labels

    for sample in dataset:
        input = GetInputs(sample)
        if with_PSF:
            pred = sample['input']['image']
            pred = pred / pred.max() #TODO: proper normalization!!!!!
        else:
            pred = GetLabels(sample)
        x.append(input)
        y.append(pred)
    
    if with_PSF:
        return torch.Tensor(np.vstack(x)).float().to(toy.device), \
               torch.Tensor(np.dstack(y)).permute([2,0,1]).float().to(toy.device)
    else:
        return torch.Tensor(np.vstack(x)).float().to(toy.device), \
               torch.Tensor(np.vstack(y)).float().to(toy.device)


#%%
#validation_ids = np.unique(np.random.randint(0, high=len(database_wvl_good), size=30, dtype=int)).tolist()
validation_ids = np.arange(len(database_wvl_good))
np.random.shuffle(validation_ids)
validation_ids = validation_ids[:30]
database_train, database_val = database_wvl_good.split(validation_ids)

X_train, y_train = GenerateDataset(database_train, with_PSF=False)
X_val, y_val = GenerateDataset(database_val, with_PSF=False)

print(str(X_train.shape[0])+' samples in train dataset, '+str(X_val.shape[0])+' in validation')

#%%
class Gnosis(torch.nn.Module):
    def __init__(self, input_size, hidden_size, psf_model=None, tranform_fun=lambda x:x, device='cpu'):
        self.device = device
        super(Gnosis, self).__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size

        self.fc1  = torch.nn.Linear(self.input_size, self.hidden_size*2, device=self.device)
        self.act1 = torch.nn.Tanh()
        self.fc2  = torch.nn.Linear(self.hidden_size*2, hidden_size, device=self.device)
        self.act2 = torch.nn.Tanh()
        self.fc3  = torch.nn.Linear(self.hidden_size, 10, device=self.device)

        self.inp_normalizer = torch.ones(self.input_size, device=self.device)
        self.out_normalizer = torch.ones(10, device=self.device)
        self.inp_bias = torch.zeros(self.input_size, device=self.device)
        self.out_bias = torch.zeros(10, device=self.device)

        self.psf_model = psf_model
        self.tranform_fun = tranform_fun
        self.buf_outin = 0.0

    def forward(self, x):
        hidden1 = self.fc1(x * self.inp_normalizer + self.inp_bias)
        act1 = self.act1(hidden1)
        hidden2 = self.fc2(act1)
        act2 = self.act2(hidden2)
        model_inp = self.fc3(act2) * self.out_normalizer + self.out_bias
        if self.psf_model is None:
            return model_inp
        else:
            self.buf_outin = model_inp
            return self.psf_model(self.tranform_fun(model_inp)) # to rescale r0 mostly


gnosis = Gnosis(input_size=9, hidden_size=200, device=toy.device)
gnosis.inp_normalizer = torch.tensor([5, 50, 1/50, 1/360, 1, 0.5, 2, 2, 1e6], device=toy.device).unsqueeze(0)
gnosis.inp_bias = torch.tensor([0, 0, 0, 0, -1, -3, 0, 0, 0],   device=toy.device).unsqueeze(0)
gnosis.out_normalizer = 1.0/torch.tensor([1, 1, 1, 10, 10, 2e6, 1, .1, .1, .1], device=toy.device).unsqueeze(0)

#%%
loss_fn = nn.L1Loss() #reduction='sum')
optimizer = optim.SGD([{'params': gnosis.fc1.parameters()},
                       {'params': gnosis.act1.parameters()},
                       {'params': gnosis.fc2.parameters()},
                       {'params': gnosis.act2.parameters()},
                       {'params': gnosis.fc3.parameters()}], lr=1e-3, momentum=0.9)

for i in range(6000):
    optimizer.zero_grad()
    loss = loss_fn(gnosis(X_train), y_train)
    loss.backward()
    if not i % 1000: print(loss.item())
    optimizer.step()

print('Validation loss: '+str(loss_fn(gnosis(X_val), y_val).item()))

#torch.save(gnosis.state_dict(), 'gnosis_weights_psfao.dict')
#gnosis.load_state_dict(torch.load('gnosis_weights_psfao.dict'))
#gnosis.eval()


#%%
def PSFcomparator(dataset):
    # Predicted PSFs
    test_batch = [sample['input'] for sample in dataset]
    toy.Update(test_batch)
    toy.norm_regime = 'max' #TODO: proper normalization regime choice
    gnosis.psf_model = toy

    X_val, _ = GenerateDataset(dataset, with_PSF=False)
    PSFs_2 = gnosis(X_val)

    # Input PSFs
    PSFs_0 = torch.tensor([sample['input']['image'] for sample in dataset], device=toy.device)
    norma  = PSFs_0.amax(dim=(1,2)).unsqueeze(1).unsqueeze(2) #TODO: proper normalization!
    PSFs_0 = PSFs_0 / norma

    # Direct prediction from the telemetry
    N_src = len(test_batch)
    r0  = torch.tensor([sample['input']['r0'] for sample in dataset], device=toy.device)
    L0  = torch.tensor([25.]*N_src,  device=toy.device)
    F   = torch.tensor([1.0]*N_src,  device=toy.device)
    dx  = torch.tensor([0.0]*N_src,  device=toy.device)
    dy  = torch.tensor([0.0]*N_src,  device=toy.device)
    bg  = torch.tensor([0.0]*N_src,  device=toy.device)
    dn  = torch.tensor([0.0]*N_src,  device=toy.device)
    Jx  = torch.tensor([10.0]*N_src, device=toy.device)
    Jy  = torch.tensor([10.0]*N_src, device=toy.device)
    Jxy = torch.tensor([2.0]*N_src,  device=toy.device)

    parameters = [r0, L0, F, dx, dy, bg, dn, Jx, Jy, Jxy]
    PSFs_3 = toy(*parameters)

    # Cross-check inputs from the fitted dataset
    PSFs_1 = torch.tensor([sample['fitted']['Img. fit'] for sample in dataset],  device=toy.device)
    PSFs_1 = PSFs_1 / norma
    return PSFs_0.detach().cpu().numpy(), \
           PSFs_1.detach().cpu().numpy(), \
           PSFs_2.detach().cpu().numpy(), \
           PSFs_3.detach().cpu().numpy()

# Input data
# fitted PSFs
# NN-predicted PSFs
# telemetry-predicted PSFs

PSF_0s, PSF_1s, PSF_2s, PSF_3s = PSFcomparator(database_val)

fit_diff    = np.abs(PSF_0s-PSF_1s).max(axis=(1,2))
gnosis_diff = np.abs(PSF_0s-PSF_2s).max(axis=(1,2))
direct_diff = np.abs(PSF_0s-PSF_3s).max(axis=(1,2))

profile_0s = radial_profile(PSF_0s)[:,:32] * 100
profile_1s = radial_profile(PSF_1s)[:,:32] * 100
profile_2s = radial_profile(PSF_2s)[:,:32] * 100
profile_3s = radial_profile(PSF_3s)[:,:32] * 100

#%%
fig = plt.figure(figsize=(6,4), dpi=150)
plt.grid()

def plot_std(x,y, label, color, style):
    y_m = np.nanmedian(y, axis=0)
    y_s = np.nanstd(y, axis=0)
    lower_bound = y_m-y_s
    upper_bound = y_m+y_s

    plt.fill_between(x, lower_bound, upper_bound, color=color, alpha=0.3)
    plt.plot(x, y_m, label=label, color=color, linestyle=style)

x = np.arange(profile_0s.shape[1])
plot_std(x, np.abs(profile_0s), 'Input PSF', 'darkslategray', '-')
plot_std(x, np.abs(profile_0s-profile_1s), '$\Delta$ Fit', 'royalblue', '--')
plot_std(x, np.abs(profile_0s-profile_2s), '$\Delta$ Gnosis', 'darkgreen', ':')
plot_std(x, np.abs(profile_0s-profile_3s), '$\Delta$ Direct', 'orchid', 'dashdot')

plt.title('Accuracy comparison (avg. for validation dataset)')
plt.yscale('symlog')
plt.xlim([x.min(), x.max()])
plt.legend()
plt.ylabel('Abs. relative diff., [%]')
plt.xlabel('Pixels')

#%%
fd = np.nanmedian(fit_diff)
gd = np.nanmedian(gnosis_diff)
dd = np.nanmedian(direct_diff)

print('Fitting: '+str(np.round((1.-fd)*100).astype('int'))+'% median accuracy')
print('Gnosis:  '+str(np.round((1.-gd)*100).astype('int'))+'% median accuracy')
print('Direct:  '+str(np.round((1.-dd)*100).astype('int'))+'% median accuracy')

#%% ==========================================================================================
gnosis.psf_model.Update([sample['input'] for sample in database_train])

X_train, y_train = GenerateDataset(database_train, with_PSF=False)
_, pred_train = GenerateDataset(database_train, with_PSF=True)

test = gnosis(X_train)

#%%

optimizer.zero_grad()
loss = loss_fn(gnosis(X_train), pred_train)
loss.backward()

print(loss.item())

#%%
testo = test.detach().cpu().numpy()
for i in range(test.shape[0]):
    el_croppo = slice(test.shape[1]//2-32, test.shape[2]//2+32)
    el_croppo = (0, el_croppo, el_croppo)
    plt.imshow(torch.log(test[el_croppo]).detach().cpu())
    plt.show()

#%%
loss_fn = nn.L1Loss() #reduction='sum')
optimizer = optim.SGD([{'params': gnosis.fc1.parameters()},
                       {'params': gnosis.act1.parameters()},
                       {'params': gnosis.fc2.parameters()},
                       {'params': gnosis.act2.parameters()},
                       {'params': gnosis.fc3.parameters()}], lr=1e-3, momentum=0.9)

for i in range(6000):
    optimizer.zero_grad()
    loss = loss_fn(gnosis(X_train), y_train)
    loss.backward()
    if not i % 1000: print(loss.item())
    optimizer.step()

'''
# %%
