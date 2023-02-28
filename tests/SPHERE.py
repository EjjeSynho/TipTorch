#%%
%reload_ext autoreload
%autoreload 2

import sys
sys.path.insert(0, '..')

import torch
import numpy as np
from torch import nn
from data_processing.SPHERE_data import SPHERE_database, SPHERE_dataset, LoadSPHEREsampleByID
from tools.parameter_parser import ParameterParser
from tools.config_manager import ConfigManager, GetSPHEREonsky
from tools.utils import OptimizeTRF, OptimizeLBFGS
from tools.utils import radial_profile, plot_radial_profile, SR
from tools.utils import BackgroundEstimate
import matplotlib.pyplot as plt

device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')

#%% Initialize data sample
data_samples = []
#data_samples.append( LoadSPHEREsampleByID('C:/Users/akuznets/Data/SPHERE/test/', 0)[1] )
#data_samples.append( LoadSPHEREsampleByID('C:/Users/akuznets/Data/SPHERE/test/', 100)[1] )
data_samples.append( LoadSPHEREsampleByID('C:/Users/akuznets/Data/SPHERE/test/', 347)[1] )
#data_samples.append( LoadSPHEREsampleByID('C:/Users/akuznets/Data/SPHERE/test/', 351)[1] )
# data_samples.append( LoadSPHEREsampleByID('C:/Users/akuznets/Data/SPHERE/test/', 340)[1] )
# data_samples.append( LoadSPHEREsampleByID('C:/Users/akuznets/Data/SPHERE/test/', 342)[1] )
#data_samples.append( LoadSPHEREsampleByID('C:/Users/akuznets/Data/SPHERE/test/', 349)[1] )

path_ini = '../data/parameter_files/irdis.ini'

config_file = ParameterParser(path_ini).params
config_file['atmosphere']['Cn2Weights'] = [0.95, 0.05]
config_file['atmosphere']['Cn2Heights'] = [0, 10000]

config_manager = ConfigManager(GetSPHEREonsky())
merged_config  = config_manager.Merge([config_manager.Modify(config_file, sample) for sample in data_samples])
config_manager.Convert(merged_config, framework='pytorch', device=device)

# merged_config['atmosphere']['WindSpeed'] = torch.tensor([[20, 10], [25, 15], [30, 20]]).to(device)
# merged_config['atmosphere']['WindDirection'] = torch.tensor([[200, 100], [250, 150], [300, 200]]).to(device)

#%% Initialize model
from PSF_models.TipToy_SPHERE_multisrc import TipToy

norm_regime = 'sum'
toy = TipToy(merged_config, norm_regime, device)

#%%
ims = []
for sample in data_samples:
    im = sample['image']
    if    toy.norm_regime == 'max': norma = im.max()
    elif  toy.norm_regime == 'sum': norma = im.sum()
    else: norma = 1.0
    ims.append(torch.tensor(im/norma, device=toy.device))

PSF_0 = torch.tensor(torch.dstack(ims)).permute([2,0,1])
bg_0 = toy.make_tensor([BackgroundEstimate(PSF_, radius=90).item() for PSF_ in ims])

N_src = len(data_samples)

init_torch_param = lambda x, N=None: \
    torch.tensor([x]*N, requires_grad=True, device=device).flatten() \
    if N != None and N != 0 \
    else torch.tensor(x, device=device, requires_grad=True).flatten()

r0  = init_torch_param([sample['r0'] for sample in data_samples])
L0  = init_torch_param(25.0, N_src)
F   = init_torch_param(1.0,  N_src)
dx  = init_torch_param(0.0,  N_src)
dy  = init_torch_param(0.0,  N_src)
dn  = init_torch_param(0.05, N_src)
Jx  = init_torch_param(5.0,  N_src)
Jy  = init_torch_param(5.0,  N_src)
Jxy = init_torch_param(2.0,  N_src)
bg  = bg_0

toy.WFS_Nph.requires_grad = True

#L0.requires_grad = False

parameters = [r0, L0, F, dx, dy, bg, dn, Jx, Jy, Jxy]
x = torch.stack(parameters).T.unsqueeze(0)
#toy.StartTimer()
#%%
PSF_1 = toy.PSD2PSF(*parameters)
#print(toy.EndTimer())
PSF_DL = toy.DLPSF()

def draw_result(PSF_in, PSF_out):
    for i in range(PSF_out.shape[0]):
        el_croppo = slice(PSF_in.shape[1]//2-32, PSF_in.shape[2]//2+32)
        el_croppo = (el_croppo, el_croppo)

        plt.imshow(torch.log(torch.hstack((
            PSF_in[i,:,:].abs()[el_croppo], PSF_out[i,:,:].abs()[el_croppo], \
            ((PSF_out[i,:,:]-PSF_in[i,:,:]).abs()[el_croppo])
        ) )).detach().cpu())
        plt.show()

draw_result(PSF_0, PSF_1)

#%% PSF fitting (no early-stopping)
loss = nn.L1Loss(reduction='sum')

# Confines a value between 0 and the specified value
window_loss = lambda x, x_max: \
    torch.gt(x,0)*(0.01/x)**2 + torch.lt(x,0)*100 + 100*torch.gt(x,x_max)*(x-x_max)**2

# TODO: specify loss weights
def loss_fn(a,b):
    z = loss(a,b) + \
        window_loss(r0, 0.5).sum() * 5.0 + \
        window_loss(Jx, 50).sum()*0.5 + \
        window_loss(Jy, 50).sum()*0.5 + \
        window_loss(Jxy, 400).sum()*0.5 + \
        window_loss(dn + toy.NoiseVariance(r0), 1.5).sum()
    return z

optimizer_lbfgs = OptimizeLBFGS(toy, parameters, loss_fn)

for i in range(10):
    optimizer_lbfgs.Optimize(PSF_0, [F, dx, dy], 3)
    optimizer_lbfgs.Optimize(PSF_0, [r0, dn, toy.WFS_Nph], 5)
    optimizer_lbfgs.Optimize(PSF_0, [bg], 2)
    optimizer_lbfgs.Optimize(PSF_0, [Jx, Jy, Jxy], 3)

PSF_1 = toy.PSD2PSF(*parameters)
print('\nStrehl ratio: ', SR(PSF_1, PSF_DL))


draw_result(PSF_0, PSF_1)
for i in range(PSF_0.shape[0]):
    plot_radial_profile(PSF_0[i,:,:], PSF_1[i,:,:], 'TipToy', title='IRDIS PSF', dpi=100)

#%%

reconstruction_result = np.save('C:/Users/akuznets/Data/SPHERE/PSF_TipToy.npy', PSF_1.detach().cpu().numpy())
initial_PSF = np.save('C:/Users/akuznets/Data/SPHERE/PSF_init.npy', PSF_0.detach().cpu().numpy())


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
    PSFs_3 = toy.PSD2PSF(*parameters)

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
