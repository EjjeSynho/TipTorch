#%%
%reload_ext autoreload
%autoreload 2

import sys
sys.path.insert(0, '..')

import pickle
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from tools.plotting import plot_radial_PSF_profiles, plot_radial_PSF_profiles_relative, SR, draw_PSF_stack, rad2mas, mask_circle
from data_processing.MUSE_preproc_utils import GetConfig, LoadImages, LoadMUSEsampleByID, rotate_PSF
from project_settings import MUSE_DATA_FOLDER, device
from torchmin import minimize
from tools.normalizers import CreateTransformSequenceFromFile
from managers.input_manager import InputsTransformer
from tqdm import tqdm
from project_settings import MUSE_DATA_FOLDER
from machine_learning.MUSE_onsky_df import *

# Load auxiliary data
with open(MUSE_DATA_FOLDER+'muse_df_norm_imputed.pickle', 'rb') as handle:
    muse_df_norm = pickle.load(handle)
    
df_transforms_onsky  = CreateTransformSequenceFromFile('../data/temp/muse_df_norm_transforms.pickle')
df_transforms_fitted = CreateTransformSequenceFromFile('../data/temp/muse_df_fitted_transforms.pickle')

#%%
with open(MUSE_DATA_FOLDER+'/muse_df.pickle', 'rb') as handle:
    muse_df = pickle.load(handle)
    
derotate_PSF    = True
Moffat_absorber = True
include_sausage = True

# sample_id = 296
sample_id = 292

data_sample = LoadMUSEsampleByID(sample_id)

PSF_onsky, var_mask, norms, bgs = LoadImages(data_sample)
config_file, PSF_onsky = GetConfig(data_sample, PSF_onsky)
N_wvl = PSF_onsky.shape[1]

PSF_onsky = PSF_onsky[...,1:,1:]

config_file['sensor_science']['FieldOfView'] = PSF_onsky.shape[-1]
config_file['NumberSources'] = config_file['NumberSources'].int().item()

# Derotate PSF
PSF_onsky = rotate_PSF(PSF_onsky, -data_sample['All data']['Pupil angle'].item())
config_file['telescope']['PupilAngle'] = 0

# Select only 7 wavelengths
wavelength = config_file['sources_science']['Wavelength'].clone()
ids_wavelength_selected = np.arange(0, wavelength.shape[-1], 2)
wavelength_selected = wavelength[..., ids_wavelength_selected]
config_file['sources_science']['Wavelength'] = wavelength_selected
N_wvl = len(ids_wavelength_selected)

PSF_onsky = PSF_onsky[..., ids_wavelength_selected, :, :]
var_mask  =  var_mask[..., ids_wavelength_selected, :, :]


#%% Initialize the model
# from PSF_models.TipToy_MUSE_multisrc import TipTorch
from PSF_models.TipTorch import TipTorch
from tools.utils import SausageFeature
from tools.utils import PupilVLT

pupil = torch.tensor( PupilVLT(samples=320, rotation_angle=0), device=device )
PSD_include = {
    'fitting':         True,
    'WFS noise':       True,
    'spatio-temporal': True,
    'aliasing':        False,
    'chromatism':      True,
    'diff. refract':   True,
    'Moffat':          True
}
PSF_model = TipTorch(config_file, 'LTAO', pupil, PSD_include, 'sum', device, oversampling=1)
PSF_model.to_float()

sausage_absorber = SausageFeature(PSF_model)
sausage_absorber.OPD_map = sausage_absorber.OPD_map.flip(dims=(-1,-2))

inputs_tiptorch = {
    # 'r0':  torch.tensor([0.09561153075597545], device=toy.device),
    'F':   torch.tensor([[1.0,]*N_wvl], device=PSF_model.device),
    # 'L0':  torch.tensor([47.93], device=toy.device),
    'dx':  torch.tensor([[0.0,]*N_wvl], device=PSF_model.device),
    'dy':  torch.tensor([[0.0,]*N_wvl], device=PSF_model.device),
    # 'dx':  torch.tensor([[0.0]], device=toy.device),
    # 'dy':  torch.tensor([[0.0]], device=toy.device),
    'bg':  torch.tensor([[1e-06,]*N_wvl], device=PSF_model.device),
    'dn':  torch.tensor([1.5], device=PSF_model.device),
    'Jx':  torch.tensor([[10,]*N_wvl], device=PSF_model.device),
    'Jy':  torch.tensor([[10,]*N_wvl], device=PSF_model.device),
    # 'Jx':  torch.tensor([[10]], device=toy.device),
    # 'Jy':  torch.tensor([[10]], device=toy.device),
    'Jxy': torch.tensor([[45]], device=PSF_model.device)
}

if Moffat_absorber:
    inputs_psfao = {
        'amp':   torch.ones (PSF_model.N_src, device=PSF_model.device)*0.0, # Phase PSD Moffat amplitude [rad²]
        'b':     torch.ones (PSF_model.N_src, device=PSF_model.device)*0.0, # Phase PSD background [rad² m²]
        'alpha': torch.ones (PSF_model.N_src, device=PSF_model.device)*0.1, # Phase PSD Moffat alpha [1/m]
        'beta':  torch.ones (PSF_model.N_src, device=PSF_model.device)*2,   # Phase PSD Moffat beta power law
        'ratio': torch.ones (PSF_model.N_src, device=PSF_model.device),     # Phase PSD Moffat ellipticity
        'theta': torch.zeros(PSF_model.N_src, device=PSF_model.device),     # Phase PSD Moffat angle
    }
else:
    inputs_psfao = {}


#%%
NN_inp = torch.as_tensor(muse_df_norm.loc[sample_id].to_numpy(), device=device, dtype=torch.float32).unsqueeze(0)

#%%
transforms = {
    'r0':    df_transforms_fitted['r0'],
    'F':     df_transforms_fitted['F'],
    'bg':    df_transforms_fitted['bg'],
    'dx':    df_transforms_fitted['dx'],
    'dy':    df_transforms_fitted['dy'],
    'Jx':    df_transforms_fitted['Jx'],
    'Jy':    df_transforms_fitted['Jy'],
    'Jxy':   df_transforms_fitted['Jxy'],
    'dn':    df_transforms_fitted['dn'],
    's_pow': df_transforms_fitted['s_pow'],
    'amp':   df_transforms_fitted['amp'],
    'b':     df_transforms_fitted['b'],
    'alpha': df_transforms_fitted['alpha'],
    'beta':  df_transforms_fitted['beta'],
    'ratio': df_transforms_fitted['ratio'],
    'theta': df_transforms_fitted['theta']
}

predicted_entries  = ['r0', 'F', 'dn', 'Jx', 'Jy', 's_pow', 'amp', 'b', 'alpha']

normalizer = InputsTransformer({ entry: transforms[entry] for entry in predicted_entries })

inp_dict = {
    'r0':    torch.ones ( PSF_model.N_src, device=PSF_model.device)*0.1,
    'F':     torch.ones ([PSF_model.N_src, N_wvl], device=PSF_model.device),
    'Jx':    torch.ones ([PSF_model.N_src, N_wvl], device=PSF_model.device)*10,
    'Jy':    torch.ones ([PSF_model.N_src, N_wvl], device=PSF_model.device)*10,
    'dn':    torch.ones (PSF_model.N_src, device=PSF_model.device)*1.5,
    's_pow': torch.zeros(PSF_model.N_src, device=PSF_model.device),
    'amp':   torch.zeros(PSF_model.N_src, device=PSF_model.device),
    'b':     torch.zeros(PSF_model.N_src, device=PSF_model.device),
    'alpha': torch.ones (PSF_model.N_src, device=PSF_model.device)*0.1,
}

inp_dict_ = { entry: inp_dict[entry] for entry in predicted_entries if entry in inp_dict.keys() }
_ = normalizer.stack(inp_dict_, no_transform=True)

#%%
class Gnosis(nn.Module):
    def __init__(self, in_size, out_size, hidden_size=100, dropout_p=0.25):
        super(Gnosis, self).__init__()
        self.fc1 = nn.Linear(in_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.dropout3 = nn.Dropout(dropout_p)
        self.fc4 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.dropout3(x)
        x = torch.tanh(self.fc4(x))
        return x
    
# Initialize the network, loss function and optimizer
net = Gnosis(NN_inp.shape[-1], normalizer.get_stacked_size(), 200, 0.1)
net.to(device)
net.float()

net.load_state_dict(torch.load('../data/weights/gnosis_MUSE_v3_7wvl_yes_Mof_no_ssg.dict', map_location=torch.device('cpu')))
net.eval()

with torch.no_grad():
    pred_inputs = normalizer.unstack(net(NN_inp))
    PSF_pred = PSF_model(pred_inputs, None, lambda: sausage_absorber(pred_inputs['s_pow'].flatten()))

    
# %%
wvl_select = np.s_[0, 3, 6]

draw_PSF_stack( PSF_onsky.cpu().numpy()[0, wvl_select, ...], PSF_pred.cpu().numpy()[0, wvl_select, ...], average=True, crop=120 )

PSF_disp = lambda x, w: (x[0,w,...]).cpu().numpy()
fig, ax = plt.subplots(1, len(wvl_select), figsize=(10, len(wvl_select)))
for i, lmbd in enumerate(wvl_select):
    plot_radial_PSF_profiles( PSF_disp(PSF_onsky, lmbd),  PSF_disp(PSF_pred, lmbd),  'Data', 'TipTorch', cutoff=30,  y_min=3e-1, linthresh=1e-2, ax=ax[i] )
plt.show()

#%%

plt.imshow(PSF_pred[0,...].mean(dim=0).log10().cpu().numpy(), vmin=0, vmax=0.01)
plt.show()

# %%
from managers.config_manager import ConfigManager
from copy import deepcopy

config_list = deepcopy(config_file)

config_manager = ConfigManager()
config_manager.Convert(config_list, framework='list', device=device)

del config_list['NumberSources']
del config_list['sensor_science']['Saturation']

config_list['telescope']['Resolution'] = int(config_list['telescope']['Resolution'])
config_list['telescope']['TechnicalFoV'] = 120

for key in ['Seeing', 'L0', 'Cn2Weights', 'Cn2Heights', 'WindSpeed', 'WindDirection']:
    config_list['atmosphere'][key] = config_list['atmosphere'][key][0]

for x in ['SizeLenslets', 'SpectralBandwidth', 'Transmittance', 'Dispersion', 'NoiseVariance', 'WfsType', 'ClockRate']:
    del config_list['sensor_LO'][x]

config_list['sensor_LO']['FieldOfView'] = int(config_list['sensor_LO']['FieldOfView'])
config_list['sensor_LO']['Binning'] = int(config_list['sensor_LO']['Binning'])

config_list['telescope']['ZenithAngle'] = config_list['telescope']['ZenithAngle'][0]
config_list['telescope']['Azimuth'] = config_list['telescope']['Azimuth'][0]

config_list['sensor_science']['FieldOfView'] = int(config_list['sensor_science']['FieldOfView']) + 1

config_list['sources_science']['Wavelength'] = config_list['sources_science']['Wavelength'][0]
config_list['sources_HO']['Wavelength'] = config_list['sources_HO']['Wavelength'][0]

config_list['sensor_LO']['NumberLenslets'] = [int(config_list['sensor_LO']['NumberLenslets'][0])]
config_list['sensor_HO']['NumberLenslets'] = [int(config_list['sensor_HO']['NumberLenslets']),]*4
config_list['sensor_HO']['SizeLenslets']   = [config_list['sensor_HO']['SizeLenslets'],]*4

noise_var_HO = (PSF_model.dn.unsqueeze(-1) + PSF_model.NoiseVariance()).abs().mean().cpu().numpy().item()
config_list['sensor_HO']['NoiseVariance'] = [noise_var_HO]
config_list['sensor_LO']['NoiseVariance'] = [None]
config_list['sensor_LO']['Gain'] = config_list['sensor_LO']['Gain'][0]

config_list['sensor_science']['SpotFWHM'] = [[PSF_model.Jx.mean().cpu().item(), PSF_model.Jy.mean().cpu().item(), 0.0]]

# config_list['sensor_science']['Transmittance'] = [1.0,] * N_wvl
# config_list['sensor_science']['Dispersion']    = [[0.0,]*N_wvl, [0.0,]*N_wvl]

config_list['sensor_science']['Name'] = 'SCIENCE CAM'

for key in ['LoopGain_HO', 'SensorFrameRate_HO', 'LoopDelaySteps_HO', 'SensorFrameRate_LO']:
    config_list['RTC'][key] = config_list['RTC'][key][0]

config_list['DM']['NumberActuators'] = [int(config_list['DM']['NumberActuators'][0])]
config_list['DM']['NumberReconstructedLayers'] = int(config_list['DM']['NumberReconstructedLayers'])


#%%
def convert_value(val):
    """
    Recursively convert a value to a string suitable for .ini output.
    - For None, return an empty string.
    - For lists or tuples, format as [item1, item2, ...].
    - For int and float values, return their string representation (unquoted).
    - For strings, replace occurrences of '..\' with '/' and '\' with '/',
      then wrap the result in single quotes.
    - Otherwise, fallback to str(val).
    """
    if val is None:
        return ''
    elif isinstance(val, (list, tuple)):
        # Recursively convert each element in the list/tuple.
        inner = ', '.join(convert_value(item) for item in val)
        return f'[{inner}]'
    elif isinstance(val, (int, float)):
        return str(val)
    elif isinstance(val, str):
        # Replace "..\" with "/" and then replace remaining "\" with "/"
        modified_val = val.replace("..\\", "/").replace("\\", "/")
        return f"'{modified_val}'"
    else:
        return str(val)


def save_ini(config_dict, filename):
    """
    Given a dictionary with section names as keys and dictionaries as values,
    write an .ini file to the given filename.
    Keys with value None are skipped.
    """
    with open(filename, 'w') as f:
        for section, params in config_dict.items():
            f.write(f'[{section}]\n')
            for key, value in params.items():
                # Skip keys with a None value
                if value is None:
                    continue
                value_str = convert_value(value)
                f.write(f'{key} = {value_str}\n')
            f.write('\n')  # add a newline between sections
            

output_file = data_sample['All data']['name'].loc[0].replace('.fits','.ini')

save_ini(config_list, '../data/parameter_files/' + output_file)


# %%
from astropy.io import fits

path = 'C:/Users/akuznets/Desktop/M.MUSE.2022-11-02T17-33-46.586.fits'

# Open FITS
with fits.open(path) as hdul:
    # Access the primary HDU (header data unit)
    PSF = hdul[1].data.squeeze()
    
#%%

# plt.imshow(np.log10(PSF.mean(axis=0)), origin='lower')
plt.imshow(np.log10(PSF[3,90:110,90:110]), origin='lower')
plt.show()