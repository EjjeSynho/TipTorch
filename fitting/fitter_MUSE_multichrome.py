#%%
# %reload_ext autoreload
# %autoreload 2

import sys
sys.path.insert(0, '..')

import os
import pickle
import torch
import numpy as np
from tools.utils import SR, FitGauss2D, SausageFeature
from PSF_models.TipToy_MUSE_multisrc import TipTorch
from data_processing.MUSE_preproc_utils import GetConfig, LoadImages, LoadMUSEsampleByID, rotate_PSF, GetRadialBackround
from project_globals import MUSE_DATA_FOLDER, MUSE_FITTING_FOLDER, device
from tools.config_manager import ConfigManager
from torchmin import minimize

from data_processing.normalizers import TransformSequence, Uniform, InputsTransformer
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")

default_device = device
start_id, end_id = -1, -1
to_float = True

if len(sys.argv) > 1:
    param1 = sys.argv[1]
else:
    device = default_device
    print("No device specified, using default device.")

if param1.startswith("cuda:") or param1.startswith("cpu"):
    device = torch.device(device_choice:=param1)
    print(f"Using device: {device_choice}")
else:
    device = default_device
    print("No device specified, using default device.")

if len(sys.argv) > 2:
    start_id, end_id = int(sys.argv[2]), int(sys.argv[3])

if len(sys.argv) > 4:
    if sys.argv[4]   == '-d': to_float = False
    elif sys.argv[4] == '-f': to_float = True
    else:
        print("Unknown option. Use -d for double precision and -f for float precision.")

#% Initialize data sample
with open(MUSE_DATA_FOLDER + 'muse_df.pickle', 'rb') as handle:
    psf_df = pickle.load(handle)

psf_df = psf_df[psf_df['Corrupted']   == False]
psf_df = psf_df[psf_df['Bad quality'] == False]

good_ids = psf_df.index.values.tolist()
# good_ids = [103, 240, 259, 264, 296, 319, 349, 367, 407]

# If no start or end ID is specified, use the first and last good IDs
if start_id == -1:
    start_id = good_ids[0]
else:
    start_id = min(good_ids, key=lambda x:abs(x-start_id))

if end_id == -1:
    end_id = good_ids[-1]
else:
    end_id = min(good_ids, key=lambda x:abs(x-end_id))

good_ids = good_ids[good_ids.index(start_id):good_ids.index(end_id)+1]

print(
    f"Device:   {device}\n"
    f"Start ID: {start_id}\n"
    f"End ID:   {end_id}"
)


#%%
def to_store(x):
    if x is not None:
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
    else:
        return None


def gauss_fitter(PSF_stack):
    FWHMs = np.zeros([PSF_stack.shape[0], PSF_stack.shape[1], 2])
    for i in range(PSF_stack.shape[0]):
        for l in range(PSF_stack.shape[1]):
            f_x, f_y = FitGauss2D(PSF_stack[i,l,:,:].float())
            FWHMs[i,l,0] = f_x.item()
            FWHMs[i,l,1] = f_y.item()
    return FWHMs


def compress_PSDs(PSDs_dict):
    PSDs_comressed = {}
    for key, values in PSDs_dict.items():
        buf = np.copy(to_store(values.squeeze()))
        # Cut half of the PSD with center preserved
        if buf.ndim > 0:
            if key == 'fitting':
                # Anyway, it is radially symmetric
                buf = buf[..., :buf.shape[-1]//2+1, :buf.shape[-1]//2+1]
            else:
                buf = buf[..., :buf.shape[-1]//2+1]

            PSDs_comressed[key] = buf.astype(np.float32)
            # print(key, buf.shape)
    return PSDs_comressed


# TipTorch
norm_F     = TransformSequence(transforms=[ Uniform(a=0.0,   b=1.0) ])
norm_bg    = TransformSequence(transforms=[ Uniform(a=-5e-6, b=5e-6)])
norm_r0    = TransformSequence(transforms=[ Uniform(a=0,     b=1)   ])
norm_dxy   = TransformSequence(transforms=[ Uniform(a=-1,    b=1)   ])
norm_J     = TransformSequence(transforms=[ Uniform(a=0,     b=50)  ])
norm_Jxy   = TransformSequence(transforms=[ Uniform(a=-180,  b=180) ])
norm_dn    = TransformSequence(transforms=[ Uniform(a=0,     b=5)   ])
# PSFAO
norm_amp   = TransformSequence(transforms=[ Uniform(a=0,     b=10)  ])
norm_b     = TransformSequence(transforms=[ Uniform(a=0,     b=0.1) ])
norm_alpha = TransformSequence(transforms=[ Uniform(a=-1,    b=1)   ])
norm_beta  = TransformSequence(transforms=[ Uniform(a=0,     b=10)  ])
norm_ratio = TransformSequence(transforms=[ Uniform(a=0,     b=2)   ])
norm_theta = TransformSequence(transforms=[ Uniform(a=-np.pi/2, b=np.pi/2)])
# To absord the phase feature appearing in the PSF (yes, we called it the sausage)
norm_sausage_pow = TransformSequence(transforms=[ Uniform(a=0, b=1) ])

Moffat_absorber = True
include_sausage = True
derotate_PSF = True

#%%
def load_and_fit_sample(id):
    # id = 240
    
    sample = LoadMUSEsampleByID(id)
    PSF_0, _, norms, bgs = LoadImages(sample, device)
    config_file, PSF_0 = GetConfig(sample, PSF_0, None, device)
    N_wvl = PSF_0.shape[1]

    if derotate_PSF:
        PSF_0 = rotate_PSF(PSF_0, -sample['All data']['Pupil angle'].item()) 
        config_file['telescope']['PupilAngle'] = 0

    #% Initialize the model
    model = TipTorch(config_file, 'sum', device, TipTop=True, PSFAO=Moffat_absorber, oversampling=1)
    sausage_absorber = SausageFeature(model)
    sausage_absorber.OPD_map = sausage_absorber.OPD_map.flip(dims=(-1,-2))

    model.PSD_include['fitting'] = True
    model.PSD_include['WFS noise'] = True
    model.PSD_include['spatio-temporal'] = True
    model.PSD_include['aliasing'] = False
    model.PSD_include['chromatism'] = True
    model.PSD_include['Moffat'] = Moffat_absorber

    if(to_float):
        model.to_float()
    else:
        model.to_double()

    inputs_tiptorch = {
        'F':   torch.tensor([[1.0,]*N_wvl], device=model.device),
        'dx':  torch.tensor([[0.0,]*N_wvl], device=model.device),
        'dy':  torch.tensor([[0.0,]*N_wvl], device=model.device),
        'bg':  torch.tensor([[1e-06,]*N_wvl], device=model.device),
        'dn':  torch.tensor([1.5], device=model.device),
        'Jx':  torch.tensor([[10,]*N_wvl], device=model.device),
        'Jy':  torch.tensor([[10,]*N_wvl], device=model.device),
        'Jxy': torch.tensor([[45]], device=model.device)
    }

    if Moffat_absorber:
        inputs_psfao = {
            'amp':   torch.ones (model.N_src, device=model.device)*0.0,   # Phase PSD Moffat amplitude [rad²]
            'b':     torch.ones (model.N_src, device=model.device)*0.0,  # Phase PSD background [rad² m²]
            'alpha': torch.ones (model.N_src, device=model.device)*0.1,   # Phase PSD Moffat alpha [1/m]
            'beta':  torch.ones (model.N_src, device=model.device)*2,     # Phase PSD Moffat beta power law
            'ratio': torch.ones (model.N_src, device=model.device),       # Phase PSD Moffat ellipticity
            'theta': torch.zeros(model.N_src, device=model.device),       # Phase PSD Moffat angle
        }
    else:
        inputs_psfao = {}

    _ = model(x = inputs_tiptorch | inputs_psfao)
    PSF_DL = model.DLPSF()


    # Initialize params
    # The order of definition matters here!
    transformer_dict = {
        'r0':  norm_r0,
        'F':   norm_F,
        'dx':  norm_dxy,
        'dy':  norm_dxy,
        'bg':  norm_bg,
        'dn':  norm_dn,
        'Jx':  norm_J,
        'Jy':  norm_J,
        'Jxy': norm_Jxy,
    }
    if include_sausage:
        transformer_dict.update({
            's_pow': norm_sausage_pow
        })

    if Moffat_absorber:
        transformer_dict.update({
            'amp'  : norm_amp,
            'b'    : norm_b,
            'alpha': norm_alpha,
        })


    transformer = InputsTransformer(transformer_dict)

    if include_sausage:
        setattr(model, 's_pow', torch.zeros([1,1], device=model.device).float())

    _ = transformer.stack({ attr: getattr(model, attr) for attr in transformer_dict }) # to create index mapping

    x0 = [\
        norm_r0.forward(model.r0).item(), # r0
        *([1.0,]*N_wvl),  # F
        *([0.0,]*N_wvl),  # dx
        *([0.0,]*N_wvl),  # dy
        *([0.0,]*N_wvl),  # bg
        0.7,              # dn
        *([-0.9,]*N_wvl), # Jx
        *([-0.9,]*N_wvl), # Jy
        0.0,              # Jxy
    ]
    # PSFAO realm
    if Moffat_absorber:
        x0 += [\
            *([ -0.9,]*N_wvl),
            *([-0.5,]*N_wvl),
            *([ -0.9,]*N_wvl),
        ]
    # If sausage absorber enabled
    if include_sausage:
        x0 += [0.5]

    x0 = torch.tensor(x0).float().to(device).unsqueeze(0)

    def func(x_, include_list=None):
        x_torch = transformer.destack(x_)
        
        if include_sausage and 's_pow' in x_torch:
            phase_func = lambda: sausage_absorber(model.s_pow.flatten())
        else:
            phase_func = None
        
        if include_list is not None:
            return model({ key: x_torch[key] for key in include_list }, None, phase_generator=phase_func)
        else:
            return model(x_torch, None, phase_generator=phase_func)

    # Loss functions
    wvl_weights = torch.linspace(1.0, 0.5, N_wvl).to(device).view(1, N_wvl, 1, 1) * 2

    def loss_MSE(x_):
        diff = (func(x_) - PSF_0) * wvl_weights
        return diff.pow(2).sum() * 200 / PSF_0.shape[0] / PSF_0.shape[1]

    def loss_MAE(x_):
        diff = (func(x_) - PSF_0) * wvl_weights
        return diff.abs().sum() / PSF_0.shape[0] / PSF_0.shape[1]

    def loss_fn(x_):
        return loss_MSE(x_) + loss_MAE(x_) * 0.4


    #% PSF fitting
    _ = func(x0)

    result = minimize(loss_MAE, x0, max_iter=100, tol=1e-3, method='bfgs', disp=0)
    x0 = result.x

    result = minimize(loss_fn, x0, max_iter=100, tol=1e-3, method='bfgs', disp=0)
    x0 = result.x

    x_torch = transformer.destack(x0)

    if include_sausage:
        PSF_1 = model(x_torch, None, phase_generator=lambda: sausage_absorber(model.s_pow.flatten()))
    else:
        PSF_1 = model(x_torch)


    #%
    config_manager = ConfigManager()
    config_manager.Convert(config_file, framework='numpy')

    df_transforms_store = { entry: transformer_dict[entry].store() for entry in transformer_dict }

    save_data = {
        'comments':      '1st multichrom fitting',
        'config':        config_file,
        'Moffat':        Moffat_absorber,
        'Sausage':       include_sausage,
        'bg':            to_store(model.bg),
        'F':             to_store(model.F),
        'dx':            to_store(model.dx),
        'dy':            to_store(model.dy),
        'n':             to_store(model.NoiseVariance(model.r0.abs())),
        'dn':            to_store(model.dn),
        'r0':            to_store(model.r0),
        'Jx':            to_store(model.Jx),
        'Jy':            to_store(model.Jy),
        'Jxy':           to_store(model.Jxy),
        'sausage_pow':   to_store(model.s_pow) if include_sausage else None,
        'amp':           to_store(model.amp)   if Moffat_absorber else None,
        'b':             to_store(model.b)     if Moffat_absorber else None,
        'alpha':         to_store(model.alpha) if Moffat_absorber else None,
        'Nph WFS':       to_store(model.WFS_Nph),
        'SR data':       SR(PSF_0, PSF_DL).detach().cpu().numpy(),
        'SR fit':        SR(PSF_1, PSF_DL).detach().cpu().numpy(),
        'FWHM fit':      gauss_fitter(PSF_0),
        'FWHM data':     gauss_fitter(PSF_1),
        'Img. data':     to_store(PSF_0.float()),
        'Img. fit':      to_store(PSF_1.float()),
        'PSD':           compress_PSDs(model.PSDs),
        'Data norms':    to_store(norms),
        'Model norms':   to_store(model.norm_scale),
        'loss':          loss_fn(x0).item(),
        'Transforms':    df_transforms_store
    }
    return save_data

#%%

for id in good_ids:
    try:
        filename = MUSE_FITTING_FOLDER + str(id) + '.pickle'
        print('>>>>>>>>>>>> Fitting sample', id)
        save_data = load_and_fit_sample(id)
        if save_data is not None:
            with open(filename, 'wb') as handle:
                pickle.dump(save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    except Exception as e:
        print(' ============= Error fitting sample ===========', id)
        print(e)

# %%
