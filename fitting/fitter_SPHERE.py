#%%
# %reload_ext autoreload
# %autoreload 2

import os
import sys
sys.path.insert(0, '..')

import pickle
import torch
import numpy as np
from torch import nn, optim
from tools.static_phase import SR, pdims, BuildPTTBasis, decompose_WF, project_WF, calc_WFE, rad2mas, LWEBasis
from tools.utils import cropper, FitGauss2D
from PSF_models.TipTorch import TipTorch
from data_processing.SPHERE_create_STD_dataset import SPHERE_preprocess, SamplesByIds
from tools.normalizers import TransformSequence, Uniform,
from managers.input_manager import InputsTransformer
from managers.config_manager import GetSPHEREonsky, ConfigManager
from data_processing.project_settings import SPHERE_DATA_FOLDER, SPHERE_FITTING_FOLDER
from torch.autograd.functional import hessian
from torchmin import minimize
from project_settings import device

import matplotlib.pyplot as plt
from tools.plotting import plot_radial_PSF_profiles, draw_PSF_stack

import warnings

warnings.filterwarnings("ignore")

default_device = device
start_id, end_id = -1, -1

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

if len(sys.argv) == 4:
    start_id, end_id = int(sys.argv[2]), int(sys.argv[3])


# device = torch.device('cuda:0')
# device = torch.device('cuda:1')

#%% Initialize data sample
with open(SPHERE_DATA_FOLDER + 'sphere_df.pickle', 'rb') as handle:
    psf_df = pickle.load(handle)

psf_df = psf_df[psf_df['Corrupted'] == False]
# psf_df = psf_df[psf_df['Low quality'] == False]
# psf_df = psf_df[psf_df['Medium quality'] == False]
# psf_df = psf_df[psf_df['LWE'] == True]
# psf_df = psf_df[psf_df['mag R'] < 7]
# psf_df = psf_df[psf_df['Num. DITs'] < 50]
# psf_df = psf_df[psf_df['Class A'] == True]
# psf_df = psf_df[np.isfinite(psf_df['λ left (nm)']) < 1700]
# psf_df = psf_df[psf_df['Δλ left (nm)'] < 80]

good_ids = psf_df.index.values.tolist()

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
norm_F    = TransformSequence(transforms=[ Uniform(a=0.0,   b=1.0)  ])
norm_bg   = TransformSequence(transforms=[ Uniform(a=-1e-6, b=1e-6) ])
norm_r0   = TransformSequence(transforms=[ Uniform(a=0,     b=0.5)  ])
norm_dxy  = TransformSequence(transforms=[ Uniform(a=-1,    b=1)    ])
norm_J    = TransformSequence(transforms=[ Uniform(a=0,     b=30)   ])
norm_Jxy  = TransformSequence(transforms=[ Uniform(a=0,     b=50)   ])
norm_LWE  = TransformSequence(transforms=[ Uniform(a=-20,   b=20)   ])
norm_dn   = TransformSequence(transforms=[ Uniform(a=-0.02, b=0.02) ])
norm_FWHM = TransformSequence(transforms=[ Uniform(a=0,     b=5)    ])
norm_wind_spd = TransformSequence(transforms=[ Uniform(a=0, b=20)   ])
norm_wind_dir = TransformSequence(transforms=[ Uniform(a=0, b=360)  ])

# Dump transforms
transforms_dump = {
    'F R':         norm_F,
    'F L':         norm_F,
    'bg R':        norm_bg,
    'bg L':        norm_bg,
    'r0':          norm_r0,
    'dx L':        norm_dxy,
    'dx R':        norm_dxy,
    'dy L':        norm_dxy,
    'dy R':        norm_dxy,
    'dn':          norm_dn,
    'Jx':          norm_J,
    'Jy':          norm_J,
    'Jxy':         norm_Jxy,
    'Wind dir':    norm_wind_dir,
    'Wind speed':  norm_wind_spd,
    'LWE coefs':   norm_LWE,
    'FWHM fit L':  norm_FWHM,
    'FWHM fit R':  norm_FWHM,
    'FWHM data L': norm_FWHM,
    'FWHM data R': norm_FWHM
}

path_transforms = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '../data/temp/fitted_df_norm_transforms.pickle'
)

with open(path_transforms, 'wb') as handle:
    df_transforms_store = {}
    for entry in transforms_dump:
        df_transforms_store[entry] = transforms_dump[entry].store()
    pickle.dump(df_transforms_store, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

def to_store(x):
    if x is not None:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        else:
            return x
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


def GetNewPhotons(model):
    def func_Nph(x):
        model.WFS_Nph = x
        var = model.NoiseVariance(model.r0.abs())
        return (WFS_noise_var-var).flatten().abs().sum()
    
    try:
        WFS_noise_var = model.dn + model.NoiseVariance(model.r0.abs())
        N_ph_0 = model.WFS_Nph.clone()
        result_photons = minimize(func_Nph, N_ph_0, method='bfgs', disp=0)
        model.WFS_Nph = N_ph_0.clone()

        return result_photons.x
    
    except Exception as e:
        print(e)
        return None

#%%
def load_and_fit_sample(id):
    try:
        PSF_data, _, merged_config = SPHERE_preprocess(
            sample_ids    = [id],
            norm_regime   = 'sum',
            split_cube    = False,
            PSF_loader    = lambda x: SamplesByIds(x, synth=False),
            config_loader = GetSPHEREonsky,
            framework     = 'pytorch',
            device        = device)

        PSF_0    = PSF_data[0]['PSF (mean)'].unsqueeze(0)
        PSF_var  = PSF_data[0]['PSF (var)'].unsqueeze(0)
        PSF_mask = PSF_data[0]['mask (mean)'].unsqueeze(0)
        norms    = PSF_data[0]['norm (mean)']
        del PSF_data

        # if psf_df.loc[sample_id]['Nph WFS'] < 10:
        PSF_mask   = PSF_mask * 0 + 1
        # LWE_flag   = psf_df.loc[sample_id]['LWE']
        LWE_flag = True
        wings_flag = True #psf_df.loc[id]['Wings']

        
    except Exception as e:
        print(e)
        print('Failed to load sample', id)
        return None

    try:
        model = TipTorch(merged_config, None, device)

        _ = model()

        PSF_DL = model.DLPSF()

        basis = LWEBasis(model)

        transformer = InputsTransformer({
            'r0':  norm_r0,
            'F':   norm_F,
            'dx':  norm_dxy,
            'dy':  norm_dxy,
            'bg':  norm_bg,
            'dn':  norm_dn,
            'Jx':  norm_J,
            'Jy':  norm_J,
            'Jxy': norm_Jxy,
            'wind_speed':  norm_wind_spd,
            'wind_dir':    norm_wind_dir,
            'basis_coefs': norm_LWE
        })


        inp_dict = {}

        # Loop through the class attributes
        for attr in ['r0', 'F', 'dx', 'dy', 'bg', 'dn', 'Jx', 'Jy', 'Jxy']:
            inp_dict[attr] = getattr(model, attr)

        if wings_flag:
            inp_dict['wind_dir'] = model.wind_dir
            # inp_dict['wind_speed'] = toy.wind_speed
            
        if LWE_flag:
            inp_dict['basis_coefs'] = basis.coefs

        _ = transformer.stack(inp_dict) # to create index mapping


        x0 = [norm_r0.forward(model.r0).item(),
            1.0, 1.0,
            0.0, 0.0,
            0.0, 0.0,
            0.0, 0.0,
            0.0,
            0.5,
            0.5,
            0.1]

        if wings_flag:
            x0 = x0 + [
                norm_wind_dir.forward(model.wind_dir).item(),
                # norm_wind_spd.forward(model.wind_speed).item()
            ]   

        if LWE_flag:
            x0 = x0 + [0,]*4 + [0,]*8

        x0 = torch.tensor(x0).float().to(device).unsqueeze(0)
        
        def func(x_):
            x_torch = transformer.unstack(x_)
            if 'basis_coefs' in x_torch:
                return model(x_torch, None, lambda: basis(x_torch['basis_coefs'].float()))
            else:
                return model(x_torch)
            
        # crop_all = cropper(PSF_0, 100)
        
        if LWE_flag:
            A = 50.0

            pattern_pos = torch.tensor([[0,0,0,0,  0,-1,1,0,  1,0,0,-1]]).to(device).float() * A
            pattern_neg = torch.tensor([[0,0,0,0,  0,1,-1,0, -1,0,0, 1]]).to(device).float() * A
            pattern_1   = torch.tensor([[0,0,0,0,  0,-1,1,0, -1,0,0, 1]]).to(device).float() * A
            pattern_2   = torch.tensor([[0,0,0,0,  0,1,-1,0,  1,0,0,-1]]).to(device).float() * A
            pattern_3   = torch.tensor([[0,0,0,0,  1,0,0,-1,  0,1,-1,0]]).to(device).float() * A
            pattern_4   = torch.tensor([[0,0,0,0,  -1,0,0,1,  0,-1,1,0]]).to(device).float() * A

            gauss_penalty = lambda A, x, x_0, sigma: A * torch.exp(-torch.sum((x - x_0) ** 2) / (2 * sigma ** 2))
            img_punish = lambda x: ( (func(x)-PSF_0) * PSF_mask ).flatten().abs().sum()
            Gauss_err  = lambda pattern, coefs: (pattern * gauss_penalty(5, coefs, pattern, A/2)).flatten().abs().sum()
                    
            LWE_regularizer = lambda c: \
                Gauss_err(pattern_pos, c) + Gauss_err(pattern_neg, c) + \
                Gauss_err(pattern_1, c)   + Gauss_err(pattern_2, c) + \
                Gauss_err(pattern_3, c)   + Gauss_err(pattern_4, c)
            
            def loss_fn(x_):
                coefs_ = transformer.unstack(x_)['basis_coefs']
                loss = img_punish(x_) + LWE_regularizer(coefs_) + (coefs_**2).mean()*1e-4
                return loss
            
        else:
            def loss_fn(x_):
                loss = (func(x_)-PSF_0)*PSF_mask
                return loss.flatten().abs().sum()

        result = minimize(lambda x_: loss_fn(x_), x0, method='bfgs', disp=0)

        x0 = result.x

        LWE_coefs = transformer.unstack(x0)['basis_coefs'].clone()
        PTT_basis = BuildPTTBasis(model.pupil.cpu().numpy(), True).to(device).float()

        TT_max = PTT_basis.abs()[1,...].max().item()
        pixel_shift = lambda coef: 4 * TT_max * rad2mas / model.psInMas / model.D * 1e-9 * coef

        LWE_OPD   = torch.einsum('mn,nwh->mwh', LWE_coefs, basis.modal_basis)
        PPT_OPD   = project_WF  (LWE_OPD, PTT_basis, model.pupil)
        PTT_coefs = decompose_WF(LWE_OPD, PTT_basis, model.pupil)

        x0_new = transformer.unstack(x0)
        x0_new['basis_coefs'] = decompose_WF(LWE_OPD-PPT_OPD, basis.modal_basis, model.pupil) 
        x0_new['dx'] -= pixel_shift(PTT_coefs[:, 2])
        x0_new['dy'] -= pixel_shift(PTT_coefs[:, 1])
        x0 = transformer.stack(x0_new)

        PSF_1 = func(x0)
        
        with torch.no_grad():
            PSF_1 = func(x0)
            fig, ax = plt.subplots(1, 2, figsize=(10, 3))
            plot_radial_PSF_profiles( PSF_0[:,0,...].cpu().numpy(), PSF_1[:,0,...].cpu().numpy(), 'Data', 'TipTorch', title='Left PSF',  ax=ax[0] )
            plot_radial_PSF_profiles( PSF_0[:,1,...].cpu().numpy(), PSF_1[:,1,...].cpu().numpy(), 'Data', 'TipTorch', title='Right PSF', ax=ax[1] )
            plt.show()
        
            draw_PSF_stack(PSF_0, PSF_1, average=True, crop=80)
            
        plt.imshow((LWE_OPD-PPT_OPD).cpu().numpy()[0,...])
        plt.colorbar()
        
    except Exception as e:
        print(e)
        print('Failed to fit sample', id)
        return None

    hessian_mat  = None
    hessian_mat_ = None
    hessian_inv  = None
    variance_estimates = None
    L = None
    V = None

    try:
        hessian_mat  = hessian(lambda x_: loss_fn(x_).log(), x0).squeeze()
        hessian_mat_ = hessian_mat.clone()
        hessian_mat_ = hessian_mat_.nan_to_num()
        hessian_mat_[1:,0] = hessian_mat_[0,1:]
        hessian_mat_[hessian_mat.abs() < 1e-11] = 1e-11
        
    except Exception as e:
        pass
    else:
        try:
            hessian_inv = torch.inverse(hessian_mat_)
            variance_estimates = torch.diag(hessian_inv).abs()
            L_complex, V_complex = torch.linalg.eig(hessian_mat_)
            L = L_complex.real.cpu().numpy()
            V = V_complex.real.cpu().numpy()
            V = V / np.linalg.norm(V, axis=0)
            
        except Exception as e:
            pass

    config_manager = ConfigManager(GetSPHEREonsky())
    config_manager.Convert(merged_config, framework='numpy')

    save_data = {
        'comments':      'New LWE process',
        'config':        merged_config,
        'bg':            to_store(model.bg),
        'F':             to_store(model.F),
        'dx':            to_store(model.dx),
        'dy':            to_store(model.dy),
        'dn':            to_store(model.dn),
        'r0':            to_store(model.r0),
        'n':             to_store(model.NoiseVariance(model.r0.abs())),
        'Jx':            to_store(model.Jx),
        'Jy':            to_store(model.Jy),
        'Jxy':           to_store(model.Jxy),
        'Wind dir':      to_store(model.wind_dir),
        'Nph WFS':       to_store(model.WFS_Nph),
        'Nph WFS (new)': to_store(GetNewPhotons(model)),
        'LWE coefs':     to_store(basis.coefs),
        'Hessian':       to_store(hessian_mat_),
        'Variances':     to_store(variance_estimates),
        'Inv.H Ls':      to_store(L),
        'Inv.H Vs':      to_store(V),
        'SR data':       SR(PSF_0, PSF_DL).detach().cpu().numpy(),
        'SR fit':        SR(PSF_1, PSF_DL).detach().cpu().numpy(),
        'FWHM fit':      gauss_fitter(PSF_0),
        'FWHM data':     gauss_fitter(PSF_1),
        'Img. data':     to_store(PSF_0),
        'Img. fit':      to_store(PSF_1),
        'PSF mask':      to_store(PSF_mask),
        'PSD':           to_store(model.PSD),
        'Data norms':    to_store(norms),
        'Model norms':   to_store(model.norm_scale),
        'loss':          loss_fn(x0).item(),
        'Transforms':    df_transforms_store,
    }
    return save_data

#%%

# good_ids = [444]
# test = load_and_fit_sample(444)

#%%
for id in good_ids:
    filename = SPHERE_FITTING_FOLDER + str(id) + '.pickle'
    print('>>>>>>>>>>>> Fitting sample', id)
    save_data = load_and_fit_sample(id)
    if save_data is not None:
        with open(filename, 'wb') as handle:
            pickle.dump(save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
# %%
