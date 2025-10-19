#%%
# %reload_ext autoreload
# %autoreload 2

import sys
import os
from pathlib import Path

# Ensure the project root is in the Python path
if __name__ == '__main__':
    # When running as a script
    project_root = Path(__file__).parent.parent.resolve()
else:
    # When running interactively (cell by cell)
    project_root = Path(os.getcwd()).resolve()
    # Try to find the project root by looking for project_config.json
    while not (project_root / 'project_config.json').exists() and project_root != project_root.parent:
        project_root = project_root.parent

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'data_processing'))


import gc
import pickle
import torch
import numpy as np
from torch.autograd.functional import hessian
from torchmin import minimize

from data_processing.SPHERE_create_STD_dataset import LoadSTDStarData, process_mask, STD_FOLDER
from PSF_models.IRDIS_wrapper import PSFModelIRDIS
from managers.config_manager import ConfigManager
from tools.utils import SR, GradientLoss, mask_circle, FWHM_fitter
from tools.normalizers import TransformSequence, Uniform

from project_settings import device, DATA_FOLDER
import warnings

warnings.filterwarnings("ignore")

SPHERE_FITTING_FOLDER = STD_FOLDER / 'fitted'

# Create directory if it doesn't exist
SPHERE_FITTING_FOLDER.mkdir(parents=True, exist_ok=True)

default_device = device
start_id, end_id = -1, -1

if len(sys.argv) > 1:
    param1 = sys.argv[1]
else:
    device = default_device
    print("No device specified, using default device.")

if len(sys.argv) > 1 and (param1.startswith("cuda:") or param1.startswith("cpu")):
    device = torch.device(device_choice:=param1)
    print(f"Using device: {device_choice}")
else:
    device = default_device
    print("No device specified, using default device.")

if len(sys.argv) > 2:
    start_id, end_id = int(sys.argv[2]), int(sys.argv[3])

# Example: python fitter_SPHERE_wrapper.py cpu 100 200

#%% Initialize data sample
with open(STD_FOLDER / 'sphere_df.pickle', 'rb') as handle:
    psf_df = pickle.load(handle)

psf_df = psf_df[psf_df['Corrupted'] == False]
# Additional filtering can be uncommented as needed
# psf_df = psf_df[psf_df['Low quality'] == False]
# psf_df = psf_df[psf_df['Medium quality'] == False]
# psf_df = psf_df[psf_df['LWE'] == True]

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
def to_store(x):
    if x is None:
        return None
    elif isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, dict):
        return {k: to_store(v) for k, v in x.items()}
    else:
        return x
    

def GetNewPhotons(model):
    """Calculate new photon count based on current noise parameters"""
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
        print(f"Error in GetNewPhotons: {e}")
        return None


#%%
def load_and_fit_sample(id):
    """Load and fit a single sample using the IRDIS wrapper"""
    try:
        # Load data using the original loading function
        PSF_data, data_sample, model_config = LoadSTDStarData(
            id,
            normalize=True,
            subtract_background=True,
            ensure_odd_pixels=True,
            device=device
        )

        PSF_0 = PSF_data[0]['PSF (mean)']
        # PSF_var = PSF_data[0]['PSF (var)']
        PSF_mask = PSF_data[0]['mask (mean)']
        norms = PSF_data[0]['norm (mean)']
        del PSF_data

        # Process mask like in original
        PSF_mask = process_mask(PSF_mask)
        LWE_flag = True
        fit_wind = True
        use_Zernike = True

        # Handle central hole if present
        if psf_df.loc[id]['Central hole'] == True:
            circ_mask = 1-mask_circle(PSF_0.shape[-1], 3, centered=True)
            PSF_mask *= torch.tensor(circ_mask[None, None, ...]).to(device)

    except Exception as e:
        print(f"Failed to load sample {id}: {e}")
        return None

    try:
        # Initialize PSF model using wrapper
        PSF_model = PSFModelIRDIS(
            model_config,
            LWE_flag=LWE_flag,
            fit_wind=fit_wind,
            use_Zernike=use_Zernike,
            N_modes=9,
            LO_map_size=31,
            device=device
        )

        func = lambda x_: PSF_model(PSF_model.inputs_manager.unstack(x_))

        # Create loss functions (like in original and refactored version)
        img_loss = lambda x: ((func(x) - PSF_0) * PSF_mask).flatten().abs().sum()

        if LWE_flag:
            A = 50.0
            patterns = [
                [0,0,0,0,  0,-1,1,0,  1,0,0,-1], # pattern_pos
                [0,0,0,0,  0,1,-1,0, -1,0,0, 1], # pattern_neg
                [0,0,0,0,  0,-1,1,0, -1,0,0, 1], # pattern_1
                [0,0,0,0,  0,1,-1,0,  1,0,0,-1], # pattern_2
                [0,0,0,0,  1,0,0,-1,  0,1,-1,0], # pattern_3
                [0,0,0,0,  -1,0,0,1,  0,-1,1,0], # pattern_4
                [-1,1,1,-1,  0,0,0,0,  0,0,0,0], # pattern_piston_horiz
                [1,-1,-1,1,  0,0,0,0,  0,0,0,0]  # pattern_piston_vert
            ]
            patterns = [torch.tensor([p]).to(device).float() * A for p in patterns]
            
            gauss_penalty = lambda A, x, x_0, sigma: A * torch.exp(-torch.sum((x - x_0) ** 2) / (2 * sigma ** 2))
            Gauss_err = lambda pattern, coefs: (pattern * gauss_penalty(5, coefs, pattern, A/2)).flatten().abs().sum()
            LWE_regularizer = lambda c: sum(Gauss_err(pattern, c) for pattern in patterns)

            if not use_Zernike:  
                grad_loss_fn = GradientLoss(p=1, reduction='mean')

            def loss_fn(x_):
                loss = img_loss(x_)
                
                # Update inputs manager to access coefficients
                PSF_model.inputs_manager.unstack(x_, include_all=True, update=True)
                
                coefs_LWE = PSF_model.inputs_manager['LWE_coefs']
                coefs_LO = PSF_model.inputs_manager['LO_coefs']
                
                if use_Zernike:  
                    grad_loss = 0.0
                else:  
                    grad_loss = grad_loss_fn(coefs_LO.view(1, 1, PSF_model.LO_map_size, PSF_model.LO_map_size)) * 1e-4
                L2_loss_coefs = coefs_LO.pow(2).sum()*5e-9
                
                loss += LWE_regularizer(coefs_LWE) + (coefs_LWE**2).mean()*1e-4
                return loss + L2_loss_coefs + grad_loss 
        else:
            def loss_fn(x_):
                return img_loss(x_)


        # Perform fitting
        x0 = PSF_model.inputs_manager.stack()
        result = minimize(loss_fn, x0, max_iter=300, tol=1e-5, method='l-bfgs', disp=2)
        x0 = result.x


        # PTT compensation
        LWE_OPD, PPT_OPD, NCPA_OPD = PSF_model.compensate_PTT_coupling()
        x0_compensated = PSF_model.inputs_manager.stack()

        PSF_1 = func(x0_compensated)
        PSF_DL = PSF_model.model.DLPSF()

        loss_val = loss_fn(x0).item()


        # Get all fitted parameters
        fitted_values = to_store(PSF_model.inputs_manager.unstack(x0_compensated, include_all=True, update=True))

        # Calculate Hessian like in original
        hessian_mat = None
        hessian_mat_ = None
        hessian_inv = None
        variance_estimates = None
        L = None
        V = None

        try:
            hessian_mat = hessian(lambda x_: loss_fn(x_).log(), x0_compensated).squeeze()
            hessian_mat_ = hessian_mat.clone()
            hessian_mat_ = hessian_mat_.nan_to_num()
            hessian_mat_[1:,0] = hessian_mat_[0,1:]
            hessian_mat_[hessian_mat.abs() < 1e-11] = 1e-11
        except Exception as e:
            print(f"Warning: Hessian calculation failed for sample {id}: {e}")
        else:
            try:
                hessian_inv = torch.inverse(hessian_mat_)
                variance_estimates = torch.diag(hessian_inv).abs()
                L_complex, V_complex = torch.linalg.eig(hessian_mat_)
                L = L_complex.real.cpu().numpy()
                V = V_complex.real.cpu().numpy()
                V = V / np.linalg.norm(V, axis=0)
            except Exception as e:
                print(f"Warning: Hessian decomposition failed for sample {id}: {e}")

        # Prepare config for saving
        config_manager = ConfigManager()
        config_manager.Convert(model_config, framework='numpy')

        # Calculate OPD maps
        LWE_coefs_final = PSF_model.inputs_manager['LWE_coefs']
        
        # Get individual OPD components
        if LWE_flag and hasattr(PSF_model, 'LWE_basis'):
            LWE_OPD_final = PSF_model.LWE_basis.compute_OPD(LWE_coefs_final) * 1e9  # [nm]
        else:
            LWE_OPD_final = None
            
        if hasattr(PSF_model, 'NCPAs_basis'):
            if use_Zernike:
                NCPA_OPD_final = PSF_model.NCPAs_basis.compute_OPD(PSF_model.inputs_manager['LO_coefs']) * 1e9
            else:
                NCPA_OPD_final = PSF_model.NCPAs_basis.interp_upscale(
                    PSF_model.inputs_manager['LO_coefs'].view(1, PSF_model.LO_map_size, PSF_model.LO_map_size)
                )[0,...] 
        else:
            NCPA_OPD_final = None 

        # Prepare save data (preserve original structure)
        save_data = {
            'comments':      'New LWE process with IRDIS wrapper',
            'config':        model_config,
            'bg':            to_store(PSF_model.model.bg),
            'F':             to_store(PSF_model.model.F),
            'dx':            to_store(PSF_model.model.dx),
            'dy':            to_store(PSF_model.model.dy),
            'dn':            to_store(PSF_model.model.dn),
            'r0':            to_store(PSF_model.model.r0),
            'n':             to_store(PSF_model.model.NoiseVariance()),
            'Jx':            to_store(PSF_model.model.Jx),
            'Jy':            to_store(PSF_model.model.Jy),
            'Jxy':           to_store(PSF_model.model.Jxy),
            'Wind dir':      to_store(PSF_model.model.wind_dir),
            'Nph WFS':       to_store(PSF_model.model.WFS_Nph),
            'Nph WFS (new)': to_store(GetNewPhotons(PSF_model.model)),
            'LWE coefs':     to_store(LWE_coefs_final),
            'LO coefs':      to_store(PSF_model.inputs_manager['LO_coefs']),
            'LWE OPD':       to_store(LWE_OPD_final),
            'NCPA OPD':      to_store(NCPA_OPD_final),
            'Hessian':       to_store(hessian_mat_),
            'Variances':     to_store(variance_estimates),
            'Inv.H Ls':      to_store(L),
            'Inv.H Vs':      to_store(V),
            'SR data':       SR(PSF_0, PSF_DL).detach().cpu().numpy(),
            'SR fit':        SR(PSF_1, PSF_DL).detach().cpu().numpy(),
            'FWHM fit':      FWHM_fitter(PSF_0.detach().cpu().numpy(), verbose=False).squeeze(),
            'FWHM data':     FWHM_fitter(PSF_1.detach().cpu().numpy(), verbose=False).squeeze(),
            'Img. data':     to_store(PSF_0),
            'Img. fit':      to_store(PSF_1),
            'PSF mask':      to_store(PSF_mask),
            'PSD':           to_store(PSF_model.model.PSD),
            'Data norms':    to_store(norms),
            'Model norms':   to_store(PSF_model.model.norm_scale) if hasattr(PSF_model.model, 'norm_scale') else None,
            'loss':          loss_val,
            'use_Zernike':   use_Zernike,
            'LWE_flag':      LWE_flag,
            'fit_wind':      fit_wind,
        }

        # Add fitted parameters to save data
        save_data.update(fitted_values)

        # Clean up
        del PSF_0, PSF_1, PSF_DL, PSF_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return save_data

    except Exception as e:
        print(f"Failed to fit sample {id}: {e}")
        import traceback
        traceback.print_exc()
        return None


#%%
# Test with a single sample (uncomment for testing)
# good_ids = [1778]  # Use a known good sample
# test_result = load_and_fit_sample(1778)

# if test_result is not None:
#     print("Test successful!")
#     print(f"Loss: {test_result['loss']:.6f}")
#     print(f"SR data: {test_result['SR data']:.3f}")
#     print(f"SR fit: {test_result['SR fit']:.3f}")


#%%
for id in good_ids:
    try:
        filename = SPHERE_FITTING_FOLDER / f'{id}.pickle'
        print(f'>>>>>>>>>>>> Fitting sample {id}')
        save_data = load_and_fit_sample(id)
        if save_data is not None:
            with open(filename, 'wb') as handle:
                pickle.dump(save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f'Successfully saved {filename}')

    except Exception as e:
        print(f' ============= Error fitting sample {id} ===========')
        print(e)
        import traceback
        traceback.print_exc()

# %%