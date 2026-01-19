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

from torchmin import minimize

from data_processing.MUSE_STD_dataset_utils import *
from PSF_models.NFM_wrapper import PSFModelNFM
from managers.config_manager import ConfigManager
from tools.utils import SR, GradientLoss, FWHM_fitter
from tools.static_phase import ArbitraryBasis, PixelmapBasis, ZernikeBasis

from project_settings import device
import warnings


MUSE_FITTING_FOLDER = STD_FOLDER / 'fitted/'

warnings.filterwarnings("ignore")

default_device = device

start_id, end_id = -1, -1

# Flexible argument parsing
input_args = sys.argv[1:]
device_specified = False

for arg in input_args:
    if arg.startswith("cuda") or arg.startswith("cpu"):
        device = torch.device(arg)
        print(f"Using device: {arg}")
        device_specified = True
    elif arg.lstrip('-').isdigit(): # Simple check for integer IDs
        if start_id == -1:
            start_id = int(arg)
        elif end_id == -1:
            end_id = int(arg)
    else:
        # Assume it is a path
        MUSE_FITTING_FOLDER = Path(arg)
        print(f"Output folder set to: {MUSE_FITTING_FOLDER}")

if not device_specified:
    print("No device specified, using default device.")

if not MUSE_FITTING_FOLDER.exists():
    try:
        MUSE_FITTING_FOLDER.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Warning: Could not create directory {MUSE_FITTING_FOLDER}: {e}")

# Example: python fitter_MUSE.py cpu 50 150

#% Initialize data sample
with open(MUSE_DATA_FOLDER / 'standart_stars' / 'muse_df.pickle', 'rb') as handle:
    psf_df = pickle.load(handle)

psf_df = psf_df[psf_df['Corrupted']   == False]
psf_df = psf_df[psf_df['Bad quality'] == False]

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


def to_store(x):
    if x is None:
        return None
    elif isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, dict):
        return {k: to_store(v) for k, v in x.items()}
    else:
        return x


wvl_ids = np.clip(np.arange(0, (N_wvl_max:=30)+1, 2), a_min=0, a_max=N_wvl_max-1)

#%%
def load_and_fit_sample(id):
    # Load data
    PSF_0, norms, bgs, configs = LoadSTDStarData(
        ids                 = [id],
        derotate_PSF        = True,
        normalize           = True,
        subtract_background = True,
        wvl_ids             = wvl_ids,
        ensure_odd_pixels   = True,
        device              = device
    )

    # Initialize the model
    PSF_model = PSFModelNFM(
        configs,
        multiple_obs    = True,
        LO_NCPAs        = True,
        chrom_defocus   = False,
        use_splines     = True,
        Moffat_absorber = False,
        Z_mode_max      = 9,
        device          = device
    )

    func = lambda x_: PSF_model( PSF_model.inputs_manager.unstack(x_) )
    PSF_1 = func(x0 := PSF_model.inputs_manager.stack() )

    N_wvl = PSF_0.shape[1]
    N_src = PSF_0.shape[0]
    wavelengths = PSF_model.wavelengths

    # Loss functions setup
    wvl_weights = torch.linspace(0.5, 1.0, N_wvl).to(device).view(1, N_wvl, 1, 1)
    # wvl_weights = torch.linspace(1.0, 0.5, N_wvl).to(device).view(1, N_wvl, 1, 1)
    wvl_weights = N_wvl / wvl_weights.sum() * wvl_weights # Normalize so that the total energy is preserved

    # loss_Huber = torch.nn.HuberLoss(reduction='mean', delta=0.05)
    # loss_MAE   = torch.nn.L1Loss(reduction='mean')
    # loss_MSE   = torch.nn.MSELoss(reduction='mean')
    grad_loss_fn = GradientLoss(p=1, reduction='mean')


    def loss_fn(x_, w_MSE, w_MAE):    
        diff = (func(x_)-PSF_0) * wvl_weights # NOTE: wvl_weights removed for now
        w = 2e4
        MSE_loss = diff.pow(2).mean() * w * w_MSE
        MAE_loss = diff.abs().mean()  * w * w_MAE
        LO_loss  = loss_LO_fn() if PSF_model.LO_NCPAs else 0.0
        Moffat_loss = Moffat_loss_fn() if PSF_model.Moffat_absorber else 0.0

        return MSE_loss + MAE_loss + LO_loss + Moffat_loss


    def Moffat_loss_fn():
        amp = PSF_model.inputs_manager['amp']
        # alpha = PSF_model.inputs_manager['alpha']
        # beta = PSF_model.inputs_manager['beta']
        # b = PSF_model.inputs_manager['b']
        
        # Enforce positive amplitude
        amp_penalty = amp.pow(2).mean() * 2.5e-2
        
        # Enforce beta > 1.5
        # beta_penalty = torch.clamp(1.5 - beta, min=0).pow(2).mean() * 1e-3
        
        # # Enforce alpha > 0
        # alpha_penalty = torch.clamp(-alpha, min=0).pow(2).mean() * 1e-3
        
        # # Enforce b > 0
        # b_penalty = torch.clamp(-b, min=0).pow(2).mean() * 1e-3
        
        return amp_penalty #+ b_penalty + beta_penalty + alpha_penalty


    def loss_LO_fn():
        if isinstance(PSF_model.LO_basis, PixelmapBasis):
            LO_loss = grad_loss_fn(PSF_model.OPD_func.unsqueeze(1)) * 5e-5
            
        elif isinstance(PSF_model.LO_basis, ZernikeBasis) or isinstance(PSF_model.LO_basis, ArbitraryBasis):
            LO_loss = PSF_model.inputs_manager['LO_coefs'].pow(2).sum(-1).mean() * 1e-7
            # Constraint to enforce first element of LO_coefs to be positive
            first_coef_penalty = torch.clamp(-PSF_model.inputs_manager['LO_coefs'][:, 0], min=0).pow(2).mean() * 5e-5
            LO_loss += first_coef_penalty
        
        return LO_loss

    # def loss_fn_Huber(x_):
    #     PSF_1 = func(x_)
    #     huber_loss = loss_Huber(PSF_1*wvl_weights*5e5, PSF_0*wvl_weights*5e5)
    #     MSE_loss = loss_MSE(PSF_1*wvl_weights, PSF_0*wvl_weights) * 2e4 * 800.0
    #     LO_loss = loss_LO_fn() if PSF_model.LO_NCPAs else 0.0

    #     return huber_loss + LO_loss + MSE_loss

    loss_fn1 = lambda x_: loss_fn(x_, w_MSE=900.0, w_MAE=1.6)
    # loss_fn2 = lambda x_: loss_fn(x_, w_MSE=1.0,   w_MAE=2.0)

    # Minimization function
    def minimize_params(loss_fn, include_list, exclude_list, max_iter, verbose=True, force_BFGS=False):
        if not include_list:
            raise ValueError('include_list is empty')
        
        if type(include_list) is str: include_list = [include_list]
        if type(exclude_list) is str: exclude_list = [exclude_list]    
        if type(include_list) is set: include_list = list(include_list)
        if type(exclude_list) is set: exclude_list = list(exclude_list)
            
        PSF_model.inputs_manager.set_optimizable(include_list, True)
        PSF_model.inputs_manager.set_optimizable(exclude_list, False)

        x_backup = PSF_model.inputs_manager.stack().clone()
        
        lim_lambda = lambda method, tol: minimize(
            loss_fn, PSF_model.inputs_manager.stack(), 
            max_iter=max_iter, tol=tol, method=method, 
            disp=2 if verbose else 0
        )

        # Try L-BFGS first
        result = lim_lambda('l-bfgs' if not force_BFGS else 'bfgs', 1e-4)
        
        acceptable_loss = 1.0
        
        # Retry with BFGS if convergence issues detected
        if result['nit'] < max_iter * 0.3 and result['fun'] > acceptable_loss: # If converged earlier than 30% of the iterations and final loss is too high
            if verbose:
                reason = "stopped too early" if result['nit'] < max_iter * 0.3 else "final loss is high"
                print(f"Warning: minimization {reason}. Perhaps, convergence wasn't reached? Trying BFGS...")
                
            _ = PSF_model.inputs_manager.unstack(x_backup, include_all=True, update=True)
            result = lim_lambda('bfgs', 1e-5)

        OPD_map = PSF_model.OPD_func(PSF_model.inputs_manager['LO_coefs']).detach().cpu().numpy() if PSF_model.LO_NCPAs else None
            
        if verbose: print('-'*50)

        success = result['fun'] < acceptable_loss
        
        if not success:
            print("Warning: Minimization did not converge.")

        return result.x, func(result.x), OPD_map, success

    # Define what to fit
    fit_wind_speed  = True
    fit_wind_dir    = False
    fit_outerscale  = True
    fit_Cn2_profile = True

    include_general = ['r0', 'dn'] + \
                      (['LO_coefs'] if PSF_model.LO_NCPAs else []) + \
                      ([x+'_ctrl' for x in PSF_model.polychromatic_params] if PSF_model.use_splines else PSF_model.polychromatic_params) + \
                      (['L0'] if fit_outerscale else []) + \
                      (['wind_speed_single'] if fit_wind_speed else []) + \
                      (['wind_dir_single'] if fit_wind_dir else []) + \
                      (['Cn2_weights'] if fit_Cn2_profile else []) + \
                      (['amp', 'alpha', 'b'] if PSF_model.Moffat_absorber else [])

    exclude_general = ['ratio', 'theta', 'beta'] if PSF_model.Moffat_absorber else []

    include_general, exclude_general = set(include_general), set(exclude_general)

    # Perform fitting
    x0, PSF_1, OPD_map, success = minimize_params(loss_fn1, include_general, exclude_general, 200, verbose=False)

    if not success:
        print("Retrying minimization without Cn2 profile fitting...")
        PSF_model.reset_parameters()
        
        include_updated = include_general - set(['Cn2_weights', 'L0', 'wind_speed_single', 'wind_dir_single'])
        x0, PSF_1, OPD_map, success = minimize_params(loss_fn1, include_updated, exclude_general, 300, verbose=False, force_BFGS=True)

    loss_val = loss_fn1(x0).item()

    fitted_values = to_store(PSF_model.inputs_manager.unstack(x0, include_all=True, update=True))

    # Compute diffraction-limited PSF for SR calculation
    PSF_DL = PSF_model.model.DLPSF()

    # Prepare data for saving
    config_manager = ConfigManager()
    config_dict = configs[0] if isinstance(configs, list) else configs
    config_manager.Convert(config_dict, framework='numpy')

    save_data = {
        'comments':    'NFM fitting with splines and LO NCPAs',
        'input_vec':   to_store(x0),
        'config':      config_dict,
        'OPD_map':     OPD_map,
        'FWHM fit':    FWHM_fitter(PSF_0.cpu().numpy(), verbose=False).squeeze(),
        'FWHM data':   FWHM_fitter(PSF_1.cpu().numpy(), verbose=False).squeeze(),
        'Nph WFS':     to_store(PSF_model.model.WFS_Nph),
        'SR data':     SR(PSF_0, PSF_DL).detach().cpu().numpy(),
        'SR fit':      SR(PSF_1, PSF_DL).detach().cpu().numpy(),
        'Img. data':   to_store(PSF_0),
        'Img. fit':    to_store(PSF_1),
        'PSD':         to_store(PSF_model.model.PSDs),
        'Data norms':  to_store(norms),
        'Data bgs':    to_store(bgs),
        'Model norms': to_store(PSF_model.model.norm_scale) if hasattr(PSF_model.model, 'norm_scale') else None,
        'loss':        loss_val,
        'wavelengths': to_store(wavelengths),
        'is_Moffat':   PSF_model.Moffat_absorber,
        'is_LO_NCPAs': PSF_model.LO_NCPAs,
        'is_splines':  PSF_model.use_splines
    }
        
    save_data = save_data | fitted_values

    del PSF_0, PSF_1, PSF_DL, PSF_model

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return save_data


#%%
for id in good_ids:
    try:
        filename = MUSE_FITTING_FOLDER / f'{id}.pickle'
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
