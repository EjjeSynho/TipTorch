#%%
# %reload_ext autoreload
# %autoreload 2

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import gc
import pickle
import torch
import numpy as np

from torchmin import minimize

from STD_dataset_utils import *
from tiptorch.PSF_models.NFM_wrapper import PSFModelNFM
from tiptorch.managers.config_manager import ConfigManager
from tiptorch.tools.utils import SR, GradientLoss, FWHM_fitter, BalancedPSFLoss
from tiptorch.tools.static_phase import ArbitraryBasis, PixelmapBasis, ZernikeBasis

from tiptorch._config import project_settings, default_device

import warnings

"""
    Does the fitting of the PSF model parameters for MUSE_NFM STD stars dataset and stores the fitted values
    Arguments are be passed as:
    - Device specification: e.g., "cuda:N" or "cpu"
    - Start and end IDs for fitting a subset of the dataset: e.g., "50 150"
    - Output folder path for saving fitted parameters: e.g., "F:/ESO/Data/MUSE

    Example usage: python STD_fitter.py cuda:0 50 150
"""

# Default output folder for fitted parameters, can be overridden by passing a path as an argument
MUSE_FITTING_FOLDER = Path(project_settings['MUSE_fitted_STD_folder'])

# PSF loss. 'balanced' is the dynamic-range-balanced loss also used to train the NFM calibrator (see BalancedPSFLoss in
# tiptorch.tools.utils and the "loss" section of machine_learning/calibrators/configs/NFM_calibrator_config.json); keep both in sync so the fitted
# values are consistent with the calibrator's objective. 'legacy' is the linear MSE + MAE loss used for the previous fits.
PSF_LOSS_CFG = {
    "type":             "balanced",
    "radial_power":     2.0,
    "r_max":            40.0,
    "asinh_k":          3.0,
    "huber_delta":      0.1,
    "scale":            3.0,
    "match_background": False,
}

# Fit the derotated PSFs (config PupilAngle 0, the regime of the stored fits) or the raw frames, with the PSF model rotated to the header
# pupil angle of the sample. The fitted dx / dy and the LO coefficients are in the frame of the fitted data (raw frames: dx / dy of the raw
# image, LO coefficients in the rotated pupil frame), so keep the two regimes in separate output folders
DEROTATE_PSF = True

warnings.filterwarnings("ignore")

device = default_device

start_id, end_id = -1, -1
kc_fraction_override = None # 'kc_fraction=<v>' argument: fit with the model k_c at v x the nominal one regardless of the 'k_c mismatch' label (experiments)

# Flexible argument parsing
input_args = sys.argv[1:]
device_specified = False

for arg in input_args:
    if arg.startswith("cuda") or arg.startswith("cpu"):
        device = torch.device(arg)
        print(f"Using device: {arg}")
        device_specified = True
        
    elif arg.lstrip('-').isdigit(): # Simple check for integer IDs
        if start_id == -1:  start_id = int(arg)
        elif end_id == -1:  end_id   = int(arg)
    elif arg.startswith('kc_fraction='):
        kc_fraction_override = float(arg.split('=', 1)[1])
        print(f"Model k_c forced to {100 * kc_fraction_override:.1f} % of the nominal one")
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

#% Initialize data sample
with open(STD_FOLDER / 'muse_df.pickle', 'rb') as handle:
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


wvl_ids = np.clip(np.arange(0, (N_wvl_max:=31)+1, 2), a_min=0, a_max=N_wvl_max-1)

#%%
def load_and_fit_sample(id):
    # Load data
    PSF_0, norms, bgs, configs = LoadSTDStarCache(
        ids                 = [id],
        derotate_PSF        = DEROTATE_PSF,
        normalize           = True,
        subtract_background = True,
        wvl_ids             = wvl_ids,
        ensure_odd_pixels   = True,
        device              = device
    )

    if kc_fraction_override is not None: # the configs already carry the k_c fraction of the sample's label: rescale to the forced one
        configs = [apply_kc_fraction(c, kc_fraction_override / kc_fraction_from_config(c)) for c in configs]

    # Initialize the model
    PSF_model = PSFModelNFM(
        configs,
        multiple_obs    = True,
        LO_NCPAs        = True,
        chrom_defocus   = False,
        use_Moffat      = False,
        retain_PSDs     = True,
        Z_mode_max      = 9,
        N_spline_nodes  = 5,
        device          = device
    )

    func = lambda x_: PSF_model( PSF_model.inputs_manager.unstack(x_) )
    PSF_1 = func(x0 := PSF_model.inputs_manager.stack() )

    N_wvl = PSF_0.shape[1]
    N_src = PSF_0.shape[0]
    wavelengths = PSF_model.λ_sim

    loss_cfg  = dict(PSF_LOSS_CFG)
    loss_type = loss_cfg.pop('type')
    if loss_type == 'balanced':
        PSF_loss_balanced = BalancedPSFLoss(PSF_0.shape[-2], PSF_0.shape[-1], device, **loss_cfg)
    elif loss_type == 'legacy':
        PSF_loss_balanced = None
    else:
        raise ValueError(f"Unknown PSF loss type '{loss_type}', expected 'balanced' or 'legacy'")

    # Retrieve flags from dataframe
    blurry_PSF_flag = psf_df.loc[id]['Bad quality'].item()
    non_point_flag  = psf_df.loc[id]['Non-point'].item()
    # Samples recorded with fewer corrected DM modes: LoadSTDStarCache already scaled the DM pitch of the config, so the model's k_c is
    # kc_fraction x the nominal one (see add_kc_mismatch_flag). Recorded in the output for the dataset cache / training
    kc_fraction = float(psf_df.loc[id][KC_FRACTION_FEATURE]) if KC_FRACTION_FEATURE in psf_df.columns else 1.0
    kc_group    = int(psf_df.loc[id][KC_GROUP_FEATURE])    if KC_GROUP_FEATURE    in psf_df.columns else 0
    if kc_fraction_override is not None:
        kc_fraction, kc_group = kc_fraction_override, -1 # -1: forced k_c, not a label group
    if kc_group != 0:
        print(f"Sample {id}: {'forced k_c' if kc_group == -1 else f'flagged k_c mismatch (group {kc_group})'}: model k_c = {100 * kc_fraction:.1f} % of the nominal, "
              f"DM pitch {np.asarray(configs[0]['DM']['DmPitchs']).ravel()[0]:.4f} m, model k_c = {PSF_model.model.kc:.3f} 1/m")

    fit_wind_dir    = False
    fit_wind_speed  = True #False
    fit_outerscale  = True #False
    fit_Cn2_profile = True

    if N_src > 1:
        suppress_bump_flag = False
        suppress_LO_flag   = False
        fit_wind_speed     = True
        fit_outerscale     = True
    else:
        suppress_bump_flag = blurry_PSF_flag | (not non_point_flag)
        suppress_LO_flag   = blurry_PSF_flag
        fit_wind_speed     = (not blurry_PSF_flag)
        fit_outerscale     = (not blurry_PSF_flag)
        fit_Cn2_profile    = (not blurry_PSF_flag)

    suppress_bump = 1e3 if suppress_bump_flag else 1
    suppress_LO   = 1e3 if suppress_LO_flag   else 1

    grad_loss_fn = GradientLoss(p=1, reduction='mean')
    
    force_positive = lambda x: torch.clamp(-x, min=0).pow(2).mean()

    # Priors initialization
    prior_mean = { param: PSF_model.inputs_manager[param].detach().clone() for param in ['r0', 'L0', 'wind_speed_single'] }

    # Pre-defined STDs
    prior_STD = {
        'r0': 0.2,
        'L0': 25,
        'wind_speed_single': 5.0
    }


    def compute_MAP_err(param: str):
        diff = PSF_model.inputs_manager[param] - prior_mean[param]
        loss_values = diff / prior_STD[param]
        return loss_values.pow(2).mean()


    def loss_MAP():
        # Enforce priors on parameters
        loss  = compute_MAP_err('r0')
        loss += compute_MAP_err('L0') if fit_outerscale else 0.0
        loss += compute_MAP_err('wind_speed_single') if fit_wind_speed else 0.0
        
        # Also, force L0, r0, and wind speed to be positive to avoid unphysical values
        loss += force_positive(PSF_model.inputs_manager['r0'])
        if fit_outerscale:
            loss += force_positive(PSF_model.inputs_manager['L0'])
        if fit_wind_speed:
            loss += force_positive(PSF_model.inputs_manager['wind_speed_single'])
        
        w = 0.025 # Empirical weight to balance the loss magnitude with the PSF loss
        return loss * w


    def loss_fn(x_, w_MSE, w_MAE, w_bump, w_LO):
        PSF_pred = func(x_)
        if PSF_loss_balanced is not None:
            PSF_loss = PSF_loss_balanced(PSF_0, PSF_pred).mean()
        else:
            diff = PSF_pred - PSF_0 #* wvl_weights
            w = 2e4
            PSF_loss = (diff.pow(2).mean() * w_MSE + diff.abs().mean() * w_MAE) * w

        LO_loss  = loss_LO_fn(w_bump, w_LO) if PSF_model.LO_NCPAs else 0.0
        Moffat_loss = Moffat_loss_fn() if PSF_model.use_Moffat else 0.0
        MAP_loss    = loss_MAP()

        return PSF_loss + LO_loss + Moffat_loss + MAP_loss


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


    def loss_LO_fn(w_bump, w_LO):
        if isinstance(PSF_model.LO_basis, PixelmapBasis):
            LO_loss = grad_loss_fn(PSF_model.OPD_func.unsqueeze(1)) * 5e-5
            
        elif isinstance(PSF_model.LO_basis, ZernikeBasis) or isinstance(PSF_model.LO_basis, ArbitraryBasis):
            LO_loss = PSF_model.inputs_manager['LO_coefs'].pow(2).sum(-1).mean() * w_LO * suppress_LO
            # Constraint to enforce first element of LO_coefs to be positive
            phase_bump = PSF_model.inputs_manager['LO_coefs'][:, 0]
            phase_bump_positive   = force_positive(phase_bump) * w_bump * suppress_bump
            # PSFs without the "Non-point" flag: the phase bump is pulled to 0, not only kept positive. Otherwise the fit uses it to
            # absorb other errors, and the fitted bump goes into the calibrator training (re-attached per sample) while the
            # calibrator bundles predict 0 (bump 1 costs ~40 % of a typical fit loss: the fitted bumps of these PSFs end up < ~0.1)
            phase_bump_penalty    = phase_bump.pow(2).mean() * w_bump * suppress_bump if suppress_bump_flag else 0.0
            # Force defocus to be positive to mitigate sign ambiguity
            first_defocus_penalty = force_positive(PSF_model.inputs_manager['LO_coefs'][:, 2]) * w_LO * suppress_LO

            LO_loss += phase_bump_positive + phase_bump_penalty + first_defocus_penalty
        
        return LO_loss


    # w_MSE / w_MAE only apply to the legacy PSF loss
    loss_fn1 = lambda x_: loss_fn(x_, w_MSE=900.0, w_MAE=1.6, w_bump=5e-5, w_LO=1e-7)

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
            loss_fn,
            PSF_model.inputs_manager.stack(), 
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

    include_general = ['r0', 'dn'] + \
                      (['LO_coefs'] if PSF_model.LO_NCPAs else []) + \
                      ([x+'_ctrl' for x in PSF_model.polychromatic_params if x != 'F_norm_λ'] if PSF_model.use_splines else [x for x in PSF_model.polychromatic_params if x != 'F_norm_λ']) + \
                      (['L0'] if fit_outerscale else []) + \
                      (['wind_speed_single'] if fit_wind_speed else []) + \
                      (['wind_dir_single'] if fit_wind_dir else []) + \
                      (['Cn2_weights'] if fit_Cn2_profile else []) + \
                      (['amp', 'alpha', 'b'] if PSF_model.use_Moffat else [])

    exclude_general = ['ratio', 'theta', 'beta'] if PSF_model.use_Moffat else []

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
        'comments':    f'NFM fitting with re-normalized dn, {loss_type} PSF loss, phase bump pulled to 0 unless flagged "Non-point"',
        'loss config': PSF_LOSS_CFG,
        'bump suppressed': bool(suppress_bump_flag),
        'k_c fraction': kc_fraction, # model k_c relative to the nominal one (1.0 unless the sample is flagged 'k_c mismatch')
        'k_c group':    kc_group,
        'input_vec':   to_store(x0),
        'config':      config_dict,
        'OPD_map':     OPD_map,
        'FWHM data':   FWHM_fitter(PSF_0.cpu().numpy(), verbose=False).squeeze(),
        'FWHM fit':    FWHM_fitter(PSF_1.cpu().numpy(), verbose=False).squeeze(),
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
        'is_Moffat':   PSF_model.use_Moffat,
        'is_LO_NCPAs': PSF_model.LO_NCPAs,
        'use_splines': PSF_model.use_splines,
        'N spline nodes': PSF_model.N_wvl_ctrl if hasattr(PSF_model, 'N_wvl_ctrl') else None
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
