#%%
# %reload_ext autoreload
# %autoreload 2

import sys
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
from tiptorch.tools.utils import r0 as r0_


from tiptorch._config import project_settings, default_device

import warnings

"""
Does the fitting of the PSF model parameters for MUSE_NFM STD stars dataset and stores the fitted values
Arguments are be passed as:
- Device specification: e.g., "cuda:N" or "cpu"
- Start and end IDs for fitting a subset of the dataset: e.g., "50 150"
- Output folder path for saving fitted parameters: e.g., "F:/ESO/Data/MUSE

Example usage: python STD_fitter.py cuda:0 50 150

The fitting regime is set by the flags below (DEROTATE_PSF, APPLY_KC_CORRECTION), not by arguments. `load_and_fit_sample` can also be
imported and called directly (kc_mismatch.py does so to scan the k_c of the flagged samples).
"""

# Default output folder for fitted parameters, can be overridden by passing a path as an argument
MUSE_FITTING_FOLDER = Path(project_settings['MUSE_fitted_STD_folder'])

# PSF loss. 'balanced' is the dynamic-range-balanced loss also used to train the NFM calibrator (see BalancedPSFLoss in
# tiptorch.tools.utils and the "loss" section of machine_learning/calibrators/configs/NFM_calibrator_config.json); keep both in sync so the fitted
# values are consistent with the calibrator's objective. 'legacy' is the linear MSE + MAE loss used for the previous fits.
PSF_LOSS_CFG = {
    "type":             "balanced",
    # "type":             "legacy",
    # BalancedPSFLoss parameters ('balanced' only)
    "radial_power":     2.0,
    "r_max":            40.0,
    "asinh_k":          3.0,
    "huber_delta":      0.1,
    "scale":            3.0,
    "match_background": False,
    # Linear MSE term added to the balanced loss (BalancedPSFLoss.mse_weight, 'balanced' only; the calibrator loss config takes the same key).
    # The balanced loss compares the pixels in log space, so the peak pixel weighs as much as any other core pixel and the fits trade a
    # 4-6 % too faint / too wide core for the halo; the linear MSE is dominated by the brightest pixels and restores the peak. 1.8e7 is the
    # legacy MSE term (w_MSE x 2e4): 30-sample test, 2026-10-19: median |peak err| 5.7 % -> 4.3 % with the halo errors unchanged (legacy
    # alone 3.7 %). Not the full legacy loss: its MAE part is dominated by the wing pixels and pulls the halo to the legacy solution. 0 = off.
    # Changed from 0 to 1.8e7 on 2026-10-19: the fits in fitted_balanced / fitted_legacy predate it
    "mse_weight":       1.8e7,
    # Per-wavelength peak term of the balanced loss (BalancedPSFLoss.peak_weight): the peak of a seeing-dominated blue PSF is a flat direction
    # of the loss otherwise (equal-loss fits differ by 10-20 % there). 30-sample test, 2026-10-20: weight 3 halves the peak error (3.7 -> 2.5 %,
    # bluest bins 4.2 -> 1.7 %, FWHM error 3.3 -> 1.6 %) for +1.4 % of the asinh + MSE loss, halo errors unchanged. 0 = off
    "peak_weight":      3.0,
    # Weights of the MSE and MAE terms ('legacy' only)
    "w_MSE":            900.0,
    "w_MAE":            1.6,
    # Regularization weights of the LO NCPAs (both loss types): phase bump (1st LO coefficient) and the LO coefficients' amplitude / defocus sign
    "w_bump":           5e-5,
    "w_LO":             1e-7,
    # Pull of the LO coefficients and of the wing fraction towards the dataset priors (medians over the sharp stars, fitted_priors.pickle made by
    # STD_fitted_df.py / compute_fit_priors). Scaled by 1e3 for blurry PSFs, where these are not constrained by the data (see the blurry branch
    # below): the pull then dominates and the values stay at the priors. Sharp PSFs feel only the base weights (0 = free)
    "w_LO_prior":       0.0,
    "w_wing_prior":     1e-2,
}

# Fit the derotated PSFs (config PupilAngle 0, the regime of the stored fits) or the raw frames, with the PSF model rotated to the header
# pupil angle of the sample. The fitted dx / dy and the LO coefficients are in the frame of the fitted data (raw frames: dx / dy of the raw
# image, LO coefficients in the rotated pupil frame), so keep the two regimes in separate output folders
DEROTATE_PSF = False

# Samples flagged 'k_c mismatch' in muse_df (recorded with fewer corrected DM modes, see kc_mismatch.py) are fitted with the model k_c of
# their label group: LoadSTDStarCache scales the DM pitch of their configs by 1 / 'k_c fraction'. With False every sample is fitted with
# the nominal k_c = 1 / (2 x 0.22 m) instead
APPLY_KC_CORRECTION = True

# Chromatic blur and TT jitter kernel of the model (PSFModelNFM flags). CHROM_DEFOCUS: the chromatic part of the core blur is a chromatic
# defocus term (spline over λ, added to the defocus coefficient of the LO NCPAs) and the jitter kernel is achromatic; otherwise the jitter
# itself is chromatic (spline). ANISOTROPIC_JITTER: the kernel is (Jx, Jy) along its own axes rotated by J_angle [deg] in the IMAGE frame
# (the kernel does not turn with the pupil angle), instead of one isotropic J. Introduced on 2026-10-20 for the detector-frame quadrupole
# of the median fit residual (the data cores are elongated along the image x axis, which an isotropic kernel cannot follow); the fits before
# that date are isotropic and chromatic-jitter fits (CHROM_DEFOCUS = ANISOTROPIC_JITTER = False).
# 30-sample test (agents/execplans/anisotropic-jitter-execplan.md): the anisotropic kernel lowers the loss by ~10 % (Jy 18 mas along the columns
# vs Jx 13 mas, J_angle ~0) at +0.5 % peak error, but does not remove the 3-8 px detector-frame quadrupole of the residual; the chromatic
# defocus stays ~0 nm and costs 0.6-1.5 % of peak error vs the chromatic jitter. Both False = the isotropic chromatic jitter (J_ctrl) of the
# fits before 2026-10-20. Adopted regime (user, 2026-10-20): anisotropic chromatic jitter (Jx_ctrl, Jy_ctrl, J_angle) without chromatic
# defocus, together with the image-frame wing below (30-sample subset: loss 0.109 vs 0.150, |peak err| 3.9 % vs 4.4 %)
CHROM_DEFOCUS      = False
ANISOTROPIC_JITTER = True

# One-dimensional wing of the PSF along the image columns (PSFModelNFM wing_kernel, TipTorch.WingKernel): the detector-frame halo excess of the
# data (5-25 % brighter along the columns than along the rows from 3 to 35 px, growing to the red), which neither the pupil-frame PSDs nor a
# jitter represent. Fitted: wing_frac (spline over λ) and wing_width [px]; the axis angle stays at 0 unless FIT_WING_ANGLE
WING_KERNEL    = True
FIT_WING_ANGLE = False
# The wing fraction and width are degenerate along a valley (a ~1 px wing holding 10-40 % of the flux fits as well as a 30-50 px wing holding
# a few %, and f > 1 gives an unphysical kernel): the fraction is kept in [0, 1] and the width in WING_WIDTH_BOUNDS by penalties, and the width
# can be fixed at WING_WIDTH_PX (its initial value) with FIT_WING_WIDTH = False. The width is fixed by default (a dataset-wide wing, as it
# will be in the calibrator): only the chromatic fraction is fitted per star, which removes the degeneracy
FIT_WING_WIDTH    = False
WING_WIDTH_PX     = 1.0    # initial (or fixed) half-width of the wing [px]
WING_WIDTH_BOUNDS = (0.5, 20.0)

# Sign convention of the static aberrations when the phase bump is suppressed. A PSF is invariant under the joint sign flip of all the EVEN
# Zernike modes (defocus, astigmatisms; the odd ones, comas and trefoils, keep their sign), so without the bump (which breaks the symmetry
# and makes the sign observable) the fit lands on either branch, and a penalty on defocus alone only pushes defocus to 0 and moves the content
# into the astigmatisms. 'flip': after the fit, if defocus is negative, the even modes are reflected and the fit is re-run from there (the
# mirror branch, same loss), 'penalty': a strong positivity penalty on defocus (bump weight), 'free': the weak w_LO penalty only.
# Phase-bump stars (bump fitted) are never flipped: the data select the branch (the dataset shows both signs, 56 % positive).
# DEFOCUS_INIT_NM: starting defocus of the bump-suppressed stars, on the positive side of the symmetric point 0 (30-sample test, 2026-10-20:
# same loss as a start at 0 or -20 nm, fewest reflections and the best peak error, 3.7 % vs 4.0-4.1 %)
DEFOCUS_SIGN    = 'flip'
DEFOCUS_INIT_NM = 20.0

warnings.filterwarnings("ignore")

#% Initialize data sample
with open(STD_FOLDER / 'muse_df.pickle', 'rb') as handle:
    psf_df = pickle.load(handle)

# Dataset priors for the blurry PSFs (None: LO coefficients pulled to 0 and the wing fraction to its bootstrap curve, with a warning)
FIT_PRIORS = load_fit_priors()
if FIT_PRIORS is None:
    warnings.warn(f'{FIT_PRIORS_FILE} not found: the blurry PSFs are fitted with LO coefficients pulled to 0 and the wing fraction to the built-in curve; '
                  'run STD_fitted_df.py (compute_fit_priors) after a fit of the sharp stars to create it')
WING_FRAC_PRIOR_DEFAULT = np.array([0.036, 0.068, 0.095, 0.129, 0.184]) # 2026-10-20, 25 sharp stars of the 30-sample subset, 1 px width, 5 nodes 475-935 nm

psf_df = psf_df[psf_df['Corrupted']   == False]
psf_df = psf_df[psf_df['Bad quality'] == False]

good_ids = psf_df.index.values.tolist()

# Prior STDs of the MAP term, taken as the spread over the dataset of the telemetry values the model is initialized with (see
# MUSE_data_utils: r0 from 'Seeing (header)' at the 500 nm of muse_ltao.ini, L0 from 'L0Tot', wind speed from 'Wind speed (header)')
PRIOR_STD = {
    'r0':                float(np.nanstd(r0_(psf_df['Seeing (header)'], 500e-9).replace(0, np.nan))) * 3, # let's crack it up a bit
    'L0':                float(np.nanstd(psf_df['L0Tot'])),
    'wind_speed_single': float(np.nanstd(psf_df['Wind speed (header)'])),
}

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
def load_and_fit_sample(id, device=default_device, kc_fraction=None):
    """
    Fits one STD star and returns the dictionary stored in the fit pickle.
    kc_fraction: model k_c relative to the nominal one. None (default): the k_c of the sample's 'k_c mismatch' label when
    APPLY_KC_CORRECTION, the nominal one otherwise; a value forces that k_c (used by kc_mismatch.py to scan the k_c of a sample).
    """
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

    # The configs carry the k_c of the sample's label (scaled DM pitch): rescale them to the nominal / forced k_c when asked
    if kc_fraction is None and not APPLY_KC_CORRECTION:
        kc_fraction = 1.0
    if kc_fraction is not None:
        configs = [apply_kc_fraction(c, kc_fraction / kc_fraction_from_config(c)) for c in configs]

    # Initialize the model
    PSF_model = PSFModelNFM(
        configs,
        multiple_obs    = True,
        LO_NCPAs        = True,
        chrom_defocus   = CHROM_DEFOCUS,
        anisotropic_jitter = ANISOTROPIC_JITTER,
        wing_kernel     = WING_KERNEL,
        use_Moffat      = False,
        retain_PSDs     = True,
        Z_mode_max      = 9,
        N_spline_nodes  = 5,
        device          = device
    )

    if WING_KERNEL:
        PSF_model.inputs_manager['wing_width'][:] = WING_WIDTH_PX
    # The blurry / Non-point flags are needed for the defocus start; they are also read again below for the regularization
    _bump_suppressed_init = bool(psf_df.loc[id]['Bad quality']) or (bool(psf_df.loc[id]['Medium quality']) and not bool(psf_df.loc[id]['Good quality'])) or not bool(psf_df.loc[id]['Non-point'])
    if PSF_model.LO_NCPAs and DEFOCUS_SIGN == 'flip' and _bump_suppressed_init:
        PSF_model.inputs_manager['LO_coefs'][:, 1] = DEFOCUS_INIT_NM # start on the positive-defocus branch (see DEFOCUS_SIGN)
    PSF_model.backup_manager = PSF_model.inputs_manager.copy()

    func = lambda x_: PSF_model( PSF_model.inputs_manager.unstack(x_) )
    PSF_1 = func(x0 := PSF_model.inputs_manager.stack() )

    N_src = PSF_0.shape[0]
    wavelengths = PSF_model.λ_sim

    loss_cfg  = dict(PSF_LOSS_CFG)
    loss_type = loss_cfg.pop('type')
    w_MSE, w_MAE, w_bump, w_LO = (loss_cfg.pop(key) for key in ('w_MSE', 'w_MAE', 'w_bump', 'w_LO'))
    w_LO_prior, w_wing_prior   = loss_cfg.pop('w_LO_prior', 0.0), loss_cfg.pop('w_wing_prior', 0.0)
    
    if loss_type == 'balanced':
        PSF_loss_balanced = BalancedPSFLoss(PSF_0.shape[-2], PSF_0.shape[-1], device, **loss_cfg)
    elif loss_type == 'legacy':
        PSF_loss_balanced = None
    else:
        raise ValueError(f"Unknown PSF loss type '{loss_type}', expected 'balanced' or 'legacy'")

    # Retrieve flags from dataframe
    blurry_PSF_flag = \
         bool(psf_df.loc[id]['Bad quality']) or \
        (bool(psf_df.loc[id]['Medium quality']) and not \
         bool(psf_df.loc[id]['Good quality']))
        
    non_point_flag  = psf_df.loc[id]['Non-point'].item()
    
    # Model k_c relative to the nominal one, recorded during dataset cache generation before fitting
    # During training: samples with k_c mismatch are grouped based on their fraction (group 1..N), 1.0 for the others (group 0), or a forced value (group -1)
    
    kc_fraction = kc_fraction_from_config(configs[0])
    label_fraction = float(psf_df.loc[id][KC_FRACTION_FEATURE]) if KC_FRACTION_FEATURE in psf_df.columns else 1.0
    kc_group = int(psf_df.loc[id][KC_GROUP_FEATURE]) if KC_GROUP_FEATURE in psf_df.columns and np.isclose(kc_fraction, label_fraction) else (0 if np.isclose(kc_fraction, 1.0) else -1)
    
    if not np.isclose(kc_fraction, 1.0):
        print(f"Sample {id}: {f'flagged k_c mismatch (group {kc_group})' if kc_group > 0 else 'forced k_c'}: model k_c = {100 * kc_fraction:.1f} % of the nominal "
              f"({PSF_model.model.kc:.3f} 1/m, DM pitch {np.asarray(configs[0]['DM']['DmPitchs']).ravel()[0]:.4f} m)")

    fit_wind_dir    = False
    fit_wind_speed  = False #True
    fit_outerscale  = False #True
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

    # Dataset priors of the parameters a blurry PSF does not constrain: the LO coefficients (not the bump, which is suppressed) and the wing
    # fraction are pulled towards the medians of the sharp stars; the other poorly constrained ones (L0, wind speed, Cn2 profile) are not
    # fitted at all for blurry PSFs and stay at the reduced-telemetry values of the config (see the include / exclude lists below)
    def _prior(key, default):
        if FIT_PRIORS is not None and key in FIT_PRIORS:
            return torch.tensor(np.asarray(FIT_PRIORS[key], dtype=np.float32), device=device)
        return None if default is None else torch.tensor(np.asarray(default, dtype=np.float32), device=device)
    LO_prior   = _prior('LO_coefs', None) if PSF_model.LO_NCPAs else None
    wing_prior = _prior('wing_frac_ctrl', WING_FRAC_PRIOR_DEFAULT) if PSF_model.wing_kernel else None

    grad_loss_fn = GradientLoss(p=1, reduction='mean')
    
    force_positive = lambda x: torch.clamp(-x, min=0).pow(2).mean()

    # Priors initialization
    # Should be comming from the config which comes from the reduced telemetry
    prior_mean = { param: PSF_model.inputs_manager[param].detach().clone() for param in ['r0', 'L0', 'wind_speed_single'] }
    # Shoud be coming from the reduced telemetry directly
    prior_STD  = PRIOR_STD


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
        if PSF_model.wing_kernel: # keep the wing fraction in [0, 1] and the width within its bounds (see WING_WIDTH_BOUNDS)
            wf = PSF_model.inputs_manager['wing_frac_ctrl' if PSF_model.use_splines else 'wing_frac']
            ww = PSF_model.inputs_manager['wing_width']
            loss += force_positive(wf) + force_positive(1.0 - wf) * 1e2
            loss += (force_positive(ww - WING_WIDTH_BOUNDS[0]) + force_positive(WING_WIDTH_BOUNDS[1] - ww)) * 1e2
            # Pull towards the dataset prior curve, dominant for blurry PSFs (the division by w undoes the overall weight below)
            if wing_prior is not None and wf.shape[-1] == wing_prior.shape[-1]:
                loss += (wf - wing_prior).pow(2).mean() * w_wing_prior * suppress_LO / 0.025
        if fit_outerscale:
            loss += force_positive(PSF_model.inputs_manager['L0'])
        if fit_wind_speed:
            loss += force_positive(PSF_model.inputs_manager['wind_speed_single'])
        
        w = 0.025 # Empirical weight to balance the loss magnitude with the PSF loss
        return loss * w


    def loss_fn(x_):
        PSF_pred = func(x_)
        
        if PSF_loss_balanced is not None:
            PSF_loss = PSF_loss_balanced(PSF_0, PSF_pred).mean()     
        else:
            diff = PSF_pred - PSF_0 #* wvl_weights
            w = 2e4
            PSF_loss = (diff.pow(2).mean() * w_MSE + diff.abs().mean() * w_MAE) * w

        LO_loss     = loss_LO_fn()     if PSF_model.LO_NCPAs   else 0.0
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


    def loss_LO_fn():
        if isinstance(PSF_model.LO_basis, PixelmapBasis):
            LO_loss = grad_loss_fn(PSF_model.OPD_func.unsqueeze(1)) * 5e-5
            
        elif isinstance(PSF_model.LO_basis, ZernikeBasis) or isinstance(PSF_model.LO_basis, ArbitraryBasis):
            LO_coefs = PSF_model.inputs_manager['LO_coefs']
            if suppress_LO_flag and LO_prior is not None and LO_coefs.shape[-1] == LO_prior.shape[-1]:
                # Blurry PSF: the LO coefficients (bump excluded, see below) are pulled towards the dataset medians instead of 0
                LO_loss = (LO_coefs[:, 1:] - LO_prior[1:]).pow(2).sum(-1).mean() * (w_LO * suppress_LO + w_LO_prior * suppress_LO)
            else:
                LO_loss = LO_coefs.pow(2).sum(-1).mean() * w_LO * suppress_LO
                if LO_prior is not None and w_LO_prior > 0 and LO_coefs.shape[-1] == LO_prior.shape[-1]:
                    LO_loss += (LO_coefs[:, 1:] - LO_prior[1:]).pow(2).sum(-1).mean() * w_LO_prior
            # Constraint to enforce first element of LO_coefs to be positive
            phase_bump = PSF_model.inputs_manager['LO_coefs'][:, 0]
            phase_bump_positive   = force_positive(phase_bump) * w_bump * suppress_bump
            # PSFs without the "Non-point" flag: the phase bump is pulled to 0, not only kept positive. Otherwise the fit uses it to
            # absorb other errors, and the fitted bump goes into the calibrator training (re-attached per sample) while the
            # calibrator bundles predict 0 (bump 1 costs ~40 % of a typical fit loss: the fitted bumps of these PSFs end up < ~0.1)
            phase_bump_penalty    = phase_bump.pow(2).mean() * w_bump * suppress_bump if suppress_bump_flag else 0.0
            # Force defocus to be positive to mitigate sign ambiguity. The composite LO basis is [phase bump, defocus, astig, astig, coma, ...]
            # (ZernikeBasis starts at Noll 2 and the wrapper takes its modes from index 2 on), so defocus is index 1; until 2026-10-20 the
            # penalty acted on index 2 (astigmatism) and the defocus sign was free. With the phase bump suppressed the sign of defocus is a
            # pure convention (the PSF is symmetric under the flip), so the penalty gets the bump weight to actually select the positive branch
            # (w_LO alone, 1e-7, lets -70 nm survive at a cost of 5e-4); with a fitted bump the sign is identifiable and the pull stays weak
            first_defocus_penalty = force_positive(PSF_model.inputs_manager['LO_coefs'][:, 1]) * ((w_bump if DEFOCUS_SIGN == 'penalty' else w_LO) if suppress_bump_flag else w_LO) * suppress_LO

            LO_loss += phase_bump_positive + phase_bump_penalty + first_defocus_penalty
        
        return LO_loss


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

    # Achromatic jitter entries (the chromatic ones, J_ctrl / Jx_ctrl / Jy_ctrl / chrom_defocus_ctrl, come with the polychromatic parameters)
    jitter_achromatic = (PSF_model.jitter_params if PSF_model.chrom_defocus else []) + (['J_angle'] if PSF_model.anisotropic_jitter else [])
    # Achromatic wing entries (wing_frac_ctrl comes with the polychromatic parameters)
    wing_achromatic = ((['wing_width'] if FIT_WING_WIDTH else []) + (['wing_angle'] if FIT_WING_ANGLE else [])) if PSF_model.wing_kernel else []

    include_general = ['r0', 'dn'] + jitter_achromatic + wing_achromatic + \
                      (['LO_coefs'] if PSF_model.LO_NCPAs else []) + \
                      ([x+'_ctrl' for x in PSF_model.polychromatic_params if x != 'F_norm_λ'] if PSF_model.use_splines else [x for x in PSF_model.polychromatic_params if x != 'F_norm_λ']) + \
                      (['L0'] if fit_outerscale else []) + \
                      (['wind_speed_single'] if fit_wind_speed else []) + \
                      (['wind_dir_single'] if fit_wind_dir else []) + \
                      (['Cn2_weights'] if fit_Cn2_profile else []) + \
                      (['amp', 'alpha', 'b'] if PSF_model.use_Moffat else [])

    # NOTE: set_optimizable only touches the listed names, so an input created optimizable (wing_width, L0, wind_speed_single) stays fitted unless
    # it is excluded explicitly
    # Blurry PSFs: L0 and the wind speed are not fitted and stay at the reduced-telemetry values of the config (they are created optimizable, so
    # leaving them out of include_general is not enough)
    exclude_general = (['ratio', 'theta', 'beta'] if PSF_model.use_Moffat else []) + \
                      (['wing_width'] if (PSF_model.wing_kernel and not FIT_WING_WIDTH) else []) + \
                      ([] if fit_outerscale else ['L0']) + ([] if fit_wind_speed else ['wind_speed_single'])

    include_general, exclude_general = set(include_general), set(exclude_general)

    # Perform fitting
    x0, PSF_1, OPD_map, success = minimize_params(loss_fn, include_general, exclude_general, 200, verbose=False)

    if not success:
        print("Retrying minimization without Cn2 profile fitting...")
        PSF_model.reset_parameters()
        
        include_updated = include_general - set(['Cn2_weights', 'L0', 'wind_speed_single', 'wind_dir_single'])
        x0, PSF_1, OPD_map, success = minimize_params(loss_fn, include_updated, exclude_general, 300, verbose=False, force_BFGS=True)

    loss_val = loss_fn(x0).item()

    # Sign convention of the even static modes (see DEFOCUS_SIGN): reflect the even group and re-fit when defocus came out negative
    even_flip = False
    if DEFOCUS_SIGN == 'flip' and suppress_bump_flag and PSF_model.LO_NCPAs and float(PSF_model.inputs_manager['LO_coefs'][0, 1]) < 0:
        even_ids = even_zernike_composite_ids(PSF_model.Z_mode_max)
        LO_orig  = PSF_model.inputs_manager['LO_coefs'].detach().clone()
        PSF_model.inputs_manager['LO_coefs'][:, even_ids] = -LO_orig[:, even_ids]
        x_flip, PSF_flip, OPD_flip, _ = minimize_params(loss_fn, include_general, exclude_general, 100, verbose=False)
        loss_flip = loss_fn(x_flip).item()
        if loss_flip <= loss_val * 1.005: # the mirror branch (equal loss up to the asymmetries of the pupil / kernels)
            x0, PSF_1, OPD_map, loss_val, even_flip = x_flip, PSF_flip, OPD_flip, loss_flip, True
        else:
            print(f"Sample {id}: the reflected even modes cost {100 * (loss_flip / loss_val - 1):.1f} % of loss, keeping the negative-defocus branch")
            PSF_model.inputs_manager['LO_coefs'][:] = LO_orig
            _ = PSF_model.inputs_manager.unstack(x0, include_all=True, update=True)

    fitted_values = to_store(PSF_model.inputs_manager.unstack(x0, include_all=True, update=True))

    # Compute diffraction-limited PSF for SR calculation
    PSF_DL = PSF_model.model.DLPSF()

    # Prepare data for saving
    config_manager = ConfigManager()
    config_dict = configs[0] if isinstance(configs, list) else configs
    config_manager.Convert(config_dict, framework='numpy')

    save_data = {
        'comments':    f'NFM fitting with re-normalized dn, {loss_type} PSF loss, phase bump pulled to 0 unless flagged "Non-point", '
                       f'{"chromatic defocus + achromatic" if CHROM_DEFOCUS else "chromatic"} {"anisotropic (Jx, Jy, J_angle)" if ANISOTROPIC_JITTER else "isotropic"} jitter'
                       f'{", image-frame wing (wing_frac, wing_width)" if WING_KERNEL else ""}',
        'loss config': PSF_LOSS_CFG,
        'chrom_defocus':      bool(CHROM_DEFOCUS),
        'anisotropic_jitter': bool(ANISOTROPIC_JITTER),
        'wing_kernel':        bool(WING_KERNEL),
        'wing width fitted':  bool(FIT_WING_WIDTH),
        'blurry':             bool(blurry_PSF_flag), # LO coefficients and wing fraction pulled to the dataset priors, L0 / wind speed / Cn2 at the config values
        'defocus sign':       DEFOCUS_SIGN,
        'even modes flipped': bool(even_flip),
        'priors':             None if FIT_PRIORS is None else {k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v) for k, v in FIT_PRIORS.items()},
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
if __name__ == '__main__':
    # Arguments: device ("cuda:N" / "cpu"), start and end IDs, output folder (any order, all optional)
    device = default_device
    start_id, end_id = -1, -1
    for arg in sys.argv[1:]:
        if arg.startswith("cuda") or arg.startswith("cpu"):
            device = torch.device(arg)
        elif arg.lstrip('-').isdigit():
            if start_id == -1:  start_id = int(arg)
            elif end_id == -1:  end_id   = int(arg)
        else:
            MUSE_FITTING_FOLDER = Path(arg)
    MUSE_FITTING_FOLDER.mkdir(parents=True, exist_ok=True)

    # Fit the samples between the closest good IDs (the whole dataset by default)
    start_id = good_ids[0]  if start_id == -1 else min(good_ids, key=lambda x: abs(x - start_id))
    end_id   = good_ids[-1] if end_id   == -1 else min(good_ids, key=lambda x: abs(x - end_id))
    ids = good_ids[good_ids.index(start_id):good_ids.index(end_id) + 1]

    print(f"Device: {device} | IDs {start_id}..{end_id} ({len(ids)} samples) | output: {MUSE_FITTING_FOLDER} | "
          f"derotated PSFs: {DEROTATE_PSF} | k_c correction of the flagged samples: {APPLY_KC_CORRECTION} | "
          f"chromatic defocus: {CHROM_DEFOCUS} | anisotropic jitter: {ANISOTROPIC_JITTER} | wing kernel: {WING_KERNEL}")

    for id in ids:
        try:
            filename = MUSE_FITTING_FOLDER / f'{id}.pickle'
            print(f'>>>>>>>>>>>> Fitting sample {id}')
            save_data = load_and_fit_sample(id, device)
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
