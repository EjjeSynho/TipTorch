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

# Elongated wing of the PSF along the image columns (PSFModelNFM wing_kernel, TipTorch.WingKernel): the detector-frame halo excess of the
# data (5-25 % brighter along the columns than along the rows from 3 to 35 px, growing to the red), which neither the pupil-frame PSDs nor a
# jitter represent. Fitted: wing_frac (spline over λ); the widths are dataset-wide (below) and the axis angle stays at 0 unless FIT_WING_ANGLE
WING_KERNEL    = True
FIT_WING_ANGLE = False
# Transverse half-width of the wing [px] (TipTorch wing_width_t): 0 makes the wing a one-dimensional line, which is too thin for the data (the
# excess of the data over the no-wing model is a band of ~3-4 px HWHM across the axis in the red, the line is as thin as the PSF core and shows
# as a bright streak in the reddest PSFs); with WING_WIDTH_T_PX > 0 the wing is an elliptical Moffat (beta = 1.5) halo with half-widths
# WING_WIDTH_PX along and WING_WIDTH_T_PX across the axis. Fixed by default, like the width along the axis
FIT_WING_WIDTH_T  = False
WING_WIDTH_T_PX   = 1.5
# Upper bound of the wing fraction (penalty above it); 1.0 = only the physical bound f <= 1
WING_FRAC_MAX     = 1.0
# Exponent p of the wing MTF exp(-(2 pi w nu)^p) (TipTorch wing_power): 1 = Lorentzian / Moffat tails (r^-2 line, r^-3 halo), 2 = Gaussian
WING_POWER        = 1.0
# The wing fraction and widths are degenerate along a valley (a ~1 px wing holding 10-40 % of the flux fits as well as a 30-50 px wing holding
# a few %, and f > 1 gives an unphysical kernel): the fraction is kept in [0, WING_FRAC_MAX] and the widths in WING_WIDTH_BOUNDS by penalties,
# and the widths are fixed at WING_WIDTH_PX / WING_WIDTH_T_PX (their initial values) with FIT_WING_WIDTH / FIT_WING_WIDTH_T = False. They are
# fixed by default (a dataset-wide wing, as it will be in the calibrator): only the chromatic fraction is fitted per star, which removes the
# degeneracy. 5 x 1.5 px (30-star subset, 2026-10-20): the same loss and peak error as the 1 px line (loss 0.122 vs 0.121, |peak err| 2.5 %),
# the quadrupole residual still suppressed (0.05 % of the peak at 3-8 px vs 0.2 % without a wing), the model streak no brighter than the data's
# band in any star (the line overshot it by up to 28 % of the local halo in the sharp red PSFs), half the wing flux (10 % vs 18 % at 935 nm) and
# a smaller halo deficit beyond 40 px (22 % vs 25 %). Widths fitted freely go back to a ~1.5 x 0.5 px line (loss -4 %) that keeps the streak
FIT_WING_WIDTH    = False
WING_WIDTH_PX     = 5.0    # initial (or fixed) half-width of the wing along its axis [px]
WING_WIDTH_BOUNDS = (0.5, 20.0)

# Sign convention of the static aberrations when the phase bump is suppressed. A PSF is invariant under the joint sign flip of all the EVEN
# Zernike modes (defocus, astigmatisms; the odd ones, comas and trefoils, keep their sign), so without the bump (which breaks the symmetry
# and makes the sign observable) the fit lands on either branch, and a penalty on defocus alone only pushes defocus to 0 and moves the content
# into the astigmatisms. 'flip': after the fit, if defocus is negative, the even modes are reflected and the fit is re-run from there (the
# mirror branch, same loss up to FLIP_TOL), 'penalty': a strong positivity penalty on defocus (bump weight), 'free': the weak w_LO penalty only.
# Phase-bump stars (bump fitted) are never flipped: the data select the branch (the dataset shows both signs, 56 % positive).
DEFOCUS_SIGN = 'flip'
FLIP_TOL     = 0.005
# Multi-start fit: the loss landscape of the static modes has local minima (the same star lands 30-50 nm apart in defocus from different
# starts, up to 13 % apart in loss), so the fit is run once per starting defocus value [nm] in DEFOCUS_STARTS and one solution is kept:
# SELECT_BY = 'peak' keeps the start with the least median |peak error| over the wavelengths (the quantity the loss resolves worst: runs of
# equal loss differ by 10-20 % at the blue peak), 'loss' the lowest loss. (20.0,) = the single start of the fits before 2026-10-20
DEFOCUS_STARTS = (20.0, -20.0)
SELECT_BY      = 'peak'

warnings.filterwarnings("ignore")

#% Initialize data sample
with open(STD_FOLDER / 'muse_df.pickle', 'rb') as handle:
    psf_df = pickle.load(handle)
muse_df_all = psf_df # unfiltered (sample_fit_flags also serves the blurry / corrupted samples fitted from the dev script)

# Dataset priors for the blurry PSFs (None: LO coefficients pulled to 0 and the wing fraction to its bootstrap curve, with a warning)
FIT_PRIORS = load_fit_priors()
if FIT_PRIORS is None:
    warnings.warn(f'{FIT_PRIORS_FILE} not found: the blurry PSFs are fitted with LO coefficients pulled to 0 and the wing fraction to the built-in curve; '
                  'run STD_fitted_df.py (compute_fit_priors) after a fit of the sharp stars to create it')
WING_FRAC_PRIOR_DEFAULT = np.array([0.046, 0.055, 0.066, 0.065, 0.098]) # 2026-10-20, 25 sharp stars of the 30-sample subset, 5 x 1.5 px wing, 5 nodes 475-935 nm
if FIT_PRIORS is not None and WING_KERNEL and 'wing_width' in FIT_PRIORS and \
   not (np.isclose(float(np.ravel(FIT_PRIORS['wing_width'])[0]), WING_WIDTH_PX) and np.isclose(float(np.ravel(FIT_PRIORS['wing_width_t'])[0]), WING_WIDTH_T_PX)):
    warnings.warn(f"{FIT_PRIORS_FILE} was made from fits with a {float(np.ravel(FIT_PRIORS['wing_width'])[0]):g} x {float(np.ravel(FIT_PRIORS['wing_width_t'])[0]):g} px wing, "
                  f"the fitter uses {WING_WIDTH_PX:g} x {WING_WIDTH_T_PX:g} px: the wing-fraction prior of the blurry PSFs does not match this wing shape")

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
def sample_fit_flags(ids):
    """
    Fitting regime of a sample (or of several OBs fitted in one model) from the quality flags of muse_df. Blurry PSFs ('Bad quality', or
    'Medium quality' without 'Good quality'): the phase bump is suppressed, the LO coefficients and the wing fraction are pulled to the
    dataset priors (FIT_PRIORS) and L0, the wind speed and the Cn2 profile stay at the config values. PSFs without the 'Non-point' flag:
    the phase bump is suppressed. Several OBs: nothing is suppressed.
    """
    ids   = [ids] if np.isscalar(ids) else list(ids)
    rows  = muse_df_all.loc[ids]
    multi = len(ids) > 1
    blurry    = bool((rows['Bad quality'].astype(bool) | (rows['Medium quality'].astype(bool) & ~rows['Good quality'].astype(bool))).any())
    non_point = bool(rows['Non-point'].astype(bool).all())
    return {
        'blurry':          blurry,
        'suppress_bump':   (blurry or not non_point) and not multi,
        'suppress_LO':     blurry and not multi,
        'fit_outerscale':  (not blurry) or multi,
        'fit_wind_speed':  (not blurry) or multi,
        'fit_wind_dir':    False,
        'fit_Cn2_profile': (not blurry) or multi,
    }


def build_PSF_model(configs, device=default_device):
    """ PSFModelNFM in the regime of the module flags (jitter, wing and its fixed widths); the backup state is the initial one. """
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
        PSF_model.inputs_manager['wing_width'][:]   = WING_WIDTH_PX
        PSF_model.inputs_manager['wing_width_t'][:] = WING_WIDTH_T_PX
        PSF_model.model.wing_power = WING_POWER
        
    PSF_model.backup_manager = PSF_model.inputs_manager.copy()
    return PSF_model


def build_loss(PSF_model, PSF_0, flags, device=default_device):
    """
    The fit objective for the data PSF_0: `func(x)` evaluates the model from the stacked parameter vector and `loss_fn(x)` is the PSF loss
    (PSF_LOSS_CFG) + the LO NCPAs regularization (bump, defocus sign, dataset priors) + the MAP terms (r0, L0, wind speed) and the wing
    bounds. `flags` is the dictionary of sample_fit_flags. Returns (func, loss_fn).
    """
    func = lambda x_: PSF_model( PSF_model.inputs_manager.unstack(x_) )

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

    suppress_bump_flag, suppress_LO_flag = flags['suppress_bump'], flags['suppress_LO']
    suppress_bump = 1e3 if suppress_bump_flag else 1
    suppress_LO   = 1e3 if suppress_LO_flag   else 1

    # Dataset priors of the parameters a blurry PSF does not constrain: the LO coefficients (not the bump, which is suppressed) and the wing
    # fraction are pulled towards the medians of the sharp stars; the other poorly constrained ones (L0, wind speed, Cn2 profile) are not
    # fitted at all for blurry PSFs and stay at the reduced-telemetry values of the config (see parameter_sets)
    def _prior(key, default):
        if FIT_PRIORS is not None and key in FIT_PRIORS:
            return torch.tensor(np.asarray(FIT_PRIORS[key], dtype=np.float32), device=device)
        
        return None if default is None else torch.tensor(np.asarray(default, dtype=np.float32), device=device)
    
    LO_prior   = _prior('LO_coefs', None) if PSF_model.LO_NCPAs else None
    wing_prior = _prior('wing_frac_ctrl', WING_FRAC_PRIOR_DEFAULT) if PSF_model.wing_kernel else None

    grad_loss_fn   = GradientLoss(p=1, reduction='mean')
    force_positive = lambda x: torch.clamp(-x, min=0).pow(2).mean()

    # MAP priors: the reduced-telemetry values the model is initialized with, with the dataset spread as width (PRIOR_STD)
    prior_mean = { param: PSF_model.inputs_manager[param].detach().clone() for param in ['r0', 'L0', 'wind_speed_single', 'wind_dir_single'] }
    prior_STD  = PRIOR_STD | {'wind_dir_single': 10.0}


    def compute_MAP_err(param: str):
        return ((PSF_model.inputs_manager[param] - prior_mean[param]) / prior_STD[param]).pow(2).mean()


    def loss_MAP():
        loss  = compute_MAP_err('r0')
        loss += compute_MAP_err('L0') if flags['fit_outerscale'] else 0.0
        loss += compute_MAP_err('wind_speed_single') if flags['fit_wind_speed'] else 0.0
        loss += compute_MAP_err('wind_dir_single') if flags['fit_wind_dir'] else 0.0

        # Also, force L0, r0, and wind speed to be positive to avoid unphysical values
        loss += force_positive(PSF_model.inputs_manager['r0'])
        
        if PSF_model.wing_kernel: # keep the wing fraction in [0, WING_FRAC_MAX] and the widths within their bounds (see WING_WIDTH_BOUNDS)
            wing_f = PSF_model.inputs_manager['wing_frac_ctrl' if PSF_model.use_splines else 'wing_frac']
            wing_w = PSF_model.inputs_manager['wing_width']
            wing_t = PSF_model.inputs_manager['wing_width_t']
            
            loss += force_positive(wing_f) + force_positive(min(1.0, WING_FRAC_MAX) - wing_f) * 1e2
            loss += (force_positive(wing_w - WING_WIDTH_BOUNDS[0]) + force_positive(WING_WIDTH_BOUNDS[1] - wing_w)) * 1e2
            
            if FIT_WING_WIDTH_T:
                loss += (force_positive(wing_t) + force_positive(WING_WIDTH_BOUNDS[1] - wing_t)) * 1e2
            # Pull towards the dataset prior curve, dominant for blurry PSFs (the division by w undoes the overall weight below)
            if wing_prior is not None and wing_f.shape[-1] == wing_prior.shape[-1]:
                loss += (wing_f - wing_prior).pow(2).mean() * w_wing_prior * suppress_LO / 0.025
        
        if flags['fit_outerscale']:
            loss += force_positive(PSF_model.inputs_manager['L0'])
            
        if flags['fit_wind_speed']:
            loss += force_positive(PSF_model.inputs_manager['wind_speed_single'])

        w = 0.025 # Empirical weight to balance the loss magnitude with the PSF loss
        return loss * w


    def Moffat_loss_fn():
        amp = PSF_model.inputs_manager['amp']
        return amp.pow(2).mean() * 2.5e-2 # Enforce positive amplitude


    def loss_LO_fn():
        if isinstance(PSF_model.LO_basis, PixelmapBasis):
            return grad_loss_fn(PSF_model.OPD_func.unsqueeze(1)) * 5e-5

        LO_coefs = PSF_model.inputs_manager['LO_coefs']
        if suppress_LO_flag and LO_prior is not None and LO_coefs.shape[-1] == LO_prior.shape[-1]:
            # Blurry PSF: the LO coefficients (bump excluded, see below) are pulled towards the dataset medians instead of 0
            LO_loss = (LO_coefs[:, 1:] - LO_prior[1:]).pow(2).sum(-1).mean() * (w_LO * suppress_LO + w_LO_prior * suppress_LO)
        else:
            LO_loss = LO_coefs.pow(2).sum(-1).mean() * w_LO * suppress_LO
            if LO_prior is not None and w_LO_prior > 0 and LO_coefs.shape[-1] == LO_prior.shape[-1]:
                LO_loss += (LO_coefs[:, 1:] - LO_prior[1:]).pow(2).sum(-1).mean() * w_LO_prior
        
        # Constraint to enforce first element of LO_coefs to be positive
        phase_bump = LO_coefs[:, 0]
        phase_bump_positive = force_positive(phase_bump) * w_bump * suppress_bump
        # PSFs without the "Non-point" flag: the phase bump is pulled to 0, not only kept positive. Otherwise the fit uses it to
        # absorb other errors, and the fitted bump goes into the calibrator training (re-attached per sample) while the
        # calibrator bundles predict 0 (bump 1 costs ~40 % of a typical fit loss: the fitted bumps of these PSFs end up < ~0.1)
        phase_bump_penalty  = phase_bump.pow(2).mean() * w_bump * suppress_bump if suppress_bump_flag else 0.0
        # Defocus sign: the composite LO basis is [phase bump, defocus, astig, astig, coma, ...], so defocus is index 1. With the phase bump
        # suppressed the sign is a convention (the PSF is symmetric under the flip of the even modes, see DEFOCUS_SIGN): the 'penalty'
        # convention selects the positive branch with the bump weight (w_LO alone, 1e-7, lets -70 nm survive at a cost of 5e-4); with a
        # fitted bump the sign is identifiable and the pull stays weak
        first_defocus_penalty = force_positive(LO_coefs[:, 1]) * ((w_bump if DEFOCUS_SIGN == 'penalty' else w_LO) if suppress_bump_flag else w_LO) * suppress_LO

        return LO_loss + phase_bump_positive + phase_bump_penalty + first_defocus_penalty


    def loss_fn(x_):
        PSF_pred = func(x_)
        if PSF_loss_balanced is not None:
            PSF_loss = PSF_loss_balanced(PSF_0, PSF_pred).mean()
        else:
            diff = PSF_pred - PSF_0
            PSF_loss = (diff.pow(2).mean() * w_MSE + diff.abs().mean() * w_MAE) * 2e4

        LO_loss     = loss_LO_fn()     if PSF_model.LO_NCPAs   else 0.0
        Moffat_loss = Moffat_loss_fn() if PSF_model.use_Moffat else 0.0
        
        return PSF_loss + LO_loss + Moffat_loss + loss_MAP()

    return func, loss_fn


def make_minimizer(PSF_model, func, loss_fn):
    """
    `minimize_params(include, exclude, max_iter, verbose=False, force_BFGS=False)`: L-BFGS on the parameters of `include` (the ones of `exclude`
    frozen; set_optimizable only touches the listed names), with a BFGS retry when it stops early at a high loss.
    Returns (x, PSF, OPD_map, success) and leaves the model at x.
    """
    def minimize_params(include_list, exclude_list, max_iter, verbose=False, force_BFGS=False):
        if not include_list:
            raise ValueError('include_list is empty')
        include_list = [include_list] if isinstance(include_list, str) else list(include_list)
        exclude_list = [exclude_list] if isinstance(exclude_list, str) else list(exclude_list)

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
                print("Warning: minimization stopped too early at a high loss. Perhaps, convergence wasn't reached? Trying BFGS...")
            _ = PSF_model.inputs_manager.unstack(x_backup, include_all=True, update=True)
            result = lim_lambda('bfgs', 1e-5)

        OPD_map = PSF_model.OPD_func(PSF_model.inputs_manager['LO_coefs']).detach().cpu().numpy() if PSF_model.LO_NCPAs else None

        if verbose: print('-'*50)

        success = result['fun'] < acceptable_loss
        if not success:
            print("Warning: Minimization did not converge.")

        return result.x, func(result.x), OPD_map, success

    return minimize_params


def parameter_sets(PSF_model, flags):
    """ The sets of parameters to fit (include) and to freeze (exclude) for the regime `flags` of sample_fit_flags. """
    # Achromatic jitter entries (the chromatic ones, J_ctrl / Jx_ctrl / Jy_ctrl / chrom_defocus_ctrl, come with the polychromatic parameters)
    jitter_achromatic = (PSF_model.jitter_params if PSF_model.chrom_defocus else []) + (['J_angle'] if PSF_model.anisotropic_jitter else [])
    # Achromatic wing entries (wing_frac_ctrl comes with the polychromatic parameters)
    wing_achromatic = ((['wing_width'] if FIT_WING_WIDTH else []) + (['wing_width_t'] if FIT_WING_WIDTH_T else []) + (['wing_angle'] if FIT_WING_ANGLE else [])) \
                      if PSF_model.wing_kernel else []

    include = ['r0', 'dn'] + jitter_achromatic + wing_achromatic + \
              (['LO_coefs'] if PSF_model.LO_NCPAs else []) + \
              ([x+'_ctrl' for x in PSF_model.polychromatic_params if x != 'F_norm_λ'] if PSF_model.use_splines else [x for x in PSF_model.polychromatic_params if x != 'F_norm_λ']) + \
              (['L0'] if flags['fit_outerscale'] else []) + \
              (['wind_speed_single'] if flags['fit_wind_speed'] else []) + \
              (['wind_dir_single'] if flags['fit_wind_dir'] else []) + \
              (['Cn2_weights'] if flags['fit_Cn2_profile'] else []) + \
              (['amp', 'alpha', 'b'] if PSF_model.use_Moffat else [])

    # NOTE: set_optimizable only touches the listed names, so an input created optimizable (wing_width, L0, wind_speed_single) stays fitted
    # unless it is excluded explicitly. Blurry PSFs: L0 and the wind speed stay at the reduced-telemetry values of the config
    exclude = (['ratio', 'theta', 'beta'] if PSF_model.use_Moffat else []) + \
              (['wing_width'] if (PSF_model.wing_kernel and not FIT_WING_WIDTH) else []) + \
              ([] if flags['fit_outerscale'] else ['L0']) + ([] if flags['fit_wind_speed'] else ['wind_speed_single'])

    return set(include), set(exclude)


def peak_error(PSF_data, PSF_pred, half=3):
    """ Median over the wavelengths of |data peak - model peak| / data peak, peaks taken in the central (2 half + 1)^2 px of the first source. """
    c = PSF_data.shape[-1] // 2
    sl = np.s_[0, :, c-half:c+half+1, c-half:c+half+1]
    d, m = PSF_data[sl].amax(dim=(-2, -1)), PSF_pred[sl].amax(dim=(-2, -1))
    return float(((d - m).abs() / d).median())


def multistart_fit(PSF_model, PSF_0, loss_fn, minimize_params, include, exclude, flags, starts=None, select_by=None, verbose=False):
    """
    The fit from each starting defocus value of `starts` (DEFOCUS_STARTS), with the even-mode sign convention (DEFOCUS_SIGN) applied per
    start, keeping the start selected by `select_by` (SELECT_BY: 'peak' or 'loss'). Returns the dictionary of the kept solution:
    x, PSF, OPD_map, loss, peak_error, flipped, success, start, and 'starts': {start: (loss, peak error, defocus [nm], flipped, success)}.
    The model is left in the kept state.
    """
    starts    = DEFOCUS_STARTS if starts is None else starts
    select_by = SELECT_BY if select_by is None else select_by
    flip      = DEFOCUS_SIGN == 'flip' and flags['suppress_bump'] and PSF_model.LO_NCPAs

    def fit_from(start_defocus):
        def _restart():
            PSF_model.reset_parameters()
            if PSF_model.LO_NCPAs:
                PSF_model.inputs_manager['LO_coefs'][:, 1] = start_defocus

        _restart()
        x, PSF, OPD, ok = minimize_params(include, exclude, 200, verbose=verbose)
        if not ok:
            if verbose:
                print("Retrying minimization without Cn2 profile fitting...")
            _restart()
            x, PSF, OPD, ok = minimize_params(include - {'Cn2_weights', 'L0', 'wind_speed_single', 'wind_dir_single'}, exclude, 300, verbose=verbose, force_BFGS=True)

        loss_val, flipped = loss_fn(x).item(), False
       
        # Sign convention of the even static modes: reflect the even group and re-fit when defocus came out negative
        if flip and float(PSF_model.inputs_manager['LO_coefs'][0, 1]) < 0:
            even_ids = even_zernike_composite_ids(PSF_model.Z_mode_max)
            LO_orig  = PSF_model.inputs_manager['LO_coefs'].detach().clone()
            PSF_model.inputs_manager['LO_coefs'][:, even_ids] = -LO_orig[:, even_ids]
            x_f, PSF_f, OPD_f, _ = minimize_params(include, exclude, 100, verbose=False)
            
            if (loss_f := loss_fn(x_f).item()) <= loss_val * (1 + FLIP_TOL): # the mirror branch (equal loss up to the asymmetries of the pupil / kernels)
                x, PSF, OPD, loss_val, flipped = x_f, PSF_f, OPD_f, loss_f, True
            else:
                print(f"The reflected even modes cost {100 * (loss_f / loss_val - 1):.1f} % of loss, keeping the negative-defocus branch")
                PSF_model.inputs_manager['LO_coefs'][:] = LO_orig
                _ = PSF_model.inputs_manager.unstack(x, include_all=True, update=True)

        defocus = float(PSF_model.inputs_manager['LO_coefs'][0, 1]) if PSF_model.LO_NCPAs else float('nan')
        return dict(x=x, PSF=PSF, OPD_map=OPD, loss=loss_val, peak_error=peak_error(PSF_0, PSF.detach()), flipped=flipped, success=ok, defocus=defocus)

    results = {}
    
    for start in starts:
        results[start] = r = fit_from(start)
        if verbose:
            print(f"start defocus {start:+.0f} nm: loss {r['loss']:.4f}, median |peak err| {100 * r['peak_error']:.2f} %, defocus {r['defocus']:+.1f} nm, "
                  f"reflected {r['flipped']}, converged {r['success']}")

    best_start = min(results, key=lambda s: results[s]['peak_error' if select_by == 'peak' else 'loss'])
    best = dict(results[best_start], start=best_start, starts={s: (r['loss'], r['peak_error'], r['defocus'], r['flipped'], r['success']) for s, r in results.items()})
    _ = PSF_model.inputs_manager.unstack(best['x'], include_all=True, update=True) # leave the model in the kept state
    
    return best


def load_and_fit_sample(id, device=default_device, kc_fraction=None, verbose=False):
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

    # The configs carry the k_c of the samples' labels (scaled DM pitch): rescale them to the nominal / forced k_c when asked
    kc_fraction = kc_fraction if kc_fraction is not None else (None if APPLY_KC_CORRECTION else 1.0)
    if kc_fraction is not None:
        configs = [apply_kc_fraction(c, kc_fraction / kc_fraction_from_config(c)) for c in configs]

    PSF_model = build_PSF_model(configs, device)

    # Model k_c relative to the nominal one, recorded during dataset cache generation before fitting
    # During training: samples with k_c mismatch are grouped based on their fraction (group 1..N), 1.0 for the others (group 0), or a forced value (group -1)
    kc_fraction    = kc_fraction_from_config(configs[0])
    label_fraction = float(psf_df.loc[id][KC_FRACTION_FEATURE]) if KC_FRACTION_FEATURE in psf_df.columns else 1.0
    kc_group = int(psf_df.loc[id][KC_GROUP_FEATURE]) if KC_GROUP_FEATURE in psf_df.columns and np.isclose(kc_fraction, label_fraction) else (0 if np.isclose(kc_fraction, 1.0) else -1)

    if not np.isclose(kc_fraction, 1.0):
        print(f"Sample {id}: {f'flagged k_c mismatch (group {kc_group})' if kc_group > 0 else 'forced k_c'}: model k_c = {100 * kc_fraction:.1f} % of the nominal "
              f"({PSF_model.model.kc:.3f} 1/m, DM pitch {np.asarray(configs[0]['DM']['DmPitchs']).ravel()[0]:.4f} m)")

    flags = sample_fit_flags(id)
    func, loss_fn    = build_loss(PSF_model, PSF_0, flags, device)
    minimize_params  = make_minimizer(PSF_model, func, loss_fn)
    include, exclude = parameter_sets(PSF_model, flags)

    fit = multistart_fit(PSF_model, PSF_0, loss_fn, minimize_params, include, exclude, flags, verbose=verbose)
    x0, PSF_1, OPD_map, loss_val = fit['x'], fit['PSF'], fit['OPD_map'], fit['loss']
    print(f"Sample {id}: kept the start at {fit['start']:+.0f} nm by {SELECT_BY}: loss {loss_val:.4f}, median |peak err| {100 * fit['peak_error']:.2f} %"
          + (' (other start(s): ' + ', '.join(f'{s:+.0f} nm loss {v[0]:.4f} peak {100 * v[1]:.2f} %' for s, v in fit['starts'].items() if s != fit['start']) + ')' if len(fit['starts']) > 1 else ''))

    fitted_values = to_store(PSF_model.inputs_manager.unstack(x0, include_all=True, update=True))

    # Compute diffraction-limited PSF for SR calculation
    PSF_DL = PSF_model.model.DLPSF()

    # Prepare data for saving
    config_manager = ConfigManager()
    config_dict = configs[0] if isinstance(configs, list) else configs
    config_manager.Convert(config_dict, framework='numpy')

    loss_type = PSF_LOSS_CFG['type']
    save_data = {
        'comments':    f'NFM fitting with re-normalized dn, {loss_type} PSF loss, phase bump pulled to 0 unless flagged "Non-point", '
                       f'{"chromatic defocus + achromatic" if CHROM_DEFOCUS else "chromatic"} {"anisotropic (Jx, Jy, J_angle)" if ANISOTROPIC_JITTER else "isotropic"} jitter'
                       f'{", image-frame wing (wing_frac, wing_width, wing_width_t)" if WING_KERNEL else ""}, '
                       f'{len(DEFOCUS_STARTS)}-start fit selected by {SELECT_BY}',
                       
        'loss config':        PSF_LOSS_CFG,
        'chrom_defocus':      bool(CHROM_DEFOCUS),
        'anisotropic_jitter': bool(ANISOTROPIC_JITTER),
        'wing_kernel':        bool(WING_KERNEL),
        'wing width fitted':  bool(FIT_WING_WIDTH),
        'wing transverse width fitted': bool(FIT_WING_WIDTH_T),
        'wing frac max':      float(WING_FRAC_MAX),
        'wing power':         float(WING_POWER),
        'blurry':             bool(flags['blurry']), # LO coefficients and wing fraction pulled to the dataset priors, L0 / wind speed / Cn2 at the config values
        'defocus sign':       DEFOCUS_SIGN,
        'even modes flipped': bool(fit['flipped']),
        'defocus starts':     tuple(DEFOCUS_STARTS),
        'selected by':        SELECT_BY,
        'start':              fit['start'],   # starting defocus [nm] of the kept solution
        'start results':      fit['starts'],  # {start: (loss, median |peak error|, defocus [nm], even modes flipped, converged)}
        'peak error':         fit['peak_error'],
        'priors':             None if FIT_PRIORS is None else {k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v) for k, v in FIT_PRIORS.items()},
        'bump suppressed': bool(flags['suppress_bump']),
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
        'wavelengths': to_store(PSF_model.λ_sim),
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
          f"chromatic defocus: {CHROM_DEFOCUS} | anisotropic jitter: {ANISOTROPIC_JITTER} | wing kernel: {WING_KERNEL} | "
          f"defocus starts {DEFOCUS_STARTS} nm selected by {SELECT_BY}")

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
