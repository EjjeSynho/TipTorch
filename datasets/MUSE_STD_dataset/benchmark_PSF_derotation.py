#%%
"""
Benchmark of the accuracy lost around the PSF peak when MUSE NFM STD star PSFs are derotated with RotatePSF.

MUSE NFM cubes are strongly undersampled (25 mas pixels vs. lambda/D = 12-24 mas), so the cubic spline interpolation
inside RotatePSF smooths the core and redistributes flux from the peak into the neighboring pixels.
Two tests are done:
    1. Real data, round-trip: the cube is derotated by -theta and rotated back by +theta, then compared to the original.
       There is no ground truth for a single derotation of the real data, but the round trip returns to the original
       pixel grid, so the difference is purely the interpolation error accumulated over two passes.
    2. Synthetic data, exact ground truth: long-exposure NFM-like PSFs (UT4 pupil, AO-corrected von Karman PSD) are computed
       7x oversampled and integrated over the 25 mas pixels at pupil angle theta (observed) and 0 (truth). The observed PSF is
       derotated with RotatePSF and compared to the truth. Star offsets from the rotation center and pupil angles are taken from
       the real samples. The synthetic round trip is computed as well to relate the real-data round trip to a single derotation.
Errors are normalized to the peak value of the reference PSF and binned radially around the PSF centroid.
"""
try:
    ipy = get_ipython() # NameError if not running under IPython
    if ipy:
        ipy.run_line_magic('reload_ext', 'autoreload')
        ipy.run_line_magic('autoreload', '2')
except NameError:
    pass

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pickle
import warnings
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as sfft
from astropy.io import fits
from scipy.ndimage import map_coordinates
from tqdm import tqdm

from STD_dataset_utils import STD_FOLDER, LoadSTDStarCache, RotatePSF
from tiptorch.tools.utils import safe_centroid

warnings.filterwarnings('ignore')

N_real   = 100  # number of real STD stars used for the round-trip test
N_synth  = 24   # number of synthetic PSF cubes (pupil angles and star offsets are borrowed from the first N_synth real samples)
r_max    = 40   # [pix], outer radius of the radial profiles
r_core   = 12   # [pix], outer radius shown in the plots of the signed error
seed     = 42
save_dir = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else None # optional folder to save the figures

pixel_scale = 25.0 # [mas]

# Synthetic PSF parameters
oversampling = 7     # odd, so that the center of the high-res grid coincides with the center of the central detector pixel
N_fft        = 1400  # high-res FFT grid, must be >= N_pix*oversampling
D_tel        = 8.0   # [m]
DM_pitch     = 0.22  # [m], defines the AO control radius
r0_500       = 0.12  # [m] at 500 nm
L0           = 25.0  # [m]
AO_residual  = 0.02  # fraction of the uncorrected PSD remaining inside the control radius (servo-lag, aliasing, noise lumped together)

pupil_path = Path.home() / '.tiptorch/calibrations/VLT_CALIBRATION/VLT_PUPIL/ut4pupil320.fits'


#%% ------------------------------------------- Helpers -------------------------------------------
def derotate(PSF, angle):
    return RotatePSF(PSF, -angle)


def round_trip(PSF, angle):
    return RotatePSF(RotatePSF(PSF, -angle), angle)


def radial_errors(PSF_est, PSF_ref, r_max=r_max):
    """
    Per-wavelength error metrics of [N_wvl x N x N] estimate vs. reference, normalized to the peak of the reference.
    Returns signed annular mean [N_wvl x r_max], annular RMS [N_wvl x r_max], and the error in the peak pixel [N_wvl].
    """
    N_wvl, ny, nx = PSF_ref.shape
    yy, xx = np.mgrid[:ny, :nx]
    signed, rms, peak = np.zeros([N_wvl, r_max]), np.zeros([N_wvl, r_max]), np.zeros(N_wvl)

    for i in range(N_wvl):
        ref, diff = PSF_ref[i], PSF_est[i] - PSF_ref[i]
        x0, y0 = safe_centroid(ref)
        r_ids  = np.floor(np.hypot(xx - x0, yy - y0)).astype(int).ravel()
        mask   = r_ids < r_max
        counts = np.bincount(r_ids[mask], minlength=r_max)
        p      = np.unravel_index(np.argmax(ref), ref.shape)
        norm   = ref[p]

        signed[i] = np.bincount(r_ids[mask], weights=diff.ravel()[mask],    minlength=r_max) / counts / norm
        rms[i]    = np.sqrt(np.bincount(r_ids[mask], weights=diff.ravel()[mask]**2, minlength=r_max) / counts) / norm
        peak[i]   = diff[p] / norm

    return signed, rms, peak


def stack_metrics(metrics):
    return {key: np.stack([m[k] for m in metrics]) for k, key in enumerate(['signed', 'rms', 'peak'])}


#%% ------------------------------------------- Real data: round trip -------------------------------------------
with open(STD_FOLDER / 'muse_df.pickle', 'rb') as handle:
    psf_df = pickle.load(handle)

psf_df   = psf_df[(psf_df['Corrupted'] == False) & (psf_df['Bad quality'] == False)]
rng      = np.random.default_rng(seed)
real_ids = np.sort(rng.choice(psf_df.index.values, size=min(N_real, len(psf_df)), replace=False)).tolist()

# Same preprocessing as in STD_fitter.py, but without derotation
PSF_kwargs = dict(derotate_PSF=False, normalize=True, subtract_background=True, ensure_odd_pixels=True, device='cpu')

metrics_real, angles, offsets = [], [], []

for id in tqdm(real_ids, desc='Real data'):
    PSF_0, _, _, configs = LoadSTDStarCache([id], **PSF_kwargs)
    PSF_0 = PSF_0[0].numpy()
    angle = psf_df.loc[id, 'Pupil angle'].item()

    metrics_real.append(radial_errors(round_trip(PSF_0, angle), PSF_0))

    # Star offset (x,y) from the rotation center of scipy.ndimage.rotate, in pixels
    angles.append(angle)
    offsets.append(np.array(safe_centroid(PSF_0.mean(axis=0))) - (PSF_0.shape[-1]-1) / 2)

wvls   = np.array(configs[0]['sources_science']['Wavelength']).flatten() * 1e9 # [nm]
N_pix  = PSF_0.shape[-1]
real   = stack_metrics(metrics_real)
angles, offsets = np.array(angles), np.array(offsets)

print(f'Star offset from the rotation center: median |offset| = {np.median(np.linalg.norm(offsets, axis=1)):.2f} pix')


#%% ------------------------------------------- Synthetic NFM PSFs with exact ground truth -------------------------------------------
assert oversampling % 2 == 1 and N_fft >= N_pix*oversampling

pupil_fits = fits.getdata(pupil_path).astype(np.float64)
pix_hr_rad = pixel_scale / oversampling / 1e3 / 206264.806 # high-res pixel [rad]

J = sfft.fftfreq(N_fft) * N_fft # signed integer frequency indices
crop = slice(N_fft//2 - N_pix*oversampling//2, N_fft//2 + N_pix*oversampling//2 + 1)


def rot_xy(v, angle):
    """ Rotates (x,y) vector(s) by angle [deg] counterclockwise in the (column, row) array coordinates """
    a = np.deg2rad(angle)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return v @ R.T


def pupil_coords(λ):
    dx = λ / (N_fft * pix_hr_rad) # pupil plane sampling [m]
    u  = (np.arange(N_fft) - N_fft//2) * dx
    return (*np.meshgrid(u, u), dx)


def sample_pupil(λ, angle):
    """
    UT4 pupil on the FFT grid, rotated so that the resulting PSF matches scipy.ndimage.rotate(PSF_0, angle).
    scipy.ndimage.rotate(img, angle) moves the content at (x,y) to rot_xy((x,y), -angle), so the pupil is sampled at rot_xy(u, angle).
    """
    x, y, _ = pupil_coords(λ)
    xy = rot_xy(np.stack([x.ravel(), y.ravel()], axis=1), angle)
    c  = (pupil_fits.shape[0]-1) / 2
    scale = pupil_fits.shape[0] / D_tel # [pix/m]
    P = map_coordinates(pupil_fits, [c + xy[:,1]*scale, c + xy[:,0]*scale], order=1, mode='constant', cval=0.0)
    return P.reshape(N_fft, N_fft)


def telescope_OTF(λ, angle):
    E = sfft.fft2(sfft.ifftshift(sample_pupil(λ, angle)).astype(np.complex64), workers=-1)
    return sfft.ifft2(np.abs(E)**2, workers=-1) # zero lag at [0,0]


def atmospheric_OTF(λ, angle):
    """
    Long-exposure OTF of the AO residuals: von Karman PSD outside of the control radius, scaled down inside.
    The square control region of the DM is attached to the pupil, so it rotates together with it.
    """
    _, _, dx = pupil_coords(λ)
    f  = sfft.fftfreq(N_fft, d=dx)
    fx, fy = np.meshgrid(f, f)
    fx, fy = np.moveaxis(rot_xy(np.stack([fx, fy], axis=-1), angle), -1, 0)
    f2 = fx**2 + fy**2
    r0 = r0_500 * (λ/500e-9)**1.2
    PSD = 0.023 * r0**(-5/3) * (f2 + 1/L0**2)**(-11/6) # [rad^2 m^2]
    PSD[np.maximum(np.abs(fx), np.abs(fy)) < 1/(2*DM_pitch)] *= AO_residual # square control region of the Fried-geometry DM
    PSD[0,0] = 0.0
    df = f[1] - f[0]
    B  = np.real(sfft.ifft2(PSD, workers=-1)) * N_fft**2 * df**2 # phase covariance vs. lag
    return np.exp(-(B[0,0] - B)).astype(np.float32) # exp(-D_phi/2), D_phi = 2*(B(0)-B(ρ))


def detector_image(OTF, shift_xy):
    """ PSF centered at shift_xy [detector pixels] from the central pixel, integrated over the detector pixels, normalized to 1 """
    sx, sy = np.asarray(shift_xy) * oversampling
    ramp = np.exp(2j*np.pi*J[:,None]*sy/N_fft) * np.exp(2j*np.pi*J[None,:]*sx/N_fft)
    PSF_hr = np.real(sfft.fftshift(sfft.fft2(OTF * ramp, workers=-1)))[crop, crop]
    PSF = PSF_hr.reshape(N_pix, oversampling, N_pix, oversampling).sum(axis=(1,3))
    return PSF / PSF.sum()


λs = wvls * 1e-9
OTF_atm0 = [atmospheric_OTF(λ, 0.0) for λ in tqdm(λs, desc='Atmospheric OTFs')]
OTF_tel0 = [telescope_OTF(λ, 0.0) for λ in λs]

SR_synth = np.array([np.real(np.sum(T*A)) / np.real(np.sum(T)) for T, A in zip(OTF_tel0, OTF_atm0)])
print('Synthetic PSF Strehl ratio: ' + ', '.join(f'{w:.0f} nm: {s*100:.1f}%' for w, s in zip(wvls[::10], SR_synth[::10])))

metrics_synth_1, metrics_synth_rt = [], []

for k in tqdm(range(min(N_synth, len(angles))), desc='Synthetic data'):
    angle = angles[k]
    shift_obs   = offsets[k]
    shift_truth = rot_xy(shift_obs, angle) # star position after a perfect derotation

    PSF_obs   = np.stack([detector_image(telescope_OTF(λ, angle) * atmospheric_OTF(λ, angle), shift_obs) for λ in λs])
    PSF_truth = np.stack([detector_image(T * A, shift_truth) for T, A in zip(OTF_tel0, OTF_atm0)])

    metrics_synth_1.append (radial_errors(derotate(PSF_obs, angle), PSF_truth))
    metrics_synth_rt.append(radial_errors(round_trip(PSF_obs, angle), PSF_obs))

    # Sanity check of the rotation direction: derotating in the opposite direction must be much worse (except for ~0/180 deg)
    if k == 0 and 20 < angle % 180 < 160:
        err_good = np.abs(derotate(PSF_obs, angle)  - PSF_truth).sum()
        err_bad  = np.abs(derotate(PSF_obs, -angle) - PSF_truth).sum()
        assert err_good < err_bad, 'Rotation direction of the synthetic PSFs is inconsistent with RotatePSF'

synth_1  = stack_metrics(metrics_synth_1)
synth_rt = stack_metrics(metrics_synth_rt)


#%% ------------------------------------------- Summary -------------------------------------------
def q(x, axis=0):
    return np.percentile(x, [16, 50, 84], axis=axis)

print('\nPeak-pixel error, % of peak (median [16th, 84th percentile] over samples)')
print(f"{'Wvl, nm':>7} | {'real, round trip':>24} | {'synthetic, round trip':>24} | {'synthetic, single (exact)':>25}")
for i in list(range(0, len(wvls), 5)) + ['avg']:
    sel = (lambda x: x.mean(axis=1)) if i == 'avg' else (lambda x, i=i: x[:,i])
    cells = [q(sel(m['peak'])*100) for m in (real, synth_rt, synth_1)]
    label = 'mean' if i == 'avg' else f'{wvls[i]:.0f}'
    print(f'{label:>7} | ' + ' | '.join(f'{c[1]:+6.2f} [{c[0]:+6.2f}, {c[2]:+6.2f}]'.rjust(24) for c in cells))

ratio = np.median(synth_1['peak'].mean(axis=1) / synth_rt['peak'].mean(axis=1))
print(f'\nSynthetic single-pass / round-trip peak error ratio: {ratio:.2f}')


#%% ------------------------------------------- Plots -------------------------------------------
INK, INK_2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
SERIES = {'Real data, round trip': '#2a78d6', 'Synthetic, round trip': '#eb6834', 'Synthetic, single derotation (exact)': '#1baf7a'}

plt.rcParams.update({
    'axes.edgecolor': INK_2, 'axes.labelcolor': INK, 'xtick.color': INK_2, 'ytick.color': INK_2,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.titlesize': 11, 'axes.labelsize': 10, 'legend.frameon': False, 'lines.linewidth': 1.5,
})

r_mas = (np.arange(r_max) + 0.5) * pixel_scale # annulus centers
cmap  = plt.matplotlib.colors.LinearSegmentedColormap.from_list('wvl', plt.cm.Blues(np.linspace(0.35, 1.0, 256)))
norm  = plt.Normalize(wvls.min(), wvls.max())


def save(fig, name):
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / f'{name}.png', dpi=150, bbox_inches='tight')


# Per-wavelength radial profiles
fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex='col', constrained_layout=True)

for row, (data, title) in enumerate([(real, 'Real data, round trip (−θ, +θ)'), (synth_1, 'Synthetic, single derotation vs. exact truth')]):
    signed_med = np.median(data['signed'], axis=0) * 100
    rms_med    = np.median(data['rms'],    axis=0) * 100
    for i in range(len(wvls)):
        axs[row,0].plot(r_mas, signed_med[i], color=cmap(norm(wvls[i])), lw=1.0)
        axs[row,1].plot(r_mas, rms_med[i],    color=cmap(norm(wvls[i])), lw=1.0)
    axs[row,0].axhline(0, color=INK_2, lw=0.8)
    axs[row,0].set_ylabel('Mean error in annulus, % of peak')
    axs[row,1].set_ylabel('RMS error in annulus, % of peak')
    axs[row,1].set_yscale('log')
    axs[row,0].set_title(f'{title}: signed', loc='left')
    axs[row,1].set_title(f'{title}: RMS',    loc='left')

for ax in axs[1]:
    ax.set_xlabel('Radius from PSF centroid, mas')
axs[0,1].set_xlim(0, r_max * pixel_scale)
for ax in axs[:,0]:
    ax.set_xlim(0, r_core * pixel_scale)

fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs, label='Wavelength, nm', shrink=0.6)
fig.suptitle(f'RotatePSF error per wavelength bin, median over samples (real: {len(real_ids)}, synthetic: {len(synth_1["peak"])})', color=INK)
save(fig, 'derotation_radial_per_bin')


# Spectrally averaged radial profiles
fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

for (label, color), data in zip(SERIES.items(), (real, synth_rt, synth_1)):
    for ax, key in zip(axs, ['signed', 'rms']):
        lo, med, hi = q(data[key].mean(axis=1) * 100) # average over wavelengths first, then spread over samples
        ax.fill_between(r_mas, lo, hi, color=color, alpha=0.12, lw=0)
        ax.plot(r_mas, med, color=color, label=label)

axs[0].axhline(0, color=INK_2, lw=0.8)
axs[0].set_ylabel('Mean error in annulus, % of peak')
axs[1].set_ylabel('RMS error in annulus, % of peak')
axs[1].set_yscale('log')
axs[0].set_title('Signed', loc='left')
axs[1].set_title('RMS',    loc='left')
for ax, r_lim in zip(axs, [r_core, r_max]):
    ax.set_xlabel('Radius from PSF centroid, mas')
    ax.set_xlim(0, r_lim * pixel_scale)
axs[1].legend(loc='upper right')
fig.suptitle('RotatePSF error, spectrally averaged (median, band: 16th–84th percentile over samples)', color=INK)
save(fig, 'derotation_radial_averaged')


# Peak-pixel error vs. wavelength
fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
for (label, color), data in zip(SERIES.items(), (real, synth_rt, synth_1)):
    lo, med, hi = q(data['peak'] * 100)
    ax.fill_between(wvls, lo, hi, color=color, alpha=0.12, lw=0)
    ax.plot(wvls, med, color=color, label=label)
ax.axhline(0, color=INK_2, lw=0.8)
ax.set_xlabel('Wavelength, nm')
ax.set_ylabel('Peak-pixel error, % of peak')
ax.set_title('Error in the peak pixel', loc='left')
ax.legend(loc='lower right')
save(fig, 'derotation_peak_vs_wavelength')

plt.show()

# %%
