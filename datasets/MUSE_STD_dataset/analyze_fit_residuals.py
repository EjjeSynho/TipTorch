#%%
"""
2D maps of the median PSF-space residual of a set of STD-star fits (fitted/<folder>/{ID}.pickle, 'Img. data' vs 'Img. fit', both sum-normalized).

Per pixel, over the samples:
  median residual      median of (model - data) / peak(data)              linear, the core and the AO-corrected area, in % of the data peak
  median log ratio     median of log10(model / data) where data > 3 sigma   the halo, where a linear residual is invisible
  sign consistency     fraction of the samples with model > data            systematic (0 or 1) vs random (0.5) residuals
  median |residual|    median of |model - data| / peak(data)               the typical pixel error, whatever its sign
for the selected wavelength bins and for the wavelength average of each sample, in two frames:
  raw     the detector frame of the fits (pixel-exact: the right frame for the core and anything fixed on the detector / the sky)
  pupil   each sample's maps rotated by -PupilAngle (bilinear), the frame of the spiders, the LO NCPAs and the AO-corrected area; the
          rotation blurs the central pixels, so read the core in the raw frame
plus the azimuthal profiles of the median maps and a comparison panel of a second fit folder if given.

    python datasets/MUSE_STD_dataset/analyze_fit_residuals.py        (or cell by cell; the settings are the constants in the first cell)

Outputs: data/analysis/NFM_STD_residuals/<folder>/residual_maps_{full,zoom}_{raw,pupil}.png, residual_profiles.png, residual_maps.npz,
residual_summary.csv [and compare_<folder>_<compare>.png]. CPU, ~3 min for 534 samples (needs ~3 GB of RAM).
"""
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import rotate

_here = Path(__file__).resolve().parent if '__file__' in dir() else Path.cwd().resolve()
sys.path.insert(0, str(_here)); sys.path.insert(0, str(_here.parent.parent))
from tiptorch._config import project_settings, DATA_FOLDER

#%% Settings (edit here; the script is cell-runnable, no command-line arguments)
FOLDER  = 'fitted'           # fit folder under the STD-star folder
COMPARE = 'fitted_balanced'  # second fit folder for a side-by-side comparison panel, None to skip
WVL_IDS = [0, 5, 10, 15]     # wavelength bins to map (of the 16 fitted ones)
ZOOM    = 31                 # side of the central zoom [px]
N_LIMIT = 0                  # limit the number of samples (0 = all)
NOISE_K = 3.0                # data > NOISE_K x sigma defines where the log ratio is taken

STD_FOLDER = Path(project_settings['MUSE_STD_data_folder'])
OUT = DATA_FOLDER / 'analysis' / 'NFM_STD_residuals' / FOLDER
OUT.mkdir(parents=True, exist_ok=True)
C_DIV, C_SEQ, C_GREY = 'RdBu_r', 'viridis', '#9AA0A6'


def noise_sigma(D, r, r_min=85.0):
    """ Robust noise level of the background per wavelength, [N_wvl] (same definition as BalancedPSFLoss). """
    bg  = D[:, r >= r_min]
    med = np.median(bg, axis=-1, keepdims=True)
    return np.maximum(np.median(np.abs(bg - med), axis=-1) * 1.4826, 1e-6 * D.max(axis=(-2, -1)))


def to_pupil_frame(cube, angle):
    """ Rotate [N_wvl, H, W] maps by -angle about the array centre (the derotation of LoadSTDStarCache), bilinear. """
    return np.stack([rotate(c, -angle, reshape=False, order=1, mode='constant', cval=np.nan) for c in cube])


def per_sample_maps(d, wvl_ids, noise_k):
    """ Residual maps of one fit: dict frame -> (lin [N_sel+1, H, W], logratio [N_sel+1, H, W]); the last entry is the wavelength average. """
    D, P = d['Img. data'][0].astype(np.float32), d['Img. fit'][0].astype(np.float32)
    N_wvl, H, W = D.shape; c = H // 2
    yy, xx = np.mgrid[:H, :W]; r = np.hypot(yy - c, xx - c)
    angle = float(np.asarray(d['config']['telescope']['PupilAngle']).ravel()[0])
    out = {}
    
    for frame in ('raw', 'pupil'):
        Df, Pf = (D, P) if frame == 'raw' else (to_pupil_frame(D, angle), to_pupil_frame(P, angle))
        peak = np.nanmax(Df[:, c-3:c+4, c-3:c+4], axis=(-2, -1))[:, None, None]
        lin  = (Pf - Df) / peak
        sig  = noise_sigma(np.nan_to_num(Df), r)[:, None, None] * noise_k
        
        with np.errstate(divide='ignore', invalid='ignore'):
            logr = np.where((Df > sig) & (Pf > 0), np.log10(Pf / Df), np.nan)
            
        sel = lambda a: np.concatenate([a[wvl_ids], np.nanmean(a, axis=0, keepdims=True)]).astype(np.float32)
        out[frame] = (sel(lin), sel(logr))
        
    return out, angle


def load_folder(folder, ids=None):
    files = sorted((STD_FOLDER / folder).glob('*.pickle'), key=lambda p: int(p.stem))
    if ids is not None:
        files = [f for f in files if int(f.stem) in ids]
        
    if N_LIMIT:
        files = files[::max(1, len(files) // N_LIMIT)][:N_LIMIT]
        
    stacks = {fr: ([], []) for fr in ('raw', 'pupil')}
    meta = []
    
    for f in tqdm(files, desc=folder):
        d = pickle.load(open(f, 'rb'))
        maps, angle = per_sample_maps(d, WVL_IDS, NOISE_K)
        
        for fr in stacks:
            stacks[fr][0].append(maps[fr][0]); stacks[fr][1].append(maps[fr][1])
            
        meta.append(dict(ID=int(f.stem), angle=angle, λ_nm=(np.asarray(d['wavelengths']).ravel() * 1e9)[WVL_IDS].round(0).tolist(),
                         SR=float(np.median(d['SR data']))))
        
    stacks = {fr: (np.stack(v[0]), np.stack(v[1])) for fr, v in stacks.items()}  # [N, N_sel+1, H, W]
    
    return stacks, pd.DataFrame(meta)


def aggregate(stacks):
    """ Per-pixel statistics over the samples: dict frame -> dict(name -> [N_sel+1, H, W]). """
    agg = {}
    for fr, (lin, logr) in stacks.items():
        agg[fr] = {
            'median residual [% of peak]': 100 * np.nanmedian(lin, axis=0),
            'median log10(model / data)':  np.nanmedian(logr, axis=0),
            'fraction model > data':       np.nanmean(lin > 0, axis=0),
            'median |residual| [% of peak]': 100 * np.nanmedian(np.abs(lin), axis=0),
        }
    return agg


#%% Load and aggregate
stacks, meta = load_folder(FOLDER)
agg = aggregate(stacks)
labels = [f'{λ:.0f} nm' for λ in meta['λ_nm'].iloc[0]] + ['λ average']
H = agg['raw']['median residual [% of peak]'].shape[-1]; c = H // 2
print(f'{len(meta)} samples from {FOLDER}, bins {labels[:-1]}, pupil angles {meta.angle.min():.0f}..{meta.angle.max():.0f} deg')

np.savez_compressed(OUT / 'residual_maps.npz', labels=np.array(labels), ids=meta.ID.values,
                    **{f'{fr}__{k}': v for fr, dd in agg.items() for k, v in dd.items()})

def plot_maps(agg_frame, frame, zoom, fname, title):
    names = list(agg_frame.keys())
    s = np.s_[c - zoom // 2: c + zoom // 2 + 1, c - zoom // 2: c + zoom // 2 + 1] if zoom else np.s_[:, :]
    fig, axes = plt.subplots(len(labels), len(names), figsize=(3.6 * len(names), 3.4 * len(labels)), squeeze=False)
    
    for j, name in enumerate(names):
        maps = agg_frame[name][:, s[0], s[1]]
        
        if 'fraction' in name:
            vmin, vmax, cmap = 0.0, 1.0, C_DIV
        elif '|residual|' in name:
            vmin, vmax, cmap = 0.0, np.nanpercentile(maps, 99), C_SEQ
        else:
            lim = np.nanpercentile(np.abs(maps), 99); vmin, vmax, cmap = -lim, lim, C_DIV
            
        for i in range(len(labels)):
            ax = axes[i, j]
            im = ax.imshow(maps[i], vmin=vmin, vmax=vmax, cmap=cmap, origin='lower', interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            
            if i == 0:
                ax.set_title(name, fontsize=10)
            if j == 0:
                ax.set_ylabel(labels[i], fontsize=10)
            if zoom:  # mark the centre pixel
                ax.plot(zoom // 2, zoom // 2, '+', color='k', ms=6, mew=0.8, alpha=0.6)
                
        fig.colorbar(im, ax=axes[:, j].tolist(), shrink=0.6, pad=0.02)
        
    fig.suptitle(title, fontsize=11)
    fig.savefig(OUT / fname, dpi=130, bbox_inches='tight'); plt.close(fig)


for frame in ('raw', 'pupil'):
    fr_txt = 'detector frame (as fitted)' if frame == 'raw' else 'pupil frame (rotated by -PupilAngle, core blurred by the rotation)'
    plot_maps(agg[frame], frame, 0,          f'residual_maps_full_{frame}.png', f'{FOLDER}: median PSF residual over {len(meta)} stars, full field, {fr_txt}')
    plot_maps(agg[frame], frame, ZOOM, f'residual_maps_zoom_{frame}.png', f'{FOLDER}: median PSF residual over {len(meta)} stars, central {ZOOM} px, {fr_txt}')

#%% Azimuthal profiles of the median maps (raw frame) and a per-band summary
yy, xx = np.mgrid[:H, :H]; r = np.hypot(yy - c, xx - c)
edges = np.concatenate([np.arange(0, 6, 1.0), np.arange(6, 20, 2.0), np.arange(20, 100, 5.0)])
rc = 0.5 * (edges[1:] + edges[:-1])
def profile(m):
    return np.array([np.nanmean(m[(r >= a) & (r < b)]) for a, b in zip(edges[:-1], edges[1:])])

fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
colors = plt.cm.viridis(np.linspace(0, 0.9, len(labels) - 1)).tolist() + ['k']
for i, lab in enumerate(labels):
    ax[0].plot(rc, profile(agg['raw']['median residual [% of peak]'][i]), '-o', ms=3, color=colors[i], label=lab)
    ax[1].plot(rc, profile(agg['raw']['median log10(model / data)'][i]), '-o', ms=3, color=colors[i], label=lab)
    ax[2].plot(rc, profile(agg['raw']['median |residual| [% of peak]'][i]), '-o', ms=3, color=colors[i], label=lab)
    
for a in ax:
    a.set_xscale('symlog', linthresh=5); a.set_xlabel('radius [px]'); a.axhline(0, color=C_GREY, lw=1); a.legend(fontsize=8)

ax[0].set_ylabel('azimuthal mean of the median residual [% of peak]'); ax[0].set_title('Linear residual (model - data)')
ax[1].set_ylabel('azimuthal mean of the median log10(model / data)'); ax[1].set_title('Log ratio (halo)')
ax[2].set_ylabel('azimuthal mean of the median |residual| [% of peak]'); ax[2].set_title('Typical pixel error')
fig.suptitle(f'{FOLDER}: radial profiles of the median residual maps (detector frame)', fontsize=11)
fig.tight_layout(); fig.savefig(OUT / 'residual_profiles.png', dpi=130); plt.close(fig)

rows = []
for i, lab in enumerate(labels):
    for a, b in [(0, 1.5), (1.5, 3.5), (3.5, 10.5), (10.5, 20.5), (20.5, 40.5), (40.5, 80.5)]:
        m = (r >= a) & (r < b)
        rows.append(dict(bin=lab, band=f'{a:g}-{b:g}',
                         med_residual_pct=float(np.nanmean(agg['raw']['median residual [% of peak]'][i][m])),
                         med_log_ratio=float(np.nanmean(agg['raw']['median log10(model / data)'][i][m])),
                         frac_model_gt_data=float(np.nanmean(agg['raw']['fraction model > data'][i][m])),
                         med_abs_residual_pct=float(np.nanmean(agg['raw']['median |residual| [% of peak]'][i][m]))))
        
summary = pd.DataFrame(rows); summary.to_csv(OUT / 'residual_summary.csv', index=False)

print('\nazimuthal means of the median maps (detector frame):')
print(summary.pivot(index='band', columns='bin', values='med_residual_pct').round(2).to_string(), '\n  ^ median residual [% of peak]')
print(summary.pivot(index='band', columns='bin', values='med_log_ratio').round(3).to_string(), '\n  ^ median log10(model / data)')
print('centre pixel, λ average: median residual %.2f %% of peak, model > data in %.0f %% of the stars, median |residual| %.2f %%' % (
    agg['raw']['median residual [% of peak]'][-1][c, c], 100 * agg['raw']['fraction model > data'][-1][c, c], agg['raw']['median |residual| [% of peak]'][-1][c, c]))

#%% Optional comparison with a second fit set (λ average, raw frame)
if COMPARE:
    stacks2, meta2 = load_folder(COMPARE, ids=set(meta.ID))
    agg2 = aggregate(stacks2)
    s = np.s_[c - ZOOM // 2: c + ZOOM // 2 + 1, c - ZOOM // 2: c + ZOOM // 2 + 1]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    
    for row, (name, s_) in enumerate([('median residual [% of peak]', s), ('median log10(model / data)', np.s_[:, :])]):
        a1, a2 = agg['raw'][name][-1][s_], agg2['raw'][name][-1][s_]
        lim = np.nanpercentile(np.abs(np.concatenate([a1, a2])), 99)
        
        for col, (m, t) in enumerate([(a1, FOLDER), (a2, COMPARE), (a1 - a2, f'{FOLDER} - {COMPARE}')]):
            im = axes[row, col].imshow(m, vmin=-lim, vmax=lim, cmap=C_DIV, origin='lower', interpolation='nearest')
            axes[row, col].set_title(f'{t}\n{name}, λ average', fontsize=9); axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
        
        fig.colorbar(im, ax=axes[row].tolist(), shrink=0.7, pad=0.02)
    
    fig.savefig(OUT / f'compare_{FOLDER}_{COMPARE}.png', dpi=130, bbox_inches='tight'); plt.close(fig)
   
    print(f'comparison panel saved for {len(meta2)} common samples')

print(f'\nSaved to {OUT}')

# %%
