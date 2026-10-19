#%%
"""

ssh -L 8765:localhost:8765 ...

Interactive PSF shape explorer for the MUSE NFM STD stars: a page in the browser (standalone) or ipywidgets (VS Code interactive window,
Jupyter).

Pick a sample on the right and press "Load": the PSF model (PSFModelNFM, set up as in datasets/MUSE_STD_dataset/STD_fitter.py) starts
from the per-sample fit (<MUSE_fitted_STD_folder>/<id>.pickle), in the regime of that fit (derotated cube for a config PupilAngle of 0,
else the raw frame with the model at the fit's pupil angle; the DM pitch, i.e. the k_c, of the fit), and every fitted input gets a slider
+ a text box. The LGS asterism knobs act in the pupil frame; the source pad draws the asterism turned by the pupil angle, on the image. The left side shows the
model PSF at one wavelength or the white (λ-averaged) PSF in log / linear scale, or its anisotropy (asinh image minus its azimuthal mean
about the model's optical axis, where the tomographic cross and the spider spikes stand out); the "RGB" colour map shows all wavelengths
as colours, as tools/observations.py (DisplayField). "Data & residual" adds the data, the residual (the asinh residual of the balanced
loss in log scale, model - data in linear scale) and the azimuthal profiles, with the balanced loss of STD_fitter.py as a score for
fitting by hand. Beyond the fitted inputs, the "Pointing" knobs move the science source off-axis for anisoplanatic PSFs (on the page
also by dragging it on a 2D pad of the field, with the LGS asterism and the NFM field drawn; double-click: back on-axis) and set the
telescope elevation (airmass: r0 along the line of sight, layer and LGS distances, differential refraction), the
"Cn2 profile" knobs set the weight (kept normalized to a sum of 1) and the altitude of every layer, the "LGS asterism" knobs its radius,
rotation and altitude, and the "PSD terms" checkboxes switch the PSD contributions on and off. A knob change only reruns the model stages it affects (STAGES), e.g. F and bg none, dx, dy and J only the OTF -> PSF step.
Note the TipTorch convention: dx shifts the PSF along the image rows, dy along the columns; the images show the rows (x) increasing upwards
and the columns (y) to the right, the pupil OPD too, as the source pad.

Needs the cached cubes and muse_df.pickle of the STD folder and the fits.
Standalone:
    python development/PSF_explorer.py [--port 8765] [--device cuda:0] [--sample ID]
starts a local web server (page in PSF_explorer.html) and prints its address; VS Code (Remote-SSH) forwards the port, otherwise
ssh -L 8765:localhost:8765 <host>. "Print values" prints the knob values in the page log and in the terminal.
VS Code interactive window / Jupyter: run the cells below (the last one shows the widgets), then
    explorer.values()                           # current knob values, e.g. to start a fit from them
or serve the page from there, with the values at hand: session, server = serve(block=False); session.values(); server.shutdown()
One model with 31 wavelengths at 199 x 199 pixels takes a few GB of GPU memory; the figure (~0.1-0.3 s) takes longer than the model.
"""
try:
    ipy = get_ipython()
    if ipy:
        ipy.run_line_magic('reload_ext', 'autoreload')
        ipy.run_line_magic('autoreload', '2')
except NameError:
    ipy = None

import io
import sys
import json
import time
import base64
import pickle
import argparse
import threading
import traceback
import contextlib
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

_here = Path(__file__).resolve().parent if '__file__' in dir() else Path.cwd().resolve()
ROOT  = next(p for p in (_here, *_here.parents) if (p / 'datasets').is_dir()) # repository root
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import ipywidgets as widgets
from IPython.display import display
from matplotlib.figure import Figure
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize, ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

from datasets.MUSE_STD_dataset.STD_dataset_utils import STD_FOLDER, LoadSTDStarCache
from tiptorch.PSF_models.NFM_wrapper import PSFModelNFM
from tiptorch.tools.utils import BalancedPSFLoss
from tiptorch._config import default_device, project_settings
from tools.plotting import _cube_to_RGB_array, wavelength_to_RGB

FIT_FOLDER  = Path(project_settings['MUSE_fitted_STD_folder'])
PAGE_FILE   = ROOT / 'development' / 'PSF_explorer.html' # the page of the standalone explorer
FIT_WVL_IDS = np.clip(np.arange(0, 32, 2), 0, 30) # wavelength bins of the fits (STD_fitter.py)
LOSS_CFG    = dict(radial_power=2.0, r_max=40.0, asinh_k=3.0, huber_delta=0.1, scale=3.0, match_background=False) # STD_fitter.PSF_LOSS_CFG
PSD_TERMS   = ['fitting', 'WFS noise', 'spatio-temporal', 'aliasing', 'chromatism', 'diff. refract']
FLAGS       = {'T': 'Tomography error', 'S': 'Has streaks', 'NP': 'Non-point', 'LA': 'Low AO errs', 'M': 'Medium quality'}
LO_LABELS   = ['bump [a.u.]', 'Z4 defocus', 'Z5 astig.', 'Z6 astig.', 'Z7 coma', 'Z8 coma', 'Z9 trefoil', 'Z10 trefoil'] # Zernikes in nm RMS
SCALES      = ['log', 'linear', 'anisotropy']
CMAPS       = ['magma', 'gray', 'viridis', 'inferno', 'RGB'] # 'RGB': the wavelengths as colours
# Colour settings of DisplayField in tools/observations.py, which also stretches the λ axis of the cube over 440-750 nm (no wavelengths
# given to _cube_to_RGB_array): with its true 480-930 nm, half of the NFM cube would be grey
RGB_KW      = dict(saturation=2.0, contrast=1.0, wb_shift=-0.1, mg_shift=0.075)
VIEW        = dict(wvl=15, white=False, scale='log', show_data=False, crop=40, decades=4.5, cmap='magma') # display options, defaults

# Slider ranges (min, max, step), widened at load time when a fitted value falls outside
RANGES = {
    'r0':                (0.03, 0.40, 0.001),  # [m] at 500 nm
    'L0':                (2.0,  60.0, 0.1),    # [m]
    'dn':                (0.0,  150., 0.5),    # HO WFSing error term
    'wind_speed_single': (0.0,  40.0, 0.1),    # [m/s], ground layer
    'wind_dir_single':   (0.0,  360., 1.0),    # [deg], ground layer
    'Cn2_weights':       (0.0,  1.0,  0.001),  # relative, kept normalized to a sum of 1
    'J_ctrl':            (0.0,  60.0, 0.1),    # [mas], TT jitter at the spline nodes
    'F_ctrl':            (0.5,  1.5,  0.001),
    'dx_ctrl':           (-3.0, 3.0,  0.01),   # [px], shifts the PSF along the image rows
    'dy_ctrl':           (-3.0, 3.0,  0.01),   # [px], along the columns
    'bg_ctrl':           (-2e-5, 2e-5, 1e-7),
    'LO_coefs':          (-200., 200., 0.5),
    'h_km':              (0.0,  25.0, 0.05),   # layer altitudes above the telescope [km], at zenith (the model applies the airmass)
    'LGS_radius':        (0.0,  20.0, 0.05),   # [arcsec]
    'LGS_rotation':      (-90., 90.,  0.5),    # [deg], added to the azimuths of the config
    'LGS_altitude':      (70.0, 110., 0.5),    # [km] at zenith
    'src_x':             (-15., 15.,  0.05),   # [arcsec], science source off-axis, in the frame of the LGS directions
    'src_y':             (-15., 15.,  0.05),
    'elevation':         (20.0, 90.0, 0.1),    # [deg], telescope elevation: airmass 1 / sin(elevation)
}
# Source pad of the page: x up, y right, as the images show the field. TipTorch's x is the image row axis for dx and the source / LGS
# directions alike (y the columns), and the images are drawn with the rows increasing upwards: an off-axis source elongates the PSF core
# towards its own direction on the pad, +dx moves the PSF up
FIELD_HALF = 15.0 # [arcsec], half-size of the pad
NFM_FOV    = 7.5  # [arcsec], side of the MUSE NFM field, drawn on the pad

GROUPS = [ # accordion sections: (title, knob keys)
    ('Pointing',           ['src_x', 'src_y', 'elevation']), # the source position also on a 2D pad on the page
    ('Atmosphere & AO',    ['r0', 'L0', 'dn', 'wind_speed_single', 'wind_dir_single']),
    ('Cn2 profile',        ['Cn2_weights', 'h_km']), # the layer weights, then their altitudes (the ground layer stays at 0)
    ('LGS asterism',       ['LGS_radius', 'LGS_rotation', 'LGS_altitude']),
    ('TT jitter',          ['J_ctrl']),
    ('NCPAs & phase bump', ['LO_coefs']),
    ('Flux & astrometry',  ['F_ctrl', 'dx_ctrl', 'dy_ctrl', 'bg_ctrl']),
]

# What a knob invalidates. A forward pass of the NFM model runs ComputePSD (controller, reconstruction filter, tomographic reconstructor,
# PSD terms), ComputeStaticOTF (NCPA phase -> OTF) and PSD2PSF (OTF of the PSD x static OTF x shift phasor x jitter kernel -> PSF,
# normalized, x F + bg); a knob reruns its own stage and the ones downstream only:
#   'PSD'     r0, L0, dn, the Cn2 profile and the LGS asterism enter the tomographic reconstructor W_tomo, the wind only the controller,
#             W_alpha and the spatio-temporal / aliasing terms (ComputePSD rebuilds W_tomo regardless, a few ms). The source position
#             enters the layer projection P_beta_L and the DM projector P_beta_DM (rebuilt by ComputePSD off-axis); the differential
#             refraction term also takes its direction from the source azimuth (x on-axis), not from the parallactic angle. The elevation
#             sets the airmass: r0 along the line of sight (r0_() scales the zenith r0 by airmass^-3/5), the layer and LGS distances
#             (the altitude knobs are at zenith) and the differential refraction (tan of the zenith angle). The optimal
#             DM projector P_opt, rebuilt by Update() only, would depend on the altitudes for off-axis optimization directions: the NFM
#             optimizes on-axis. The cone effect term (off for the NFM) would read the LGS and source zeniths from the config
#   'static'  LO_coefs (NCPAs, bump): the static OTF; the PSD is reused
#   'OTF'     J, dx, dy: the OTF products and the OTF -> PSF FFT; PSD and static OTF reused
#   'flux'    F, bg: nothing in the model, the cached PSF for F = 1, bg = 0 is scaled and offset
STAGES = {
    **dict.fromkeys(['r0', 'L0', 'dn', 'wind_speed_single', 'wind_dir_single', 'Cn2_weights', 'h_km', 'LGS_radius', 'LGS_rotation',
                     'LGS_altitude', 'src_x', 'src_y', 'elevation'], 'PSD'),
    'LO_coefs': 'static',
    'J_ctrl': 'OTF', 'dx_ctrl': 'OTF', 'dy_ctrl': 'OTF',
    'F_ctrl': 'flux', 'bg_ctrl': 'flux',
}
MODEL_STAGES = {'PSD', 'static', 'OTF'} # the ones run in TipTorch


def azimuthal_mean(img, r):
    """Azimuthal mean of every image [n, H, W] about its radius map r [n, H, W]: integer-radius bins interpolated linearly at the exact
    pixel radius (the bins alone would turn a steep radial profile into a fake cross: the pixels on the axes sit at the bin centres, the
    diagonal ones do not); empty bins (r = 0 when the axis is off the pixel centres) take their neighbours' values. Returns the map
    [n, H, W] and the profiles [n, N_bins]."""
    n_bins = int(np.rint(r.max())) + 1
    maps, profs = [], []
    for im, rr in zip(img, r):
        rbin   = np.rint(rr).astype(int).ravel()
        counts = np.bincount(rbin, minlength=n_bins)
        prof   = np.bincount(rbin, im.ravel(), minlength=n_bins) / np.maximum(counts, 1)
        prof   = np.interp(np.arange(n_bins), np.flatnonzero(counts), prof[counts > 0])
        maps.append(np.interp(rr.ravel(), np.arange(n_bins), prof).reshape(im.shape))
        profs.append(prof)
    return np.stack(maps), np.stack(profs)


class PSFSession:
    """Model, knob values and figure of the loaded sample, without any UI: the ipywidgets (PSFExplorer) and the page (serve) drive it."""
    def __init__(self, device=default_device, dpi=90):
        self.device   = torch.device(device)
        self.dpi      = dpi
        self.model    = None
        self.psd_on   = {t: t != 'aliasing' for t in PSD_TERMS}
        self._dirty   = set(MODEL_STAGES) # model stages to rerun (STAGES)
        self._changed = set()             # knobs to write into the model

        with open(STD_FOLDER / 'muse_df.pickle', 'rb') as f:
            df = pickle.load(f)
        df = df[(df['Corrupted'] == False) & (df['Bad quality'] == False)]
        self.df  = df.loc[[i for i in df.index if (FIT_FOLDER / f'{i}.pickle').exists()]]


    def samples(self, flag='all'):
        """[(label, id)] of the samples with this flag (a FLAGS value), or of all."""
        df = self.df if flag == 'all' else self.df[self.df[flag] == True]
        label = lambda i: f"{i}  " + ' '.join(k for k, col in FLAGS.items() if df.loc[i, col]) + ('  good' if df.loc[i, 'Good quality'] else '')
        return [(label(i), int(i)) for i in df.index]


    # ── Sample ────────────────────────────────────────────────────────────────
    def load(self, sample_id):
        """Loads a sample and its fit, returns a message for the log."""
        if sample_id not in self.df.index:
            raise KeyError(f'No fitted sample {sample_id}')

        t0 = time.time()

        if self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache()

        with open(FIT_FOLDER / f'{sample_id}.pickle', 'rb') as f:
            self.fit = pickle.load(f)
        # The regime of the fit (STD_fitter.DEROTATE_PSF): derotated cube with the config PupilAngle 0, or raw frame with the model rotated to
        # the header pupil angle (dx / dy in the raw image, LO coefficients in the rotated pupil frame)
        self.pupil_angle = float(np.ravel(self.fit['config']['telescope'].get('PupilAngle', 0.0))[0]) # [deg]
        PSF_data, _, _, configs = LoadSTDStarCache(ids=[sample_id], derotate_PSF=self.pupil_angle == 0, normalize=True, subtract_background=True,
                                                    wvl_ids=None, ensure_odd_pixels=True, device=self.device)
        # The atmosphere / guide-star / telescope / DM sections of the fit's own config (the Cn2 altitudes of the cache may have changed since,
        # the DM pitch carries the k_c of the fit, the telescope section its pupil angle), the wavelengths of the full cube
        for section in ('atmosphere', 'sources_HO', 'telescope', 'DM'):
            if section in self.fit['config']:
                configs[0][section] = {**configs[0][section], **{k: v for k, v in self.fit['config'][section].items() if k != 'PathPupil'}}

        with torch.no_grad():
            self.model = PSFModelNFM(configs, multiple_obs=True, LO_NCPAs=True, chrom_defocus=False, use_Moffat=False,
                                     Z_mode_max=9, N_spline_nodes=5, device=self.device)
        self.sample_id = sample_id
        self.data      = PSF_data[0].float()                                      # [N_λ, H, W]
        self.data_np   = self.data.cpu().numpy()
        self._aliasing_grids = False
        self.λ_nm      = self.model.λ_sim.flatten().cpu().numpy() * 1e9
        self.loss_fn   = BalancedPSFLoss(*self.data.shape[-2:], self.device, **LOSS_CFG)
        self.s_noise   = (LOSS_CFG['asinh_k'] * self.loss_fn.noise_sigma(self.data[None])[0]).cpu().numpy() # [N_λ, 1, 1], asinh scale

        # Fitted inputs -> model, knob values
        missing = []
        for name in self.model.inputs_manager.get_names(optimizable_only=False):
            if name in self.fit:
                cur = self.model.inputs_manager[name]
                val = torch.as_tensor(np.asarray(self.fit[name]), dtype=cur.dtype, device=cur.device)
                if val.numel() == cur.numel():
                    self.model.inputs_manager[name] = val.reshape(cur.shape)
                else:
                    missing.append(name)

        m   = self.model.model
        cfg = configs[0]
        self.LGS_az  = np.ravel(cfg['sources_HO']['Azimuth']).astype(float)
        self.fitted  = {key: self.model.inputs_manager[key].detach().flatten().cpu().numpy().astype(float).copy()
                        for _, keys in GROUPS for key in keys if key in self.model.inputs_manager.get_names(optimizable_only=False)}
        self.fitted.update({
            'h_km':         np.ravel(cfg['atmosphere']['Cn2Heights']).astype(float) / 1e3,
            'LGS_radius':   np.array([np.mean(np.ravel(cfg['sources_HO']['Zenith']))], dtype=float),
            'LGS_rotation': np.array([0.0]),
            'LGS_altitude': np.array([np.mean(np.ravel(cfg['sources_HO']['Height']))], dtype=float) / 1e3,
        })
        src_ζ, src_az = (float(np.ravel(cfg['sources_science'][k])[0]) for k in ('Zenith', 'Azimuth')) # [arcsec], [deg]
        self.fitted['src_x'] = np.array([src_ζ * np.cos(np.deg2rad(src_az))])
        self.fitted['src_y'] = np.array([src_ζ * np.sin(np.deg2rad(src_az))])
        self.fitted['elevation'] = np.array([90.0 - float(np.ravel(cfg['telescope']['ZenithAngle'])[0])])
        self.fitted['Cn2_weights'] /= self.fitted['Cn2_weights'].sum()

        self.state   = {k: v.copy() for k, v in self.fitted.items()} # knob values
        pitch, pixel = float(np.ravel(cfg['DM']['DmPitchs'])[0]), float(np.ravel(cfg['sensor_science']['PixelScale'])[0]) # [m], [mas]
        self.r_AO    = self.λ_nm * 1e-9 / (2 * pitch) * 206264.8e3 / pixel # AO cut-off radius [px]
        self.pupil   = m.pupil_rot.cpu().numpy() > 0 # at the pupil angle, as the OPD shown
        self._x0     = {k: v.detach().clone() for k, v in self.model.inputs_manager.to_dict().items()} # shapes and dtypes of the inputs
        # The splines are linear in their node values: x(λ) = x_ctrl @ basis, [N_ctrl, N_λ] (no spline evaluation per knob change)
        self._basis  = self.model.evaluate_splines(torch.eye(self.model.N_wvl_ctrl, device=self.device), self.model.λ_sim_normed).float().cpu().numpy()
        # Colour bar of the RGB images: as DisplayField, the λ axis of the cube is stretched over the visible 440-750 nm
        self._spectrum = ListedColormap([wavelength_to_RGB(λ, show_invisible=True) for λ in np.linspace(440, 750, len(self.λ_nm))])

        self.set_PSD_terms({})
        self._sync_model()
        self._simulate()
        self.loss_fit = self.loss_now

        fit_img = np.asarray(self.fit['Img. fit'])[0]
        repro   = np.abs(self.PSF[FIT_WVL_IDS] - fit_img).max() / fit_img.max() if fit_img.shape[0] == len(FIT_WVL_IDS) else np.nan
        regime  = 'derotated' if self.pupil_angle == 0 else f'raw frame, pupil angle {self.pupil_angle:.1f}°'
        kc      = float(self.fit.get('k_c fraction', 1.0))
        return (f'Sample {sample_id} loaded in {time.time() - t0:.1f} s ({regime}' + (f', k_c {100 * kc:.1f} % of the nominal' if kc != 1 else '')
                + f'); the fitted inputs reproduce the stored fit to {repro:.1e} of the peak' + (f'; not loaded (shape mismatch): {missing}' if missing else ''))


    def knobs(self):
        """Accordion sections of the loaded sample, [(title, [(key, idx, label, value, min, max, step)])]."""
        λ_nodes = (self.model.λ_ctrl.flatten().cpu().numpy() * (1e9 if self.model.λ_ctrl.max() < 1e-3 else 1)).round().astype(int)
        N_L     = len(self.state['h_km'])
        labels  = {
            'r0': ['r0 [m] @500 nm'], 'L0': ['L0 [m]'], 'dn': ['dn'], 'wind_speed_single': ['wind GL [m/s]'], 'wind_dir_single': ['wind dir GL [°]'],
            'Cn2_weights': [f'weight layer {k}' for k in range(N_L)],
            'h_km': [f'altitude layer {k} [km]' for k in range(N_L)],
            'LGS_radius': ['LGS radius ["]'], 'LGS_rotation': ['LGS rotation [°]'], 'LGS_altitude': ['LGS altitude [km]'],
            'src_x': ['source x ["]'], 'src_y': ['source y ["]'], 'elevation': ['telescope elevation [°]'],
            'LO_coefs': LO_LABELS,
            **{key: [f'{name} @{l} nm' for l in λ_nodes] for key, name in (('J_ctrl', 'J [mas]'), ('F_ctrl', 'F'), ('dx_ctrl', 'dx [px, rows]'),
                                                                           ('dy_ctrl', 'dy [px, cols]'), ('bg_ctrl', 'bg'))},
        }
        sections = []
        for title, keys in GROUPS:
            rows = []

            for key in keys:
                if key not in self.state:
                    continue

                lo, hi, step = RANGES[key]
                for idx, v in enumerate(self.state[key]):
                    if key == 'h_km' and idx == 0:
                        continue # the ground layer stays at 0

                    v = float(v)
                    v_lo, v_hi = (lo, hi) if key == 'Cn2_weights' else (min(lo, v - abs(v) * 0.2 - step), max(hi, v + abs(v) * 0.2 + step)) # weights stay in [0, 1]
                    rows.append((key, idx, labels[key][idx] if idx < len(labels[key]) else f'{key}[{idx}]', v, v_lo, v_hi, step))

            sections.append((title, rows))
        return sections


    def set_knob(self, key, idx, value):
        """Sets one knob value; returns the knobs whose other entries changed along (the Cn2 weights, kept normalized to a sum of 1)."""
        w = self.state[key]
        if not 0 <= idx < len(w):
            raise IndexError(f'{key}[{idx}]')

        coupled = set()
        if key == 'Cn2_weights': # the other layers share the rest in their current proportions (equally if they are all 0)
            value  = float(np.clip(value, 0.0, 1.0)) if len(w) > 1 else 1.0
            others = np.arange(len(w)) != idx
            rest   = w[others].sum()
            w[others] = w[others] * (1.0 - value) / rest if rest > 0 else (1.0 - value) / max(others.sum(), 1)
            coupled.add(key)

        w[idx] = float(value)
        self._changed.add(key)
        self._dirty.add(STAGES[key])
        return coupled


    def reset(self):
        self.state    = {k: v.copy() for k, v in self.fitted.items()}
        self._changed = set(self.state)
        self._dirty  |= MODEL_STAGES


    def values(self):
        """Current knob values, {name: array} (PSFModelNFM inputs in physical units, plus the tomography overrides)."""
        return {k: v.copy() for k, v in self.state.items()}


    def values_text(self):
        return '\n'.join(f'{k:18s} ' + np.array2string(v, precision=4, max_line_width=200) for k, v in self.state.items())


    # ── Model ─────────────────────────────────────────────────────────────────
    def set_PSD_terms(self, terms):
        """Switches PSD terms on / off, {term: bool}; they persist across samples."""
        self.psd_on.update({t: bool(on) for t, on in terms.items() if t in self.psd_on})
        if self.model is None:
            return

        m = self.model.model
        for term, on in self.psd_on.items():
            m.PSD_include[term] = on
        self._dirty.add('PSD')

        if self.psd_on['aliasing'] and not self._aliasing_grids:
            m.Update(grids=True) # the aliasing grids are only built when the term is on; Update also resets the inputs to the config
            self._aliasing_grids = True
            self._sync_model()


    @torch.no_grad()
    def _sync_model(self):
        """Writes every input into the TipTorch model through the wrapper's forward (the fixed ones included), then every knob: after
        loading a sample and after Update()."""
        self.model(self.model.inputs_manager.to_dict(), update_params=False)
        self._changed = set(self.state)
        self._dirty  |= MODEL_STAGES


    def _set_inputs(self, keys):
        """Writes these knobs into the TipTorch model, as the NFM wrapper's forward derives its inputs (splines of the λ nodes, J -> Jx = Jy,
        wind in the ground layer only), plus the tomography overrides."""
        m = self.model.model
        t = lambda v: torch.as_tensor(np.asarray(v, dtype=np.float32), device=self.device)
        zenith  = np.deg2rad(90.0 - self.state['elevation'][0])
        airmass = 1.0 / np.cos(zenith)
        x = {}
        if 'elevation' in keys: # the layer and LGS distances along the line of sight follow
            keys = set(keys) | {'h_km', 'LGS_altitude'}
        for key in keys:
            v = self.state[key]
            if STAGES[key] == 'flux':
                continue # applied to the PSF directly
            elif key == 'Cn2_weights':
                v = np.clip(v, 1e-6, None)
                x['Cn2_weights'] = t(v / v.sum()).view_as(self._x0[key])
            elif key.endswith('_ctrl'):
                x[key.removesuffix('_ctrl')] = t(v @ self._basis).view(1, -1) # [1, N_λ]
            elif key in ('wind_speed_single', 'wind_dir_single'):
                x[key.removesuffix('_single')] = torch.nn.functional.pad(t(v).view(1, 1), (0, m.N_L - 1))
            elif key == 'h_km':
                x['Cn2_heights'] = t(v * 1e3 * airmass).view(1, -1)
            elif key in ('LGS_radius', 'LGS_rotation'):
                az = np.deg2rad(self.LGS_az + self.state['LGS_rotation'][0])
                r  = np.tan(self.state['LGS_radius'][0] / 206264.8)
                x['GS_dirs_x'], x['GS_dirs_y'] = t(r * np.cos(az)).view(1, -1), t(r * np.sin(az)).view(1, -1)
            elif key == 'LGS_altitude':
                x['GS_height']  = t(np.full(len(self.LGS_az), v[0] * 1e3 * airmass)).view(1, -1)
                x['LGS_height'] = x['GS_height'].mean(dim=-1, keepdim=True)
            elif key == 'elevation':
                x['zenith_angle'], x['airmass'] = t(zenith).view_as(m.zenith_angle), t(airmass).view_as(m.airmass)
            elif key in ('src_x', 'src_y'): # as InitValues: tan(zenith) along the azimuth
                sx, sy = self.state['src_x'][0], self.state['src_y'][0]
                ζ, az  = np.hypot(sx, sy) / 206264.8, np.arctan2(sy, sx)
                x['src_dirs_x'], x['src_dirs_y'] = t([np.tan(ζ) * np.cos(az)]), t([np.tan(ζ) * np.sin(az)]) # [N_src]
            else:
                x[key] = t(v).view_as(self._x0[key]) # r0, L0, dn, LO_coefs

        if 'J' in x:
            x['Jx'] = x['Jy'] = x['J']
        for name, value in x.items():
            if hasattr(m, name):
                setattr(m, name, value)


    @torch.no_grad()
    def _simulate(self):
        """Model PSFs [N_λ, H, W] for the current knob values, rerunning only the stages the changed knobs affect, and the balanced loss."""
        m, dirty = self.model.model, self._dirty
        self._set_inputs(self._changed)

        if 'PSD' in dirty:
            self._PSD = m.ComputePSD()

        if 'static' in dirty:
            LO = torch.as_tensor(self.state['LO_coefs'], dtype=torch.float32, device=self.device).view_as(self._x0['LO_coefs'])
            self._OTF_static = m.ComputeStaticOTF(self.model.phase_func(LO, None))
            self.OPD = (self.model.LO_basis.rotate_OPD(self.model.OPD_func(LO))[0] * 1e9).cpu().numpy() # [nm], at the pupil angle

        if dirty & MODEL_STAGES:
            F, bg = m.F, m.bg
            m.F, m.bg = torch.ones_like(F), torch.zeros_like(bg)
            self._PSF_unit = m.PSD2PSF(self._PSD, self._OTF_static)[0].float() # [N_λ, H, W] for F = 1, bg = 0
            m.F, m.bg = F, bg

        F, bg = (torch.as_tensor(self.state[k] @ self._basis, dtype=torch.float32, device=self.device).view(-1, 1, 1) for k in ('F_ctrl', 'bg_ctrl'))
        PSF = self._PSF_unit * F + bg
        self.loss_λ   = self.loss_fn.pixel_map(self.data[None], PSF[None]).sum(dim=(-2, -1))[0].cpu().numpy() # [N_λ], mean = the loss
        self.loss_now = float(self.loss_λ.mean())
        self.PSF      = PSF.cpu().numpy()
        self.shift_rows, self.shift_cols = (self.state[k] @ self._basis for k in ('dx_ctrl', 'dy_ctrl'))
        self.stages_run = [s for s in ('PSD', 'static', 'OTF') if s in dirty] or ['flux']
        self._dirty, self._changed = set(), set()


    def _on_device(self):
        """Puts the default CUDA device on the model's one (_cube_to_RGB_array works on 'cuda')."""
        return torch.cuda.device(self.device) if self.device.type == 'cuda' else contextlib.nullcontext()


    # ── Display ───────────────────────────────────────────────────────────────
    def render(self, wvl=15, white=False, scale='log', show_data=False, crop=40, decades=4.5, cmap='magma'):
        """Figure for these display options (VIEW), recomputing the model PSFs if a knob changed: (PNG bytes, status line in HTML)."""
        if scale not in SCALES or cmap not in CMAPS:
            raise ValueError(f'scale {scale!r}, cmap {cmap!r}')

        t0, simulated = time.time(), bool(self._dirty or self._changed)
        if simulated:
            self._simulate()

        t_sim = time.time() - t0

        H, W = self.PSF.shape[-2:]
        wvl  = int(np.clip(wvl, 0, len(self.λ_nm) - 1))
        c, h = H // 2, int(np.clip(crop, 1, H // 2))
        sl   = np.s_[..., c-h:c+h+1, c-h:c+h+1]
        yy, xx = np.mgrid[-h:h+1, -h:h+1].astype(float)
        rgb  = cmap == 'RGB' and scale != 'anisotropy' # all wavelengths as colours (the anisotropy keeps its diverging map)
        ids  = slice(None) if white or rgb else [wvl]

        D = self.data_np[sl][ids]                                # [n, 2h+1, 2h+1]
        P = self.PSF[sl][ids]
        s = self.s_noise[ids]
        img   = {'model': P.mean(0), 'data': D.mean(0)}
        asinh = {'model': np.arcsinh(P / s), 'data': np.arcsinh(D / s)}
        # Radius about the optical axis of the model, i.e. its shift (dx along the rows, dy along the columns), for model and data alike:
        # about the pixel centre, the ~1 px offset of the PSF leaves a dipole over the whole halo in the anisotropy; the core centroid is
        # biased towards the pixel centre (undersampled core) and pulled by the coma
        r     = np.hypot(yy - self.shift_rows[ids][:, None, None], xx - self.shift_cols[ids][:, None, None]) # [n, 2h+1, 2h+1]
        aniso = lambda name: (asinh[name] - azimuthal_mean(asinh[name], r)[0]).mean(0)
        outer = np.hypot(yy, xx) >= 3 # colour limits of the anisotropy maps without the core

        fig = Figure(figsize=(13.5, 8.6 if show_data else 4.3))
        gs  = fig.add_gridspec(2 if show_data else 1, 3, left=0.05, right=0.95, top=0.93 if show_data else 0.9, bottom=0.08 if show_data else 0.15, wspace=0.35, hspace=0.25)
        extent = [-h - 0.5, h + 0.5, -h - 0.5, h + 0.5] # [px] from the centre, rows (x) upwards as on the source pad: origin='lower'
        cbar_ax = lambda ax: make_axes_locatable(ax).append_axes('right', size='4%', pad=0.05)

        def image(ax, im, title, norm, cmap, cbar_label=None, centred=True):
            cm = self._cmap(cmap)
            artist = ax.imshow(np.ma.masked_invalid(im), extent=extent if centred else None, origin='lower', norm=norm, cmap=cm, interpolation='nearest')
            ax.set_title(title, fontsize=10)
            fig.colorbar(artist, cax=cbar_ax(ax), label=cbar_label)

        peak = max(img['model'].max(), img['data'].max() if show_data else 0)
        panels = ['model', 'data'] if show_data else ['model']
        cubes  = {'model': P, 'data': D}
        if rgb: # as DisplayField: λ axis over 440-750 nm, upper limit 0.85 x the brightest pixel (common to model and data), `decades` below
            to_RGB = lambda cube, lo, hi: _cube_to_RGB_array(cube, None, lo, hi, scale, **RGB_KW)
            with self._on_device():
                hi = max(to_RGB(cubes[name], 1e-30, None)[2] for name in panels)
                lo = hi * 10**-decades if scale == 'log' else 0.0
                rgb_img = {name: to_RGB(cubes[name], lo, hi)[0] for name in panels}

        for k, name in enumerate(panels):
            ax = fig.add_subplot(gs[0, k])
            if scale == 'anisotropy':
                a   = aniso(name)
                lim = np.percentile(np.abs(a[outer]), 99)
                image(ax, a, f'{name}: anisotropy', Normalize(-lim, lim), 'RdBu_r', 'asinh - azimuthal mean')
            elif rgb:
                ax.imshow(rgb_img[name], extent=extent, origin='lower', interpolation='nearest')
                ax.set_title(f'{name}: RGB, {scale}', fontsize=10)
                fig.colorbar(ScalarMappable(Normalize(self.λ_nm[0], self.λ_nm[-1]), self._spectrum), cax=cbar_ax(ax), label='λ [nm]')
            elif scale == 'log':
                im = np.where(img[name] > 0, img[name], np.nan)
                image(ax, im, name, LogNorm(vmin=peak * 10**-decades, vmax=peak), cmap)
            else:
                image(ax, img[name], name, Normalize(vmin=0, vmax=peak), cmap)
            if k == 0:
                ax.set_xlabel('y = columns [px]', fontsize=8)
                ax.set_ylabel('x = rows [px]', fontsize=8)

        if show_data:
            ax = fig.add_subplot(gs[0, 2])
            if scale == 'linear':
                res, label = (P - D).mean(0) / D.max() * 100, '(model - data) / data peak [%]'
            elif scale == 'log':
                res, label = (asinh['model'] - asinh['data']).mean(0), 'asinh residual ≈ ln(model / data)'
            else:
                res, label = aniso('model') - aniso('data'), 'anisotropy model - data'
            lim = max(np.percentile(np.abs(res[outer]), 99), 1e-6)
            image(ax, res, 'residual', Normalize(-lim, lim), 'RdBu_r', label)

        # Azimuthal profiles (+ relative difference) and the static OPD
        ax = fig.add_subplot(gs[1, :2] if show_data else gs[0, 1])
        prof_m = azimuthal_mean(img['model'][None], r.mean(0, keepdims=True))[1][0, :h+1]
        ax.semilogy(np.clip(prof_m, peak * 1e-7, None), 'r', label='model')
        if show_data:
            prof_d = azimuthal_mean(img['data'][None], r.mean(0, keepdims=True))[1][0, :h+1]
            ax.semilogy(np.clip(prof_d, peak * 1e-7, None), 'k', label='data')
            ax2 = ax.twinx()
            ax2.plot((prof_m - prof_d) / np.abs(prof_d).clip(peak * 1e-7) * 100, color='tab:blue', lw=0.8, alpha=0.7)
            ax2.axhline(0, color='tab:blue', lw=0.5)
            ax2.set_ylim(-50, 50)
            ax2.set_ylabel('(model - data) / data [%]', color='tab:blue')
        for rr in (self.r_AO[[0, -1]] if white or rgb else self.r_AO[ids]):
            ax.axvline(rr, color='gray', ls='--', lw=0.8)
        ax.set_xlim(0, h)
        ax.set_xlabel('r [px]  (dashed: AO cut-off λ / 2 pitch)')
        ax.set_title('azimuthal profile', fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(alpha=0.3)

        ax = fig.add_subplot(gs[1, 2] if show_data else gs[0, 2])
        OPD = np.where(self.pupil, self.OPD, np.nan)
        lim = max(np.nanpercentile(np.abs(OPD), 99.5), 1.0)
        image(ax, OPD, 'static OPD: bump + NCPAs', Normalize(-lim, lim), 'RdBu_r', 'nm', centred=False)
        ax.set_xticks([]), ax.set_yticks([])

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=self.dpi)

        λ_text = 'RGB, all λ' if rgb else 'white' if white else f'{self.λ_nm[wvl]:.0f} nm'
        flags = ', '.join(col for col in [*FLAGS.values(), 'Good quality'] if self.df.loc[self.sample_id, col])
        peak_ratio = img['model'][h, h] / img['data'][h, h] if img['data'][h, h] > 0 else np.nan
        status = (f"<b>Sample {self.sample_id}</b> ({flags}) &nbsp;|&nbsp; {λ_text} &nbsp;|&nbsp; balanced loss: "
                  f"<b>{self.loss_now:.4f}</b> (at the fit {self.loss_fit:.4f}), this λ {self.loss_λ[ids].mean():.4f} &nbsp;|&nbsp; "
                  f"central pixel model / data {peak_ratio:.3f} &nbsp;|&nbsp; "
                  + (f"model [{' + '.join(self.stages_run)}] {t_sim:.2f} s, " if simulated else '') + f"render {time.time() - t0:.2f} s")
        return buf.getvalue(), status


    @staticmethod
    def _cmap(name):
        import matplotlib as mpl
        cmap = mpl.colormaps.get_cmap(name).copy()
        cmap.set_bad('black' if name != 'RdBu_r' else 'lightgray')
        return cmap


# ── ipywidgets front end ──────────────────────────────────────────────────────
class PSFExplorer:
    """The explorer as ipywidgets, for the VS Code interactive window or Jupyter."""
    def __init__(self, device=default_device, dpi=90):
        self.session = PSFSession(device, dpi)
        self._build_widgets()
        display(self.widget)


    def _build_widgets(self):
        W = widgets
        self.filter = W.Dropdown(options=['all'] + list(FLAGS.values()), value='all', description='Flag', layout=W.Layout(width='260px'))
        self.sample = W.Dropdown(description='Sample', layout=W.Layout(width='340px'))
        self.load_button  = W.Button(description='Load', button_style='primary', layout=W.Layout(width='80px'))
        self.reset_button = W.Button(description='Reset to fit', layout=W.Layout(width='110px'))
        self.print_button = W.Button(description='Print values', layout=W.Layout(width='110px'))
        self.filter.observe(lambda _: self._update_sample_list(), 'value')
        self.load_button.on_click(lambda _: self._guard(self.load, self.sample.value))
        self.reset_button.on_click(lambda _: self._guard(self.reset))
        self.print_button.on_click(lambda _: self._guard(self._print_values))
        self._update_sample_list()

        self.psd_boxes = {t: W.Checkbox(value=on, description=t, indent=False, layout=W.Layout(width='140px')) for t, on in self.session.psd_on.items()}
        for box in self.psd_boxes.values():
            box.observe(lambda _: self._guard(self._set_PSD_terms), 'value')
        self.knob_box = W.Accordion(layout=W.Layout(width='100%'))

        self.wvl        = W.IntSlider(min=0, max=30, value=VIEW['wvl'], description='λ', continuous_update=False, layout=W.Layout(width='330px'))
        self.white      = W.Checkbox(value=VIEW['white'], description='white', indent=False, layout=W.Layout(width='70px'))
        self.scale      = W.ToggleButtons(options=SCALES, value=VIEW['scale'], style={'button_width': '80px'})
        self.show_data  = W.ToggleButton(value=VIEW['show_data'], description='Data & residual', icon='eye', layout=W.Layout(width='150px'))
        self.crop       = W.IntSlider(min=5, max=99, value=VIEW['crop'], description='half-size', continuous_update=False, layout=W.Layout(width='300px'))
        self.decades    = W.FloatSlider(min=2.0, max=7.0, value=VIEW['decades'], step=0.25, description='log range', continuous_update=False, layout=W.Layout(width='300px'))
        self.cmap       = W.Dropdown(options=CMAPS, value=VIEW['cmap'], layout=W.Layout(width='100px'))
        for w in (self.wvl, self.white, self.scale, self.show_data, self.crop, self.decades, self.cmap):
            w.observe(lambda _: self._guard(self.render), 'value')

        self.image  = W.Image(format='png', layout=W.Layout(width='100%'))
        self.status = W.HTML()
        self.log    = W.Output(layout=W.Layout(max_height='160px', overflow='auto'))

        left = W.VBox([
            self.image,
            W.HBox([self.wvl, self.white, self.scale, self.show_data]),
            W.HBox([self.crop, self.decades, self.cmap]),
            self.status, self.log,
        ], layout=W.Layout(flex='1 1 auto', min_width='600px'))
        right = W.VBox([
            W.HBox([self.filter]), W.HBox([self.sample, self.load_button]),
            W.HBox([self.reset_button, self.print_button]),
            W.HTML('<b>PSD terms</b>'), W.GridBox(list(self.psd_boxes.values()), layout=W.Layout(grid_template_columns='repeat(3, 140px)')),
            self.knob_box,
        ], layout=W.Layout(width='470px', min_width='470px', max_height='900px', overflow_y='auto'))
        self.widget = W.HBox([left, right])


    def _update_sample_list(self):
        self.sample.options = self.session.samples(self.filter.value)


    def _guard(self, fn, *args):
        """Runs a callback with its errors printed in the widget (ipywidgets callbacks swallow them otherwise)."""
        try:
            return fn(*args)
        except Exception:
            with self.log:
                traceback.print_exc()


    def _knob(self, key, idx, label, value, lo, hi, step):
        slider = widgets.FloatSlider(value=value, min=lo, max=hi, step=step, continuous_update=False, readout=False, layout=widgets.Layout(width='190px'))
        text   = widgets.FloatText(value=value, step=step, layout=widgets.Layout(width='105px'))

        def set_value(v, source):
            if self._syncing:
                return
            self._syncing = True
            if source is text and not (slider.min <= v <= slider.max): # typed values may leave the slider range
                slider.min, slider.max = min(slider.min, v), max(slider.max, v)
            (text if source is slider else slider).value = v
            self._syncing = False
            self._show_values(self.session.set_knob(key, idx, v))
            self._guard(self.render)

        slider.observe(lambda ch: set_value(ch['new'], slider), 'value')
        text.observe(lambda ch: set_value(ch['new'], text), 'value')
        self._widgets[(key, idx)] = (slider, text)
        return widgets.HBox([widgets.Label(label, layout=widgets.Layout(width='130px')), slider, text], layout=widgets.Layout(margin='0 0 0 0'))


    def _show_values(self, keys):
        """Shows the session's values of these knobs (e.g. all Cn2 weights after one changed), without triggering their callbacks."""
        self._syncing = True
        for (key, idx), (slider, text) in self._widgets.items():
            if key in keys:
                v = float(self.session.state[key][idx])
                slider.min, slider.max = min(slider.min, v), max(slider.max, v)
                slider.value = text.value = v
        self._syncing = False


    def _build_knobs(self):
        self._widgets, self._syncing = {}, False
        sections = self.session.knobs()
        self.knob_box.children = [widgets.VBox([self._knob(*row) for row in rows]) for _, rows in sections]
        for i, (title, _) in enumerate(sections):
            self.knob_box.set_title(i, title)
        self.knob_box.selected_index = 0


    def load(self, sample_id):
        self.status.value = f'Loading sample {sample_id}...'
        self.log.clear_output()
        message = self.session.load(sample_id)
        self._build_knobs()
        self.wvl.max = len(self.session.λ_nm) - 1
        self.render()
        with self.log:
            print(message)


    def reset(self):
        self.session.reset()
        self._show_values(set(self.session.state))
        self.render()


    def values(self):
        """Current knob values, {name: array} (PSFModelNFM inputs in physical units, plus the tomography overrides)."""
        return self.session.values()


    def _print_values(self):
        with self.log:
            print(self.session.values_text())


    def _set_PSD_terms(self):
        self.session.set_PSD_terms({t: box.value for t, box in self.psd_boxes.items()})
        self.render()


    def render(self):
        if self.session.model is None:
            return
        self.image.value, self.status.value = self.session.render(
            wvl=self.wvl.value, white=self.white.value, scale=self.scale.value, show_data=self.show_data.value,
            crop=self.crop.value, decades=self.decades.value, cmap=self.cmap.value)
        self.wvl.description = f'λ {self.session.λ_nm[self.wvl.value]:.0f}'


# ── Standalone page ───────────────────────────────────────────────────────────
def _api(session, path, req):
    """One request of the page (POST /api/<name>, JSON): applies its changes to the session and returns what the page shows next.
    Every request carries the display options ('view'); the figure comes back whenever a sample is loaded, and 'values' holds the knobs
    the server changed (all after a reset, the Cn2 weights after one of them changed)."""
    out = {}
    if path == '/api/samples':
        return {'samples': session.samples(req.get('flag', 'all'))}
    if path == '/api/state': # page (re)opened: everything, including the sample loaded so far
        out.update(flags=['all', *FLAGS.values()], samples=session.samples(), psd=session.psd_on)
    elif path == '/api/load':
        out['log'] = session.load(int(req['id']))
    elif path == '/api/reset':
        session.reset()
    elif path == '/api/values':
        out['log'] = f'Sample {session.sample_id}:\n{session.values_text()}'
        print(out['log'], flush=True)
    elif path != '/api/update':
        raise KeyError(f'Unknown request {path}')

    changed = set(session.state) if path == '/api/reset' else set()
    for key, idx, value in req.get('knobs', []):
        changed |= session.set_knob(key, int(idx), float(value))
    if changed:
        out['values'] = {k: session.state[k].tolist() for k in changed}
    if 'psd' in req:
        session.set_PSD_terms(req['psd'])
    if session.model is not None:
        if path in ('/api/state', '/api/load'):
            out.update(sample=session.sample_id, knobs=session.knobs(), wvl_nm=session.λ_nm.round().astype(int).tolist(),
                       field=dict(half=FIELD_HALF, fov=NFM_FOV, lgs_az=session.LGS_az.tolist(), pupil_angle=session.pupil_angle)) # source pad geometry
        view = {k: type(VIEW[k])(v) for k, v in req.get('view', {}).items() if k in VIEW}
        png, out['status'] = session.render(**view)
        out['image'] = base64.b64encode(png).decode()
    return out


def serve(port=8765, device=default_device, dpi=90, sample=None, block=True):
    """Serves the explorer as a page on http://localhost:<port> (the next free port if taken), only to this machine: over Remote-SSH,
    VS Code forwards the port. block=False runs the server in a background thread and returns (session, server), e.g. for
    session.values() in the interactive window; server.shutdown() stops it."""
    session = PSFSession(device, dpi)
    if sample is not None:
        print(session.load(sample), flush=True)
    lock = threading.Lock() # one model, one request at a time

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): # no access log
            pass

        def _send(self, body, content_type, code=200):
            self.send_response(code)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.split('?')[0] in ('/', '/index.html'):
                self._send(PAGE_FILE.read_bytes(), 'text/html; charset=utf-8') # re-read: edits of the page show on reload
            else:
                self._send(b'Not found', 'text/plain', 404)

        def do_POST(self):
            try:
                req = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))) or b'{}')
                with lock:
                    out, code = _api(session, self.path, req), 200
            except Exception:
                out, code = {'error': traceback.format_exc()}, 500
                print(out['error'], file=sys.stderr, flush=True)
            self._send(json.dumps(out).encode(), 'application/json', code)

    for p in range(port, port + 20):
        try:
            server = ThreadingHTTPServer(('127.0.0.1', p), Handler)
            break
        except OSError:
            continue
    else:
        raise OSError(f'No free port in {port}-{port + 19}')
    server.daemon_threads = True
    print(f'PSF explorer: http://localhost:{p}  (Ctrl+C to stop)', flush=True)

    if not block:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return session, server
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description='PSF shape explorer for the MUSE NFM STD stars, as a page in the browser')
    parser.add_argument('--port',   type=int, default=8765, help='local port, the next free one if taken (default: 8765)')
    parser.add_argument('--device', default=str(default_device), help=f'torch device of the model (default: {default_device})')
    parser.add_argument('--sample', type=int, default=None, help='sample to load at start')
    parser.add_argument('--dpi',    type=int, default=90, help='resolution of the figure (default: 90)')
    serve(**vars(parser.parse_args()))


#%%
if __name__ == '__main__':
    if type(ipy).__name__ == 'ZMQInteractiveShell':
        explorer = PSFExplorer() # VS Code interactive window / Jupyter: the widgets below the cell
    else:
        main()                   # python development/PSF_explorer.py: the page in the browser

# %%
