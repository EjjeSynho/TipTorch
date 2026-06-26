#%%
# HUGS NGC 6441 processing follows the Omega Cen routine:
# load HST catalog, select sources around a MUSE NFM pointing, convert HST
# magnitudes to MUSE flux units, align HST/MUSE detections, and export CSV.
import sys
import glob
import json
import os
import pickle
from pathlib import Path

FORCE_CPU_CUBE_LOADING = False
if FORCE_CPU_CUBE_LOADING:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))
sys.path.append(str(SCRIPT_DIR.parent))

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.table import Table
from astropy.time import Time
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from matplotlib import colors
from scipy.spatial import cKDTree
from skimage.transform import AffineTransform
from tqdm import tqdm

from tiptorch._config import default_device, project_settings
from data_processing.MUSE_data_utils import LoadCachedDataMUSE
from MUSE_STD_dataset.STD_dataset_utils import FindDuplicatesFITS, MatchRawWithCubes
from tools.multisources import DetectSources

#%%
MUSE_DATA_FOLDER = Path(project_settings["MUSE_data_folder"])
data_folder = MUSE_DATA_FOLDER / "omega_cluster/NGC6441_data"
RAW_FOLDER = data_folder / "../raw_data"
CUBES_FOLDER = data_folder / "../reduced_cubes"
METADATA_FOLDER = data_folder / "../metadata"

CATALOG_METHOD = 1
MATCH_RADIUS = 30 * u.arcsec
USE_FLUX_WEIGHTED_ALIGNMENT = True
APPLY_HACKS_PROPER_MOTION = True
HACKS_PM_REF_EPOCH = 2009.1
HACKS_PM_MATCH_RADIUS = 0.10 * u.arcsec
PROPER_MOTION_CACHE_VERSION = 1

MUSE_units = u.Unit("1e-20 erg / (s cm2 Angstrom)")

FILTERS = ("F275W", "F336W", "F438W", "F606W", "F814W")
FILTER_BANDS = {
    "F275W": "WFC3,UVIS1,F275W",
    "F336W": "WFC3,UVIS1,F336W",
    "F438W": "WFC3,UVIS1,F438W",
    "F606W": "ACS,WFC1,F606W",
    "F814W": "ACS,WFC1,F814W",
}

#%%
def hugs_columns():
    columns = ["x_ref", "y_ref"]
    for filt in FILTERS:
        columns += [
            filt,
            f"{filt}_rms",
            f"{filt}_qfit",
            f"{filt}_sharp",
            f"{filt}_nfound",
            f"{filt}_ngood",
        ]
    columns += ["membership", "RA", "DEC", "ID", "iteration"]
    return columns


def load_hugs_catalog(method=1):
    catalog_path = data_folder / f"hlsp_hugs_hst_wfc3-uvis-acs-wfc_ngc6441_multi_v1_catalog-meth{method}.txt"
    df = pd.read_csv(
        catalog_path,
        comment="#",
        sep=r"\s+",
        names=hugs_columns(),
        engine="python",
    )
    df.set_index("ID", inplace=True)
    df[list(FILTERS)] = df[list(FILTERS)].replace(-99.9999, np.nan)

    df_astro = df[["RA", "DEC", "x_ref", "y_ref"]].rename(
        columns={"x_ref": "x", "y_ref": "y"}
    )
    df_mag = df[list(FILTERS)]
    return df_astro, df_mag


def exposure_epoch_from_header(header):
    if "MJD-OBS" in header:
        return Time(header["MJD-OBS"], format="mjd", scale="utc").jyear
    if "DATE-OBS" in header:
        return Time(header["DATE-OBS"], format="isot", scale="utc").jyear
    raise KeyError("Cube header has neither MJD-OBS nor DATE-OBS")


def load_hacks_pm_catalog():
    pm_path = data_folder / "hlsp_hacks_hst_acs-wfc3_ngc6441_multi_v1.0_pm-catalog.fits"
    if not pm_path.is_file():
        raise FileNotFoundError(f"HACKS proper-motion catalog not found: {pm_path}")

    pm = Table.read(str(pm_path)).to_pandas()
    return pm[[
        "ID",
        "RA",
        "DEC",
        "PMRAC",
        "EPMRAC",
        "PMDECC",
        "EPMDECC",
        "CHI2RX",
        "CHI2RY",
        "NF",
        "NU",
        "TIME",
        "CORRFLAG",
    ]]


def apply_hacks_proper_motion(
    df_astro,
    target_epoch_jyear,
    match_radius=HACKS_PM_MATCH_RADIUS,
    ref_epoch_jyear=HACKS_PM_REF_EPOCH,
):
    """
    Replace matched HUGS coordinates with HACKS coordinates propagated to target_epoch_jyear.

    HACKS PMRAC is d(RA*cos(DEC))/dt in mas/yr, so RA needs the cos(DEC)
    correction when converting to degrees.
    """
    pm = load_hacks_pm_catalog()
    df = df_astro.copy()

    hugs_coord = SkyCoord(df["RA"].to_numpy() * u.deg, df["DEC"].to_numpy() * u.deg)
    pm_coord = SkyCoord(pm["RA"].to_numpy() * u.deg, pm["DEC"].to_numpy() * u.deg)
    pm_idx, sep, _ = match_coordinates_sky(hugs_coord, pm_coord)
    matched = sep <= match_radius

    for col in [
        "HACKS_ID",
        "HACKS_RA_REF",
        "HACKS_DEC_REF",
        "PMRAC",
        "EPMRAC",
        "PMDECC",
        "EPMDECC",
        "PMA",
        "EPMA",
        "PMD",
        "EPMD",
        "PM_CHI2RX",
        "PM_CHI2RY",
        "PM_NF",
        "PM_NU",
        "PM_BASELINE",
        "PM_CORRFLAG",
        "PM_MATCH_SEP_MAS",
    ]:
        df[col] = np.nan

    if not np.any(matched):
        print(f"HACKS proper-motion match: 0 sources within {match_radius.to(u.arcsec):.3f}; coordinates unchanged")
        return df

    pm_match = pm.iloc[pm_idx[matched]].reset_index(drop=True)
    matched_pos = np.flatnonzero(matched)

    def set_matched(col, values):
        df.iloc[matched_pos, df.columns.get_loc(col)] = np.asarray(values)

    set_matched("HACKS_ID", pm_match["ID"].to_numpy())
    set_matched("HACKS_RA_REF", pm_match["RA"].to_numpy())
    set_matched("HACKS_DEC_REF", pm_match["DEC"].to_numpy())
    set_matched("PMRAC", pm_match["PMRAC"].to_numpy())
    set_matched("EPMRAC", pm_match["EPMRAC"].to_numpy())
    set_matched("PMDECC", pm_match["PMDECC"].to_numpy())
    set_matched("EPMDECC", pm_match["EPMDECC"].to_numpy())
    set_matched("PMA", pm_match["PMRAC"].to_numpy())
    set_matched("EPMA", pm_match["EPMRAC"].to_numpy())
    set_matched("PMD", pm_match["PMDECC"].to_numpy())
    set_matched("EPMD", pm_match["EPMDECC"].to_numpy())
    set_matched("PM_CHI2RX", pm_match["CHI2RX"].to_numpy())
    set_matched("PM_CHI2RY", pm_match["CHI2RY"].to_numpy())
    set_matched("PM_NF", pm_match["NF"].to_numpy())
    set_matched("PM_NU", pm_match["NU"].to_numpy())
    set_matched("PM_BASELINE", pm_match["TIME"].to_numpy())
    set_matched("PM_CORRFLAG", pm_match["CORRFLAG"].to_numpy())
    set_matched("PM_MATCH_SEP_MAS", sep[matched].to_value(u.mas))

    dt = target_epoch_jyear - ref_epoch_jyear
    dec_ref = pm_match["DEC"].to_numpy(dtype=float)
    cos_dec = np.cos(np.deg2rad(dec_ref))
    dra_deg = pm_match["PMRAC"].to_numpy(dtype=float) * dt / (1000.0 * 3600.0 * cos_dec)
    ddec_deg = pm_match["PMDECC"].to_numpy(dtype=float) * dt / (1000.0 * 3600.0)

    if "RA_HUGS" not in df.columns:
        df["RA_HUGS"] = df["RA"]
    if "DEC_HUGS" not in df.columns:
        df["DEC_HUGS"] = df["DEC"]
    set_matched("RA", pm_match["RA"].to_numpy(dtype=float) + dra_deg)
    set_matched("DEC", pm_match["DEC"].to_numpy(dtype=float) + ddec_deg)
    print(
        "HACKS proper-motion correction: "
        f"{matched.sum()} / {len(df)} HUGS sources matched within {match_radius.to(u.mas):.1f}; "
        f"propagated from J{ref_epoch_jyear:.1f} to J{target_epoch_jyear:.4f} "
        f"(dt={dt:+.3f} yr)"
    )
    return df

def select_reduced_cube(center_coord, max_sep=MATCH_RADIUS):
    matches, _ = MatchRawWithCubes(RAW_FOLDER, CUBES_FOLDER, verbose=False)
    rows = []
    for cube_name in matches["cube"]:
        cube_path = CUBES_FOLDER / cube_name
        with fits.open(cube_path, memmap=True) as hdul:
            header = hdul[0].header
            coord = SkyCoord(header["RA"] * u.deg, header["DEC"] * u.deg)
        rows.append((cube_name, coord.separation(center_coord)))

    if not rows:
        raise FileNotFoundError(f"No reduced MUSE cubes found in {CUBES_FOLDER}")

    cube_name, sep = min(rows, key=lambda item: item[1])
    if sep > max_sep:
        raise ValueError(
            f"Nearest MUSE cube is {cube_name}, {sep.to(u.arcmin):.2f} from NGC 6441. "
            f"Provide an NGC 6441 cube in {CUBES_FOLDER} or set reduced_cube explicitly."
        )
    return cube_name, matches.loc[matches["cube"] == cube_name, "raw"].values[0]


def compute_fluxes(df_mag):
    filters_config = data_folder / "HST_filters_folder.json"
    if not filters_config.is_file():
        filters_config = data_folder / "../OmegaCentaury_data/HST_filters_folder.json"

    with open(filters_config, "r") as f:
        os.environ["PYSYN_CDBS"] = json.load(f)["filters_folder"]

    from synphot import Observation, SourceSpectrum
    from synphot.models import ConstFlux1D
    import stsynphot as stsyn

    df_flux = pd.DataFrame(index=df_mag.index)
    filters_data = {}

    for filter_name in df_mag.columns:
        print(f"Processing targets for filter {filter_name}...", flush=True)
        bp = stsyn.band(FILTER_BANDS[filter_name])
        mag = df_mag[filter_name].to_numpy(dtype=float)
        sp0 = SourceSpectrum(ConstFlux1D, amplitude=0 * u.ABmag)
        flux0 = Observation(sp0, bp).effstim(MUSE_units).value * 1e20
        df_flux[filter_name] = flux0 * np.power(10.0, -0.4 * mag)
        filters_data[filter_name] = (bp.pivot().value / 10, bp.fwhm().value / 10)

    return df_flux, filters_data


def weighted_affine_fit_proximity(
    df_src: pd.DataFrame,
    df_dst: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    weight_col: str = "flux",
    max_distance: float = None,
    max_iters: int = 30,
    tol: float = 1e-7,
    exclude_brightest_src: int = 0,
    exclude_brightest_dst: int = 0,
) -> AffineTransform:
    """
    ICP-style weighted affine fit.

    Each iteration:
    1. Apply current transform to src points.
    2. Match via mutual nearest neighbours.
    3. Weight each pair by geometric-mean flux / (1 + distance).
    4. Re-solve the affine system from the original src coords to dst.
    5. Repeat until the matrix change drops below tol.
    """
    if exclude_brightest_src > 0:
        drop_idx = df_src[weight_col].nlargest(exclude_brightest_src).index
        df_src = df_src.drop(index=drop_idx)
    if exclude_brightest_dst > 0:
        drop_idx = df_dst[weight_col].nlargest(exclude_brightest_dst).index
        df_dst = df_dst.drop(index=drop_idx)

    src_coords = df_src[[x_col, y_col]].to_numpy().copy()
    dst_coords = df_dst[[x_col, y_col]].to_numpy()
    src_weights = df_src[weight_col].to_numpy().astype(float)
    dst_weights = df_dst[weight_col].to_numpy().astype(float)
    M = np.eye(3)

    for _ in range(max_iters):
        homog = np.column_stack([src_coords, np.ones(len(src_coords))])
        src_warped = (M @ homog.T).T[:, :2]

        tree_src = cKDTree(src_warped)
        tree_dst = cKDTree(dst_coords)
        dists_fwd, idxs_fwd = tree_src.query(dst_coords, k=1)
        _, idxs_bwd = tree_dst.query(src_warped, k=1)

        mutual = np.array([idxs_bwd[idxs_fwd[i]] == i for i in range(len(dst_coords))])
        if max_distance is not None:
            mutual &= dists_fwd <= max_distance
        if mutual.sum() < 3:
            if max_distance is not None:
                mutual = dists_fwd <= max_distance
            else:
                mutual = np.ones(len(dists_fwd), dtype=bool)
        if mutual.sum() < 3:
            raise ValueError(f"Need at least 3 matched pairs, got {mutual.sum()}")

        src_idx = idxs_fwd[mutual]
        src_pts = src_coords[src_idx]
        dst_pts = dst_coords[mutual]
        weights = np.sqrt(src_weights[src_idx] * dst_weights[mutual]) / (1.0 + dists_fwd[mutual])
        weights = np.clip(weights, 0, None)

        A = np.zeros((2 * len(src_pts), 6))
        b = np.zeros(2 * len(src_pts))
        for i, ((x, y), (u_dst, v_dst), weight) in enumerate(zip(src_pts, dst_pts, weights)):
            sw = np.sqrt(weight)
            A[2 * i] = sw * np.array([x, y, 1, 0, 0, 0])
            A[2 * i + 1] = sw * np.array([0, 0, 0, x, y, 1])
            b[2 * i] = sw * u_dst
            b[2 * i + 1] = sw * v_dst

        theta, *_ = np.linalg.lstsq(A, b, rcond=None)
        M_new = np.array([[theta[0], theta[1], theta[2]], [theta[3], theta[4], theta[5]], [0.0, 0.0, 1.0]])
        if np.max(np.abs(M_new - M)) < tol:
            M = M_new
            break
        M = M_new

    return AffineTransform(matrix=M)

#%%
FindDuplicatesFITS(CUBES_FOLDER)
df_astro, df_phot = load_hugs_catalog(CATALOG_METHOD)
cluster_center = SkyCoord(df_astro["RA"].median() * u.deg, df_astro["DEC"].median() * u.deg)

# Leave as None to auto-select a reduced cube whose pointing lies near NGC 6441.
reduced_cube = None

if reduced_cube is None:
    reduced_cube, raw_file = select_reduced_cube(cluster_center)
    print(f"Selected MUSE cube: {reduced_cube}", flush=True)
else:
    files_matches, _ = MatchRawWithCubes(RAW_FOLDER, CUBES_FOLDER, verbose=False)
    raw_file = files_matches.loc[files_matches["cube"] == reduced_cube, "raw"].values[0]

#%%
cube_path = CUBES_FOLDER / reduced_cube
raw_path = RAW_FOLDER / raw_file
cache_path = data_folder / f"../cached_cubes/{cube_path.stem}.pickle"

image_path = data_folder / "hlsp_hugs_hst_wfc3-uvis_ngc6441_f438w_v1_stack-0128s.fits"

with fits.open(image_path, memmap=True) as hdul:
    stacked_image = hdul[0].data
    wcs = WCS(hdul[0].header)

with fits.open(cube_path, memmap=True) as hdul:
    cube_header = hdul[0].header
    center_coord = SkyCoord(
        ra  = cube_header["RA"]  * u.deg,
        dec = cube_header["DEC"] * u.deg
    )
    muse_epoch_jyear = exposure_epoch_from_header(cube_header)

if APPLY_HACKS_PROPER_MOTION:
    df_astro = apply_hacks_proper_motion(df_astro, muse_epoch_jyear)

coords = SkyCoord(ra=df_astro["RA"].values * u.deg, dec=df_astro["DEC"].values * u.deg)
radius = (7 * np.sqrt(2) / 2) * u.arcsec * 2
mask = coords.separation(center_coord) < radius

df_sel = df_astro.iloc[np.nonzero(mask)[0]].copy()
if df_sel.empty:
    raise ValueError(f"No HUGS NGC 6441 sources found within {radius:.2f} of {cube_path.name}")

df_mag = df_phot.iloc[np.nonzero(mask)[0]].copy()
x_pix_sel, y_pix_sel = wcs.world_to_pixel(SkyCoord(ra=df_sel["RA"].values * u.deg, dec=df_sel["DEC"].values * u.deg))
x0, y0 = wcs.world_to_pixel(center_coord)

pixel_scale = np.mean(np.abs(proj_plane_pixel_scales(wcs))) * 3600.0
df_sel["x"] = (x_pix_sel - x0) * pixel_scale
df_sel["y"] = (y_pix_sel - y0) * pixel_scale
df_sel = df_sel.rename(columns={"x": "x, [asec]", "y": "y, [asec]"})

ROI_size_px = 280
cutout = Cutout2D(stacked_image, position=(x0, y0), size=(2 * ROI_size_px, 2 * ROI_size_px))
extent = [-ROI_size_px * pixel_scale, ROI_size_px * pixel_scale, -ROI_size_px * pixel_scale, ROI_size_px * pixel_scale]

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cutout.data + 15, origin="lower", extent=extent, norm=colors.LogNorm(vmin=10, vmax=500000)) #, cmap="Greys")
ax.scatter(df_sel["x, [asec]"], df_sel["y, [asec]"], s=40, facecolors="none", edgecolors="red", linewidths=0.5)
ax.scatter(0, 0, marker="o", color="green", s=80)
ax.set_xlabel("ΔRA [arcsec]")
ax.set_ylabel("ΔDEC [arcsec]")
ax.set_title(f"NGC 6441 sources near {cube_path.name}")
plt.tight_layout()
plt.show()


#%%
cache_file = METADATA_FOLDER / f"selected_srcs_{cube_path.stem}.pkl"
data_store = None
cache_needs_update = True

if cache_file.is_file():
    with open(cache_file, "rb") as f:
        data_store = pickle.load(f)
    cache_epoch = data_store.get("proper_motion_epoch_jyear", np.nan)
    cache_needs_update = (
        data_store.get("proper_motion_cache_version") != PROPER_MOTION_CACHE_VERSION
        or data_store.get("proper_motion_applied") != APPLY_HACKS_PROPER_MOTION
        or not np.isclose(float(cache_epoch), float(muse_epoch_jyear), rtol=0.0, atol=1e-6)
    )
    if cache_needs_update:
        print("Cached photometry predates this proper-motion solution; recomputing selection/flux cache...", flush=True)

if cache_needs_update:
    print("Computing photometry...\n")
    df_flux, filters_data = compute_fluxes(df_mag)
    data_store = {
        "Astrometry": df_sel,
        "AB magnitudes": df_mag,
        "Fluxes (MUSE units):": df_flux,
        "Filters data (pivot, FWHM) [nm]": filters_data,
        "proper_motion_applied": APPLY_HACKS_PROPER_MOTION,
        "proper_motion_epoch_jyear": muse_epoch_jyear,
        "proper_motion_reference_epoch_jyear": HACKS_PM_REF_EPOCH,
        "proper_motion_cache_version": PROPER_MOTION_CACHE_VERSION,
    }
    with open(cache_file, "wb") as f:
        pickle.dump(data_store, f)
    print(f"Data saved to {cache_file}")
else:
    print("Loading photometry from cache...\n")
    df_sel = data_store["Astrometry"]
    df_mag = data_store["AB magnitudes"]
    df_flux = data_store["Fluxes (MUSE units):"]
    filters_data = data_store["Filters data (pivot, FWHM) [nm]"]
    print(f"Data loaded from {cache_file}")
if len(df_sel) != len(df_mag) or len(df_flux) != len(df_mag):
    print("Cached photometry length mismatch; recomputing...", flush=True)
    df_mag = df_phot.iloc[np.nonzero(mask)[0]].copy()
    df_flux, filters_data = compute_fluxes(df_mag)
    data_store = {
        "Astrometry": df_sel,
        "AB magnitudes": df_mag,
        "Fluxes (MUSE units):": df_flux,
        "Filters data (pivot, FWHM) [nm]": filters_data,
        "proper_motion_applied": APPLY_HACKS_PROPER_MOTION,
        "proper_motion_epoch_jyear": muse_epoch_jyear,
        "proper_motion_reference_epoch_jyear": HACKS_PM_REF_EPOCH,
        "proper_motion_cache_version": PROPER_MOTION_CACHE_VERSION,
    }
    with open(cache_file, "wb") as f:
        pickle.dump(data_store, f)


#%%
spectral_cubes, spectral_info, TELEMETRY_CACHE, model_config = LoadCachedDataMUSE(
    raw_path, cube_path, cache_path, save_cache=True, device=default_device, verbose=True
)
cube_full, cube_sparse, valid_mask = spectral_cubes["cube_full"], spectral_cubes["cube_binned"], spectral_cubes["mask"]

sources = DetectSources(cube_sparse, nsigma=25, display=True, draw_box_size=21)
yy, xx = np.where(valid_mask.cpu().numpy().squeeze() > 0)
field_center = np.stack([xx.mean(), yy.mean()])[None, ...]
sources_coords = np.stack([sources["x_peak"].values, sources["y_peak"].values], axis=1)
sources_coords = (sources_coords - field_center) * 25 / 1000

#%%
df_MUSE_sources = pd.DataFrame({"x": sources_coords[:, 0], "y": sources_coords[:, 1]})
df_MUSE_sources["flux"] = sources["peak_value"]

df_HST_sources = df_sel.drop(columns=["RA", "DEC", "x_ref", "y_ref"], errors="ignore")
df_HST_sources = df_HST_sources.rename(columns={"x, [asec]": "x", "y, [asec]": "y"})
df_HST_sources["_flux_row"] = np.arange(len(df_HST_sources))
df_HST_sources["flux"] = df_flux.sum(axis=1).to_numpy()

dx_emp = -1.75
dy_emp = -0.25

df_HST_sources['x'] += dx_emp
df_HST_sources['y'] += dy_emp


flux_HST = df_HST_sources["flux"].values
flux_MUSE = df_MUSE_sources["flux"].values
n_compare = min(len(flux_HST), len(flux_MUSE))
flux_ratio_median = np.median(np.sort(flux_HST)[-n_compare:] / np.sort(flux_MUSE)[-n_compare:])

print(f"Affine fit uses flux weights: {USE_FLUX_WEIGHTED_ALIGNMENT}")

tform = weighted_affine_fit_proximity(
    df_HST_sources,
    df_MUSE_sources,
    max_distance=0.25
)

print("Affine matrix:\n", tform.params)

homog = np.vstack([df_HST_sources["x"], df_HST_sources["y"], np.ones(len(df_HST_sources))]).T
warped = (tform.params @ homog.T).T

df_HST_sources_filtered = df_HST_sources.copy()
df_HST_sources_filtered["x"], df_HST_sources_filtered["y"] = warped[:, 0], warped[:, 1]

x_hst = df_HST_sources_filtered["x"].to_numpy()
y_hst = df_HST_sources_filtered["y"].to_numpy()
x_muse = df_MUSE_sources["x"].to_numpy()
y_muse = df_MUSE_sources["y"].to_numpy()
flux_HST_norm = flux_HST / np.max(flux_HST) / flux_ratio_median
flux_MUSE_norm = flux_MUSE / np.max(flux_HST)

eps = -0.1
df_HST_sources_filtered = df_HST_sources_filtered[df_HST_sources_filtered["x"].between(-3.75 - eps, 3.75 + eps)]
df_HST_sources_filtered = df_HST_sources_filtered[df_HST_sources_filtered["y"].between(-3.75 - eps, 3.75 + eps)]
df_HST_sources_filtered = df_HST_sources_filtered.sort_values(by="flux", ascending=False)

x_hst_sel = df_HST_sources_filtered["x"].to_numpy()
y_hst_sel = df_HST_sources_filtered["y"].to_numpy()
flux_hst_norm_sel = df_HST_sources_filtered["flux"].to_numpy() / np.max(flux_HST) / flux_ratio_median

#%
plt.figure(figsize=(10, 10))
dot_scaler = 500

def norm_size(flux, max_flux, scale=dot_scaler):
    return scale * np.clip(flux / max_flux, 0, None) ** 0.8

max_flux_all = max(flux_HST_norm.max(), flux_MUSE_norm.max(), flux_hst_norm_sel.max())

plt.scatter(x_hst, y_hst, s=norm_size(flux_HST_norm, max_flux_all), facecolors="none", edgecolors="blue", linewidths=0.5, label="HST sources")
plt.scatter(x_hst_sel, y_hst_sel, s=norm_size(flux_hst_norm_sel, max_flux_all), facecolors="none", edgecolors="green", linewidths=0.7, label="HST sources (selected)")
plt.scatter(x_muse, y_muse, marker=".", color="red", s=norm_size(flux_MUSE_norm, max_flux_all), alpha=0.5, label="MUSE sources")
plt.plot([-3.75, -3.75, 3.75, 3.75, -3.75], [-3.75, 3.75, 3.75, -3.75, -3.75], color="black", linestyle="--", label="MUSE NFM FoV")

plt.axis("equal")
plt.xlabel("ΔRA [arcsec]")
plt.ylabel("ΔDEC [arcsec]")
plt.title("NGC 6441 HST sources aligned to MUSE detections")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

for filter_name in df_mag.columns:
    rows = df_HST_sources_filtered["_flux_row"].to_numpy(dtype=int)
    df_HST_sources_filtered[filter_name] = df_flux[filter_name].to_numpy()[rows]
df_HST_sources_filtered.drop(columns="_flux_row", inplace=True)

df_HST_sources_filtered.rename(
    columns={"flux": "flux (total, normalized)", "x": "x, [asec]", "y": "y, [asec]"},
    inplace=True,
)

csv_path = METADATA_FOLDER / f"HST_srcs_{cube_path.stem}.csv"
df_HST_sources_filtered.to_csv(csv_path, index=True, index_label="ID")
print("DataFrame saved successfully to", csv_path)

#%%
