"""
PSF parameters shared by the on-sky / TIPTOP / calibrated-TipTorch ETC comparison.

The same functions are applied to the TIPTOP PSFs (`data/Enrico_ETC/generated_random_PSFs/{wave}nm/*.fits`) and to the TipTorch
PSFs (`data/Enrico_ETC/tiptorch_random_PSFs/{wave}nm/*.fits`, written by `write_psf_file` in the same FITS layout), so the two
simulators are measured identically:

- `compute_fwhm`         Enrico's FWHM [mas] from a 5x oversampled radial profile around the peak.
- `compute_strehl_psfao` Enrico's Strehl from a MAOPPY PSFAO fit (`maoppy.psfmodel.Psfao`, MUSE NFM sampling at the PSF wavelength).
- `tiptop_metrics`       TIPTOP's own header metrics (`baseSimulation.computeMetrics`): P3 `getStrehl(method='max', psfInOnePix=True)`,
                         `getFWHM(method='contour')` and the encircled energy within 50 mas (`EE50`).

PSFs are 2D arrays [H, W] (100 x 100 px at 25 mas) normalized to unit sum before any measurement.
"""

import io
import sys
import contextlib
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.ndimage import zoom
from scipy.interpolate import interp1d

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tiptop_env  # noqa: F401
from p3.aoSystem import np as p3np  # cupy when P3 runs on the GPU, numpy otherwise
from p3.aoSystem.FourierUtils import getStrehl, getFWHM, getEncircledEnergy
from mastsel.mavisPsf import centeredPixelCoords
from tiptop.tiptopUtils import cpuArray

REPO_ROOT       = Path(__file__).resolve().parents[1]
PUPIL_PATH      = REPO_ROOT / 'data' / 'calibrations' / 'VLT_CALIBRATION' / 'VLT_PUPIL' / 'ut4pupil320.fits'
RAD2MAS         = 3600 * 180 * 1000 / np.pi
TEL_DIAMETER    = 8.0   # [m]
PIXEL_SCALE_MAS = 25.0  # MUSE NFM
EE_RADIUS_MAS   = 50.0  # TIPTOP default `eeRadiusInMas` (header key EE50)
PSF_SIZE        = 100   # [px], TIPTOP `sensor_science FieldOfView`


def normalize_psf(data):
    return data / np.nansum(data)


def compute_fwhm(image, oversampling=5, pixel_scale_mas=PIXEL_SCALE_MAS):  # OPTIMIZATION: reduced from 10 to 5 (4x faster)
    """ Enrico's FWHM [mas]: radius at which the radial profile around the peak of the oversampled image crosses half maximum """
    # Oversample the image
    image_oversampled = zoom(image, oversampling, order=1)  # OPTIMIZATION: order=1 instead of 3

    # Find the maximum value in the oversampled image
    max_value = np.nanmax(image_oversampled)

    # Find the coordinates of the maximum
    max_coords = np.unravel_index(np.nanargmax(image_oversampled), image_oversampled.shape)

    # Build a radial profile
    y, x = np.indices(image_oversampled.shape)
    r = np.sqrt((x - max_coords[1])**2 + (y - max_coords[0])**2)
    r = r.astype(int)

    tbin = np.bincount(r.ravel(), image_oversampled.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr

    # Half-maximum level
    half_max = max_value / 2.0

    # Radius at which the radial profile crosses the half-maximum level
    fwhm = 2 * np.interp(half_max, radialprofile[::-1], np.arange(len(radialprofile))[::-1])

    return fwhm / oversampling * pixel_scale_mas  # Convert from oversampled pixels to original pixels, then to mas


def compute_strehl_psfao(data, wave_nm):
    """ Enrico's Strehl: PSFAO (MAOPPY) fit of the PSF with the MUSE NFM instrument model at `wave_nm` """
    from maoppy.psfmodel import Psfao
    from maoppy.psffit import psffit
    from maoppy.instrument import muse_nfm

    wvl = wave_nm * 1e-9
    samp = muse_nfm.samp(wvl)

    r0 = 0.15 # Fried parameter [m]
    bck = 1e-5 # background [rad2 m2]
    amp = 5.0 # Moffat amplitude [rad2]
    alpha = 0.1 # Moffat alpha [1/m]
    ratio = 1.2 # alpha_x/alpha_y
    theta = 0 # rotation angle
    beta = 1.6 # Moffat beta power law, 1.6 good for MUSE NFM
    param = [r0, bck, amp, alpha, ratio, theta, beta]
    psfmodel = Psfao(data.shape, system=muse_nfm, samp=samp)

    with contextlib.redirect_stdout(io.StringIO()):  # psffit prints every iteration
        out = psffit(data, psfmodel, param)

    return float(psfmodel.strehlOTF(out.x)), out.x


def load_pupil(path=PUPIL_PATH):
    return fits.getdata(path).astype(float)


def tiptop_metrics(psf, wave_nm, pupil, pixel_scale_mas=PIXEL_SCALE_MAS, ee_radius_mas=EE_RADIUS_MAS, D=TEL_DIAMETER):
    """
    TIPTOP header metrics of one PSF, as in `tiptop.baseSimulation.computeMetrics`: (Strehl, FWHM [mas], EE within `ee_radius_mas`).
    `pupil` is the telescope pupil array used by TIPTOP (`telescope PathPupil`).
    """
    psf_xp   = p3np.asarray(np.array(psf, dtype=float))  # copy: getStrehl normalizes in place
    pupil_xp = p3np.asarray(pupil)
    samp = wave_nm * 1e-9 * RAD2MAS / (pixel_scale_mas * D)

    sr   = float(cpuArray(getStrehl(psf_xp, pupil_xp, samp, method='max', psfInOnePix=True)))
    fwhm = float(cpuArray(getFWHM(psf_xp, pixel_scale_mas, method='contour', nargout=1)))  # TIPTOP 1.4.3: no rebin
    ee_, rr_ = getEncircledEnergy(psf_xp, pixelscale=pixel_scale_mas, center=centeredPixelCoords(psf.shape[0]), nargout=2)
    ee_fn = interp1d(cpuArray(rr_), cpuArray(ee_), kind='cubic', bounds_error=False)
    ee = float(ee_fn(ee_radius_mas))
    return sr, fwhm, ee


def psf_parameters(psf, wave_nm, pupil, pixel_scale_mas=PIXEL_SCALE_MAS, do_psfao=True):
    """
    All PSF parameters of the comparison for one unit-sum PSF:
    `fwhm` (Enrico), `strehl` (PSFAO fit), `strehl_tt`, `fwhm_tt`, `ensquared_tt` (TIPTOP-style SR, FWHM, EE50; the on-sky
    column names are kept, `ensquared_tt` is TIPTOP's encircled energy within a 50 mas radius).
    """
    psf = normalize_psf(np.asarray(psf, dtype=float))
    sr_tt, fwhm_tt, ee_tt = tiptop_metrics(psf, wave_nm, pupil, pixel_scale_mas)
    out = {
        'fwhm':         compute_fwhm(psf, pixel_scale_mas=pixel_scale_mas),
        'strehl':       np.nan,
        'strehl_tt':    sr_tt,
        'fwhm_tt':      fwhm_tt,
        'ensquared_tt': ee_tt,
    }
    if do_psfao:
        try:
            out['strehl'], _ = compute_strehl_psfao(psf, wave_nm)
        except Exception as e:  # a failed fit must not stop the batch
            print(f'PSFAO fit failed: {type(e).__name__}: {e}')
    return out


# ---------------------------------------------------- FITS I/O ----------------------------------------------------
def _find_hdu(hdul, content):
    for hdu in hdul[1:]:
        if hdu.header.get('CONTENT', '').upper().startswith(content):
            return hdu
    return None


def read_psf_file(path):
    """
    Read a single-wavelength PSF file in TIPTOP's layout (`PSF CUBE` HDU with WL_NM / PIX_MAS / SR0000 / FWHM0000 / EE500000,
    `DIFFRACTION LIMITED PSF` HDU). Returns a dict with `psf` [H, W] (first source), `psf_dl`, `wave_nm`, `pix_mas` and the
    header metrics `sr_hdr`, `fwhm_hdr`, `ee_hdr`.
    """
    with fits.open(path) as hdul:
        cube_hdu = _find_hdu(hdul, 'PSF CUBE') or hdul[1]
        dl_hdu   = _find_hdu(hdul, 'DIFFRACTION LIMITED PSF')
        hdr  = cube_hdu.header
        data = np.asarray(cube_hdu.data, dtype=float)
        if data.ndim == 4:  # multi-wavelength TIPTOP cube [N_wvl, N_src, H, W]: not used here
            raise ValueError(f'{path}: multi-wavelength PSF cubes are not supported')
        psf = data[0] if data.ndim == 3 else data
        return {
            'psf':      psf,
            'psf_dl':   None if dl_hdu is None else np.asarray(dl_hdu.data, dtype=float),
            'wave_nm':  float(hdr['WL_NM']),
            'pix_mas':  float(hdr['PIX_MAS']),
            'sr_hdr':   hdr.get('SR0000', np.nan),
            'fwhm_hdr': hdr.get('FWHM0000', np.nan),
            'ee_hdr':   hdr.get(f'EE{int(EE_RADIUS_MAS)}0000', np.nan),
        }


def write_psf_file(path, psf, psf_dl, wave_nm, pix_mas, metrics, header_items=None, source='TipTorch'):
    """ Write one PSF (and its diffraction-limited PSF) in TIPTOP's single-wavelength FITS layout, with the TIPTOP-style metrics """
    primary = fits.PrimaryHDU()
    primary.header['SOURCE'] = source
    for key, value in (header_items or {}).items():
        primary.header[f'HIERARCH {key}'] = str(value)

    cube = fits.ImageHDU(data=np.asarray(psf, dtype=np.float64)[None])
    cube.header['CONTENT']  = 'PSF CUBE'
    cube.header['SIZE']     = str(cube.data.shape)
    cube.header['WL_NM']    = str(int(round(wave_nm)))
    cube.header['PIX_MAS']  = str(float(pix_mas))
    cube.header['SR0000']   = float(np.round(metrics['sr'], 5))
    cube.header['FWHM0000'] = float(np.round(metrics['fwhm'], 3))
    cube.header[f'EE{int(EE_RADIUS_MAS)}0000'] = float(np.round(metrics['ee'], 5))

    dl = fits.ImageHDU(data=np.asarray(psf_dl, dtype=np.float64))
    dl.header['CONTENT'] = 'DIFFRACTION LIMITED PSF'
    dl.header['SIZE']    = str(dl.data.shape)

    fits.HDUList([primary, cube, dl]).writeto(path, overwrite=True)


def crop_center(image, size=PSF_SIZE, center=None):
    """
    Central crop of a PSF stack [..., H, W] to `size` x `size`, keeping the peak pixel at index `size // 2` like TIPTOP's PSFs
    (`center` defaults to the array center H // 2, W // 2).
    """
    H, W = image.shape[-2:]
    cy, cx = (H // 2, W // 2) if center is None else center
    half = size // 2
    return image[..., cy - half:cy - half + size, cx - half:cx - half + size]
