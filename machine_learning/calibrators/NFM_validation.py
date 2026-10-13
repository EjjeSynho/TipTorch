"""
Validation helpers shared by the NFM calibrator scripts: PSF cubes of a parameter predictor on a data loader and the core / halo
metrics used to compare predictors (the loss value alone does not tell where the error is).
"""
import numpy as np
import torch
from torch.utils.data import Subset

from tools.plotting import plot_radial_PSF_profiles


@torch.no_grad()
def predict_cubes(trainer, predict, loader=None):
    """
    PSF cubes of a predictor on a loader (validation set by default), in the loader order of the samples.

    predict(tel_b, fit_b, idxs_b) -> x_dict of PSF-model parameters for the batch (e.g. the calibrator, or the fitted values).
    Returns (PSF_data [N, N_wvl, H, W], PSF_pred [N, N_wvl, H, W], dataset positions [N]) on the CPU.
    """
    loader = loader or trainer.val_loader
    trainer.calibrator.eval()

    N = len(loader.dataset)
    local_to_global = torch.tensor(loader.dataset.indices if isinstance(loader.dataset, Subset) else list(range(N)), dtype=torch.long)
    global_to_local = {int(g): l for l, g in enumerate(local_to_global.tolist())}

    PSF_pred = torch.zeros((N, trainer.N_wvl_total, trainer.H, trainer.W))
    PSF_data = torch.zeros_like(PSF_pred)

    for PSF_data_b, tel_b, fit_b, config_b, idxs_b in loader:
        lpos   = torch.tensor([global_to_local[int(i)] for i in idxs_b.cpu().tolist()], dtype=torch.long)
        x_dict = predict(tel_b, fit_b, idxs_b)

        for lambda_ids in trainer.lambda_id_sets:
            PSF_pred_b = trainer.run_model(x_dict, config_b, idxs_b, lambda_ids)
            for wi, li in enumerate(lambda_ids):
                PSF_pred[lpos, li] = PSF_pred_b[:, wi].cpu()
                PSF_data[lpos, li] = PSF_data_b[:, li].cpu()

    return PSF_data, PSF_pred, local_to_global


@torch.no_grad()
def core_halo_metrics(PSF_data, PSF_pred, trainer, wavelengths_nm, wvl_select=None, bands=None, chunk=8):
    """
    Validation metrics of a predictor, medians over the samples unless stated otherwise:
      balanced / legacy loss (means and medians of the per-sample losses, using the trainer's loss functions),
      peak ratio model / data (|error| median, 16-84 % range, median per selected wavelength),
      encircled-energy ratios within 2 and 8 px, radial-profile error in % of the peak (the "Max. err." of
      plot_radial_PSF_profiles) per selected wavelength, and the relative bias of the azimuthal profile per radial band.
    """
    N, C, H, W = PSF_data.shape
    wvl_select = wvl_select or [0, C // 2, C - 1]
    bands      = bands or [(0, 1.5), (1.5, 3.5), (3.5, 6.5), (6.5, 10.5), (10.5, 20.5), (20.5, 40.5), (40.5, 70.5)]
    lam_nm     = np.asarray(wavelengths_nm).round().astype(int)
    device     = trainer.device

    losses = {'balanced': [], 'legacy': []}
    for i in range(0, N, chunk):
        d, p = PSF_data[i:i+chunk].to(device), PSF_pred[i:i+chunk].to(device)
        losses['legacy'].append(trainer._legacy_PSF_loss(d, p).cpu())
        if trainer.balanced_loss is not None:
            losses['balanced'].append(trainer.balanced_loss(d, p).cpu())

    m = {}
    for name, vals in losses.items():
        if vals:
            vals = torch.cat(vals).numpy()
            m[f'{name}_loss_mean'], m[f'{name}_loss_median'] = float(vals.mean()), float(np.median(vals))

    D, P = PSF_data.numpy(), PSF_pred.numpy()
    yy, xx = np.mgrid[:H, :W]
    R  = np.hypot(yy - H // 2, xx - W // 2)
    c  = H // 2
    c3 = np.s_[..., c-3:c+4, c-3:c+4]

    peak = P[c3].max(axis=(-2, -1)) / D[c3].max(axis=(-2, -1)) # [N, C]
    m['peak_err_pct_median']  = float(np.median(np.abs(peak - 1)) * 100)
    m['peak_ratio_p16_p84']   = [float(x) for x in np.percentile(peak, [16, 84])]
    m['peak_ratio_median']    = {int(lam_nm[l]): float(np.median(peak[:, l])) for l in wvl_select}

    for rr in (2, 8):
        EE = P[..., R <= rr].sum(-1) / D[..., R <= rr].sum(-1)
        m[f'EE{rr}_ratio_median']   = float(np.median(EE))
        m[f'EE{rr}_err_pct_median'] = float(np.median(np.abs(EE - 1)) * 100)

    m['profile_max_err_pct'] = {}
    for l in wvl_select:
        _, _, p_err, _ = plot_radial_PSF_profiles(D[:, l], P[:, l], cutoff=40, return_profiles=True, suppress_plot=True)
        m['profile_max_err_pct'][int(lam_nm[l])] = float(np.median(p_err, axis=0).max())

    m['radial_rel_bias'] = {}
    for a, b in bands:
        mask = (R >= a) & (R < b)
        rel  = (P[..., mask].mean(-1) - D[..., mask].mean(-1)) / np.abs(D[..., mask].mean(-1)).clip(1e-12)
        m['radial_rel_bias'][f'{a:g}-{b:g}'] = [float(np.median(rel)), float(np.median(np.abs(rel)))]

    m['N'] = int(N)
    return m


def metrics_table(results: dict) -> str:
    """Side-by-side text table of core_halo_metrics results, {name: metrics}."""
    names = list(results)
    first = results[names[0]]
    rows  = [('', [n[:20] for n in names])]

    def add(label, f):
        rows.append((label, [f(results[n]) for n in names]))

    for key in ('balanced_loss_mean', 'legacy_loss_mean'):
        if key in first:
            add(key, lambda r: f"{r[key]:.4f}")
    add('|peak err| % median', lambda r: f"{r['peak_err_pct_median']:.1f}")
    add('peak ratio p16-p84',  lambda r: f"{r['peak_ratio_p16_p84'][0]:.2f}-{r['peak_ratio_p16_p84'][1]:.2f}")
    for l in first['peak_ratio_median']:
        add(f'peak ratio median @{l} nm', lambda r, l=l: f"{r['peak_ratio_median'][l]:.3f}")
    for rr in (2, 8):
        add(f'EE{rr} ratio / |err| %', lambda r, rr=rr: f"{r[f'EE{rr}_ratio_median']:.3f} / {r[f'EE{rr}_err_pct_median']:.1f}")
    for l in first['profile_max_err_pct']:
        add(f'profile max err % @{l} nm', lambda r, l=l: f"{r['profile_max_err_pct'][l]:.2f}")
    for b in first['radial_rel_bias']:
        add(f'rel. bias {b} px (med / |med|)', lambda r, b=b: f"{r['radial_rel_bias'][b][0]:+.3f} / {r['radial_rel_bias'][b][1]:.3f}")

    width = max(len(label) for label, _ in rows) + 2
    return '\n'.join(f'{label:{width}s}' + ''.join(f'{v:>22}' for v in vals) for label, vals in rows)
