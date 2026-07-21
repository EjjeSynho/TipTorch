#%%
try:
    ipy = get_ipython()        # NameError if not running under IPython
    if ipy:
        ipy.run_line_magic('reload_ext', 'autoreload')
        ipy.run_line_magic('autoreload', '2')
        import linecache
        ipy.events.register('post_execute', lambda: linecache.clearcache())
except NameError:
    pass

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tiptorch._config import default_device, project_settings
import matplotlib.pyplot as plt
from tools.observations import MUSEObservation

from pathlib import Path

# Define the location of your NFM data. It can be whenever. One option is to add it to the project config file 
MUSE_DATA_FOLDER = Path(project_settings["MUSE_data_folder"])

device = default_device

#%%
# Define the paths to the raw and reduced MUSE NFM cubes. The cached data cube will be generated based on them
data_folder = MUSE_DATA_FOLDER / 'quasars/' # change to your actual path with the MUSE NFM data

if not isinstance(data_folder, Path):
    data_folder = Path(data_folder)

raw_path   = data_folder / "J0259/MUSE.2024-12-05T03_15_37.598.fits.fz"
cube_path  = data_folder / "J0259/J0259-0901_all.fits"
cache_path = data_folder / "J0259/J0259-0901_all.pickle"

# cube_path  = data_folder / "J0144/J0144-5745.fits"
# cache_path = data_folder / "J0144/J0144-5745_cache.pickle"
# raw_path   = data_folder / "J0144/J0144_raw.114.fits.fz"


ob = MUSEObservation(raw_path, cube_path, cache_path, device=device)

ob.DetectSources(nsigma=35, threshold='auto')
# ob.AddSources([[100, 200]], weights=0.0)
# NOTE: If some sources must be filtered out, do it here by modifying the ob.sources_table before initializing the sources for simulation

#%%
ob.ExtractSources()
ob.DisplayField(draw_box_size=20)

#%%
from tiptorch._config import WEIGHTS_FOLDER

# ob.PlotSourceSpectra()
# ob.InitSimulation()
ob.InitSimulation(WEIGHTS_FOLDER / 'NFM_calibrator/J0259-0901_all_science_tuned_bundle.pth')
# ob.InitSimulation()

#%%
ob.FitPSFModel(fit=['astrometry'], repeat=1, max_iter=200)

#%%
# ── Science-domain calibrator fine-tuning ─────────────────────────────────────
# Tunes the calibrator NN weights directly on this science observation.
# PSF rendering goes through MUSEObservation (_simulate_sparse_from_calibrator,
# _loss_PSF, _loss_LO) so field geometry, source overlaps, and measured spectra
# are all preserved during training — exactly as in the normal fitting path.
# ─────────────────────────────────────────────────────────────────────────────
import gc
import torch
from copy import deepcopy
from datetime import datetime

# ── Hyper-parameters ──────────────────────────────────────────────────────────
TUNE_PARAMS    = ('F_ctrl', 'dn', 'J_ctrl', 'r0')
NUM_EPOCHS     = 1000
LR             = 2e-3   # higher LR to allow meaningful adaptation in ~100-200 steps
PATIENCE       = 30
WEIGHT_DECAY   = 1e-4
# Anchor on FROZEN output dims only — stops the network from silently shifting
# parameters it is not supposed to change. Keep small since frozen dims are few.
FROZEN_ANCHOR_W = 5e-5
# L2 penalty on NN weight displacement from initialization. Prevents catastrophic
# forgetting while still allowing the network to move noticeably.
WEIGHT_DRIFT_W  = 5e-5
GRAD_CLIP       = 1.0
# ─────────────────────────────────────────────────────────────────────────────

calibrator = ob.calibrator            # NFMCalibrator loaded by InitSimulation()
calibrator.net.train()

# Frozen reference: the initial calibrator prediction for this OB's telemetry
tel = calibrator.prepare_telemetry(ob.reduced_telemetry)   # [1, N_features]
with torch.no_grad():
    x_ref = calibrator.net(tel).detach()                   # [1, N_out]

# tuned_mask  — which output dims to freely optimise
# frozen_mask — which output dims must stay at x_ref (penalised, not hard-clamped)
tuned_mask  = ob._calibrator_output_mask(calibrator, TUNE_PARAMS).unsqueeze(0)  # [1, N_out]
frozen_mask = ~tuned_mask.squeeze(0)                                              # [N_out]

# Pre-compute fitting weights and source subset once — constant throughout fine-tuning
fit_weights = ob._compute_fitting_weights()
src_subset  = fit_weights.subset

# Snapshot of initial NN weights for drift regularization
ref_state = {name: p.detach().clone() for name, p in calibrator.net.named_parameters()}

optimizer = torch.optim.AdamW(calibrator.net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=max(1, PATIENCE // 3), min_lr=1e-8
)

best_loss  = float('inf')
best_state = deepcopy(calibrator.net.state_dict())
bad_epochs = 0
history    = []

for epoch in range(NUM_EPOCHS):
    optimizer.zero_grad(set_to_none=True)

    x_pred     = calibrator.net(tel)                          # [1, N_out]
    # Hard-clamp non-tuned dims to reference — prevents accidental changes there
    x_combined = torch.where(tuned_mask, x_pred, x_ref)      # [1, N_out]

    # ── Render the full field via MUSEObservation ──────────────────────────────
    # Calls ob.PSF_model internally; preserves off-axis positions and source overlaps
    rendered = ob._simulate_sparse_from_calibrator(x_combined, calibrator, src_subset)

    # ── Losses ────────────────────────────────────────────────────────────────
    data_loss = ob._loss_PSF(ob.cube_sparse, rendered, fit_weights)   # PSF shape fidelity
    lo_loss   = ob._loss_LO(fit_weights)                               # LO modes regularization

    # Anchor on FROZEN dims: penalises the raw network output (x_pred) for moving
    # the frozen outputs away from x_ref. This regularises the shared network weights
    # so they don't silently corrupt parameters we are not tuning.
    # NOTE: tuned dims are intentionally NOT penalised here — that was the bug in the
    #       previous version and was the main reason why fine-tuning had no effect.
    if frozen_mask.any():
        anchor_loss = (x_pred[:, frozen_mask] - x_ref[:, frozen_mask]).pow(2).mean()
    else:
        anchor_loss = x_pred.new_zeros(())

    # Weight drift: soft L2 on NN weight displacement from initialization
    drift_loss = sum(
        (p - ref_state[n]).pow(2).mean() for n, p in calibrator.net.named_parameters()
    ) / max(len(ref_state), 1)

    total_loss = data_loss + lo_loss + FROZEN_ANCHOR_W * anchor_loss + WEIGHT_DRIFT_W * drift_loss
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(calibrator.net.parameters(), max_norm=GRAD_CLIP)
    optimizer.step()

    # Keep PSF model inputs in sync so _loss_LO reads up-to-date values next iteration
    with torch.no_grad():
        ob.PSF_model.update_manager_params(
            ob._calibrator_prediction_to_PSF_inputs(calibrator, x_combined.detach())
        )

    total_val  = float(total_loss.detach().cpu())
    data_val   = float(data_loss.detach().cpu())
    history.append(dict(
        epoch  = epoch + 1,
        total  = total_val,
        data   = data_val,
        anchor = float(anchor_loss.detach().cpu()),
        drift  = float(drift_loss.detach().cpu()),
        lr     = optimizer.param_groups[0]['lr'],
    ))
    scheduler.step(total_val)

    is_best = total_val < best_loss
    if is_best:
        best_loss  = total_val
        best_state = deepcopy(calibrator.net.state_dict())
        bad_epochs = 0
    else:
        bad_epochs += 1

    print(
        f"[{epoch+1:03d}/{NUM_EPOCHS}]  total={total_val:.4f}  data={data_val:.4f}  "
        f"frozen_anchor={history[-1]['anchor']:.4f}  lr={optimizer.param_groups[0]['lr']:.1e}"
        + ("  *" if is_best else "")
    )

    del rendered, x_pred, x_combined, total_loss, data_loss, lo_loss, anchor_loss, drift_loss
    if bad_epochs >= PATIENCE:
        print(f"Early stopping at epoch {epoch + 1}.")
        break

# Restore best checkpoint, switch to eval, and re-sync the PSF model
calibrator.net.load_state_dict(best_state)
calibrator.net.eval()

with torch.no_grad():
    x_best     = calibrator.net(tel)
    x_best     = torch.where(tuned_mask, x_best, x_ref)
    ob.PSF_model.update_manager_params(ob._calibrator_prediction_to_PSF_inputs(calibrator, x_best))
    ob._update_flux_norm()

gc.collect()
torch.cuda.empty_cache()
print(f"\nFine-tuning complete.  Best total loss: {best_loss:.6f}")

#%%
# Save the science-tuned calibrator as a standalone bundle
save_path = WEIGHTS_FOLDER / 'NFM_calibrator' / f'{Path(ob.cube_path).stem}_science_tuned_bundle.pth'
calibrator.save(
    save_path,
    source       = 'MUSE_quasar.py/science_finetune',
    tuned_params = list(TUNE_PARAMS),
    num_epochs   = len(history),
    best_loss    = best_loss,
    cube_path    = str(ob.cube_path),
    completed_at = datetime.now().isoformat(timespec='seconds'),
)
print(f"Science-tuned calibrator bundle saved to:\n  {save_path}")

# Training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
epochs_ = [h['epoch'] for h in history]
axes[0].plot(epochs_, [h['total'] for h in history], label='total loss')
axes[0].plot(epochs_, [h['data']  for h in history], label='data loss')
axes[0].set(title='Fine-tune losses', xlabel='Epoch', ylabel='Loss')
axes[0].legend(); axes[0].grid(True)
axes[1].plot(epochs_, [h['anchor'] for h in history], label='prediction anchor')
axes[1].plot(epochs_, [h['drift']  for h in history], label='weight drift')
axes[1].set(title='Regularization terms', xlabel='Epoch', ylabel='Loss')
axes[1].legend(); axes[1].grid(True)
plt.tight_layout()
plt.show()

#%%
ob.FitPSFModel(repeat=3, max_iter=200)
# ob.FitPSFModel(fit=['astrometry'], repeat=3, max_iter=200)
# ob.FitPSFModel(fit=['astrometry', 'photometry'], repeat=3, max_iter=200)

#%%
model = ob.SimulateField()
ob.DisplaySimulation(plot_profiles=True)

#%%
Strehls_per_λ = ob.PSF_model.ComputeStrehl()
plt.title('Strehl ratio vs. λ (for the 1st source)')
plt.plot(ob.λ_sparse, 100.0 * Strehls_per_λ.flatten().cpu())
plt.ylabel('Strehl ratio, [%]')
plt.xlabel('Wavelength, [nm]')
plt.grid()
plt.show()

#%%
_ = ob.SimulateField(full_spectrum=True)
# ob.DisplaySimulation(plot_profiles=True, plot_full_spectrum=True)

#%%
# ob.PlotSourceSpectra(title='Sources spectra (residual)', show_sparse=False, plot_residual=True, smooth_kernel=15)
# %% ===========================================================

cube_path  = data_folder / "J0144/J0144-5745.fits"
cache_path = data_folder / "J0144/J0144-5745_cache.pickle"
raw_path   = data_folder / "J0144/J0144_raw.114.fits.fz"

ob = MUSEObservation(raw_path, cube_path, cache_path, device=device)

ob.DetectSources(nsigma=35, threshold='auto')
# ob.AddSources([[100, 200]], weights=0.0)
# NOTE: If some sources must be filtered out, do it here by modifying the ob.sources_table before initializing the sources for simulation

#%%
ob.ExtractSources()
ob.DisplayField(draw_box_size=20)

#%%
from tiptorch._config import WEIGHTS_FOLDER

# ob.PlotSourceSpectra()

# ob.InitSimulation(WEIGHTS_FOLDER / 'NFM_calibrator/J0259-0901_all_science_tuned_bundle.pth')
ob.InitSimulation()

#%%
ob.FitPSFModel(fit=['astrometry'], repeat=1, max_iter=200)

#%%
ob.FitPSFModel(repeat=3, max_iter=200)

#%%
model = ob.SimulateField()
ob.DisplaySimulation(plot_profiles=True)


#%%
from tools.observations import PlotSourcesProfiles

PlotSourcesProfiles(ob.cube_sparse, ob.simulated_sparse, ob.sources.table, radius=16, title='Source radial profiles (sparse spectrum)', show=False)

plt.savefig(data_folder / f"plots/profile_predicted_{Path(cache_path).stem}.pdf", dpi=300)

# %%
from tools.plotting import PlotSpetralCubeInRGB

color_kwargs = {
    'saturation':  1.5,
    'contrast'  :  0.8,
    'wb_shift'  :  0.05,
    'mg_shift'  :  0.0,
    'min_val'   :  10,
    'max_val'   :  3e2,
    'show'      :  False,
    'fig_size'  :  (4, 4)
}


residual_sparse = (ob.cube_sparse - ob.simulated_sparse).abs()

_ = PlotSpetralCubeInRGB(
    ob.cube_sparse.cpu().numpy()[ob.ROI_plot],
    title = "Data",
    **color_kwargs
)

plt.savefig(data_folder / f"plots/data_{Path(cache_path).stem}.pdf", dpi=300)

_ = PlotSpetralCubeInRGB(
    ob.simulated_sparse.cpu().numpy()[ob.ROI_plot],
    title = "Model",
    **color_kwargs
)

plt.savefig(data_folder / f"plots/model_{Path(cache_path).stem}_no-tuned.pdf", dpi=300)

# _ = PlotSpetralCubeInRGB(
#     residual_sparse.cpu().numpy()[ob.ROI_plot],
#     title = "Residual",
#     **color_kwargs
# )

# %%
