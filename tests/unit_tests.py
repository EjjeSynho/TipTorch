
#%%
%reload_ext autoreload
%autoreload 2

import sys
sys.path.append('..')

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from project_settings import DATA_FOLDER, DATA_FOLDER
from tiptorch.psf_models.TipTorch import TipTorch
from tiptorch.managers.config_manager import ConfigManager
from project_settings import default_device, DATA_FOLDER, default_torch_type


#%%
config_manager = ConfigManager()
config = config_manager.Load(DATA_FOLDER / "parameter_files/muse_ltao.ini")
config = config_manager.Convert(config, framework='pytorch', device=default_device, dtype=default_torch_type)
config['sources_science']['Wavelength'] = torch.tensor([300e-9, 500e-9, 700e-9, 900e-9], dtype=default_torch_type, device=default_device)  # [m]
# config['sources_science']['Wavelength'] = torch.tensor([300e-9], dtype=default_torch_type, device=device)  # [m]

# Initialize TipTorch model
model = TipTorch(
    AO_config=config,
    AO_type='LTAO',
    norm_regime=None,
    device=default_device,
    oversampling=1
)

x_dict = {
    'dn': model.dn.clone() + 10,
    'Jx': model.Jx.clone() + 50,
    'Jy': model.Jy.clone() + 50,
}

# PSF_pred_small = model().clone() # First initial prediction of the "small" PSF
# plt.imshow(dd, origin='lower')
# plt.imshow(PSF_pred_small.squeeze().cpu().log().numpy().mean(axis=0), origin='lower')
# plt.colorbar()

# print(PSF_pred_small.sum(dim=(-2,-1)))


#%%
from tiptorch.tools.utils import mask_circle

PSF_size = 111

flux_core_radius = 2  # [pix]

with torch.no_grad():
    config['sensor_science']['FieldOfView'] = 511
    model.Update(config=config, grids=True, pupils=True, tomography=True)
    PSF_pred_big = model(x_dict).clone()
    
    config['sensor_science']['FieldOfView'] = PSF_size
    model.Update(config=config, grids=True, pupils=True, tomography=True)
    PSF_pred_small = model(x_dict).clone()
    
torch.cuda.empty_cache()

# plt.imshow(PSF_pred_small.squeeze().cpu().log().numpy().mean(axis=0), origin='lower')
# plt.colorbar()


crop_ratio_1 = (PSF_pred_big.amax(dim=(-2,-1)) / PSF_pred_small.amax(dim=(-2,-1))).squeeze()

c = PSF_pred_big.shape[-1] // 2
w = PSF_size // 2

# core_mask     = torch.tensor(mask_circle(PSF_size, flux_core_radius+1)[None,None,...], dtype=default_torch_type, device=device)
# core_mask_big = torch.tensor(mask_circle(PSF_pred_big.shape[-2], flux_core_radius+1)[None,None,...], dtype=default_torch_type, device=device)

# crop_ratio_2 = torch.where(
#     core_mask == 1,
#     PSF_pred_big[..., c-w:c+w+1, c-w:c+w+1] / PSF_pred_small,
#     torch.zeros_like(PSF_pred_small)
# ).sum(dim=(-2,-1)) / core_mask.sum(dim=(-2,-1)).squeeze()

# crop_ratio_3 = (PSF_pred_big[..., c-w:c+w+1, c-w:c+w+1] / PSF_pred_small).mean(dim=(-2,-1))

# print((crop_ratio_1/crop_ratio_2-1)*100, '% difference in crop ratio estimation by two methods')
# print((crop_ratio_1/crop_ratio_3-1)*100, '% difference in crop ratio estimation by two methods')

#%%
dd = (PSF_pred_big[..., c-w:c+w+1, c-w:c+w+1] / PSF_pred_small).mean(dim=(0,1)).cpu().numpy()

plt.imshow(dd, origin='lower')
# plt.imshow(PSF_pred_small.cpu().log().numpy().mean(axis=(0,1)), origin='lower', vmin=-11, vmax=-5.5)
plt.colorbar()

#%% ==================== PSD comparison: big vs small ====================

with torch.no_grad():
    config['sensor_science']['FieldOfView'] = 511
    model.Update(config=config, grids=True, pupils=True, tomography=True)
    model(x_dict)  # triggers ComputePSD and OTF internally
    PSD_big = model.PSD.clone()

    config['sensor_science']['FieldOfView'] = PSF_size
    model.Update(config=config, grids=True, pupils=True, tomography=True)
    model(x_dict)
    PSD_small = model.PSD.clone()

torch.cuda.empty_cache()

# Crop the big PSD to the size of the small PSD (center crop, same as for PSFs)
c_psd = PSD_big.shape[-1] // 2
w_psd = PSD_small.shape[-1] // 2

PSD_big_cropped = PSD_big[..., c_psd-w_psd:c_psd+w_psd+1, c_psd-w_psd:c_psd+w_psd+1]

# Ratio map (averaged over wavelengths)
PSD_ratio = (PSD_big_cropped / PSD_small).squeeze().mean(dim=0).abs().cpu().numpy()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

im0 = axes[0].imshow(PSD_big_cropped.mean(dim=(0,1)).abs().log().cpu().numpy(), origin='lower')
axes[0].set_title('PSD big (cropped), log')
fig.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(PSD_small.mean(dim=(0,1)).abs().log().cpu().numpy(), origin='lower')
axes[1].set_title('PSD small, log')
fig.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(np.log(PSD_ratio), origin='lower')
axes[2].set_title('log(PSD big cropped / PSD small)')
fig.colorbar(im2, ax=axes[2])

plt.tight_layout()

#%% ==================== PSF (no interp) comparison: big vs small ====================

with torch.no_grad():
    config['sensor_science']['FieldOfView'] = 511
    model.Update(config=config, grids=True, pupils=True, tomography=True)
    PSD_big = model.ComputePSD()
    OTF_static_big = model.ComputeStaticOTF()
    # Build the combined OTF manually (same as PSD2PSF but without the final OTF2PSF call)
    from tiptorch.tools.utils import pdims, min_2d
    cov_big = model._rfft2_to_full(torch.fft.fftshift(torch.fft.rfft2(torch.fft.ifftshift(PSD_big.abs(), dim=(-2,-1)), dim=(-2,-1)), dim=-2).real)
    SF_big = 2*(cov_big.abs().amax(dim=(-2,-1), keepdim=True) - cov_big).real
    OTF_turb_big = torch.exp(-0.5 * SF_big * pdims(2*torch.pi*1e-9/model.wvl, 2)**2)
    Jx  = pdims(min_2d(model.Jx.abs()),  2)
    Jy  = pdims(min_2d(model.Jy.abs()),  2)
    Jxy = pdims(min_2d(model.Jxy.abs()), 2)
    OTF_jitter_big = model.JitterKernel(Jx, Jy, Jxy)
    OTF_combined_big = OTF_turb_big * OTF_static_big * OTF_jitter_big
    OTF_combined_big = OTF_combined_big / pdims(OTF_combined_big.abs()[..., model.nOtf//2, model.nOtf//2], 2)
    PSF_noi_big = model.OTF2PSF_no_interp(OTF_combined_big)

    config['sensor_science']['FieldOfView'] = PSF_size
    model.Update(config=config, grids=True, pupils=True, tomography=True)
    PSD_small = model.ComputePSD()
    OTF_static_small = model.ComputeStaticOTF()
    cov_small = model._rfft2_to_full(torch.fft.fftshift(torch.fft.rfft2(torch.fft.ifftshift(PSD_small.abs(), dim=(-2,-1)), dim=(-2,-1)), dim=-2).real)
    SF_small = 2*(cov_small.abs().amax(dim=(-2,-1), keepdim=True) - cov_small).real
    OTF_turb_small = torch.exp(-0.5 * SF_small * pdims(2*torch.pi*1e-9/model.wvl, 2)**2)
    OTF_jitter_small = model.JitterKernel(Jx, Jy, Jxy)
    OTF_combined_small = OTF_turb_small * OTF_static_small * OTF_jitter_small
    OTF_combined_small = OTF_combined_small / pdims(OTF_combined_small.abs()[..., model.nOtf//2, model.nOtf//2], 2)
    PSF_noi_small = model.OTF2PSF_no_interp(OTF_combined_small)

torch.cuda.empty_cache()

# Crop big PSF to match the small one (center crop)
c_noi = PSF_noi_big.shape[-1] // 2
w_noi = PSF_noi_small.shape[-1] // 2
PSF_noi_big_cropped = PSF_noi_big[..., c_noi-w_noi:c_noi+w_noi+1, c_noi-w_noi:c_noi+w_noi+1]

PSF_noi_ratio = (PSF_noi_big_cropped / PSF_noi_small).mean(dim=(0,1)).cpu().numpy()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

im0 = axes[0].imshow(PSF_noi_big_cropped.mean(dim=(0,1)).log().cpu().numpy(), origin='lower')
axes[0].set_title('PSF big no_interp (cropped), log')
fig.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(PSF_noi_small.mean(dim=(0,1)).log().cpu().numpy(), origin='lower')
axes[1].set_title('PSF small no_interp, log')
fig.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(PSF_noi_ratio, origin='lower')
axes[2].set_title('PSF big cropped / PSF small (no interp)')
fig.colorbar(im2, ax=axes[2])

plt.tight_layout()

#%%
