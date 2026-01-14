#%%
%reload_ext autoreload
%autoreload 2

import sys

sys.path.insert(0, '..')

import math
import numpy as np
import torch
import matplotlib.pyplot as plt

import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS

# Optional: for corner plotting
try:
    import corner
    _HAVE_CORNER = True
except Exception:
    _HAVE_CORNER = False


from tools.plotting import plot_radial_PSF_profiles, draw_PSF_stack
from data_processing.SPHERE_STD_dataset_utils import LoadSTDStarData
from PSF_models.IRDIS_wrapper import PSFModelIRDIS

from project_settings import device

pyro.set_rng_seed(0)
pyro.enable_validation(False)

torch_dtype = torch.float32
torch.set_default_dtype(torch_dtype)

# For Sequential MCMC (NUTS), we only need 1 simultaneous model evaluation.
# Large batch sizes are only useful for vectorized samplers.
BATCH_SIZE = 1 

# torch.amp.autocast('cuda', dtype=torch.float16 or torch.bfloat16)
# torch.backends.cuda.matmul.allow_tf32 = True

#%%
sample_id = 1778

print(f"Loading sample {sample_id}...")
PSF_data, data_sample, configs = LoadSTDStarData(
    sample_id,
    normalize=True,
    subtract_background=True,
    ensure_odd_pixels=True,
    device=device
)

PSF_0    = PSF_data[0]['PSF (mean)']
PSF_mask = PSF_data[0]['mask (mean)']
del PSF_data

# Move mask to boolean for easier indexing
mask_bool = PSF_mask.bool()
target_pixels = PSF_0[mask_bool] # (N_valid_pix,)
N_valid_pix = target_pixels.numel()
target_valid = target_pixels.unsqueeze(0) # Target: (1, N_valid_pix)

#%%
LWE_flag = True
fit_wind = True
LO_NCPAs = False
use_Zernike = True

PSF_model = PSFModelIRDIS(
    configs,
    multiple_obs=True,
    LWE_flag=LWE_flag,
    fit_wind=fit_wind,
    LO_NCPAs=LO_NCPAs,
    use_Zernike=use_Zernike,
    N_modes=9,
    LO_map_size=31,
    device=device
)

PSF_model.inputs_manager.to_float()
PSF_model.model.to_float()

#%%
manager = PSF_model.inputs_manager
# manager.set_optimizable('bg', False)
# manager.set_optimizable('Jxy', False)

print(manager)

# 3. Configure Parameter Space using InputsManager
param_names = manager.get_names(optimizable_only=True, flattened=True)
param_names.append('log_sigma') # Add Noise parameter (handled externally)

print(f"Model parameters stacked vector size: {manager.get_stacked_size()}")
print(f"UltraNest Parameters ({len(param_names)}): {param_names}")


#%% --------------------------------------------------------------------------
span = 2.0
logsig_low, logsig_high = -8.0, 0.0

ndim = len(param_names)  # includes log_sigma

# -------------------------
# Helper: unconstrained -> bounded theta (vectorized)
# z in R^ndim -> u in (0,1) -> theta in bounds
# -------------------------
def z_to_theta(z: torch.Tensor) -> torch.Tensor:
    """
    z: (B, ndim) unconstrained
    returns theta: (B, ndim) in the same parameterization as ultranest loglike:
      theta[:, :-1] in [-span, span]
      theta[:,  -1] in [logsig_low, logsig_high]
    """
    u = torch.sigmoid(z)  # (0,1)
    theta = torch.empty_like(u)

    theta[:, :-1] = (-span) + (2.0 * span) * u[:, :-1]
    theta[:, -1]  = logsig_low + (logsig_high - logsig_low) * u[:, -1]
    return theta

# Torch PSF batch generator
def compute_PSF_grad(params_latent_batch: torch.Tensor):
    """
    params_latent_batch: (BATCH_SIZE, n_model_params) in manager latent space ([-span,span])
    returns PSF_batch: (BATCH_SIZE, H, W) or raises RuntimeError
    """
    x_dict = manager.unstack(params_latent_batch, include_all=True)
    PSF_batch = PSF_model(x_dict)  # must be differentiable
    return PSF_batch


# -------------------------
# Differentiable log-likelihood in torch
# Matches your gaussian pixel model on masked pixels.
# Works for input theta of shape (B, ndim).
# -------------------------
def loglike_torch(theta: torch.Tensor) -> torch.Tensor:
    """
    theta: (B, ndim), float32 on GPU
    returns logL: (B,), float32 on GPU
    """
    assert theta.ndim == 2 and theta.shape[1] == ndim
    B = theta.shape[0]
    n_model_params = ndim - 1

    params_latent = theta[:, :n_model_params]            # (B, n_model_params), in [-span,span]
    log_sigma     = theta[:,  n_model_params]            # (B,)

    # Forward model
    PSF_batch_padded = compute_PSF_grad(params_latent.float())  # (BATCH_SIZE, H, W)
    PSF_batch = PSF_batch_padded[:B]

    # Select valid pixels
    pred_valid = PSF_batch.reshape(B, -1)[:, mask_bool.view(-1)]  # (B, N_valid)
    sigma = torch.exp(log_sigma).unsqueeze(1).clamp_min(1e-6)     # (B, 1)

    resid = target_valid.to(device) - pred_valid                  # (B, N_valid)
    chi2 = (resid / sigma).pow(2).sum(dim=1)                      # (B,)

    log_norm = N_valid_pix * (math.log(2.0 * math.pi) + 2.0 * log_sigma)  # (B,)
    logL = -0.5 * (chi2 + log_norm)
    return logL


# Prior in unconstrained z-space
# If you want "start near x_init" and also help geometry:
# set init_scale smaller.
# -------------------------
init_scale = 1.0
prior = dist.Normal(
    torch.tensor(0.0, device=device, dtype=torch_dtype),
    torch.tensor(init_scale, device=device, dtype=torch_dtype)
).expand([ndim]).to_event(1)

# -------------------------
# Pyro model for NUTS
# We observe nothing explicitly; we add your log-likelihood via pyro.factor.
# -------------------------
    
def pyro_model():
    z = pyro.sample("z", prior)  # (ndim,)
    theta = z_to_theta(z.unsqueeze(0)).squeeze(0)  # (ndim,)
    # Add the log-likelihood term
    logL = loglike_torch(theta.unsqueeze(0)).squeeze(0)  # scalar
    # print(f"DEBUG: logL shape: {logL.shape}, value: {logL.item()}")
    pyro.factor("loglike", logL)


pyro_model()

#%%
vals_dict = {
    'r0': torch.tensor([0.0290], device=device),
    'F': torch.tensor([[1.5756, 1.5762]], device=device),
    'dx': torch.tensor([[-0.4894, -0.5194]], device=device),
    'dy': torch.tensor([[0.1962, 0.1548]], device=device),
    'bg': torch.tensor([[-4.9721e-05, -5.2300e-05]], device=device),
    'dn': torch.tensor([0.0010], device=device),
    'Jx': torch.tensor([16.7010], device=device),
    'Jy': torch.tensor([15.9819], device=device),
    'Jxy': torch.tensor([0.], device=device),
    'wind_dir': torch.tensor([[-404.8192,    8.0000]], device=device),
    'LWE_coefs': torch.tensor([[
        -4.9881, -10.5176,   7.6777,   5.7780,  27.6069,   3.5581,
         8.6913,   3.8902, -22.8046,  -1.0048,  13.8456, -25.7376
    ]], device=device)
}

for name, val in vals_dict.items():
    if name in manager.parameters:
        manager[name] = val

x_init = manager.stack()


# x_testo = manager.unstack(x_init, include_all=True, update=True)
# x_init_2 = manager.stack()
# assert torch.allclose(x_init, x_init_2), "Stack/Unstack consistency check failed!"

# PSF_1 = compute_PSF_batch_grad(x_init)[5, ...].unsqueeze(0)  # (1, H, W)

# Test run (wrapped in trace to silence 'outside of inference' warning)
print("Running initial model check...")
with pyro.poutine.trace() as tr:
    pyro_model()

# Compute trace log prob
tr.trace.compute_log_prob()
log_prob = tr.trace.log_prob_sum()
print(f"Check successful. Initial Trace Log-Probability: {log_prob.item():.4f}")


#%% -------------------------
# Run NUTS
# Tips:
# - Start with fewer warmup steps; increase if divergences.
# - Keep target_accept_prob high for stiff posteriors (0.85-0.95).
# -------------------------
pyro.clear_param_store()

nuts_kernel = NUTS(
    pyro_model,
    adapt_step_size=True,
    adapt_mass_matrix=True,
    target_accept_prob=0.9,
)

mcmc = MCMC(
    nuts_kernel,
    num_samples=800,     # posterior draws 
    warmup_steps=600,    # adaptation
    num_chains=1,        # multi-chain is expensive with VRAM-heavy model
)

mcmc.run()

#%%
# -------------------------
# Convert samples to theta-space and summarize
# -------------------------
samples = mcmc.get_samples()           # dict: {"z": (S, ndim)}
z_samps = samples["z"].to(device)
theta_samps = z_to_theta(z_samps).detach().cpu().numpy()  # (S, ndim)

print("Posterior samples shape:", theta_samps.shape)

# Quick summary for each dimension
q16, q50, q84 = np.quantile(theta_samps, [0.16, 0.50, 0.84], axis=0)
for j, name in enumerate(param_names):
    print(f"{name:>12s}: {q50[j]:+.4f}  [{q16[j]:+.4f}, {q84[j]:+.4f}]")

# -------------------------
# Corner plot
# -------------------------
if _HAVE_CORNER:
    fig = corner.corner(theta_samps, labels=param_names, show_titles=True)
    plt.show()
else:
    print("Install 'corner' for corner plots: pip install corner")

# -------------------------
# Posterior predictive check (draw a few curves / PSFs)
# Here we draw K samples and compare to the target PSF.
# -------------------------
K = 8
idx = np.random.choice(theta_samps.shape[0], size=K, replace=False)
theta_k = torch.from_numpy(theta_samps[idx]).to(device=device, dtype=torch_dtype)

# Pad to fixed BATCH_SIZE for a single batched call
params_latent_k = theta_k[:, :-1]
if K < BATCH_SIZE:
    pad = BATCH_SIZE - K
    params_latent_k = torch.cat([params_latent_k, params_latent_k[-1:].expand(pad, -1)], dim=0)

with torch.no_grad():
    PSF_pred = compute_PSF_grad(params_latent_k)[:K]  # (K, H, W)

# Visualize (left/right channels)
fig, ax = plt.subplots(K, 2, figsize=(6, 2*K))
for i in range(K):
    ax[i, 0].imshow((PSF_pred[i, 0] * PSF_mask[0, 0]).detach().cpu().numpy(), origin="lower")
    ax[i, 0].set_title(f"Pred L {i}")
    ax[i, 0].axis("off")

    ax[i, 1].imshow((PSF_pred[i, 1] * PSF_mask[0, 1]).detach().cpu().numpy(), origin="lower")
    ax[i, 1].set_title(f"Pred R {i}")
    ax[i, 1].axis("off")

plt.tight_layout()
plt.show()
# %%
