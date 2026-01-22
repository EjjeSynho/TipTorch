#%%
%reload_ext autoreload
%autoreload 2

import sys

sys.path.insert(0, '..')

import torch
import numpy as np
import matplotlib.pyplot as plt
import ultranest

from tools.plotting import plot_radial_PSF_profiles, draw_PSF_stack
from data_processing.SPHERE_STD_dataset_utils import LoadSTDStarData
from PSF_models.IRDIS_wrapper import PSFModelIRDIS

from project_settings import device

BATCH_SIZE = 128 # Adjust based on GPU memory

# torch.amp.autocast('cuda', dtype=torch.float16 or torch.bfloat16)
# torch.backends.cuda.matmul.allow_tf32 = True

#%%
sample_id = 1778

print(f"Loading sample {sample_id}...")
PSF_data, data_sample, configs = LoadSTDStarData(
    [sample_id,] * BATCH_SIZE,
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

_ = PSF_model()

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
from tools.normalizers import Uniform0_1

span = 2.0
to_manager_space = Uniform0_1(a = -span, b = span)

def transform(cube):
    """
    Transforms unit cube [0, 1] to the parameter space used by likelihood.
    
    We interpret the likelihood parameter space as:
    1. Model Params: InputsManager's normalized space (typically [-1, 1])
    2. Noise Param: Explicit physical log_sigma
    """
    # cube: (B, ndim)
    B, ndim = cube.shape
    output = np.empty_like(cube)
    
    # 1. Model Parameters (Indices 0 to N-1)
    # Map [0, 1] -> [-1, 1] (Uniform Prior in Physical Space via linear transform)
    # This assumes the InputsManager transforms are Uniform(a,b) -> [-1,1]
    output[:, :-1] = to_manager_space.inverse(cube[:, :-1])
    
    # 2. Noise Parameter (Index -1)
    # Prior: Uniform in log_sigma between -8 and 0 (sigma ~3e-4 to 1.0)
    output[:, -1] = -8.0 + cube[:, -1] * 8.0
    
    return output

@torch.no_grad()
def compute_PSF_batch(params_latent_batch):
    """
    Generates a batch of PSFs from normalized parameters using fixed BATCH_SIZE.
    params_latent_batch: Tensor (BATCH_SIZE, n_model_params)
    Returns: Tensor (BATCH_SIZE, H, W) or None if NaN/Error detected
    """
    
    # Unstack using Manager (Latent [-1,1] -> Physical)
    x_dict = manager.unstack(params_latent_batch, include_all=True)
    
    try:
        PSF_batch = PSF_model(x_dict) # (B, H, W)
        
    except RuntimeError as e:
        # Catch errors (e.g. singular matrix in model internals)
        print(f"RuntimeError in PSF model evaluation: {e}")
        return None

    # 4. Check for NaNs/Infs in output
    if torch.isnan(PSF_batch).any() or torch.isinf(PSF_batch).any():
        return None

    return PSF_batch


@torch.no_grad()
def loglike(theta):
    """
    Computes log-likelihood for a batch of parameters.
    Splits input into chunks of fixed BATCH_SIZE to preserve memory stability.
    theta: (N_samples, ndim)
    """
    N_samples = theta.shape[0]
    logL_results = []
    
    # Loop over chunks
    for i in range(0, N_samples, BATCH_SIZE):
        # 1. Extract Chunk
        theta_chunk = theta[i : i + BATCH_SIZE]
        current_batch_size = theta_chunk.shape[0]
                
        # Convert to Tensor
        params_latent = torch.from_numpy(theta_chunk[:, :-1]).to(device, dtype=torch.float32)
        log_sigma     = torch.from_numpy(theta_chunk[:, -1] ).to(device, dtype=torch.float32)
        
        # 2. Pad to fixed BATCH_SIZE if necessary
        # This prevents GPU memory reallocation by ensuring input size is always BATCH_SIZE
        if current_batch_size < BATCH_SIZE:
            pad_size = BATCH_SIZE - current_batch_size
            # Repeat the last element to fill the batch safely (valid physical params)
            params_latent_padded = torch.cat([params_latent, params_latent[-1:].expand(pad_size, -1)], dim=0)
        else:
            params_latent_padded = params_latent
            
        # 3. Generate PSFs via Model
        PSF_batch_padded = compute_PSF_batch(params_latent_padded)
        
        if PSF_batch_padded is None:
            # Model failed (NaNs or Runtime error) -> Assign extremely low likelihood
            logL_chunk = np.full(current_batch_size, -1e15)
        else:
            # 4. Slice back to actual data size
            PSF_batch = PSF_batch_padded[:current_batch_size]
            
            # 5. Compute Likelihood
            # Flatten spatial dimensions and select valid pixels: (B, H, W) -> (B, N_valid)
            pred_valid = PSF_batch.reshape(current_batch_size, -1)[:, mask_bool.view(-1)]
            
            sigma = torch.exp(log_sigma).unsqueeze(1) # (B, 1)
            resid = target_valid.to(device) - pred_valid # Ensure target is strictly on device
            
            # Chi2 Calculation
            chi2 = (resid / sigma).pow(2).sum(dim=1)
            log_norm = N_valid_pix * (np.log(2.0 * np.pi) + 2.0 * log_sigma)
            
            logL_chunk = -0.5 * (chi2 + log_norm)
            logL_chunk = logL_chunk.cpu().numpy()
            
            # Catch NaNs in likelihood result (e.g. from log_sigma issues)
            if np.isnan(logL_chunk).any():
                logL_chunk[np.isnan(logL_chunk)] = -1e15
                
        logL_results.append(logL_chunk)
        
    return np.concatenate(logL_results)

#%%
# Manually assign known values to the manager for initialization or fixing
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
x_init = x_init.repeat(BATCH_SIZE, 1)

# x_testo = manager.unstack(x_init, include_all=True, update=True)
# x_init_2 = manager.stack()
# assert torch.allclose(x_init, x_init_2), "Stack/Unstack consistency check failed!"
#%%
PSF_1 = compute_PSF_batch(x_init)[5, ...].unsqueeze(0)  # (1, H, W)

#%% Results visualization
fig, ax = plt.subplots(1, 2, figsize=(10, 3))
plot_radial_PSF_profiles(
    (PSF_0*PSF_mask)[:,0,...].cpu().numpy(), 
    (PSF_1*PSF_mask)[:,0,...].cpu().numpy(), 
    'Data', 'TipTorch', title='Left PSF', ax=ax[0]
)
plot_radial_PSF_profiles(
    (PSF_0*PSF_mask)[:,1,...].cpu().numpy(), 
    (PSF_1*PSF_mask)[:,1,...].cpu().numpy(), 
    'Data', 'TipTorch', title='Right PSF', ax=ax[1]
)
plt.show()

draw_PSF_stack(PSF_0*PSF_mask, PSF_1*PSF_mask, min_val=1e-6, average=True, crop=80)

#%% --------------------------------------------------------------------------
# Debugging: Batch Logic Verification
# --------------------------------------------------------------------------
print(f"--- Debugging Batch Logic (Fixed BATCH_SIZE={BATCH_SIZE}) ---")
import time

# Test various batch sizes to stress test padding and chunking logic
# 1. Sub-batch (needs padding)
# 2. Exact batch (no padding)
# 3. Cross-boundary (needs chunking + padding)
test_sizes = [BATCH_SIZE // 2, BATCH_SIZE, BATCH_SIZE + 3]
n_params = len(param_names)

for size in test_sizes:
    if size == 0: continue
    
    print(f"\nTesting Input Batch Size: {size}")
    
    # 1. Generate random samples in unit cube [0, 1]
    # cube_test = np.random.rand(size, n_params)
    cube_test = np.random.uniform(0, 1, (size, n_params))
    
    # 2. Transform to likelihood parameter space
    theta_test = transform(cube_test)
    
    # 3. Measure loglike execution
    t0 = time.time()
    logL_test = loglike(theta_test)
    dt = time.time() - t0
    
    # 4. Check results
    print(f"  Execution Time: {dt*1000:.2f} ms")
    print(f"  Output Shape: {logL_test.shape}")
    
    if logL_test.shape[0] != size:
        print(f"  [FAIL] Size mismatch! Expected {size}, got {logL_test.shape[0]}")
    else:
        print(f"  [PASS] Output size matches input.")

    # Check for memory reallocation warnings or failures (simulated via strict shape checks inside loglike)
    if np.any(np.isnan(logL_test)):
         print(f"  [WARN] NaNs produced (might be expected for random un-physical params).")
    else:
         print(f"  [PASS] Values computed (No NaNs).")

print("\n--- Benchmarking Completed ---\n")


#%% ------------------------------------------------------------------------
# Run Nested Sampling
sampler_args = {
    'param_names': param_names,
    'loglike': loglike,
    'transform': transform,
    'vectorized': True,
    'ndraw_max': BATCH_SIZE,
    # 'log_dir': 'ns_output_sphere_manager'
}

sampler = ultranest.ReactiveNestedSampler(**sampler_args)

# Run
result = sampler.run(
    min_num_live_points=200, 
    max_ncalls=10000000,
    show_status=True,
    viz_callback="auto"
)

sampler.print_results()
torch.cuda.empty_cache()

#%% Analyze Results
sampler.plot_corner()
plt.show()

# Get best parameters
idx_max = np.argmax(result['weighted_samples']['logl'])
best_theta = result['weighted_samples']['points'][idx_max]

#%%
best_dict = PSF_model.inputs_manager.unstack(
    torch.from_numpy(best_theta[:-1]).to(device, dtype=torch.float32).unsqueeze(0),
    include_all=True
)
x_out = torch.from_numpy(best_theta[:-1]).to(device, dtype=torch.float32).unsqueeze(0).repeat(BATCH_SIZE, 1)
PSF_2 = compute_PSF_batch(x_out)[0, ...].unsqueeze(0)  # (1, H, W)

#%%
fig, ax = plt.subplots(1, 2, figsize=(10, 3))
plot_radial_PSF_profiles(
    (PSF_0*PSF_mask)[:,0,...].cpu().numpy(), 
    (PSF_2*PSF_mask)[:,0,...].cpu().numpy(), 
    'Data', 'TipTorch', title='Left PSF', ax=ax[0]
)
plot_radial_PSF_profiles(
    (PSF_0*PSF_mask)[:,1,...].cpu().numpy(), 
    (PSF_2*PSF_mask)[:,1,...].cpu().numpy(), 
    'Data', 'TipTorch', title='Right PSF', ax=ax[1]
)
plt.show()

draw_PSF_stack(PSF_0*PSF_mask, PSF_2*PSF_mask, min_val=1e-6, average=True, crop=80)


#%%
# Print Physical Values
print("\nRecovered Physical Parameters:")
for k, v in best_dict.items():
    if manager.is_optimizable(k):
        # Print scalar mean/values
        val = v.squeeze().cpu().numpy()
        print(f"  {k}: {val}")
        
        
# %%
