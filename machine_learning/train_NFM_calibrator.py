#%%
try:
    ipy = get_ipython()        # NameError if not running under IPython
    if ipy:
        ipy.run_line_magic('reload_ext', 'autoreload')
        ipy.run_line_magic('autoreload', '2')
except NameError:
    pass

import sys
sys.path.insert(0, '..')

import gc
import logging
import argparse
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset, random_split
from sklearn.model_selection import train_test_split, KFold
from pathlib import Path
from datetime import datetime
from project_settings import *
from torch.utils.data import Dataset

from managers.config_manager import MultipleTargetsInDifferentObservations

MUSE_DATA_FOLDER = Path(project_settings["MUSE_data_folder"])
STD_FOLDER = MUSE_DATA_FOLDER / 'standart_stars/'
DATASET_CACHE = STD_FOLDER / 'dataset_cache'

pre_init_astrometry = True
optimize_astrometry = False
predict_LOs = True

# Set up logging
log_dir = Path('../data/logs')
log_dir.mkdir(parents=True, exist_ok=True)
log_filename = log_dir / f'training_NFM_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()  # Also log to console
    ]
)
logger = logging.getLogger(__name__)

logger.info("="*60)
logger.info("NFM Calibrator Training Script")
logger.info("="*60)
logger.info(f"Log file: {log_filename}")


# Add argument inputs
parser = argparse.ArgumentParser(description="Train NFM Calibrator")
# parser.add_argument('--weights', type=str, default=str(WEIGHTS_FOLDER / 'NFM_calibrator_new/best_calibrator_checkpoint.pth'), help='Path to the best calibrator weights checkpoint')
parser.add_argument('--continue-training', action='store_true', help='Whether to continue training from a checkpoint')

# Handle both command line and iPython environments
try:
    args = parser.parse_args()
except SystemExit:
    # In iPython/Jupyter, parse_args() fails, so use default values
    args = argparse.Namespace(continue_training=False)

# from data_processing.MUSE_STD_dataset_utils import *

N_wvl_total = 30
batch_size = 16

#%%
class NFMDataset(Dataset):
    def __init__(self):
        self.PSF_cubes  = np.load(DATASET_CACHE / 'muse_STD_stars_PSFs.npy',      mmap_mode="r")
        self.telemetry  = np.load(DATASET_CACHE / 'muse_STD_stars_telemetry.npy', mmap_mode="r")
        self.configs = torch.load(DATASET_CACHE / 'muse_STD_stars_configs.pt')

        for i, config in enumerate(self.configs):
            if isinstance(config, list):
                self.configs[i] = self.configs[i][0]  # Unwrap single-item lists

        self.N, self.H, self.W, self.C = self.PSF_cubes.shape

    def __len__(self): return self.N

    def __getitem__(self, idx):
        # Make writable copies of memory-mapped arrays
        cube = self.PSF_cubes[idx].copy()  # Force copy to make it writable
        vec  = self.telemetry[idx].copy()  # Force copy to make it writable
        conf = self.configs[idx]
        
        # Create tensors from writable arrays
        cube_tensor = torch.from_numpy(cube.astype(np.float32))  # Ensure float32 and writable
        vec_tensor  = torch.from_numpy(vec.astype(np.float32))    # Ensure float32 and writable
    
        return cube_tensor.permute(2,0,1), vec_tensor, conf, idx  # (C, H, W), (vector,), config, index


def collate_batch(batch, device):
    PSF_cubes, telemetry_vecs, configs, idxs = zip(*batch)
        
    PSF_cubes = torch.stack(PSF_cubes, 0).to(device=device, non_blocking=True)
    telemetry_vecs = torch.stack(telemetry_vecs, 0).to(device=device, non_blocking=True)
    idxs = torch.tensor(idxs, dtype=torch.long, device=device)

    batch_config = MultipleTargetsInDifferentObservations(list(configs), device=device)
    batch_config['PathPupil'] = str(DATA_FOLDER / 'calibrations/VLT_CALIBRATION/VLT_PUPIL/ut4pupil320.fits')
    batch_config['telescope']['PupilAngle'] = 0.0
    
    return PSF_cubes, telemetry_vecs, batch_config, idxs


#%%
NFM_dataset = NFMDataset()

print(f"Total dataset size: {len(NFM_dataset)}")
logger.info(f"Total dataset size: {len(NFM_dataset)}")

random_idxs = np.random.randint(0, len(NFM_dataset), size=batch_size)
cubes, vecs, configs, idxs = [], [], [], []
for random_idx in random_idxs:
    cube, vec, config, idx = NFM_dataset[random_idx]
    cubes.append(cube)
    vecs.append(vec)
    configs.append(config)
    idxs.append(idx)

cubes = torch.stack(cubes, 0).to(device=device, non_blocking=True)
vecs  = torch.stack(vecs,  0).to(device=device, non_blocking=True)
idxs  = torch.tensor(idxs, dtype=torch.long, device=device)

N_features = vecs.shape[-1]

#%% =============================================================================
# Example model class placeholder (replace with your actual model)
# ==============================================================================
from PSF_models.NFM_wrapper import PSFModelNFM

if 'PSF_model' in locals():
    PSF_model.cleanup()  # Recursively cleans everything
    del PSF_model
    gc.collect()
    torch.cuda.empty_cache()


with torch.no_grad():
    PSF_model = PSFModelNFM(
        configs,
        multiple_obs    = True,
        LO_NCPAs        = True,
        chrom_defocus   = False,
        use_splines     = True,
        Moffat_absorber = False,
        Z_mode_max      = 9,
        device          = device
    )
    PSF_model.inputs_manager.delete('Jxy')
    PSF_model.inputs_manager.delete('bg_ctrl')
    PSF_model.inputs_manager.delete('dx_ctrl')
    PSF_model.inputs_manager.delete('dy_ctrl')
    PSF_model.inputs_manager.delete('L0')
    
    if PSF_model.Moffat_absorber:
        # PSF_model.inputs_manager.delete('beta')
        # PSF_model.inputs_manager.delete('b')
        PSF_model.inputs_manager.delete('theta')
        PSF_model.inputs_manager.delete('ratio')
    
    if not predict_LOs:
        PSF_model.inputs_manager.set_optimizable(['LO_coefs'], False)

    print(PSF_model.inputs_manager)
    N_outputs = PSF_model.inputs_manager.get_stacked_size()

    inputs_transformer = PSF_model.inputs_manager.get_transformer()

print(f" >>>>>>>>>>>> Model inputs: {N_features}, outputs: {N_outputs}")

gc.collect()
torch.cuda.empty_cache()

# %%
λ_full = PSF_model.wavelengths.clone().cpu() # [nm]

# λ_id_sets = [
#     [0, 5, 10, 15, 20, 25, 29],
#     [1, 6, 11, 16, 21, 26, 29],
#     [2, 7, 12, 17, 22, 27, 29],
#     [0, 3,  8, 13, 18, 23, 28],
#     [0, 4,  9, 14, 19, 24, 28]
# ]

# λ_id_sets = [
#     [0, 3, 6,  9, 12, 15, 18, 21, 24, 27, 29],  
#     [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 29],
#     [0, 2, 5,  8, 11, 14, 17, 20, 23, 26, 29],
# ]

λ_id_sets = [
    [0, 4,  8, 12, 16, 20, 24, 28],
    [1, 5,  9, 13, 17, 21, 25, 29],
    [2, 6, 10, 14, 18, 22, 26, 29],
    [0, 3,  7, 11, 15, 19, 23, 27]
]

λ_sets = [λ_full[list_id] for list_id in λ_id_sets]

#%%
logger.info(f"Pre-initialize astrometry: {pre_init_astrometry}")
logger.info(f"Optimize astrometry (dx/dy): {optimize_astrometry}")

if optimize_astrometry:
    logger.info("✓ Astrometry parameters (dx/dy) will be optimized during training")
else:
    logger.info("✗ Astrometry parameters (dx/dy) will be fixed during training")

if pre_init_astrometry:
    logger.info("✓ Using pre-fitted astrometry values for initialization")
else:
    logger.info("✗ Initializing astrometry with zeros")

fitted_df = pickle.load(open(STD_FOLDER / 'muse_fitted_df.pkl', 'rb'))

wvl_ids = np.clip(np.arange(0, (N_wvl_max:=30)+1, 2), a_min=0, a_max=N_wvl_max-1)
wvl_ids_shift = np.clip(wvl_ids + 1, a_min=0, a_max=N_wvl_max-1)[:-2]

dx_df = fitted_df['dx_df'].sort_index().to_numpy().astype(np.float32)
dy_df = fitted_df['dy_df'].sort_index().to_numpy().astype(np.float32)

dx_full_arr = np.zeros((dx_df.shape[0], N_wvl_total), dtype=np.float32)
dy_full_arr = np.zeros((dy_df.shape[0], N_wvl_total), dtype=np.float32)

for i, wvl_id in enumerate(wvl_ids):
    dx_full_arr[:, wvl_id] = dx_df[:, i]
    dy_full_arr[:, wvl_id] = dy_df[:, i]
# Linear interpolation for missing wavelengths
for i, wvl_id in enumerate(wvl_ids_shift):
    dx_full_arr[:, wvl_id] = 0.5*(dx_df[:, i] + dx_df[:, i+1])
    dy_full_arr[:, wvl_id] = 0.5*(dy_df[:, i] + dy_df[:, i+1])

if optimize_astrometry:
    if pre_init_astrometry:
        # Use pre-fitted astrometry values as initialization
        dx = torch.from_numpy(dx_full_arr).to(device=device, dtype=torch.float32)
        dy = torch.from_numpy(dy_full_arr).to(device=device, dtype=torch.float32)
        dx.requires_grad_(True)
        dy.requires_grad_(True)
    else:
        # Initialize astrometry with zeros
        dx = torch.zeros((len(NFM_dataset), N_wvl_total), device=device, dtype=torch.float32, requires_grad=True)
        dy = torch.zeros((len(NFM_dataset), N_wvl_total), device=device, dtype=torch.float32, requires_grad=True)
else:
    if pre_init_astrometry:
        # Use pre-fitted astrometry values (fixed, no gradients)
        dx = torch.from_numpy(dx_full_arr).to(device=device, dtype=torch.float32)
        dy = torch.from_numpy(dy_full_arr).to(device=device, dtype=torch.float32)
    else:
        # No astrometry optimization, use zeros (fixed)
        dx = torch.zeros((len(NFM_dataset), N_wvl_total), device=device, dtype=torch.float32)
        dy = torch.zeros((len(NFM_dataset), N_wvl_total), device=device, dtype=torch.float32)


if not predict_LOs:
    logger.info("✓ Using pre-fitted NCPAs values for initialization")
    phase_bump_dataset = fitted_df['LO_df']['Phase bump'].sort_index().to_numpy().astype(np.float32)
    phase_bump_dataset = torch.from_numpy(phase_bump_dataset).to(device=device, dtype=torch.float32)
    LO_median = torch.from_numpy(fitted_df['LO_df'].median().to_numpy()[1:]).to(device=device, dtype=torch.float32)
else:
    logger.info("✗ Initializing NCPAs with zeros")


#%%
# def run_model(predicted_inputs_vec, config, idx, λ_ids):
def run_model(x_dict, config, idx, λ_ids):
    # Set dx/dy for this batch (indexed by sample IDs)
    x_dict['dx'] = dx[idx[:,None], λ_ids]
    x_dict['dy'] = dy[idx[:,None], λ_ids]

    # Update simulated wavelengths
    config['sources_science']['Wavelength'] = λ_full[λ_ids].unsqueeze(0).to(device=device)

    if not predict_LOs:
        # Set LO NCPAs information from fitting
        x_dict['LO_coefs'] = torch.hstack( (phase_bump_dataset[idx].unsqueeze(-1), LO_median.repeat(len(idx),1)) )

    # Update internal state of the PSF model for the given batch config. Update just model parameters, not grids
    PSF_model.model.Update(config=config, init_grids=False, init_pupils=False, init_tomography=True) 
    return PSF_model(x_dict)

#%%
# Keep it shallow - 1-2 hidden layers max
# Keep it narrow - 24-32 hidden units
# Strong regularization - Dropout + weight decay + L1
# Small batches - 8-16 samples
# Early stopping - Essential to prevent overfitting
# Data augmentation - If possible, add small noise to inputs
# Cross-validation - Consider k-fold CV for better estimates

import torch.nn as nn

class SmallCalibratorNet(nn.Module):
    """
    Compact neural network for calibration with strong regularization.
    Designed for small datasets (~500 samples).
    
    Args:
        n_features: Number of input features (22)
        n_outputs: Number of output values (31)
        hidden_dim: Size of hidden layers (default: 32)
        dropout_rate: Dropout probability (default: 0.3)
    """
    def __init__(self, n_features=22, n_outputs=31, hidden_dim=32, dropout_rate=0.3):
        super().__init__()
        
        self.network = nn.Sequential(
            # Input layer
            nn.Linear(n_features, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Better than BatchNorm for small batches
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Hidden layer
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Output layer
            nn.Linear(hidden_dim, n_outputs),
        )
        
        # Initialize weights with smaller values for better regularization
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)  # Smaller gain
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.network(x)


'''
# Alternative: Even more compact version
class TinyCalibratorNet(nn.Module):
    """Ultra-compact version for very small datasets."""
    def __init__(self, n_features=22, n_outputs=31, hidden_dim=24, dropout_rate=0.4):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, n_outputs),
            nn.Tanh()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.network(x)
'''
#%%
import torch.optim as optim

# Data loaders
train_idx, test_idx = train_test_split(np.arange(len(NFM_dataset)), test_size=0.20, random_state=42)

train_dataset = Subset(NFM_dataset, train_idx)
val_dataset   = Subset(NFM_dataset, test_idx)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    # pin_memory=True if torch.cuda.is_available() else False,
    collate_fn=lambda batch: collate_batch(batch, device=device),
    drop_last=False
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    # pin_memory=True if torch.cuda.is_available() else False,
    collate_fn=lambda batch: collate_batch(batch, device=device),
    drop_last=False
)

# Initialize calibrator NN
calibrator = SmallCalibratorNet(
    n_features=N_features,
    n_outputs=N_outputs,
    hidden_dim=32,
    dropout_rate=0.3
).to(device)


if args.continue_training:
    checkpoint_path = WEIGHTS_FOLDER / 'NFM_calibrator_new/best_calibrator_checkpoint.pth'
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        calibrator.load_state_dict(checkpoint['calibrator_state_dict'])
        dx = checkpoint['dx'].to(device)
        dy = checkpoint['dy'].to(device)

        dx.requires_grad_(True)
        dy.requires_grad_(True)

        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    except Exception as e:
        logger.warning(f"Failed to load checkpoint from {checkpoint_path}: {e}. Starting training from scratch.")


default_lr = 1e-4

# Optimizer with weight decay (L2 regularization)
if optimize_astrometry:
    optimizer = optim.AdamW([
        {'params': calibrator.parameters(), 'lr': default_lr, 'weight_decay': 5e-4},
        {'params': [dx, dy], 'lr': 1e-3, 'weight_decay': 1e-5}  # Lower weight decay for dx/dy
    ], lr=1e-3)  # Default values
else:
    # Only optimize calibrator parameters, not dx/dy
    optimizer = optim.AdamW([
        {'params': calibrator.parameters(), 'lr': default_lr, 'weight_decay': 5e-4}
    ], lr=default_lr)  # Default values


scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode     = 'min',
    factor   = 0.2,
    patience = 3,
    verbose  = True,
    min_lr   = 1e-6
)

logger.info(f"Scheduler: ReduceLROnPlateau with patience={3}, factor={0.7}, min_lr={1e-5}")

#%%
wvl_weights = torch.linspace(1.0, 0.5, N_wvl_total).to(device).view(1, N_wvl_total, 1, 1)
wvl_weights = N_wvl_total / wvl_weights.sum() * wvl_weights # Normalize so that the total energy is preserved

# logger.info(f"Wavelength weights range: {wvl_weights.min().item():.4f} - {wvl_weights.max().item():.4f}")

def loss_LO_fn(LO_coefs, w_L2=1e-7, w_first=5e-5):
    coefs_L2_penalty = LO_coefs.pow(2).sum(-1).mean() * w_L2
    first_coef_penalty = torch.clamp(-LO_coefs[:, 0], min=0).pow(2).mean() * w_first
    return coefs_L2_penalty + first_coef_penalty


def Moffat_loss_fn(x_dict):
    amp = x_dict['amp']
    # alpha = x_dict['alpha']
    # beta = x_dict['beta']
    # b = x_dict['b']

    # Enforce positive amplitude
    amp_penalty = amp.pow(2).mean() * 2.5e-2 * 2
    
    # Enforce beta > 1.5
    # beta_penalty = torch.clamp(1.5 - beta, min=0).pow(2).mean() * 1e-3
    
    # # Enforce alpha > 0
    # alpha_penalty = torch.clamp(-alpha, min=0).pow(2).mean() * 1e-3
    
    # # Enforce b > 0
    # b_penalty = torch.clamp(-b, min=0).pow(2).mean() * 1e-3
    
    return amp_penalty #+ beta_penalty + alpha_penalty + b_penalty


def loss_fn(PSF_pred, PSF_data, x_dict, w_MSE, w_MAE, w_L2, w_first, λ_ids):  
    diff = (PSF_pred-PSF_data) * wvl_weights[:, λ_ids, ...]
    w = 2e4
    MSE_loss = diff.pow(2).mean() * w * w_MSE
    MAE_loss = diff.abs().mean()  * w * w_MAE
    LO_loss  = loss_LO_fn(x_dict['LO_coefs'], w_L2=w_L2, w_first=w_first) if PSF_model.LO_NCPAs and predict_LOs else 0.0
    Moffat_loss = Moffat_loss_fn(x_dict) if PSF_model.Moffat_absorber else 0.0
    
    # Add safety check for NaN in loss components (helps debug NaN sources)
    if torch.isnan(MSE_loss) or torch.isnan(MAE_loss):
        logger.error(f"NaN in loss components - MSE: {MSE_loss.item()}, MAE: {MAE_loss.item()}, LO: {LO_loss}")
        logger.error(f"PSF_pred range: [{PSF_pred.min().item()}, {PSF_pred.max().item()}]")
        logger.error(f"PSF_data range: [{PSF_data.min().item()}, {PSF_data.max().item()}]")

    return MSE_loss + MAE_loss + LO_loss + Moffat_loss


criterion = lambda pred, data, x_dict, λ_ids: loss_fn(pred, data, x_dict, w_MSE=900.0, w_MAE=1.6, w_L2=1e-4, w_first=5e-5, λ_ids=λ_ids)
# criterion = lambda pred, data, x_dict, λ_ids: loss_fn(pred, data, x_dict, w_MSE=900.0, w_MAE=3.2, w_L2=5e-7, w_first=5e-5, λ_ids=λ_ids)

# L1 regularization of NN parameters
def l1_regularization(model, lambda_l1=1e-4):
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    return lambda_l1 * l1_norm

loss_Huber = torch.nn.HuberLoss(reduction='mean', delta=0.05)

def loss_fn_Huber(PSF_pred, PSF_data, x_dict, w_L2, w_first, λ_ids):
    w = 5e5
    huber_loss = loss_Huber(
        PSF_pred * wvl_weights[:, λ_ids, ...] * w,
        PSF_data * wvl_weights[:, λ_ids, ...] * w
    )
    LO_loss  = loss_LO_fn(x_dict['LO_coefs'], w_L2=w_L2, w_first=w_first) if PSF_model.LO_NCPAs and predict_LOs else 0.0
    Moffat_loss = Moffat_loss_fn(x_dict) if PSF_model.Moffat_absorber else 0.0
    
    if torch.isnan(huber_loss):
        logger.error(f"NaN in Huber loss - Huber: {huber_loss.item()}, LO: {LO_loss}")
        logger.error(f"PSF_pred range: [{PSF_pred.min().item()}, {PSF_pred.max().item()}]")
        logger.error(f"PSF_data range: [{PSF_data.min().item()}, {PSF_data.max().item()}]")
    
    return huber_loss + LO_loss + Moffat_loss

criterion_Huber = lambda pred, data, x_dict, λ_ids: loss_fn_Huber(pred, data, x_dict, w_L2=5e-7, w_first=5e-5, λ_ids=λ_ids)

# criterion = criterion_Huber

# %%
'''
num_epochs = 5

for epoch in range(num_epochs):
    for current_λ_set_id in range(len(λ_sets)): # Iterate through all wavelength sets
        PSF_model.SetWavelengths(λ_sets[current_λ_set_id].to(device=device))

        λ_id_set = λ_id_sets[current_λ_set_id] # which wavelengths ids are currently selected

        for batch_idx, batch in enumerate(train_loader):
            PSF_cubes, telemetry_inputs, batch_config, idxs = batch

            optimizer.zero_grad()
            
            #  Calibrator NN predicts corrections
            NN_output = calibrator(telemetry_inputs)
            
            x_pred = inputs_transformer.unstack(NN_output)

            PSF_pred = run_model(x_pred, batch_config, idxs, λ_id_set)

            # Select only the relevant wavelengths
            loss = criterion(PSF_pred, PSF_cubes[:, λ_id_set, ...], x_pred, λ_id_set)

            loss.backward() # gradients flow to both calibrator and dx/dy
            # print(loss.item())

            # Gradient clipping (prevents explosion if there is an outlier in the batch)
            torch.nn.utils.clip_grad_norm_(calibrator.parameters(), max_norm=1.0)
            
            optimizer.step()
            
        # gc.collect()
        print(f"\rλ set {current_λ_set_id + 1}/{len(λ_sets)}, batch {batch_idx + 1}: val_loss = {final_loss.item():.6f}", end='', flush=True)

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
'''

def validate_with_astrometry_optimization(num_opt_steps=20, lr_dx_dy=1e-3):
    """
    Validation routine where:
    - Calibrator NN weights are FROZEN (in eval mode, no gradients)
    - dx/dy values for validation samples are OPTIMIZED (if optimize_astrometry=True)
    
    Args:
        num_opt_steps: Number of optimization steps per validation sample
        lr_dx_dy: Learning rate for dx/dy optimization
    
    Returns:
        average validation loss after optimizing dx/dy (or just evaluation if astrometry disabled)
    """
    # Set calibrator to eval mode (freezes BatchNorm, Dropout, etc.)
    calibrator.eval()
    
    total_loss  = 0
    num_batches = 0
    
    # Iterate through all wavelength sets
    for current_λ_set_id in range(len(λ_sets)):
        PSF_model.SetWavelengths(λ_sets[current_λ_set_id].to(device=device))
        λ_id_set = λ_id_sets[current_λ_set_id]
        
        for batch_idx, batch in enumerate(val_loader):
            PSF_cubes, telemetry_inputs, batch_config, idxs = batch
            
            if optimize_astrometry:
                # Create optimizer for ONLY dx/dy (NOT calibrator!)
                # Only optimize dx/dy for the current validation samples
                val_dx_dy_optimizer = optim.Adam([
                    {'params': [dx], 'lr': lr_dx_dy},
                    {'params': [dy], 'lr': lr_dx_dy}
                ])
                
                # Optimize dx/dy for this validation batch
                for opt_step in range(num_opt_steps):
                    val_dx_dy_optimizer.zero_grad()
                    
                    # Forward pass with FROZEN calibrator
                    with torch.no_grad():  # No gradients for calibrator!
                        NN_output = calibrator(telemetry_inputs)
                    
                    x_pred = inputs_transformer.unstack(NN_output)

                    PSF_pred = run_model(x_pred, batch_config, idxs, λ_id_set)
                    loss = criterion(PSF_pred, PSF_cubes[:, λ_id_set, ...], x_pred, λ_id_set)
                    
                    loss.backward() # Only dx/dy get gradients
                    
                    # torch.nn.utils.clip_grad_norm_([dx, dy], max_norm=1.0)
        
                    val_dx_dy_optimizer.step() # Update ONLY dx/dy
            
            # Record final loss for this batch
            with torch.no_grad():
                NN_output = calibrator(telemetry_inputs)
                x_pred = inputs_transformer.unstack(NN_output)

                PSF_pred = run_model(x_pred, batch_config, idxs, λ_id_set)
                final_loss = criterion(PSF_pred, PSF_cubes[:, λ_id_set, ...], x_pred, λ_id_set)

                print(f"\rλ set {current_λ_set_id + 1}/{len(λ_sets)}, batch {batch_idx + 1}: val_loss = {final_loss.item():.6f}", end='', flush=True)
                logger.debug(f"Validation: λ set {current_λ_set_id + 1}/{len(λ_sets)}, batch {batch_idx + 1}/{len(val_loader)}: val_loss = {final_loss.item():.6f}")
                total_loss += final_loss.item()
                num_batches += 1

    # Clear cache between epochs
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    # Set calibrator back to train mode
    calibrator.train()
    
    return avg_loss


# Simpler validation without optimization (just evaluate)
def validate_fixed():
    """
    Simple validation: just evaluate with current dx/dy values.
    No optimization, just forward pass.
    
    Returns:
        PSFs_pred_cube: Tensor of shape (N_val_samples_actual, N_wvl_total, H, W) with all predicted PSFs
        PSFs_data_cube: Tensor of shape (N_val_samples_actual, N_wvl_total, H, W) with all data PSFs
        validation_ids: Tensor of shape (N_val_samples_actual,) with validation sample IDs
        avg_loss: Average validation loss
    """
    calibrator.eval()
    total_loss = 0
    num_batches = 0
    
    # Get dimensions from first batch to initialize the output cubes
    first_batch = next(iter(val_loader))
    PSF_cubes_sample, _, _, _ = first_batch
    
    # Calculate actual number of validation samples (accounting for drop_last=True)
    N_val_dataset = len(val_dataset)
    N_val_batches = len(val_loader)  # This accounts for drop_last
    N_val_samples = N_val_dataset  # Actual samples processed
    
    _, H, W = PSF_cubes_sample.shape[1:]  # Get H, W from (batch, C, H, W)
    
    # Initialize output cubes with ACTUAL number of samples that will be processed
    PSFs_pred_cube = torch.zeros((N_val_samples, N_wvl_total, H, W), dtype=torch.float32, device='cpu')
    PSFs_data_cube = torch.zeros((N_val_samples, N_wvl_total, H, W), dtype=torch.float32, device='cpu')
    validation_ids = torch.zeros((N_val_samples,), dtype=torch.long, device='cpu')

    NN_predictions = torch.zeros((N_val_samples, N_outputs), dtype=torch.float32, device='cpu')

    logger.info(f"Validation dataset size: {N_val_dataset}, batches: {N_val_batches}, actual samples: {N_val_samples}")
    logger.info(f"Initializing PSF cubes: predictions and data, shape: {PSFs_pred_cube.shape}")
    logger.info(f"Initializing validation IDs array, shape: {validation_ids.shape}")
    
    with torch.no_grad():  # No gradients at all
        batch_start_idx = 0
        
        # Outer loop: iterate through batches
        for batch_idx, batch in enumerate(val_loader):
            PSF_cubes, telemetry_inputs, batch_config, idxs = batch
            batch_size_current = PSF_cubes.shape[0]  # May be different for last batch if drop_last=False
            batch_end_idx = batch_start_idx + batch_size_current
            
            # Store validation sample IDs for this batch (only once per batch)
            validation_ids[batch_start_idx:batch_end_idx] = idxs.cpu()
            
            # Get calibrator predictions (same for all wavelength sets)
            NN_output = calibrator(telemetry_inputs)
            x_pred = inputs_transformer.unstack(NN_output)
            
            NN_predictions[batch_start_idx:batch_end_idx, :] = NN_output.cpu()
            
            # Inner loop: iterate through wavelength sets
            for current_λ_set_id in range(len(λ_sets)):
                PSF_model.SetWavelengths(λ_sets[current_λ_set_id].to(device=device))
                λ_id_set = λ_id_sets[current_λ_set_id]  # Which wavelength indices for this set
                
                # Run model for this wavelength set
                PSF_pred = run_model(x_pred, batch_config, idxs, λ_id_set)
                
                # Calculate loss for this batch and wavelength set
                loss = criterion(PSF_pred, PSF_cubes[:, λ_id_set, ...], x_pred, λ_id_set)
                total_loss += loss.item()
                num_batches += 1
                
                # Fill in the wavelengths for this batch
                for wvl_idx, λ_id in enumerate(λ_id_set):
                    PSFs_pred_cube[batch_start_idx:batch_end_idx, λ_id, :, :] = PSF_pred[:, wvl_idx, :, :].cpu()
                    PSFs_data_cube[batch_start_idx:batch_end_idx, λ_id, :, :] = PSF_cubes[:, λ_id, :, :].cpu()
                
                print(f"\rBatch {batch_idx + 1}/{len(val_loader)}, λ set {current_λ_set_id + 1}/{len(λ_sets)}", end='', flush=True)
            
            batch_start_idx = batch_end_idx

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    calibrator.train()
    
    print()  # New line after progress
    logger.info(f"PSFs_pred cube shape: {PSFs_pred_cube.shape}")
    logger.info(f"PSFs_data cube shape: {PSFs_data_cube.shape}")
    logger.info(f"Validation IDs shape: {validation_ids.shape}")
    logger.info(f"PSF pred range: [{PSFs_pred_cube.min():.4e}, {PSFs_pred_cube.max():.4e}]")
    logger.info(f"PSF data range: [{PSFs_data_cube.min():.4e}, {PSFs_data_cube.max():.4e}]")
    logger.info(f"Validation IDs range: [{validation_ids.min()}, {validation_ids.max()}]")

    return PSFs_pred_cube, PSFs_data_cube, validation_ids, NN_predictions, total_loss / num_batches if num_batches > 0 else 0.0


# Helper function to check for NaN values
def check_for_nan(loss, model, epoch, batch_idx, phase="train"):
    """
    Check if loss or model parameters contain NaN values.
    Returns True if NaN detected, False otherwise.
    """
    if torch.isnan(loss) or torch.isinf(loss):
        logger.error(f"❌ NaN/Inf detected in {phase} loss at epoch {epoch}, batch {batch_idx}: {loss.item()}")
        return True
    
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            logger.error(f"❌ NaN/Inf detected in {phase} parameter '{name}' at epoch {epoch}, batch {batch_idx}")
            return True
    
    return False


def save_checkpoint(calibrator, dx, dy, optimizer, epoch, train_loss, val_loss, path='checkpoint.pth'):
    """Save a training checkpoint."""
    checkpoint_data = {
        'epoch': epoch,
        'calibrator_state_dict': calibrator.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'optimize_astrometry': optimize_astrometry,
        'pre_init_astrometry': pre_init_astrometry,
    }
    
    if optimize_astrometry:
        checkpoint_data['dx'] = dx.detach().clone()
        checkpoint_data['dy'] = dy.detach().clone()
    
    torch.save(checkpoint_data, path)
    logger.debug(f"Checkpoint saved to {path}")


def load_checkpoint(calibrator, dx, dy, optimizer, path='checkpoint.pth'):
    """Load a training checkpoint and restore model state."""
    checkpoint = torch.load(path)
    calibrator.load_state_dict(checkpoint['calibrator_state_dict'])
    
    # Load astrometry parameters if they were saved
    if optimize_astrometry and 'dx' in checkpoint and 'dy' in checkpoint:
        dx.data = checkpoint['dx']
        dy.data = checkpoint['dy']
        logger.info("✓ Astrometry parameters (dx/dy) loaded from checkpoint")
    elif optimize_astrometry:
        logger.warning("⚠ Astrometry optimization enabled but no dx/dy found in checkpoint")
    else:
        logger.info("✗ Astrometry optimization disabled, skipping dx/dy loading")
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    logger.warning(f"🔄 Checkpoint loaded from {path} (epoch {checkpoint['epoch']})")
    return checkpoint['epoch'], checkpoint['train_loss'], checkpoint['val_loss']


# Complete training loop with validation
def train_with_validation(num_epochs=50, patience=10, val_dx_dy_opt_steps=20, val_dx_dy_lr=1e-3, 
                          nan_recovery=True, max_nan_recoveries=3):
    """
    Training loop with validation where dx/dy are optimized during validation.
    
    Args:
        num_epochs: Maximum number of training epochs
        patience: Early stopping patience
        val_dx_dy_opt_steps: Number of optimization steps for dx/dy during validation
        val_dx_dy_lr: Learning rate for dx/dy optimization during validation
        nan_recovery: If True, recover from NaN by loading last good checkpoint
        max_nan_recoveries: Maximum number of times to recover from NaN before giving up
    """
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    nan_recovery_count = 0
    patience_counter   = 0
    
    # Initialize loss tracking for detailed logging
    loss_stats = {
        'train_losses_per_epoch': [],
        'val_losses_per_epoch':   [],
        'train_losses_per_batch': [],
        'val_losses_per_batch':   [],
        'learning_rates': [],
        'best_epoch': 0,
        'best_val_loss': float('inf'),
        'nan_recoveries': 0
    }
    
    logger.info(f"Starting training for {num_epochs} epochs")
    logger.info(f"Patience: {patience}, Val dx/dy opt steps: {val_dx_dy_opt_steps}")
    logger.info(f"NaN recovery: {'enabled' if nan_recovery else 'disabled'}, max recoveries: {max_nan_recoveries}")

    for epoch in range(num_epochs):
        # ========== TRAINING ==========
        calibrator.train()
        epoch_train_loss = 0
        train_batch_count = 0
        nan_detected_this_epoch = False
        
        for current_λ_set_id in range(len(λ_sets)):
            PSF_model.SetWavelengths(λ_sets[current_λ_set_id].to(device=device))
            λ_id_set = λ_id_sets[current_λ_set_id]
            
            for batch_idx, batch in enumerate(train_loader):
                PSF_cubes, telemetry_inputs, batch_config, idxs = batch
                
                optimizer.zero_grad()
                
                # Forward pass
                NN_output = calibrator(telemetry_inputs)
                x_pred = inputs_transformer.unstack(NN_output)

                PSF_pred = run_model(x_pred, batch_config, idxs, λ_id_set)

                loss = criterion(PSF_pred, PSF_cubes[:, λ_id_set, ...], x_pred, λ_id_set)

                # ========== NaN DETECTION ==========
                if check_for_nan(loss, calibrator, epoch, batch_idx, phase="train"):
                    nan_detected_this_epoch = True
                    
                    if nan_recovery and nan_recovery_count < max_nan_recoveries:
                        nan_recovery_count += 1
                        loss_stats['nan_recoveries'] = nan_recovery_count
                        
                        logger.error(f"🚨 NaN detected! Recovery attempt {nan_recovery_count}/{max_nan_recoveries}")
                        print(f"\n🚨 NaN detected! Attempting recovery {nan_recovery_count}/{max_nan_recoveries}...")
                        
                        # Load last good checkpoint
                        try:
                            load_checkpoint(calibrator, dx, dy, optimizer, WEIGHTS_FOLDER / 'NFM_calibrator_new/best_calibrator_checkpoint.pth')
                            
                            # Reduce learning rate
                            for param_group in optimizer.param_groups:
                                old_lr = param_group['lr']
                                param_group['lr'] = old_lr * 0.7
                                logger.warning(f"Reduced LR: {old_lr:.2e} -> {param_group['lr']:.2e}")

                            print(f"✓ Recovered from checkpoint, LR reduced by 25%")
                            logger.info("Model recovered from checkpoint, continuing training...")
                            break  # Skip rest of this epoch, start fresh
                            
                        except FileNotFoundError:
                            logger.error("❌ No checkpoint found! Cannot recover from NaN.")
                            print("❌ No checkpoint available for recovery!")
                            raise ValueError("NaN detected but no checkpoint to recover from")
                    else:
                        logger.error(f"❌ Max NaN recoveries ({max_nan_recoveries}) reached or recovery disabled. Stopping training.")
                        print(f"\n❌ Training stopped due to NaN (recovery limit reached)")
                        raise ValueError(f"Training failed due to NaN after {nan_recovery_count} recovery attempts")
                
                loss.backward()
                
                # Gradient clipping (helps prevent NaN)
                torch.nn.utils.clip_grad_norm_(calibrator.parameters(), max_norm=1.0)
                if optimize_astrometry:
                    torch.nn.utils.clip_grad_norm_([dx, dy], max_norm=10.0)  # Clip dx/dy too
                
                # Update calibrator and optionally dx/dy
                optimizer.step()
                
                epoch_train_loss += loss.item()
                train_batch_count += 1
                
                # Log batch loss
                loss_stats['train_losses_per_batch'].append(loss.item())
        
                # Running loss info during wavelength iteration
          
                current_lr = optimizer.param_groups[0]['lr']

                print(f"\rλ set {current_λ_set_id + 1}/{len(λ_sets)} complete, batch {batch_idx + 1}/{len(train_loader)} | "
                      f"Running loss: {loss.item():.6f} | LR: {current_lr:.2e}", end='', flush=True)

                logger.debug(f"Epoch {epoch}, λ set {current_λ_set_id + 1}/{len(λ_sets)}, "
                           f"batch {batch_idx + 1}/{len(train_loader)}: train_loss = {loss.item():.6f}")
            
            # If NaN was detected, break out of wavelength loop too
            if nan_detected_this_epoch:
                break
        
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
        # If NaN was detected, skip to next epoch
        if nan_detected_this_epoch:
            logger.warning(f"Epoch {epoch} skipped due to NaN recovery")
            continue
        
        avg_train_loss = epoch_train_loss / train_batch_count if train_batch_count > 0 else 0
        train_losses.append(avg_train_loss)
        loss_stats['train_losses_per_epoch'].append(avg_train_loss)
        
        logger.info(f"Epoch {epoch}: Average training loss = {avg_train_loss:.6f}")
        
        
        # ========== VALIDATION ==========
        # Option 1: Validate with dx/dy optimization
        val_loss = validate_with_astrometry_optimization(num_opt_steps=val_dx_dy_opt_steps, lr_dx_dy=val_dx_dy_lr)
        
        # Option 2: Simple validation
        # val_loss = validate_fixed()
        
        # Check for NaN in validation
        if np.isnan(val_loss) or np.isinf(val_loss):
            logger.error(f"❌ NaN/Inf in validation loss at epoch {epoch}")
            if nan_recovery and nan_recovery_count < max_nan_recoveries:
                nan_recovery_count += 1
                logger.warning(f"Attempting recovery from validation NaN...")
                load_checkpoint(calibrator, dx, dy, optimizer, WEIGHTS_FOLDER / 'NFM_calibrator_new/best_calibrator_checkpoint.pth')
                continue
            else:
                raise ValueError("NaN in validation loss")
        
        val_losses.append(val_loss)
        loss_stats['val_losses_per_epoch'].append(val_loss)
        
        logger.info(f"Epoch {epoch}: Validation loss = {val_loss:.6f}")
        
        # Learning rate scheduling
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        if old_lr != new_lr:
            logger.warning(f"Learning rate reduced: {old_lr:.2e} -> {new_lr:.2e}")
            print(f"⚠ Learning rate reduced: {old_lr:.2e} -> {new_lr:.2e}")
        
        current_lr = optimizer.param_groups[0]['lr']
        loss_stats['learning_rates'].append(current_lr)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            loss_stats['best_epoch'] = epoch
            loss_stats['best_val_loss'] = val_loss
            
            # Save best model
            save_checkpoint(calibrator, dx, dy, optimizer, epoch, avg_train_loss, val_loss, 
                            WEIGHTS_FOLDER / 'NFM_calibrator_new/best_calibrator_checkpoint.pth')
            logger.info(f"✓ Saved best model at epoch {epoch} with val_loss = {val_loss:.6f}")
            print(f"✓ Saved best model at epoch {epoch}")
        else:
            patience_counter += 1
        
        # Save periodic checkpoint every 5 epochs (for extra safety)
        if epoch % 5 == 0:
            save_checkpoint(calibrator, dx, dy, optimizer, epoch, avg_train_loss, val_loss, 
                            WEIGHTS_FOLDER/f'NFM_calibrator_new/checkpoint_epoch_{epoch}.pth')
            logger.info(f"Periodic checkpoint saved at epoch {epoch}")
        
        # Print progress
        current_lr = optimizer.param_groups[0]['lr']
        progress_msg = (f"Epoch {epoch:3d}/{num_epochs} | "
                       f"Train Loss: {avg_train_loss:.6f} | "
                       f"Val Loss: {val_loss:.6f} | "
                       f"LR: {current_lr:.2e} | "
                       f"Patience: {patience_counter}/{patience}")
        print(progress_msg)
        logger.info(progress_msg)
        
        # Early stopping check
        if patience_counter >= patience:
            logger.warning(f"Early stopping triggered at epoch {epoch}")
            print(f"\n⚠ Early stopping at epoch {epoch}")
            break
    
    # Save loss statistics to file
    loss_save_path = Path(DATA_FOLDER / 'temp')
    loss_save_path.mkdir(parents=True, exist_ok=True)
    
    np.save(loss_save_path / 'loss_stats_train.npy', np.array(loss_stats['train_losses_per_epoch']))
    np.save(loss_save_path / 'loss_stats_val.npy',   np.array(loss_stats['val_losses_per_epoch']))
    
    # Save complete loss statistics
    import pickle
    with open(loss_save_path / f'training_stats_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl', 'wb') as f:
        pickle.dump(loss_stats, f)
    
    logger.info(f"Loss statistics saved to {loss_save_path}")
    
    # Log NaN recovery statistics
    if nan_recovery_count > 0:
        logger.warning(f"⚠ Training recovered from NaN {nan_recovery_count} time(s)")
        print(f"\n⚠ Note: Training recovered from NaN {nan_recovery_count} time(s)")
    
    # Load best model
    logger.info("Loading best model...")
    print("\nLoading best model...")
    checkpoint = torch.load(WEIGHTS_FOLDER / 'NFM_calibrator_new/best_calibrator_checkpoint.pth')
    calibrator.load_state_dict(checkpoint['calibrator_state_dict'])

    if optimize_astrometry and 'dx' in checkpoint and 'dy' in checkpoint:
        dx.data = checkpoint['dx']
        dy.data = checkpoint['dy']
    
    logger.info(f"Best validation loss: {checkpoint['val_loss']:.6f} at epoch {checkpoint['epoch']}")
    print(f"Best validation loss: {checkpoint['val_loss']:.6f} at epoch {checkpoint['epoch']}")
    
    return calibrator, train_losses, val_losses


# %%
def main():
    logger.info("="*60)
    logger.info("Starting Training with Validation")
    logger.info("="*60)

    # Train the model with NaN recovery enabled
    calibrator, train_losses, val_losses = train_with_validation(
        num_epochs=500,
        patience=20,
        val_dx_dy_opt_steps=3,
        val_dx_dy_lr=1e-4,
        nan_recovery=True,
        max_nan_recoveries=10
    )

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)

    logger.info("="*60)
    logger.info("Training Complete!")
    logger.info("="*60)
    logger.info(f"Final train loss: {train_losses[-1]:.6f}")
    logger.info(f"Final val loss: {val_losses[-1]:.6f}")
    logger.info(f"Best val loss: {min(val_losses):.6f}")

    # Plot losses (optional)
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Losses')
        plt.grid(True)
        plt.show()
    except:
        pass


if __name__ == "__main__":
    main()

# %%
# Load best model
logger.info("Loading best model...")
print("\nLoading best model...")
checkpoint = torch.load(WEIGHTS_FOLDER / 'NFM_calibrator_new/best_calibrator_checkpoint.pth')
calibrator.load_state_dict(checkpoint['calibrator_state_dict'])

if optimize_astrometry and 'dx' in checkpoint and 'dy' in checkpoint:
    dx.data = checkpoint['dx']
    dy.data = checkpoint['dy']

logger.info(f"Best validation loss: {checkpoint['val_loss']:.6f} at epoch {checkpoint['epoch']}")
print(f"Best validation loss: {checkpoint['val_loss']:.6f} at epoch {checkpoint['epoch']}")


# Validate and collect all PSF predictions and data
PSFs_pred, PSFs_data, validation_ids, NN_predictions, final_val_loss = validate_fixed()

print(f"\n{'='*60}")
print(f"Validation Results:")
print(f"{'='*60}")
print(f"Final validation loss (fixed dx/dy): {final_val_loss:.6f}")
print(f"PSF predictions shape: {PSFs_pred.shape}")
print(f"PSF data shape: {PSFs_data.shape}")
print(f"Validation sample IDs shape: {validation_ids.shape}")
print(f"Validation IDs range: [{validation_ids.min()}, {validation_ids.max()}]")
print(f"{'='*60}\n")

logger.info(f"Final validation loss (fixed dx/dy): {final_val_loss:.6f}")
logger.info(f"PSF predictions collected: {PSFs_pred.shape}")
logger.info(f"PSF data collected: {PSFs_data.shape}")
logger.info(f"Validation sample IDs collected: {validation_ids.shape}")

# Optionally save the cubes for later analysis
# torch.save(PSFs_pred, WEIGHTS_FOLDER / 'NFM_calibrator_new/validation_PSFs_predicted.pt')
# torch.save(PSFs_data, WEIGHTS_FOLDER / 'NFM_calibrator_new/validation_PSFs_data.pt')
# torch.save(validation_ids, WEIGHTS_FOLDER / 'NFM_calibrator_new/validation_sample_ids.pt')
# torch.save(PSFs_data, WEIGHTS_FOLDER / 'NFM_calibrator_new/validation_PSFs_data.pt')

PSFs_pred = PSFs_pred.cpu()
PSFs_data = PSFs_data.cpu()
validation_ids = validation_ids.cpu()

#%%
from tools.plotting import plot_radial_PSF_profiles, draw_PSF_stack

id_src = np.random.randint(0, PSFs_data.shape[0])
print(f"Randomly selected validation sample ID: {id_src}")
print(f"Corresponding original dataset index: {validation_ids[id_src].item()}")

PSF_0 = PSFs_data[id_src]
PSF_1 = PSFs_pred[id_src]

# NN_pred = NN_predictions[id_src].cpu().unsqueeze(0).numpy()
# x_pred = inputs_transformer.unstack(NN_pred)

vmin = np.percentile(PSF_0[PSF_0 > 0].cpu().numpy(), 10)
vmax = np.percentile(PSF_0[PSF_0 > 0].cpu().numpy(), 99.995)
wvl_select = np.s_[0, N_wvl_total//2, -1]

draw_PSF_stack(
    PSF_0.numpy()[wvl_select, ...],
    PSF_1.numpy()[wvl_select, ...],
    average=True,
    min_val=vmin,
    max_val=vmax,
    crop=100
)

#%%
PSF_disp = lambda x, w: (x[w,...]).cpu().numpy()

fig, ax = plt.subplots(1, len(wvl_select), figsize=(10, len(wvl_select)))
for i, lmbd in enumerate(wvl_select):
    plot_radial_PSF_profiles(
        PSF_disp(PSF_0, lmbd),
        PSF_disp(PSF_1, lmbd),
        'Data',
        'TipTorch',
        cutoff=40,
        y_min=3e-2,
        linthresh=1e-2,
        return_profiles=True,
        ax=ax[i]        
    )
plt.show()

#%%
wvl_select = np.s_[0, N_wvl_total//2, -1]

fig, ax = plt.subplots(1, len(wvl_select), figsize=(10, len(wvl_select)))
for i, lmbd in enumerate(wvl_select):
    plot_radial_PSF_profiles(
        PSFs_data[:, lmbd, ...].cpu().numpy(),
        PSFs_pred[:, lmbd, ...].cpu().numpy(),
        'Data',
        'TipTorch',
        cutoff=40,
        ax=ax[i]
        #, centers=center)
    )
plt.show()

#%%
fig = plt.figure(figsize=(10, 6))
plt.title('Polychromatic PSF')
PSF_avg = lambda x: np.mean(x.cpu().numpy(), axis=1)
plot_radial_PSF_profiles( PSF_avg(PSFs_data), PSF_avg(PSFs_pred), 'Data', 'TipTorch', cutoff=40, ax=fig.add_subplot(111) )
plt.show()


#%% ============================================================================================================================================================================================
# Dummy optimization routine for debugging loss function and optimization behavior
def debug_dummy_optimization(num_iters=100, initial_guess=None):
    """
    Short debugging optimization loop that directly optimizes model inputs.
    
    Args:
        num_iters: Number of optimization iterations
        use_calibrator: If True, uses calibrator predictions as starting point; 
                       if False, optimizes dummy vector directly
    """
    logger.info("=" * 40)
    logger.info("DUMMY OPTIMIZATION DEBUG")
    logger.info("=" * 40)
    
    # Use first batch for debugging
    dummy_batch = list(val_loader)[0]
    PSF_cubes, _, batch_config, idxs = dummy_batch
    
    logger.info(f"Debug batch size: {PSF_cubes.shape[0]}")
    logger.info(f"PSF cube shape: {PSF_cubes.shape}")
    
    # Use single wavelength set for faster debugging
    current_λ_set_id = 0
    PSF_model.SetWavelengths(λ_sets[current_λ_set_id].to(device=device))
    λ_id_set = λ_id_sets[current_λ_set_id]
    
    logger.info(f"Using wavelength set {current_λ_set_id}: {len(λ_id_set)} wavelengths")
    
    if initial_guess is not None:
        # Start with calibrator prediction and add noise
        with torch.no_grad():
            dummy_vec = initial_guess.detach().clone().to(device=device)
            dummy_vec.requires_grad_(True)
        logger.info("Using initial guess")
    else:
        # Random initialization
        dummy_vec = torch.randn((PSF_cubes.shape[0], N_outputs), device=device, requires_grad=True)
        logger.info("Using random initialization")
    
    # Setup optimizer and scheduler
    if optimize_astrometry:
        optimizer = optim.Adam([
            {'params': [dx], 'lr': 1e-3},
            {'params': [dy], 'lr': 1e-3},
            {'params': [dummy_vec], 'lr': 1e-1}
        ], weight_decay=1e-4)
    else:
        optimizer = optim.Adam([
            {'params': [dummy_vec], 'lr': 1e-1}
        ], weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=10, min_lr=1e-6, verbose=True
    )
    
    # Loss tracking
    losses, grad_norms, param_norms = [], [], []
    
    # Initial evaluation
    with torch.no_grad():
        x_pred = inputs_transformer.unstack(dummy_vec)
        PSF_pred = run_model(x_pred, batch_config, idxs, λ_id_set)
        initial_loss = criterion(PSF_pred, PSF_cubes[:, λ_id_set, ...], x_pred, λ_id_set)
        logger.info(f"Initial loss: {initial_loss.item():.6f}")
    
    logger.info(f"Starting optimization for {num_iters} iterations...")
    
    
    # Optimization loop
    for epoch in range(num_iters):
        optimizer.zero_grad()
        
        # Forward pass
        x_pred = inputs_transformer.unstack(dummy_vec)
        PSF_pred = run_model(x_pred, batch_config, idxs, λ_id_set)
        loss = criterion(PSF_pred, PSF_cubes[:, λ_id_set, ...], x_pred, λ_id_set)
        
        # Check for NaN
        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(f"❌ NaN/Inf loss at iteration {epoch}: {loss.item()}")
            break
        
        # Backward pass
        loss.backward()
        
        # Gradient monitoring
        dummy_grad_norm = torch.nn.utils.clip_grad_norm_(dummy_vec, max_norm=1.0)
        if optimize_astrometry:
            dx_dy_grad_norm = torch.nn.utils.clip_grad_norm_([dx, dy], max_norm=10.0)
        else:
            dx_dy_grad_norm = torch.tensor(0.0)  # No dx/dy gradients
        
        # Optimizer step
        optimizer.step()
        
        # Learning rate scheduling
        if epoch % 10 == 0:
            scheduler.step(loss)
        
        # Record metrics
        losses.append(loss.item())
        grad_norms.append(dummy_grad_norm.item())
        param_norms.append(dummy_vec.norm().item())
        
        # Progress logging
        if (epoch + 1) % 20 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            improvement = (initial_loss.item() - loss.item()) / initial_loss.item() * 100
            
            progress_msg = (f"Iter {epoch+1:4d}/{num_iters} | "
                           f"Loss: {loss.item():.6f} | "
                           f"Improve: {improvement:+.2f}% | "
                           f"LR: {current_lr:.2e} | "
                           f"Grad: {dummy_grad_norm:.3f}")
            print(progress_msg)
            logger.info(progress_msg)
    
    # Final evaluation
    with torch.no_grad():
        x_pred = inputs_transformer.unstack(dummy_vec)
        PSF_pred = run_model(x_pred, batch_config, idxs, λ_id_set)
        final_loss = criterion(PSF_pred, PSF_cubes[:, λ_id_set, ...], x_pred, λ_id_set)
        
        total_improvement = (initial_loss.item() - final_loss.item()) / initial_loss.item() * 100
        
        logger.info("=" * 40)
        logger.info("DUMMY OPTIMIZATION RESULTS")
        logger.info("=" * 40)
        logger.info(f"Initial loss:    {initial_loss.item():.6f}")
        logger.info(f"Final loss:      {final_loss.item():.6f}")
        logger.info(f"Total improvement: {total_improvement:.2f}%")
        logger.info(f"Iterations:      {len(losses)}")
        logger.info(f"Final param norm: {dummy_vec.norm().item():.6f}")
    
    # Simple visualization
    if len(losses) > 1:
        try:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            # Loss curve
            axes[0].plot(losses)
            axes[0].set_title('Loss Evolution')
            axes[0].set_xlabel('Iteration')
            axes[0].set_ylabel('Loss')
            axes[0].set_yscale('log')
            axes[0].grid(True)
            
            # Gradient norms
            axes[1].plot(grad_norms)
            axes[1].set_title('Gradient Norm')
            axes[1].set_xlabel('Iteration')
            axes[1].set_ylabel('Grad Norm')
            axes[1].set_yscale('log')
            axes[1].grid(True)
            
            # Parameter norms
            axes[2].plot(param_norms)
            axes[2].set_title('Parameter Norm')
            axes[2].set_xlabel('Iteration')
            axes[2].set_ylabel('Param Norm')
            axes[2].grid(True)
            
            plt.tight_layout()
            plt.show()
                        
        except Exception as e:
            logger.warning(f"Could not create plots: {e}")
    
    # Cleanup
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    return {
        'losses': losses,
        'grad_norms': grad_norms,
        'param_norms': param_norms,
        'initial_loss': initial_loss.item(),
        'final_loss': final_loss.item(),
        'improvement_percent': total_improvement,
        'dummy_vec': dummy_vec.detach().clone(),
        'final_x_pred': x_pred,
        'PSF_pred': PSF_pred.detach().clone(),
        'PSF_cubes': PSF_cubes.detach().clone()
    }



#%%
# results = debug_dummy_optimization(num_iters=100, initial_guess=dummy_vec)
results = debug_dummy_optimization(num_iters=1000)

#%%
with torch.no_grad():
    PSFs_data = results['PSF_cubes'].cpu()
    PSFs_pred = results['PSF_pred'].cpu()

    x_pred = results['final_x_pred']
    N_wvl_temp = PSFs_data.shape[1]

    dummy_vec = results['dummy_vec'].cpu()
    x_test = inputs_transformer.unstack(dummy_vec)


#%%
from tools.plotting import plot_radial_PSF_profiles, draw_PSF_stack

id_src = np.random.randint(0, PSFs_data.shape[0])

PSF_0 = PSFs_data[id_src]
PSF_1 = PSFs_pred[id_src]

vmin = np.percentile(PSF_0[PSF_0 > 0].cpu().numpy(), 10)
vmax = np.percentile(PSF_0[PSF_0 > 0].cpu().numpy(), 99.995)
wvl_select = np.s_[0, N_wvl_total//2, -1]

draw_PSF_stack(
    PSF_0.numpy()[np.s_[0, 16, 28], ...],
    PSF_1.numpy()[np.s_[0,  4,  7], ...],
    average=True,
    min_val=vmin,
    max_val=vmax,
    crop=100
)

#%%
centers_test = []
# centers_new = np.stack(centers_test).transpose((1,0,2)) - np.array(PSF_0.shape[-1])//2
_,_,_,idx = list(val_loader)[0]

# dx_new = dx[idx][:, [0, 16, 28]].detach().cpu().numpy()
# dy_new = dy[idx][:, [0, 16, 28]].detach().cpu().numpy()

# dx_dy = np.stack((dx_new, dy_new), axis=-1)

fig, ax = plt.subplots(1, 3, figsize=(10, 3))
for ax_idx, (i, j) in enumerate(zip([0, 16, 28], [0, 4, 7])):
    p_0, p_1, p_err, centers = plot_radial_PSF_profiles(
        PSFs_data[:,i,...].cpu().numpy(),
        PSFs_pred[:,j,...].cpu().numpy(),
        'Data',
        'TipTorch',
        cutoff=40,
        y_min=3e-2,
        linthresh=1e-2,
        return_profiles=True,
        ax=ax[ax_idx],
    )
    centers_test.append(centers)

plt.show()

# %%
