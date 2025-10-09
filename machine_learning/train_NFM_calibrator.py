#%%
%reload_ext autoreload
%autoreload 2

import sys
import gc
import logging
from datetime import datetime

from tests.MUSE_quasars import PSFs_pred

sys.path.insert(0, '..')

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split
from sklearn.model_selection import train_test_split, KFold
from pathlib import Path
from project_settings import *
from torch.utils.data import Dataset

from managers.config_manager import MultipleTargetsInDifferentObservations

MUSE_DATA_FOLDER = Path(project_settings["MUSE_data_folder"])
STD_FOLDER     = MUSE_DATA_FOLDER / 'standart_stars/'
DATASET_CACHE  = STD_FOLDER / 'dataset_cache'

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

# from data_processing.MUSE_STD_dataset_utils import *

N_wvl_total = 30
batch_size = 16

#%%
class NFMDataset(Dataset):
    def __init__(self):
        self.PSF_cubes  = np.load(DATASET_CACHE / 'muse_STD_stars_PSFs.npy',      mmap_mode="r")
        self.telemetry  = np.load(DATASET_CACHE / 'muse_STD_stars_telemetry.npy', mmap_mode="r")
        self.configs = torch.load(DATASET_CACHE / 'muse_STD_stars_configs.pt')
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
        device          = device
    )
    PSF_model.inputs_manager.delete('Jxy')
    PSF_model.inputs_manager.delete('bg_ctrl')
    PSF_model.inputs_manager.delete('dx_ctrl')
    PSF_model.inputs_manager.delete('dy_ctrl')

    print(PSF_model.inputs_manager)
    N_outputs = PSF_model.inputs_manager.get_stacked_size()

    inputs_transformer = PSF_model.inputs_manager.get_transformer()

gc.collect()
torch.cuda.empty_cache()

# %%
λ_full = PSF_model.wavelengths.clone().cpu()
λ_id_sets = [
    [0, 5, 10, 15, 20, 25, 29],
    [1, 6, 11, 16, 21, 26, 29],
    [2, 7, 12, 17, 22, 27, 29],
    [0, 3,  8, 13, 18, 23, 28],
    [0, 4,  9, 14, 19, 24, 28]
]
λ_sets = [λ_full[list_id] for list_id in λ_id_sets]


#%%
dx = torch.zeros((len(NFM_dataset), N_wvl_total), device=device, dtype=torch.float32, requires_grad=True)
dy = torch.zeros((len(NFM_dataset), N_wvl_total), device=device, dtype=torch.float32, requires_grad=True)
# b  = torch.zeros((len(NFM_dataset), N_wvl_total), device=device, dtype=torch.float32)

#%%
# def run_model(predicted_inputs_vec, config, idx, λ_ids):
def run_model(x_dict, config, idx, λ_ids):
    # Unpacking to dictionary format which is understood by PSF_model
    # x_dict = inputs_transformer.unstack(predicted_inputs_vec)
    
    # Set dx/dy for this batch (indexed by sample IDs)
    x_dict['dx'] = dx[idx[:,None], λ_ids]
    x_dict['dy'] = dy[idx[:,None], λ_ids]

    # Update simulated wavelengths
    config['sources_science']['Wavelength'] = λ_sets[λ_ids].unsqueeze(0).to(device=device)
    
    # Update internal state of the PSF model for the given batch config
    PSF_model.model.Update(config=config, init_grids=False, init_pupils=False, init_tomography=True) # Update just AO + model parameters, not grids
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
    drop_last=True
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    # pin_memory=True if torch.cuda.is_available() else False,
    collate_fn=lambda batch: collate_batch(batch, device=device),
    drop_last=True
)

# Initialize calibrator NN
calibrator = SmallCalibratorNet(
    n_features=N_features,
    n_outputs=N_outputs,
    hidden_dim=32,
    dropout_rate=0.3
).to(device)


# Optimizer with weight decay (L2 regularization)
optimizer = optim.AdamW([
    {'params': calibrator.parameters(), 'lr': 1e-3, 'weight_decay': 1e-3},
    {'params': [dx, dy], 'lr': 1e-3, 'weight_decay': 1e-4}  # Lower weight decay for dx/dy
], lr=1e-3, weight_decay=1e-3)  # Default values

# More aggressive scheduler for 50 epochs: reduce LR every 3-5 epochs without improvement
# Key: scheduler patience (3) << early stopping patience (10) so LR actually reduces during training
# Expected LR schedule over 50 epochs: 1e-3 -> 5e-4 -> 2.5e-4 -> 1.25e-4 -> ... (every ~3-4 epochs of plateau)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.5,      # Reduce by half
    patience=3,      # Wait only 3 epochs (much less than early stopping patience)
    verbose=True,
    min_lr=1e-6      # Don't go below this
)

logger.info(f"Scheduler: ReduceLROnPlateau with patience={3}, factor={0.5}, min_lr={1e-6}")


# Loss function
wvl_weights = torch.linspace(1.0, 0.5, N_wvl_total).to(device).view(1, N_wvl_total, 1, 1)
wvl_weights = N_wvl_total / wvl_weights.sum() * wvl_weights # Normalize so that the total energy is preserved

# logger.info(f"Wavelength weights range: {wvl_weights.min().item():.4f} - {wvl_weights.max().item():.4f}")


def loss_LO_fn(LO_coefs, w_L2=1e-7, w_first=5e-5):
    coefs_L2_penalty = LO_coefs.pow(2).sum(-1).mean() * w_L2
    first_coef_penalty = torch.clamp(-LO_coefs[:, 0], min=0).pow(2).mean() * w_first
    return coefs_L2_penalty + first_coef_penalty


def loss_fn(PSF_pred, PSF_data, coefs, w_MSE, w_MAE, w_L2, w_first, λ_ids):    
    diff = (PSF_pred-PSF_data) * wvl_weights[0, λ_ids, ...]
    w = 2e4
    MSE_loss = diff.pow(2).mean() * w * w_MSE
    MAE_loss = diff.abs().mean()  * w * w_MAE
    LO_loss  = loss_LO_fn(coefs, w_L2=w_L2, w_first=w_first) if PSF_model.LO_NCPAs else 0.0
    
    # Add safety check for NaN in loss components (helps debug NaN sources)
    if torch.isnan(MSE_loss) or torch.isnan(MAE_loss):
        logger.error(f"NaN in loss components - MSE: {MSE_loss.item()}, MAE: {MAE_loss.item()}, LO: {LO_loss}")
        logger.error(f"PSF_pred range: [{PSF_pred.min().item()}, {PSF_pred.max().item()}]")
        logger.error(f"PSF_data range: [{PSF_data.min().item()}, {PSF_data.max().item()}]")

    return MSE_loss + MAE_loss + LO_loss


criterion = lambda pred, data, coefs, λ_ids: loss_fn(pred, data, coefs, w_MSE=800.0, w_MAE=1.6, w_L2=1e-7, w_first=5e-5, λ_ids=λ_ids)

# L1 regularization of NN parameters
def l1_regularization(model, lambda_l1=1e-4):
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    return lambda_l1 * l1_norm

#%%

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
            coefs = x_pred['LO_coefs']

            PSF_pred = run_model(x_pred, batch_config, idxs, current_λ_set_id)

            # Select only the relevant wavelengths
            loss = criterion(PSF_pred, PSF_cubes[:, λ_id_set, ...], coefs, λ_id_set)

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
    - dx/dy values for validation samples are OPTIMIZED
    
    Args:
        num_opt_steps: Number of optimization steps per validation sample
        lr_dx_dy: Learning rate for dx/dy optimization
    
    Returns:
        average validation loss after optimizing dx/dy
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
                coefs = x_pred['LO_coefs']

                PSF_pred = run_model(x_pred, batch_config, idxs, current_λ_set_id)
                loss = criterion(PSF_pred, PSF_cubes[:, λ_id_set, ...], coefs, λ_id_set)
                
                loss.backward() # Only dx/dy get gradients
                
                # torch.nn.utils.clip_grad_norm_([dx, dy], max_norm=1.0)
    
                val_dx_dy_optimizer.step() # Update ONLY dx/dy
            
            # Record final loss for this batch
            with torch.no_grad():
                NN_output = calibrator(telemetry_inputs)
                x_pred = inputs_transformer.unstack(NN_output)
                coefs = x_pred['LO_coefs']

                PSF_pred = run_model(x_pred, batch_config, idxs, current_λ_set_id)
                final_loss = criterion(PSF_pred, PSF_cubes[:, λ_id_set, ...], coefs, λ_id_set)

                print(f"\rλ set {current_λ_set_id + 1}/{len(λ_sets)}, batch {batch_idx + 1}: val_loss = {final_loss.item():.6f}", end='', flush=True)
                logger.debug(f"Validation: λ set {current_λ_set_id + 1}/{len(λ_sets)}, batch {batch_idx + 1}: val_loss = {final_loss.item():.6f}")
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
        PSFs_pred_cube: Tensor of shape (N_val_samples, N_wvl_total, H, W) with all predicted PSFs
        PSFs_data_cube: Tensor of shape (N_val_samples, N_wvl_total, H, W) with all data PSFs
        avg_loss: Average validation loss
    """
    calibrator.eval()
    total_loss = 0
    num_batches = 0
    
    # Get dimensions from first batch to initialize the output cubes
    first_batch = next(iter(val_loader))
    PSF_cubes_sample, _, _, _ = first_batch
    N_val_samples = len(val_dataset)
    _, H, W = PSF_cubes_sample.shape[1:]  # Get H, W from (batch, C, H, W)
    
    # Initialize output cubes: (N_val_samples, N_wvl_total, H, W)
    # We'll fill these in as we process different wavelength sets
    PSFs_pred_cube = torch.zeros((N_val_samples, N_wvl_total, H, W), dtype=torch.float32, device='cpu')
    PSFs_data_cube = torch.zeros((N_val_samples, N_wvl_total, H, W), dtype=torch.float32, device='cpu')
    
    logger.info(f"Initializing PSF cubes: predictions and data, shape: {PSFs_pred_cube.shape}")
    
    with torch.no_grad():  # No gradients at all
        for current_λ_set_id in range(len(λ_sets)):
            PSF_model.SetWavelengths(λ_sets[current_λ_set_id].to(device=device))
            λ_id_set = λ_id_sets[current_λ_set_id]  # Which wavelength indices for this set
            
            batch_start_idx = 0
            
            for batch_idx, batch in enumerate(val_loader):
                PSF_cubes, telemetry_inputs, batch_config, idxs = batch
                
                NN_output = calibrator(telemetry_inputs)
                x_pred = inputs_transformer.unstack(NN_output)
                coefs = x_pred['LO_coefs']

                PSF_pred = run_model(x_pred, batch_config, idxs, current_λ_set_id)

                loss = criterion(PSF_pred, PSF_cubes[:, λ_id_set, ...], coefs, λ_id_set)
                total_loss += loss.item()
                num_batches += 1
                
                # Store predictions and data in the cubes
                # PSF_pred shape: (batch_size, n_wavelengths_in_set, H, W)
                # PSF_cubes shape: (batch_size, N_wvl_total, H, W) - but we only need λ_id_set wavelengths
                batch_size = PSF_pred.shape[0]
                batch_end_idx = batch_start_idx + batch_size
                
                # Fill in the wavelengths for this batch
                for wvl_idx, λ_id in enumerate(λ_id_set):
                    PSFs_pred_cube[batch_start_idx:batch_end_idx, λ_id, :, :] = PSF_pred[:, wvl_idx, :, :].cpu()
                    PSFs_data_cube[batch_start_idx:batch_end_idx, λ_id, :, :] = PSF_cubes[:, λ_id, :, :].cpu()
                
                batch_start_idx = batch_end_idx
                
                print(f"\rλ set {current_λ_set_id + 1}/{len(λ_sets)}, batch {batch_idx + 1}/{len(val_loader)}", end='', flush=True)

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    calibrator.train()
    
    print()  # New line after progress
    logger.info(f"PSFs_pred cube shape: {PSFs_pred_cube.shape}")
    logger.info(f"PSFs_data cube shape: {PSFs_data_cube.shape}")
    logger.info(f"PSF pred range: [{PSFs_pred_cube.min():.4e}, {PSFs_pred_cube.max():.4e}]")
    logger.info(f"PSF data range: [{PSFs_data_cube.min():.4e}, {PSFs_data_cube.max():.4e}]")

    return PSFs_pred_cube, PSFs_data_cube, total_loss / num_batches if num_batches > 0 else 0.0


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
    torch.save({
        'epoch': epoch,
        'calibrator_state_dict': calibrator.state_dict(),
        'dx': dx.detach().clone(),
        'dy': dy.detach().clone(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }, path)
    logger.debug(f"Checkpoint saved to {path}")


def load_checkpoint(calibrator, dx, dy, optimizer, path='checkpoint.pth'):
    """Load a training checkpoint and restore model state."""
    checkpoint = torch.load(path)
    calibrator.load_state_dict(checkpoint['calibrator_state_dict'])
    dx.data = checkpoint['dx']
    dy.data = checkpoint['dy']
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
                coefs = x_pred['LO_coefs']

                PSF_pred = run_model(x_pred, batch_config, idxs, current_λ_set_id)

                loss = criterion(PSF_pred, PSF_cubes[:, λ_id_set, ...], coefs, λ_id_set)
                
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
                            
                            # Reduce learning rate aggressively
                            for param_group in optimizer.param_groups:
                                old_lr = param_group['lr']
                                param_group['lr'] = old_lr * 0.1
                                logger.warning(f"Reduced LR: {old_lr:.2e} -> {param_group['lr']:.2e}")
                            
                            print(f"✓ Recovered from checkpoint, LR reduced by 10x")
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
                torch.nn.utils.clip_grad_norm_([dx, dy], max_norm=10.0)  # Clip dx/dy too
                
                # Update both calibrator and dx/dy
                optimizer.step()
                
                epoch_train_loss += loss.item()
                train_batch_count += 1
                
                # Log batch loss
                loss_stats['train_losses_per_batch'].append(loss.item())
        
                # Running loss info during wavelength iteration
          
                current_lr = optimizer.param_groups[0]['lr']
                
                print(f"\rλ set {current_λ_set_id + 1}/{len(λ_sets)} complete | "
                      f"Running loss: {loss.item():.6f} | LR: {current_lr:.2e}", end='', flush=True)

                logger.debug(f"Epoch {epoch}, λ set {current_λ_set_id + 1}/{len(λ_sets)}, "
                           f"batch {batch_idx + 1}: train_loss = {loss.item():.6f}")
            
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
    np.save(loss_save_path / 'loss_stats_val.npy', np.array(loss_stats['val_losses_per_epoch']))
    
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
    dx.data = checkpoint['dx']
    dy.data = checkpoint['dy']
    
    logger.info(f"Best validation loss: {checkpoint['val_loss']:.6f} at epoch {checkpoint['epoch']}")
    print(f"Best validation loss: {checkpoint['val_loss']:.6f} at epoch {checkpoint['epoch']}")
    
    return calibrator, train_losses, val_losses


#%%
logger.info("="*60)
logger.info("Starting Training with Validation")
logger.info("="*60)

# Train the model with NaN recovery enabled
calibrator, train_losses, val_losses = train_with_validation(
    num_epochs=50,
    patience=5,
    val_dx_dy_opt_steps=5,
    val_dx_dy_lr=1e-3,
    nan_recovery=True,
    max_nan_recoveries=3
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
    import matplotlib.pyplot as plt
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

# %%

# Validate and collect all PSF predictions and data
PSFs_pred, PSFs_data, final_val_loss = validate_fixed()

print(f"\n{'='*60}")
print(f"Validation Results:")
print(f"{'='*60}")
print(f"Final validation loss (fixed dx/dy): {final_val_loss:.6f}")
print(f"PSF predictions shape: {PSFs_pred.shape}")
print(f"PSF data shape: {PSFs_data.shape}")
print(f"{'='*60}\n")

logger.info(f"Final validation loss (fixed dx/dy): {final_val_loss:.6f}")
logger.info(f"PSF predictions collected: {PSFs_pred.shape}")
logger.info(f"PSF data collected: {PSFs_data.shape}")

# Optionally save the cubes for later analysis
# torch.save(PSFs_pred, WEIGHTS_FOLDER / 'NFM_calibrator_new/validation_PSFs_predicted.pt')
# torch.save(PSFs_data, WEIGHTS_FOLDER / 'NFM_calibrator_new/validation_PSFs_data.pt')


PSFs_pred = PSFs_pred.cpu()
PSFs_data = PSFs_data.cpu()

#%%
from tools.plotting import plot_radial_PSF_profiles, draw_PSF_stack

id_src = np.random.randint(0, PSFs_data.shape[0])
print(f"Randomly selected validation sample ID: {id_src}")

PSF_0 = PSFs_data[id_src]
PSF_1 = PSFs_pred[id_src]

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

# if PSF_model.LO_NCPAs:
#     plt.imshow(OPD_map[id_src,...]*1e9)
#     plt.colorbar()
#     plt.show()
