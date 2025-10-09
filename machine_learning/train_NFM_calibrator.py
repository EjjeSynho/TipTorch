#%%
%reload_ext autoreload
%autoreload 2

import sys
import gc

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

# from data_processing.MUSE_STD_dataset_utils import *

N_wvl_total = 31
batch_size = 4

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
# current_λ_set_id = 2

# PSF_model.SetWavelengths(λ_sets[current_λ_set_id].to(device=device))
# calibrator = lambda x: x @ torch.ones([N_features, N_outputs], device=device, dtype=torch.float32)

# configs_ = MultipleTargetsInDifferentObservations(configs, device=device)
# configs_['telescope']['PupilAngle'] = 0.0

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


# for epoch in range(2):
#     for λ_id in range(len(λ_sets)):
#         PSF_model.SetWavelengths(λ_sets[λ_id].to(device=device))

#         for batch in range(5):
#             PSF_1 = run_model(vecs, configs_, idxs, λ_id)
        
#         torch.cuda.empty_cache()
#         torch.cuda.synchronize()


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

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

# Loss function
wvl_weights = torch.linspace(1.0, 0.5, N_wvl_total).to(device).view(1, N_wvl_total, 1, 1)
wvl_weights = N_wvl_total / wvl_weights.sum() * wvl_weights # Normalize so that the total energy is preserved


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
            
            PSF_pred = run_model(x_pred, batch_config, idxs, current_λ_set_id)

            # Select only the relevant wavelengths
            loss = criterion(PSF_pred, PSF_cubes[:, λ_id_set, ...], λ_id_set)

            loss.backward() # gradients flow to both calibrator and dx/dy
            # print(loss.item())

            # Gradient clipping (prevents explosion if there is an outlier in the batch)
            torch.nn.utils.clip_grad_norm_(calibrator.parameters(), max_norm=1.0)
            
            optimizer.step()
            
        # gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        print(f"Epoch {epoch}, λ set #{current_λ_set_id}: Loss = {loss.item():.6f}")
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

                PSF_pred = run_model(x_pred, batch_config, idxs, current_λ_set_id)
                loss = criterion(PSF_pred, PSF_cubes[:, λ_id_set, ...], λ_id_set)
                
                loss.backward() # Only dx/dy get gradients
                
                # torch.nn.utils.clip_grad_norm_([dx, dy], max_norm=1.0)
    
                val_dx_dy_optimizer.step() # Update ONLY dx/dy
            
            # Record final loss for this batch
            with torch.no_grad():
                NN_output = calibrator(telemetry_inputs)
                x_pred = inputs_transformer.unstack(NN_output)

                PSF_pred = run_model(x_pred, batch_config, idxs, current_λ_set_id)
                final_loss = criterion(PSF_pred, PSF_cubes[:, λ_id_set, ...], λ_id_set)
                
                print(f"λ set {current_λ_set_id + 1}/{len(λ_sets)}, batch {batch_idx + 1}: val_loss = {final_loss.item():.6f}")
                
                total_loss += final_loss.item()
                num_batches += 1

        # Clear cache between wavelength sets
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
    """
    calibrator.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():  # No gradients at all
        for current_λ_set_id in range(len(λ_sets)):
            PSF_model.SetWavelengths(λ_sets[current_λ_set_id].to(device=device))
            λ_id_set = λ_id_sets[current_λ_set_id]
            
            for batch in val_loader:
                PSF_cubes, telemetry_inputs, batch_config, idxs = batch
                
                NN_output = calibrator(telemetry_inputs)
                x_pred = inputs_transformer.unstack(NN_output)
                
                PSF_pred = run_model(x_pred, batch_config, idxs, current_λ_set_id)

                loss = criterion(PSF_pred, PSF_cubes[:, λ_id_set, ...], λ_id_set)
                total_loss += loss.item()
                num_batches += 1
            
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    calibrator.train()
    return total_loss / num_batches if num_batches > 0 else 0.0


# Complete training loop with validation
def train_with_validation(num_epochs=50, patience=10, val_dx_dy_opt_steps=20, val_dx_dy_lr=1e-3):
    """
    Training loop with validation where dx/dy are optimized during validation.
    """
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []


    for epoch in range(num_epochs):
        # ========== TRAINING ==========
        calibrator.train()
        epoch_train_loss = 0
        train_batch_count = 0
        
        for current_λ_set_id in range(len(λ_sets)):
            PSF_model.SetWavelengths(λ_sets[current_λ_set_id].to(device=device))
            λ_id_set = λ_id_sets[current_λ_set_id]
            
            for batch_idx, batch in enumerate(train_loader):
                PSF_cubes, telemetry_inputs, batch_config, idxs = batch
                
                optimizer.zero_grad()
                
                # Forward pass
                NN_output = calibrator(telemetry_inputs)
                x_pred = inputs_transformer.unstack(NN_output)

                PSF_pred = run_model(x_pred, batch_config, idxs, current_λ_set_id)

                loss = criterion(PSF_pred, PSF_cubes[:, λ_id_set, ...], λ_id_set)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(calibrator.parameters(), max_norm=1.0)
                
                # Update both calibrator and dx/dy
                optimizer.step()
                
                epoch_train_loss += loss.item()
                train_batch_count += 1
        
                # Running loss info during wavelength iteration
          
                current_lr = optimizer.param_groups[0]['lr']
                
                print(f"\rλ set {current_λ_set_id + 1}/{len(λ_sets)} complete | "
                      f"Running loss: {loss.item():.6f} | LR: {current_lr:.2e}")#, end='', flush=True)
                
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        avg_train_loss = epoch_train_loss / train_batch_count if train_batch_count > 0 else 0
        train_losses.append(avg_train_loss)
        
        
        # ========== VALIDATION ==========
        # Option 1: Validate with dx/dy optimization
        val_loss = validate_with_astrometry_optimization(num_opt_steps=val_dx_dy_opt_steps, lr_dx_dy=val_dx_dy_lr)
        
        # Option 2: Simple validation
        # val_loss = validate_fixed()
        
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'calibrator_state_dict': calibrator.state_dict(),
                'dx': dx.detach().clone(),
                'dy': dy.detach().clone(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
            }, 'best_calibrator_checkpoint.pth')
            print(f"✓ Saved best model at epoch {epoch}")
        else:
            patience_counter += 1
        
        # Print progress
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:3d}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"LR: {current_lr:.2e} | "
              f"Patience: {patience_counter}/{patience}")
        
        # Early stopping check
        if patience_counter >= patience:
            print(f"\n⚠ Early stopping at epoch {epoch}")
            break
    
    # Load best model
    print("\nLoading best model...")
    checkpoint = torch.load('best_calibrator_checkpoint.pth')
    calibrator.load_state_dict(checkpoint['calibrator_state_dict'])
    dx.data = checkpoint['dx']
    dy.data = checkpoint['dy']
    
    print(f"Best validation loss: {checkpoint['val_loss']:.6f} at epoch {checkpoint['epoch']}")
    
    return calibrator, train_losses, val_losses


#%%
print("="*60)
print("Starting Training with Validation")
print("="*60)

# Train the model
calibrator, train_losses, val_losses = train_with_validation(
    num_epochs=50,
    patience=10,
    val_dx_dy_opt_steps=20,  # Optimize dx/dy for 20 steps during validation
    val_dx_dy_lr=1e-3
)

print("\n" + "="*60)
print("Training Complete!")
print("="*60)

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
