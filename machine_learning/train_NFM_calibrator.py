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
from copy import deepcopy

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
BEST_CALIB_PATH = WEIGHTS_FOLDER / 'NFM_calibrator/best_calibrator_checkpoint.pth'
DEBUG = True # TODO: make it an argument
BATCH_SIZE = 16 # TODO: make it an argument

pre_init_astrometry = True
optimize_astrometry = False
predict_Cn2_profile = True
predict_LO_NCPAs = True

# Set up logging
log_dir = Path('../data/logs')
log_dir.mkdir(parents=True, exist_ok=True)
log_filename = log_dir / f'training_NFM_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()  # Also log to console
    ]
)

# Silence noisy libraries
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

logger.info("="*60)
logger.info("NFM Calibrator Training Script")
logger.info("="*60)
logger.info(f"Log file: {log_filename}")

# Add argument inputs
parser = argparse.ArgumentParser(description="Train NFM Calibrator")
# parser.add_argument('--weights', type=str, default=str(BEST_CALIB_PATH), help='Path to the best calibrator weights checkpoint')
parser.add_argument('--continue-training', action='store_true', help='Whether to continue training from a checkpoint')

# Handle both command line and iPython environments
try:
    args = parser.parse_args()   #TODO: implement continue training
except SystemExit:
    # In iPython/Jupyter, parse_args() fails, so use default values
    args = argparse.Namespace(
        continue_training = True
    )

#%%
class NFMDataset(Dataset):
    def __init__(self, data_path):
        data = torch.load(data_path, weights_only=False)
        
        self.PSF_cubes   = data['PSF_cubes']
        self.telemetry   = data['telemetry']
        self.sample_ids  = data['sample_ids']
        self.configs     = data['model_configs']
        self.fitted_vals = data['fitted_param_values']
                
        self.features = list(data['telemetry'][0].keys())
        
        vals_to_spline_ctrl = { p: p+'_ctrl' for p in ['J', 'F', 'bg', 'dx', 'dy'] }
        
        for i in range(len(self.sample_ids)):
            if isinstance(self.configs[i], list):
                self.configs[i] = self.configs[i][0]  # Unwrap single-item lists
                # Rename parameter keys to their control versions
                
                for legacy_key, control_key in vals_to_spline_ctrl.items():
                    if legacy_key in self.fitted_vals[i] and control_key not in self.fitted_vals[i]:
                        self.fitted_vals[i][control_key] = self.fitted_vals[i][legacy_key]
                        del self.fitted_vals[i][legacy_key]
         
        self.N, self.H, self.W, self.C = self.PSF_cubes.shape

    def __len__(self): return self.N

    def __getitem__(self, idx):
        ''' This function accesses one sample from the dataset and converts it to tensors. '''
        conf = self.configs[idx]
        PSFs_tensor = torch.from_numpy(self.PSF_cubes[idx].astype(np.float32)).permute(2,0,1)
        telemetry_tensor = torch.from_numpy(np.array(list(self.telemetry[idx].values())).astype(np.float32))
        fitted_dict = {key: torch.tensor(value, dtype=torch.float32) for key, value in self.fitted_vals[idx].items()}
    
        return PSFs_tensor, telemetry_tensor, fitted_dict, conf, idx  # [C, H, W], [N_features], ...


def collate_batch(batch, device):
    ''' This function PSF and telemetry prepares batches to be model-ready. '''
    PSF_cubes, telemetry_vecs, fitted_vals, configs, idxs = zip(*batch)
        
    telemetry_vecs = torch.stack(telemetry_vecs, 0).to(device=device, non_blocking=True)
    PSF_cubes = torch.stack(PSF_cubes, 0).to(device=device, non_blocking=True)
    idxs = torch.tensor(idxs, dtype=torch.long, device=device)

    fitted_vals = {
        key: torch.stack([fv[key] for fv in fitted_vals], 0).to(device=device, non_blocking=True)
        for key in fitted_vals[0].keys()
    }
    
    batch_config = MultipleTargetsInDifferentObservations(configs, device=device)
    batch_config['PathPupil'] = str(DATA_FOLDER / 'calibrations/VLT_CALIBRATION/VLT_PUPIL/ut4pupil320.fits')
    batch_config['telescope']['PupilAngle'] = 0.0 #TODO: make this changable with flag?
    
    return PSF_cubes, telemetry_vecs, fitted_vals, batch_config, idxs


#%%
dataset = NFMDataset(DATASET_CACHE / 'muse_STD_stars_dataset.pt')
N_features = len(dataset.features)
N_wvl_total = dataset.C # The number of spectral channels from the PSF data

# Data loaders
train_idx, test_idx = train_test_split(np.arange(len(dataset)), test_size=0.20, random_state=42)

H, W = dataset.H, dataset.W
dx, dy = None, None

logger.info(f"Dataset loaded.")

#%%
train_dataset = Subset(dataset, train_idx)
val_dataset   = Subset(dataset, test_idx)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    # pin_memory=True if torch.cuda.is_available() else False,
    collate_fn=lambda batch: collate_batch(batch, device=device),
    drop_last=False
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    # pin_memory=True if torch.cuda.is_available() else False,
    collate_fn=lambda batch: collate_batch(batch, device=device),
    drop_last=False
)

logger.info(f"Number of samples in the dataset: {len(dataset)}")
logger.info(f"Number of training samples: {len(train_dataset)}")
logger.info(f"Number of validation samples: {len(val_dataset)}")
logger.info(f"PSF cubes are ({dataset.C}, {H}, {W})")
logger.info(f"Batch size: {BATCH_SIZE}")

#%%
random_idxs = np.random.randint(0, len(dataset), size=BATCH_SIZE)
test_batch = tuple([dataset[random_idx] for random_idx in random_idxs])
PSF_cubes, telemetry_vecs, fitted_vals, batch_config, idxs = collate_batch(test_batch, device=device)

#%% ==============================================================================
# Initialize the PSF model
# ==============================================================================
from PSF_models.NFM_wrapper import PSFModelNFM

if 'PSF_model' in locals():
    del PSF_model

with torch.no_grad():
    PSF_model = PSFModelNFM(
        batch_config,
        multiple_obs    = True,
        LO_NCPAs        = True,
        chrom_defocus   = False,
        use_splines     = True,
        Moffat_absorber = False,
        Z_mode_max      = 9,
        device          = device
    )
    
# Delete extra parameters that are pre-defined and thus don't need to be treated my the manager
# Could be just disabled but instead deleted to save a bit of memory
PSF_model.inputs_manager.delete('Jxy')     # Always zero inside
PSF_model.inputs_manager.delete('bg_ctrl') # Already subtracted from data
PSF_model.inputs_manager.delete('dx_ctrl') # Astrometric shifts are managed externally
PSF_model.inputs_manager.delete('dy_ctrl') # -//-
PSF_model.inputs_manager.delete('L0')      # Managed by configs

PSF_model.inputs_manager.delete('F_norm') # No photometric correction per source
PSF_model.inputs_manager.delete('flux_crop_ctrl') # -//-
PSF_model.inputs_manager.delete('src_dirs_x') # On-axis sources
PSF_model.inputs_manager.delete('src_dirs_y') # -//-

PSF_model.inputs_manager.delete('wind_speed_single') # Wind vector is not predicted
PSF_model.inputs_manager.delete('wind_dir_single') # -//-

if PSF_model.Moffat_absorber:
    # PSF_model.inputs_manager.delete('beta')
    # PSF_model.inputs_manager.delete('b')
    PSF_model.inputs_manager.delete('theta')
    PSF_model.inputs_manager.delete('ratio')

PSF_model.inputs_manager.set_optimizable(['LO_coefs'], predict_LO_NCPAs)
PSF_model.inputs_manager.set_optimizable(['Cn2_weights'], predict_Cn2_profile)   

print(PSF_model.inputs_manager)

# Note that the inputs manager for the calibrator is different from the default one,
# as some parameters are removed (GL weight can be computed) and some are set externally (phase bump)
#%%
calibrator_outputs_transformer = deepcopy(PSF_model.inputs_manager.get_transformer()) # The calibrator predicts the same parameters as the default model, but some of them are removed or set externally before running the model

# Get the dict of input tensors from the default model, to change their dimensions
buf_x_dict = calibrator_outputs_transformer.unstack(PSF_model.inputs_manager.stack())

input_features  = dataset.features
output_features = [k for k in buf_x_dict.keys()]

# Remove phase bump from the input dict, as it's set externally and not predicted by the NN
if 'LO_coefs' in buf_x_dict:
    buf_x_dict['LO_coefs'] = buf_x_dict['LO_coefs'][:, 1:] 

# Remove the first Cn2 weight, as it's determined by the others and the normalization to sum of 1
if 'Cn2_weights' in buf_x_dict:
    buf_x_dict['Cn2_weights'] = buf_x_dict['Cn2_weights'][:,1:]

_ = calibrator_outputs_transformer.stack(buf_x_dict) # Update stacking dimensions

N_outputs = calibrator_outputs_transformer.get_stacked_size()

del buf_x_dict

# Make sure the space from the removed inputs is freed up in GPU memory
gc.collect()
torch.cuda.empty_cache()

#%%
# NN is shallow and narrow - to limit capacity and overfitting
# Strong regularization - Dropout + weight decay + L1 TODO: include L1
# Small batches - 8-16 samples (to limit the influence of outliers and increase regularization effect)
# Early stopping - to prevent overfitting
# Data augmentation - TODO: if possible, add small noise to telemetry inputs
# Cross-validation - TODO: add k-fold CV for better estimates

import torch.nn as nn

class SmallCalibratorNet(nn.Module):
    """
    Compact neural network for calibration with strong regularization.
    Designed for small datasets (~500 samples).
    
    Args:
        n_features: Number of input features
        n_outputs: Number of output values
        hidden_dim: Size of hidden layers (default: 32)
        dropout_rate: Dropout probability (default: 0.3)
    """
    def __init__(self, n_features, n_outputs, hidden_dim=32, dropout_rate=0.3):
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
    def __init__(self, n_features, n_outputs, hidden_dim=24, dropout_rate=0.4):
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

# Initialize calibrator NN
calibrator = SmallCalibratorNet(
    n_features=N_features,
    n_outputs=N_outputs,
    hidden_dim=32,
    dropout_rate=0.3
).to(device)

default_lr = 1e-2

print(f"\n>>>>>>>>>>>> Calibrator inputs: {N_features}, outputs: {N_outputs}")

def save_checkpoint(calibrator, dx, dy, optimizer, epoch, train_loss, val_loss, path='checkpoint.pth'):
    """ Save a training checkpoint. """
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


def load_checkpoint(calibrator, optimizer, path):
    """ Load a training checkpoint and restore model state."""
    global dx, dy
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=device)
        calibrator.load_state_dict(checkpoint['calibrator_state_dict'])
        
        # Load astrometry parameters if they were saved
        if optimize_astrometry and 'dx' in checkpoint and 'dy' in checkpoint:
            dx = checkpoint['dx'].to(device)
            dy = checkpoint['dy'].to(device)
            logger.info("✅ Astrometry parameters (dx/dy) loaded from checkpoint")
            
        elif dx is None or dy is None:
            initialize_astrometry() # If dx/dy are not defined, initialize them (either from fitted values or with zeros)
            
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("✅ Optimizer state loaded from checkpoint")
            
        logger.info(f"🔄 Checkpoint loaded from {path} (epoch {checkpoint['epoch']})")
        return checkpoint['epoch'], checkpoint['train_loss'], checkpoint['val_loss']
    else:
        logger.error(f"❌ Checkpoint file not found at {path}. No weights initialized.")
        return None, None, None

#%%
# Pre training the calibrator with fitted parameters (optional, can help convergence if the fitted values are close to optimal)
# At this point, PSF model is no invloved yet, so it is a much easier task than predicting PSF cubes.
# It may help the model to converge faster when we start training with the full PSF model in the loop.

if not args.continue_training:
    optimizer = optim.AdamW(calibrator.parameters(), lr=default_lr, weight_decay=5e-4)
    scheduler_pretrain = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-4)

    num_epochs_pretrain = 100

    def calibrator_friendly_fit_dict(x_dict_fitted):
        ''' This function modifies the input dict to be more suitable for the calibrator NN, by removing parameters that are not predicted. '''
        x_dict_fitted = {key: x_dict_fitted[key] for key in output_features if key in x_dict_fitted} # Keep only the parameters that are predicted by the NN
        if 'LO_coefs' in x_dict_fitted:
            x_dict_fitted['LO_coefs'] = x_dict_fitted['LO_coefs'][:, 1:]
        # And Cn2_weights are already stored without the first weight in the fitting dataset
        return x_dict_fitted

    # We can also use a simpler loss function for this pre-training, such as MSE between the predicted parameters and the fitted values
    loss_pretrain = nn.MSELoss()

    calibrator.train()

    best_pretrain_loss = float('inf')
    best_calibrator_state = None


    for epoch in range(num_epochs_pretrain):
        calibrator.train()
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            _, telemetry_vecs, fitted_vals, _, _ = batch
        
            optimizer.zero_grad()
            
            y_data = calibrator_outputs_transformer.stack( calibrator_friendly_fit_dict(fitted_vals) )
            y_pred = calibrator(telemetry_vecs)
            
            loss = loss_pretrain(y_pred, y_data)
            loss.backward()

            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)

        # Validation
        calibrator.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                _, telemetry_vecs, fitted_vals, _, _ = batch
                
                y_data = calibrator_outputs_transformer.stack( calibrator_friendly_fit_dict(fitted_vals) )
                y_pred = calibrator(telemetry_vecs)
                
                val_loss = loss_pretrain(y_pred, y_data)
                epoch_val_loss += val_loss.item()
                
        avg_val_loss = epoch_val_loss / len(val_loader)
        
        scheduler_pretrain.step(avg_val_loss)

        if avg_val_loss < best_pretrain_loss:
            best_pretrain_loss = avg_val_loss
            best_calibrator_state = deepcopy(calibrator.state_dict())

        if (epoch + 1) % 10 == 0:
            logger.info(f"Pre-train Epoch {epoch+1}: train_loss = {avg_loss:.6f}, val_loss = {avg_val_loss:.6f}")

    if best_calibrator_state is not None:
        calibrator.load_state_dict(best_calibrator_state)
        logger.info(f"Restored best pre-training weights (val_loss: {best_pretrain_loss:.6f})")

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Store best pretrain calibrator state in the file
    pretrain_weights_path = WEIGHTS_FOLDER / 'NFM_calibrator/pretrain_weights.pth'
    torch.save(best_calibrator_state, pretrain_weights_path)
    logger.info(f"Saved best pre-training weights to {pretrain_weights_path}")
else:
    # Load best pre-training weights if continue_training flag is set
    pretrain_weights_path = WEIGHTS_FOLDER / 'NFM_calibrator/pretrain_weights.pth'
    if pretrain_weights_path.exists():
        best_pretrain_state = torch.load(pretrain_weights_path, map_location=device)
        calibrator.load_state_dict(best_pretrain_state)
        logger.info(f"🔄 Loaded pre-training weights from {pretrain_weights_path}")
    else:
        logger.warning(f"❌ Pre-training weights not found at {pretrain_weights_path}. Continuing with current weights.")

#%%
fill_from_fitted = lambda key: np.array([d[key] for d in dataset.fitted_vals], dtype=np.float32) if key in dataset.fitted_vals[0] else None

def initialize_astrometry():
    global optimize_astrometry
    global dx, dy
    # Init astrometric shifts dataset with fitted values if available, otherwise with zeros.
    if pre_init_astrometry:
        logger.info("✅ Using pre-fitted astrometry values for initialization")
        dx_data = fill_from_fitted('dx_ctrl')
        dy_data = fill_from_fitted('dy_ctrl')
    else:
        logger.info("ℹ️ Initializing astrometry with zeros and forcing optimization")
        # If zeros are used, we set optimize_astrometry=True to allow the model to learn them from the data
        optimize_astrometry = True
        dx_data = np.zeros((len(dataset), len(dataset.fitted_vals[0]['dx_ctrl'])), dtype=np.float32)
        dy_data = np.zeros((len(dataset), len(dataset.fitted_vals[0]['dy_ctrl'])), dtype=np.float32)
    
    dx = torch.tensor(dx_data, device=device, dtype=torch.float32, requires_grad=optimize_astrometry)
    dy = torch.tensor(dy_data, device=device, dtype=torch.float32, requires_grad=optimize_astrometry)


if args.continue_training:
    if BEST_CALIB_PATH.exists():
        load_checkpoint(calibrator, None, BEST_CALIB_PATH)                      
    else:
        logger.info(f"ℹ️ Training model from the pre-trained weights (if available) or from scratch.")
        initialize_astrometry()
        
# Initialize NCPAs with fitted values if available, otherwise with median values from the fitting dataset (same for all samples).
LO_dataset = fill_from_fitted('LO_coefs')
phase_bump = torch.tensor(LO_dataset[:, 0], device=device, dtype=torch.float32)  # Extract phase bump values separately
LO_dataset = LO_dataset[:, 1:] # Remove phase bump from the dataset, leaving only Zernike coefficients

if predict_LO_NCPAs:
    logger.info(f"ℹ️ Prediction of modes Z3-Z{PSF_model.Z_mode_max} is enabled")
else:
    logger.info(f"ℹ️ Initializing NCPAs with median values from fitting (same for all samples)")
    NCPAs_median = torch.tensor(np.median(LO_dataset, axis=0)[None,...], device=device, dtype=torch.float32)

# %%
# To save GPU memory, PSFs are predicted and compared to data only for a subset of wavelengths at a time
λ_full = PSF_model.wavelengths.clone() # [nm]

def generate_wavelength_sets(step, max_val):
    """ Generates wavelength index sets for training, ensuring coverage of all wavelengths with a given step size. """
    sets = []
    thresh = step // 2
    for offset in range(step):
        indices = list(range(offset, max_val + 1, step))
        if indices[0] > thresh: indices.insert(0, 0)
        if (max_val - indices[-1]) > thresh: indices.append(max_val)
        sets.append(indices)
    return sets

λ_step = 2

λ_id_sets = generate_wavelength_sets(λ_step, N_wvl_total-1)
λ_sets = [λ_full[list_id] for list_id in λ_id_sets]

print(f"Total wavelengths: {N_wvl_total}, generated {len(λ_id_sets)} sets with step {λ_step}, set sizes are: {[len(s) for s in λ_id_sets]}")

#%%
# Now, we put the model in the loop and train it end-to-end to predict PSF cubes, using the full loss function that compares predicted and true PSFs.

def run_model(x_dict_NN, config, idx, λ_ids):
    ''' This function runs the PSF model to predict the batch of PSFs based on the dictionary of parameters predicted by the NN and the batch configuration. '''
    # NOTE: idx here are dataset ids, not the PSF samples ids
    # Get the current dict of input tensors from the PSF model's inputs manager    
    x_dict = PSF_model.inputs_manager.to_dict()
    # Update only the parameters that are predicted by the NN, keep the rest unchanged
    x_dict = {key: x_dict_NN[key] for key in x_dict_NN.keys() if key in x_dict}
    # Set dx/dy for this batch (indexed by array's index)
    x_dict['dx_ctrl'] = dx[idx]
    x_dict['dy_ctrl'] = dy[idx]

    # Append phase bump
    x_dict['LO_coefs'] = torch.hstack( (phase_bump[idx].unsqueeze(-1), x_dict['LO_coefs']) ) if predict_LO_NCPAs \
                    else torch.hstack( (phase_bump[idx].unsqueeze(-1), NCPAs_median.repeat(len(idx),1))
    )
    # Append Cn2 ground layer weight
    if predict_Cn2_profile:
        GL_fraction = 1.0 - x_dict['Cn2_weights'].sum(dim=-1, keepdim=True)
        x_dict['Cn2_weights'] = torch.hstack( (GL_fraction, x_dict['Cn2_weights']) )

    current_wavelengths = λ_full[λ_ids].to(device=device)
    
    config['sources_science']['Wavelength'] = current_wavelengths.view(1,-1) # [1, N_wvl_selected]
    
    if current_wavelengths.shape == PSF_model.wavelengths.shape and torch.allclose(current_wavelengths, PSF_model.wavelengths, atol=1e-12):
        PSF_model.model.Update(config=config, grids=False, pupils=False, tomography=True)
    else:
        # Update the internal state of the PSF model for the given batch config. Update just model parameters, not grids
        # This could be done with the SetWavelengths method, but it does some extra re-initializations, so we do it manually to save
        PSF_model.wavelengths = current_wavelengths.clone()
        PSF_model.model.Update(config=config, grids=True, pupils=False, tomography=True)
    
    return PSF_model(x_dict) # Run given the predicted parameters and the updated internal state, get the predicted PSF cubes

#%%
# Initialize training and validation loss function

λ_weighting = False # no weighting of PSFs at different wavelengths, as it may bias the fit towards the redder wavelengths with higher SNR

if λ_weighting:
    wvl_weights = torch.linspace(0.5, 1.0, N_wvl_total).to(device).view(1, N_wvl_total, 1, 1)
    wvl_weights = N_wvl_total / wvl_weights.sum() * wvl_weights # Normalize so that the total energy is preserved
else:
    wvl_weights = torch.ones((1, N_wvl_total, 1, 1), device=device)

# Enforce positive values for modal coefficients to mitigate sign ambiguity and improve convergence
force_positive = lambda x: torch.clamp(-x, min=0).pow(2).mean()
# TODO: positive r0 regularization?

def loss_PSF(PSF_data, PSF_model, w_MSE, w_MAE):
    diff = (PSF_model-PSF_data) #* wvl_weights[:, λ_ids, ...] #TODO: add wvl selection
    w = 2e4 # Empirical weight to balance the loss magnitude with the regularization terms
    MSE_loss = diff.pow(2).mean() * w_MSE
    MAE_loss = diff.abs().mean()  * w_MAE
    return w * (MSE_loss + MAE_loss)


def loss_LO(coefs_vec, w_bump, w_LO):
    # L2 regularization on all LO coefficients
    LO_loss = coefs_vec.pow(2).sum(-1).mean() * w_LO
    # Constraint to enforce first element of LO_coefs to be positive
    phase_bump_positive = force_positive(coefs_vec[:, 0]) * w_bump
    # Force defocus to be positive to mitigate sign ambiguity
    first_defocus_penalty = force_positive(coefs_vec[:, 2]) * w_LO #NOTE: won't work with the chromatic defocus
    return LO_loss + phase_bump_positive + first_defocus_penalty


def loss_fn(PSF_data, PSF_model, x_dict_pred):
    # PSF_loss = loss_PSF(PSF_data, PSF_model, w_MSE=900.0, w_MAE=1.6) Used to be 900
    PSF_loss = loss_PSF(PSF_data, PSF_model, w_MSE=1200.0, w_MAE=1.6)
    LO_loss  = loss_LO(x_dict_pred['LO_coefs'], w_bump=5e-5, w_LO=1e-7) if predict_LO_NCPAs else 0.0
    return PSF_loss + LO_loss

#%%
def validate(loader=None, return_cubes=True, verbose=False):
    """
    Validaion function to evaluate the calibrator on the validation dataset.
    It can return either just the average loss, or the full predicted and data PSF cubes for all samples and wavelengths in addition.
    NOTE: validation dataset is ordered, so we can fill it batch by batch in the correct order to return full cubes without running out of memory.
    This is possible because we set shuffle=False and drop_last=False in the validation DataLoader.
    
    Args:
        loader: DataLoader to use (default: val_loader)
        return_cubes: If True, returns full PSF cubes (memory intensive). If False, returns only loss.
        verbose: If True, prints progress and statistics
    
    Returns:
        If return_cubes is True:
            PSFs_pred_cube: Tensor of shape (N_val_samples_actual, N_wvl_total, H, W) with all predicted PSFs
            PSFs_data_cube: Tensor of shape (N_val_samples_actual, N_wvl_total, H, W) with all data PSFs
            validation_ids: Tensor of shape (N_val_samples_actual,) with validation sample IDs
            NN_predictions: Tensor of shape (N_val_samples_actual, N_outputs) with NN predictions
            avg_loss: Average validation loss
        If return_cubes is False:
            avg_loss: Average validation loss
    """
    if loader is None:
        loader = val_loader

    calibrator.eval()
    total_loss = 0
    total_simulated_batches = 0
    
    PSFs_pred_cube = None
    PSFs_data_cube = None
    validation_ids = []
    NN_predictions = None

    if return_cubes:
        N_samples = len(dataset)

        PSFs_pred_cube = torch.zeros((N_samples, N_wvl_total, H, W), dtype=torch.float32, device='cpu')
        PSFs_data_cube = torch.zeros((N_samples, N_wvl_total, H, W), dtype=torch.float32, device='cpu')
        NN_predictions = torch.zeros((N_samples, N_outputs), dtype=torch.float32, device='cpu')

        if verbose: logger.info(f"Validation dataset size: {N_samples} (full), batches: {len(loader)}")
    
    with torch.no_grad():
        # Outer loop: iterate through batches
        for batch_idx, batch in enumerate(loader):
            PSF_data, telemetry_inputs, _, batch_config, idxs = batch
            
            # IDs are used directly to place data in the correct position in the full cubes
            # This is robust to shuffling and lack of ordering
            idxs_cpu = idxs.cpu()

            if return_cubes:
                validation_ids.extend(idxs_cpu.tolist())
            
            # Get calibrator predictions (same for all wavelength sets since prediction is λ-agnostic)
            x_pred = calibrator(telemetry_inputs)
            x_dict_pred = calibrator_outputs_transformer.unstack(x_pred)
            
            if return_cubes:
                NN_predictions[idxs_cpu, :] = x_pred.cpu()
            
            if verbose: logger.info(f"Processed batch {batch_idx + 1}/{len(loader)}")
            
            # Inner loop: iterate through wavelength sets
            for current_λ_set_id in range(len(λ_sets)):
                λ_id_set = λ_id_sets[current_λ_set_id]  # Which wavelength indices for this set
                
                # Run model for this wavelength set
                PSF_pred = run_model(x_dict_pred, batch_config, idxs, λ_id_set)
                
                # Calculate loss for this batch and wavelength set
                loss = loss_fn(PSF_pred, PSF_data[:, λ_id_set, ...], x_dict_pred)
                total_loss += loss.item()
                total_simulated_batches += 1
                
                if return_cubes:
                    # Fill in the wavelengths for this batch
                    for wvl_idx, λ_id in enumerate(λ_id_set):
                        PSFs_pred_cube[idxs_cpu, λ_id, :, :] = PSF_pred[:, wvl_idx, :, :].cpu()
                        PSFs_data_cube[idxs_cpu, λ_id, :, :] = PSF_data[:, λ_id, :, :].cpu()
                
                if verbose: logger.info(f"  >> λ set {current_λ_set_id + 1}/{len(λ_sets)}")

    if verbose and return_cubes:
        logger.info(f"PSF pred range: [{PSFs_pred_cube.min():.4e}, {PSFs_pred_cube.max():.4e}]")
        logger.info(f"PSF data range: [{PSFs_data_cube.min():.4e}, {PSFs_data_cube.max():.4e}]")

    if return_cubes:
        # Select only those samples that are actually in the validation set (in case of Subset with non-contiguous indices)
        validation_ids = torch.tensor(validation_ids, dtype=torch.long)
        PSFs_pred_cube = PSFs_pred_cube[validation_ids]
        PSFs_data_cube = PSFs_data_cube[validation_ids]
        NN_predictions = NN_predictions[validation_ids]
        
        return PSFs_pred_cube, PSFs_data_cube, validation_ids, NN_predictions, total_loss / total_simulated_batches if total_simulated_batches > 0 else 0.0
    else:
        return total_loss / total_simulated_batches if total_simulated_batches > 0 else 0.0

#%%
# Validate after calibrator pre-training, before the main training loop, to check that everything is working and to have a baseline for comparison
PSFs_pred_cube, PSFs_data_cube, validation_ids, NN_predictions, val_loss_total = validate(loader=val_loader, verbose=True, return_cubes=True)

gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()

#%%
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
    min_lr   = 1e-6
)

logger.info(f"Scheduler: ReduceLROnPlateau with patience={3}, factor={0.7}, min_lr={1e-5}")


# L1 regularization of NN parameters TODO: use it
def l1_regularization(model, lambda_l1=1e-4):
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    return lambda_l1 * l1_norm


# Helper function to check for NaN values
def check_for_nan(loss, model, epoch, batch_idx):
    """
    Check if loss or model parameters contain NaN values.
    Returns True if NaN detected, False otherwise.
    """
    if torch.isnan(loss) or torch.isinf(loss):
        logger.error(f"❌ NaN/Inf detected at epoch {epoch}, batch {batch_idx}: {loss.item()}")
        return True
    
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            logger.error(f"❌ NaN/Inf detected in parameter '{name}' at epoch {epoch}, batch {batch_idx}")
            return True
    
    return False


# Complete training loop with validation
def train(num_epochs=50, patience=10, nan_recovery=True, max_nan_recoveries=3):
    """
    Training loop with validation where dx/dy are optimized during validation.
    
    Args:
        num_epochs: Maximum number of training epochs
        patience: Early stopping patience
        nan_recovery: If True, recover from NaN by loading last good checkpoint
        max_nan_recoveries: Maximum number of times to recover from NaN before giving up
    """
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    nan_recovery_count = 0
    patience_counter   = 0
    
    lr_decay = 0.7  # Factor to reduce learning rate on NaN recovery
    
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
                PSF_data, telemetry_inputs, _, batch_config, idxs = batch
                
                optimizer.zero_grad()
                
                x_pred = calibrator(telemetry_inputs)
                x_dict_pred = calibrator_outputs_transformer.unstack(x_pred)

                PSF_pred = run_model(x_dict_pred, batch_config, idxs, λ_id_set)
                loss = loss_fn(PSF_pred, PSF_data[:, λ_id_set, ...], x_dict_pred)

                # ========== NaN detection ==========
                if check_for_nan(loss, calibrator, epoch, batch_idx):
                    nan_detected_this_epoch = True
                    
                    if nan_recovery and nan_recovery_count < max_nan_recoveries:
                        nan_recovery_count += 1
                        loss_stats['nan_recoveries'] = nan_recovery_count
                        logger.error(f"🚨 NaN detected! Recovery attempt {nan_recovery_count}/{max_nan_recoveries}")
                        
                        try:
                            # Load last good checkpoint
                            load_checkpoint(calibrator, optimizer, BEST_CALIB_PATH)
                            # Reduce learning rate
                            for param_group in optimizer.param_groups:
                                old_lr = param_group['lr']
                                param_group['lr'] = old_lr * lr_decay
                                logger.warning(f"Reduced LR: {old_lr:.2e} -> {param_group['lr']:.2e}")

                            logger.info(f"✅ Recovered from checkpoint, LR reduced by {(1.-lr_decay) * 100:.0f}%")
                            logger.info("Model recovered from checkpoint, continuing training...")
                            break  # Skip rest of this epoch, start fresh
                            
                        except FileNotFoundError:
                            logger.error("❌ No checkpoint found! Cannot recover from NaN.")
                            raise ValueError("NaN detected but no checkpoint to recover from")
                    else:
                        logger.error(f"❌ Max NaN recoveries ({max_nan_recoveries}) reached or recovery disabled. Stopping training.")
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

                logger.debug(f"Epoch {epoch}, λ set {current_λ_set_id + 1}/{len(λ_sets)}, "
                             f"batch {batch_idx + 1}/{len(train_loader)}: train_loss = {loss.item():.6f}, LR = {current_lr:.2e}")
            
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
        val_loss = validate(return_cubes=False)

        # Check for NaN in validation
        if np.isnan(val_loss) or np.isinf(val_loss):
            logger.error(f"❌ NaN/Inf in validation loss at epoch {epoch}")
            if nan_recovery and nan_recovery_count < max_nan_recoveries:
                nan_recovery_count += 1
                logger.warning(f"Attempting recovery from validation NaN...")
                load_checkpoint(calibrator, optimizer, BEST_CALIB_PATH)
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
            print(f"⚠️️️ Learning rate reduced: {old_lr:.2e} -> {new_lr:.2e}")
        
        current_lr = optimizer.param_groups[0]['lr']
        loss_stats['learning_rates'].append(current_lr)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            loss_stats['best_epoch'] = epoch
            loss_stats['best_val_loss'] = val_loss
            
            # Save best model
            save_checkpoint(calibrator, dx, dy, optimizer, epoch, avg_train_loss, val_loss, BEST_CALIB_PATH)
            logger.info(f"✅ Saved best model at epoch {epoch} with val_loss = {val_loss:.6f}")
        else:
            patience_counter += 1
        
        # Save periodic checkpoint every 5 epochs (for extra safety)
        if epoch % 5 == 0:
            save_checkpoint(calibrator, dx, dy, optimizer, epoch, avg_train_loss, val_loss, 
                            WEIGHTS_FOLDER/f'NFM_calibrator/checkpoint_epoch_{epoch}.pth')
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
            print(f"\n ⚠️️️ Early stopping at epoch {epoch}")
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
        logger.warning(f"⚠️️️ Training recovered from NaN {nan_recovery_count} time(s)")
        print(f"\n ⚠️️ Note: Training recovered from NaN {nan_recovery_count} time(s)")
    
    # Load best model
    logger.info("Loading best model...")
    print("\nLoading best model...")
    checkpoint = torch.load(BEST_CALIB_PATH)
    calibrator.load_state_dict(checkpoint['calibrator_state_dict'])

    if optimize_astrometry and 'dx' in checkpoint and 'dy' in checkpoint:
        dx.data = checkpoint['dx']
        dy.data = checkpoint['dy']
    
    logger.info(f"Best validation loss: {checkpoint['val_loss']:.6f} at epoch {checkpoint['epoch']}")
    print(f"Best validation loss: {checkpoint['val_loss']:.6f} at epoch {checkpoint['epoch']}")
    
    return calibrator, train_losses, val_losses


# %%
# def main():
logger.info("="*60)
logger.info("Starting Training with Validation")
logger.info("="*60)

# Backup the previous best checkpoint if it exists
if os.path.exists(BEST_CALIB_PATH):
    backup_path = BEST_CALIB_PATH.with_name(BEST_CALIB_PATH.stem + '_backup.pth')
    os.rename(BEST_CALIB_PATH, backup_path)
    logger.debug(f"Previous checkpoint backed up to {backup_path}")

# Train the model with NaN recovery enabled
calibrator, train_losses, val_losses = train(
    num_epochs=500,
    patience=20,
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

#%%
tuned_params = ['F_ctrl', 'dn']

def tune_calibrator(calibrator, train_loader, val_loader, tuned_params=None, num_epochs=50, lr=1e-3):
    """
    Fine-tune specific parameters of the calibrator while keeping others close to the reference model.
    
    Args:
        calibrator: The model to tune
        train_loader: Training data loader
        val_loader: Validation data loader
        tuned_params: List of parameter names to tune
        num_epochs: Number of tuning epochs
        lr: Learning rate
    """
    logger.info("="*60)
    logger.info(f"Starting Tuning for parameters: {tuned_params}")
    logger.info("="*60)
    
    if tuned_params is None:
        ValueError("tuned_params list must be specified for tuning")
    
    # Create reference model (frozen)
    calibrator_ref = deepcopy(calibrator)
    calibrator_ref.eval()
    for param in calibrator_ref.parameters():
        param.requires_grad = False
    
    # Create mask for tuned parameters
    tuned_slices = [calibrator_outputs_transformer.slices[param] for param in tuned_params]
    tuned_idx_mask = torch.zeros(N_outputs, dtype=torch.bool, device=device)
    for slc in tuned_slices:
        tuned_idx_mask[slc] = True
    tuned_idx_mask = tuned_idx_mask.unsqueeze(0)  # [1, N_outputs] for broadcasting
    
    # Optimizer for tuning
    optimizer_tune = optim.AdamW(calibrator.parameters(), lr=lr, weight_decay=5e-4) # Tune all weights, but loss will constrain outputs
    scheduler_tune = optim.lr_scheduler.ReduceLROnPlateau(optimizer_tune, mode='min', factor=0.5, patience=5)
    
    best_loss = float('inf')
    best_state = None
    
    for epoch in range(num_epochs):
        calibrator.train()
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            PSF_data, telemetry_inputs, _, batch_config, idxs = batch
            
            optimizer_tune.zero_grad()
            
            # Get predictions from both models
            x_pred_tuned = calibrator(telemetry_inputs)
            with torch.no_grad():
                x_pred_ref = calibrator_ref(telemetry_inputs)
            
            # Soft-fix strategy: 
            # 1. Use tuned predictions for tuned parameters
            # 2. Use reference predictions for fixed parameters (hard constraint in forward pass)
            # This allows the NN to update its weights to optimize tuned params, while trying to maintain 
            # the fixed params (via the fact that weights are shared).
            # Note: Since weights are shared, changing them to improve tuned_params WILL change fixed_params outputs.
            # To strictly enforce fixed values, we must replace them in the output used for PSF generation.
            
            x_pred_combined = torch.where(tuned_idx_mask, x_pred_tuned, x_pred_ref)
            
            # Loss computation
            # x_dict_pred = calibrator_outputs_transformer.unstack(x_pred_combined)
            
            # Anchor loss
            # We negate the mask to get fixed parameters
            # Since tuned_idx_mask is [1, N_outputs], let's strip the first dim for indexing or broadcast properly
            mask_fixed = ~tuned_idx_mask.squeeze(0) # [N_outputs]
            
            # Use masking on columns (dim 1)
            # anchor_loss = (x_pred_tuned[:, mask_fixed] - x_pred_ref[:, mask_fixed]).pow(2).mean() * 100.0
            
            if mask_fixed.any():
                anchor_loss = (x_pred_tuned[:, mask_fixed] - x_pred_ref[:, mask_fixed]).pow(2).mean() * 100.0
                anchor_loss.backward() # Compute gradients for anchor loss AND FREE GRAPH
            else:
                 anchor_loss = torch.tensor(0.0, device=device)

            # Free memory from initial pass
            del x_pred_tuned, x_pred_ref, x_pred_combined
            
            running_batch_loss = 0.0

            # Iterate over wavelength sets & backprop immediately to free memory
            for i, λ_id_set in enumerate(λ_id_sets): 
                 # Re-run the NN forward pass to create a fresh graph for this iteration
                 # This avoids retain_graph=True across iterations, allowing TipTorch buffers to be freed
                 # The NN is small so this is cheap.
                 
                 x_pred_tuned_loop = calibrator(telemetry_inputs)
                 
                 with torch.no_grad():
                     x_pred_ref_loop = calibrator_ref(telemetry_inputs)
                 
                 x_pred_combined_loop = torch.where(tuned_idx_mask, x_pred_tuned_loop, x_pred_ref_loop)
                 x_dict_pred_loop = calibrator_outputs_transformer.unstack(x_pred_combined_loop)
                 
                 PSF_model.SetWavelengths(λ_full[λ_id_set].to(device))
                 PSF_pred = run_model(x_dict_pred_loop, batch_config, idxs, λ_id_set)
                 
                 loss = loss_fn(PSF_pred, PSF_data[:, λ_id_set, ...], x_dict_pred_loop)
                 loss_item = loss.item() # Save item before backward

                 # Backward without retaining graph!
                 loss.backward() 
                 
                 running_batch_loss += loss_item
                 
                 current_lr = optimizer_tune.param_groups[0]['lr']
                 logger.debug(f"Tune Epoch {epoch}, λ set {i + 1}/{len(λ_sets)}, "
                              f"batch {batch_idx + 1}/{len(train_loader)}: batch_loss = {loss_item:.6f}, LR = {current_lr:.2e}")

                 # Explicitly delete large tensors to help GC
                 del PSF_pred, loss, x_pred_tuned_loop, x_pred_combined_loop, x_dict_pred_loop
            
            optimizer_tune.step()
            
            epoch_loss += running_batch_loss + anchor_loss.item()
            
            current_lr = optimizer_tune.param_groups[0]['lr']
            logger.debug(f"Tune Epoch {epoch:3d}/{num_epochs} | "
                         f"Batch {batch_idx + 1:3d}/{len(train_loader)} | "
                         f"Total Batch Loss: {running_batch_loss:.6f} | "
                         f"Anchor Loss: {anchor_loss.item():.6f}")
            
        avg_loss = epoch_loss / len(train_loader)
        
        # Validation
        calibrator.eval()
        val_loss = 0.0
        with torch.no_grad():
             for batch in val_loader:
                PSF_data, telemetry_inputs, _, batch_config, idxs = batch
                
                x_pred_tuned = calibrator(telemetry_inputs)
                x_pred_ref = calibrator_ref(telemetry_inputs)
                x_pred_combined = torch.where(tuned_idx_mask, x_pred_tuned, x_pred_ref)
                x_dict_pred = calibrator_outputs_transformer.unstack(x_pred_combined)
                
                batch_val_loss = 0
                for λ_id_set in λ_id_sets:
                     PSF_model.SetWavelengths(λ_full[λ_id_set].to(device))
                     PSF_pred = run_model(x_dict_pred, batch_config, idxs, λ_id_set)
                     batch_val_loss += loss_fn(PSF_pred, PSF_data[:, λ_id_set, ...], x_dict_pred)
                
                val_loss += batch_val_loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        scheduler_tune.step(avg_val_loss)
        
        logger.info(f"Tune Epoch {epoch+1}: train_loss={avg_loss:.6f}, val_loss={avg_val_loss:.6f}")
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_state = deepcopy(calibrator.state_dict())
            
    if best_state is not None:
        calibrator.load_state_dict(best_state)
        logger.info(f"Tuning complete. Best val loss: {best_loss:.6f}")


#%%
gc.collect()
torch.cuda.empty_cache()

#%%
tune_calibrator(calibrator, train_loader, val_loader, tuned_params=['F_ctrl'])


# %%
# if __name__ == "__main__":
#     main()

# Load best model
logger.info("Loading best model...")
print("\nLoading best model...")

epoch, train_loss, val_loss = load_checkpoint(calibrator, optimizer, BEST_CALIB_PATH)
logger.info(f"Best validation loss: {val_loss:.6f} at epoch {epoch}")

# Validate and collect all PSF predictions and data
PSFs_pred_cube, PSFs_data_cube, validation_ids, NN_predictions, final_val_loss = validate()

print(f"\n{'='*60}")
print(f"Validation Results:")
print(f"{'='*60}")
print(f"Final validation loss (fixed dx/dy): {final_val_loss:.6f}")
print(f"PSF predictions shape: {PSFs_pred_cube.shape}")
print(f"PSF data shape: {PSFs_data_cube.shape}")
print(f"Validation sample IDs shape: {validation_ids.shape}")
print(f"Validation IDs range: [{validation_ids.min()}, {validation_ids.max()}]")
print(f"{'='*60}\n")

logger.info(f"Final validation loss (fixed dx/dy): {final_val_loss:.6f}")
logger.info(f"PSF predictions collected: {PSFs_pred_cube.shape}")
logger.info(f"PSF data collected: {PSFs_data_cube.shape}")
logger.info(f"Validation sample IDs collected: {validation_ids.shape}")

# Optionally save the cubes for later analysis
# torch.save(PSF_pred_cube, WEIGHTS_FOLDER / 'NFM_calibrator/validation_PSFs_predicted.pt')
# torch.save(PSF_data_cube, WEIGHTS_FOLDER / 'NFM_calibrator/validation_PSFs_data.pt')
# torch.save(validation_ids, WEIGHTS_FOLDER / 'NFM_calibrator/validation_sample_ids.pt')
# torch.save(PSF_data_cube, WEIGHTS_FOLDER / 'NFM_calibrator/validation_PSFs_data.pt')

PSFs_pred_cube = PSFs_pred_cube.cpu()
PSFs_data_cube = PSFs_data_cube.cpu()
validation_ids = validation_ids.cpu()

#%%
from tools.plotting import plot_radial_PSF_profiles, draw_PSF_stack

id_src = np.random.randint(0, PSFs_data_cube.shape[0])
print(f"Randomly selected validation sample ID: {id_src}")
print(f"Corresponding original dataset index: {validation_ids[id_src].item()}")

PSF_0 = PSFs_data_cube[id_src]
PSF_1 = PSFs_pred_cube[id_src]

# NN_pred = NN_predictions[id_src].cpu().unsqueeze(0).numpy()
# x_pred = inputs_transformer.unstack(NN_pred)

vmin = np.percentile(PSF_0[PSF_0 > 0].cpu().numpy(), 10)
vmax = np.percentile(PSF_0[PSF_0 > 0].cpu().numpy(), 99.995)
wvl_select = np.s_[0, N_wvl_total//2, -1]
PSF_disp = lambda x, w: (x[w,...]).cpu().numpy()

draw_PSF_stack(
    PSF_0.numpy()[wvl_select, ...],
    PSF_1.numpy()[wvl_select, ...],
    average=True,
    min_val=vmin,
    max_val=vmax,
    crop=100
)

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
fig, ax = plt.subplots(1, len(wvl_select), figsize=(10, len(wvl_select)))
for i, lmbd in enumerate(wvl_select):
    plot_radial_PSF_profiles(
        PSFs_data_cube[:, lmbd, ...].cpu().numpy(),
        PSFs_pred_cube[:, lmbd, ...].cpu().numpy(),
        'Data',
        'TipTorch',
        cutoff=40,
        ax=ax[i],
        title=f"Wavelength {λ_full[lmbd].item():.2f} m"
        #, centers=center)
    )
plt.title('Radial PSF Profiles at Selected Wavelengths')
plt.show()

#%%
fig = plt.figure(figsize=(10, 6))
plt.title('Polychromatic PSF')
PSF_avg = lambda x: np.mean(x.cpu().numpy(), axis=1)
plot_radial_PSF_profiles(
    PSF_avg(PSFs_data_cube),
    PSF_avg(PSFs_pred_cube),
    'Data',
    'TipTorch',
    title='Spectrally averaged PSF',
    cutoff=40,
    ax=fig.add_subplot(111)
)
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
        x_pred = calibrator_outputs_transformer.unstack(dummy_vec)
        PSF_pred = run_model(x_pred, batch_config, idxs, λ_id_set)
        initial_loss = criterion(PSF_pred, PSF_cubes[:, λ_id_set, ...], x_pred, λ_id_set)
        logger.info(f"Initial loss: {initial_loss.item():.6f}")
    
    logger.info(f"Starting optimization for {num_iters} iterations...")
    
    
    # Optimization loop
    for epoch in range(num_iters):
        optimizer.zero_grad()
        
        # Forward pass
        x_pred = calibrator_outputs_transformer.unstack(dummy_vec)
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
        x_pred = calibrator_outputs_transformer.unstack(dummy_vec)
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
    PSFs_data_cube = results['PSF_cubes'].cpu()
    PSFs_pred_cube = results['PSF_pred'].cpu()

    x_pred = results['final_x_pred']
    N_wvl_temp = PSFs_data_cube.shape[1]

    dummy_vec = results['dummy_vec'].cpu()
    x_test = calibrator_outputs_transformer.unstack(dummy_vec)


#%%
from tools.plotting import plot_radial_PSF_profiles, draw_PSF_stack

id_src = np.random.randint(0, PSFs_data_cube.shape[0])

PSF_0 = PSFs_data_cube[id_src]
PSF_1 = PSFs_pred_cube[id_src]

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
        PSFs_data_cube[:,i,...].cpu().numpy(),
        PSFs_pred_cube[:,j,...].cpu().numpy(),
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
