#%%
try:
    ipy = get_ipython()        # NameError if not running under IPython
    if ipy:
        ipy.run_line_magic('reload_ext', 'autoreload')
        ipy.run_line_magic('autoreload', '2')
except NameError:
    pass


import gc
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from copy import deepcopy

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path
from datetime import datetime
from tiptorch._config import *
from torch.utils.data import Dataset

from tiptorch.managers.config_manager import MultipleTargetsInDifferentObservations

MUSE_DATA_FOLDER = Path(project_settings["MUSE_data_folder"])
STD_FOLDER       = Path(project_settings["MUSE_STD_data_folder"])
DATASET_CACHE    = STD_FOLDER / 'dataset_cache'
BEST_CALIB_PATH  = WEIGHTS_FOLDER / 'NFM_calibrator/best_calibrator_checkpoint.pth'
DEBUG            = True # TODO: make it an argument
BATCH_SIZE       = 16 # TODO: make it an argument

pre_init_astrometry = True
optimize_astrometry = False
predict_Cn2_profile = True
predict_LO_NCPAs    = True

# Set up logging
log_dir = Path('../data/logs')
log_dir.mkdir(parents=True, exist_ok=True)
log_filename = log_dir / f'training_NFM_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

# Configure logging (force=True so it works even when IPython has already set up the root logger)
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)  # Also log to console
    ],
    force=True
)

# Silence noisy libraries
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def release_gpu_memory(sync=False):
    """Best-effort VRAM cleanup without affecting model state."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if sync:
            torch.cuda.synchronize()

logger.info("="*60)
logger.info("NFM Calibrator Training Script")
logger.info("="*60)
logger.info(f"Log file: {log_filename}")

# Add argument inputs
parser = argparse.ArgumentParser(description="Train NFM Calibrator")
# parser.add_argument('--weights', type=str, default=str(BEST_CALIB_PATH), help='Path to the best calibrator weights checkpoint')
parser.add_argument('--continue-training', action='store_true', help='Whether to continue training from a checkpoint')
parser.add_argument('--run-kfold-difficulty', action='store_true',
                    help='Run K-fold out-of-fold training to estimate telemetry-conditioned difficulty weights before final training')
parser.add_argument('--kfolds', type=int, default=5, help='Number of folds for difficulty-weight estimation')
parser.add_argument('--kfold-epochs', type=int, default=120, help='Maximum epochs per K-fold difficulty model')
parser.add_argument('--kfold-patience', type=int, default=12, help='Early-stopping patience per K-fold difficulty model')
parser.add_argument('--difficulty-alpha', type=float, default=0.30, help='Strength of RF log-loss difficulty weighting')
parser.add_argument('--difficulty-clip-min', type=float, default=0.50, help='Minimum sample weight after clipping')
parser.add_argument('--difficulty-clip-max', type=float, default=2.00, help='Maximum sample weight after clipping')

# Handle both command line and iPython environments
try:
    args = parser.parse_args()   #TODO: implement continue training
except SystemExit:
    # In iPython/Jupyter, parse_args() fails, so use default values
    args = argparse.Namespace(
        continue_training=True,
        run_kfold_difficulty=True,
        kfolds=5,
        kfold_epochs=120,
        kfold_patience=12,
        difficulty_alpha=0.30,
        difficulty_clip_min=0.50,
        difficulty_clip_max=2.00,
    )

# watch -n 0.1 'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits -i 1 | awk -F, "{printf \"GPU: %s%%  VRAM: %s / %s MiB\n\",\$1,\$2,\$3}"'

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
    batch_config['DM']['NumberReconstructedLayers'] = torch.tensor(3.0, device=device)
    return PSF_cubes, telemetry_vecs, fitted_vals, batch_config, idxs


#%%
dataset = NFMDataset(DATASET_CACHE / 'muse_STD_stars_dataset.pt')
N_features = len(dataset.features)
N_wvl_total = dataset.C # The number of spectral channels from the PSF data

# Data loaders
train_idx, test_idx = train_test_split(np.arange(len(dataset)), test_size=0.20, random_state=42) #TODO make it a part of the STD dataset

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
    collate_fn=lambda batch: collate_batch(batch, device=default_device),
    drop_last=False
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    # pin_memory=True if torch.cuda.is_available() else False,
    collate_fn=lambda batch: collate_batch(batch, device=default_device),
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
_, _, _, batch_config, _ = collate_batch(test_batch, device=default_device)

# ==============================================================================
# Initialize the PSF model
# ==============================================================================
from tiptorch.PSF_models.NFM_wrapper import PSFModelNFM

if 'PSF_model' in locals():
    del PSF_model

with torch.no_grad():
    PSF_model = PSFModelNFM(
        batch_config,
        multiple_obs    = True,
        LO_NCPAs        = True,
        chrom_defocus   = False,
        use_Moffat      = False,
        retain_PSDs     = False,
        N_spline_nodes  = 5,
        Z_mode_max      = 9,
        device          = default_device
    )

# Initialization batch is no longer needed and can hold a large GPU allocation.
del test_batch, batch_config, random_idxs
release_gpu_memory(sync=True)

# Delete extra parameters that are pre-defined and thus don't need to be treated my the manager
# Could be just disabled but instead deleted to save a bit of memory
PSF_model.inputs_manager.delete('Jxy')     # Always zero inside
PSF_model.inputs_manager.delete('bg_ctrl') # Already subtracted from data
PSF_model.inputs_manager.delete('dx_ctrl') # Astrometric shifts are managed externally
PSF_model.inputs_manager.delete('dy_ctrl') # -//-
PSF_model.inputs_manager.delete('L0')      # Managed by configs

PSF_model.inputs_manager.delete('F_norm') # No photometric correction per source
PSF_model.inputs_manager.delete('F_norm_λ_ctrl') # -//-
PSF_model.inputs_manager.delete('src_dirs_x') # Only on-axis sources used for training
PSF_model.inputs_manager.delete('src_dirs_y') # -//-

PSF_model.inputs_manager.delete('wind_speed_single') # Wind vector is not predicted
PSF_model.inputs_manager.delete('wind_dir_single') # -//-

if PSF_model.use_Moffat:
    # PSF_model.inputs_manager.delete('beta')
    # PSF_model.inputs_manager.delete('b')
    PSF_model.inputs_manager.delete('theta')
    PSF_model.inputs_manager.delete('ratio')

PSF_model.inputs_manager.set_optimizable(['LO_coefs'],    predict_LO_NCPAs)
PSF_model.inputs_manager.set_optimizable(['Cn2_weights'], predict_Cn2_profile)   

print(PSF_model.inputs_manager)

# Note that the inputs manager for the calibrator is different from the default one,
# as some parameters are set externally (phase bump)
#%%
calibrator_outputs_transformer = deepcopy(PSF_model.inputs_manager.get_transformer()) # The calibrator predicts the same parameters as the default model, but some of them are removed or set externally before running the model

# Get the dict of input tensors from the default model, to change their dimensions
buf_x_dict = calibrator_outputs_transformer.unstack(PSF_model.inputs_manager.stack())

input_features  = dataset.features
output_features = [k for k in buf_x_dict.keys()]

# Remove phase bump from the input dict, as it's set externally and not predicted by the NN
if 'LO_coefs' in buf_x_dict:
    buf_x_dict['LO_coefs'] = buf_x_dict['LO_coefs'][:, 1:] 

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
# Cross-validation  - TODO: add k-fold CV for better estimates
from calibrators.NFM_calibrator import SmallCalibratorNet

# Initialize calibrator NN
calibrator = SmallCalibratorNet(
    n_features=N_features,
    n_outputs=N_outputs,
    hidden_dim=48,
    dropout_rate=0.2
).to(default_device)

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
        checkpoint = torch.load(path, map_location=default_device)
        calibrator.load_state_dict(checkpoint['calibrator_state_dict'])
        
        # Load astrometry parameters if they were saved
        if optimize_astrometry and 'dx' in checkpoint and 'dy' in checkpoint:
            dx = checkpoint['dx'].to(default_device)
            dy = checkpoint['dy'].to(default_device)
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

pretrain_weights_path = WEIGHTS_FOLDER / 'NFM_calibrator/pretrain_weights.pth'
should_run_pretraining = (not args.continue_training) or (args.continue_training and not pretrain_weights_path.exists())

if should_run_pretraining:
    if args.continue_training and not pretrain_weights_path.exists():
        logger.warning(f"Pre-training weights not found at {pretrain_weights_path}. Starting pre-training.")

    optimizer = optim.AdamW(calibrator.parameters(), lr=default_lr, weight_decay=5e-4)
    scheduler_pretrain = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-4)

    num_epochs_pretrain  = 100
    pretrain_patience    = 15
    pretrain_pat_counter = 0

    def calibrator_friendly_fit_dict(x_dict_fitted):
        ''' This function modifies the input dict to be more suitable for the calibrator NN, by removing parameters that are not predicted. '''
        x_dict_fitted = {key: x_dict_fitted[key] for key in output_features if key in x_dict_fitted} # Keep only the parameters that are predicted by the NN
        if 'LO_coefs' in x_dict_fitted:
            x_dict_fitted['LO_coefs'] = x_dict_fitted['LO_coefs'][:, 1:]
        # The dataset stores Cn2_weights without the first (GL) layer. Reconstruct it so the target matches the full-profile output the calibrator
        if 'Cn2_weights' in x_dict_fitted:
            cn2 = x_dict_fitted['Cn2_weights'].clamp(min=1e-6) # NOTE: clamp all values to min=1e-6 to avoid log(0) in SoftmaxInv transform
            GL_fraction = (1.0 - cn2.sum(dim=-1, keepdim=True)).clamp(min=1e-6)
            x_dict_fitted['Cn2_weights'] = torch.hstack((GL_fraction, cn2))
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
            
            batch_loss = loss.item()
            epoch_loss += batch_loss
            running_avg = epoch_loss / (batch_idx + 1)
            
            print(f"\r  Pre-train Epoch {epoch+1:3d}/{num_epochs_pretrain} | "
                  f"Batch {batch_idx+1:3d}/{len(train_loader)} | "
                  f"running_loss = {running_avg:.6f}",
                  end='', flush=True)
            
            # logger.debug(f"Pre-train Epoch {epoch+1}/{num_epochs_pretrain} | "
            #              f"Batch {batch_idx+1}/{len(train_loader)} | loss = {batch_loss:.6f}")
            
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

        is_best = avg_val_loss < best_pretrain_loss
        if is_best:
            best_pretrain_loss = avg_val_loss
            best_calibrator_state = deepcopy(calibrator.state_dict())
            pretrain_pat_counter = 0
        else:
            pretrain_pat_counter += 1

        current_lr = optimizer.param_groups[0]['lr']
        progress_msg = (f"Pre-train Epoch {epoch+1:3d}/{num_epochs_pretrain} | "
                        f"train_loss = {avg_loss:.6f} | val_loss = {avg_val_loss:.6f} | "
                        f"LR = {current_lr:.2e} | patience = {pretrain_pat_counter}/{pretrain_patience}" +
                        (" *" if is_best else ""))
        # Overwrite the running-loss line, then move to next line
        print(f"\r{progress_msg}")
        logger.info(progress_msg)

        if pretrain_pat_counter >= pretrain_patience:
            logger.info(f"Pre-train early stopping at epoch {epoch+1} (patience={pretrain_patience})")
            print(f"  Early stopping triggered at epoch {epoch+1}")
            break

    if best_calibrator_state is not None:
        calibrator.load_state_dict(best_calibrator_state)
        logger.info(f"Restored best pre-training weights (val_loss: {best_pretrain_loss:.6f})")

    # Store best pretrain calibrator state in the file
    torch.save(best_calibrator_state, pretrain_weights_path)
    logger.info(f"Saved best pre-training weights to {pretrain_weights_path}")
else:
    # Load best pre-training weights if continue_training flag is set
    best_pretrain_state = torch.load(pretrain_weights_path, map_location=default_device)
    calibrator.load_state_dict(best_pretrain_state)
    logger.info(f"🔄 Loaded pre-training weights from {pretrain_weights_path}")

# Release pre-training artifacts that can pin optimizer states in VRAM.
for _name in ('optimizer', 'scheduler_pretrain', 'loss_pretrain', 'best_calibrator_state', 'best_pretrain_state'):
    globals().pop(_name, None)

release_gpu_memory(sync=True)

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
    
    dx = torch.tensor(dx_data, device=default_device, dtype=torch.float32, requires_grad=optimize_astrometry)
    dy = torch.tensor(dy_data, device=default_device, dtype=torch.float32, requires_grad=optimize_astrometry)


if args.continue_training:
    if BEST_CALIB_PATH.exists():
        load_checkpoint(calibrator, None, BEST_CALIB_PATH)                      
    else:
        logger.info(f"ℹ️ Training model from the pre-trained weights (if available) or from scratch.")
        initialize_astrometry()
        
# Initialize NCPAs with fitted values if available, otherwise with median values from the fitting dataset (same for all samples).
LO_dataset = fill_from_fitted('LO_coefs')
phase_bump = torch.tensor(LO_dataset[:, 0], device=default_device, dtype=torch.float32)  # Extract phase bump values separately
LO_dataset = LO_dataset[:, 1:] # Remove phase bump from the dataset, leaving only Zernike coefficients

if predict_LO_NCPAs:
    logger.info(f"ℹ️ Prediction of modes Z3-Z{PSF_model.Z_mode_max} is enabled")
else:
    logger.info(f"ℹ️ Initializing NCPAs with median values from fitting (same for all samples)")
    NCPAs_median = torch.tensor(np.median(LO_dataset, axis=0)[None,...], device=default_device, dtype=torch.float32)


# %%
# To save GPU memory, PSFs are predicted and compared to data only for a subset of wavelengths at a time
λ_full = PSF_model.λ_sim.clone() # [nm]

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
# Now, we put the model in the loop and train it end-to-end to predict PSF cubes, using the full loss function that compares PSFs

def run_model(x_dict_NN, config, idx, λ_ids):
    ''' This function runs the PSF model to predict the batch of PSFs based on the dictionary of parameters predicted by the NN and the batch configuration. '''
    # NOTE: idx here are dataset ids, not the PSF samples ids
    # Get the current dict of input tensors from the PSF model's inputs manager    
    x_dict = PSF_model.inputs_manager.to_dict()
    # Update only the parameters that are predicted by the NN, keep the rest unchanged
    x_dict.update({key: x_dict_NN[key] for key in x_dict_NN if key in x_dict})
    # Set dx/dy for this batch (indexed by array's index)
    x_dict['dx_ctrl'] = dx[idx]
    x_dict['dy_ctrl'] = dy[idx]

    # Append phase bump
    x_dict['LO_coefs'] = torch.hstack( (phase_bump[idx].unsqueeze(-1), x_dict['LO_coefs']) ) if predict_LO_NCPAs \
                    else torch.hstack( (phase_bump[idx].unsqueeze(-1), NCPAs_median.repeat(len(idx),1))
    )
    current_wavelengths = λ_full[λ_ids].to(device=default_device)
    
    config['sources_science']['Wavelength'] = current_wavelengths.view(1,-1) # [1, N_wvl_selected]
    PSF_model.model.config = config
    PSF_model.SetWavelengths(current_wavelengths)
    
    # Do not update internal inputs_manager with graph-connected tensors to avoid cross-batch graph retention.
    return PSF_model(x_dict, update_params=False) # Run given the predicted parameters and the updated internal state, get the predicted PSF cubes

#%%
# Initialize training and validation loss function

λ_weighting = False # no weighting of PSFs at different wavelengths, as it may bias the fit towards the redder wavelengths with higher SNR

if λ_weighting:
    wvl_weights = torch.linspace(0.5, 1.0, N_wvl_total).to(default_device).view(1, N_wvl_total, 1, 1)
    wvl_weights = N_wvl_total / wvl_weights.sum() * wvl_weights # Normalize so that the total energy is preserved
else:
    wvl_weights = torch.ones((1, N_wvl_total, 1, 1), device=default_device)

# Enforce positive values for modal coefficients to mitigate sign ambiguity and improve convergence
force_positive = lambda x: torch.clamp(-x, min=0).pow(2).mean()
# TODO: positive r0 regularization?


def loss_PSF(PSF_data, PSF_pred, w_MSE, w_MAE):
    diff = PSF_pred - PSF_data #* wvl_weights[:, λ_ids, ...] #TODO: add wvl selection
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


def loss_fn_per_sample_core(PSF_data, PSF_pred, x_dict_pred):
    """
    Per-sample version of the full PSF objective.

    Returns:
        loss_per_sample: [B] tensor.

    This is used both for difficulty diagnostics and for weighted training.
    The scalar loss_fn below reduces it either uniformly or with sample weights.
    """
    diff = PSF_pred - PSF_data
    w = 2e4  # Empirical weight to balance the PSF loss magnitude with regularization terms

    MSE_loss = diff.pow(2).mean(dim=(1, 2, 3)) * 1200.0
    MAE_loss = diff.abs().mean(dim=(1, 2, 3))  * 1.6
    PSF_loss = w * (MSE_loss + MAE_loss)

    if predict_LO_NCPAs:
        coefs_vec = x_dict_pred['LO_coefs']
        LO_loss = coefs_vec.pow(2).sum(-1) * 1e-7
        phase_bump_positive = torch.clamp(-coefs_vec[:, 0], min=0).pow(2) * 5e-5
        first_defocus_penalty = torch.clamp(-coefs_vec[:, 2], min=0).pow(2) * 1e-7
        LO_loss_per = LO_loss + phase_bump_positive + first_defocus_penalty
    else:
        LO_loss_per = torch.zeros(PSF_data.shape[0], device=PSF_data.device, dtype=PSF_data.dtype)

    return PSF_loss + LO_loss_per


def loss_fn(PSF_data, PSF_pred, x_dict_pred, sample_weights=None, return_per_sample=False, eps=1e-12):
    """
    Full PSF objective. If sample_weights is supplied, returns a weighted mean.

    sample_weights should be a [B] tensor indexed by the global dataset IDs in the batch.
    Weights are normalized by their sum, so the loss scale stays comparable to the
    unweighted objective.
    """
    loss_per_sample = loss_fn_per_sample_core(PSF_data, PSF_pred, x_dict_pred)

    if return_per_sample:
        return loss_per_sample

    if sample_weights is None:
        return loss_per_sample.mean()

    w = sample_weights.to(device=loss_per_sample.device, dtype=loss_per_sample.dtype).view(-1)
    return (w * loss_per_sample).sum() / (w.sum() + eps)


#%%
def validate(loader=None, return_cubes=False, verbose=False):
    """
    Validaion function to evaluate the calibrator on the validation dataset.
    It can return either just the average loss, or the full predicted and data PSF cubes for all samples and wavelengths in addition.
    NOTE: validation dataset is ordered, so we can fill it batch by batch in the correct order to return full cubes without running out of memory.
    This is possible because we set shuffle=False and drop_last=False in the validation DataLoader.
    """
    if loader is None:
        loader = val_loader

    calibrator.eval()
    total_loss = 0
    total_simulated_batches = 0
    
    PSFs_pred_cube = None
    PSFs_data_cube = None
    NN_predictions = None
    telemetry_vecs = None
    
    validation_ids = []

    if return_cubes:
        N_samples = len(loader.dataset)

        PSFs_pred_cube = torch.zeros((N_samples, N_wvl_total, H, W), dtype=torch.float32, device='cpu')
        PSFs_data_cube = torch.zeros((N_samples, N_wvl_total, H, W), dtype=torch.float32, device='cpu')
        NN_predictions = torch.zeros((N_samples, N_outputs), dtype=torch.float32, device='cpu')
        telemetry_vecs = torch.zeros((N_samples, N_features), dtype=torch.float32, device='cpu')

        # Map global dataset IDs to local positions within this loader's dataset split.
        if isinstance(loader.dataset, Subset):
            local_to_global = torch.tensor(loader.dataset.indices, dtype=torch.long)
        else:
            local_to_global = torch.arange(N_samples, dtype=torch.long)
        
        global_to_local = { int(global_id): local_pos for local_pos, global_id in enumerate(local_to_global.tolist()) }

        if verbose:
            logger.info(f"Validation dataset size: {N_samples}, batches: {len(loader)}")
    
    with torch.no_grad():
        # Outer loop: iterate through batches
        for batch_idx, batch in enumerate(loader):
            PSF_data, telemetry_inputs, _, batch_config, idxs = batch
            
            # IDs are used directly to place data in the correct position in the full cubes
            # This is robust to shuffling and lack of ordering
            idxs_cpu = idxs.cpu()

            if return_cubes:
                validation_ids.extend(idxs_cpu.tolist())
                local_positions = torch.tensor([global_to_local[int(global_id)] for global_id in idxs_cpu.tolist()], dtype=torch.long)
            
            # Get calibrator predictions (same for all wavelength sets since prediction is λ-agnostic)
            x_pred = calibrator(telemetry_inputs)
            x_dict_pred = calibrator_outputs_transformer.unstack(x_pred)
            
            if return_cubes:
                NN_predictions[local_positions, :] = x_pred.cpu()
                telemetry_vecs[local_positions, :] = telemetry_inputs.cpu()
            
            if verbose: logger.info(f"Processed batch {batch_idx + 1}/{len(loader)}")
            
            # Inner loop: iterate through wavelength sets
            for current_λ_set_id in range(len(λ_sets)):
                λ_id_set = λ_id_sets[current_λ_set_id]  # Which wavelength indices for this set
                
                # Run model for this wavelength set
                PSF_pred = run_model(x_dict_pred, batch_config, idxs, λ_id_set)
                
                # Calculate loss for this batch and wavelength set
                loss = loss_fn(PSF_data[:, λ_id_set, ...], PSF_pred, x_dict_pred)
                total_loss += loss.item()
                total_simulated_batches += 1
                
                if return_cubes:
                    # Fill in the wavelengths for this batch
                    for wvl_idx, λ_id in enumerate(λ_id_set):
                        PSFs_pred_cube[local_positions, λ_id, :, :] = PSF_pred[:, wvl_idx, :, :].cpu()
                        PSFs_data_cube[local_positions, λ_id, :, :] = PSF_data[:, λ_id, :, :].cpu()
                
                if verbose: logger.info(f"  >> λ set {current_λ_set_id + 1}/{len(λ_sets)}")

    if verbose and return_cubes:
        logger.info(f"PSF pred range: [{PSFs_pred_cube.min():.4e}, {PSFs_pred_cube.max():.4e}]")
        logger.info(f"PSF data range: [{PSFs_data_cube.min():.4e}, {PSFs_data_cube.max():.4e}]")

    if return_cubes:
        # Return global dataset IDs matching the row order in returned cubes/predictions.
        validation_ids = local_to_global
        return PSFs_pred_cube, PSFs_data_cube, validation_ids, NN_predictions, telemetry_vecs, total_loss / total_simulated_batches if total_simulated_batches > 0 else 0.0
    else:
        return total_loss / total_simulated_batches if total_simulated_batches > 0 else 0.0

#%%
# Validate after calibrator pre-training, before the main training loop, to check that everything is working and to have a baseline for comparison
val_loss_total = validate(loader=val_loader, verbose=True, return_cubes=False)
release_gpu_memory(sync=True)

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
    factor   = 0.5,
    patience = 7,
    min_lr   = 1e-6
)

logger.info(f"Scheduler: ReduceLROnPlateau with patience={7}, factor={0.5}, min_lr={1e-6}")


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
def train(num_epochs=50, patience=10, nan_recovery=True, max_nan_recoveries=3,
          train_loader_override=None, val_loader_override=None, sample_weights=None,
          checkpoint_path=None):
    """
    Training loop with validation where dx/dy are optimized during validation.
    
    Args:
        num_epochs: Maximum number of training epochs
        patience: Early stopping patience
        nan_recovery: If True, recover from NaN by loading last good checkpoint
        max_nan_recoveries: Maximum number of times to recover from NaN before giving up
    """
    train_loader_used = train_loader if train_loader_override is None else train_loader_override
    val_loader_used = val_loader if val_loader_override is None else val_loader_override
    checkpoint_path = BEST_CALIB_PATH if checkpoint_path is None else Path(checkpoint_path)

    if sample_weights is not None and not torch.is_tensor(sample_weights):
        sample_weights = torch.as_tensor(sample_weights, dtype=torch.float32, device=default_device)
    elif sample_weights is not None:
        sample_weights = sample_weights.to(device=default_device, dtype=torch.float32)

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
    if sample_weights is not None:
        logger.info(f"Difficulty weighting enabled: min={sample_weights.min().item():.3f}, mean={sample_weights.mean().item():.3f}, max={sample_weights.max().item():.3f}")

    for epoch in range(num_epochs):
        # ========== TRAINING ==========
        calibrator.train()
        epoch_train_loss = 0
        train_batch_count = 0
        nan_detected_this_epoch = False
        
        for batch_idx, batch in enumerate(train_loader_used):
            PSF_data, telemetry_inputs, _, batch_config, idxs = batch

            optimizer.zero_grad()
            batch_loss_value = 0.0

            # Backpropagate per wavelength subset to reduce peak graph memory.
            for current_λ_set_id in range(len(λ_sets)):
                λ_id_set = λ_id_sets[current_λ_set_id]

                x_pred = calibrator(telemetry_inputs)
                x_dict_pred = calibrator_outputs_transformer.unstack(x_pred)

                PSF_pred = run_model(x_dict_pred, batch_config, idxs, λ_id_set)
                batch_sample_weights = sample_weights[idxs] if sample_weights is not None else None
                loss = loss_fn(PSF_data[:, λ_id_set, ...], PSF_pred, x_dict_pred,
                               sample_weights=batch_sample_weights)

                # ========== NaN detection ==========
                if check_for_nan(loss, calibrator, epoch, batch_idx):
                    nan_detected_this_epoch = True

                    if nan_recovery and nan_recovery_count < max_nan_recoveries:
                        nan_recovery_count += 1
                        loss_stats['nan_recoveries'] = nan_recovery_count
                        logger.error(f"🚨 NaN detected! Recovery attempt {nan_recovery_count}/{max_nan_recoveries}")

                        try:
                            # Load last good checkpoint
                            load_checkpoint(calibrator, optimizer, checkpoint_path)
                            # Reduce learning rate
                            for param_group in optimizer.param_groups:
                                old_lr = param_group['lr']
                                param_group['lr'] = old_lr * lr_decay
                                logger.warning(f"Reduced LR: {old_lr:.2e} -> {param_group['lr']:.2e}")

                            logger.info(f"✅ Recovered from checkpoint, LR reduced by {(1.-lr_decay) * 100:.0f}%")
                            logger.info("Model recovered from checkpoint, continuing training...")
                            optimizer.zero_grad(set_to_none=True)
                            break  # Skip rest of this epoch, start fresh

                        except FileNotFoundError:
                            logger.error("❌ No checkpoint found! Cannot recover from NaN.")
                            raise ValueError("NaN detected but no checkpoint to recover from")
                    else:
                        logger.error(f"❌ Max NaN recoveries ({max_nan_recoveries}) reached or recovery disabled. Stopping training.")
                        raise ValueError(f"Training failed due to NaN after {nan_recovery_count} recovery attempts")

                loss.backward()
                batch_loss_value += loss.item()

                # Explicitly clear large intermediates between wavelength subsets.
                del PSF_pred, x_pred, x_dict_pred, loss

            if nan_detected_this_epoch:
                break

            # Gradient clipping (helps prevent NaN)
            torch.nn.utils.clip_grad_norm_(calibrator.parameters(), max_norm=1.0)

            if optimize_astrometry:
                torch.nn.utils.clip_grad_norm_([dx, dy], max_norm=10.0)  # Clip dx/dy too

            # Update calibrator and optionally dx/dy
            optimizer.step()

            epoch_train_loss += batch_loss_value
            train_batch_count += 1

            # Log batch loss
            loss_stats['train_losses_per_batch'].append(batch_loss_value)

            # Running loss info
            current_lr = optimizer.param_groups[0]['lr']
            logger.debug(f"Epoch {epoch}, batch {batch_idx + 1}/{len(train_loader_used)}: "
                         f"train_loss = {batch_loss_value:.6f}, LR = {current_lr:.2e}")

        release_gpu_memory(sync=True)
    
        # If NaN was detected, skip to next epoch
        if nan_detected_this_epoch:
            logger.warning(f"Epoch {epoch} skipped due to NaN recovery")
            continue
        
        avg_train_loss = epoch_train_loss / train_batch_count if train_batch_count > 0 else 0
        train_losses.append(avg_train_loss)
        loss_stats['train_losses_per_epoch'].append(avg_train_loss)
        
        logger.info(f"Epoch {epoch}: Average training loss = {avg_train_loss:.6f}")
        
        
        # ========== VALIDATION ==========
        val_loss = validate(loader=val_loader_used, return_cubes=False)

        # Check for NaN in validation
        if np.isnan(val_loss) or np.isinf(val_loss):
            logger.error(f"❌ NaN/Inf in validation loss at epoch {epoch}")
            if nan_recovery and nan_recovery_count < max_nan_recoveries:
                nan_recovery_count += 1
                logger.warning(f"Attempting recovery from validation NaN...")
                load_checkpoint(calibrator, optimizer, checkpoint_path)
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
            save_checkpoint(calibrator, dx, dy, optimizer, epoch, avg_train_loss, val_loss, checkpoint_path)
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
    checkpoint = torch.load(checkpoint_path)
    calibrator.load_state_dict(checkpoint['calibrator_state_dict'])

    if optimize_astrometry and 'dx' in checkpoint and 'dy' in checkpoint:
        dx.data = checkpoint['dx']
        dy.data = checkpoint['dy']
    
    logger.info(f"Best validation loss: {checkpoint['val_loss']:.6f} at epoch {checkpoint['epoch']}")
    print(f"Best validation loss: {checkpoint['val_loss']:.6f} at epoch {checkpoint['epoch']}")
    
    return calibrator, train_losses, val_losses


#%%
def make_loader(indices, shuffle, batch_size=BATCH_SIZE):
    """Create a DataLoader over a Subset while preserving global dataset indices in batches."""
    return DataLoader(
        dataset=Subset(dataset, np.asarray(indices, dtype=int)),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=lambda batch: collate_batch(batch, device=default_device),
        drop_last=False,
    )


def make_optimizer_and_scheduler(model, lr=default_lr):
    """Create the optimizer/scheduler pair used by the main PSF training loop."""
    if optimize_astrometry:
        opt = optim.AdamW([
            {'params': model.parameters(), 'lr': lr, 'weight_decay': 5e-4},
            {'params': [dx, dy], 'lr': 1e-3, 'weight_decay': 1e-5},
        ], lr=1e-3)
    else:
        opt = optim.AdamW([
            {'params': model.parameters(), 'lr': lr, 'weight_decay': 5e-4},
        ], lr=lr)

    sch = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=7, min_lr=1e-6
    )
    return opt, sch


def reset_calibrator_for_fold(initial_state_dict):
    """Reset the global calibrator/optimizer/scheduler for a fold or final retraining."""
    global calibrator, optimizer, scheduler
    calibrator = SmallCalibratorNet(
        n_features=N_features,
        n_outputs=N_outputs,
        hidden_dim=48,
        dropout_rate=0.2,
    ).to(default_device)
    calibrator.load_state_dict(deepcopy(initial_state_dict))
    optimizer, scheduler = make_optimizer_and_scheduler(calibrator, lr=default_lr)
    return calibrator, optimizer, scheduler


def telemetry_matrix_from_indices(indices):
    """Return telemetry feature matrix [N, D] in the dataset feature order."""
    return np.asarray([
        np.asarray(list(dataset.telemetry[int(i)].values()), dtype=np.float32)
        for i in indices
    ], dtype=np.float32)


def difficulty_to_weights(pred_log_loss, alpha=0.30, clip=(0.5, 2.0)):
    """
    Convert RF-predicted log10(loss) difficulty into normalized sample weights.

    Because the input is already a log-loss, exponentiating the centered difficulty
    produces gentle multiplicative weights. alpha controls the strength.
    """
    d = np.asarray(pred_log_loss, dtype=np.float64)
    d_centered = d - np.median(d)
    w = np.exp(alpha * d_centered)
    w = w / np.mean(w)
    w = np.clip(w, clip[0], clip[1])
    w = w / np.mean(w)
    return w.astype(np.float32)


def evaluate_per_sample_losses(loader):
    """
    Evaluate unweighted per-sample losses for every sample in loader.

    Returns:
        global_ids: [N] global dataset indices in loader order
        losses:     [N] average loss over wavelength subsets
    """
    calibrator.eval()

    if isinstance(loader.dataset, Subset):
        global_ids = np.asarray(loader.dataset.indices, dtype=int)
    else:
        global_ids = np.arange(len(loader.dataset), dtype=int)

    loss_sum = {int(i): 0.0 for i in global_ids}
    loss_count = {int(i): 0 for i in global_ids}

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            PSF_data, telemetry_inputs, _, batch_config, idxs = batch
            x_pred = calibrator(telemetry_inputs)
            x_dict_pred = calibrator_outputs_transformer.unstack(x_pred)
            idxs_cpu = idxs.detach().cpu().numpy().astype(int)

            for λ_id_set in λ_id_sets:
                PSF_pred = run_model(x_dict_pred, batch_config, idxs, λ_id_set)
                per = loss_fn(
                    PSF_data[:, λ_id_set, ...], PSF_pred, x_dict_pred,
                    return_per_sample=True,
                ).detach().cpu().numpy()

                for global_id, loss_value in zip(idxs_cpu, per):
                    loss_sum[int(global_id)] += float(loss_value)
                    loss_count[int(global_id)] += 1

                del PSF_pred, per

            del x_pred, x_dict_pred

    losses = np.asarray([
        loss_sum[int(i)] / max(loss_count[int(i)], 1)
        for i in global_ids
    ], dtype=np.float32)
    return global_ids, losses


def run_kfold_difficulty_weighting(
    base_indices,
    initial_state_dict,
    n_splits=5,
    fold_epochs=120,
    fold_patience=12,
    alpha=0.30,
    clip=(0.5, 2.0),
    random_state=42,
):
    """
    Estimate out-of-fold PSF losses, fit RF(telemetry -> log10 loss), and return
    dataset-length weights for final training.

    Only base_indices are weighted/fitted; all other samples receive weight 1.0.
    This avoids leaking the held-out test split into the difficulty model.
    """
    base_indices = np.asarray(base_indices, dtype=int)
    n_splits = min(int(n_splits), len(base_indices))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    oof_losses = np.full(len(dataset), np.nan, dtype=np.float32)
    fold_summaries = []
    kfold_dir = WEIGHTS_FOLDER / 'NFM_calibrator' / 'kfold_difficulty'
    kfold_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"Starting {n_splits}-fold out-of-fold difficulty estimation")
    logger.info("=" * 60)

    for fold_id, (tr_rel, va_rel) in enumerate(kf.split(base_indices)):
        fold_train_idx = base_indices[tr_rel]
        fold_val_idx = base_indices[va_rel]

        logger.info(f"Fold {fold_id+1}/{n_splits}: train={len(fold_train_idx)}, val={len(fold_val_idx)}")
        reset_calibrator_for_fold(initial_state_dict)

        fold_train_loader = make_loader(fold_train_idx, shuffle=True)
        fold_val_loader = make_loader(fold_val_idx, shuffle=False)
        fold_ckpt = kfold_dir / f'fold_{fold_id:02d}_best.pth'

        train(
            num_epochs=fold_epochs,
            patience=fold_patience,
            nan_recovery=True,
            max_nan_recoveries=5,
            train_loader_override=fold_train_loader,
            val_loader_override=fold_val_loader,
            sample_weights=None,
            checkpoint_path=fold_ckpt,
        )

        # train() reloads the best fold checkpoint, so these are out-of-fold losses.
        val_ids, val_losses_fold = evaluate_per_sample_losses(fold_val_loader)
        oof_losses[val_ids] = val_losses_fold

        fold_summary = {
            'fold': fold_id,
            'train_indices': fold_train_idx,
            'val_indices': fold_val_idx,
            'val_loss_mean': float(np.mean(val_losses_fold)),
            'val_loss_median': float(np.median(val_losses_fold)),
        }
        fold_summaries.append(fold_summary)
        logger.info(
            f"Fold {fold_id+1}/{n_splits} OOF loss: "
            f"mean={fold_summary['val_loss_mean']:.6f}, "
            f"median={fold_summary['val_loss_median']:.6f}"
        )
        release_gpu_memory(sync=True)

    valid = np.isfinite(oof_losses[base_indices])
    if not np.all(valid):
        missing = base_indices[~valid]
        raise RuntimeError(f"Missing OOF losses for {len(missing)} samples: {missing[:10]}")

    X = telemetry_matrix_from_indices(base_indices)
    y = np.log10(oof_losses[base_indices] + 1e-12)

    rf = RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=-1)
    cv_splits = min(n_splits, len(base_indices))
    rf_scores = cross_val_score(rf, X, y, cv=cv_splits, scoring='r2')
    logger.info(f"RF difficulty CV R2 on OOF log-loss: {rf_scores.mean():.4f} ± {rf_scores.std():.4f}")

    rf.fit(X, y)
    pred_log_loss = rf.predict(X)
    weights_train = difficulty_to_weights(pred_log_loss, alpha=alpha, clip=clip)

    weights_all = np.ones(len(dataset), dtype=np.float32)
    weights_all[base_indices] = weights_train

    stats = {
        'base_indices': base_indices,
        'oof_losses': oof_losses,
        'rf_cv_r2_mean': float(rf_scores.mean()),
        'rf_cv_r2_std': float(rf_scores.std()),
        'pred_log_loss': pred_log_loss.astype(np.float32),
        'weights_all': weights_all,
        'fold_summaries': fold_summaries,
        'alpha': float(alpha),
        'clip': tuple(float(c) for c in clip),
        'feature_names': input_features,
        'rf_feature_importances': rf.feature_importances_.astype(np.float32),
    }

    stats_path = kfold_dir / 'difficulty_weighting_stats.pth'
    torch.save(stats, stats_path)
    logger.info(f"Saved difficulty-weighting diagnostics to {stats_path}")
    logger.info(
        f"Difficulty weights on training split: min={weights_train.min():.3f}, "
        f"mean={weights_train.mean():.3f}, max={weights_train.max():.3f}"
    )

    return torch.as_tensor(weights_all, dtype=torch.float32, device=default_device), stats


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

# Optional: estimate sample weights from K-fold out-of-fold losses before final training.
# The held-out test_idx split is not used to fit the RF difficulty model.
BASE_CALIBRATOR_STATE = deepcopy(calibrator.state_dict())
sample_weights_for_training = None
kfold_difficulty_stats = None

#%%
if args.run_kfold_difficulty:
    sample_weights_for_training, kfold_difficulty_stats = run_kfold_difficulty_weighting(
        base_indices=train_idx,
        initial_state_dict=BASE_CALIBRATOR_STATE,
        n_splits=args.kfolds,
        fold_epochs=args.kfold_epochs,
        fold_patience=args.kfold_patience,
        alpha=args.difficulty_alpha,
        clip=(args.difficulty_clip_min, args.difficulty_clip_max),
        random_state=42,
    )
    # Reset to the same starting point before the final weighted run.
    reset_calibrator_for_fold(BASE_CALIBRATOR_STATE)
else:
    logger.info("K-fold difficulty weighting is disabled. Use --run-kfold-difficulty to enable it.")

#%%
# Train the final model with optional RF difficulty weights.
calibrator, train_losses, val_losses = train(
    num_epochs=500,
    patience=20,
    nan_recovery=True,
    max_nan_recoveries=10,
    sample_weights=sample_weights_for_training,
    checkpoint_path=BEST_CALIB_PATH,
)

release_gpu_memory()

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
import numpy as np

def compare_by_bins(metric_baseline, metric_weighted, difficulty, n_bins=4):
    metric_baseline = np.asarray(metric_baseline)
    metric_weighted = np.asarray(metric_weighted)
    difficulty = np.asarray(difficulty)

    qs = np.quantile(difficulty, np.linspace(0, 1, n_bins + 1))

    print(f"{'bin':>4} {'N':>5} {'base mean':>12} {'w mean':>12} {'rel change':>12}")
    print("-" * 55)

    for k in range(n_bins):
        lo, hi = qs[k], qs[k + 1]
        if k == n_bins - 1:
            m = (difficulty >= lo) & (difficulty <= hi)
        else:
            m = (difficulty >= lo) & (difficulty < hi)

        b = metric_baseline[m].mean()
        w = metric_weighted[m].mean()
        rel = (w - b) / b

        print(f"{k:4d} {m.sum():5d} {b:12.4e} {w:12.4e} {rel:12.3%}")


compare_by_bins(
    metric_baseline=loss_baseline,
    metric_weighted=loss_weighted,
    difficulty=rf_pred_log_loss,
    n_bins=4,
)

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
        raise ValueError("tuned_params list must be specified for tuning")
    
    # Create reference model (frozen)
    calibrator_ref = deepcopy(calibrator)
    calibrator_ref.eval()
    for param in calibrator_ref.parameters():
        param.requires_grad = False
    
    # Create mask for tuned parameters
    tuned_slices = [calibrator_outputs_transformer.slices[param] for param in tuned_params]
    tuned_idx_mask = torch.zeros(N_outputs, dtype=torch.bool, device=default_device)
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
                 anchor_loss = torch.tensor(0.0, device=default_device)

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
                 
                 PSF_pred = run_model(x_dict_pred_loop, batch_config, idxs, λ_id_set)
                 
                 loss = loss_fn(PSF_data[:, λ_id_set, ...], PSF_pred, x_dict_pred_loop)
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
                     PSF_pred = run_model(x_dict_pred, batch_config, idxs, λ_id_set)
                     batch_val_loss += loss_fn(PSF_data[:, λ_id_set, ...], PSF_pred, x_dict_pred)
                
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
tune_calibrator(calibrator, train_loader, val_loader, tuned_params=['F_ctrl', 'dn'], num_epochs=20, lr=1e-3)

# %%
# if __name__ == "__main__":
#     main()

# Load best model
logger.info("Loading best model...")
print("\nLoading best model...")

epoch, train_loss, val_loss = load_checkpoint(calibrator, None, BEST_CALIB_PATH)

calibrator.eval()
# logger.info(f"Best validation loss: {val_loss:.6f} at epoch {epoch}")

# Validate and collect all PSF predictions and data
PSFs_pred_cube, PSFs_data_cube, validation_ids, NN_predictions, telemetry_vecs, final_val_loss = validate(return_cubes=True)

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
# torch.save(PSF_pred_cube,  WEIGHTS_FOLDER / 'NFM_calibrator/validation_PSFs_predicted.pt')
# torch.save(PSF_data_cube,  WEIGHTS_FOLDER / 'NFM_calibrator/validation_PSFs_data.pt')
# torch.save(validation_ids, WEIGHTS_FOLDER / 'NFM_calibrator/validation_sample_ids.pt')
# torch.save(PSF_data_cube,  WEIGHTS_FOLDER / 'NFM_calibrator/validation_PSFs_data.pt')

PSFs_pred_cube = PSFs_pred_cube.cpu()
PSFs_data_cube = PSFs_data_cube.cpu()
validation_ids = validation_ids.cpu()

release_gpu_memory()


#%%
from tools.plotting import plot_radial_PSF_profiles, draw_PSF_stack
import pickle

with open(STD_FOLDER / 'muse_df.pickle', 'rb') as handle:
    muse_df = pickle.load(handle)

id_src = np.random.randint(0, PSFs_data_cube.shape[0])
true_sample_id = validation_ids[id_src].item()

cube_filename = muse_df.loc[true_sample_id]['Filename']

# print(f"Randomly selected validation sample ID: {id_src}")
# print(f"Corresponding original dataset index: {true_sample_id}")
# print(f"Original cube filename: {true_sample_id}_{cube_filename}")

PSF_0 = PSFs_data_cube[id_src]
PSF_1 = PSFs_pred_cube[id_src]

vmin = np.percentile(PSF_0[PSF_0 > 0].cpu().numpy(), 10)
vmax = np.percentile(PSF_0[PSF_0 > 0].cpu().numpy(), 99.995)
wvl_select = np.s_[0, N_wvl_total//2, -1]
PSF_disp = lambda x, w: (x[w,...]).cpu().numpy()

#%
# plottis = True
# diff = (PSF_1 - PSF_0).abs()[wvl_select, ...].amax(dim=(1, 2)) / PSF_0[wvl_select, ...].amax(dim=(1, 2)) * 100.0
# print(f"Maximums {diff.median().item()} %")

#%
fig, ax = plt.subplots(1, len(wvl_select), figsize=(15, 1.35*len(wvl_select)))

p_errs = []
for i, lmbd in enumerate(wvl_select):
    p_errs.append(
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
        )[2].squeeze().max().item()
    )
    ax[i].set_title(f"λ = {(λ_full[lmbd]*1e9).round().int().item()} nm")
plt.show()

p_errs = np.array(p_errs)
print(p_errs)

draw_PSF_stack(
    PSF_0.numpy()[wvl_select, ...],
    PSF_1.numpy()[wvl_select, ...],
    average=True,
    min_val=vmin,
    max_val=vmax,
    crop=80,
    cmap='inferno'
)

_ax = plt.gca()
_crop = 80
for _i, (_lmbd, _p_err) in enumerate(zip(wvl_select, p_errs)):
    _wvl_nm = int((λ_full[_lmbd] * 1e9).round().item())
    _ax.text(3 * _crop - 2, _i * _crop + 4, f'λ={_wvl_nm} nm  ΔSR={_p_err:.1f}%',
             color='white', fontsize=6, va='top', ha='right',
             bbox=dict(boxstyle='round,pad=0.1', fc='black', alpha=0.4, lw=0))

plt.title(cube_filename)

# if plottis:
#     plt.savefig(STD_FOLDER / f'plots/predicted_PSF_{true_sample_id}.pdf', dpi=300)


#%%
fig, ax = plt.subplots(1, len(wvl_select), figsize=(15, 1.35*len(wvl_select)))
for i, lmbd in enumerate(wvl_select):
    plot_radial_PSF_profiles(
        PSFs_data_cube[:, lmbd, ...].cpu().numpy(),
        PSFs_pred_cube[:, lmbd, ...].cpu().numpy(),
        'Data',
        'TipTorch',
        cutoff=40,
        ax=ax[i],
        title=f"λ = {(λ_full[lmbd]*1e9).round().int().item()} nm"
        #, centers=center)
    )
plt.suptitle('Averaged radial PSF Profiles at selected Wavelengths')
plt.show()

#%%
fig = plt.figure(figsize=(10, 6))
PSF_avg = lambda x: np.mean(x.cpu().numpy(), axis=1)
white_profiles = plot_radial_PSF_profiles(
    PSF_avg(PSFs_data_cube),
    PSF_avg(PSFs_pred_cube),
    'Data',
    'TipTorch',
    title='Spectrally averaged PSF',
    cutoff=40,
    ax=fig.add_subplot(111),
    return_profiles=True
)
plt.title('Spectrally averaged median PSF profile')
# plt.show()
# plt.savefig(STD_FOLDER / 'plots/calibrated_prediction_white.pdf', dpi=300)

#%%
SR_err = white_profiles[2].max(axis=1)
_ = plt.hist(SR_err, bins=20)
# Median line\
plt.axvline((np.median(SR_err)), color='r', linestyle='--', label=f'Median ΔSR = {np.median(SR_err):.2f}%')
plt.xlabel('Maximum ΔSR (%)')
plt.ylabel('Number of samples')
plt.title('Distribution of maximum Strehl Ratio errors across validation samples')
plt.grid(True)
plt.legend()
plt.xlim(0, None)
# plt.show()
# plt.savefig(STD_FOLDER / 'plots/calibrated_prediction_hist.pdf', dpi=300)

#%%

diff = (PSFs_pred_cube - PSFs_data_cube).abs().amax(dim=(-2,-1)) / PSFs_data_cube.amax(dim=(-2,-1)) * 100.0

diff = diff.mean(dim=-1).cpu().numpy()

_ = plt.hist(diff, bins=50)
plt.xlabel('Maximum ΔSR (%)')
plt.ylabel('Number of samples')
plt.title('Distribution of maximum Strehl Ratio errors across validation samples')
plt.grid(True)
plt.show()

#%%
def save_calibrator(path):
    """
    Save the complete calibrator state needed to restore inference,
    without requiring the training script to be re-run.
    """
    state = {
        # --- NN weights and architecture ---
        'net_state_dict':   calibrator.state_dict(),
        'net_arch': {
            'n_features':   N_features,
            'n_outputs':    N_outputs,
            'hidden_dim':   calibrator.network[0].out_features,   # read back, not hardcoded
            'dropout_rate': calibrator.network[3].p,
        },

        # --- Telemetry / parameters ---
        'input_feature_names':  input_features,   # ordered list, matches telemetry tensor columns
        'output_feature_names': output_features,  # ordered list of predicted parameter names
        # Saved via InputsTransformer.save() so no lambdas/closures leak into the pickle
        'outputs_transformer':  calibrator_outputs_transformer.save(),

        # --- Prediction parameters ---
        'predict_LO_NCPAs':    predict_LO_NCPAs,
        'predict_Cn2_profile': predict_Cn2_profile,
        'LO_modes_max':        PSF_model.Z_mode_max,
        'N_spline_nodes':      PSF_model.N_wvl_ctrl,
        'predict_phase_bump':  False,
    }

    torch.save(state, path)
    logger.info(f"Calibrator state saved to {path}")


save_calibrator(WEIGHTS_FOLDER / 'NFM_calibrator/NFM_calibrator_bundle.pth')

#%%
all_loader = DataLoader(
    dataset=dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=lambda batch: collate_batch(batch, device=default_device),
    drop_last=False
)

PSFs_pred_cube, PSFs_data_cube, _, NN_predictions, telemetry_vecs, _ = validate(loader=all_loader, return_cubes=True, verbose=False)

#%%
def loss_fn_per_sample(PSF_data, PSF_pred, x_dict_pred):
    """Per-sample version of loss_fn. Returns a 1-D tensor of shape [B] instead of a scalar.
    x_dict_pred may be a stacked [B, N_outputs] tensor or an already-unstacked dict."""
    if isinstance(x_dict_pred, torch.Tensor):
        x_dict_pred = calibrator_outputs_transformer.unstack(x_dict_pred)
    
    diff = PSF_pred - PSF_data                          # [B, C, H, W]
    w = 2e4
    # Reduce over all dims except the batch dim
    MSE_loss = diff.pow(2).mean(dim=(1, 2, 3)) * 1200.0
    MAE_loss = diff.abs() .mean(dim=(1, 2, 3)) * 1.6
    PSF_loss = w * (MSE_loss + MAE_loss)               # [B]

    if predict_LO_NCPAs:
        coefs_vec = x_dict_pred['LO_coefs']            # [B, N_modes]
        LO_loss              = coefs_vec.pow(2).sum(-1)              * 1e-7
        phase_bump_positive  = torch.clamp(-coefs_vec[:, 0], min=0).pow(2) * 5e-5
        first_defocus_pen    = torch.clamp(-coefs_vec[:, 2], min=0).pow(2) * 1e-7
        LO_loss_per          = LO_loss + phase_bump_positive + first_defocus_pen  # [B]
    else:
        LO_loss_per = torch.zeros(PSF_data.shape[0], device=PSF_data.device)

    return PSF_loss + LO_loss_per  # [B]


loss_SR = loss_fn_per_sample(PSFs_data_cube, PSFs_pred_cube, NN_predictions)

release_gpu_memory()

# loss_SR = torch.amax((PSFs_data_cube - PSFs_pred_cube).abs().mean(dim=1), dim=[-2,-1]) / torch.amax(PSFs_data_cube.abs().mean(dim=1), dim=[-2,-1])

#%%
num_best_samples  = 50
num_worst_samples = 15

best_sample_ids  = torch.topk(loss_SR, k=num_best_samples, largest=False).indices
worst_sample_ids = torch.topk(loss_SR, k=num_worst_samples, largest=True).indices

bad_border_start  = loss_SR[worst_sample_ids].min().item()
sus_boarder_start = 0.41

sus_ids = torch.where((loss_SR >= sus_boarder_start) & (loss_SR < bad_border_start))[0]

_ = plt.hist(loss_SR.cpu().numpy(), bins=50, log=True)
plt.axvline(loss_SR[best_sample_ids].max().item(), color='g', linestyle='--', label='Best samples (low loss)')
plt.axvline(bad_border_start, color='r', linestyle='--', label='Bad samples (high loss)')
plt.axvline(sus_boarder_start, color='y', linestyle='--', label='Suspicious samples (medium loss)')

print(f"Best predicted sample IDs (lowest loss): {best_sample_ids.cpu().numpy()}")
print(f"Worst predicted sample IDs (highest loss): {worst_sample_ids.cpu().numpy()}")
print(f"Suspicious sample IDs (medium loss): {sus_ids.cpu().numpy()}")


#%%
tel_vecs_best  = telemetry_vecs[best_sample_ids]
tel_vecs_worst = telemetry_vecs[worst_sample_ids]
tel_vecs_sus   = telemetry_vecs[sus_ids]

# Single summary plot: median ± 1-sigma quantiles per feature for best / sus / worst samples
def feature_stats(vecs):
    arr = vecs.cpu().numpy()
    med = np.median(arr, axis=0)
    q16 = np.percentile(arr, 16, axis=0)
    q84 = np.percentile(arr, 84, axis=0)
    return med, q16, q84

med_best,  q16_best,  q84_best  = feature_stats(tel_vecs_best)
med_worst, q16_worst, q84_worst = feature_stats(tel_vecs_worst)
med_sus,   q16_sus,   q84_sus   = feature_stats(tel_vecs_sus)

x = np.arange(len(input_features))
width = 0.125

groups = [
    (med_best,  q16_best,  q84_best,  'Best (low loss)',    'tab:green'),
    (med_sus,   q16_sus,   q84_sus,   'Suspicious',         'tab:orange'),
    (med_worst, q16_worst, q84_worst, 'Worst (high loss)',  'tab:red'),
]

fig, ax = plt.subplots(figsize=(max(12, len(input_features) * 0.5), 5))
for offset, (med, q16, q84, label, color) in enumerate(groups):
    pos = x + (offset - 1) * width
    ax.errorbar(pos, med,
                yerr=[med - q16, q84 - med],
                fmt='o', capsize=4, markersize=5,
                color=color, label=label, linestyle='none')

ax.set_xticks(x)
ax.set_xticklabels(input_features, rotation=45, ha='right')
ax.set_ylabel('Feature value')
ax.set_title('Feature distributions: median ± 1σ quantile (best / suspicious / worst samples)')
ax.legend()
ax.grid(True, axis='y', alpha=0.4)
plt.tight_layout()
plt.show()

#%%
# Scatter plot: selected feature value vs loss
# feature_to_plot = 'Derot. angle'  # Change to any feature name in input_features
feature_to_plot = 'theta0'  # Change to any feature name in input_features

if feature_to_plot in input_features:
    feat_idx = input_features.index(feature_to_plot)
    feat_vals = telemetry_vecs[:, feat_idx].cpu().numpy()
    loss_vals = loss_SR.cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(feat_vals, loss_vals, s=12, alpha=0.6, c=loss_vals, cmap='viridis')
    plt.colorbar(sc, ax=ax, label='Loss')
    ax.set_xlabel(feature_to_plot)
    ax.set_ylabel('Loss')
    ax.set_title(f"Loss vs '{feature_to_plot}'")
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()
else:
    print(f"Feature '{feature_to_plot}' not found. Available features: {input_features}")

#%%
# Correlation of each feature with loss
from scipy.stats import pearsonr, spearmanr

loss_vals = loss_SR.cpu().numpy()
tel_np    = telemetry_vecs.cpu().numpy()

pearson_r  = []
spearman_r = []

for i in range(len(input_features)):
    pr, _ = pearsonr(tel_np[:, i],  loss_vals)
    sr, _ = spearmanr(tel_np[:, i], loss_vals)
    pearson_r.append(pr)
    spearman_r.append(sr)

pearson_r  = np.array(pearson_r)
spearman_r = np.array(spearman_r)

sort_idx = np.argsort(np.abs(spearman_r))[::-1]
x_corr   = np.arange(len(input_features))

fig, ax = plt.subplots(figsize=(max(12, len(input_features) * 0.9), 5))
ax.bar(x_corr - 0.2, pearson_r[sort_idx],  width=0.4, label='Pearson r',  alpha=0.8, color='tab:blue')
ax.bar(x_corr + 0.2, spearman_r[sort_idx], width=0.4, label='Spearman ρ', alpha=0.8, color='tab:orange')
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xticks(x_corr)
ax.set_xticklabels([input_features[i] for i in sort_idx], rotation=45, ha='right')
ax.set_ylabel('Correlation with loss')
ax.set_title('Feature–loss correlation (sorted by |Spearman ρ|)')
ax.legend()
ax.grid(True, axis='y', alpha=0.4)
plt.tight_layout()
plt.show()

# Print ranked table
print(f"{'Feature':<30} {'Pearson r':>10} {'Spearman ρ':>12}")
print("-" * 54)
for i in sort_idx:
    print(f"{input_features[i]:<30} {pearson_r[i]:>10.3f} {spearman_r[i]:>12.3f}")


#%%
# Example diagnostic
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# X_tel: [N, D] reduced telemetry
# losses: [N] validation/per-sample loss from current model
reg = RandomForestRegressor(n_estimators=300, random_state=0)
scores = cross_val_score(reg, telemetry_vecs.cpu().numpy(), np.log10(loss_SR.cpu().numpy() + 1e-12), cv=5, scoring="r2")

print("CV R2 predicting log-loss from telemetry:", scores.mean(), scores.std())


#%%
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# X_np = NN_predictions
X_np = telemetry_vecs

tsne = TSNE(
    n_components=2,
    metric="cosine",
    perplexity=50,
    init="pca",
    learning_rate="auto",
    random_state=0,
)


X_tsne = tsne.fit_transform(X_np)

# Color scatter by loss values using colormap
loss_values = loss_SR.log().cpu().numpy()

plt.figure(figsize=(7, 6))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=12, c=loss_values, cmap='viridis')
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.title("t-SNE projection, cosine metric")
plt.colorbar(scatter, label='Loss')
plt.grid(True, alpha=0.3)
plt.show()


#%%
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

X_np = telemetry_vecs
X_np = NN_predictions

tsne = TSNE(
    n_components=3,
    metric="euclidean",
    perplexity=50,
    init="pca",
    learning_rate="auto",
    random_state=0,
)

X_tsne = tsne.fit_transform(X_np)

loss_values = loss_SR.log().cpu().numpy()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], s=16, c=loss_values, cmap='viridis')
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
ax.set_zlabel("t-SNE 3")
ax.set_title("t-SNE projection (3D), euclidean metric")
plt.colorbar(scatter, label='Loss', ax=ax, pad=0.1)
plt.show()



#%% ============================================================================================================================================================================================
# Dummy optimization routine for debugging loss function and optimization behavior
def debug_dummy_optimization(num_iters=100, criterion=None, initial_guess=None):
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
    PSF_model.SetWavelengths(λ_sets[current_λ_set_id].to(device=default_device))
    λ_id_set = λ_id_sets[current_λ_set_id]
    
    logger.info(f"Using wavelength set {current_λ_set_id}: {len(λ_id_set)} wavelengths")
    
    if initial_guess is not None:
        # Start with calibrator prediction and add noise
        with torch.no_grad():
            dummy_vec = initial_guess.detach().clone().to(device=default_device)
            dummy_vec.requires_grad_(True)
        logger.info("Using initial guess")
    else:
        # Random initialization
        dummy_vec = torch.randn((PSF_cubes.shape[0], N_outputs), device=default_device, requires_grad=True)
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

PSFs_data_cube = results['PSF_cubes'].cpu()
PSFs_pred_cube = results['PSF_pred'].cpu()

x_pred = results['final_x_pred']
N_wvl_temp = PSFs_data_cube.shape[1]

dummy_vec = results['dummy_vec'].cpu()
x_test = calibrator_outputs_transformer.unstack(dummy_vec)

