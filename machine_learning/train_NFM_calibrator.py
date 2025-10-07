#%%
%reload_ext autoreload
%autoreload 2

import sys
sys.path.insert(0, '..')

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from sklearn.model_selection import train_test_split, KFold

from managers.config_manager import MultipleTargetsInDifferentObservations
from data_processing.MUSE_STD_dataset_utils import *

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATASET_CACHE = STD_FOLDER / 'dataset_cache'

N_wvl_total = 30

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
    cubes, vecs, confs, idxs = zip(*batch)
    cubes = torch.stack(cubes, 0).to(device=device, non_blocking=True)
    vecs = torch.stack(vecs, 0).to(device=device, non_blocking=True)
    
    batch_config = MultipleTargetsInDifferentObservations(list(confs), device=device)
    batch_config['PathPupil'] = str(DATA_FOLDER / 'calibrations/VLT_CALIBRATION/VLT_PUPIL/ut4pupil320.fits')
    
    return cubes, vecs, batch_config, idxs


#%%
NFM_dataset = NFMDataset()
print(f"Total dataset size: {len(NFM_dataset)}")

#%%
# Three-way split: Train/Val (for k-fold cross-validation (CV) + Test (held-out)
# Strategy: Hold-out test set first, then do k-fold CV on remaining data

def create_train_test_split(dataset, test_size=0.15, random_state=42):
    """
    Create initial train/test split with stratification if possible
    test_size=0.15 means 15% for test, 85% for train/val CV
    """
    total_size = len(dataset)
    indices = np.arange(total_size)
    
    # Try to stratify by seeing conditions if telemetry data is available
    try:
        # Get seeing values for stratification
        seeing_values = []
        for i in range(len(NFM_dataset)):
            _, _, conf, _ = NFM_dataset[i]
            conf['atmosphere']['Seeing']
            seeing_values.append(conf['atmosphere']['Seeing'].item())  # Assuming first element is seeing

        seeing_values = np.array(seeing_values)
        # Create stratification bins based on seeing quartiles
        seeing_bins = np.digitize(seeing_values, np.percentile(seeing_values, [25, 50, 75]))
        
        train_val_idx, test_idx = train_test_split(
            indices, 
            test_size=test_size, 
            stratify=seeing_bins,
            random_state=random_state
        )
        print(f"Stratified split by seeing conditions")
        
    except Exception as e:
        print(f"Stratification failed ({e}), using random split")
        train_val_idx, test_idx = train_test_split(
            indices, 
            test_size=test_size, 
            random_state=random_state
        )
    
    return train_val_idx, test_idx

# Create the initial split
train_val_indices, test_indices = create_train_test_split(NFM_dataset, test_size=0.15)

print(f"Train/Val indices: {len(train_val_indices)} ({len(train_val_indices)/len(NFM_dataset)*100:.1f}%)")
print(f"Test indices: {len(test_indices)} ({len(test_indices)/len(NFM_dataset)*100:.1f}%)")

# Create test dataset (never used during training/validation)
test_dataset = Subset(NFM_dataset, test_indices)
train_val_dataset = Subset(NFM_dataset, train_val_indices)

print(f"Test set size: {len(test_dataset)}")
print(f"Available for CV: {len(train_val_dataset)}")

#%%
# K-fold Cross-Validation setup on the train/val subset
def create_kfold_splits(dataset, indices, n_splits=5, random_state=42):
    """
    Create k-fold cross-validation splits from the train/val dataset
    Returns list of (train_subset, val_subset, fold_num) tuples
    """
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    cv_splits = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(indices)):
        # Convert back to original dataset indices
        fold_train_indices = indices[train_idx]
        fold_val_indices = indices[val_idx]
        
        # Create subsets
        fold_train_dataset = Subset(dataset, fold_train_indices)
        fold_val_dataset   = Subset(dataset, fold_val_indices)
        
        cv_splits.append({
            'fold': fold,
            'train_dataset': fold_train_dataset,
            'val_dataset':   fold_val_dataset,
            'train_indices': fold_train_indices,
            'val_indices':   fold_val_indices,
            'train_size':    len(fold_train_indices),
            'val_size':      len(fold_val_indices)
        })
        
        print(f"Fold {fold}: Train={len(fold_train_indices)}, Val={len(fold_val_indices)}")
    
    return cv_splits

# Create k-fold splits (typically 5-fold for small datasets)
n_folds = 5
cv_splits = create_kfold_splits(NFM_dataset, train_val_indices, n_splits=n_folds)

print(f"\nCreated {len(cv_splits)} CV folds")
print(f"Average train size per fold: {np.mean([split['train_size'] for split in cv_splits]):.1f}")
print(f"Average val size per fold: {np.mean([split['val_size'] for split in cv_splits]):.1f}")

#%%
# Data loader creation for k-fold CV
def create_fold_data_loaders(cv_split, batch_size=16, num_workers=0):
    """Create data loaders for a specific CV fold"""
    
    train_loader = DataLoader(
        dataset=cv_split['train_dataset'],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        # pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=lambda batch: collate_batch(batch, device=device),
        drop_last=True
    )
    
    val_loader = DataLoader(
        dataset=cv_split['val_dataset'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        # pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=lambda batch: collate_batch(batch, device=device),
        drop_last=False
    )
    
    return train_loader, val_loader

# Create test data loader (for final evaluation)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,
    # pin_memory=True if torch.cuda.is_available() else False,
    collate_fn=lambda batch: collate_batch(batch, device=device),
    drop_last=False
)

print(f"Test loader batches: {len(test_loader)}")

#%%
# Example training loop structure for k-fold CV
def train_with_kfold_cv(cv_splits, model_class, **training_kwargs):
    """
    Template for k-fold cross-validation training
    """
    fold_results = []
    
    for cv_split in cv_splits:
        print(f"\n{'='*50}")
        print(f"Training Fold {cv_split['fold']}")
        print(f"{'='*50}")
        
        # Create data loaders for this fold
        train_loader, val_loader = create_fold_data_loaders(cv_split)
        
        # Initialize fresh model for this fold
        model = model_class()  # Your model initialization here
        
        # Train the model on this fold
        fold_result = train_single_fold(
            model, train_loader, val_loader, 
            fold_num=cv_split['fold'],
            **training_kwargs
        )
        
        fold_results.append(fold_result)
    
    return fold_results


def train_single_fold(model, train_loader, val_loader, fold_num, num_epochs=50):
    """Train model on a single fold"""
    print(f"Fold {fold_num}: Training for {num_epochs} epochs")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Your training loop here
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            # Your training step here
            pass
        
        # Validation phase
        if epoch % 5 == 0:
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    # Your validation step here
                    pass
            
            print(f"Fold {fold_num}, Epoch {epoch}: Val Loss = {val_loss:.6f}")
    
    return {'fold': fold_num, 'best_val_loss': best_val_loss}

#%%
# Test the CV setup
print("Testing k-fold CV setup...")

# Test first fold
fold_0 = cv_splits[0]
train_loader_0, val_loader_0 = create_fold_data_loaders(fold_0, batch_size=4)

print(f"Fold 0 - Train batches: {len(train_loader_0)}, Val batches: {len(val_loader_0)}")

# Test data loading
for batch in train_loader_0:
    cubes, vecs, batch_config, idxs = batch
    print(f"Fold 0 train batch: Cubes {cubes.shape}, Vecs {vecs.shape}")
    break

for batch in val_loader_0:
    cubes, vecs, batch_config, idxs = batch
    print(f"Fold 0 val batch: Cubes {cubes.shape}, Vecs {vecs.shape}")
    break

# Test test set
for batch in test_loader:
    cubes, vecs, batch_config, idxs = batch
    print(f"Test batch: Cubes {cubes.shape}, Vecs {vecs.shape}")
    break

#%%
# Wavelength selection and parameter initialization
wvl_ids = np.clip(np.arange(0, (N_wvl_max:=N_wvl_total)+1, 2), a_min=0, a_max=N_wvl_max-1)

# Test wavelength selection with first fold training data
print("Testing wavelength selection...")
fold_0 = cv_splits[0]
train_loader_test, _ = create_fold_data_loaders(fold_0, batch_size=4)

for batch in train_loader_test:
    cubes, vecs, batch_config, idxs = batch
    cubes = cubes[:, wvl_ids, :, :]
    batch_config['sources_science']['Wavelength'] = batch_config['sources_science']['Wavelength'][:, wvl_ids]
    print(f"After wavelength selection: {cubes.shape}")
    break

# %%
# Initialize learnable parameters per sample
# Astrometric shifts and photometric backgrounds are learned by the network
print("Initializing learnable parameters...")

# Create parameter tensors for each sample in the dataset
dx = torch.zeros((len(NFM_dataset), N_wvl_total), dtype=torch.float32, requires_grad=True, device=device)
dy = torch.zeros((len(NFM_dataset), N_wvl_total), dtype=torch.float32, requires_grad=True, device=device)
b  = torch.zeros((len(NFM_dataset), N_wvl_total), dtype=torch.float32, requires_grad=True, device=device)

print(f"Parameter shapes: dx={dx.shape}, dy={dy.shape}, b={b.shape}")

#%%
# Alternative: Create parameter dictionaries for train/val splits
def create_split_parameters(train_indices, val_indices, n_wavelengths, device):
    """Create separate parameter tensors for train and validation sets"""
    
    train_params = {
        'dx': torch.zeros((len(train_indices), n_wavelengths), dtype=torch.float32, requires_grad=True, device=device),
        'dy': torch.zeros((len(train_indices), n_wavelengths), dtype=torch.float32, requires_grad=True, device=device),
        'b':  torch.zeros((len(train_indices), n_wavelengths), dtype=torch.float32, requires_grad=True, device=device)
    }
    
    val_params = {
        'dx': torch.zeros((len(val_indices), n_wavelengths), dtype=torch.float32, requires_grad=False, device=device),
        'dy': torch.zeros((len(val_indices), n_wavelengths), dtype=torch.float32, requires_grad=False, device=device),
        'b':  torch.zeros((len(val_indices), n_wavelengths), dtype=torch.float32, requires_grad=False, device=device)
    }
    
    return train_params, val_params

# Example usage (uncomment if you want split parameters)
# if hasattr(train_dataset, 'indices') and hasattr(val_dataset, 'indices'):
#     train_params, val_params = create_split_parameters(
#         train_dataset.indices, val_dataset.indices, N_wvl_total, device
#     )
#     print(f"Split parameters created - Train: {len(train_dataset.indices)}, Val: {len(val_dataset.indices)}")

#%%
# K-fold Cross-Validation Training Configuration
cv_training_config = {
    'n_folds': n_folds,
    'num_epochs_per_fold': 50,  # Fewer epochs per fold since you have multiple folds
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'batch_size': 16,
    'validate_every': 5,
    'early_stopping_patience': 15,  # Per fold
    'save_best_models': True,  # Save best model from each fold
    'final_test_evaluation': True,  # Evaluate on test set after CV
}

print("K-fold CV Training configuration:")
for key, value in cv_training_config.items():
    print(f"  {key}: {value}")

print(f"\nTotal training: {cv_training_config['n_folds']} folds × {cv_training_config['num_epochs_per_fold']} epochs = {cv_training_config['n_folds'] * cv_training_config['num_epochs_per_fold']} total epochs")

#%%
# Model evaluation strategy for k-fold CV
def evaluate_kfold_results(fold_results, test_loader=None):
    """
    Evaluate and summarize k-fold cross-validation results
    """
    print(f"\n{'='*60}")
    print("K-FOLD CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    
    val_losses = [result['best_val_loss'] for result in fold_results]
    
    print(f"Validation Loss Statistics:")
    print(f"  Mean: {np.mean(val_losses):.6f}")
    print(f"  Std:  {np.std(val_losses):.6f}")
    print(f"  Min:  {np.min(val_losses):.6f}")
    print(f"  Max:  {np.max(val_losses):.6f}")
    
    # Find best fold
    best_fold_idx = np.argmin(val_losses)
    print(f"  Best fold: {best_fold_idx} (loss: {val_losses[best_fold_idx]:.6f})")
    
    # If test set is provided, evaluate final performance
    if test_loader is not None:
        print("Final Test Set Evaluation:")
        # Load best model and evaluate on test set
        # Implementation depends on your model saving strategy
        print("  Test evaluation would go here...")
    
    return {
        'mean_val_loss': np.mean(val_losses),
        'std_val_loss': np.std(val_losses),
        'best_fold': best_fold_idx,
        'fold_results': fold_results
    }


