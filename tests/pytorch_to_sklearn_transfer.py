#%%
%reload_ext autoreload
%autoreload 2


import pickle
import os
import numpy as np
import torch
import shap
import torch.nn as nn
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tools.utils import mask_circle
from data_processing.normalizers import Uniform
from managers.input_manager import InputsManager
from machine_learning.calibrator import Calibrator, Gnosis
from data_processing.normalizers import CreateTransformSequenceFromFile
from data_processing.MUSE_data_settings import MUSE_DATA_FOLDER, MUSE_DATASET_FOLDER
from data_processing.MUSE_onsky_df import *
from tools.utils import rad2mas

device = torch.device('cpu')

#%%
df_transforms_onsky  = CreateTransformSequenceFromFile('../data/temp/muse_df_norm_transforms.pickle')
df_transforms_fitted = CreateTransformSequenceFromFile('../data/temp/muse_df_fitted_transforms.pickle')

inputs_manager = InputsManager()

N_wvl = 7

inputs_manager.add('r0',    torch.tensor([0.16]),         df_transforms_fitted['r0'])
inputs_manager.add('F',     torch.tensor([[1.0,]*N_wvl]), df_transforms_fitted['F'])
inputs_manager.add('dn',    torch.tensor([1.5]),          df_transforms_fitted['dn'])
inputs_manager.add('Jx',    torch.tensor([[10,]*N_wvl]),  df_transforms_fitted['Jx'])
inputs_manager.add('Jy',    torch.tensor([[10,]*N_wvl]),  df_transforms_fitted['Jy'])
inputs_manager.add('s_pow', torch.tensor([0.0]),          df_transforms_fitted['s_pow'])
inputs_manager.add('amp',   torch.tensor([0.0]),          df_transforms_fitted['amp'])
inputs_manager.add('b',     torch.tensor([0.0]),          df_transforms_fitted['b'])
inputs_manager.add('alpha', torch.tensor([4.5]),          df_transforms_fitted['alpha'])

inputs_manager.to_float()
inputs_manager.to(device)

print(inputs_manager)


#%%
with open(MUSE_DATA_FOLDER+'muse_df_norm_imputed.pickle', 'rb') as handle:
    muse_df_norm = pickle.load(handle)

# Open pickle file
with open(MUSE_RAW_FOLDER+'../muse_df.pickle', 'rb') as handle:
    muse_df = pickle.load(handle)

muse_df_pruned  = prune_columns(muse_df.copy())
muse_df_reduced = reduce_columns(muse_df_pruned.copy())

selected_entries_input = muse_df_norm.columns.values.tolist()

df_transforms = CreateTransformSequenceFromFile('../data/temp/muse_df_norm_transforms.pickle')

# Load processed data file
with open(MUSE_DATA_FOLDER + f"quasars/J0259_reduced/J0259_2024-12-05T03_15_37.598.pickle", 'rb') as f:
    data_sample = pickle.load(f)

df = data_sample['All data']
df['ID'] = 0
df.loc[0, 'Pupil angle'] = 0.0

df_pruned  = prune_columns(df.copy())
df_reduced = reduce_columns(df_pruned.copy())

df_norm = normalize_df(df_reduced, df_transforms)
df_norm = df_norm.fillna(0)

X = df_norm[selected_entries_input].loc[0].to_numpy().reshape(1,-1)

#%%
calibrator = Calibrator(
    inputs_manager=inputs_manager,
    predicted_values = ['r0', 'F', 'dn', 'Jx', 'Jy', 's_pow', 'amp', 'b', 'alpha'],
    device=device,
    calibrator_network = {
        'artichitecture': Gnosis,
        'inputs_size': len(selected_entries_input),
        'NN_kwargs': {
            'hidden_size': 200,
            'dropout_p': 0.1
        },
        'weights_folder': '../data/weights/gnosis_MUSE_v3_7wvl_yes_Mof_no_ssg.dict'
    }
)

calibrator.eval()


#%%
def extract_pytorch_weights(pytorch_model):
    """
    Extract weights and biases from an arbitrary PyTorch model by scanning through all linear layers
    """
    weights, biases = [], []

    # Iterate through all modules in the model
    for name, module in pytorch_model.named_modules():
        if isinstance(module, nn.Linear):
            weights.append(module.weight.data.numpy().T) # Extract weights (transposed to match sklearn's format)
            biases.append(module.bias.data.numpy().reshape(-1, 1)) # Extract bias and reshape to column vector

    # Make sure we found at least one layer
    if not weights:
        raise ValueError("No linear layers found in the model")

    return weights, biases


def transfer_to_sklearn(pytorch_model, X):
    """
    Transfer knowledge from PyTorch model to sklearn MLPRegressor
    by directly setting the weights and automatically determining the architecture
    """
    # Extract weights and biases from PyTorch model
    weights, biases = extract_pytorch_weights(pytorch_model)
    
    # Determine the hidden layer sizes from the extracted weights
    hidden_layer_sizes = []
    for i in range(len(weights) - 1):
        hidden_layer_sizes.append(weights[i].shape[1])

    # Initialize a new MLPRegressor with the derived architecture
    sklearn_model = MLPRegressor(
        hidden_layer_sizes=tuple(hidden_layer_sizes),
        activation='tanh',
        solver='adam',
        max_iter=1,  # We'll just set the weights, no need to train
        warm_start=True
    )

    # Fit the model with a single iteration to initialize the weights
    output_size = weights[-1].shape[1]
    dummy_y = np.zeros((X.shape[0], output_size))
    sklearn_model.fit(X, dummy_y)

    # Set the weights in the sklearn model directly
    sklearn_model.coefs_ = weights
    sklearn_model.intercepts_ = [b.flatten() for b in biases]
    
    return sklearn_model


PSF_calibrator = transfer_to_sklearn(calibrator.net, X)

#%%
with torch.no_grad():
    Y_torch = calibrator.net(torch.as_tensor(X, dtype=torch.float32)).numpy()
    
Y_sklearn = PSF_calibrator.predict(X)

print('Mean absolute difference between PyTorch and sklearn predictions:', np.abs(Y_sklearn - Y_torch).mean())

#%%
batches_train, batches_val = [], []
train_ids, val_ids = [], []
batches_train, batches_val = [], []


train_files = [ MUSE_DATASET_FOLDER+'train/'+file      for file in os.listdir(MUSE_DATASET_FOLDER+'train')      if '.pkl' in file ]
val_files   = [ MUSE_DATASET_FOLDER+'validation/'+file for file in os.listdir(MUSE_DATASET_FOLDER+'validation') if '.pkl' in file ]

print('Loading train batches...')
for file in tqdm(train_files):
    with open(file, 'rb') as handle:
        batches_train.append( pickle.load(handle) )
        
print('Loading validation batches...')
for file in tqdm(val_files):
    with open(file, 'rb') as handle:
        batches_val.append( pickle.load(handle) )

train_ids, valid_ids = [], []

for batch in batches_train: train_ids.extend(batch['IDs'])
for batch in batches_val:   valid_ids.extend(batch['IDs'])
    
train_ids.sort()
valid_ids.sort()

del batches_train, batches_val, train_files, val_files

#%%
X_train = torch.as_tensor(muse_df_norm[selected_entries_input].iloc[train_ids].to_numpy(), device=device).float()
X_test  = torch.as_tensor(muse_df_norm[selected_entries_input].iloc[valid_ids].to_numpy(), device=device).float()

explainer = shap.DeepExplainer(calibrator.net, X_train)
shap_values = explainer.shap_values(X_test, check_additivity=False)

shap_values = shap_values[0] if isinstance(shap_values, list) else shap_values

#%%
mean_shap_values = np.mean(shap_values, axis=-1)

# shap.summary_plot(shap_values[...,3], X_test, max_display=61, feature_names=selected_entries_input)
shap.summary_plot(mean_shap_values, X_test, max_display=61, feature_names=selected_entries_input)


#%%
# interaction_values = explainer.shap_interaction_values(X)
cor_matrix = []
for output_idx in range(shap_values.shape[-1]):
    shap_slice = shap_values[:, :, output_idx]  # shape: (N_samples, N_features)
    shap_df = pd.DataFrame(shap_slice, columns=selected_entries_input)
    cor_matrix.append( shap_df.corr().values )

cor_matrix = np.array(cor_matrix)
# cor_matrix = np.mean(cor_matrix, axis=0)
cor_matrix = cor_matrix[-1,...]

#%
# Plot heatmap
# Plot heatmap using imshow with nearest interpolation
plt.figure(figsize=(12, 10))
plt.imshow(cor_matrix, cmap='coolwarm', interpolation='nearest', vmin=-1, vmax=1)
plt.colorbar(label='Correlation')
plt.title('SHAP Value Correlation Matrix')

# Set ticks for both axes
plt.xticks(range(len(selected_entries_input)), selected_entries_input, rotation=90, fontsize=8)
plt.yticks(range(len(selected_entries_input)), selected_entries_input, fontsize=8)

# Annotate each cell with the correlation value
# for i in range(len(cor_matrix)):
#     for j in range(len(cor_matrix)):
#         text_color = 'white' if abs(cor_matrix.values[i, j]) > 0.5 else 'black'
#         plt.text(j, i, f'{cor_matrix.values[i, j]:.2f}',
#                  ha='center', va='center', color=text_color, fontsize=7)

plt.tight_layout()
plt.show()


#%%
# Define the features and targets for our MLP model
input_features = [
    'Wind dir (header)',
    'NGS mag (from ph.)',
    'Seeing (header)',
    'Airmass',
    'Tau0 (header)',
]


median_features = [
    'Tel. altitude',
    'Tel. azimuth',
    'RA (science)',
    'DEC (science)',
    'Derot. angle',
    'NGS RA',
    'NGS DEC',
    'window',
    'frequency',
    'gain',
    'plate scale, [mas/pix]',
    'conversion, [e-/ADU]',
    'RON, [e-]',
    'LGS1 photons, [photons/m^2/s]',
    'LGS2 photons, [photons/m^2/s]',
    'LGS3 photons, [photons/m^2/s]',
    'LGS4 photons, [photons/m^2/s]',
    'Exp. time',
    'Temperature (header)',
    'MASS-DIMM Turb Velocity [m/s]',
    'LGS_STREHL',
    'IA_FWHM',
    'MASS_TURB',
    'RON, [e-]',
    'ASM_RFLRMS',
    'Relative Flux RMS',
    'LGS_FWHM_GAIN',
    'MASS-DIMM Cn2 fraction at ground',
    'Par. angle',
    'LGS_TURVAR_RES',
    'MASS_FRACGL',
]

predicted_entries = list(set(df_norm.columns.tolist()) - set(input_features) - set(median_features))

# Sort entries in alphabetical order
input_features.sort()
median_features.sort()
predicted_entries.sort()


#%%
# Calculate the median for each feature
median_values = {}
for feature in median_features:
    if feature in muse_df_norm.columns:
        median_values[feature] = muse_df_norm[feature].median()
    else:
        print(f"Warning: Feature '{feature}' not found in muse_df_norm")
        median_values[feature] = np.nan

# Create a DataFrame with the median values
median_df = pd.DataFrame([median_values])

# Save median values dataframe with pickle
with open('../data/temp/median_values_norm.pkl', 'wb') as f:
    pickle.dump(median_df, f)

# Display the first few rows of the median DataFrame
# print("Median values for specified features:")
# print(median_df.head())

#%
# from machine_learning.MUSE_onsky_df import normalize_df
# median_df_unnorm = normalize_df(median_df, df_transforms, backward=True)
# median_df_unnorm.head()


#%%
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# Prepare the dataset for training
X_data = muse_df_norm[input_features].values
y_data = muse_df_norm[predicted_entries].values

# Use train_ids and valid_ids for splitting the data instead of train_test_split
X_train, y_train = X_data[train_ids], y_data[train_ids]
X_test,  y_test  = X_data[valid_ids], y_data[valid_ids]

# Create and train the MLP regressor
tc2sparta = MLPRegressor(
    hidden_layer_sizes=(100, 1000, 200),  # Two hidden layers with 200 and 100 neurons
    activation='tanh',
    solver='adam',
    alpha=0.01,  # L2 regularization parameter
    batch_size='auto',
    learning_rate='adaptive',
    max_iter=100,
    validation_fraction=0.1,
    # Since we're using our own validation set, we don't need early stopping with validation_fraction
    early_stopping=True,
    random_state=42,
    verbose=True
)

tc2sparta.fit(X_train, y_train)

# Evaluate the model on validation set
y_pred = tc2sparta.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Validation Mean Squared Error: {mse:.4f}")
print(f"Validation R² Score: {r2:.4f}")

#%%
with open('../data/models/tc2sparta.pkl', 'wb') as f:
    pickle.dump(tc2sparta, f)
    print("Model saved successfully.")

with open('../data/models/sparta2PSF.pkl', 'wb') as f:
    pickle.dump(PSF_calibrator, f)
    print("Model saved successfully.")


#%%
# Analyze feature importance using permutation importance
from sklearn.inspection import permutation_importance
import numpy as np

result = permutation_importance(
    tc2sparta, X_test, y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

# Get the mean importance for each feature
feature_importance = result.importances_mean
feature_importance_std = result.importances_std

# Create a DataFrame to display feature importance
importance_df = pd.DataFrame({
    'Feature': input_features,
    'Importance': feature_importance,
    'Std Dev': feature_importance_std
})

# Sort by importance
importance_df = importance_df.sort_values('Importance', ascending=False)
print("\nFeature Importance:")
print(importance_df)

#%%
# Visualize predictions vs actual values for all target variables with a grid layout
import math

# Calculate a reasonable grid size based on the number of targets
num_targets = len(predicted_entries)
cols = min(4, num_targets)  # Max 4 columns
rows = math.ceil(num_targets / cols)  # Calculate needed rows

# Create the figure with the calculated grid size
fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*4))

# Flatten the axes array for easier iteration
if num_targets > 1:
    axes = axes.flatten()
else:
    axes = [axes]  # Handle the case with only one subplot

predicted_features_r2 = {}

# Loop through all target variables
for i, target in enumerate(predicted_entries):
    if i < len(axes):  # Safety check
        target_idx = predicted_entries.index(target)

    # Plot the scatter points
    axes[i].scatter(y_test[:, target_idx], y_pred[:, target_idx], alpha=0.5)
        
    # Add a perfect prediction line
    min_val = min(y_test[:, target_idx].min(), y_pred[:, target_idx].min())
    max_val = max(y_test[:, target_idx].max(), y_pred[:, target_idx].max())
    axes[i].plot([min_val, max_val], [min_val, max_val], 'r--')

    # Set labels and title
    axes[i].set_xlabel('True Values')
    axes[i].set_ylabel('Predicted Values')
    axes[i].set_title(f'{target}')
    # Calculate and display R² for this specific target
    target_r2 = r2_score(y_test[:, target_idx], y_pred[:, target_idx])
    predicted_features_r2[target] = target_r2
    
    axes[i].text(0.05, 0.95, f'R² = {target_r2:.4f}', transform=axes[i].transAxes, verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))

# Hide any unused subplots
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.show()

#%%
print('R² anti-records:')
for target, r2 in predicted_features_r2.items():
    if r2 < 0.1:
        print(f'{target}: {r2}')

#%%
random_id = np.random.choice(valid_ids)
print(f'Random ID: {random_id}')

#%%
# Get the row as a Series first
X = muse_df[input_features].loc[[random_id]]  # Note the double brackets to keep it as a DataFrame

inputs = {
    'Wind direction': X['Wind dir (header)'].item(),
    'NGS magnitude': X['NGS mag (from ph.)'].item(),
    'Seeing': X['Seeing (header)'].item(),
    'Airmass': X['Airmass'].item(),
    'Tau0': X['Tau0 (header)'].item(),
    'Wavelengths': [500e-9, 600e-9],
    'Image size': 200,
}

#%%
inputs_map = {
    'Wind direction': 'Wind dir (header)',
    'NGS magnitude': 'NGS mag (from ph.)',
    'Seeing': 'Seeing (header)',
    'Airmass': 'Airmass',
    'Tau0': 'Tau0 (header)',
}

inputs_df = pd.DataFrame({x: inputs[x] for x in inputs_map.keys()}, index=[0])
inputs_df = inputs_df.rename(columns=inputs_map)
inputs_df = normalize_df(inputs_df, df_transforms)
inputs_df = inputs_df[input_features]
inputs_df.index = [0]
inputs_df = inputs_df.sort_index(axis=1)

#%%
sparta_df = pd.DataFrame([tc2sparta.predict(inputs_df.to_numpy().reshape(1,-1))[0]], columns=predicted_entries)
full_df = pd.concat([inputs_df, sparta_df, median_df], axis=1)
full_df = full_df[muse_df_norm.columns]
full_df = full_df.sort_index(axis=1)

full_df_denorm = normalize_df(full_df, df_transforms, backward=True)
full_df_denorm = full_df_denorm.sort_index(axis=1)

Y = PSF_calibrator.predict(full_df.to_numpy().reshape(1,-1))
Y_dict = calibrator.normalizer.unstack(Y)

atmo_wvl = 500e-9 # [m]
D_tel = 8.1 #[m]
TT_max = 2

TT_jitter = 0.5*(Y_dict['Jx'] + Y_dict['Jy']).mean() / rad2mas # [rad]
TT_WFE_nm = D_tel/2/TT_max * TT_jitter * 1e9 # [nm]

WFS_add_noise = Y_dict['dn'].item() # [rad^2]
WFS_add_noise_nm = np.sqrt(WFS_add_noise) * atmo_wvl*1e9 / (2 * np.pi)  # [nm]

print(f"Tip-tilt jitter: {TT_jitter*rad2mas:.2f} mas = {TT_WFE_nm:.2f} nm WFE RMS")
print(f"WFS noise: {WFS_add_noise_nm:.2f} nm RMS")

#%%
def parse_ini_file(file_path):
    """
    Parse a configuration file in INI format into a nested dictionary

    Parameters
    ----------
    file_path : str
        Path to the INI file

    Returns
    -------
    dict
        Nested dictionary with configuration sections and key-value pairs
    """
    config = {}
    current_section = None

    with open(file_path, 'r') as f:
        for line in f:
            # Remove comments and strip whitespace
            if ';' in line:
                line = line.split(';')[0]
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Process section headers
            if line.startswith('[') and line.endswith(']'):
                current_section = line[1:-1]
                config[current_section] = {}
                continue

            # Process key-value pairs
            if '=' in line and current_section is not None:
                key, value = [part.strip() for part in line.split('=', 1)]

                # Process list values
                if value.startswith('[') and value.endswith(']'):
                    value = value[1:-1]
                    value_list = [item.strip() for item in value.split(',')]

                    # Convert numeric values in lists
                    parsed_list = []
                    for item in value_list:
                        try:
                            if '.' in item:
                                parsed_list.append(float(item))
                            else:
                                parsed_list.append(int(item))
                        except ValueError:
                            parsed_list.append(item.strip('"\''))

                    config[current_section][key] = parsed_list
                else:
                    # Handle non-list values
                    try:
                        # Try to convert to numeric value
                        if value.lower() == 'true':
                            config[current_section][key] = True
                        elif value.lower() == 'false':
                            config[current_section][key] = False
                        elif value.lower() == 'none' or value.lower() == 'null':
                            config[current_section][key] = None
                        elif value.startswith("'") and value.endswith("'"):
                            config[current_section][key] = value[1:-1]
                        elif '.' in value or 'e' in value.lower():
                            config[current_section][key] = float(value)
                        else:
                            config[current_section][key] = int(value)
                    except ValueError:
                        # Keep as string if conversion fails
                        if value.startswith("'") and value.endswith("'"):
                            config[current_section][key] = value[1:-1]
                        elif value.startswith('"') and value.endswith('"'):
                            config[current_section][key] = value[1:-1]
                        else:
                            config[current_section][key] = value

    return config

def write_ini_file(config, file_path):
    """
    Write a configuration dictionary to an INI file

    Parameters
    ----------
    config : dict
        Nested dictionary with configuration sections and key-value pairs
    file_path : str
        Path to the output INI file
    """
    with open(file_path, 'w') as f:
        for section, keys in config.items():
            f.write(f'[{section}]\n')

            for key, value in keys.items():
                # Format lists
                if isinstance(value, list):
                    formatted_list = ', '.join(str(item) for item in value)
                    f.write(f'{key} = [{formatted_list}]\n')
                # Format strings with quotes if they contain spaces
                elif isinstance(value, str) and (' ' in value or ',' in value):
                    f.write(f"{key} = '{value}'\n")
                # Format booleans, None, and other values
                else:
                    f.write(f'{key} = {value}\n')

            f.write('\n')


#%%

config = parse_ini_file('../data/parameter_files/muse_base.ini')

config['telescope']['ZenithAngle'] = np.rad2deg(np.arccos(1.0/full_df_denorm['Airmass'].item()))

config['atmosphere']['Seeing'] = full_df_denorm['Seeing (header)'].item()
config['atmosphere']['L0'] = 27.44 # TODO: get it from somewhere

GL_frac = full_df_denorm['Cn2 fraction below 2000m'].item()

config['atmosphere']['Cn2Weights'] = [GL_frac, 1-GL_frac]
config['atmosphere']['Cn2Heights'] = [0, 2000.0]

num_layers = len(config['atmosphere']['Cn2Weights'])

assert num_layers == len(config['atmosphere']['Cn2Heights'])

config['atmosphere']['WindSpeed']     = [full_df_denorm['Wind speed (header)'].item(),] * num_layers
config['atmosphere']['WindDirection'] = [full_df_denorm['Wind dir (header)'].item(),]   * num_layers
    
config['sources_science']['Wavelength'] = inputs['Wavelengths']
config['sensor_science']['FieldOfView'] = inputs['Image size']

median_LGS_photons = sum([full_df_denorm[f'LGS{i} photons, [photons/m^2/s]'].item() for i in range(1,5)]) / 4 / 1000 / 1240
IRLOS_photons = full_df_denorm['IRLOS photons, [photons/s/m^2]'].item() / full_df_denorm['frequency'].item() / 4

config['sensor_HO']['NumberPhotons'] = [median_LGS_photons,]*4
config['sensor_HO']['extraErrorNm']  = WFS_add_noise_nm

config['sensor_LO']['PixelScale']  = full_df_denorm['plate scale, [mas/pix]'].item()
config['sensor_LO']['FieldOfView'] = int(full_df_denorm['window'].item())
config['sensor_LO']['Gain'] = full_df_denorm['gain'].item()

config['sensor_LO']['NumberPhotons'] = [IRLOS_photons]
config['sensor_LO']['extraErrorNm']  = TT_WFE_nm

#%%
write_ini_file(config, 'config.ini')