#%%
%reload_ext autoreload
%autoreload 2

import sys
sys.path.insert(0, '..')

import pickle
import re
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from project_globals import MUSE_RAW_FOLDER, MUSE_DATA_FOLDER
import pandas as pd
from tools.utils import GetJmag
import seaborn as sns


def plot_data_filling(df):
    # Convert DataFrame to a boolean matrix: True for non-NaN values, False for NaN values
    data_filling = ~df.isna()
    
    # Plotting
    plt.figure(figsize=(20, 20))
    plt.imshow(data_filling, cmap='Greens', interpolation='none', aspect=6./35)
    plt.title('Data Filling Plot')
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    # plt.colorbar(label='Data Presence (1: non-NaN, 0: NaN)', ticks=[0, 1])
    plt.xticks(ticks=np.arange(len(df.columns)), labels=df.columns, fontsize=7, rotation=90)
    # plt.grid(axis='x', color='black', linestyle='-', linewidth=0.5)
    for x in np.arange(-0.5, len(df.columns.values), 1):
        plt.axvline(x=x, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.show()


#%%
# Load labels information
all_labels = []
labels_df  = { 'ID': [], 'Filename': [] }

if os.path.exists(MUSE_DATA_FOLDER+'labels.txt'):
    with open(MUSE_DATA_FOLDER+'labels.txt', 'r') as f:
        for line in f:
            filename, labels = line.strip().split(': ')

            ID = filename.split('_')[0]
            pure_filename = filename.replace(ID+'_', '').replace('.png', '')

            labels_df['ID'].append(int(ID))
            labels_df['Filename'].append(pure_filename)
            all_labels.append(labels.split(', '))
else:
    raise ValueError('Labels file does not exist!')

labels_list = list(set( [x for xs in all_labels for x in xs] ))
labels_list.sort()

for i in range(len(labels_list)):
    labels_df[labels_list[i]] = []

for i in range(len(all_labels)):
    for label in labels_list:
        labels_df[label].append(label in all_labels[i])

labels_df = pd.DataFrame(labels_df)
labels_df.set_index('ID', inplace=True)


# Read flattened DataFrames and concatenate them
muse_df = []
files = os.listdir(MUSE_RAW_FOLDER+'../DATA_reduced/')

for file in tqdm(files):
    with open(MUSE_RAW_FOLDER+'../DATA_reduced/'+file, 'rb') as f:
        data = pickle.load(f)
        df_ = data['All data']
        df_['ID'] = int(file.split('_')[0])
        muse_df.append(df_)
        
muse_df = pd.concat(muse_df)
muse_df.set_index('ID', inplace=True)
muse_df.sort_index(inplace=True)

muse_df = muse_df.join(labels_df)

# Save as pickle
with open(MUSE_RAW_FOLDER+'../muse_df.pickle', 'wb') as handle:
    pickle.dump(muse_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

#%%
# Open pickle file
with open(MUSE_RAW_FOLDER+'../muse_df.pickle', 'rb') as handle:
    muse_df = pickle.load(handle)

#%%
for entry in muse_df.columns.values:
    sns.displot(data=muse_df, x=entry, kde=True, bins=100)
    plt.show()

#%%
muse_df_pruned = muse_df.copy()

muse_df_pruned.loc[muse_df_pruned['SLODAR_FRACGL_300'] < 1e-12, 'SLODAR_FRACGL_300'] = np.nan
muse_df_pruned.loc[muse_df_pruned['SLODAR_FRACGL_500'] < 1e-12, 'SLODAR_FRACGL_500'] = np.nan
muse_df_pruned.loc[muse_df_pruned['SLODAR_TOTAL_CN2']  < 1e-12, 'SLODAR_TOTAL_CN2'] = np.nan
muse_df_pruned.loc[muse_df_pruned['IA_FWHM'] > 5, 'IA_FWHM'] = np.nan
muse_df_pruned.loc[muse_df_pruned['ASM_RFLRMS'] > 0.25, 'ASM_RFLRMS'] = np.nan
muse_df_pruned.loc[muse_df_pruned['Strehl (header)'] < 0.1, 'Strehl (header)'] = np.nan
muse_df_pruned.loc[muse_df_pruned['LGS3 flux, [ADU/frame]'] < 1e-12, 'LGS3 flux, [ADU/frame]'] = np.nan
muse_df_pruned.loc[muse_df_pruned['LGS3 photons, [photons/m^2/s]'] < 1e-12, 'LGS3 photons, [photons/m^2/s]'] = np.nan
muse_df_pruned.loc[muse_df_pruned['IRLOS photons, [photons/s/m^2]'] < 1e-12, 'IRLOS photons, [photons/s/m^2]'] = np.nan


for i in range(1, 9):
    entry = f'ALT{i}'
    muse_df_pruned.loc[muse_df_pruned[entry] >= 1000, entry] = np.nan
    muse_df_pruned.loc[muse_df_pruned[entry] < 0, entry] = np.nan

    entry = f'L0_ALT{i}'
    muse_df_pruned.loc[muse_df_pruned[entry] >= 96, entry] = np.nan
    muse_df_pruned.loc[muse_df_pruned[entry] < 0, entry] = np.nan

    entry = f'CN2_FRAC_ALT{i}'
    muse_df_pruned.loc[muse_df_pruned[entry] >= 1, entry] = np.nan
    muse_df_pruned.loc[muse_df_pruned[entry] < 1e-10, entry] = np.nan

    entry = f'CN2_ALT{i}'
    muse_df_pruned.loc[muse_df_pruned[entry] > 1e-11, entry] = np.nan
    muse_df_pruned.loc[muse_df_pruned[entry] < 1e-15, entry] = np.nan


for i in range(0, 7):
    entry = f'MASS_TURB{i}'
    muse_df_pruned.loc[muse_df_pruned[entry] < 1e-16, entry] = np.nan

for i in range(1, 5):   
    entry = f'LGS{i}_STREHL'
    muse_df_pruned.loc[muse_df_pruned[entry] < 0.1, entry] = np.nan
    
    entry = f'LGS{i}_L0'
    muse_df_pruned.loc[muse_df_pruned[entry] < 1000, entry] = np.nan
    
    muse_df_pruned.loc[muse_df_pruned[entry:=f'AOS.LGS{i}.SLOPERMSX'] < 1e-9, entry] = np.nan
    muse_df_pruned.loc[muse_df_pruned[entry:=f'AOS.LGS{i}.SLOPERMSY'] < 1e-9, entry] = np.nan
    muse_df_pruned.loc[muse_df_pruned[entry:=f'LGS{i}_TURVAR_TOT'] < 1e-9, entry] = np.nan
    muse_df_pruned.loc[muse_df_pruned[entry:=f'LGS{i}_TUR_GND'] < 1e-9, entry] = np.nan

plot_data_filling(muse_df_pruned)

#%%
for entry in muse_df.columns.values:
    sns.displot(data=muse_df_pruned, x=entry, kde=True, bins=100)
    plt.show()


# %% Dataset cleaning
muse_df_reduced = muse_df_pruned.copy()

muse_df_reduced['LGS_R0']         = muse_df_reduced[['LGS1_R0', 'LGS2_R0', 'LGS3_R0', 'LGS4_R0']].mean(axis=1, skipna=True)
muse_df_reduced['LGS_SEEING']     = muse_df_reduced[['LGS1_SEEING', 'LGS2_SEEING', 'LGS3_SEEING', 'LGS4_SEEING']].mean(axis=1, skipna=True)
muse_df_reduced['LGS_STREHL']     = muse_df_reduced[['LGS1_SEEING', 'LGS2_SEEING', 'LGS3_SEEING', 'LGS4_SEEING']].mean(axis=1, skipna=True)
muse_df_reduced['LGS_TURVAR_RES'] = muse_df_reduced[['LGS1_TURVAR_RES', 'LGS2_TURVAR_RES', 'LGS3_TURVAR_RES', 'LGS4_TURVAR_RES']].mean(axis=1, skipna=True)
muse_df_reduced['LGS_TURVAR_TOT'] = muse_df_reduced[['LGS1_TURVAR_TOT', 'LGS2_TURVAR_TOT', 'LGS3_TURVAR_TOT', 'LGS4_TURVAR_TOT']].mean(axis=1, skipna=True)
muse_df_reduced['LGS_TUR_ALT']    = muse_df_reduced[['LGS1_TUR_ALT', 'LGS2_TUR_ALT', 'LGS3_TUR_ALT', 'LGS4_TUR_ALT']].mean(axis=1, skipna=True)
muse_df_reduced['LGS_TUR_GND']    = muse_df_reduced[['LGS1_TUR_GND', 'LGS2_TUR_GND', 'LGS3_TUR_GND', 'LGS4_TUR_GND']].mean(axis=1, skipna=True)
muse_df_reduced['LGS_FWHM_GAIN']  = muse_df_reduced[['LGS1_FWHM_GAIN', 'LGS2_FWHM_GAIN', 'LGS3_FWHM_GAIN', 'LGS4_FWHM_GAIN']].mean(axis=1, skipna=True)

for i in range(1, 5):
    muse_df_reduced[f'AOS.LGS{i}.SLOPERMS'] = (muse_df_reduced[f'AOS.LGS{i}.SLOPERMSX']**2 + muse_df_reduced[f'AOS.LGS{i}.SLOPERMSY']**2)**0.5
    muse_df_reduced[f'LGS{i}_SLOPERMS']     = (muse_df_reduced[f'LGS{i}_SLOPERMSX']**2 + muse_df_reduced[f'LGS{i}_SLOPERMSY']**2)**0.5

muse_df_reduced['AOS.LGS.SLOPERMS'] = muse_df_reduced[[f'AOS.LGS{i}.SLOPERMS' for i in range(1, 5)]].mean(axis=1, skipna=True)
muse_df_reduced['LGS_SLOPERMS']     = muse_df_reduced[[f'LGS{i}_SLOPERMS' for i in range(1, 5)]].mean(axis=1, skipna=True)
muse_df_reduced['MASS_TURB']        = muse_df_reduced[[f'MASS_TURB{i}' for i in range(0, 7)]].sum(axis=1, skipna=True)

muse_df_reduced.loc[muse_df_reduced['MASS_TURB'] < 1e-16, 'MASS_TURB'] = np.nan


muse_df_reduced['Pupil angle'] = muse_df_reduced['Pupil angle'].map(lambda x: x % 360)
muse_df_reduced['Pupil angle'] = muse_df_reduced['Pupil angle'].map(lambda x: x if x < 180 else x - 360)


muse_df_reduced[[f'AOS.LGS{i}.SLOPERMS' for i in range(1, 5)]].mean(axis=1)

GL_fracs = []
GL_frac_SLODAR = []
h_GL = 2000

for j in range(len(muse_df_reduced)):
    Cn2_weights = np.array([muse_df_reduced.iloc[j][f'CN2_FRAC_ALT{i}'] for i in range(1, 9)])
    
    if not np.isnan(Cn2_weights).all():
        altitudes = np.array([muse_df_reduced.iloc[j][f'ALT{i}'] for i in range(1, 9)])*100 # in meters

        Cn2_weights_GL = Cn2_weights[altitudes < h_GL]
        altitudes_GL   = altitudes  [altitudes < h_GL]

        GL_frac  = np.nansum(Cn2_weights_GL)  # Ground layer fraction
        # Cn2_w_GL = np.interp(h_GL, altitudes, Cn2_weights)

        if GL_frac > 1.0 or GL_frac < 0.0:
            GL_frac = np.nan

        GL_fracs.append(GL_frac)
    else:
        GL_fracs.append(np.nan)
        
muse_df_reduced['Cn2 fraction below 2000m'] = GL_fracs


droppies = [
    *[f'LGS{i}_R0' for i in range(1, 5)],
    *[f'LGS{i}_L0' for i in range(1, 5)],
    *[f'LGS{i}_SEEING' for i in range(1, 5)],
    *[f'LGS{i}_STREHL' for i in range(1, 5)],
    *[f'LGS{i} works'  for i in range(1, 5)],
    *[f'LGS{i} flux, [ADU/frame]' for i in range(1, 5)],
    *[f'LGS{i} flux, [ADU/frame]' for i in range(1, 5)],
    *[f'LGS{i} flux, [ADU/frame]' for i in range(1, 5)],
    *[f'AOS.LGS{i}.SLOPERMSX' for i in range(1, 5)],
    *[f'AOS.LGS{i}.SLOPERMSY' for i in range(1, 5)],
    *[f'AOS.LGS{i}.SLOPERMS'  for i in range(1, 5)],
    *[f'LGS{i}_SLOPERMSX'  for i in range(1, 5)],
    *[f'LGS{i}_SLOPERMSY'  for i in range(1, 5)],
    *[f'LGS{i}_SLOPERMS'   for i in range(1, 5)],
    *[f'LGS{i}_TURVAR_RES' for i in range(1, 5)],
    *[f'LGS{i}_TURVAR_TOT' for i in range(1, 5)],
    *[f'LGS{i}_TUR_ALT'    for i in range(1, 5)],
    *[f'LGS{i}_TUR_GND'    for i in range(1, 5)],
    *[f'LGS{i}_FWHM_GAIN'  for i in range(1, 5)],
    
    *[f'CN2_FRAC_ALT{i}' for i in range(1, 9)],
    *[f'CN2_ALT{i}' for i in range(1, 9)],
    *[f'ALT{i}' for i in range(1, 9)],
    *[f'L0_ALT{i}' for i in range(1, 9)],
    *[f'SLODAR_CNSQ{i}' for i in range(2, 9)],
    *[f'MASS_TURB{i}' for i in range(0, 7)],
    
    'SLODAR_TOTAL_CN2',
    'SLODAR_FRACGL_300',
    'SLODAR_FRACGL_500',
    'ADUs (from 2x2)',
    'NGS mag',
    'NGS mag (from 2x2)',
    'Cn2 above UTs [10**(-15)m**(1/3)]',
    'Cn2 fraction below 300m',
    'Cn2 fraction below 500m',
    'Surface layer profile [10**(-15)m**(1/3)]',
    'Air Temperature at 30m [C]',
    'Seeing ["]',
    'scale',
    'AIRMASS',
    'seeingTot',
    'L0Tot',
    'r0Tot',
    'ADUs (header)',
    'NGS mag (header)',
    'Strehl (header)',
    'IRLOS photons (cube), [photons/s/m^2]',
    'Pixel scale (science)'
]

# Remove columns with specific names
muse_df_reduced.drop(columns=droppies, inplace=True)

print('Columns left:', len(muse_df_reduced.columns.values))

plot_data_filling(muse_df_reduced)


#%%
# for entry in muse_df_reduced.columns.values:
#     sns.displot(data=muse_df_reduced, x=entry, kde=True, bins=100)
#     plt.show()
#%%
from data_processing.normalizers import Invert, Gauss, Logify, YeoJohnson, Uniform, CreateTransformSequence, TransformSequence

entry_data = [
    ('name', []),
    ('time', []),
    ('Filename',     []),
    ('Bad quality',  []),
    ('Corrupted',    []),
    ('Good quality', []),
    ('Has streaks',  []),
    ('Low AO errs',  []),
    ('Non-point',    []),
    ('Science target', []),
    ('Medium quality', []),

    ('MASS_TURB',        [Logify, Gauss]),
    ('LGS_FWHM_GAIN',    [YeoJohnson, Gauss]),
    ('AOS.LGS.SLOPERMS', [YeoJohnson, Gauss]),
    ('LGS_SLOPERMS',     [YeoJohnson, Gauss]),
    ('LGS_R0',           [YeoJohnson, Gauss]),
    ('LGS_SEEING',       [YeoJohnson, Gauss]),
    ('LGS_STREHL',       [YeoJohnson, Gauss]),
    ('LGS_TURVAR_RES',   [YeoJohnson, Gauss]),
    ('LGS_TURVAR_TOT',   [YeoJohnson, Gauss]),
    ('LGS_TUR_ALT',      [YeoJohnson, Gauss]),
    ('DIMM Seeing ["]',  [YeoJohnson, Gauss]),
    ('MASS-DIMM Seeing ["]', [YeoJohnson, Gauss]),
    ('MASS-DIMM Tau0 [s]',   [YeoJohnson, Gauss]),
    ('MASS-DIMM Turb Velocity [m/s]', [YeoJohnson, Gauss]),
    ('Wind Speed at 30m [m/s]',       [YeoJohnson, Gauss]),
    ('Free Atmosphere Seeing ["]',    [YeoJohnson, Gauss]),
    ('MASS Tau0 [s]',       [YeoJohnson, Gauss]),
    ('ASM_WINDSPEED_10',    [YeoJohnson, Gauss]),
    ('ASM_WINDSPEED_30',    [YeoJohnson, Gauss]),
    ('Seeing (header)',     [YeoJohnson, Gauss]),
    ('Tau0 (header)',       [YeoJohnson, Gauss]),
    ('Wind dir (header)',   [Uniform]),
    ('Pupil angle',         [YeoJohnson, Gauss]),
    ('R0',                  [YeoJohnson, Gauss]),
    ('DIMM_SEEING',         [YeoJohnson, Gauss]),
    ('IA_FWHMLIN',          [YeoJohnson, Gauss]),
    ('IA_FWHMLINOBS',       [YeoJohnson, Gauss]),
    ('Wind speed (header)', [YeoJohnson, Gauss]),

    ('LGS_TUR_GND', [Invert, YeoJohnson, Gauss]),
    ('MASS-DIMM Cn2 fraction at ground', [Invert, YeoJohnson, Gauss]),
    ('Temperature (header)', [Invert, YeoJohnson, Gauss]),
    ('Relative Flux RMS',    [Logify, Gauss]),
    ('ASM_RFLRMS',  [Logify, Gauss]),
    ('Airmass',     [YeoJohnson, Uniform]),
    ('IRLOS photons, [photons/s/m^2]',        [Logify, Gauss]),
    # ('IRLOS photons (cube), [photons/s/m^2]', [Logify, Gauss]),

    ('Cn2 fraction below 2000m', [YeoJohnson, Uniform]),
    ('MASS_FRACGL',              [YeoJohnson, Gauss]),

    ('IA_FWHM', [Gauss]),

    ('ASM_WINDDIR_10', [Uniform]),
    ('ASM_WINDDIR_30', [Uniform]),
    ('Wind Direction at 30m (0/360) [deg]', [Uniform]),
    ('LGS1 photons, [photons/m^2/s]', [Uniform]),
    ('LGS2 photons, [photons/m^2/s]', [Uniform]),
    ('LGS3 photons, [photons/m^2/s]', [Uniform]),
    ('LGS4 photons, [photons/m^2/s]', [Uniform]),
    ('NGS mag (from ph.)', [Uniform]),
    ('RA (science)',  [Uniform]),
    ('DEC (science)', [Uniform]),
    ('Exp. time',     [Gauss]),
    ('Tel. altitude', [Uniform]),
    ('Tel. azimuth',  [Uniform]),
    ('Par. angle',    [Uniform]),
    ('Derot. angle',  [Uniform]),
    ('NGS RA',        [Uniform]),
    ('NGS DEC',       [Uniform]),
    # ('Pixel scale (science)', [Uniform]),
    ('window',    [Uniform]),
    ('frequency', [Uniform]),
    ('gain',      [Uniform]),
    ('plate scale, [mas/pix]', [Uniform]),
    ('conversion, [e-/ADU]',   [Uniform]),
    ('RON, [e-]', [Uniform])
]

# Find transforms parameters for each entry
df_transforms = {}
for entry, transforms_list in entry_data:
    if entry in muse_df_reduced.columns.values and len(transforms_list) > 0:
        print(f' -- Processing \"{entry}\"')
        df_transforms[entry] = CreateTransformSequence(entry, muse_df_reduced, transforms_list)

#%%
muse_df_normalized = {}

for entry in df_transforms:
    series = muse_df_reduced[entry].replace([np.inf, -np.inf], np.nan)
    transformed_values = df_transforms[entry].forward(series.dropna().values)
    full_length_values = np.full_like(series, np.nan, dtype=np.float64)
    full_length_values[~series.isna()] = transformed_values 
    muse_df_normalized[entry] = full_length_values   

muse_df_normalized = pd.DataFrame(muse_df_normalized)
muse_df_normalized['ID'] = muse_df_reduced.index
muse_df_normalized.set_index('ID', inplace=True)

# for entry in muse_df_normalized.columns.values:
#     sns.displot(data=muse_df_normalized, x=entry, kde=True, bins=100)
#     plt.show()



#%%
# Storing for later usages in the project
with open('../data/temp/muse_df_norm.pickle', 'wb') as handle:
    pickle.dump(muse_df_normalized, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('../data/temp/muse_df_norm_transforms.pickle', 'wb') as handle:
    df_transforms_store = {}
    for entry in df_transforms:
        df_transforms_store[entry] = df_transforms[entry].store()
    pickle.dump(df_transforms_store, handle, protocol=pickle.HIGHEST_PROTOCOL)
        

#%% Data imputation
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

# Clone dataframe
muse_df_imputed = muse_df_normalized.copy()

# Delete non-numeric string columns
# muse_df_imputed = muse_df_imputed.select_dtypes(exclude=['object'])
# muse_df_imputed = muse_df_imputed.select_dtypes(exclude=['bool'])

droppies_for_imputter = [
    # 'RA (science)',
    # 'DEC (science)',
    # 'Tel. altitude',
    # 'Tel. azimuth',
    # 'NGS RA',
    # 'NGS DEC',
    # 'NGS mag (from ph.)' # delete to restore later
]

# Change scaling
# for i in range(1, 5):
#     df_imputed[f'LGS{i} photons, [photons/m^2/s]'] /= 1e9
    
# df_imputed['MASS_TURB'] *= 1e13
# df_imputed['IRLOS photons, [photons/s/m^2]'] /= 1e4
# df_imputed['IRLOS photons (cube), [photons/s/m^2]'] /= 1e4

if len(droppies_for_imputter) > 0:
    muse_df_imputed.drop(columns=droppies_for_imputter, inplace=True)

muse_df_imputed.replace([np.inf, -np.inf], np.nan, inplace=True)

# scaler_imputer = StandardScaler()
# df_imputed_data = scaler_imputer.fit_transform(df_imputed)
# df_imputed = pd.DataFrame(df_imputed_data, columns=df_imputed.columns)

# plot_data_filling(df_imputed)
#%%

# imputer = IterativeImputer(estimator=BayesianRidge(), random_state=0, max_iter=200, verbose=2)
# imputer = IterativeImputer(estimator=RandomForestRegressor(n_estimators=100, n_jobs=16), max_iter=15, random_state=0, verbose=2)

imputer = IterativeImputer(
    estimator = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=16),
    random_state = 42,
    max_iter = 7,
    verbose = 2)

imputed_data = imputer.fit_transform(muse_df_imputed)
muse_df_imputed = pd.DataFrame(imputed_data, columns=muse_df_imputed.columns)

#%%
muse_df_normalized_new = muse_df_normalized.copy()

# Copy missing values back to the original DataFrame
for col in muse_df_imputed.columns:
    muse_df_normalized_new[col] = muse_df_imputed[col]

# muse_df_normalized_new['NGS mag (from ph.)'] = GetJmag(muse_df_normalized_new['IRLOS photons, [photons/s/m^2]'])
plot_data_filling(muse_df_normalized_new)

#%%
with open(MUSE_RAW_FOLDER+'../muse_df_norm_imputed.pickle', 'wb') as handle:
    pickle.dump(muse_df_normalized_new, handle, protocol=pickle.HIGHEST_PROTOCOL)

#%%%   
muse_df_reduced = muse_df_reduced[muse_df_reduced['Corrupted'] == False]
muse_df_reduced = muse_df_reduced[muse_df_reduced['Bad quality'] == False]

plot_data_filling(muse_df_reduced)

#% Compute percentage of missing values in each column
data_filling = np.array(~muse_df_reduced.isna()).prod(axis=1)
print(f'Fraction of rows with all data present: {data_filling.sum()/len(data_filling):.2f}')

missing_values = ( muse_df_reduced.isna().sum() / len(muse_df_reduced) ).sort_values(ascending=False)

plt.figure(figsize=(20, 7))
plt.bar(missing_values.index, missing_values.values)
plt.xticks(rotation=90)
plt.show()
