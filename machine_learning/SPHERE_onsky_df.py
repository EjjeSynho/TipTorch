#%%
%reload_ext autoreload
%autoreload 2

import sys
sys.path.append('..')

import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from project_globals import SPHERE_DATA_FOLDER
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tools.utils import corr_plot


#%% Initialize data sample
with open(SPHERE_DATA_FOLDER + 'sphere_df.pickle', 'rb') as handle:
    psf_df = pickle.load(handle)

# psf_df['Nph WFS (log)' ] = np.log(psf_df['Nph WFS'])
# psf_df['Flux WFS (log)'] = np.log(psf_df['Flux WFS'])
psf_df['Temperature (200 mbar)'] = psf_df['Temperature (200 mbar)'].apply(lambda x: np.nan if x > 100 else x)
psf_df = psf_df[psf_df['Corrupted'] == False]
psf_df = psf_df[psf_df['Multiples'] == False]
psf_df = psf_df[psf_df['Central hole'] == False]

ids_low         = psf_df['Low quality'] == True
ids_medium      = psf_df['Medium quality'] == True
ids_high        = psf_df['High quality'] == True
ids_corrupted   = psf_df['Corrupted'] == True
ids_medium_low  = ids_medium &  ids_low
ids_medium_high = ids_medium &  ids_high
ids_high        = ids_high   & ~ids_medium_high
ids_medium      = ids_medium & ~ids_medium_high
ids_medium      = ids_medium & ~ids_medium_low
ids_low         = ids_low    & ~ids_medium_low

ids_select = ids_high | ids_medium_high | ids_medium | ids_medium_low

quality_data = {
    'Low quality':         ids_low.sum()         / len(psf_df) * 100,
    'Low-Medium quality':  ids_medium_low.sum()  / len(psf_df) * 100,
    'Medium quality':      ids_medium.sum()      / len(psf_df) * 100,
    'Medium-High quality': ids_medium_high.sum() / len(psf_df) * 100,
    'High quality':        ids_high.sum()        / len(psf_df) * 100,
}

plt.bar(quality_data.keys(), quality_data.values())
plt.title('PSF samples quality distribution')
plt.xticks(rotation=45)
plt.ylabel('Percentage')
plt.show()

psf_df = psf_df[ids_select]
print(f'Remaining number of samples: {len(psf_df)}')

#%%
# Compute number of NaNs
presence_factor = {}

for entry in psf_df.columns.values.tolist():
    num_nans = psf_df[entry].isnull().sum()
    presence = 100 - num_nans / len(psf_df)*100
    presence_factor[entry] = presence
    
def percent_presence_gt(percentage, comparator, verbose=True):
    fields = []
    for entry in presence_factor:
        if comparator(presence_factor[entry], percentage):
            if verbose:
                print(f'-- {entry}: {presence_factor[entry]:.1f}%')
            fields.append(entry)
    return fields

good_fields = percent_presence_gt(75, np.greater_equal, True)

remove_entries = ['Filter WFS', 'Jitter X', 'Jitter Y', 'mag V', 'mag R', 'mag G', 'mag J', 'mag H', 'mag K', 'RA', 'DEC']
for entry in remove_entries:
    good_fields.pop(good_fields.index(entry))

# Copy only the good fields into a new dataframe
psf_df_filtered = psf_df[good_fields].dropna()

print('\n>>>> Samples remained:', len(psf_df_filtered))

# Remove all columns with NaNs
# psf_df_processed = psf_df_processed.dropna(axis=1)

#%%
from data_processing.normalizers import Gauss, Logify, YeoJohnson, Uniform, CreateTransformSequence

entry_data = [\
    ['Tau0 (header)',              0.0,   0.025,  [YeoJohnson, Gauss]],
    ['λ left (nm)',                1000,  2300,   [Uniform]],
    ['λ right (nm)',               1000,  2300,   [Uniform]],
    ['FWHM',                       0.5,   3.0,    [YeoJohnson, Gauss]],
    ['Jitter X',                   0.0,   60.0,   [Gauss]],
    ['Jitter Y',                   0.0,   60.0,   [Gauss]],
    ['RA',                         0,     360,    [Uniform]],
    ['DEC',                       -85,    35,     [Uniform]],
    ['Airmass',                    1.0,   1.5,    [YeoJohnson, Gauss]],
    ['Azimuth',                    0,     360,    [Uniform]],
    ['Altitude',                   20,    90,     [Uniform]],
    ['Strehl',                     0,     1,      [YeoJohnson, Gauss]],
    ['r0 (SPARTA)',                0.1,   0.5,    [YeoJohnson, Gauss]],
    ['Rate',                       0,     1380,   [Uniform]],
    ['Nph WFS',                    0,     200,    [Logify, YeoJohnson, Gauss]],
    ['Flux WFS',                   0,     2000,   [Logify, YeoJohnson, Gauss]],
    ['Wind speed (header)',        0,     20,     [YeoJohnson, Gauss]],
    ['Wind speed (200 mbar)',      0,     70,     [YeoJohnson, Gauss]],
    ['Wind direction (header)',    0,     360,    [Uniform]],
    ['Wind direction (200 mbar)',  0,     360,    [Uniform]],
    ['Focus',                     -2,     2,      [Gauss]],
    ['Pressure',                   740,   748,    [Gauss]],
    ['Humidity',                   3,     50,     [YeoJohnson, Gauss]],
    ['Temperature',                5,     20,     [YeoJohnson, Gauss]],
    ['Temperature (200 mbar)',    -70,   -40,     [YeoJohnson, Gauss]],
    ['mag V',                      2,     16,     [YeoJohnson, Gauss]],
    ['mag R',                      2,     14,     [YeoJohnson, Gauss]],
    ['mag G',                      2,     14,     [YeoJohnson, Gauss]],
    ['mag J',                      2,     11,     [YeoJohnson, Gauss]],
    ['mag H',                      2,     11,     [YeoJohnson, Gauss]],
    ['mag K',                      2,     11,     [YeoJohnson, Gauss]],
    ['Seeing (SPARTA)',            0.1,   1.5,    [YeoJohnson, Gauss]],
    ['Seeing (MASSDIMM)',          0.3,   1.75,   [YeoJohnson, Gauss]],
    ['Seeing (DIMM)',              0.3,   2.5,    [YeoJohnson, Gauss]],
    ['Turb. speed',                2,     24,     [YeoJohnson, Gauss]],
    ['Wind direction (MASSDIMM)',  0,     360,    [YeoJohnson, Gauss]],
    ['Wind speed (SPARTA)',        0,     30,     [YeoJohnson, Gauss]],
    ['Wind speed (MASSDIMM)',      0,     17.5,   [YeoJohnson, Gauss]],
    ['Tau0 (SPARTA)',              0,     0.075,  [YeoJohnson, Gauss]],
    ['Tau0 (MASSDIMM)',            0,     0.0175, [YeoJohnson, Gauss]],
    ['Tau0 (MASS)',                0,     0.025,  [YeoJohnson, Gauss]],
    ['ITTM pos (avg)',            -0.2,   0.1,    [YeoJohnson, Gauss]],
    ['ITTM pos (std)',             0,     0.2,    [YeoJohnson, Gauss]],
    ['DM pos (avg)',              -0.025, 0.025,  [Gauss]],
    ['DM pos (std)',               0.1,   0.3,    [YeoJohnson, Gauss]],
    ['Sequence time',              0,     300,    [Logify, Uniform]],
]

df_transforms = {}
for entry, _, _, transforms_list in entry_data:
    if entry in psf_df_filtered.columns.values:
        print(f' -- Processing \"{entry}\"')
        df_transforms[entry] = CreateTransformSequence(entry, psf_df_filtered, transforms_list)
        
#%%        
psf_df_normalized = psf_df_filtered.copy()
retain_entries = ['Filename', 'Date', 'Observation', 
                  'Filename', 'Bad center', 'Corrupted',
                  'Crossed', 'Extra', 'High SNR',
                  'High quality', 'LWE', 'Low quality',
                  'Medium quality', 'Multiples', 'Wings']

# Delete all columns except the ones in the entry_data
for entry in psf_df_normalized.columns.values:
    if entry not in df_transforms and entry not in retain_entries:
        psf_df_normalized.pop(entry)

for entry in df_transforms:
    psf_df_normalized[entry] = df_transforms[entry].forward(psf_df_normalized[entry].values)

#%%
# Storing for later usages in the project
with open('../data/temp/psf_df_norm.pickle', 'wb') as handle:
    pickle.dump(psf_df_normalized, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('../data/temp/psf_df_norm_transforms.pickle', 'wb') as handle:
    df_transforms_store = {}
    for entry in df_transforms:
        df_transforms_store[entry] = df_transforms[entry].store()
    pickle.dump(df_transforms_store, handle, protocol=pickle.HIGHEST_PROTOCOL)

#%%
# psf_df_processed
for entry, _, _, _ in entry_data:
    # pass
    if entry in psf_df_normalized.columns.values:
        sns.displot(data=psf_df_normalized, x=entry, kde=True, bins=100)
        plt.show()