#%%
%reload_ext autoreload
%autoreload 2

import sys
from typing import Any
sys.path.append('..')

import torch
import pickle
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from project_globals import SPHERE_DATA_FOLDER, DATA_FOLDER, device
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
from tools.utils import seeing

from project_globals import WEIGHTS_FOLDER

#%% Initialize data sample
with open(SPHERE_DATA_FOLDER+'sphere_df.pickle', 'rb') as handle:
    psf_df = pickle.load(handle)

with open('E:/ESO/Data/SPHERE/fitted_df.pickle', 'rb') as handle:
    fitted_df = pickle.load(handle)

with open('E:/ESO/Data/SPHERE/synth_df.pickle', 'rb') as handle:
    synth_df = pickle.load(handle)

with open('E:/ESO/Data/SPHERE/synth_fitted_df.pickle', 'rb') as handle:
    synth_fitted_df = pickle.load(handle)

psf_df = psf_df[psf_df['invalid'] == False]
psf_df = psf_df[psf_df['LWE'] == False]
psf_df = psf_df[psf_df['doubles'] == False]
psf_df = psf_df[psf_df['No coronograph'] == False]
psf_df = psf_df[~pd.isnull(psf_df['r0 (SPARTA)'])]
psf_df = psf_df[~pd.isnull(psf_df['Nph WFS'])]
psf_df = psf_df[~pd.isnull(psf_df['Strehl'])]
psf_df = psf_df[~pd.isnull(psf_df['FWHM'])]
# psf_df = psf_df[~pd.isnull(psf_df['Wind direction (MASSDIMM)'])]
# psf_df = psf_df[~pd.isnull(psf_df['Wind speed (MASSDIMM)'])]
# psf_df = psf_df[~pd.isnull(psf_df['Tau0 (MASSDIMM)'])]
# psf_df = psf_df[~pd.isnull(psf_df['Seeing (MASSDIMM)'])]

# psf_df = psf_df[psf_df['Nph WFS'] < 5000]
# psf_df = psf_df[psf_df['λ left (nm)'] > 1600]
# psf_df = psf_df[psf_df['λ left (nm)'] < 1700]
# psf_df = psf_df[~pd.isnull(psf_df['Turb. speed'])]
# psf_df = psf_df[~pd.isnull(psf_df['Pressure'])]
# psf_df = psf_df[~pd.isnull(psf_df['Humidity'])]
# psf_df = psf_df[~pd.isnull(psf_df['Temperature'])]

psf_df['Seeing (SPARTA)'] = seeing(psf_df['r0 (SPARTA)'],500e-9)
synth_df = synth_df[synth_df['invalid'] == False]

#%%
good_fits_folder = 'E:/ESO/Data/SPHERE/good_fits_TipTorch/'

files = os.listdir(good_fits_folder)
good_ids = [int(file.split('.')[0]) for file in files]

# psf_df = psf_df.loc[good_ids]
fitted_ids = list( set( fitted_df.index.values.tolist() ).intersection( set(psf_df.index.values.tolist()) ) )
fitted_df = fitted_df[fitted_df.index.isin(fitted_ids)]

# Absolutize positive-only values
for entry in ['r0','Jx','Jy','Jxy']: #,'F (left)','F (right)']:
    fitted_df[entry] = fitted_df[entry].abs()
fitted_df.sort_index(inplace=True)

fitted_df['SR fit']  = 0.5 * (fitted_df['SR fit (left)'] + fitted_df['SR fit (right)'])
fitted_df['SR data'] = 0.5 * (fitted_df['SR data (left)'] + fitted_df['SR data (right)'])
fitted_df['J']       = np.sqrt(fitted_df['Jx'].pow(2) + fitted_df['Jy'].pow(2))
fitted_df['F']       = 0.5 * (fitted_df['F (left)'] + fitted_df['F (right)'])

synth_fitted_df['SR fit']  = 0.5 * (fitted_df['SR fit (left)'] + fitted_df['SR fit (right)'])
synth_fitted_df['SR data'] = 0.5 * (fitted_df['SR data (left)'] + fitted_df['SR data (right)'])
synth_fitted_df['J']       = np.sqrt(fitted_df['Jx'].pow(2) + fitted_df['Jy'].pow(2))
synth_fitted_df['F']       = 0.5 * (fitted_df['F (left)'] + fitted_df['F (right)'])

#%% Compute data transformations
from data_processing.normalizers import Uniform, TransformSequence

transforms_input = {}
transforms_input['Airmass']                   = TransformSequence( transforms = [ Uniform(a=1.0, b=2.2) ])
transforms_input['r0 (SPARTA)']               = TransformSequence( transforms = [ Uniform(a=0.05, b=0.45) ])
transforms_input['Seeing (MASSDIMM)']         = TransformSequence( transforms = [ Uniform(a=0.3, b=1.5) ])
transforms_input['Wind direction (header)']   = TransformSequence( transforms = [ Uniform(a=0, b=360) ])
transforms_input['Wind speed (header)']       = TransformSequence( transforms = [ Uniform(a=0.0, b=17.5) ])
transforms_input['Wind direction (MASSDIMM)'] = TransformSequence( transforms = [ Uniform(a=0, b=360) ])
transforms_input['Wind speed (MASSDIMM)']     = TransformSequence( transforms = [ Uniform(a=0.0, b=17.5) ])
transforms_input['Tau0 (header)']             = TransformSequence( transforms = [ Uniform(a=0.0, b=0.025) ])
transforms_input['Tau0 (MASSDIMM)']           = TransformSequence( transforms = [ Uniform(a=0.0, b=0.02) ])
transforms_input['λ left (nm)']               = TransformSequence( transforms = [ Uniform(a=psf_df['λ left (nm)'].min(), b=psf_df['λ left (nm)'].max()) ])
transforms_input['λ right (nm)']              = TransformSequence( transforms = [ Uniform(a=psf_df['λ right (nm)'].min(), b=psf_df['λ right (nm)'].max()) ])
transforms_input['Rate']                      = TransformSequence( transforms = [ Uniform(a=psf_df['Rate'].min(), b=psf_df['Rate'].max()) ])
transforms_input['FWHM']                      = TransformSequence( transforms = [ Uniform(a=0.5, b=3.0) ])
transforms_input['Nph WFS']                   = TransformSequence( transforms = [ Uniform(a=0, b=2e6) ])
transforms_input['Strehl']                    = TransformSequence( transforms = [ Uniform(a=0.0, b=1.0) ])
transforms_input['Jitter X']                  = TransformSequence( transforms = [ Uniform(a=0.0, b=60.0) ])
transforms_input['Jitter Y']                  = TransformSequence( transforms = [ Uniform(a=0.0, b=60.0) ])
transforms_input['Turb. speed']               = TransformSequence( transforms = [ Uniform(a=0.0, b=25.0) ])
transforms_input['Humidity']                  = TransformSequence( transforms = [ Uniform(a=3.0, b=51.0) ])
transforms_input['Pressure']                  = TransformSequence( transforms = [ Uniform(a=740, b=750) ])
transforms_input['Temperature']               = TransformSequence( transforms = [ Uniform(a=0, b=25) ])
transforms_input['ITTM pos (std)']            = TransformSequence( transforms = [ Uniform(a=0, b=0.25) ])
transforms_input['Focus']                     = TransformSequence( transforms = [ Uniform(a=-1, b=1.5) ])
transforms_input['DM pos (std)']              = TransformSequence( transforms = [ Uniform(a=0.125, b=0.275) ])
transforms_input['ITTM pos (avg)']            = TransformSequence( transforms = [ Uniform(a=-0.1, b=0.1) ])
transforms_input['DM pos (avg)']              = TransformSequence( transforms = [ Uniform(a=-0.025, b=0.025) ])
transforms_input['Seeing (SPARTA)']           = TransformSequence( transforms = [ Uniform(a=0.2, b=1.1) ])

input_df = pd.DataFrame( {a: transforms_input[a].forward(psf_df[a].values) for a in transforms_input.keys()} )
input_df.index = psf_df.index


transforms_output = {}
transforms_output['r0']              = TransformSequence( transforms = [ Uniform(a=0.05, b=0.45) ])
transforms_output['bg (left)']       = TransformSequence( transforms = [ Uniform(a=-0.2e-5, b=0.4e-5) ])
transforms_output['bg (right)']      = TransformSequence( transforms = [ Uniform(a=-0.2e-5, b=0.4e-5) ])
transforms_output['dx']              = TransformSequence( transforms = [ Uniform(a=-1, b=1) ])
transforms_output['dy']              = TransformSequence( transforms = [ Uniform(a=-1, b=1) ])
transforms_output['F (left)']        = TransformSequence( transforms = [ Uniform(a=0.3, b=1.5) ])
transforms_output['F (right)']       = TransformSequence( transforms = [ Uniform(a=0.3, b=1.5) ])
transforms_output['F']               = TransformSequence( transforms = [ Uniform(a=0.3, b=1.5) ])
transforms_output['Jx']              = TransformSequence( transforms = [ Uniform(a=0, b=60) ])
transforms_output['Jy']              = TransformSequence( transforms = [ Uniform(a=0, b=60) ])
transforms_output['J']               = TransformSequence( transforms = [ Uniform(a=0, b=60) ])
transforms_output['Jxy']             = TransformSequence( transforms = [ Uniform(a=0, b=200) ])
transforms_output['Nph WFS']         = TransformSequence( transforms = [ Uniform(a=0, b=2e6) ])
transforms_output['n']               = TransformSequence( transforms = [ Uniform(a=0, b=0.005) ])
transforms_output['dn']              = TransformSequence( transforms = [ Uniform(a=-0.2, b=0.2) ])
transforms_output['SR data (left)']  = TransformSequence( transforms = [ Uniform(a=0.0, b=1.0) ])
transforms_output['SR data (right)'] = TransformSequence( transforms = [ Uniform(a=0.0, b=1.0) ])
transforms_output['SR fit (left)']   = TransformSequence( transforms = [ Uniform(a=0.0, b=1.0) ])
transforms_output['SR fit (right)']  = TransformSequence( transforms = [ Uniform(a=0.0, b=1.0) ])
transforms_output['SR data']         = TransformSequence( transforms = [ Uniform(a=0.0, b=1.0) ])
transforms_output['SR fit']          = TransformSequence( transforms = [ Uniform(a=0.0, b=1.0) ])

output_df = pd.DataFrame( {a: transforms_output[a].forward(fitted_df[a].values) for a in transforms_output.keys()} )
output_df.index = fitted_df.index


rows_with_nan = output_df.index[output_df.isna().any(axis=1)].values.tolist()
rows_with_nan += fitted_df.index[fitted_df['F (right)'] > 2].values.tolist()
rows_with_nan += fitted_df.index[fitted_df['Jxy'] > 300].values.tolist()
rows_with_nan += fitted_df.index[fitted_df['Jx'] > 70].values.tolist()
rows_with_nan += fitted_df.index[fitted_df['Jy'] > 70].values.tolist()

psf_df.drop(rows_with_nan, inplace=True)
input_df.drop(rows_with_nan, inplace=True)
fitted_df.drop(rows_with_nan, inplace=True)
output_df.drop(rows_with_nan, inplace=True)

good_ids = psf_df.index.values.tolist()
print(len(good_ids), 'samples are in the dataset')

def corr_plot(data, entry_x, entry_y, lims=None):
    j = sns.jointplot(data=data, x=entry_x, y=entry_y, kind="kde", space=0, alpha = 0.8, fill=True, colormap='royalblue' )
    i = sns.scatterplot(data=data, x=entry_x, y=entry_y, alpha=0.5, ax=j.ax_joint, color = 'black', s=10)
    sns.set_style("darkgrid")
    
    j.ax_joint.set_aspect('equal')
    lims = [np.min([j.ax_joint.get_xlim(), j.ax_joint.get_ylim()]),
            np.max([j.ax_joint.get_xlim(), j.ax_joint.get_ylim()])]

    j.ax_joint.set_xlim(lims)
    j.ax_joint.set_ylim(lims)
    j.ax_joint.plot([lims[0], lims[1]], [lims[0], lims[1]], 'gray', linewidth=1.5, linestyle='--')
    
    # plt.grid()
    plt.show()

#%%
from tools.utils import seeing

# sns.kdeplot(data=fitted_df, x='F', y='J', fill=True, thresh=0.02, palette='viridis')
# sns.kdeplot(data=psf_df, x='r0 (SPARTA)', y='DM pos (std)', fill=True, thresh=0.02, palette='viridis')
sns.kdeplot(data=psf_df, x='Strehl', y='DM pos (avg)', fill=True, thresh=0.02, palette='viridis')


#%% ================ Telemetry SR from telemetry ===================
#% Select the entries to be used in training
selected_entries_X = [
        'Airmass',
        # 'Seeing (MASSDIMM)',
        # 'FWHM',
        # 'Wind direction (MASSDIMM)',
        # 'Wind speed (MASSDIMM)',
        # 'Tau0 (MASSDIMM)',
        'Seeing (SPARTA)',
        'Wind direction (header)',
        'Wind speed (header)',
        'Tau0 (header)',
        'r0 (SPARTA)',
        'Nph WFS',
        'Rate',
        'λ left (nm)',
        'λ right (nm)',
        # 'Jitter X',
        # 'Jitter Y',
        # 'Turb. speed',
        # 'Pressure',
        # 'Humidity'
        # 'Temperature'
        # 'ITTM pos (std)',
        # 'Focus',
        # 'DM pos (std)'
    ]

selected_entries_Y = ['Strehl']

X_df = input_df[selected_entries_X]
Y_df = input_df[selected_entries_Y]

X = X_df.to_numpy()
Y = Y_df.to_numpy()[:,0]

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
#%
gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=7, random_state=42)
reg = LinearRegression()
reg.fit(X_train, y_train)
gbr.fit(X_train, y_train)
y_pred = gbr.predict(X_test)
y_pred_reg = reg.predict(X_test)

# y_pred = y_pred_reg

err = np.abs(y_test-y_pred) * 100
print("The mean absolute error (MAE) on test set: {:.4f}".format(err.mean()), "%")
print("The median absolute error (MAE) on test set: {:.4f}".format(np.median(err)), "%")
print("The std absolute error (MAE) on test set: {:.4f}".format(err.std()), "%")
print("The max absolute error (MAE) on test set: {:.4f}".format(err.max()), "%")
print("The min absolute error (MAE) on test set: {:.4f}".format(err.min()), "%")

test_df = pd.DataFrame({
    'SR predicted': transforms_input['Strehl'].backward(y_pred),
    'SR from data': transforms_input['Strehl'].backward(y_test),
})

corr_plot(test_df, 'SR predicted', 'SR from data')

#%% ================ Fitted SR from telemetry ===================
#% Select the entries to be used in training
selected_entries_X = [
        'Airmass',
        'r0 (SPARTA)',
        # 'Seeing (MASSDIMM)',
        # 'FWHM',
        # 'Wind direction (MASSDIMM)',
        # 'Wind speed (MASSDIMM)',
        'Wind direction (header)',
        'Wind speed (header)',
        'Tau0 (header)',
        # 'Tau0 (MASSDIMM)',
        'Nph WFS',
        'Rate',
    #   'Jitter X',
    #   'Jitter Y',
        'λ left (nm)',
        'λ right (nm)',
    #   'Turb. speed',
    #   'Pressure',
    #   'Humidity',
    #   'Temperature'
    ]

selected_entries_Y = ['SR fit']

X_df = input_df[selected_entries_X]
Y_df = output_df[selected_entries_Y]

X = X_df.to_numpy()
Y = Y_df.to_numpy()[:,0]

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
#%
gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=7, random_state=42)
reg = LinearRegression()
reg.fit(X_train, y_train)
gbr.fit(X_train, y_train)
y_pred = gbr.predict(X_test)
y_pred_reg = reg.predict(X_test)

# y_pred = y_pred_reg

err = np.abs(y_test-y_pred) * 100
print("The mean absolute error (MAE) on test set: {:.4f}".format(err.mean()), "%")
print("The median absolute error (MAE) on test set: {:.4f}".format(np.median(err)), "%")
print("The std absolute error (MAE) on test set: {:.4f}".format(err.std()), "%")
print("The max absolute error (MAE) on test set: {:.4f}".format(err.max()), "%")
print("The min absolute error (MAE) on test set: {:.4f}".format(err.min()), "%")

test_df = pd.DataFrame({
    'SR predicted': transforms_input['Strehl'].backward(y_pred),
    'SR from data': transforms_input['Strehl'].backward(y_test),
})

corr_plot(test_df, 'SR predicted', 'SR from data')
#%% ================ Fitted parameters (J,F) from telemetry ===================
#% Select the entries to be used in training
selected_entries_X = [
        'Airmass',
        'r0 (SPARTA)',
        # 'Seeing (MASSDIMM)',
        # 'FWHM',
        # 'Wind direction (MASSDIMM)',
        # 'Wind speed (MASSDIMM)',
        'Wind direction (header)',
        'Wind speed (header)',
        'Tau0 (header)',
        # 'Tau0 (MASSDIMM)',
        'Nph WFS',
        'Rate',
    #   'Jitter X',
    #   'Jitter Y',
        'λ left (nm)',
        'λ right (nm)',
    #   'Turb. speed',
    #   'Pressure',
    #   'Humidity',
    #   'Temperature'
    ]

selected_entries_Y = ['J']

X_df = input_df[selected_entries_X]
Y_df = output_df[selected_entries_Y]

X = X_df.to_numpy()
Y = Y_df.to_numpy()[:,0]

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
#%
gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=7, random_state=42)
reg = LinearRegression()
reg.fit(X_train, y_train)
gbr.fit(X_train, y_train)
y_pred = gbr.predict(X_test)
y_pred_reg = reg.predict(X_test)

# y_pred = y_pred_reg

err = np.abs(y_test-y_pred) * 100
print("The mean absolute error (MAE) on test set: {:.4f}".format(err.mean()), "%")
print("The median absolute error (MAE) on test set: {:.4f}".format(np.median(err)), "%")
print("The std absolute error (MAE) on test set: {:.4f}".format(err.std()), "%")
print("The max absolute error (MAE) on test set: {:.4f}".format(err.max()), "%")
print("The min absolute error (MAE) on test set: {:.4f}".format(err.min()), "%")

test_df = pd.DataFrame({
    'J predicted': transforms_output['J'].backward(y_pred),
    'J test': transforms_output['J'].backward(y_test),
})

corr_plot(test_df, 'J predicted', 'J test')

#%% ================ Land of synths ===================

test = {
    'x': synth_fitted_df['n'] + synth_fitted_df['dn'],
    'y': synth_df['WFS noise (nm)'],
    'z': synth_df['Rate'],
   
}

sns.scatterplot(test, x='x', y='y', hue='z', cmap='viridis')


#%%
with open(SPHERE_DATA_FOLDER+'sphere_df.pickle', 'rb') as handle:
    synth_psf_df = pickle.load(handle)
    selected_ids = list( set( synth_df.index.values.tolist() ).intersection( set(synth_df.index.values.tolist()) ) )
    synth_psf_df = synth_psf_df.loc[selected_ids]

synth_input_df = pd.DataFrame( {a: transforms_input[a].forward(synth_df[a].values) for a in transforms_input.keys()} )
synth_input_df.index = synth_df.index

synth_output_df = pd.DataFrame( {a: transforms_output[a].forward(synth_fitted_df[a].values) for a in transforms_output.keys()} )
synth_output_df.index = synth_fitted_df.index

#%%
rows_with_nan = synth_output_df.index[synth_output_df.isna().any(axis=1)].values.tolist()
rows_with_nan += synth_fitted_df.index[synth_fitted_df['F (right)'] > 2].values.tolist()
rows_with_nan += synth_fitted_df.index[synth_fitted_df['Jxy'] > 300].values.tolist()
rows_with_nan += synth_fitted_df.index[synth_fitted_df['Jx'] > 70].values.tolist()
rows_with_nan += synth_fitted_df.index[synth_fitted_df['Jy'] > 70].values.tolist()

synth_psf_df.drop(rows_with_nan, inplace=True)
synth_input_df.drop(rows_with_nan, inplace=True)
synth_fitted_df.drop(rows_with_nan, inplace=True)
synth_output_df.drop(rows_with_nan, inplace=True)

#%%
#% Select the entries to be used in training
selected_entries_X = [
        'Airmass',
        'r0 (SPARTA)',
        # 'Seeing (MASSDIMM)',
        # 'FWHM',
        # 'Wind direction (MASSDIMM)',
        # 'Wind speed (MASSDIMM)',
        'Wind direction (header)',
        'Wind speed (header)',
        'Tau0 (header)',
        # 'Tau0 (MASSDIMM)',
        'Nph WFS',
        'Rate',
    #   'Jitter X',
    #   'Jitter Y',
        'λ left (nm)',
        'λ right (nm)',
    #   'Turb. speed',
    #   'Pressure',
    #   'Humidity',
    #   'Temperature'
    ]

selected_entries_Y = ['SR fit']

X_df = synth_input_df[selected_entries_X]
Y_df = synth_output_df[selected_entries_Y]

X = X_df.to_numpy()
Y = Y_df.to_numpy()[:,0]

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
#%
gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=7, random_state=42)
reg = LinearRegression()
reg.fit(X_train, y_train)
gbr.fit(X_train, y_train)
y_pred = gbr.predict(X_test)
y_pred_reg = reg.predict(X_test)

# y_pred = y_pred_reg

err = np.abs(y_test-y_pred) * 100
print("The mean absolute error (MAE) on test set: {:.4f}".format(err.mean()), "%")
print("The median absolute error (MAE) on test set: {:.4f}".format(np.median(err)), "%")
print("The std absolute error (MAE) on test set: {:.4f}".format(err.std()), "%")
print("The max absolute error (MAE) on test set: {:.4f}".format(err.max()), "%")
print("The min absolute error (MAE) on test set: {:.4f}".format(err.min()), "%")

test_df = pd.DataFrame({
    'SR predicted': transforms_output['SR fit'].backward(y_pred),
    'SR test': transforms_output['SR fit'].backward(y_test),
})

corr_plot(test_df, 'SR predicted', 'SR test')



#%% ========================= AAAAAAAAAAA ==============================
maximum_oulier = y_test[np.where(err == err.max())[0].item()]
outlier_sample = X_df.iloc[Y.tolist().index(maximum_oulier)]
outlier_input = psf_df.loc[outlier_sample.name]

#%% 
#% Select the entries to be used in training

merged_df = pd.concat([input_df, output_df], axis=1)

selected_entries_X = [
    'SR fit',
    'r0',
    'Rate',
    'Tau0 (header)',
    'Wind speed (header)'
    ]
    # 'Wind direction (header)',

# selected_entries_Y = ['SR fit']
selected_entries_Y = ['J']

X_df = merged_df[selected_entries_X]
Y_df = merged_df[selected_entries_Y]

X = X_df.to_numpy()
Y = Y_df.to_numpy()[:,0]

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=7, random_state=42)
reg = LinearRegression()

gbr.fit(X_train, y_train)
reg.fit(X_train, y_train)

y_pred = gbr.predict(X_test)
y_pred_reg = reg.predict(X_test)

# y_pred = y_pred_reg

err = np.abs(
    transforms_output[selected_entries_Y[0]].backward(y_test) - 
    transforms_output[selected_entries_Y[0]].backward(y_pred))

print("The mean absolute error (MAE) on test set: {:.4f}".format(err.mean()))
print("The median absolute error (MAE) on test set: {:.4f}".format(np.median(err)))
print("The std absolute error (MAE) on test set: {:.4f}".format(err.std()))
print("The max absolute error (MAE) on test set: {:.4f}".format(err.max()))
print("The min absolute error (MAE) on test set: {:.4f}".format(err.min()))

test_df = pd.DataFrame({
    'Test': transforms_output['Jy'].backward(y_test),
    'Pred': transforms_output['Jy'].backward(y_pred),
})

corr_plot(test_df, 'Test', 'Pred')

#%%
test_df = {
    'F pred': transforms_output['F (left)'].backward(y_pred),
    'F test': transforms_output['F (left)'].backward(y_test),
    # 'SR fit': fitted_df['SR fit'],
}

test_df = pd.DataFrame(test_df)

sns.jointplot(data=test_df, x='F pred', y='F test', kind="kde", space=0, fill=True)
# sns.kdeplot(data=test_df, x='F pred', y='F test', fill=True, palette='viridis')
# plt.axes().set_aspect('equal')

#%%
import numpy as np
from scipy.stats import (multivariate_normal as mvn, norm)
from scipy.stats._multivariate import _squeeze_output
from scipy.optimize import minimize


class multivariate_skewnorm:
    def __init__(self, shape, cov=None):
        self.dim   = len(shape)
        self.shape = np.asarray(shape)
        self.mean  = np.zeros(self.dim)
        self.cov   = np.eye(self.dim) if cov is None else np.asarray(cov)

    def pdf(self, x):
        return np.exp(self.logpdf(x))
        
    def logpdf(self, x):
        x    = mvn._process_quantiles(x, self.dim)
        pdf  = mvn(self.mean, self.cov).logpdf(x)
        cdf  = norm(0, 1).logcdf(np.dot(x, self.shape))
        return _squeeze_output(np.log(2) + pdf + cdf)

    def rvs_slow(self, size=1):
        std_mvn = mvn(np.zeros(self.dim), np.eye(self.dim))
        x       = np.empty((size, self.dim))
        
        # Apply rejection sampling.
        n_samples = 0
        while n_samples < size:
            z = std_mvn.rvs(size=1)
            u = np.random.uniform(0, 2*std_mvn.pdf(z))
            if not u > self.pdf(z):
                x[n_samples] = z
                n_samples += 1
        
        # Rescale based on correlation matrix.
        chol = np.linalg.cholesky(self.cov)
        x = (chol @ x.T).T
        return x
    
    def rvs_fast(self, size=1):
        aCa      = self.shape @ self.cov @ self.shape
        delta    = (1 / np.sqrt(1 + aCa)) * self.cov @ self.shape
        cov_star = np.block([[np.ones(1),     delta],
                             [delta[:, None], self.cov]])
        
        x        = mvn(np.zeros(self.dim+1), cov_star).rvs(size)
        x0, x1   = x[:, 0], x[:, 1:]
        inds     = x0 <= 0
        x1[inds] = -1 * x1[inds]
        return x1

        
# Assuming your multivariate_skewnorm class is defined above as we have discussed

# First, let's create an instance of the multivariate_skewnorm distribution
shape = np.array([5, 1])
cov = np.array([[1, 0.9], [0.9, 1.0]])
skewnorm = multivariate_skewnorm(shape, cov)

# Now let's generate some data from this distribution
data = skewnorm.rvs_fast(size=1000)

# # Now let's fit the distribution to this data
# skewnorm.fit(data)

sns.kdeplot(x=data[:,0], y=data[:,1], fill=True)

# Now the shape and cov attributes of skewnorm should be close to the original parameters
# print(skewnorm.shape)
# print(skewnorm.cov)



#%% ====================================================================================================
#% Select the entries to be used in training
selected_entries_X = [
                      'Airmass',
                      'r0 (SPARTA)',
                      'FWHM',
                      'Wind direction (header)',
                      'Wind speed (header)',
                      'Tau0 (header)',
                      'Nph WFS',
                      'Rate',
                      'Jitter X',
                      'Jitter Y',
                      'λ left (nm)',
                      'λ right (nm)',
                    #   'Turb. speed',
                    #   'Pressure',
                    #   'Humidity',
                    #   'Temperature'
                      ]

selected_entries_Y = ['Jx', 'Jy', 'Jxy', 'F (left)', 'F (right)']

X_df = input_df[selected_entries_X]
Y_df = output_df[selected_entries_Y]
# choice = 'F (left)'
choice = 'Jx'

# X = X_df.to_numpy()
# Y = Y_df[choice].to_numpy()
# Split the dataset into a training set and a test set
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(X_df, Y_df, test_size=0.2, random_state=42)

X_train = X_train_df.to_numpy()
X_test  = X_test_df.to_numpy()
y_train = y_train_df[choice].to_numpy()
y_test  = y_test_df[choice].to_numpy()

#%
model_choice = 'gbr'

if model_choice == 'gbr':
    # Create a gradient boosting regressor
    gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=7, random_state=42)
    # Train the model
    gbr.fit(X_train, y_train)
    # Make predictions using the test set
    y_pred = gbr.predict(X_test)
    
elif model_choice == 'rf':
    rf = RandomForestRegressor(n_estimators=300, max_depth=7, random_state=42)
    # Train the model
    rf.fit(X_train, y_train)
    # Make predictions using the test set
    y_pred = rf.predict(X_test)
    
elif model_choice == 'svr':
    svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    # Train the model
    svr.fit(X_train, y_train)
    # Make predictions using the test set
    y_pred = svr.predict(X_test)
    
elif model_choice == 'mlp':
    # Create an MLPRegressor object
    mlp = MLPRegressor(hidden_layer_sizes=(50,50,), max_iter=2000, random_state=42)
    # Train the MLPRegressor
    mlp.fit(X_train, y_train)
    # Make predictions with the MLPRegressor
    y_pred = mlp.predict(X_test)

#%
y_test_1 = transforms_output[choice].backward(y_test)
y_pred_1 = transforms_output[choice].backward(y_pred)

if choice == 'Jx' or choice == 'Jy' or choice == 'Jxy':
    err = np.abs(y_test_1-y_pred_1)
    print("The mean absolute error (MAE) on test set: {:.4f}".format(err.mean()), "[mas]")
    print("The median absolute error (MAE) on test set: {:.4f}".format(np.median(err)), "[mas]")
    print("The std absolute error (MAE) on test set: {:.4f}".format(err.std()), "[mas]")
    print("The max absolute error (MAE) on test set: {:.4f}".format(err.max()), "[mas]")
    print("The min absolute error (MAE) on test set: {:.4f}".format(err.min()), "[mas]")
    
elif choice == 'F (left)' or choice == 'F (right)':
    err = np.abs(y_test_1-y_pred_1) * 100
    print("The mean absolute error (MAE) on test set: {:.4f}".format(err.mean()), "%")
    print("The median absolute error (MAE) on test set: {:.4f}".format(np.median(err)), "%")
    print("The std absolute error (MAE) on test set: {:.4f}".format(err.std()), "%")
    print("The max absolute error (MAE) on test set: {:.4f}".format(err.max()), "%")
    print("The min absolute error (MAE) on test set: {:.4f}".format(err.min()), "%")



#%% ====================================================================================================
# dict2vec = lambda Y: np.stack([transforms_output[a].forward(Y[a].values) for a in selected_entries_Y]).T
vec2dict = lambda Y: { selected_entries_Y[i]: transforms_output[selected_entries_Y[i]].backward(Y[:,i]) for i in range(len(selected_entries_Y)) }

X_df = input_df[selected_entries_X]
# X_df = psf_df[selected_entries_X]
Y_df = output_df[selected_entries_Y]

X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(X_df, Y_df, test_size=0.2, random_state=42)

X_train = X_train_df.to_numpy()
X_test  = X_test_df.to_numpy()
y_train = y_train_df.to_numpy()
y_test  = y_test_df.to_numpy()

model_choice = 'mlp'

if model_choice == 'gbr':
    # Create a gradient boosting regressor
    gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=7, random_state=42)
    # Make it a multi-output regressor
    mor = MultiOutputRegressor(gbr, n_jobs=-1)
    # Train the MultiOutputRegressor
    mor.fit(X_train, y_train)
    # Make predictions with the MultiOutputRegressor
    y_pred = mor.predict(X_test)

elif model_choice == 'mlp':
    # Create an MLPRegressor object
    mlp = MLPRegressor(hidden_layer_sizes=(50,50,), max_iter=2000, random_state=42)
    # Train the MLPRegressor
    mlp.fit(X_train, y_train)
    # Make predictions with the MLPRegressor
    y_pred = mlp.predict(X_test)


#%%
y_pred_1 = pd.DataFrame(vec2dict(y_pred))
y_test_1 = pd.DataFrame(vec2dict(y_test))
delta = y_pred_1 - y_test_1

def print_err(err, name, a):
    if a == "%": err *= 100	
    print(name + ":")
    print("\t The mean absolute error (MAE) on test set: {:.4f}".format(err.mean()), a)
    print("\t The median absolute error (MAE) on test set: {:.4f}".format(np.median(err)), a)
    print("\t The std absolute error (MAE) on test set: {:.4f}".format(err.std()), a)
    print("\t The max absolute error (MAE) on test set: {:.4f}".format(err.max()), a)
    print("\t The min absolute error (MAE) on test set: {:.4f}".format(err.min()), a)
    print()

print_err(delta['Jx'].abs(),        'Jx', "[mas]")
print_err(delta['Jy'].abs(),        'Jy', "[mas]")
print_err(delta['Jxy'].abs(),       'Jxy', "[mas]")
print_err(delta['F (left)'].abs(),  'F (left)', "%")
print_err(delta['F (right)'].abs(), 'F (right)', "%")

#%%
from project_globals import device
import torch.nn as nn
import torch.optim as optim

# Define the network architecture
class Gnosis(nn.Module):
    def __init__(self, in_size, out_size, hidden_size=50, dropout_p=0.0):
        super(Gnosis, self).__init__()
        self.fc1 = nn.Linear(in_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, out_size)
        
        self.dropout1 = nn.Dropout(dropout_p)
        self.dropout2 = nn.Dropout(dropout_p)
        self.dropout3 = nn.Dropout(dropout_p)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.dropout1(x)
        x = torch.tanh(self.fc2(x))
        x = self.dropout2(x)
        x = torch.tanh(self.fc3(x))
        x = self.dropout3(x)
        return x
    
# Initialize the network, loss function and optimizer
net = Gnosis(X_train.shape[1], y_train.shape[1], 50)
#%%
import torchbnn as bnn

net = nn.Sequential(
    bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=X_train.shape[1], out_features=50),
    nn.Tanh(),
    bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=50, out_features=50),
    nn.Tanh(),
    bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=50, out_features=y_train.shape[1]),
    nn.Tanh(),
)

ce_loss = nn.MSELoss()
kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
kl_weight = 0.1
#%
net.double().to(device)

optimizer = optim.Adam(net.parameters(), lr=0.0001)

X_train_torch = torch.from_numpy(X_train).double().to(device)
y_train_torch = torch.from_numpy(y_train).double().to(device)

num_iters = 20000
for iter in range(num_iters):  # number of epochs
    optimizer.zero_grad()   # zero the gradient buffers
    ce = ce_loss(net(X_train_torch), y_train_torch)
    kl = kl_loss(net)
    cost = ce + kl_weight*kl
    cost.backward()
    print('({:d}/{:d}) current loss: {:.2f}'.format(iter+1, num_iters, cost.item()), end='\r')
    # loss = loss_fn(net(X_train_torch), y_train_torch)  # compute loss
    # loss.backward()  # backpropagation
    optimizer.step()  # update weights

torch.save(net, WEIGHTS_FOLDER/'GnosisBNN.pth')
#%%
# y_pred_torch = net(torch.from_numpy(X_test).double().to(device)).cpu().detach().numpy()
y_test_1 = pd.DataFrame(vec2dict(y_test))

dfs = []
# y_preds_torch = []
for i in range(200):
    y_pred_torch = net(torch.from_numpy(X_test).double().to(device)).cpu().detach().numpy()
    y_pred_torch_1 = pd.DataFrame(vec2dict(y_pred_torch))
    delta_torch = y_pred_torch_1 - y_test_1
    dfs.append(delta_torch.abs())

a = lambda entry: (np.array([df[entry] for df in dfs]).mean(axis=0), np.array([df[entry] for df in dfs]).std(axis=0))

def print_err_prob(err, std, name, a):
    if a == "%": err *= 100	
    print(name + ":")
    print("\t The mean absolute error (MAE) on test set: {:.2f}+-{:.2f}".format(err.mean(), std.mean()), a)
    print("\t The median absolute error (MAE) on test set: {:.2f}+-{:.2f}".format(np.median(err), np.median(std)), a)
    print("\t The std absolute error (MAE) on test set: {:.2f}+-{:.2f}".format(err.std(), std.std()), a)
    print("\t The max absolute error (MAE) on test set: {:.2f}+-{:.2f}".format(err.max(), std.max()), a)
    print("\t The min absolute error (MAE) on test set: {:.2f}+-{:.2f}".format(err.min(), std.min()), a)
    print()

print_err_prob(*a('Jx'), 'Jx', "[mas]")
print_err_prob(*a('Jy'), 'Jy', "[mas]")
print_err_prob(*a('Jxy'), 'Jxy', "[mas]")
print_err_prob(*a('F (left)'), 'F (left)', "%")
print_err_prob(*a('F (right)'), 'F (right)', "%")

# print_err_prob(delta_torch['Jx'].abs(), y_pred_torch_std[:,0], 'Jx', "[mas]")
# print_err_prob(delta_torch['Jy'].abs(), y_pred_torch_std[:,1], 'Jy', "[mas]")
# print_err_prob(delta_torch['Jxy'].abs(), y_pred_torch_std[:,2], 'Jxy', "[mas]")
# print_err_prob(delta_torch['F (left)'].abs(), y_pred_torch_std[:,3], 'F (left)', "%")
# print_err_prob(delta_torch['F (right)'].abs(), y_pred_torch_std[:,4], 'F (right)', "%")


# %%
