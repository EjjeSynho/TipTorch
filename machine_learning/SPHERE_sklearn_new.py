#%%
%reload_ext autoreload
%autoreload 2

import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torch.optim as optim
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
from tools.utils import seeing, corr_plot
from astropy.io import fits
from data_processing.normalizers import CreateTransformSequenceFromFile
from sklearn.inspection import permutation_importance

from project_globals import WEIGHTS_FOLDER


def AnalyseImpurities(model, feature_names, X_test=None, y_test=None):
    # Compute MDI (Mean Decrease in Impurity) feature importances
    importances = model.feature_importances_
    forest_importances_MDI_std = np.std([tree[0].feature_importances_ for tree in model.estimators_], axis=0)

    forest_importances_MDI = pd.Series(importances, index=feature_names)

    fig, ax = plt.subplots()
    forest_importances_MDI.plot.bar(yerr=forest_importances_MDI_std, ax=ax)
    ax.set_title("Feature importances using mean decrease in impurity")
    ax.set_ylabel("Mean decrease in impurity")
    ax.set_xticklabels(feature_names, rotation=40, ha='right')
    fig.tight_layout()
    plt.show()

    forest_importances_perm, forest_importances_perm_std = None, None
    
    if X_test is not None and y_test is not None:
        result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)

        forest_importances_perm = pd.Series(result.importances_mean, index=feature_names)
        forest_importances_perm_std = result.importances_std

        fig, ax = plt.subplots()
        forest_importances_perm.plot.bar(yerr=result.importances_std, ax=ax)
        ax.set_title("Feature importances using permutation on full model")
        ax.set_ylabel("Mean accuracy decrease")
        ax.set_xticklabels(feature_names, rotation=40, ha='right')
        fig.tight_layout()
        plt.show()
    
    return forest_importances_MDI, forest_importances_MDI_std, forest_importances_perm, forest_importances_perm_std

#%% Initialize data sample
with open(SPHERE_DATA_FOLDER+'sphere_df.pickle', 'rb') as handle:
    psf_df = pickle.load(handle)

with open(SPHERE_DATA_FOLDER+'fitted_df.pickle', 'rb') as handle:
    fitted_df = pickle.load(handle)

with open('../data/temp/fitted_df_norm.pickle', 'rb') as handle:
    fitted_df_norm = pickle.load(handle)

with open('../data/temp/psf_df_norm.pickle', 'rb') as handle:
    psf_df_norm = pickle.load(handle)

SPHERE_pupil = fits.getdata('../data/calibrations/VLT_CALIBRATION/VLT_PUPIL/ALC2LyotStop_measured.fits')
LWE_basis = np.load('../data/LWE_basis_SPHERE.npy')

df_transforms_onsky  = CreateTransformSequenceFromFile('../data/temp/psf_df_norm_transforms.pickle')
df_transforms_fitted = CreateTransformSequenceFromFile('../data/temp/fitted_df_norm_transforms.pickle')

get_ids = lambda df: set( df.index.values.tolist() )

ids_0 = get_ids(fitted_df)
ids_1 = get_ids(psf_df)
ids_2 = get_ids(fitted_df_norm)
ids_3 = get_ids(psf_df_norm)
ids   = list( ids_0.intersection(ids_1).intersection(ids_2).intersection(ids_3) )

print('Final number of good samples:', len(ids))

psf_df = psf_df.loc[ids]
fitted_df = fitted_df.loc[ids]
psf_df_norm = psf_df_norm.loc[ids]
fitted_df_norm = fitted_df_norm.loc[ids]

for entry in ['r0', 'Jx', 'Jy', 'Jxy']: #,'F L','F R']:
    fitted_df[entry] = fitted_df[entry].abs()
    fitted_df_norm[entry] = fitted_df_norm[entry].abs()


#%% ================================== Predict Strehl ==================================
# =======================================================================================
df_norm = pd.concat([psf_df_norm, fitted_df_norm], axis=1)#.fillna(0)

selected_entries_X = [
    'Airmass',
    'r0 (SPARTA)',
    # 'Azimuth',
    # 'Altitude',
    # 'Seeing (SPARTA)',
    # 'FWHM',
    'Wind direction (header)',
    'Wind direction (MASSDIMM)',
    'Wind speed (header)',
    'Wind speed (SPARTA)',
    'Wind speed (MASSDIMM)',
    'Tau0 (header)',
    'Tau0 (SPARTA)',
    # 'Jitter X', 'Jitter Y',
    'Focus',
    'ITTM pos (avg)',
    'ITTM pos (std)',
    'DM pos (avg)',
    'DM pos (std)',
    'Pressure',
    'Humidity',
    'Temperature',
    'Nph WFS',
    # 'Flux WFS',
    'Rate',
    # 'DIT Time',
    'Sequence time',
    'λ left (nm)',  'λ right (nm)',
    # 'Δλ left (nm)', 'Δλ right (nm)',
    # 'mag V', 'mag R', 'mag G', 'mag J', 'mag H', 'mag K'
#--------------------------------------------------------------------------------------------------------
]

selected_entries_Y = ['Strehl']

for entry in selected_entries_Y:
    if entry in selected_entries_X:
        selected_entries_X.pop(selected_entries_X.index(entry))

    if entry not in df_norm.columns.values:
        selected_entries_X.pop(selected_entries_X.index(entry))

X = df_norm[selected_entries_X].to_numpy()
Y = df_norm[selected_entries_Y].to_numpy()

#%%
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=7, random_state=42)
gbr.fit(X_train, y_train)
y_pred = gbr.predict(X_test)

test_df = pd.DataFrame({
    'SR predicted': df_transforms_onsky['Strehl'].backward(y_pred),
    'SR from data': df_transforms_onsky['Strehl'].backward(y_test.flatten()),
})
corr_plot(test_df, 'SR predicted', 'SR from data')

_ = AnalyseImpurities(gbr, selected_entries_X, X_test, y_test)

#%% ================================== Predict LWE WFE ==================================
# =======================================================================================

# df_norm = df_norm[df_norm['LWE'] == True]
X = df_norm[selected_entries_X].to_numpy()
Y = df_norm['LWE coefs'].apply(lambda x: np.linalg.norm(x, ord=2)) # Just calculate the normalized LWE WFE

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=7, random_state=42)
gbr.fit(X_train, y_train)
y_pred = gbr.predict(X_test)

#%%
test_df = pd.DataFrame({
    'LWE predicted': df_transforms_fitted['LWE coefs'].backward(y_pred),
    'LWE from data': df_transforms_fitted['LWE coefs'].backward(y_test),
})
corr_plot(test_df, 'LWE predicted', 'LWE from data', lims=[0, 500])

_ = AnalyseImpurities(gbr, selected_entries_X, X_test, y_test)

#%%
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, explained_variance_score, median_absolute_error

MAPE  = mean_absolute_percentage_error(y_test, y_pred)
EVP   = explained_variance_score(y_test, y_pred) # 1.0 is the best
MedAE = median_absolute_error(y_test, y_pred)

print( 'Mean absolute percentage error (MAPE): {:.3f}'.format(MAPE) )
print( 'Explained Variance Score (EVP) {:.3f}'.format(EVP) )
print( 'Median absolute error (MedAE) {:.3f}'.format(MedAE) )


#%% ================================== Predict LWE coefficients =========================
# =======================================================================================

from sklearn.decomposition import PCA

LWE_coefs = Y = np.array([x for x in df_norm['LWE coefs'].to_numpy()])

pca = PCA(n_components=11)
pca.fit(LWE_coefs)

plt.title('PCA of LWE coefs')
plt.scatter(Y[:,0], pca.inverse_transform(pca.transform(Y))[:,0], label='Train')
plt.xlabel('Original')
plt.ylabel('Reconstructed')

#%%
# Make dataframes from the PCA results
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=7, random_state=42)
mor = MultiOutputRegressor(gbr, n_jobs=-1)
mor.fit(X_train, y_train_pca:=pca.transform(y_train))
y_pred_pca = mor.predict(X_test)
y_pred = pca.inverse_transform(y_pred_pca)

#%%
names = ['Piston',]*4 + ['Tip',]*4 + ['Tilt',]*4

for i in range(0, 4):
    names[i]   += f' {i+1}'
    names[i+4] += f' {i+1}'
    names[i+8] += f' {i+1}'
names.pop(0)

MDIs, MDI_stds, PIs, PI_stds = [], [], [], []

for i in tqdm(range(0, 11)):
    test_df = pd.DataFrame({
        names[i]+' predicted [nm RMS]': df_transforms_fitted['LWE coefs'].backward(y_pred[:,i]),
        names[i]+' from data [nm RMS]': df_transforms_fitted['LWE coefs'].backward(y_test[:,i]),
    })
    corr_plot(test_df, names[i]+' predicted [nm RMS]', names[i]+' from data [nm RMS]', title=names[i])
    # plt.savefig(f'C:/Users/akuznets/Desktop/presa_buf/LWE/LWE_coefs_corr_plots/LWE_coefs_{i}.png')

    MDI, MDI_std, PI, PI_std = AnalyseImpurities(mor.estimators_[ 0], selected_entries_X, X_test, y_test[:,i])

    MDIs.append(MDI)
    MDI_stds.append(MDI_std)
    PIs.append(PI)
    PI_stds.append(PI_std)

MDIs     = np.stack(MDIs)
PIs      = np.stack(PIs)
MDI_stds = np.stack(MDI_stds)
PI_stds  = np.stack(PI_stds)

#%%
MDIs_m     = MDIs.mean(axis=0)
PIs_m      = PIs.mean(axis=0)
MDI_stds_m = MDI_stds.mean(axis=0)
PI_stds_m  = PI_stds.mean(axis=0)

fig, ax = plt.subplots(1, 2, figsize=(15, 5))

ax[0].bar(selected_entries_X, MDIs_m, yerr=MDI_stds_m)
ax[0].set_title("Mean decrease in impurity")
ax[0].set_ylabel("Mean decrease in impurity")
ax[0].set_xticklabels(selected_entries_X, rotation=40, ha='right')

ax[1].bar(selected_entries_X, PIs_m, yerr=PI_stds_m)
ax[1].set_title("Permutation importance")
ax[1].set_ylabel("Mean accuracy decrease")
ax[1].set_xticklabels(selected_entries_X, rotation=40, ha='right')

fig.tight_layout()
# plt.savefig('C:/Users/akuznets/Desktop/presa_buf/LWE/LWE_coefs_importances.png')

#%%
phase_pred = np.dot(y_pred, LWE_basis.reshape(11,-1))
phase_pred = phase_pred.reshape(-1, LWE_basis.shape[1], LWE_basis.shape[2])

phase_test = np.dot(y_test, LWE_basis.reshape(11,-1))
phase_test = phase_test.reshape(-1, LWE_basis.shape[1], LWE_basis.shape[2])

for i in range(0, phase_pred.shape[0]):
    A = phase_test[i,...]
    B = phase_pred[i,...]

    MSE_1 = np.mean( (A-B)**2 )
    MSE_2 = np.mean( (A+B)**2 )

    if MSE_1 > MSE_2:
        phase_pred[i,...] *= -1

phase_pred *= 2
# phase_pred *= 1.5
diff = phase_test - phase_pred

def calc_WFEs(cube):
    cube = np.atleast_3d(cube)
    WFEs = np.zeros(cube.shape[0])

    for i in range(cube.shape[0]):
        WFEs[i] = np.std(cube[i,...][SPHERE_pupil>0])
        
    return np.squeeze(WFEs)

WFE_test = calc_WFEs(phase_test)
WFE_pred = calc_WFEs(phase_pred)
dWFE     = calc_WFEs(diff)

# _ = plt.hist(WFE_test, bins=20, alpha=0.5, label='Test')
# _ = plt.hist(WFE_pred, bins=20, alpha=0.5, label='Predicted')
# plt.legend()

print(WFE_test.mean(), WFE_pred.mean(), dWFE.mean())

#%%
rand_id = np.random.randint(0, phase_pred.shape[0])

A = phase_test[rand_id,...]
B = phase_pred[rand_id,...]
C = diff      [rand_id,...]

fig, ax = plt.subplots(1, 3, figsize=(10, 5))

c_lim = max(A.max(), B.max(), -A.min(), -B.min())

ax[0].imshow(A, vmin=-c_lim, vmax=c_lim, cmap='viridis')
ax[0].set_title('Fitted LWE')
ax[0].grid(False)
ax[0].axis('off')

ax[1].imshow(B, vmin=-c_lim, vmax=c_lim, cmap='viridis')
ax[1].set_title('Predicted LWE')
ax[1].grid(False)
ax[1].axis('off')

ax[2].imshow(C, vmin=-c_lim, vmax=c_lim, cmap='viridis')
ax[2].set_title('Difference')
ax[2].grid(False)
ax[2].axis('off')
plt.tight_layout()

plt.savefig(f'C:/Users/akuzne ts/Desktop/presa_buf/LWE/LWE_{rand_id}.png', dpi=300)

#%% =============================================================================================
transforms = {**transforms_input, **transforms_output}

trans_df = df.copy()
for entry in transforms:
    trans_df[entry] = transforms[entry].forward(trans_df[entry])

selected_X = ['r0 (data)',
              'Rate',
              'Nph WFS (data)',
              'Wind speed (header)',
              'Wind direction (header)',
            #   'Flux WFS',
              'Wind speed (200 mbar)',
              'Wind direction (200 mbar)']

selected_Y = [
    'Rec. noise',
    'Jx',
    'Jy',
    'F (left)',
    'F (right)']


X_df = trans_df[selected_X].fillna(0)
Y_df = trans_df[selected_Y].fillna(0)

NN2in  = lambda X: { selected: transforms[selected].backward(X[:,i]) for i,selected in enumerate(selected_X) }
NN2fit = lambda Y: { selected: transforms[selected].backward(Y[:,i]) for i,selected in enumerate(selected_Y) }
in2NN  = lambda inp: torch.from_numpy(( np.stack([transforms[a].forward(inp[a].values) for a in selected_X]))).T
fit2NN = lambda out: torch.from_numpy(( np.stack([transforms[a].forward(out[a].values) for a in selected_Y]))).T

for entry in selected_X:
    trans_df = trans_df[trans_df[entry].abs() < 3]
    # print(trans_df[entry].abs() < 3)

for entry in selected_Y:
    trans_df = trans_df[trans_df[entry].abs() < 3]

ids = X_df.index.intersection(Y_df.index)
X_df = X_df.loc[ids]
Y_df = Y_df.loc[ids]

X_df_train, X_df_test, y_df_train, y_df_test = train_test_split(X_df, Y_df, test_size=0.2, random_state=42)

class ParamPredictor(nn.Module):
    def __init__(self, in_size, out_size, hidden_size=100, dropout_p=0.0):
        super(ParamPredictor, self).__init__()
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
        return x

net = ParamPredictor(len(selected_X), len(selected_Y), 100, 0.05)
net = net.to(device)

optimizer = optim.Adam(net.parameters(), lr=0.0001)

X_train = torch.from_numpy(X_df_train.to_numpy()).to(device).float()
Y_train = torch.from_numpy(y_df_train.to_numpy()).to(device).float()
X_val   = torch.from_numpy(X_df_test.to_numpy()).to(device).float()
Y_val   = torch.from_numpy(y_df_test.to_numpy()).to(device).float()

loss_fn = nn.MSELoss()

weights_name = WEIGHTS_FOLDER/f'param_predictor_{len(selected_X)}x{len(selected_Y)}_real.pth'

if not os.path.exists(weights_name):
    num_iters = 30000
    for iter in range(num_iters):
        optimizer.zero_grad()
        loss_train = loss_fn(net(X_train), Y_train)
        loss_train.backward()
        optimizer.step()
        
        with torch.no_grad():
            loss_val = loss_fn(net(X_val), Y_val)
        print('({:d}/{:d}) train, valid: {:.2f}, {:.2f}'.format(iter+1, num_iters, loss_train.item()*100, loss_val.item()*100), end='\r')

    torch.save(net, weights_name)
else:
    net = torch.load(weights_name)

y_pred = net(X_val).cpu().detach().numpy()

del X_train, Y_train, X_val, Y_val
torch.cuda.empty_cache()

y_pred_df = pd.DataFrame(y_pred, index=X_df_test.index, columns=y_df_test.columns)

#%% ================ Telemetry SR from telemetry ===================
#% Select the entries to be used in training
selected_entries_X = [
        # 'Airmass',
        # 'Seeing (MASSDIMM)',
        # 'FWHM',
        # 'Wind direction (MASSDIMM)',
        # 'Wind speed (MASSDIMM)',
        # 'Tau0 (MASSDIMM)',
        # 'Seeing (SPARTA)',
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

#%%
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
    'J predicted': transforms_output_synth['Jx'].backward(y_pred),
    'J test': transforms_output_synth['Jx'].backward(y_test),
})

corr_plot(test_df, 'J predicted', 'J test')


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
