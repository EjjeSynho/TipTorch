#%%
%reload_ext autoreload
%autoreload 2

import os
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
from project_settings import MUSE_DATA_FOLDER
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from tools.utils import seeing, corr_plot
from astropy.io import fits
from data_processing.normalizers import CreateTransformSequenceFromFile
from sklearn.inspection import permutation_importance
from matplotlib.colors import LogNorm
from project_settings import WEIGHTS_FOLDER
from sklearn.preprocessing import StandardScaler


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
with open(MUSE_DATA_FOLDER+'muse_df.pickle', 'rb') as handle:
    psf_df = pickle.load(handle)

with open(MUSE_DATA_FOLDER+'MUSE_fitted_df.pkl', 'rb') as handle:
    fitted_df = pickle.load(handle)
    
with open(MUSE_DATA_FOLDER+'muse_df_reduced_imputed.pickle', 'rb') as handle:
    imputed_df = pickle.load(handle)

#%%
outputs = fitted_df['singular_vals_df']

scaler_outputs = StandardScaler()
outputs_data = scaler_outputs.fit_transform(outputs)
outputs_norm = pd.DataFrame(outputs_data, columns=outputs.columns)
outputs_norm.dropna(inplace=True)

relevant_inputs = [\
    'frequency', 'gain',
    'IRLOS photons, [photons/s/m^2]', 'NGS mag (from ph.)',
    'LGS1 photons, [photons/m^2/s]', 'LGS2 photons, [photons/m^2/s]',
    'LGS3 photons, [photons/m^2/s]', 'LGS4 photons, [photons/m^2/s]',
    'RA (science)', 'DEC (science)', 'Exp. time',
    'Tel. altitude', 'Tel. azimuth', 'Par. angle', 'Derot. angle',
    'NGS RA', 'NGS DEC', 'Pixel scale (science)', 'Airmass',
    'Seeing (header)', 'Tau0 (header)', 'Temperature (header)',
    'Wind dir (header)', 'Wind speed (header)', 'R0', 'MASS_FRACGL',
    'DIMM_SEEING', 'ASM_RFLRMS', 'SLODAR_FRACGL_300',
    'SLODAR_FRACGL_500', 'ASM_WINDDIR_10', 'ASM_WINDDIR_30',
    'ASM_WINDSPEED_10', 'ASM_WINDSPEED_30',
    'Wind Direction at 30m (0/360) [deg]', 'Wind Speed at 30m [m/s]',
    'Free Atmosphere Seeing ["]', 'MASS Tau0 [s]',
    'MASS-DIMM Cn2 fraction at ground', 'MASS-DIMM Seeing ["]',
    'MASS-DIMM Tau0 [s]', 'MASS-DIMM Turb Velocity [m/s]',
    'DIMM Seeing ["]', 'Relative Flux RMS', 'Cn2 fraction below 2000m'
]

# Copy only selected columns
input_df = imputed_df[relevant_inputs]

#%%
input_df = input_df.replace([np.inf, -np.inf], np.nan)
input_df.dropna(inplace=True)

scaler_inputs = StandardScaler()
input_data = scaler_inputs.fit_transform(input_df)
inputs_norm = pd.DataFrame(input_data, columns=input_df.columns)

index_in    = set(inputs_norm.index)
index_out   = set(outputs_norm.index)
index_total = list(index_in.intersection(index_out))

inputs_norm  = inputs_norm.loc[index_total]
outputs_norm = outputs_norm.loc[index_total]

#%%
X = inputs_norm.to_numpy()
Y = outputs_norm['b'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.0001, max_depth=100, random_state=42)
gbr.fit(X_train, y_train)
y_pred = gbr.predict(X_test)

#%%
plt.plot(y_test, y_pred, 'o')
plt.xlim([-0.5, -0.3])

# _ = AnalyseImpurities(gbr, selected_entries_X, X_test, y_test)

#%% ================================== Predict all ==================================
# =======================================================================================

selected_entries_Y = ['r0', 'dn', 'Jx', 'Jy', 'Jxy', 'F L', 'F R']

X = df_norm[selected_entries_X].to_numpy()
Y = df_norm[selected_entries_Y].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

mlp = MLPRegressor(hidden_layer_sizes=(50,50,), max_iter=2000, random_state=42)
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)

#%%
param_to_plot = 'F R'
id = selected_entries_Y.index(param_to_plot)

test_df = pd.DataFrame({
    param_to_plot+' predicted':   df_transforms_fitted[param_to_plot].backward(y_pred[:,id].flatten()),
    param_to_plot+' from fitted': df_transforms_fitted[param_to_plot].backward(y_test[:,id].flatten()),
})
corr_plot(test_df, param_to_plot+' predicted', param_to_plot+' from fitted')

# _ = AnalyseImpurities(gbr, selected_entries_X, X_test, y_test)



#%% ================================== Predict LWE WFE ==================================
# =======================================================================================

# df_norm = df_norm[df_norm['LWE'] == True]
X = df_norm[selected_entries_X].to_numpy()
Y = df_norm['LWE coefs'].apply(lambda x: np.linalg.norm(x, ord=2)) # Just calculate the normalized LWE WFE

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

gbr_LWE = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=7, random_state=42)
gbr_LWE.fit(X_train, y_train)
y_pred = gbr_LWE.predict(X_test)

#%%
test_df = pd.DataFrame({
    'LWE predicted':   df_transforms_fitted['LWE coefs'].backward(y_pred),
    'LWE from fitted': df_transforms_fitted['LWE coefs'].backward(y_test),
})
corr_plot(test_df, 'LWE predicted', 'LWE from fitted', lims=[0, 200])

_ = AnalyseImpurities(gbr_LWE, selected_entries_X, X_test, y_test)

#%%
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, explained_variance_score, median_absolute_error

MAPE  = mean_absolute_percentage_error(y_test, y_pred)
EVP   = explained_variance_score(y_test, y_pred) # 1.0 is the best
MedAE = median_absolute_error(y_test, y_pred)

print( 'Mean absolute percentage error (MAPE): {:.3f}'.format(MAPE) )
print( 'Explained Variance Score (EVP) {:.3f}'.format(EVP) )
print( 'Median absolute error (MedAE) {:.3f}'.format(MedAE) )


#%% ============================ Predict LWE coefficients ============================
# ====================================================================================

from sklearn.decomposition import PCA

LWE_coefs = np.array([x for x in df_norm['LWE coefs'].to_numpy()])

pca = PCA(n_components=8)
pca.fit(LWE_coefs)

plt.title('PCA of LWE coefs')
plt.scatter(LWE_coefs[:,0], pca.inverse_transform(pca.transform(LWE_coefs))[:,0], label='Train')
plt.xlabel('Original')
plt.ylabel('Reconstructed')

Y_pca = pca.transform(LWE_coefs)

#%%
# Make dataframes from the PCA results
X_train, X_test, y_train_pca, y_test_pca = train_test_split(X, Y_pca, test_size=0.2, random_state=42)

gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=7, random_state=42)
mor = MultiOutputRegressor(gbr, n_jobs=-1)
mor.fit(X_train, y_train_pca)
y_pred_pca = mor.predict(X_test)

y_pred  = df_transforms_fitted['LWE coefs'].backward(pca.inverse_transform(y_pred_pca))
y_test  = df_transforms_fitted['LWE coefs'].backward(pca.inverse_transform(y_test_pca))
y_train = df_transforms_fitted['LWE coefs'].backward(pca.inverse_transform(y_train_pca))

#%%
from joblib import dump, load

store_dict = {
    'PCA': pca,
    'inputs': selected_entries_X,
    'LWE WFE predictor': mor,
    'LWE coefs predictor': gbr_LWE,
}
dump(store_dict, '../data/LWE.predictor')

# IDs = [456, 1112, 1610, 579]

# def run_predictor(IDs):
#     X_inp = df_norm.loc[IDs][selected_entries_X].to_numpy()
#     LWE_WFE_pred = df_transforms_fitted['LWE coefs'].backward(gbr_LWE.predict(X_inp))
#     LWE_coefs_pred_pca = mor.predict(X_inp)
#     LWE_coefs_pred = df_transforms_fitted['LWE coefs'].backward(pca.inverse_transform(LWE_coefs_pred_pca))
#     return LWE_coefs_pred

#%%
WFE_pred_pow = df_transforms_fitted['LWE coefs'].backward(gbr_LWE.predict(X_test))

WFE_pred = np.linalg.norm(y_pred, ord=2, axis=1)

y_pred_norm = y_pred / WFE_pred[:,None]
y_pred_norm *= WFE_pred_pow[:,None]

WFE_pred = np.linalg.norm(y_pred_norm, ord=2, axis=1)
WFE_test = np.linalg.norm(y_test, ord=2, axis=1)

plt.scatter(WFE_test, WFE_pred, s=2)
# plt.scatter(WFE_pred, WFE_pred_pow, s=2)

# plt.xlim(0, 200)
# plt.ylim(0, 200)
y_pred = y_pred_norm

#%%
names = ['Piston',]*4 + ['Tip',]*4 + ['Tilt',]*4

for i in range(0, 4):
    names[i]   += f' {i+1}'
    names[i+4] += f' {i+1}'
    names[i+8] += f' {i+1}'
# names.pop(0)

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
plt.savefig('C:/Users/akuznets/Desktop/presa_buf/LWE/LWE_coefs_importances.png')

#%%
phase_pred  = np.dot(y_pred, LWE_basis.reshape(12,-1))
phase_pred  = phase_pred.reshape(-1, LWE_basis.shape[1], LWE_basis.shape[2])

phase_test  = np.dot(y_test, LWE_basis.reshape(12,-1))
phase_test  = phase_test.reshape(-1, LWE_basis.shape[1], LWE_basis.shape[2])

phase_train = np.dot(y_train, LWE_basis.reshape(12,-1))
phase_train = phase_train.reshape(-1, LWE_basis.shape[1], LWE_basis.shape[2])


flipped_ids = []

for i in range(0, phase_pred.shape[0]):
    A, B = phase_test[i,...], phase_pred[i,...]

    MSE_1 = np.mean( (A-B)**2 )
    MSE_2 = np.mean( (A+B)**2 )

    if MSE_1 > MSE_2:
        phase_pred[i,...] *= -1
        flipped_ids.append(i)

flipped_id = np.array(flipped_ids) 


diff = phase_test - phase_pred

def calc_WFEs(cube):
    
    if cube.ndim == 2:
        cube = cube[None,...]
    
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

# A /= calc_WFEs(A)
# B /= calc_WFEs(B)
# C = A - B

fig, ax = plt.subplots(1, 3, figsize=(8, 4))

c_lim = np.array([50, np.abs(A.max()), np.abs(B.max())]).max()

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

PCM=ax[2].get_children()[0] #get the mappable, the 1st and the 2nd are the x and y axes
plt.colorbar(PCM, ax=ax, anchor=(2.2,0.0))

plt.tight_layout()
# plt.show()
# plt.savefig(f'../data/temp/LWE_shapes/LWE_{rand_id}.png')

#%%

phase_all = np.concatenate([phase_train, phase_test], axis=0)
#%%
rand_id = np.random.randint(0, phase_all.shape[0])
# phase_all -= phase_all.mean(axis=0)[None,...]

#%
plt.imshow(phase_all[rand_id,...], cmap='viridis')
plt.grid(False)
plt.colorbar()
plt.axis('off')

#%% =================================================================================================
im_data = np.abs( images_df[rand_id][0][0,0,...] )
im_fit  = np.abs( images_df[rand_id][1][0,0,...] )

norm = LogNorm(vmin=np.maximum(im_data.min(), 1), vmax=np.minimum(im_data.max(), 1e16))
fig2, ax2 = plt.subplots(1, 2)

ax2[0].imshow(im_data, cmap='viridis', norm=norm)
ax2[0].set_title('Data')
ax2[0].grid(False)
ax2[0].axis('off')

ax2[1].imshow(im_fit, cmap='viridis', norm=norm)
ax2[1].set_title('Fitted')
ax2[1].grid(False)
ax2[1].axis('off')
plt.tight_layout()

# plt.savefig(f'C:/Users/akuznets/Desktop/presa_buf/LWE/LWE_{rand_id}.png', dpi=300)

#%%
test_df = pd.DataFrame({
    'Photons (data)'  : psf_df['Nph WFS'].apply(lambda x: np.abs(x.real)),
    'Photons (fitted)': fitted_df['Nph WFS (new)'].apply(lambda x: np.abs(x.real)),
    'IDs'             : fitted_df.index.values
})
test_df['Ratio'] = test_df['Photons (data)'] / test_df['Photons (fitted)']
test_df.set_index('IDs', inplace=True)

#%%
test_df_1 = test_df[(  test_df['Ratio'] >= 1.1) | (test_df['Ratio'] <= 0.9)]
test_df_1 = test_df_1[(1/test_df_1['Ratio'] <  0.1)]
test_df_1 = test_df_1[ (test_df_1['Photons (data)']   >= 55) & (test_df_1['Photons (data)']   <= 65)]
test_df_1 = test_df_1[ (test_df_1['Photons (fitted)'] >= 5)  & (test_df_1['Photons (fitted)'] <= 7) ]

test_df_2 = test_df[(  test_df['Ratio'] >= 1.1) | (test_df['Ratio'] <= 0.9)]
test_df_2 = test_df_2[ (test_df_2['Photons (data)'] < 1000) ]
test_df_2 = test_df_2[ (test_df_2['Photons (fitted)'] > 200) ]
test_df_2 = test_df_2[ (test_df_2['Photons (fitted)'] < 500) ]

#%%
from tools.utils import draw_PSF_stack

ids_strange_low = [1265, 1418, 1425, 1914, 2857, 2859, 2873, 3661, 3722, 3732, 3733, 3734, 3979]

ids_strange_high = [114, 549, 811, 816, 1176, 1192, 1304, 1573, 2146, 2726, 3121,
                    3613, 3651, 3706, 3875, 3882, 3886, 3906, 3909, 4002, 405]

rand_id = np.random.choice(ids_strange_low)
rand_id = np.random.choice(ids_strange_high)

print(psf_df.loc[rand_id]['Nph WFS'].real)
print(fitted_df.loc[rand_id]['Nph WFS (new)'])

PSF_0 = torch.tensor(images_df[rand_id][0])
PSF_1 = torch.tensor(images_df[rand_id][1])

draw_PSF_stack(PSF_0, PSF_1, average=True, crop=80)

#%%
test_df_2 = test_df

# Do linear fit
X = test_df_2['Photons (data)'].values.reshape(-1, 1)
Y = test_df_2['Photons (fitted)'].values

lr = LinearRegression()
lr.fit(X, Y)
Y_pred = lr.predict(X)

print('Slope:', lr.coef_[0])

# draw linear fit
# plt.plot(X, Y_pred, color='red')
plt.plot([0, 10000], [0, 10000], color='red')
plt.plot([0, 10000], [0, 9000], color='red')
plt.plot([0, 10000], [0, 1000])
plt.plot([0, 10000], [0, 1800])
plt.scatter(X, Y, s=2)
plt.xlabel('Photons (data)')
plt.ylabel('Photons (fitted)')
plt.xlim(0, 2e2)
plt.ylim(0, 2e2)
plt.show()

#%%
sns.scatterplot(data=test_df_2, x='Photons (data)', y='Photons (fitted)', s=2)
# plt.xlim(0, 4e2)
# plt.ylim(0, 4e2)

#%%
sns.displot(test_df, x='Ratio', bins=1000, kde=True)
plt.xlim(0, 4)
