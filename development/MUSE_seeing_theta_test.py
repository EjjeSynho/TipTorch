#%%
try:
    ipy = get_ipython()        # NameError if not running under IPython
    if ipy:
        ipy.run_line_magic('reload_ext', 'autoreload')
        ipy.run_line_magic('autoreload', '2')
        import linecache
        ipy.events.register('post_execute', lambda: linecache.clearcache())
except NameError:
    pass

import sys, os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tiptorch._config import WEIGHTS_FOLDER, default_device, default_torch_type, project_settings

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pickle

from pathlib import Path


MUSE_DATA_FOLDER = Path(project_settings["MUSE_data_folder"])
STD_FOLDER       = Path(project_settings["MUSE_STD_data_folder"])
DATASET_CACHE    = STD_FOLDER / 'dataset_cache'

device = default_device

#%%
with open(STD_FOLDER / 'muse_df.pickle', 'rb') as handle:
    muse_df = pickle.load(handle)

    
with open(STD_FOLDER / 'muse_fitted_df.pickle', 'rb') as handle:
    muse_fitted_df = pickle.load(handle)['fitted_values']
    
#%%
muse_df = muse_df[muse_df['Corrupted']   == False]
muse_df = muse_df[muse_df['Bad quality'] == False]

# Intersect index between muse_df and muse_fitted_df
muse_df = muse_df[muse_df.index.isin(muse_fitted_df.index)]


#%% Create fitted parameters dataset
from tqdm import tqdm

experiment_folder = 'fitted'

fitted_samples_folder = STD_FOLDER / experiment_folder

files = os.listdir(fitted_samples_folder)

SR_data, PSDs_data = {}, {}

for file in tqdm(files):
    id = int(file.split('.')[0])

    with open(fitted_samples_folder / file, 'rb') as handle:
        data = pickle.load(handle)
    
    SR_data[id] = data['SR data'].copy()
    PSDs_data[id] = data['PSD'].copy()
    
    
# %%
import pandas as pd

SR_df = pd.DataFrame.from_dict({ k: np.median(v) for k, v in SR_data.items() }, orient='index')

SR_df.set_index(muse_df.index, inplace=True)
SR_df.sort_index(inplace=True)
SR_df.rename(columns={0: 'SR'}, inplace=True)

SR_df.head()

#%%
with open(STD_FOLDER / 'muse_STD_stars_telemetry.pickle', 'rb') as handle:
    reduced_telemetry = pickle.load(handle)['telemetry imputed df']

# Align index with muse_df
reduced_telemetry = reduced_telemetry[reduced_telemetry.index.isin(muse_df.index)]
# reduced_telemetry.sort_index(inplace=True)

SR_df = SR_df[SR_df.index.isin(reduced_telemetry.index)]


# %%
def compute_WFE(PSD_):
    PSD_AO = 0.0
    PSD_chrom = 0.0

    # for col in cols:
    #     PSD_AO += PSD_test[col]
    #     # print(col, PSD_test[col].shape)

    PSD_chrom += PSD_['chromatism']
    PSD_chrom += PSD_['diff. refract']

    PSD_AO += PSD_['WFS noise']
    PSD_AO += PSD_['spatio-temporal']

    PSD_chrom = PSD_chrom.squeeze()
    PSD_AO = PSD_AO.squeeze(1)
    PSD_AO = PSD_AO + PSD_chrom
    PSD_AO = PSD_AO.mean(axis=0)

    PSD_atm = PSD_['fitting'].squeeze()

    center_crop = np.s_[
        PSD_atm.shape[0]//2-PSD_AO.shape[0]//2 : PSD_atm.shape[0]//2+PSD_AO.shape[0]//2 + 1,
        -PSD_AO.shape[1]:
    ]

    PSD_total = PSD_atm.copy()
    PSD_total = PSD_total.real
    PSD_total[center_crop] = PSD_total[center_crop] + PSD_AO.real

    dk = 0.0623
    PSD_norm = (500.0 / (2.0 * torch.pi))**2  # [nm²/rad²]
    WFE_var  = 2.0 * (PSD_total * dk**2).sum() * PSD_norm  # [nm²]
    WFE_rms  = np.sqrt(WFE_var).item()                       # [nm rms]
    return WFE_rms

#%%
WFE_df = pd.DataFrame.from_dict({ k: compute_WFE(v) for k, v in PSDs_data.items() }, orient='index', columns=['WFE_rms'])
WFE_df.set_index(muse_df.index, inplace=True)
WFE_df.sort_index(inplace=True)
WFE_df.head()


#%%
combined_df = SR_df[['SR']].join(
    reduced_telemetry[['theta0', 'Seeing (header)']], how='inner'
).join(
    WFE_df[['WFE_rms']], how='inner'
).dropna()

combined_df['ratio'] = combined_df['Seeing (header)'] / combined_df['theta0']

#%%
import seaborn as sns

fig, ax = plt.subplots(figsize=(7, 5))
sns.kdeplot(data=combined_df, x='ratio', y='SR', fill=True, cmap='Blues', thresh=0.02, levels=12, ax=ax)
sns.scatterplot(data=combined_df, x='ratio', y='SR', alpha=0.3, s=15, color='steelblue', linewidth=0, ax=ax)
ax.set_xlabel('Seeing / Theta0')
ax.set_ylabel('SR')
plt.tight_layout()
plt.show()

#%%
fig, ax = plt.subplots(figsize=(7, 5))
sns.kdeplot(data=combined_df, x='ratio', y='WFE_rms', fill=True, cmap='Blues', thresh=0.02, levels=12, ax=ax)
sns.scatterplot(data=combined_df, x='ratio', y='WFE_rms', alpha=0.3, s=15, color='steelblue', linewidth=0, ax=ax)
ax.set_xlabel('Seeing / Theta0')
ax.set_ylabel('WFE_rms')
plt.tight_layout()
plt.show()

#%%
fig, ax = plt.subplots(figsize=(7, 5))
sns.kdeplot(data=combined_df, x='SR', y='WFE_rms', fill=True, cmap='Blues', thresh=0.02, levels=12, ax=ax)
sns.scatterplot(data=combined_df, x='SR', y='WFE_rms', alpha=0.3, s=15, color='steelblue', linewidth=0, ax=ax)
ax.set_xlabel('SR')
ax.set_ylabel('WFE_rms')
plt.tight_layout()
plt.show()

#%%
from scipy.stats import pearsonr, spearmanr, kendalltau

x, y = combined_df['ratio'], combined_df['SR']
r_pearson,  p_pearson  = pearsonr(x, y)
r_spearman, p_spearman = spearmanr(x, y)
r_kendall,  p_kendall  = kendalltau(x, y)

print(f"Pearson  r = {r_pearson: .4f}  (p = {p_pearson:.2e})")
print(f"Spearman r = {r_spearman:.4f}  (p = {p_spearman:.2e})")
print(f"Kendall  τ = {r_kendall: .4f}  (p = {p_kendall:.2e})")
