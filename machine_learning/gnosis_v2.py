#%%
%reload_ext autoreload
%autoreload 2

import sys
sys.path.append('..')

import torch
import pickle
from project_globals import SPHERE_DATA_FOLDER, DATA_FOLDER, device
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from tqdm import tqdm
import os

import torch
from torch.distributions.normal import Normal


#%% Initialize data sample
with open(SPHERE_DATA_FOLDER+'sphere_df.pickle', 'rb') as handle:
    psf_df = pickle.load(handle)

psf_df = psf_df[psf_df['invalid'] == False]
psf_df = psf_df[psf_df['LWE'] == False]
psf_df = psf_df[psf_df['doubles'] == False]
psf_df = psf_df[psf_df['No coronograph'] == False]
psf_df = psf_df[~pd.isnull(psf_df['r0 (SPARTA)'])]
psf_df = psf_df[~pd.isnull(psf_df['Nph WFS'])]
psf_df = psf_df[~pd.isnull(psf_df['Strehl'])]
psf_df = psf_df[~pd.isnull(psf_df['FWHM'])]
psf_df = psf_df[psf_df['Nph WFS'] < 5000]
# psf_df = psf_df[psf_df['λ left (nm)'] > 1600]
# psf_df = psf_df[psf_df['λ left (nm)'] < 1700]

ids_class_C = set(psf_df.index[psf_df['Class C'] == True])
ids_wvls = set(psf_df.index[psf_df['λ left (nm)'] > 1600]).intersection(set(psf_df.index[psf_df['λ left (nm)'] < 1700]))

ids_to_exclude_later = ids_class_C.union(ids_wvls)

#%%
# psf_df2 = psf_df[(pd.isnull(psf_df['mag J'])) |  (pd.isnull(psf_df['mag H']))]
# print(len(psf_df2))
good_ids = psf_df.index.values.tolist()
print(len(good_ids), 'samples are in the dataset')

#% Select the entries to be used in training
selected_entries = ['Airmass',
                    'r0 (SPARTA)',
                    'FWHM',
                    'Strehl',
                    'Wind direction (header)',
                    'Wind speed (header)',
                    'Tau0 (header)',
                    'Nph WFS',
                    'Rate',
                    'λ left (nm)',
                    'λ right (nm)']

psf_df = psf_df[selected_entries]

#%%
class DataTransformer:
    def __init__(self, data, boxcox=True, gaussian=True, uniform=False, invert=False) -> None:
        self.boxcox_flag   = boxcox
        self.gaussian_flag = gaussian
        self.uniform_flag  = uniform
        self.invert_flag   = invert
        
        self.std_scaler     = lambda x, m, s: (x-m)/s
        self.inv_std_scaler = lambda x, m, s: x*s+m

        self.uniform_scaler     = lambda x, a, b: (x-a)/(b-a)
        self.inv_uniform_scaler = lambda x, a, b: x*(b-a)+a
                        
        self.one_minus     = lambda x: 1-x if self.invert_flag else x
        self.inv_one_minus = lambda x: 1-x if self.invert_flag else x
        
        self.normal_distribution = Normal(torch.tensor([0.0]).to(device), torch.tensor([1.0]).to(device))

        if data is not None: self.fit(data)
    
    def boxcox(self, x, λ):
        if isinstance(x, torch.Tensor):
            return torch.log(x) if λ == 0 else (x**λ-1)/λ
        else:
            return np.log(x) if λ == 0 else (x**λ-1)/λ
    
    def inv_boxcox(self, y, λ):
        if isinstance(y, torch.Tensor) :
            return torch.exp(y) if λ == 0 else (λ*y+1)**(1/λ)
        else:
            return np.exp(y) if λ == 0 else (λ*y+1)**(1/λ)

    def cdf(self, x):
        return self.normal_distribution.cdf(x) if isinstance(x, torch.Tensor) else  stats.norm.cdf(x)
        
    def ppf(self, q):
        return torch.sqrt(torch.tensor([2.]).to(device)) * torch.erfinv(2.*q - 1.) if isinstance(q, torch.Tensor) else stats.norm.ppf(q)
    
    def fit(self, data):
        if self.boxcox_flag and self.gaussian_flag and not self.uniform_flag:
            data_standartized, self.lmbd = stats.boxcox(self.one_minus(data))
            self.mu, self.std = stats.norm.fit(data_standartized)
            
        elif self.boxcox_flag and not self.gaussian_flag and self.uniform_flag:
            data_standartized, self.lmbd = stats.boxcox(self.one_minus(data))
            self.a = data_standartized.min()*0.99
            self.b = data_standartized.max()*1.01
        
        elif not self.boxcox_flag and self.gaussian_flag and not self.uniform_flag:
            self.mu, self.std = stats.norm.fit(self.one_minus(data))

        elif not self.boxcox_flag and not self.gaussian_flag and self.uniform_flag:
            self.a = data.min()
            self.b = data.max()

    def forward(self, x):
        if self.boxcox_flag and self.gaussian_flag and not self.uniform_flag:
            return self.std_scaler(self.boxcox(self.one_minus(x), self.lmbd), self.mu, self.std)
        
        elif self.boxcox_flag and not self.gaussian_flag and self.uniform_flag:
            return self.ppf( self.uniform_scaler(self.boxcox(self.one_minus(x), self.lmbd), self.a, self.b) )
        
        elif not self.boxcox_flag and self.gaussian_flag and not self.uniform_flag:
            return self.std_scaler(self.one_minus(x), self.mu, self.std)
        
        elif not self.boxcox_flag and not self.gaussian_flag and self.uniform_flag:
            return self.uniform_scaler(self.one_minus(x), self.a, self.b)*2-1

    def backward(self, x):
        if self.boxcox_flag and self.gaussian_flag and not self.uniform_flag:
            return self.inv_one_minus(self.inv_boxcox(self.inv_std_scaler(x, self.mu, self.std), self.lmbd))
        
        elif self.boxcox_flag and not self.gaussian_flag and self.uniform_flag:
            return self.inv_one_minus(self.inv_boxcox(self.inv_uniform_scaler(self.cdf(x), self.a, self.b), self.lmbd))
        
        elif not self.boxcox_flag and self.gaussian_flag and not self.uniform_flag:
            return self.inv_one_minus(self.inv_std_scaler(x, self.mu, self.std))
        
        elif not self.boxcox_flag and not self.gaussian_flag and self.uniform_flag:
            return (self.inv_one_minus(self.inv_uniform_scaler((x+1)/2, self.a, self.b)) )



#%% Create fitted parameters dataset
fitted_dict_raw = {key: [] for key in ['F','b','dx','dy','r0','amp','beta','alpha','theta','ratio','bg','Jx','Jy','Jxy']}
ids = []

images_data = []
images_fitted = []

fitted_folder = 'E:/ESO/Data/SPHERE/IRDIS_fitted_PAO_1P21I/'
fitted_files = os.listdir(fitted_folder)

for file in tqdm(fitted_files):
    id = int(file.split('.')[0])

    with open(fitted_folder + file, 'rb') as handle:
        data = pickle.load(handle)
        
    images_data.append( data['Img. data'] )
    images_fitted.append( data['Img. fit'] )

    for key in fitted_dict_raw.keys():
        fitted_dict_raw[key].append(data[key])
    ids.append(id)
    
fitted_dict = {}
fitted_dict['ID'] = np.array(ids)

for key in fitted_dict_raw.keys():
    fitted_dict[key] = np.squeeze(np.array(fitted_dict_raw[key]))

fitted_dict['F (left)'  ] = fitted_dict['F'][:,0]
fitted_dict['F (right)' ] = fitted_dict['F'][:,1]
fitted_dict['bg (left)' ] = fitted_dict['bg'][:,0]
fitted_dict['bg (right)'] = fitted_dict['bg'][:,1]
fitted_dict.pop('F')
fitted_dict.pop('bg')

for key in fitted_dict.keys():
    fitted_dict[key] = fitted_dict[key].tolist()

images_dict = {
    'ID': ids,
    'PSF (data)': images_data,
    'PSF (fit)': images_fitted
}

#%%
fitted_df = pd.DataFrame(fitted_dict)
fitted_df.set_index('ID', inplace=True)

del fitted_dict, fitted_dict_raw, ids, images_data, images_fitted, fitted_folder, fitted_files, data

#%
fitted_ids = list( set( fitted_df.index.values.tolist() ).intersection( set(psf_df.index.values.tolist()) ) )
fitted_df = fitted_df[fitted_df.index.isin(fitted_ids)]

# Absolutize positive-only values
for entry in ['b','r0','Jx','Jy','Jxy','ratio','alpha','F (left)','F (right)']:
    fitted_df[entry] = fitted_df[entry].abs()

# threshold = 3 # in STDs
# bad_alpha = np.where(np.abs(fitted_df['alpha']) > threshold * fitted_df['alpha'].std())[0].tolist()
# bad_beta  = np.where(np.abs(fitted_df['beta'])  > threshold * fitted_df['beta'].std())[0].tolist()
# bad_amp   = np.where(np.abs(fitted_df['amp'])   > threshold * fitted_df['amp'].std())[0].tolist()
# bad_ratio = np.where(np.abs(fitted_df['ratio']) > 2)[0].tolist()
# bad_theta = np.where(np.abs(fitted_df['theta']) > 3)[0].tolist()

# exclude_ids = set(bad_alpha + bad_beta + bad_amp + bad_ratio + bad_theta)
# exlude_samples = [fitted_df.iloc[id].name for id in exclude_ids]

# for id in exlude_samples:
#     fitted_df = fitted_df.drop(id)

# psf_df = psf_df[psf_df.index.isin(fitted_df.index.values.tolist())]

#%% Create Input parameters dataset
fl_in = {                      # boxcox gaussian uniform invert
    'Airmass':                 [True,  False, True,  False],
    'r0 (SPARTA)':             [True,  True,  False, False],
    'FWHM':                    [True,  True,  False, False],
    'Strehl':                  [True,  True,  False, True ],
    'Wind speed (header)':     [True,  True,  False, False],
    'Tau0 (header)':           [True,  True,  False, False],
    'Rate':                    [False, False, True,  False],
    'λ left (nm)':             [False, False, True,  False],
    # 'λ right (nm)':            [False, False, True,  False],
    'Wind direction (header)': [False, False, True,  False],
    'Nph WFS':                 [True,  True,  False, False]
}

transforms_input = {\
    i: DataTransformer(psf_df[i].values, boxcox=fl_in[i][0], gaussian=fl_in[i][1], uniform=fl_in[i][2], invert=fl_in[i][3]) for i in fl_in.keys() }

input_df = pd.DataFrame( {a: transforms_input[a].forward(psf_df[a].values) for a in fl_in.keys()} )

#%%
fl_out = {          # boxcox gaussian uniform invert
    'F (left)':   [True,  True,  False, False],
    # 'F (right)':  [True,  True,  False, False],
    'bg (left)':  [False, True,  False, False],
    # 'bg (right)': [False, True,  False, False],
    'b':          [True,  True,  False, False],
    'dx':         [False, False, True,  False],
    'dy':         [False, False, True,  False],
    'r0':         [True,  False, True,  False],
    'amp':        [True,  True,  False, False],
    'beta':       [True,  True,  False, False],
    'alpha':      [True,  False, True,  False],
    'theta':      [False, False, True,  False],
    'ratio':      [False, False, True,  False],
    'Jx':         [True,  True,  False, False],
    'Jy':         [True,  True,  False, False],
    'Jxy':        [True,  True,  False, False]
}

transforms_output = {\
    i: DataTransformer(fitted_df[i].values, boxcox=fl_out[i][0], gaussian=fl_out[i][1], uniform=fl_out[i][2], invert=fl_out[i][3]) for i in fl_out.keys() }

output_df = pd.DataFrame( {a: transforms_output[a].forward(fitted_df[a].values) for a in fl_out.keys()} )

rows_with_nan = output_df.index[output_df.isna().any(axis=1)].values.tolist()

# for id in rows_with_nan:
#     fitted_df = fitted_df.drop(id)

#%%
psf_df.drop(rows_with_nan, inplace=True)
input_df.drop(rows_with_nan, inplace=True)
fitted_df.drop(rows_with_nan, inplace=True)
output_df.drop(rows_with_nan, inplace=True)

print('Number of samples: {}'.format(len(psf_df)))

#%%
save_path = DATA_FOLDER/"temp/SPHERE params/psf_df"
for entry in selected_entries:
    sns.displot(data=psf_df, x=entry, kde=True, bins=20)
    plt.savefig(save_path/f"{entry}.png")
    plt.close()

save_path = DATA_FOLDER/"temp/SPHERE params/fitted_df"
for entry in fitted_df.columns.values.tolist():
    sns.displot(data=fitted_df, x=entry, kde=True, bins=20)
    plt.savefig(save_path/f"{entry}.png")
    plt.close()

save_path = DATA_FOLDER/"temp/SPHERE params/input_df"
for entry in fl_in.keys():
    sns.displot(data=input_df, x=entry, kde=True, bins=20)
    plt.savefig(save_path/f"{entry}.png")
    plt.close()

save_path = DATA_FOLDER/"temp/SPHERE params/output_df"
for entry in output_df.columns.values.tolist():
    sns.displot(data=output_df, x=entry, kde=True, bins=20)
    plt.savefig(save_path/f"{entry}.png")
    plt.close()

inp_inv_df = pd.DataFrame( {a: transforms_input[a].backward(input_df[a].values) for a in fl_in.keys()} )
save_path = DATA_FOLDER/"temp/SPHERE params/inp_inv_df"
for entry in fl_in.keys():
    sns.displot(data=inp_inv_df, x=entry, kde=True, bins=20)
    plt.savefig(save_path/f"{entry}.png")
    plt.close()

out_inv_df = pd.DataFrame( {a: transforms_output[a].backward(output_df[a].values) for a in fl_out.keys()} )
save_path = DATA_FOLDER/"temp/SPHERE params/out_inv_df"
for entry in output_df.columns.values.tolist():
    sns.displot(data=out_inv_df, x=entry, kde=True, bins=20)
    plt.savefig(save_path/f"{entry}.png")
    plt.close()

#%%
# 'Airmass'
# 'r0 (SPARTA)'
# 'FWHM'
# 'Strehl'
# 'Wind direction (header)'
# 'Wind speed (header)'
# 'Tau0 (header)'
# 'Nph WFS'
# 'Rate'
# 'λ left (nm)'
# 'λ right (nm)'

# 'b'
# 'dx'
# 'dy'
# 'r0'
# 'amp'
# 'beta'
# 'alpha'
# 'theta'
# 'ratio'
# 'Jx'
# 'Jy'
# 'Jxy'
# 'F (left)'
# 'F (right)'
# 'bg (left)'
# 'bg (right)'

# df_ultimate = pd.concat([input_df, output_df], axis=1)
# df_ultimate = pd.concat([psf_df, fitted_df], axis=1)
df_ultimate = pd.concat([psf_df], axis=1)

columns_to_drop = ['F (right)', 'bg (right)', 'λ right (nm)', 'Strehl']

for column_to_drop in columns_to_drop:
    if column_to_drop in df_ultimate.columns:
        df_ultimate = df_ultimate.drop(column_to_drop, axis=1)

df_ultimate = df_ultimate.sort_index(axis=1)

# corr_method = 'pearson'
# corr_method = 'spearman'
corr_method = 'kendall'

spearman_corr = df_ultimate.corr(method=corr_method)

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(spearman_corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))
ax.set_title("Correlation matrix ({})".format(corr_method))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(spearman_corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.xticks(rotation=45, ha='right')
# plt.yticks(rotation=45)
plt.show()
#%%

# sns.kdeplot(data = input_df,
#             x='Rate',
#             y='Nph WFS',
#             color='r', fill=True, Label='Iris_Setosa',
#             cmap="Reds", thresh=0.02)

# sns.kdeplot(data = input_df,
#             x='r0 (SPARTA)',
#             y='Tau0 (header)',
#             color='r', fill=True, Label='Iris_Setosa',
#             cmap="Reds", thresh=0.02)

sns.pairplot(data=psf_df, vars=[
    'Nph WFS',
    'r0 (SPARTA)',
    'Tau0 (header)',
    'Wind speed (header)',
    'FWHM'], corner=True, kind='kde')
    # 'FWHM'], hue='λ left (nm)', kind='kde')


#%%
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

from project_globals import device

psf_df.sort_index(inplace=True)
fitted_df.sort_index(inplace=True)

assert psf_df.index.equals(fitted_df.index)

psf_df_train,    psf_df_valid    = train_test_split(psf_df,    test_size=0.2, random_state=42)
fitted_df_train, fitted_df_valid = train_test_split(fitted_df, test_size=0.2, random_state=42)

psf_df_train.sort_index(inplace=True)
psf_df_valid.sort_index(inplace=True)

X_train_df = pd.DataFrame( {a: transforms_input[a].forward(psf_df_train[a].values) for a in fl_in.keys()} )
X_valid_df = pd.DataFrame( {a: transforms_input[a].forward(psf_df_valid[a].values) for a in fl_in.keys()} )
X_train_df.index = psf_df_train.index
X_valid_df.index = psf_df_valid.index

Y_train_df = pd.DataFrame( {a: transforms_output[a].forward(fitted_df_train[a].values) for a in fl_out.keys()} )
Y_valid_df = pd.DataFrame( {a: transforms_output[a].forward(fitted_df_valid[a].values) for a in fl_out.keys()} )
Y_train_df.index = psf_df_train.index
Y_valid_df.index = psf_df_valid.index

rows_with_nan = Y_train_df.index[Y_train_df.isna().any(axis=1)].values.tolist()
Y_train_df = Y_train_df.drop(rows_with_nan, axis=0)
X_train_df = X_train_df.drop(rows_with_nan, axis=0)

assert Y_train_df.index.values.tolist() == X_train_df.index.values.tolist()

X_train = torch.from_numpy(X_train_df.values).float().to(device)
Y_train = torch.from_numpy(Y_train_df.values).float().to(device)

X_val   = torch.from_numpy(X_valid_df.values).float().to(device)
Y_val   = torch.from_numpy(Y_valid_df.values).float().to(device)

#%%
# Define the network architecture
class Gnosis2(nn.Module):
    def __init__(self, in_size, out_size, hidden_size=100, dropout_p=0.5):
        super(Gnosis2, self).__init__()
        self.fc1 = nn.Linear(in_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(dropout_p)
        self.fc3 = nn.Linear(hidden_size, hidden_size*2)
        self.dropout3 = nn.Dropout(dropout_p)
        self.fc4 = nn.Linear(hidden_size*2, out_size)
        

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.dropout1(x)
        x = torch.tanh(self.fc2(x))
        x = self.dropout2(x)
        x = torch.tanh(self.fc3(x))
        x = self.dropout3(x)
        x = torch.tanh(self.fc4(x))
        return x
    
# Initialize the network, loss function and optimizer
net = Gnosis2(X_train.shape[1], Y_train.shape[1], 200, 0.25)
net.to(device)

# loss_fn = nn.MSELoss(reduction='mean')
loss_fn = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)


#%%
from project_globals import WEIGHTS_FOLDER
import pathlib
# Assume we have some input data in a pandas DataFrame
# inputs and targets should be your actual data
loss_trains = []
loss_vals = []

# Training loop
num_iters = 50

for epoch in range(num_iters):  # number of epochs
    optimizer.zero_grad()   # zero the gradient buffers
    y_pred = net(X_train)   # forward pass
    loss = loss_fn(y_pred, Y_train)  # compute loss
    loss_trains.append(loss.detach().cpu().item())
    loss.backward()  # backpropagation
    optimizer.step()  # update weights
    print('Epoch [%d/%d], Loss: %.4f' % (epoch+1, num_iters, loss.item()))

    with torch.no_grad():
        y_pred_val = net(X_val)
        loss_val = loss_fn(y_pred_val, Y_val)
        loss_vals.append(loss_val.detach().cpu().item())
        # print('Epoch [%d/%d], Val Loss: %.4f' % (epoch+1, 100, loss_val.item()))

loss_trains = np.array(loss_trains)
loss_vals = np.array(loss_vals)

# %
plt.plot(loss_trains, label='Train')
plt.plot(loss_vals, label='Val')
plt.legend()
plt.grid()

torch.save(net.state_dict(), WEIGHTS_FOLDER/'gnosis2_weights.dict')

#%%
from tools.utils import plot_radial_profiles #, draw_PSF_stack
from data_processing.SPHERE_preproc_utils import SPHERE_preprocess
from PSF_models.TipToy_SPHERE_multisrc import TipToy

norm_regime = 'sum'

fl_out_keys = list(fl_out.keys())
fl_in_keys  = list(fl_in.keys())

gnosis2PAO = lambda Y: { fl_out_keys[i]: transforms_output[fl_out_keys[i]].backward(Y[:,i]) for i in range(len(fl_out_keys)) }
gnosis2in  = lambda X: { fl_in_keys[i]:  transforms_input[fl_in_keys[i]].backward(X[:,i])   for i in range(len(fl_in_keys))  }
in2gnosis  = lambda inp: torch.from_numpy(( np.stack([transforms_input[a].forward(inp[a].values)  for a in fl_in_keys]) )).float().to(device).T
PAO2gnosis = lambda out: torch.from_numpy(( np.stack([transforms_output[a].forward(out[a].values) for a in fl_out_keys]) )).float().to(device).T

# psf_df --> in --> gnosis --> out --> PAO --> image

def GetImages(sample_ids, norms):
    PSF_0_ = []
    PSF_1_ = []

    for i,id_im in enumerate([images_dict['ID'].index(sample_id) for sample_id in sample_ids]):
        PSF_0_.append(images_dict['PSF (data)'][id_im][0,...] / norms[i,:][:,None, None].cpu().numpy())
        PSF_1_.append(images_dict['PSF (fit)' ][id_im][0,...] / norms[i,:][:,None, None].cpu().numpy())
        
    PSF_0_ = torch.tensor(np.stack(PSF_0_)).float()
    PSF_1_ = torch.tensor(np.stack(PSF_1_)).float()
    return PSF_0_, PSF_1_


def GetBatch(rand_ids, read_mode):
    if read_mode == 'valid':
        sample_ids = [psf_df_valid.index.values.tolist()[rand_id] for rand_id in rand_ids]
    elif read_mode == 'train':
        sample_ids = [psf_df_train.index.values.tolist()[rand_id] for rand_id in rand_ids]
    else:
        raise ValueError('Mode must be either "valid" or "train"')
    
    # PSF_0, bg, norms, data_samples, merged_config = SPHERE_preprocess(sample_ids, 'different', norm_regime)
    _, _, norms, _, merged_config = SPHERE_preprocess(sample_ids, 'different', norm_regime)
    PSF_0_, PSF_1_ = GetImages(sample_ids, norms)

    batch = {
        'ids': sample_ids,
        'confo': merged_config,
        'PSF (data)': PSF_0_,
        'PSF (fit)':  PSF_1_,
        'X': in2gnosis ( psf_df_valid.loc[sample_ids] )    if read_mode == 'valid' else in2gnosis ( psf_df_train.loc[sample_ids]    ),
        'Y': PAO2gnosis( fitted_df_valid.loc[sample_ids] ) if read_mode == 'valid' else PAO2gnosis( fitted_df_train.loc[sample_ids] )
    }
    return batch


def GetRandIDs(length):
    rand_ids = np.arange(length)
    np.random.shuffle(rand_ids)
    rand_ids = torch.split(torch.from_numpy(rand_ids), 64)
    return rand_ids


batches_train = [GetBatch(rand_id, 'train') for rand_id in tqdm(GetRandIDs(len(X_train_df)))]
batches_val   = [GetBatch(rand_id, 'valid') for rand_id in tqdm(GetRandIDs(len(X_valid_df)))]

def toy_run(model, dictionary):
        model.F     = dictionary['F (left)'].repeat(2,1).T
        model.bg    = dictionary['bg (left)'].repeat(2,1).T
        model.Jy    = dictionary['Jy']
        model.Jxy   = dictionary['Jxy']
        model.Jx    = dictionary['Jx']
        model.dx    = dictionary['dx']
        model.dy    = dictionary['dy']
        model.b     = dictionary['b']
        model.r0    = dictionary['r0']
        model.amp   = dictionary['amp']
        model.beta  = dictionary['beta']
        model.theta = dictionary['theta']
        model.alpha = dictionary['alpha']
        model.ratio = dictionary['ratio']
        
        return model.forward()

#%%
# Load weights
net.load_state_dict(torch.load(WEIGHTS_FOLDER/'gnosis2_weights.dict'))

#%%
optimizer = optim.Adam(net.parameters(), lr=0.001)
# optimizer = optim.LBFGS(net.parameters(), lr=1, history_size=20, max_iter=4, line_search_fn="strong_wolfe")

loss_PSF = nn.L1Loss(reduction='sum')

def loss_fn(output, target):
    return loss_PSF(output, target) #+ torch.amax(torch.abs(output-target), dim=(-2,-1)).sum() * 1e1
    # return torch.amax(torch.abs(output-target), dim=(-2,-1)).mean() * 1e2


epochs = 26
torch.cuda.empty_cache()

for epoch in range(epochs):
    loss_train_average = []
    
    for batch in batches_train:
        optimizer.zero_grad()
        
        toy = TipToy(batch['confo'], norm_regime, device, TipTop=False, PSFAO=True)
        toy.optimizables = []
        
        PSF_pred = toy_run(toy, gnosis2PAO(net(batch['X'])))
        
        loss = loss_fn(PSF_pred, batch['PSF (data)'].to(device))
        loss.backward()
        loss_train_average.append(loss.item())
        
        # optimizer.step(lambda: loss_fn(toy_run(toy, gnosis2PAO(net(batch['X']))), batch['PSF (data)'].to(device)))
        optimizer.step()
        torch.cuda.empty_cache()
        
    with torch.no_grad():
        loss_valid_average = []

        for batch in batches_val:
                
            toy = TipToy(batch['confo'], norm_regime, device, TipTop=False, PSFAO=True)
            toy.optimizables = []
            
            PSF_pred = toy_run(toy, gnosis2PAO(net(batch['X'])))
            
            loss = loss_fn(PSF_pred, batch['PSF (data)'].to(device))
            loss_valid_average.append(loss.item())
            torch.cuda.empty_cache()
            
    print('Epoch %d/%d: ' % (epoch+1, epochs))
    print('  Train loss:  %.4f' % (np.array(loss_train_average).mean()))
    print('  Valid. loss: %.4f' % (np.array(loss_valid_average).mean()))
    print('')

    #TODO: TipTorch config update
    
torch.save(net.state_dict(), WEIGHTS_FOLDER/'gnosis2_weights_tuned.dict')

#%%

# net.load_state_dict(torch.load(WEIGHTS_FOLDER/'gnosis2_weights.dict'))

net.load_state_dict(torch.load(WEIGHTS_FOLDER/'gnosis2_weights_tuned.dict'))

#%%
with torch.no_grad():
    batch = batches_val[2] #np.random.choice(len(batches_val))]   
    pred_test = gnosis2PAO(net(batch['X']))

    toy = TipToy(batch['confo'], norm_regime, device, TipTop=False, PSFAO=True)
    toy.optimizables = []

    PSF_pred = toy_run(toy, pred_test).detach().cpu().numpy()
    PSF_test = batch['PSF (data)'].detach().cpu().numpy()
    
# rand_psf = np.random.choice(PSF_pred.shape2[0])
# PSF_pred = PSF_pred[rand_psf,...][None,...]
# PSF_test = PSF_test[rand_psf,...][None,...]
    
destack = lambda PSF_stack: [ x for x in np.split(PSF_stack[:,0,...], PSF_stack.shape[0], axis=0) ]

PSF_test_ = np.copy( PSF_test / PSF_test.max(axis=(-2,-1))[..., None, None] )
PSF_pred_ = np.copy( PSF_pred / PSF_pred.max(axis=(-2,-1))[..., None, None] )

plot_radial_profiles(destack(PSF_test_), destack(PSF_pred_), 'Data', 'Predicted', title='NN PSFAO prediction', dpi=200, cutoff=32, scale='log')
plot_radial_profiles(destack(PSF_test),  destack(PSF_pred),  'Data', 'Predicted', title='NN PSFAO prediction', dpi=200, cutoff=32, scale='log')


#%%
batch = batches_val[np.random.choice(len(batches_val))] 

toy = TipToy(batch['confo'], norm_regime, device, TipTop=False, PSFAO=True)
toy.optimizables = []


PSF_pred = toy_run(toy, gnosis2PAO(batch['Y'])).detach().cpu().numpy()  
PSF_test = batch['PSF (fit)'].detach().cpu().numpy()    
    
# rand_psf = np.random.choice(PSF_pred.shape[0])
# PSF_pred = PSF_pred[rand_psf,...][None,...]
# PSF_test = PSF_test[rand_psf,...][None,...]
    
destack = lambda PSF_stack: [ x for x in np.split(PSF_stack[:,0,...], PSF_stack.shape[0], axis=0) ]

PSF_test_ = np.copy( PSF_test / PSF_test.max(axis=(-2,-1))[..., None, None] )
PSF_pred_ = np.copy( PSF_pred / PSF_pred.max(axis=(-2,-1))[..., None, None] )

plot_radial_profiles(destack(PSF_test_), destack(PSF_pred_), 'From dataset', 'Converted', title='Fitted', dpi=200, cutoff=64)
plot_radial_profiles(destack(PSF_test),  destack(PSF_pred),  'From dataset', 'Converted', title='Fitted', dpi=200, cutoff=64)

#%% ================================================================================================================
%reload_ext autoreload
%autoreload 2

import sys
sys.path.append('..')

from tools.config_manager import ConfigManager, GetSPHEREonsky
from PSF_models.TipToy_SPHERE_multisrc import TipToy
import matplotlib.pyplot as plt
from torch.nn.functional import interpolate

from data_processing.SPHERE_preproc_utils import SPHERE_preprocess
from tools.utils import draw_PSF_stack, plot_radial_profiles
from project_globals import device
import pickle
import torch
from pprint import pprint

norm_regime = 'sum'

fitted_folder = 'E:/ESO/Data/SPHERE/IRDIS_fitted_PAO_1P21I/'
fitted_files = os.listdir(fitted_folder)

# selected_files = [fitted_files[20], fitted_files[30], fitted_files[50], fitted_files[40] ]
selected_files = [fitted_files[20], fitted_files[30]] #, fitted_files[40] ]
# selected_files = [ fitted_files[20] ]
sample_ids = [ int(file.split('.')[0]) for file in selected_files ]

regime = 'different'

psdsaves = ['PSD_1', 'PSD_2', 'PSD_double']

#%%============================================================================
PSFs_test_1 = []
PSFs_pred_1 = []
tensy = lambda x: torch.tensor(x).to(device)

c = 0

for file, sample in zip(selected_files,sample_ids):

    PSF_0, bg, norms, data_samples, init_config = SPHERE_preprocess([sample], regime, norm_regime)
    norms = norms[:, None, None].cpu().numpy() 

    with open(fitted_folder + file, 'rb') as handle:
        data = pickle.load(handle)

    toy = TipToy(init_config, norm_regime, device, TipTop=False, PSFAO=True)
    toy.optimizables = []

    toy.F     = tensy( data['F']     ).squeeze()
    toy.bg    = tensy( data['bg']    ).squeeze()
    toy.Jy    = tensy( data['Jy']    ).flatten()
    toy.Jxy   = tensy( data['Jxy']   ).flatten()
    toy.Jx    = tensy( data['Jx']    ).flatten()
    toy.dx    = tensy( data['dx']    ).flatten()
    toy.dy    = tensy( data['dy']    ).flatten()
    toy.b     = tensy( data['b']     ).flatten()
    toy.r0    = tensy( data['r0']    ).flatten()
    toy.amp   = tensy( data['amp']   ).flatten()
    toy.beta  = tensy( data['beta']  ).flatten()
    toy.theta = tensy( data['theta'] ).flatten()
    toy.alpha = tensy( data['alpha'] ).flatten()
    toy.ratio = tensy( data['ratio'] ).flatten()

    toy.psdsave = psdsaves[c]
    c += 1

    PSFs_pred_1.append( toy().detach().clone()[0,...] )
    PSFs_test_1.append( torch.tensor( data['Img. fit'][0,...] / norms).to(device) )

PSFs_test_1 = torch.stack(PSFs_test_1)
PSFs_pred_1 = torch.stack(PSFs_pred_1)

# draw_PSF_stack(PSFs_test_1, PSFs_pred_1)

#%%============================================================================
PSF_0, bg, norms, data_samples, init_config = SPHERE_preprocess(sample_ids, regime, norm_regime)

if len(sample_ids) == 1:
    norms = norms[None,...]

datas = []
for file in selected_files:
    with open(fitted_folder + file, 'rb') as handle:
        datas.append( pickle.load(handle) )

toy = TipToy(init_config, norm_regime, device, TipTop=False, PSFAO=True)
toy.optimizables = []

tensy = lambda x: torch.tensor(x).to(device)

toy.F     = tensy( [data['F']     for data in datas] ).squeeze()
toy.bg    = tensy( [data['bg']    for data in datas] ).squeeze()
toy.Jy    = tensy( [data['Jy']    for data in datas] ).flatten()
toy.Jxy   = tensy( [data['Jxy']   for data in datas] ).flatten()
toy.Jx    = tensy( [data['Jx']    for data in datas] ).flatten()
toy.dx    = tensy( [data['dx']    for data in datas] ).flatten()
toy.dy    = tensy( [data['dy']    for data in datas] ).flatten()
toy.b     = tensy( [data['b']     for data in datas] ).flatten()
toy.r0    = tensy( [data['r0']    for data in datas] ).flatten()
toy.amp   = tensy( [data['amp']   for data in datas] ).flatten()
toy.beta  = tensy( [data['beta']  for data in datas] ).flatten()
toy.theta = tensy( [data['theta'] for data in datas] ).flatten()
toy.alpha = tensy( [data['alpha'] for data in datas] ).flatten()
toy.ratio = tensy( [data['ratio'] for data in datas] ).flatten()


PSFs_test = []
for i in range(len(datas)):
    PSFs_test.append( torch.tensor( datas[i]['Img. fit'][0,...] / norms[i,:,None,None].cpu().numpy() ).to(device) )
PSFs_test = torch.stack(PSFs_test)


if len(sample_ids) != 1:
    PSFs_test = PSFs_test.squeeze()

toy.psdsave = psdsaves[-1]
PSF_pred = toy()
    
# draw_PSF_stack(PSFs_test, PSF_pred)
draw_PSF_stack(PSFs_pred_1, PSF_pred, scale='log', min_val=1e-7, max_val=1e16)

plot_radial_profiles(PSFs_pred_1[0,0,...].unsqueeze(0), PSF_pred[0,0,...].unsqueeze(0), 'L_1', 'L', title='PSFs', dpi=200, cutoff=14, scale='linear')


#%%
# from tools.utils import register_hooks
# Q = loss_fn(PSF_pred, torch.ones_like(PSF_pred).to(device))
# get_dot = register_hooks(Q)
# Q.backward()
# dot = get_dot()
# #dot.save('tmp.dot') # to get .dot
# #dot.render('tmp') # to get SVG
# dot # in Jupyter, you can just render the variable

#%% ============================================================================
# ==============================================================================
# ==============================================================================

reado = 'C:/Users/akuznets/Projects/TipToy/data/temp/'

with open(reado + 'PSD_double.pickle', 'rb') as handle:
    PSD_double = pickle.load(handle)
with open(reado + 'PSD_1.pickle', 'rb') as handle:
    PSD_1 = pickle.load(handle)
with open(reado + 'PSD_2.pickle', 'rb') as handle:
    PSD_2 = pickle.load(handle)

#%%
psd_1 = PSD_1['MoffatPSD'].squeeze()
psd_2 = PSD_2['MoffatPSD'].squeeze()
psd_3 = PSD_double['MoffatPSD'].squeeze()

test_1 = psd_1.squeeze()
test_2 = psd_3[0,...]
# plt.imshow(torch.log10((test_1-test_2*0.01).abs()).cpu().numpy())
# test_1 = psd_2.squeeze()
# test_2 = psd_3[1,...]

test_1_ = interpolate(test_1.unsqueeze(0).unsqueeze(0), size=(test_2.shape[-2], test_2.shape[-1]), mode='bicubic').squeeze() #* (test_1.shape[0] / test_2.shape[0])**2 
test_2_ = interpolate(test_2.unsqueeze(0).unsqueeze(0), size=(test_1.shape[-2], test_1.shape[-1]), mode='bicubic').squeeze() #* (test_2.shape[0] / test_1.shape[0])**2

# el_croppo = slice(test_2.shape[0]//2-test_1.shape[0]//2, test_2.shape[0]//2+test_1.shape[0]//2)
el_croppo = slice(test_1.shape[0]//2-test_2.shape[0]//2, test_1.shape[0]//2+test_2.shape[0]//2)
el_croppo = (el_croppo, el_croppo)

test_1 = test_1[el_croppo]

# plot_radial_profiles(test_1_.unsqueeze(0), test_2.unsqueeze(0), 'Interp', 'Supersamp', title='Moffats', dpi=200, cutoff=14, scale='linear')
# plot_radial_profiles(test_1.unsqueeze(0), test_2_.unsqueeze(0), 'Supersamp', 'Interp', title='Moffats', dpi=200, cutoff=14, scale='linear')
plot_radial_profiles(test_1.unsqueeze(0), test_2.unsqueeze(0), 'Supersamp', 'Interp', title='Moffats', dpi=200, cutoff=14, scale='linear')

#%%
%matplotlib qt
test = torch.hstack([test_2, test_1,(test_1-test_2).abs()])

plt.imshow(torch.log10(test.abs()).cpu().numpy())
plt.colorbar()
plt.show()

# %matplotlib inline

#%% ============================================================================
# ==============================================================================
# ==============================================================================



%reload_ext autoreload
%autoreload 2

import sys
sys.path.append('..')

from tools.config_manager import ConfigManager, GetSPHEREonsky
from PSF_models.TipToy_SPHERE_multisrc import TipToy
import matplotlib.pyplot as plt
from torch.nn.functional import interpolate

from data_processing.SPHERE_preproc_utils import SPHERE_preprocess
from tools.utils import draw_PSF_stack, plot_radial_profiles
from project_globals import device
import pickle
import torch
from pprint import pprint

norm_regime = 'sum'

fitted_folder = 'E:/ESO/Data/SPHERE/IRDIS_fitted_PAO_1P21I/'
fitted_files = os.listdir(fitted_folder)

selected_files = [fitted_files[20]] #, fitted_files[40] ]
sample_ids = [ int(file.split('.')[0]) for file in selected_files ]

regime = '1P21I'


tensy = lambda x: torch.tensor(x).to(device)

PSF_0, bg, norms, data_samples, init_config = SPHERE_preprocess(sample_ids, regime, norm_regime)
norms = norms[:, None, None].cpu().numpy() 

with open(fitted_folder + selected_files[0], 'rb') as handle:
    data = pickle.load(handle)

toy = TipToy(init_config, norm_regime, device, TipTop=False, PSFAO=True)
toy.optimizables = []

data['dx']  = 0
data['dy']  = 0
data['Jx']  = 0
data['Jy']  = 0
data['Jxy'] = 0
data['F'] = data['F'] * 0 + 1
data['bg'] = data['bg'] * 0

toy.F     = tensy( data['F']     ).squeeze()
toy.bg    = tensy( data['bg']    ).squeeze()
toy.Jy    = tensy( data['Jy']    ).flatten()
toy.Jxy   = tensy( data['Jxy']   ).flatten()
toy.Jx    = tensy( data['Jx']    ).flatten()
toy.dx    = tensy( data['dx']    ).flatten()
toy.dy    = tensy( data['dy']    ).flatten()
toy.b     = tensy( data['b']     ).flatten()
toy.r0    = tensy( data['r0']    ).flatten()
toy.amp   = tensy( data['amp']   ).flatten()
toy.beta  = tensy( data['beta']  ).flatten()
toy.theta = tensy( data['theta'] ).flatten()
toy.alpha = tensy( data['alpha'] ).flatten()
toy.ratio = tensy( data['ratio'] ).flatten()


PSFs_pred_1 = toy().detach().clone()
# PSFs_test_1 = torch.tensor( data['Img. fit'][0,...] / norms).to(device)

#%%
toy.oversampling = 2.#0-1e-5
toy.Update()
toy.optimizables = []

toy.F     = tensy( data['F']     ).squeeze()
toy.bg    = tensy( data['bg']    ).squeeze()
toy.Jy    = tensy( data['Jy']    ).flatten()
toy.Jxy   = tensy( data['Jxy']   ).flatten()
toy.Jx    = tensy( data['Jx']    ).flatten()
toy.dx    = tensy( data['dx']    ).flatten()
toy.dy    = tensy( data['dy']    ).flatten()
toy.b     = tensy( data['b']     ).flatten()
toy.r0    = tensy( data['r0']    ).flatten()
toy.amp   = tensy( data['amp']   ).flatten()
toy.beta  = tensy( data['beta']  ).flatten()
toy.theta = tensy( data['theta'] ).flatten()
toy.alpha = tensy( data['alpha'] ).flatten()
toy.ratio = tensy( data['ratio'] ).flatten()

PSFs_pred_2 = toy().detach().clone()

plot_radial_profiles(PSFs_pred_1[:,0,...], PSFs_pred_2[:,0,...], 'Before Update', 'After', title='PSFs', dpi=200, cutoff=32, scale='log')
plt.show()

#%%
%matplotlib qt
draw_PSF_stack(PSFs_pred_1, PSFs_pred_2)

# %%
