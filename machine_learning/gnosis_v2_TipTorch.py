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
entries_init = ['F','dx','dy','r0','dn', 'bg','Jx','Jy','Jxy']
fitted_dict_raw = {key: [] for key in entries_init}
ids = []

images_data = []
images_fitted = []

fitted_folder = 'E:/ESO/Data/SPHERE/IRDIS_fitted_1P21I/'
fitted_files = os.listdir(fitted_folder)

fitted_dict_raw['wind speed'] = []
fitted_dict_raw['wind direction'] = []

for file in tqdm(fitted_files):

    id = int(file.split('.')[0])
    with open(fitted_folder + file, 'rb') as handle:
        data = pickle.load(handle)
        
    ws = data['config']['atmosphere']['WindSpeed']
    wd = data['config']['atmosphere']['WindDirection']

    test = data['dn']
    images_data.append( data['Img. data'] )
    images_fitted.append( data['Img. fit'] )

    for key in entries_init:
        fitted_dict_raw[key].append(data[key])
    fitted_dict_raw['wind speed'].append(ws)
    fitted_dict_raw['wind direction'].append(wd)
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

fitted_df = pd.DataFrame(fitted_dict)
fitted_df.set_index('ID', inplace=True)

# del fitted_dict, fitted_dict_raw, ids, images_data, images_fitted, fitted_folder, fitted_files, data

fitted_ids = list( set( fitted_df.index.values.tolist() ).intersection( set(psf_df.index.values.tolist()) ) )
fitted_df = fitted_df[fitted_df.index.isin(fitted_ids)]
#%%
# Absolutize positive-only values
for entry in ['r0','Jx','Jy','Jxy', 'F (left)','F (right)', 'wind speed']:
    fitted_df[entry] = fitted_df[entry].abs()

exlude_samples = set( fitted_df.index[fitted_df['dn'] < -1].values.tolist() )
exlude_samples = exlude_samples.union( set( fitted_df.index[fitted_df['r0'] > 0.49].values.tolist() ) )
exlude_samples = exlude_samples.union( set( fitted_df.index[fitted_df['dy'].abs() > 3].values.tolist() ) )
exlude_samples = exlude_samples.union( set( fitted_df.index[fitted_df['dx'].abs() > 3].values.tolist() ) )
exlude_samples = list(exlude_samples)

fitted_df.drop(exlude_samples, inplace=True)
psf_df.drop(exlude_samples, inplace=True)

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
    'F (left)':       [True,  True,  False, False],
    # 'F (right)':     [True,  True,  False, False],
    'bg (left)':      [False, True,  False, False],
    'dx':             [False, False, True,  False],
    'dy':             [False, False, True,  False],
    'r0':             [True,  False, True,  False],
    'Jx':             [True,  True,  False, False],
    'Jy':             [True,  True,  False, False],
    'Jxy':            [True,  True,  False, False],
    'wind speed':     [True,  True,  False, False],
    'wind direction': [False, False, True,  False],
    'dn':             [False, True,  False, False ]
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

#%
# Multiply rows in 'b' column of fitted_df where corresponding 'b' value in output_df is less than 10^-5
# indices_b = np.where(np.log10(np.abs(fitted_df['b'])) < -4.5)[0].tolist()
# fitted_df.loc[fitted_df.index[indices_b], 'b'] *= 100

# # # Add 0.01 to rows in 'r0' column of fitted_df where corresponding 'r0' value in output_df is NaN
# indices_r0 = np.where(np.isnan(output_df['r0']))[0].tolist()
# fitted_df.loc[fitted_df.index[indices_r0], 'r0'] += 0.01

# # # Subtract 0.01 from rows in 'alpha' column of fitted_df where corresponding 'alpha' value in output_df is NaN
# indices_alpha = np.where(np.isnan(output_df['alpha']))[0].tolist()
# fitted_df.loc[fitted_df.index[indices_alpha], 'alpha'] += 0.01

# indicices_Jxy = np.where(fitted_df['Jxy'] < 1.0)[0].tolist()
# fitted_df.loc[fitted_df['Jxy'].index[indicices_Jxy], 'Jxy'] *= 3

# output_df = pd.DataFrame( {a: transforms_output[a].forward(fitted_df[a].values) for a in fl_out.keys()} )

# # # Subtract 0.01 from rows in 'alpha' column of fitted_df where corresponding 'alpha' value in output_df is NaN
# indices_alpha = np.where(np.isnan(output_df['alpha']))[0].tolist()
# fitted_df.loc[fitted_df.index[indices_alpha], 'alpha'] -= 0.025

# indices_b = np.where(fitted_df['b'] > 0.4)[0].tolist()
# fitted_df.loc[fitted_df.index[indices_b], 'b'] *= 0.5

# output_df = pd.DataFrame( {a: transforms_output[a].forward(fitted_df[a].values) for a in fl_out.keys()} )

#%%
save_path = DATA_FOLDER/"temp/SPHERE params TipTorch/psf_df"
for entry in selected_entries:
    sns.displot(data=psf_df, x=entry, kde=True, bins=20)
    plt.savefig(save_path/f"{entry}.png")
    plt.close()

save_path = DATA_FOLDER/"temp/SPHERE params TipTorch/fitted_df"
for entry in fitted_df.columns.values.tolist():
    sns.displot(data=fitted_df, x=entry, kde=True, bins=20)
    plt.savefig(save_path/f"{entry}.png")
    plt.close()

save_path = DATA_FOLDER/"temp/SPHERE params TipTorch/input_df"
for entry in fl_in.keys():
    sns.displot(data=input_df, x=entry, kde=True, bins=20)
    plt.savefig(save_path/f"{entry}.png")
    plt.close()

save_path = DATA_FOLDER/"temp/SPHERE params TipTorch/output_df"
for entry in output_df.columns.values.tolist():
    sns.displot(data=output_df, x=entry, kde=True, bins=20)
    plt.savefig(save_path/f"{entry}.png")
    plt.close()

inp_inv_df = pd.DataFrame( {a: transforms_input[a].backward(input_df[a].values) for a in fl_in.keys()} )
save_path = DATA_FOLDER/"temp/SPHERE params TipTorch/inp_inv_df"
for entry in fl_in.keys():
    sns.displot(data=inp_inv_df, x=entry, kde=True, bins=20)
    plt.savefig(save_path/f"{entry}.png")
    plt.close()

out_inv_df = pd.DataFrame( {a: transforms_output[a].backward(output_df[a].values) for a in fl_out.keys()} )
save_path = DATA_FOLDER/"temp/SPHERE params TipTorch/out_inv_df"
for entry in output_df.columns.values.tolist():
    sns.displot(data=out_inv_df, x=entry, kde=True, bins=20)
    plt.savefig(save_path/f"{entry}.png")
    plt.close()

#%%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# assume we have two dataframes, df1 and df2
df1 = psf_df
df2 = fitted_df

# concat the two dataframes
df = pd.concat([df1, df2], axis=1)
df = df.sort_index(axis=1)
columns_to_drop = ['F (right)', 'bg (right)', 'λ right (nm)'] #, 'Strehl']

for column_to_drop in columns_to_drop:
    if column_to_drop in df.columns:
        df = df.drop(column_to_drop, axis=1)

# calculate the correlation
corr = df.corr(method='kendall')

# generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})

# set the xtick labels
ax.set_xticklabels(df.columns)

# color the labels: red for df1, green for df2
colors = ['Black' if label in df1.columns else 'teal' for label in df.columns]
for color, label in zip(colors, ax.get_xticklabels()):
    label.set_color(color)

for color, label in zip(colors, ax.get_yticklabels()):
    label.set_color(color)

plt.xticks(rotation=45, ha='right')
plt.title("Correlation matrix (Kendall)")
plt.show()


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

X_train = torch.nan_to_num(torch.from_numpy(X_train_df.values).float().to(device))
Y_train = torch.nan_to_num(torch.from_numpy(Y_train_df.values).float().to(device))

X_val   = torch.nan_to_num(torch.from_numpy(X_valid_df.values).float().to(device))
Y_val   = torch.nan_to_num(torch.from_numpy(Y_valid_df.values).float().to(device))

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
        print('Epoch [%d/%d], Val Loss: %.4f' % (epoch+1, 100, loss_val.item()))

loss_trains = np.array(loss_trains)
loss_vals = np.array(loss_vals)

# %
plt.plot(loss_trains, label='Train')
plt.plot(loss_vals, label='Val')
plt.legend()
plt.grid()

torch.save(net.state_dict(), WEIGHTS_FOLDER/'gnosis2_TT_weights.dict')


# %%
from tools.utils import plot_radial_profiles #, draw_PSF_stack
from data_processing.SPHERE_preproc_utils import SPHERE_preprocess
from PSF_models.TipToy_SPHERE_multisrc import TipTorch
from tools.parameter_parser import ParameterParser
from tools.config_manager import ConfigManager, GetSPHEREonsky
from project_globals import device

norm_regime = 'sum'

_, _, norms, _, config_file = SPHERE_preprocess([0], '1P21I', norm_regime, device)
toy = TipTorch(config_file, norm_regime, device)
toy.optimizables = []

fl_out_keys = list(fl_out.keys())
fl_in_keys  = list(fl_in.keys())

gnosis2TT = lambda Y: { fl_out_keys[i]: transforms_output[fl_out_keys[i]].backward(Y[:,i]) for i in range(len(fl_out_keys)) }
gnosis2in = lambda X: { fl_in_keys[i]:  transforms_input[fl_in_keys[i]].backward(X[:,i])   for i in range(len(fl_in_keys))  }
in2gnosis = lambda inp: torch.from_numpy(( np.stack([transforms_input[a].forward(inp[a].values)  for a in fl_in_keys]) )).float().to(device).T
TT2gnosis = lambda out: torch.from_numpy(( np.stack([transforms_output[a].forward(out[a].values) for a in fl_out_keys]) )).float().to(device).T

# psf_df --> in --> gnosis --> out --> PAO --> image


#%%
def toy_run(model, dictionary):
        model.F     = dictionary['F (left)'].repeat(2,1).T
        model.bg    = dictionary['bg (left)'].repeat(2,1).T
        model.Jy    = dictionary['Jy']
        model.Jxy   = dictionary['Jxy']
        model.Jx    = dictionary['Jx']
        model.dx    = dictionary['dx']
        model.dy    = dictionary['dy']
        model.dn    = dictionary['dn']
        model.r0    = dictionary['r0']
        model.wind_dir = dictionary['wind direction']
        model.wind_speed = dictionary['wind speed']
        return model.forward()

def toy_run_direct(model, dict_data):
        model.F     = torch.tensor([1.,1.]).to(device)
        model.bg    = torch.tensor([0.,0.]).to(device)
        model.Jy    = torch.tensor([1.]).to(device)
        model.Jxy   = torch.tensor([1.]).to(device)
        model.Jx    = torch.tensor([1.]).to(device)
        model.dx    = torch.tensor([0.]).to(device)
        model.dy    = torch.tensor([0.]).to(device)
        model.dn    = torch.tensor([0.]).to(device)
        model.r0    = torch.tensor([dict_data['r0 (SPARTA)'].values.item()]).to(device)
        model.wind_dir = torch.tensor([dict_data['Wind direction (header)'].values.item()]).to(device)
        model.wind_speed = torch.tensor([dict_data['Wind speed (header)'].values.item()]).to(device)
        return model.forward()
    
def toy_run_mixed(model, dict_data, dict_pred):
        model.F     = torch.tensor([1.,1.]).to(device)
        model.bg    = torch.tensor([0.,0.]).to(device)  
        model.Jy    = dict_pred['Jy']
        model.Jxy   = dict_pred['Jxy']
        model.Jx    = dict_pred['Jx']
        model.dx    = dict_pred['dx']
        model.dy    = dict_pred['dy']
        model.dn    = dict_pred['dn']
        model.r0    = dict_pred['r0']
        model.wind_dir = torch.tensor([dict_data['Wind direction (header)'].values.item()]).to(device)
        model.wind_speed = torch.tensor([dict_data['Wind speed (header)'].values.item()]).to(device)
        return model.forward()
    
destack = lambda PSF_stack: [ x for x in np.split(PSF_stack[:,0,...], PSF_stack.shape[0], axis=0) ]

temp_dict = {
    'ID': [],
    'PSF (data)': [],
    'config': []
}

for id in tqdm(psf_df.index.values.tolist()):
    PSF_0, _, norms, _, config_file = SPHERE_preprocess([id], '1P21I', norm_regime, device)
    PSF_0 = PSF_0[...,1:,1:]
    config_file['sensor_science']['FieldOfView'] = 255
    temp_dict['ID'].append(id)
    temp_dict['PSF (data)'].append(PSF_0)
    temp_dict['config'].append(config_file)


#%%
val_ids = psf_df_valid.index.values.tolist()

sample_id = np.random.choice(val_ids)

PSF_0s = []
PSF_1s = []

for sample_id in tqdm(val_ids):
    sample_ids = [sample_id]
    test_in  = psf_df_valid.loc[sample_ids]
    test_out = fitted_df_valid.loc[sample_ids]

    inp = psf_df_valid.loc[sample_ids]
    out = fitted_df_valid.loc[sample_ids]

    X = in2gnosis(inp)
    Y = TT2gnosis(out)

    with torch.no_grad():
        try:
            Y_pred = net(X)
            params = gnosis2TT(Y)
            # params = gnosis2TT(Y_pred)
            
            # PSF_0, _, norms, _, config_file = SPHERE_preprocess(sample_ids, '1P21I', norm_regime, device)
            # PSF_0 = PSF_0[...,1:,1:]
            # config_file['sensor_science']['FieldOfView'] = 255

            config_file = temp_dict['config'][temp_dict['ID'].index(sample_id)]
            PSF_0 = temp_dict['PSF (data)'][temp_dict['ID'].index(sample_id)]

            # toy = TipTorch(config_file, norm_regime, device)
            # toy.optimizables = []

            toy.config = config_file
            toy.Update(reinit_grids=True, reinit_pupils=False)
            
            PSF_1 = toy_run(toy, params)
            # PSF_1 = toy_run_direct(toy, inp)
            # PSF_1 = toy_run_mixed(toy, inp, params)
            
        except:
            print('Error in sample %d' % sample_id)
        
        PSF_0s.append(PSF_0)
        PSF_1s.append(PSF_1)

#%%

PSF_0s_ = torch.stack(PSF_0s).squeeze()
PSF_1s_ = torch.stack(PSF_1s).squeeze()

plot_radial_profiles(destack(PSF_0s_),  destack(PSF_1s_),  'Data', 'Predicted', title='Fitted TipTop', dpi=200, cutoff=32, scale='log')

# %%
