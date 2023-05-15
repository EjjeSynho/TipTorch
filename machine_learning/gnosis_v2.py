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

# % Store all the distributions in the files for later use
# save_path = DATA_FOLDER/"temp/SPHERE params/"
# for entry in selected_entries:
#     sns.displot(data=psf_df, x=entry, kde=True)
#     plt.savefig(save_path/f"{entry}.png")
#     plt.close()

'''
def determine_distribution(data, alpha=0.05):
    """Determine if the data follows a Gaussian or Uniform distribution."""
    # Perform Shapiro-Wilk test for normality
    W, p_normal = stats.shapiro(data)  
    # If p-value > alpha, we cannot reject the null hypothesis that the data is normally distributed
    if p_normal > alpha: return 'Gaussian'
    # Scale the data to be between 0 and 1
    data_scaled = (data-data.min()) / (data.max()-data.min())
    # Perform Kolmogorov-Smirnov test for uniformity
    D, p_uniform = stats.kstest(data_scaled, 'uniform')
    # If p-value > alpha, we cannot reject the null hypothesis that the data is uniformly distributed
    if p_uniform > alpha: return 'Uniform'
    return 'Neither'

'''
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
            if λ == 0 :
                return torch.log(x)
            else:
                return (x**λ-1)/λ
        else:
            if λ == 0:
                return np.log(x)
            else:
                return (x**λ-1)/λ
    
    def inv_boxcox(self, y, λ):
        if isinstance(y, torch.Tensor) :
            if λ == 0:
                return torch.exp(y)
            else:
                return  (λ*y+1)**(1/λ)
        else:
            if λ == 0:
                return np.exp(y)
            else:
                return (λ*y+1)**(1/λ)

    def cdf(self, x):
        if isinstance(x, torch.Tensor):
            return self.normal_distribution.cdf(x)
        else:
            return stats.norm.cdf(x)
        
    def ppf(self, q):
        if isinstance(q, torch.Tensor):
            return torch.sqrt(torch.tensor([2.]).to(device)) * torch.erfinv(2.*q - 1.)
        else:
            return stats.norm.ppf(q)
    

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

# loc, scale = stats.expon.fit(data_init)
# x = np.linspace(min(data_init), max(data_init), 1000)
# exponential_to_uniform = lambda x, rate: 1-np.exp(-rate*x)
# data_standartized = exponential_to_uniform(data_init, 1/loc)


#%% Create fitted parameters dataset
fitted_dict_raw = {key: [] for key in ['F','b','dx','dy','r0','amp','beta','alpha','theta','ratio','bg','Jx','Jy','Jxy']}
ids = []

images_data = []
images_fitted = []

fitted_folder = 'E:/ESO/Data/SPHERE/IRDIS_fitted_PAO_1P21I/'
fitted_files  = os.listdir(fitted_folder)

for file in tqdm(fitted_files):
    id = int(file.split('.')[0])

    with open(fitted_folder+file, 'rb') as handle:
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

del fitted_dict, fitted_dict_raw, ids, images_data, images_fitted, selected_entries, fitted_folder, fitted_files, data

#%
fitted_ids = list( set( fitted_df.index.values.tolist() ).intersection( set(psf_df.index.values.tolist()) ) )
fitted_df = fitted_df[fitted_df.index.isin(fitted_ids)]

# Absolutize positive-only values
for entry in ['b','r0','Jx','Jy','Jxy','ratio','alpha','F (left)','F (right)']:
    fitted_df[entry] = fitted_df[entry].abs()

threshold = 3 # in STDs

bad_alpha = np.where(np.abs(fitted_df['alpha']) > threshold * fitted_df['alpha'].std())[0].tolist()
bad_beta  = np.where(np.abs(fitted_df['beta'])  > threshold * fitted_df['beta'].std())[0].tolist()
bad_amp   = np.where(np.abs(fitted_df['amp'])   > threshold * fitted_df['amp'].std())[0].tolist()
bad_ratio = np.where(np.abs(fitted_df['ratio']) > 2)[0].tolist()
bad_theta = np.where(np.abs(fitted_df['theta']) > 3)[0].tolist()

exclude_ids = set(bad_alpha + bad_beta + bad_amp + bad_ratio + bad_theta)
exlude_samples = [fitted_df.iloc[id].name for id in exclude_ids]

for id in exlude_samples:
    fitted_df = fitted_df.drop(id)

#% Store all the distributions in the files for later use
# save_path = DATA_FOLDER/"temp/SPHERE params/fitted"
# for entry in fitted_df.columns.values.tolist():
#     sns.displot(data=fitted_df, x=entry, kde=True)
#     plt.savefig(save_path/f"{entry}.png")
#     plt.close()

psf_df = psf_df[psf_df.index.isin(fitted_df.index.values.tolist())]

#%% Store all the distributions in the files for later use
# save_path = DATA_FOLDER/"temp/SPHERE params/fitted"
# for entry in fitted_df.columns.values.tolist():
#     sns.displot(data=fitted_df, x=entry, kde=True)
#     plt.savefig(save_path/f"{entry}.png")
#     plt.close()

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

# save_path = DATA_FOLDER/"temp/SPHERE params/input"
# for entry in fl_in.keys():
#     sns.displot(data=input_df, x=entry, kde=True, bins=20)
#     plt.savefig(save_path/f"{entry}.png")
#     plt.close()

#%%
# save_path = DATA_FOLDER/"temp/SPHERE params/fitted"
# for entry in fitted_df.columns.values.tolist():
#     sns.displot(data=fitted_df, x=entry, kde=True)
#     plt.savefig(save_path/f"{entry}.png")
#     plt.close()
    
#%%

# xx_df = pd.DataFrame( {a: transforms_input[a].backward(input_df[a].values) for a in fl_in.keys()} )

# save_path = DATA_FOLDER/"temp/SPHERE params/aa"
# for entry in fl_in.keys():
#     sns.displot(data=xx_df, x=entry, kde=True, bins=20)
#     plt.savefig(save_path/f"{entry}.png")
#     plt.close()


#%%
fl_out = {          # boxcox gaussian uniform invert
    'F (left)':   [True,  True,  False, False],
    # 'F (right)':  [True,  True,  False, False],
    'bg (left)':  [False, True,  False, False],
    # 'bg (right)': [False, True,  False, False],
    'b':          [True,  False, True,  False],
    'dx':         [False, False, True,  False],
    'dy':         [False, False, True,  False],
    'r0':         [True,  False, True,  False],
    'amp':        [True,  False, True,  False],
    'beta':       [True,  True,  False, False],
    'alpha':      [True,  False, True,  False],
    'theta':      [False, False, True,  False],
    'ratio':      [False, False, True,  False],
    'Jx':         [True,  True,  False, False],
    'Jy':         [True,  True,  False, False],
    'Jxy':        [True,  False, True,  False]
}

transforms_output = {\
    i: DataTransformer(fitted_df[i].values, boxcox=fl_out[i][0], gaussian=fl_out[i][1], uniform=fl_out[i][2], invert=fl_out[i][3]) for i in fl_out.keys() }

output_df = pd.DataFrame( {a: transforms_output[a].forward(fitted_df[a].values) for a in fl_out.keys()} )

#%%
# Multiply rows in 'b' column of fitted_df where corresponding 'b' value in output_df is less than 10^-5
indices_b = np.where(np.log10(np.abs(fitted_df['b'])) < -4.5)[0].tolist()
fitted_df.loc[fitted_df.index[indices_b], 'b'] *= 100

# # Add 0.01 to rows in 'r0' column of fitted_df where corresponding 'r0' value in output_df is NaN
indices_r0 = np.where(np.isnan(output_df['r0']))[0].tolist()
fitted_df.loc[fitted_df.index[indices_r0], 'r0'] += 0.01

# # Subtract 0.01 from rows in 'alpha' column of fitted_df where corresponding 'alpha' value in output_df is NaN
indices_alpha = np.where(np.isnan(output_df['alpha']))[0].tolist()
fitted_df.loc[fitted_df.index[indices_alpha], 'alpha'] += 0.01

indicices_Jxy = np.where(fitted_df['Jxy'] < 1.0)[0].tolist()
fitted_df.loc[fitted_df['Jxy'].index[indicices_Jxy], 'Jxy'] *= 3

output_df = pd.DataFrame( {a: transforms_output[a].forward(fitted_df[a].values) for a in fl_out.keys()} )

# # Subtract 0.01 from rows in 'alpha' column of fitted_df where corresponding 'alpha' value in output_df is NaN
indices_alpha = np.where(np.isnan(output_df['alpha']))[0].tolist()
fitted_df.loc[fitted_df.index[indices_alpha], 'alpha'] -= 0.025

indices_b = np.where(fitted_df['b'] > 0.4)[0].tolist()
fitted_df.loc[fitted_df.index[indices_b], 'b'] *= 0.5

output_df = pd.DataFrame( {a: transforms_output[a].forward(fitted_df[a].values) for a in fl_out.keys()} )


#% Store all the distributions in the files for later use
# save_path = DATA_FOLDER/"temp/SPHERE params/output"
# for entry in output_df.columns.values.tolist():
#     sns.displot(data=output_df, x=entry, kde=True)
#     plt.savefig(save_path/f"{entry}.png")
#     plt.close()

#%%
ids_to_remain = set(psf_df.index.values).difference(ids_to_exclude_later)
psf_df = psf_df[psf_df.index.isin(ids_to_remain)]
fitted_df = fitted_df[fitted_df.index.isin(psf_df.index.values.tolist())]

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
X_columns = X_train_df.columns.tolist()
# ids_train = psf_df_train.index.values.tolist()

Y_train_df = pd.DataFrame( {a: transforms_output[a].forward(fitted_df_train[a].values) for a in fl_out.keys()} )
Y_valid_df = pd.DataFrame( {a: transforms_output[a].forward(fitted_df_valid[a].values) for a in fl_out.keys()} )
Y_train_df.index = psf_df_train.index
Y_valid_df.index = psf_df_valid.index
Y_columns = Y_train_df.columns.tolist()
# ids_valid = psf_df_valid.index.values.tolist()

assert Y_train_df.index.values.tolist() == X_train_df.index.values.tolist()

X_train = torch.from_numpy(X_train_df.values).float().to(device)
Y_train = torch.from_numpy(Y_train_df.values).float().to(device)

X_val = torch.from_numpy(X_valid_df.values).float().to(device)
Y_val = torch.from_numpy(Y_valid_df.values).float().to(device)

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

# %%

# df_ultimate = pd.concat([input_df, output_df], axis=1)
# df_ultimate = pd.concat([psf_df, fitted_df], axis=1)

# seaborn scatter r0 vs jitter
# sns.scatterplot(data=output_df, x='r0', y='alpha')

# sns.scatterplot(data=fitted_df, x='r0', y='Jy', hue='alpha', palette='viridis')
# sns.scatterplot(data=df_ultimate, x='r0', y='Jy', hue='λ left (nm)', palette='viridis')

# sns.scatterplot(data=df_ultimate, x='beta', y='alpha')

# plt.ylim(0, 0.5)
# plt.xlim(0, 1)

# sns.kdeplot(data = df_ultimate,
#             x='theta',
#             y='ratio',
#             color='r', fill=True,
#             cmap="Reds", thresh=0.02)



#%%
from tools.utils import plot_radial_profiles #, draw_PSF_stack
from data_processing.SPHERE_preproc_utils import SPHERE_preprocess
from PSF_models.TipToy_SPHERE_multisrc import TipToy

norm_regime = 'sum'
regime = 'different'

fl_out_keys = list(fl_out.keys())

gnosis2PAO = lambda Y: { fl_out_keys[i]: transforms_output[fl_out_keys[i]].backward(Y[:,i]) for i in range(len(fl_out_keys)) }
in2gnosis  = lambda inp: torch.from_numpy(( np.stack([transforms_input[a].forward(inp[a].values) for a in fl_in.keys()]) )).float().to(device).T

def GetImages(sample_ids, norms):
    PSF_0_ = []
    PSF_1_ = []

    for i,id_im in enumerate([images_dict['ID'].index(sample_id) for sample_id in sample_ids]):
        PSF_0_.append(images_dict['PSF (data)'][id_im][0,...] / norms[i,:][:,None, None].cpu().numpy())
        PSF_1_.append(images_dict['PSF (fit)' ][id_im][0,...] / norms[i,:][:,None, None].cpu().numpy())
        
    PSF_0_ = np.stack(PSF_0_)
    PSF_1_ = np.stack(PSF_1_)
    return PSF_0_, PSF_1_


def GetBatch(rand_ids):
    sample_ids = [X_valid_df.index.values.tolist()[rand_id] for rand_id in rand_ids]

    # PSF_0, bg, norms, data_samples, merged_config = SPHERE_preprocess(sample_ids, regime, norm_regime)
    _, _, norms, _, merged_config = SPHERE_preprocess(sample_ids, regime, norm_regime)
    PSF_0_, PSF_1_ = GetImages(sample_ids, norms)

    batch = {
        'confo': merged_config,
        'PSF (data)': PSF_0_,
        'PSF (fit)': PSF_1_,
        'X': in2gnosis( psf_df_valid.loc[sample_ids] )
    }
    return batch


rand_ids = np.arange(len(X_valid_df))
np.random.shuffle(rand_ids)

rand_ids = torch.split(torch.from_numpy(rand_ids), 64)

batches = [GetBatch(rand_id) for rand_id in rand_ids]

#%%
X_test = in2gnosis(sample)
Y_test = net(X_test)

pred_test = gnosis2PAO(Y_test)


#%%
toy = TipToy(merged_config, norm_regime, device, TipTop=False, PSFAO=True)

toy.optimizables = []
toy.F     = pred_test['F (left)'].repeat(2,1).T
toy.bg    = pred_test['bg (left)'].repeat(2,1).T
toy.Jy    = pred_test['Jxy']
toy.Jxy   = pred_test['Jx']
toy.Jx    = pred_test['Jy']
toy.dx    = pred_test['dx']
toy.dy    = pred_test['dy']
toy.r0    = pred_test['r0']
toy.b     = pred_test['b']
toy.amp   = pred_test['amp']
toy.beta  = pred_test['beta']
toy.theta = pred_test['theta']
toy.alpha = pred_test['alpha']
toy.ratio = pred_test['ratio']

PSF_pred  = toy.forward()

#%%
# from tools.utils import register_hooks
# Q = loss_fn(PSF_pred, torch.ones_like(PSF_pred).to(device))
# get_dot = register_hooks(Q)
# Q.backward()
# dot = get_dot()
# #dot.save('tmp.dot') # to get .dot
# #dot.render('tmp') # to get SVG
# dot # in Jupyter, you can just render the variable

#%%
# plot_radial_profiles(destack(PSF_0), destack(PSF_1), 'Data', 'Fit', title='NN PSFAO prediction', dpi=200, cutoff=64)

PSF_pred_ = PSF_pred.cpu().detach().numpy()

# PSF_pred__ = PSF_pred_ / PSF_pred_.max(axis=(-2,-1))[...,None,None]
# PSF_1__    = PSF_1_    / PSF_1_.max(axis=(-2,-1))[...,None,None]

destack = lambda PSF_stack: [ x for x in np.split(PSF_stack[:,0,...], PSF_stack.shape[0], axis=0) ]

# plot_radial_profiles(destack(PSF_1__), destack(PSF_pred__), 'Fit', 'Predicted', title='NN PSFAO prediction', dpi=200, cutoff=64)
plot_radial_profiles(destack(PSF_1_), destack(PSF_pred_), 'Fit', 'Predicted', title='NN PSFAO prediction', dpi=200, cutoff=64)
# plot_radial_profiles(destack(PSF_1_), destack(PSF_0_), 'Fit', 'Predicted', title='NN PSFAO prediction', dpi=200, cutoff=64)


#%%
