#%%
%reload_ext autoreload
%autoreload 2

import sys
from typing import Any
sys.path.append('..')

import torch
import pickle

from project_globals import SPHERE_DATA_FOLDER, DATA_FOLDER, device
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

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
                    'Jitter X',
                    'Jitter Y',
                    'λ left (nm)',
                    'λ right (nm)']

psf_df['Nph WFS'] *= psf_df['Rate']
psf_df['Jitter X'] = psf_df['Jitter X'].abs()
psf_df['Jitter Y'] = psf_df['Jitter Y'].abs()

psf_df = psf_df[selected_entries]
psf_df.sort_index(inplace=True)

#%% Create fitted parameters dataset
#check if file exists
if not os.path.isfile('E:/ESO/Data/SPHERE/fitted_df_PAO.pickle'):
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

    # images_dict = {
    #     'ID': ids,
    #     'PSF (data)': images_data,
    #     'PSF (fit)': images_fitted
    # }

    fitted_df = pd.DataFrame(fitted_dict)
    fitted_df.set_index('ID', inplace=True)

    # Save dataframe
    with open('E:/ESO/Data/SPHERE/fitted_df_PAO.pickle', 'wb') as handle:
        pickle.dump(fitted_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
else:
    with open('E:/ESO/Data/SPHERE/fitted_df_PAO.pickle', 'rb') as handle:
        print('Loading dataframe "fitted_df_PAO.pickle"...')
        fitted_df = pickle.load(handle)

fitted_ids = list( set( fitted_df.index.values.tolist() ).intersection( set(psf_df.index.values.tolist()) ) )
fitted_df = fitted_df[fitted_df.index.isin(fitted_ids)]

# Absolutize positive-only values
for entry in ['b','r0','Jx','Jy','Jxy','ratio','alpha','F (left)','F (right)']:
    fitted_df[entry] = fitted_df[entry].abs()
fitted_df.sort_index(inplace=True)

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

#%% Compute data transformations
from data_processing.normalizers import BoxCox, Uniform, TransformSequence, Invert, DataTransformer

'''
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
input_df.index = psf_df.index

fl_out = {          # boxcox gaussian uniform invert
    'F (left)':   [True,  True,  False, False],
    # 'F (right)':  [True,  True,  False, False],
    'bg (left)':  [False, True,  False, False],
    # 'bg (right)': [False, True,  False, False],
    'b':          [False,  True,  False, False],
    'dx':         [False, False, True,  False],
    'dy':         [False, False, True,  False],
    'r0':         [True,  False, True,  False],
    'amp':        [True,  True,  False, False],
    'beta':       [False,  True,  False, False],
    'alpha':      [False,  False, True,  False],
    'theta':      [False, False, True,  False],
    'ratio':      [False, False, True,  False],
    'Jx':         [False,  True,  False, False],
    'Jy':         [False,  True,  False, False],
    'Jxy':        [False,  True,  False, False]
}

transforms_output = { i: DataTransformer(fitted_df[i].values, boxcox=fl_out[i][0], gaussian=fl_out[i][1], uniform=fl_out[i][2], invert=fl_out[i][3]) for i in fl_out.keys() }
output_df = pd.DataFrame( {a: transforms_output[a].forward(fitted_df[a].values) for a in transforms_output.keys()} )
output_df.index = fitted_df.index
'''

transforms_input = {}
# transforms_input['Airmass']                 = TransformSequence( transforms = [ BoxCox(lmbda=-2.532), Uniform(a=0.0, b=0.312) ])
transforms_input['Airmass']                 = TransformSequence( transforms = [ Uniform(a=1.0, b=2.2) ])
# transforms_input['r0 (SPARTA)']             = TransformSequence( transforms = [ BoxCox(lmbda=0.20), Uniform(a=-1.8, b=-1.0) ])
transforms_input['r0 (SPARTA)']             = TransformSequence( transforms = [ Uniform(a=0.05, b=0.45) ])
transforms_input['Wind direction (header)'] = TransformSequence( transforms = [ Uniform(a=0, b=360) ])
# transforms_input['Wind speed (header)']     = TransformSequence( transforms = [ BoxCox(lmbda=0.4), Uniform(a=0.5, b=5) ])
transforms_input['Wind speed (header)']     = TransformSequence( transforms = [ Uniform(a=0.0, b=17.5) ])
# transforms_input['Tau0 (header)']           = TransformSequence( transforms = [ BoxCox(data=psf_df['Tau0 (header)']), Uniform(a=-5.5, b=-4) ])
transforms_input['Tau0 (header)']           = TransformSequence( transforms = [ Uniform(a=0.0, b=0.025) ])
transforms_input['λ left (nm)']             = TransformSequence( transforms = [ Uniform(a=psf_df['λ left (nm)'].min(), b=psf_df['λ left (nm)'].max()) ])
transforms_input['λ right (nm)']            = TransformSequence( transforms = [ Uniform(a=psf_df['λ right (nm)'].min(), b=psf_df['λ right (nm)'].max()) ])
transforms_input['Rate']                    = TransformSequence( transforms = [ Uniform(a=psf_df['Rate'].min(), b=psf_df['Rate'].max()) ])
# transforms_input['FWHM']                    = TransformSequence( transforms = [ BoxCox(lmbda=-0.7), Uniform(a=-0.2, b=0.5) ])
transforms_input['FWHM']                    = TransformSequence( transforms = [ Uniform(a=0.5, b=3.0) ])
# transforms_input['Nph WFS']                 = TransformSequence( transforms = [ BoxCox(lmbda=-0.25), Uniform(a=1, b=3) ])
transforms_input['Nph WFS']                 = TransformSequence( transforms = [ Uniform(a=0, b=1000) ])
# transforms_input['Strehl']                  = TransformSequence( transforms = [ Invert(), BoxCox(lmbda=0.1), Uniform(a=-2.25, b=-0.5) ])
transforms_input['Strehl']                  = TransformSequence( transforms = [ Uniform(a=0.015, b=1.0) ])
transforms_input['Jitter X']                = TransformSequence( transforms = [ Uniform(a=0.0,   b=60.0) ])
transforms_input['Jitter Y']                = TransformSequence( transforms = [ Uniform(a=0.0,   b=60.0) ])


input_df = pd.DataFrame( {a: transforms_input[a].forward(psf_df[a].values) for a in transforms_input.keys()} )
input_df.index = psf_df.index

transforms_output = {}
# transforms_output['amp']        = TransformSequence( transforms = [ BoxCox(lmbda=-1.5), Uniform(a=0.525, b=0.675) ])
transforms_output['amp']        = TransformSequence( transforms = [ Uniform(a=0.0, b=20.) ])
# transforms_output['r0']         = TransformSequence( transforms = [ BoxCox(lmbda=-0.0), Uniform(a=-3.75, b=2) ])
transforms_output['r0']         = TransformSequence( transforms = [ Uniform(a=0.0, b=3.0) ])
# transforms_output['alpha']      = TransformSequence( transforms = [ BoxCox(lmbda=-0.034), Uniform(a=-4, b=-1) ])
transforms_output['alpha']      = TransformSequence( transforms = [ Uniform(a=0.0, b=0.4) ])
# transforms_output['ratio']      = TransformSequence( transforms = [ BoxCox(lmbda=-0.0), Uniform(a=-0.42, b=0.6) ])
transforms_output['ratio']      = TransformSequence( transforms = [ Uniform(a=0.6, b=1.6) ])
# transforms_output['b']          = TransformSequence( transforms = [ BoxCox(lmbda=0.3454), Uniform(a=-2.5, b=-1.5) ])
transforms_output['b']          = TransformSequence( transforms = [ Uniform(a=0.0, b=0.3) ])
# transforms_output['beta']       = TransformSequence( transforms = [ BoxCox(lmbda=-1.384), Uniform(a=0.25, b=0.575) ])
transforms_output['beta']       = TransformSequence( transforms = [ Uniform(a=0.0, b=10.0) ])
transforms_output['theta']      = TransformSequence( transforms = [ Uniform(a=-1, b=1) ])
transforms_output['bg (left)']  = TransformSequence( transforms = [ Uniform(a=-0.25e-5, b=0.15e-5) ])
transforms_output['bg (right)'] = TransformSequence( transforms = [ Uniform(a=-0.25e-5, b=0.15e-5) ])
transforms_output['dx']         = TransformSequence( transforms = [ Uniform(a=-1, b=1) ])
transforms_output['dy']         = TransformSequence( transforms = [ Uniform(a=-1, b=1) ])
transforms_output['F (left)']   = TransformSequence( transforms = [ Uniform(a=0.9, b=1.1) ])
transforms_output['F (right)']  = TransformSequence( transforms = [ Uniform(a=0.9, b=1.1) ])
transforms_output['Jx']         = TransformSequence( transforms = [ Uniform(a=15, b=30) ])
transforms_output['Jy']         = TransformSequence( transforms = [ Uniform(a=15, b=30) ])
# transforms_output['Jxy']        = TransformSequence( transforms = [ BoxCox(lmbda=0.091), Uniform(a=-4, b=4) ])
transforms_output['Jxy']        = TransformSequence( transforms = [ Uniform(a=0, b=100) ])

output_df = pd.DataFrame( {a: transforms_output[a].forward(fitted_df[a].values) for a in transforms_output.keys()} )
output_df.index = fitted_df.index

rows_with_nan = output_df.index[output_df.isna().any(axis=1)].values.tolist()


'''
entry = 'ratio'

A = fitted_df[entry].values
B = transforms_output[entry].forward(A)
C = transforms_output[entry].backward(B)

test_df = pd.DataFrame({'A': A, 'B': B, 'C': C})

sns.displot(test_df, x='A', bins=20)
sns.displot(test_df, x='B', bins=20)
sns.displot(test_df, x='C', bins=20)
'''


def SaveTransformedResults():
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
    for entry in transforms_input.keys():
        sns.displot(data=input_df, x=entry, kde=True, bins=20)
        plt.savefig(save_path/f"{entry}.png")
        plt.close()

    save_path = DATA_FOLDER/"temp/SPHERE params/output_df"
    for entry in output_df.columns.values.tolist():
        sns.displot(data=output_df, x=entry, kde=True, bins=20)
        plt.savefig(save_path/f"{entry}.png")
        plt.close()

    inp_inv_df = pd.DataFrame( {a: transforms_input[a].backward(input_df[a].values) for a in transforms_input.keys()} )
    save_path = DATA_FOLDER/"temp/SPHERE params/inp_inv_df"
    for entry in transforms_input.keys():
        sns.displot(data=inp_inv_df, x=entry, kde=True, bins=20)
        plt.savefig(save_path/f"{entry}.png")
        plt.close()

    out_inv_df = pd.DataFrame( {a: transforms_output[a].backward(output_df[a].values) for a in transforms_output.keys()} )
    save_path = DATA_FOLDER/"temp/SPHERE params/out_inv_df"
    for entry in output_df.columns.values.tolist():
        sns.displot(data=out_inv_df, x=entry, kde=True, bins=20)
        plt.savefig(save_path/f"{entry}.png")
        plt.close()

# SaveTransformedResults()

#%%
psf_df.drop(rows_with_nan, inplace=True)
input_df.drop(rows_with_nan, inplace=True)
fitted_df.drop(rows_with_nan, inplace=True)
output_df.drop(rows_with_nan, inplace=True)

psf_df.sort_index(inplace=True)
fitted_df.sort_index(inplace=True)

print('Number of samples after the filtering: {}'.format(len(psf_df)))

assert psf_df.index.equals(fitted_df.index)

#%% Group by wavelengths
print('Grouping data by wavelengths...')

psf_df['λ group'] = psf_df.groupby(['λ left (nm)', 'λ right (nm)']).ngroup()
unique_wvls = psf_df[['λ left (nm)', 'λ right (nm)', 'λ group']].drop_duplicates()
group_counts = psf_df.groupby(['λ group']).size().reset_index(name='counts')
group_counts = group_counts.set_index('λ group')
print(group_counts)

# Flag train and validation sets
def generate_binary_sequence(size, percentage_of_ones):
    sequence = np.random.choice([0, 1], size=size-1, p=[1-percentage_of_ones, percentage_of_ones])
    sequence = np.append(sequence, 1)
    np.random.shuffle(sequence)
    return sequence
    
train_valid_df = { 'ID': [] , 'For validation': [] }

for group_id in group_counts.index.values:
    ids = psf_df[psf_df['λ group'] == group_id].index.values
    train_valid_ratio = generate_binary_sequence(len(ids), 0.2)
    train_valid_df['ID'] += ids.tolist()
    train_valid_df['For validation'] += train_valid_ratio.tolist()

train_valid_df = pd.DataFrame(train_valid_df)
train_valid_df.set_index('ID', inplace=True)
train_valid_df.sort_index(inplace=True)
train_valid_df[train_valid_df['For validation'] == 1] = True
train_valid_df[train_valid_df['For validation'] == 0] = False

psf_df = pd.concat([psf_df, train_valid_df], axis=1)


# Split into the batches with the same wavelength
psf_df_batches_train = []
psf_df_batches_valid = []

fitted_df_batches_train = []
fitted_df_batches_valid = []

for group_id in group_counts.index.values:
    batch_ids = psf_df.index[psf_df['λ group'] == group_id]
    
    buf_psf_df    = psf_df.loc[batch_ids]
    buf_fitted_df = fitted_df.loc[batch_ids]
    
    buf_train_ids = buf_psf_df.index[buf_psf_df['For validation'] == False]
    buf_val_ids   = buf_psf_df.index[buf_psf_df['For validation'] == True ]
    
    # Split big batches into smaller ones
    if len(buf_train_ids) > 40:
        n_parts = np.ceil(len(buf_train_ids) // 32)
        id_divisions = np.array_split(buf_train_ids, n_parts)
        psf_df_batches_train += [buf_psf_df.loc[ids] for ids in id_divisions]
        fitted_df_batches_train += [buf_fitted_df.loc[ids] for ids in id_divisions]
    else:
        psf_df_batches_train.append(buf_psf_df.loc[buf_train_ids])
        fitted_df_batches_train.append(buf_fitted_df.loc[buf_train_ids])
    
    # Split big batches into smaller ones
    if len(buf_val_ids) > 40:
        n_parts = np.ceil(len(buf_val_ids) // 32)
        id_divisions = np.array_split(buf_val_ids, n_parts)
        psf_df_batches_valid += [buf_psf_df.loc[ids] for ids in id_divisions]
        fitted_df_batches_valid += [buf_fitted_df.loc[ids] for ids in id_divisions]
    else:
        psf_df_batches_valid.append(buf_psf_df.loc[buf_val_ids])
        fitted_df_batches_valid.append(buf_fitted_df.loc[buf_val_ids])

print('Number of batches for training: {}'.format(len(psf_df_batches_train)))
print('Number of batches for validation: {}'.format(len(psf_df_batches_valid)))

for i in range(len(psf_df_batches_train)):
    assert len(psf_df_batches_train[i]) == len(fitted_df_batches_train[i])
    
for i in range(len(psf_df_batches_valid)):
    assert len(psf_df_batches_valid[i]) == len(fitted_df_batches_valid[i])

# for i in range(len(psf_df_batches_train)):
    # print(len(psf_df_batches_train[i]), len(fitted_df_batches_train[i]))
    
# for i in range(len(psf_df_batches_valid)):
#     print(len(psf_df_batches_valid[i]), len(fitted_df_batches_valid[i]))

#%%
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
# psf_df --> in --> gnosis --> out --> PAO --> image

fl_out_keys = list(transforms_output.keys())
fl_in_keys  = list(transforms_input.keys())

gnosis2PAO = lambda Y: { fl_out_keys[i]: transforms_output[fl_out_keys[i]].backward(Y[:,i]) for i in range(len(fl_out_keys)) }
gnosis2in  = lambda X: { fl_in_keys[i]:  transforms_input[fl_in_keys[i]].backward(X[:,i])   for i in range(len(fl_in_keys))  }
in2gnosis  = lambda inp: torch.from_numpy(( np.stack([transforms_input[a].forward(inp[a].values)  for a in fl_in_keys]) )).T
PAO2gnosis = lambda out: torch.from_numpy(( np.stack([transforms_output[a].forward(out[a].values) for a in fl_out_keys]))).T

psf_df_train,    psf_df_valid    = pd.concat(psf_df_batches_train),    pd.concat(psf_df_batches_valid)
fitted_df_train, fitted_df_valid = pd.concat(fitted_df_batches_train), pd.concat(fitted_df_batches_valid)

X_train, X_valid = in2gnosis(psf_df_train),     in2gnosis(psf_df_valid)
Y_train, Y_valid = PAO2gnosis(fitted_df_train), PAO2gnosis(fitted_df_valid)

#%%
from project_globals import device
import torch.nn as nn
import torch.optim as optim

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
net.double()

# loss_fn = nn.MSELoss(reduction='mean')
loss_fn = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)


#%%
from project_globals import WEIGHTS_FOLDER

# Assume we have some input data in a pandas DataFrame
# inputs and targets should be your actual data
loss_trains = []
loss_vals   = []

# Training loop
num_iters = 55

for epoch in range(num_iters):  # number of epochs
    optimizer.zero_grad()   # zero the gradient buffers
    y_pred = net(X_train.to(device))   # forward pass
    loss = loss_fn(y_pred, Y_train.to(device))  # compute loss
    loss_trains.append(loss.detach().cpu().item())
    loss.backward()  # backpropagation
    optimizer.step()  # update weights
    print('Epoch [%d/%d], Loss: %.4f' % (epoch+1, num_iters, loss.item()))

    with torch.no_grad():
        y_pred_val = net(X_valid.to(device))
        loss_val = loss_fn(y_pred_val, Y_valid.to(device))
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

torch.cuda.empty_cache()

#%%
from tools.utils import plot_radial_profiles #, draw_PSF_stack
from data_processing.SPHERE_preproc_utils import SPHERE_preprocess
from PSF_models.TipToy_SPHERE_multisrc import TipTorch

norm_regime = 'sum'

def toy_run(model, dictionary):
        model.F  = dictionary['F (left)'].unsqueeze(-1)
        model.bg = dictionary['bg (left)'].unsqueeze(-1)
        
        for attr in ['Jy','Jxy','Jx','dx','dy','b','r0','amp','beta','theta','alpha','ratio']:
            setattr(model, attr, dictionary[attr])
        
        return model.forward()

def prepare_batch_configs(batches_in, batches_out):
    batches_dict = []
    for i in tqdm(range(len(batches_in))):
        sample_ids = batches_in[i].index.tolist()
        PSF_0, _, _, _, config_files = SPHERE_preprocess(sample_ids, 'different', norm_regime, device)
        PSF_0 = PSF_0[..., 1:, 1:].cpu()
        config_files['sensor_science']['FieldOfView'] = 255
        
        batch_dict = {
            'ids': sample_ids,
            'PSF (data)': PSF_0,
            'configs': config_files,
            'X': in2gnosis ( batches_in[i].loc[sample_ids]  ),
            'Y': PAO2gnosis( batches_out[i].loc[sample_ids] )
        }
        batches_dict.append(batch_dict)
    return batches_dict

batches_dict_train = prepare_batch_configs(psf_df_batches_train, fitted_df_batches_train)
batches_dict_valid = prepare_batch_configs(psf_df_batches_valid, fitted_df_batches_valid)

#%%
# Load weights
net.load_state_dict(torch.load(WEIGHTS_FOLDER/'gnosis2_weights.dict'))

#%% Train in online fashion
epochs = 100

train_samples = psf_df_train.index.values.tolist()
val_samples   = psf_df_valid.index.values.tolist()

optimizer = optim.Adam(net.parameters(), lr=1e1)
loss_fn = nn.L1Loss(reduction='sum')

config_file = temp_dict['config'][temp_dict['ID'].index(27)]
toy = TipTorch(config_file, norm_regime, device)
toy.optimizables = []

torch.cuda.empty_cache()
loss_train = []
loss_val = []


for epoch in range(epochs):
    np.random.shuffle(train_samples)
    epoch_train_loss = []
    for i,sample in enumerate(train_samples):
        optimizer.zero_grad()

        inp = psf_df_train.loc[[sample]]
        out = fitted_df_train.loc[[sample]]

        X = in2gnosis(inp)
        Y = PAO2gnosis(out)

        Y_pred = torch.nan_to_num(net(X))
        params = gnosis2PAO(Y_pred)
        
        config_file = temp_dict['config'][temp_dict['ID'].index(sample)]
        PSF_0 = temp_dict['PSF (data)'][temp_dict['ID'].index(sample)]

        toy.config = config_file
        toy.Update(reinit_grids=True, reinit_pupils=False)
        
        PSF_1 = toy_run(toy, params)
        
        loss = loss_fn(PSF_1, PSF_0)
        loss.backward()
        optimizer.step()
        
        loss_train.append(torch.nan_to_num(loss).item())
        epoch_train_loss.append(torch.nan_to_num(loss).item())
        print(f'Current Loss ({i+1}/{len(train_samples)}): {loss.item():.4f}', end='\r')
        
    print(f'Epoch: {epoch+1}/{epochs}, train loss: {np.array(epoch_train_loss).mean().item()}')
    
    np.random.shuffle(val_samples)
    epoch_val_loss = []
    for i,sample in enumerate(val_samples):
        with torch.no_grad():
            inp = psf_df_valid.loc[[sample]]
            out = fitted_df_valid.loc[[sample]]

            X = in2gnosis(inp)
            Y = PAO2gnosis(out)

            Y_pred = torch.nan_to_num(net(X))
            params = gnosis2PAO(Y_pred)
            
            config_file = temp_dict['config'][temp_dict['ID'].index(sample)]
            PSF_0 = temp_dict['PSF (data)'][temp_dict['ID'].index(sample)]

            toy.config = config_file
            toy.Update(reinit_grids=True, reinit_pupils=False)
            
            PSF_1 = toy_run(toy, params)
            loss = loss_fn(PSF_1, PSF_0)
            
            loss_val.append(torch.nan_to_num(loss).item())
            epoch_val_loss.append(torch.nan_to_num(loss).item())
            print(f'Current Loss ({i+1}/{len(train_samples)}): {loss.item():.4f}', end='\r')
    
    print(f'Epoch: {epoch+1}/{epochs}, validation loss: {np.array(epoch_val_loss).mean().item()}')
 

#%% Test online fashion
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
    Y = PAO2gnosis(out)

    with torch.no_grad():
        try:
            Y_pred = net(X)
            params = gnosis2PAO(Y)

            config_file = temp_dict['config'][temp_dict['ID'].index(sample_id)]
            PSF_0 = temp_dict['PSF (data)'][temp_dict['ID'].index(sample_id)]

            toy.config = config_file
            toy.Update(reinit_grids=True, reinit_pupils=False)
            
            PSF_1 = toy_run(toy, params)
            
        except:
            print('Error in sample %d' % sample_id)
        
        PSF_0s.append(PSF_0)
        PSF_1s.append(PSF_1)

#%
destack = lambda PSF_stack: [ x for x in np.split(PSF_stack[:,0,...], PSF_stack.shape[0], axis=0) ]

PSF_0s_ = torch.stack(PSF_0s).squeeze()
PSF_1s_ = torch.stack(PSF_1s).squeeze()

plot_radial_profiles(destack(PSF_0s_),  destack(PSF_1s_),  'Data', 'Predicted', title='Fitted TipTop', dpi=200, cutoff=32, scale='log')


#%%
net = Gnosis2(X_train.shape[1], Y_train.shape[1], 100, 0.25)
net.to(device)
net.double()
#%%
optimizer = optim.Adam(net.parameters(), lr=0.001)
# optimizer = optim.LBFGS(net.parameters(), lr=1, history_size=20, max_iter=4, line_search_fn="strong_wolfe")

loss_PSF = nn.L1Loss(reduction='sum')

def loss_fn(output, target):
    return loss_PSF(output, target) + torch.amax(torch.abs(output-target), dim=(-2,-1)).sum() * 1e1
    # return torch.amax(torch.abs(output-target), dim=(-2,-1)).mean() * 1e2

epochs = 26
torch.cuda.empty_cache()

toy = TipTorch(batches_dict_train[0]['configs'], norm_regime, device, TipTop=False, PSFAO=True)
toy.optimizables = []


optimizer = optim.Adam(net.parameters(), lr=0.00001)
# loss_fn = nn.L1Loss(reduction='sum')

for epoch in range(epochs):
    loss_train_average = []
    for batch in batches_dict_train:
        optimizer.zero_grad()
        
        toy.config = batch['configs']
        toy.Update(reinit_grids=True, reinit_pupils=False)
        
        X = batch['X'].to(device)
        PSF_0 = batch['PSF (data)'].to(device)
        
        PSF_pred = toy_run(toy, gnosis2PAO(net(X)))
        loss = loss_fn(PSF_pred, PSF_0.to(device))
        loss.backward()
        optimizer.step()
        
        loss_train_average.append(loss.item()/PSF_0.shape[0])
        print('Current loss:', loss.item()/PSF_0.shape[0], end='\r')
 
    loss_valid_average = []
    with torch.no_grad():
        for batch in batches_dict_valid:
            
            toy.config = batch['configs']
            toy.Update(reinit_grids=True, reinit_pupils=False)

            X = batch['X'].to(device)
            PSF_0 = batch['PSF (data)'].to(device)

            PSF_pred = toy_run(toy, gnosis2PAO(net(X)))
            loss = loss_fn(PSF_pred, PSF_0.to(device))
            
            loss_valid_average.append(loss.item()/PSF_0.shape[0])
            print('Current loss:', loss.item()/PSF_0.shape[0], end='\r')
         
    print('Epoch %d/%d: ' % (epoch+1, epochs))
    print('  Train loss:  %.4f' % (np.array(loss_train_average).mean()))
    print('  Valid. loss: %.4f' % (np.array(loss_valid_average).mean()))
    print('')
    
    torch.cuda.empty_cache()
    
# torch.save(net.state_dict(), WEIGHTS_FOLDER/'gnosis2_weights_tuned.dict')

#%%
PSF_0s = []
PSF_1s = []

destack = lambda PSF_stack: [ x for x in np.split(PSF_stack[:,0,...].detach().cpu().numpy(), PSF_stack.shape[0], axis=0) ]

toy = TipTorch(batches_dict_valid[0]['configs'], norm_regime, device, TipTop=False, PSFAO=True)
toy.optimizables = []

torch.cuda.empty_cache()

for i, batch in tqdm(enumerate(batches_dict_valid)):
    # Test in batched fashion
    batch = batches_dict_valid[i]
    
    # Read data
    PSF_0 = batch['PSF (data)']
    config_file = batch['configs']
    X = batch['X'].to(device)
    Y = batch['Y'].to(device)

    toy.config = config_file
    toy.Update(reinit_grids=True, reinit_pupils=False)

    params_pred = gnosis2PAO(net(X))
    # params_pred = gnosis2PAO(Y)

    PSF_1 = toy_run(toy, params_pred )
    
    PSF_0s.append(PSF_0.detach().cpu().numpy())
    PSF_1s.append(PSF_1.detach().cpu().numpy())

    plot_radial_profiles(destack(PSF_0),  destack(PSF_1),  'Data', 'Predicted', title='NN prediction', dpi=200, cutoff=32, scale='log')
    plt.show()


#%%
def UnitTest_Batch_vs_Individuals():
    toy = TipTorch(batches_dict_valid[0]['configs'], norm_regime, device, TipTop=False, PSFAO=True)
    toy.optimizables = []

    torch.cuda.empty_cache()

    for i, batch in tqdm(enumerate(batches_dict_valid)):
        # Test in batched fashion
        batch = batches_dict_valid[i]
        
        # Read data
        sample_ids = batch['ids']
        X, Y  = batch['X'].to(device), batch['Y'].to(device)
        PSF_0 = batch['PSF (data)']
        config_file = batch['configs']
                
        toy.config = config_file
        toy.Update(reinit_grids=True, reinit_pupils=False)

        PSF_1 = toy_run(toy, gnosis2PAO(Y))

        # Now test individual samples
        PSFs_test = []
        for sample in tqdm(sample_ids):

            PSF_0, _, norms, _, init_config = SPHERE_preprocess([sample], '1P21I', norm_regime, device)
            PSF_0 = PSF_0[...,1:,1:].to(device)
            init_config['sensor_science']['FieldOfView'] = 255
            
            norms = norms[:, None, None].cpu().numpy()

            file = 'E:/ESO/Data/SPHERE/IRDIS_fitted_PAO_1P21I/' + str(sample) + '.pickle'

            with open(file, 'rb') as handle:
                data = pickle.load(handle)

            toy = TipTorch(init_config, norm_regime, device, TipTop=False, PSFAO=True)
            toy.optimizables = []

            tensy = lambda x: torch.tensor(x).to(device)
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

            PSFs_test.append( toy().detach().clone() )

        PSFs_test = torch.stack(PSFs_test).squeeze()
        if PSFs_test.ndim == 3: PSFs_test = PSFs_test.unsqueeze(0)

        destack = lambda PSF_stack: [ x for x in np.split(PSF_stack[:,0,...], PSF_stack.shape[0], axis=0) ]
        plot_radial_profiles(destack(PSF_1),  destack(PSFs_test),  'Stack', 'Singles', title='Fitted TipTop', dpi=200, cutoff=32, scale='log')
        plt.show()


def UnitTest_TransformsBackprop():
    batch = batches_dict_valid[5]
    _, Y  = batch['X'].double().to(device), batch['Y'].double().to(device)

    keys_test = [
        'F (left)',
        'bg (left)',
        'b',
        'dx',
        'dy',
        'r0',
        'amp',
        'beta',
        'alpha',
        'theta',
        'ratio',
        'Jx',
        'Jy',
        'Jxy'
    ]

    vec = torch.rand(Y.shape[-1]).to(device)
    vec.requires_grad = True

    def run_test(Y):  
        dictionary = {}
        for i in range(len(fl_out_keys)):
            dictionary[fl_out_keys[i]] = transforms_output[fl_out_keys[i]].backward(Y[:,i])
        return torch.stack([dictionary[key] for key in fl_out_keys if key in keys_test]).T

    Z_ref = run_test(Y)
    optimizer = optim.LBFGS([vec], lr=10, history_size=20, max_iter=4, line_search_fn="strong_wolfe")
    loss_fn = nn.MSELoss(reduction='mean')

    for _ in range(10):
        optimizer.zero_grad()
        Z_1 = run_test(vec * Y)
        loss = loss_fn(Z_1, Z_ref)
        loss.backward()
        optimizer.step( lambda: loss_fn(Z_ref, run_test(vec*Y)) )
        print('Current loss:', loss.item())


def UnitTest_TipTorch_transform_out():
    torch.cuda.empty_cache()
    def toy_run2(model, dictionary):
            model.F  = dictionary['F (left)'].unsqueeze(-1)
            model.bg = dictionary['bg (left)'].unsqueeze(-1)
            
            for attr in ['Jy','Jxy','Jx','dx','dy','b','r0','amp','beta','theta','alpha','ratio']:
                setattr(model, attr, dictionary[attr])
                
            return model.forward()

    batch = batches_dict_valid[5]
    Y = batch['Y'].double().to(device)

    toy = TipTorch(batch['configs'], norm_regime, device)
    toy.optimizables = []

    PSF_0_ = toy_run2(toy, gnosis2PAO(Y))

    vec = torch.rand(Y.shape[-1]).to(device)
    vec.requires_grad = True

    optimizer = optim.LBFGS([vec], lr=10, history_size=20, max_iter=4, line_search_fn="strong_wolfe")
    loss_fn = nn.L1Loss(reduction='sum')

    for _ in range(27):
        optimizer.zero_grad()
        PSF_1 = toy_run2(toy, gnosis2PAO(vec * Y))
        loss = loss_fn(PSF_1, PSF_0_)
        loss.backward()
        optimizer.step( lambda: loss_fn( PSF_0_, toy_run2(toy, gnosis2PAO(vec * Y)) ))
        print('Current loss:', loss.item(), end='\r')


def UnitTest_TipTorch_and_NN():
    class Agnosis(nn.Module):
        def __init__(self, size):
            super(Agnosis, self).__init__()
            self.fc1 = nn.Linear(size, size*2)
            self.fc2 = nn.Linear(size*2, size)

        def forward(self, x):
            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            return x

    torch.cuda.empty_cache()


    def toy_run2(model, dictionary):
            model.F  = dictionary['F (left)'].unsqueeze(-1)
            model.bg = dictionary['bg (left)'].unsqueeze(-1)
            
            for attr in ['Jy','Jxy','Jx','dx','dy','b','r0','amp','beta','theta','alpha','ratio']:
                setattr(model, attr, dictionary[attr])
                
            return model.forward()

    batch = batches_dict_valid[5]
    sample_ids = batch['ids']
    X = batch['X'].double().to(device)
    Y = batch['Y'].double().to(device)

    net = Agnosis(Y.shape[1])
    net.to(device)
    net.double()

    toy = TipTorch(batch['configs'], norm_regime, device)
    toy.optimizables = []

    PSF_0_ = toy_run2(toy, gnosis2PAO(Y))

    optimizer = optim.Adam(net.parameters(), lr=0.0005)
    loss_fn = nn.L1Loss(reduction='sum')

    for _ in range(100):
        optimizer.zero_grad()
        PSF_1 = toy_run2(toy, gnosis2PAO(net(Y)))
        loss = loss_fn(PSF_1, PSF_0_)
        loss.backward()
        optimizer.step( lambda: loss_fn( PSF_0_, toy_run2(toy, gnosis2PAO(net(Y))) ))
        print('Current loss:', loss.item(), end='\r')
       

# UnitTest_Batch_vs_Individuals()
# UnitTest_TransformsBackprop()
# UnitTest_TipTorch_transform_out()
# UnitTest_TipTorch_and_NN()


#%%

# gnosis2PAO(net(Y)))

#%%
from tools.utils import register_hooks

# PSF_1 = toy_run2(toy, gnosis2PAO(vec * Y))
Y_1 = run_test(gnosis2PAO(vec * Y))

Q = loss_fn(Y_1, Y_ref)
get_dot = register_hooks(Q)
Q.backward()
dot = get_dot()
#dot.save('tmp.dot') # to get .dot
#dot.render('tmp') # to get SVG
dot # in Jupyter, you can just render the variable
