#%%
%reload_ext autoreload
%autoreload 2

import sys
sys.path.append('..')

import os
import torch
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from data_processing.normalizers import Uniform, TransformSequence
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from project_globals import SPHERE_DATA_FOLDER, DATA_FOLDER, device

from tqdm import tqdm
from tools.utils import seeing, plot_radial_profiles, plot_radial_profiles_new
from data_processing.SPHERE_preproc_utils import SPHERE_preprocess
from PSF_models.TipToy_SPHERE_multisrc import TipTorch

from project_globals import WEIGHTS_FOLDER

#%% Initialize data sample
with open('E:/ESO/Data/SPHERE/synth_df.pickle', 'rb') as handle:
    synth_df = pickle.load(handle)

with open('E:/ESO/Data/SPHERE/synth_fitted_df.pickle', 'rb') as handle:
    synth_fitted_df = pickle.load(handle)

synth_df = synth_df[synth_df['invalid'] == False]
synth_df = synth_df[synth_df['WFS noise (nm)'] < 275]
synth_df.drop(columns=['Jitter X'], inplace=True)
synth_df.drop(columns=['Jitter Y'], inplace=True)

synth_fitted_df            = synth_fitted_df[synth_fitted_df['F (left)'] > 0.87]
synth_fitted_df['SR fit']  = 0.5 * (synth_fitted_df['SR fit (left)'] + synth_fitted_df['SR fit (right)'])
synth_fitted_df['SR data'] = 0.5 * (synth_fitted_df['SR data (left)'] + synth_fitted_df['SR data (right)'])
synth_fitted_df['J']       = np.sqrt(synth_fitted_df['Jx'].pow(2) + synth_fitted_df['Jy'].pow(2))
synth_fitted_df['J init']  = np.sqrt(synth_fitted_df['Jx init'].pow(2) + synth_fitted_df['Jy init'].pow(2))
synth_fitted_df['F']       = 0.5 * (synth_fitted_df['F (left)'] + synth_fitted_df['F (right)'])
synth_fitted_df['Jx']      = synth_fitted_df['Jx'].abs()
synth_fitted_df['Jy']      = synth_fitted_df['Jy'].abs()

synth_df.rename(columns={'r0': 'r0 (data)'}, inplace=True)
synth_df.rename(columns={'Nph WFS': 'Nph WFS (data)'}, inplace=True)

synth_fitted_df.rename(columns={'r0': 'r0 (fit)'}, inplace=True)
synth_fitted_df.rename(columns={'Nph WFS': 'Nph WFS (fit)'}, inplace=True)

synth_fitted_df['dn'] = \
    (synth_fitted_df['n'] + synth_fitted_df['dn']).abs() - synth_fitted_df['n']

synth_fitted_df['dn'] = synth_fitted_df['dn'].abs()
synth_fitted_df['n']  = synth_fitted_df['n'].abs()
synth_fitted_df['Rec. noise'] = np.sqrt((synth_fitted_df['n']+synth_fitted_df['dn']).abs())*synth_df['λ WFS (nm)']/2/np.pi * 1e9

ids = synth_df.index.intersection(synth_fitted_df.index)
synth_fitted_df = synth_fitted_df.loc[ids]
synth_df = synth_df.loc[ids]

df = pd.concat([synth_df, synth_fitted_df], axis=1).fillna(0)

def corr_plot(data, entry_x, entry_y, lims=None):
    j = sns.jointplot(data=data, x=entry_x, y=entry_y, kind="kde", space=0, alpha = 0.3, fill=True, colormap='royalblue' )
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

#%% =========================================================================
'''
x = 'WFS noise (nm)'
y = 'Rec. noise'
z = 'Rate'
test = pd.DataFrame({ x: synth_df[x], y: synth_fitted_df[y], z: synth_df[z], })
sns.scatterplot(data=test, x=x, y=y, hue=z, cmap='viridis')
plt.ylim([0, 750])
plt.xlim([0, 250])
plt.ylabel(y, color='royalblue')
plt.grid(True)

#%
x = 'Jx init'
y = 'Jx'
z = 'Rate'
test = { x: synth_fitted_df[x], y: synth_fitted_df[y], z: synth_df[z], }
sns.scatterplot(test, x=x, y=y, hue=z, cmap='viridis')
plt.xlim([0, 35])
plt.ylim([10, 40])
plt.ylabel(y, color='royalblue')
plt.grid(True)

#%
x = 'WFS noise (nm)'
y = 'Jx'
z = 'Rate'
test = { x: synth_df[x], y: synth_fitted_df[y], z: synth_df[z], }
sns.scatterplot(test, x=x, y=y, hue=z, cmap='viridis')
plt.xlim([0, 250])
plt.ylim([10, 35])
plt.ylabel(y, color='royalblue')
plt.grid(True)

#%
x = 'WFS noise (nm)'
y = 'F (left)'
z = 'Rate'
test = { x: synth_df[x], y: synth_fitted_df[y], z: synth_df[z], }
sns.scatterplot(test, x=x, y=y, hue=z, cmap='viridis')
'''
#%
transforms_input_synth = {}
transforms_input_synth['Airmass']                   = TransformSequence( transforms = [ Uniform(a=1.0, b=1.5) ])
transforms_input_synth['Strehl (IR)']               = TransformSequence( transforms = [ Uniform(a=0,   b=1)] )
transforms_input_synth['r0 (data)']                 = TransformSequence( transforms = [ Uniform(a=0.1, b=0.5)] )
transforms_input_synth['Rate']                      = TransformSequence( transforms = [ Uniform(a=0,   b=1380)] )
transforms_input_synth['Nph WFS (data)']            = TransformSequence( transforms = [ Uniform(a=0,   b=200)] )
transforms_input_synth['Wind speed']                = TransformSequence( transforms = [ Uniform(a=0,   b=20)] )
transforms_input_synth['Wind direction']            = TransformSequence( transforms = [ Uniform(a=0,   b=360)] )
transforms_input_synth['WFS noise (nm)']            = TransformSequence( transforms = [ Uniform(a=0,   b=220)] )
transforms_input_synth['Flux WFS']                  = TransformSequence( transforms = [ Uniform(a=0,   b=2000)] )
transforms_input_synth['Wind speed (200 mbar)'   ]  = TransformSequence( transforms = [ Uniform(a=0,   b=70)] )
transforms_input_synth['Wind direction (200 mbar)'] = TransformSequence( transforms = [ Uniform(a=0,   b=360)] ) 
# transforms_input_synth['Jitter X']                  = TransformSequence( transforms = [ Uniform(a=0,   b=30)] )
# transforms_input_synth['Jitter Y']                  = TransformSequence( transforms = [ Uniform(a=0,   b=30)] )

transforms_output_synth = {}
transforms_output_synth['dx']              = TransformSequence( transforms = [ Uniform(a=-0.5, b=0.5) ] )
transforms_output_synth['dy']              = TransformSequence( transforms = [ Uniform(a=-0.5, b=0.5) ] )
transforms_output_synth['r0 (fit)']        = TransformSequence( transforms = [ Uniform(a=0.0,  b=1.0) ] )
transforms_output_synth['n']               = TransformSequence( transforms = [ Uniform(a=0.0,  b=17.0 ) ] )
transforms_output_synth['dn']              = TransformSequence( transforms = [ Uniform(a=0, b=50) ] )
transforms_output_synth['Rec. noise']      = TransformSequence( transforms = [ Uniform(a=0, b=1250) ] )
transforms_output_synth['J']               = TransformSequence( transforms = [ Uniform(a=0, b=40) ] )
transforms_output_synth['Jx']              = TransformSequence( transforms = [ Uniform(a=0, b=40) ] )
transforms_output_synth['Jy']              = TransformSequence( transforms = [ Uniform(a=0, b=40) ] )
transforms_output_synth['Jxy']             = TransformSequence( transforms = [ Uniform(a=0, b=300) ] )
transforms_output_synth['Nph WFS (fit)']   = TransformSequence( transforms = [ Uniform(a=0, b=300) ] )
transforms_output_synth['J init']          = TransformSequence( transforms = [ Uniform(a=0, b=30) ] )
transforms_output_synth['Jx init']         = TransformSequence( transforms = [ Uniform(a=0, b=30) ] )
transforms_output_synth['Jy init']         = TransformSequence( transforms = [ Uniform(a=0, b=30) ] )
transforms_output_synth['F']               = TransformSequence( transforms = [ Uniform(a=0, b=1.5) ] )
transforms_output_synth['F (left)']        = TransformSequence( transforms = [ Uniform(a=0, b=1.5) ] )
transforms_output_synth['F (right)']       = TransformSequence( transforms = [ Uniform(a=0, b=1.5) ] )
transforms_output_synth['bg (left)']       = TransformSequence( transforms = [ Uniform(a=-1e-5, b=1e-5) ] )
transforms_output_synth['bg (right)']      = TransformSequence( transforms = [ Uniform(a=-1e-5, b=1e-5) ] )
transforms_output_synth['SR data']         = TransformSequence( transforms = [ Uniform(a=0.0, b=1.0) ] )
transforms_output_synth['SR fit']          = TransformSequence( transforms = [ Uniform(a=0.0, b=1.0) ] )
transforms_output_synth['SR data (left)']  = TransformSequence( transforms = [ Uniform(a=0.0, b=1.0) ] )
transforms_output_synth['SR data (right)'] = TransformSequence( transforms = [ Uniform(a=0.0, b=1.0) ] )
transforms_output_synth['SR fit (left)']   = TransformSequence( transforms = [ Uniform(a=0.0, b=1.0) ] )
transforms_output_synth['SR fit (right)']  = TransformSequence( transforms = [ Uniform(a=0.0, b=1.0) ] )

transforms = {**transforms_input_synth, **transforms_output_synth}

trans_df = df.copy()
for entry in transforms:
    trans_df[entry] = transforms[entry].forward(trans_df[entry])


#%%
selected_X = [
    # 'Strehl (IR)',
    # 'Airmass',
    'r0 (data)',
    'Rate',
    'Nph WFS (data)',
    'Wind speed',
    'Wind direction',
    # 'Flux WFS',
    'Wind speed (200 mbar)',
    'Wind direction (200 mbar)',
    # 'WFS noise (nm)',
    # 'Jx init',
    # 'Jy init',
]

selected_Y = [
    'Rec. noise',
    'Jx',
    'Jy',
    'F (left)',
    'F (right)',
]

X_df = trans_df[selected_X].fillna(0)
Y_df = trans_df[selected_Y].fillna(0)


# for entry in selected_X:
#     sns.displot(X_df[entry], kde=True)
#     plt.show()

'''
%matplotlib qt

merged_df = pd.concat([X_df, Y_df], axis=1)

vars = selected_entries_X + selected_entries_Y
vars.remove('Jy')
vars.remove('F (right)')
vars.remove('Jitter Y')
vars.remove('Rate')
vars.remove('Flux WFS')

sns.pairplot(data = merged_df,
             vars = vars,
             corner = True,
             hue = 'Rate')
    # 'FWHM'], hue='λ left (nm)', kind='kde')
'''

# Remove the outliers
for entry in selected_X:
    X_df = X_df[X_df[entry].abs() < 3]

for entry in selected_Y:
    Y_df = Y_df[Y_df[entry].abs() < 3]

#%
# intersect ids
ids = X_df.index.intersection(Y_df.index)
X_df = X_df.loc[ids]
Y_df = Y_df.loc[ids]

X_df_train, X_df_test, y_df_train, y_df_test = train_test_split(X_df, Y_df, test_size=0.2, random_state=42)

# mlp = MLPRegressor(hidden_layer_sizes=(100,100,100,), max_iter=3000, random_state=42)
# mlp.fit(X_df_train, y_df_train)
# y_pred = mlp.predict(X_df_test)


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

weights_name = WEIGHTS_FOLDER/f'param_predictor_{len(selected_X)}x{len(selected_Y)}.pth'

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

#%%
synth_fitted_df_pred = y_pred_df.copy()
for entry in selected_Y:
    synth_fitted_df_pred[entry] = transforms_output_synth[entry].backward(synth_fitted_df_pred[entry])

synth_fitted_df_test = y_df_test.copy()
for entry in selected_Y:
    synth_fitted_df_test[entry] = transforms_output_synth[entry].backward(synth_fitted_df_test[entry])

error_df = (synth_fitted_df_test-synth_fitted_df_pred).abs()
error_df.sort_index(inplace=True)

error_df['F (left)'] *= 100
# error_df['F (right)'] *= 100

summary_df = pd.DataFrame({
    'Mean':    error_df.mean(),
    'STD':     error_df.std(),
    'Median':  error_df.median(),
    'Maximum': error_df.max(),
    'Minimum': error_df.min(),
}).transpose()

synth_fitted_df['Rec. noise'] = np.sqrt((synth_fitted_df['n'] + synth_fitted_df['dn']).abs()) * synth_df['λ WFS (nm)']/2/np.pi * 1e9
synth_fitted_df_pred.rename(columns={'Jx': 'Jx pred'}, inplace=True)
synth_fitted_df_test.rename(columns={'Jx': 'Jx test'}, inplace=True)
synth_fitted_df_pred.rename(columns={'Rec. noise': 'Rec. noise pred'}, inplace=True)
synth_fitted_df_test.rename(columns={'Rec. noise': 'Rec. noise test'}, inplace=True)
df_merge = pd.concat([synth_fitted_df_test, synth_fitted_df_pred], axis=1)

corr_plot(df_merge, 'Jx test', 'Jx pred')
# corr_plot(df_merge,'Rec. noise pred', 'Rec. noise test')

#%%
NN2in  = lambda X: { selected: transforms[selected].backward(X[:,i]) for i,selected in enumerate(selected_X) }
NN2fit = lambda Y: { selected: transforms[selected].backward(Y[:,i]) for i,selected in enumerate(selected_Y) }
in2NN  = lambda inp: torch.from_numpy(( np.stack([transforms[a].forward(inp[a].values) for a in selected_X]))).T
fit2NN = lambda out: torch.from_numpy(( np.stack([transforms[a].forward(out[a].values) for a in selected_Y]))).T

def toy_run(model, data, pred):
    conv = lambda x: torch.from_numpy(np.array(x)).to(device).float()
    model.F  = torch.vstack([pred['F (left)'], pred['F (right)' ]]).T
    model.bg = torch.vstack([conv(data['bg (left)']), conv(data['bg (right)'])]).T

    for attr in ['Jy', 'Jx']:
        setattr(model, attr, pred[attr])
    for attr in ['dx', 'dy', 'r0 (data)', 'Jxy']:
        setattr(model, attr, conv(data[attr]))
        
    model.WFS_Nph = conv(data['Nph WFS (data)'])
    inv_a2 = conv( 1 / (data['λ WFS (nm)']*1e9/2/np.pi)**2 )
    model.dn = inv_a2 * pred['Rec. noise']**2 - conv(data['n'])
    
    return model.forward()


def toy_run_direct(model, data):
    conv = lambda x: torch.from_numpy(np.array(x)).to(device).float()
    model.F  = torch.tensor([1.0, 1.0]).to(device).float()
    model.bg = torch.tensor([0.0, 0.0]).to(device).float()

    model.Jx = conv(data['Jx init'])
    model.Jy = conv(data['Jy init'])
    
    for attr in ['dx', 'dy', 'r0 (data)', 'Jxy']:
        setattr(model, attr, conv(data[attr]))
        
    model.WFS_Nph = conv(data['Nph WFS (data)'])
    model.dn = torch.tensor([0.0]).to(device).float()
    
    return model.forward()
    

def prepare_batch_configs(batches):
    batches_dict = []
    for i in tqdm(range(len(batches))):
        sample_ids = batches[i].index.tolist()
        PSF_0, _, _, _, config_files = SPHERE_preprocess(sample_ids, 'different', 'sum', device, synth=True)

        batch_dict = {
            'df': batches[i],
            'ids': sample_ids,
            'PSF (data)': PSF_0,
            'configs': config_files,
            'X': in2NN ( batches[i].loc[sample_ids] ),
            'Y': fit2NN( batches[i].loc[sample_ids] )
        }
        batches_dict.append(batch_dict)
    return batches_dict

batch_test = prepare_batch_configs([df.loc[0:256]])

#%%
toy = TipTorch(batch_test[0]['configs'], 'sum', device, TipTop=True, PSFAO=False)
toy.optimizables = []

# y_pred = torch.from_numpy(mlp.predict(batch_test[0]['X'])).float()
y_pred = net(batch_test[0]['X'].float().to(device))

PSF_0 = batch_test[0]['PSF (data)'].cpu().numpy()
PSF_2 = toy_run( toy, batch_test[0]['df'], NN2fit(y_pred.to(device)) ).cpu().detach().numpy()

destack = lambda PSF_stack: [ x for x in np.split(PSF_stack[:,0,...], PSF_stack.shape[0], axis=0) ]
# plot_radial_profiles(destack(PSF_0),  destack(PSF_2), 'Data', 'Predicted', title='Synth prediction accuracy worked', dpi=200, cutoff=32, scale='log')
plot_radial_profiles_new(PSF_0[:,0,...],  PSF_2[:,0,...], 'Data', 'Predicted', title='Synth prediction accuracy worked', dpi=200, cutoff=32, scale='lin')
# plt.savefig('C:/Users/akuznets/Desktop/AO4ELT/OOPAOfitted_lin.pdf')

#%%
toy = TipTorch(batch_test[0]['configs'], 'sum', device, TipTop=True, PSFAO=False)
toy.optimizables = []

# y_pred = torch.from_numpy(mlp.predict(batch_test[0]['X'])).float()
y_pred = net(batch_test[0]['X'].float().to(device))

PSF_0 = batch_test[0]['PSF (data)'].cpu().numpy()
PSF_2 = toy_run_direct( toy, batch_test[0]['df'] ).cpu().detach().numpy()

destack = lambda PSF_stack: [ x for x in np.split(PSF_stack[:,0,...], PSF_stack.shape[0], axis=0) ]
plot_radial_profiles_new(PSF_0[:,0,...],  PSF_2[:,0,...], 'Data', 'Predicted', title='Synth prediction accuracy worked', dpi=200, cutoff=32)

# plt.savefig('C:/Users/akuznets/Desktop/AO4ELT/OOPAOdirect_log.pdf')
'''
configs_path = 'C:/Users/akuznets/Data/SPHERE/simulated/configs_onsky/'
configs = [int(f.split('.')[0]) for f in os.listdir(configs_path)]

samples = [5555, 5556, 5557]

PSFs_1_test = []
PSFs_1 = []
PSFs_0 = []

for sample in samples:
    PSF_0, _, norms, _, init_config = SPHERE_preprocess([sample], 'different', 'sum', device, synth=True)
    norms = norms[:, None, None].cpu().numpy()

    fitted_folder = SPHERE_DATA_FOLDER + 'IRDIS_fitted_OOPAO/'
    file = fitted_folder + str(sample) + '.pickle'

    with open(file, 'rb') as handle:
        data = pickle.load(handle)

    toy = TipTorch(init_config, norm_regime, device, TipTop=True, PSFAO=False)
    toy.optimizables = []

    tensy = lambda x: torch.tensor(x).to(device)
    toy.F     = tensy( data['F']   ).squeeze()
    toy.bg    = tensy( data['bg']  ).squeeze()
    toy.Jy    = tensy( data['Jy']  ).flatten()
    toy.Jxy   = tensy( data['Jxy'] ).flatten()
    toy.Jx    = tensy( data['Jx']  ).flatten()
    toy.dx    = tensy( data['dx']  ).flatten()
    toy.dy    = tensy( data['dy']  ).flatten()
    toy.dn    = tensy( data['dn']  ).flatten()
    toy.r0    = tensy( data['r0']  ).flatten()

    PSF_1 = toy().detach().clone()
        
    images_data   = data['Img. data'] / norms[None,:]
    images_fitted = data['Img. fit']  / norms[None,:]

    PSFs_0.append(PSF_0.cpu().numpy())
    PSFs_1.append(PSF_1.cpu().numpy())
    PSFs_1_test.append(images_fitted)

PSFs_0 = np.vstack(PSFs_0)
PSFs_1 = np.vstack(PSFs_1)
PSFs_1_test = np.vstack(PSFs_1_test)

destack = lambda PSF_stack: [ x for x in np.split(PSF_stack[:,0,...], PSF_stack.shape[0], axis=0) ]
plot_radial_profiles(destack(PSFs_0),  destack(PSFs_1_test),  'Data', 'Fitted', title='AAAAA', dpi=200, cutoff=32, scale='log')

x = synth_fitted_df.loc[samples]
pred_0 = fit2NN(x)

test = pd.DataFrame(NN2fit(pred_0), index=x.index)
x = x[test.columns]

#%
PSF_0, _, norms, _, init_config = SPHERE_preprocess(samples, 'different', 'sum', device, synth=True)
toy = TipTorch(init_config, norm_regime, device, TipTop=True, PSFAO=False)
toy.optimizables = []

PSF_1_1 = toy_run( toy, df.loc[samples], NN2fit(pred_0.to(device)) )

destack = lambda PSF_stack: [ x for x in np.split(PSF_stack[:,0,...], PSF_stack.shape[0], axis=0) ]
plot_radial_profiles(destack(PSFs_0),  destack(PSF_1_1),  'Data', 'Fitted', title='AAAAA', dpi=200, cutoff=32, scale='log')

'''

# %%
