#%%
%reload_ext autoreload
%autoreload 2

import os
import sys
sys.path.insert(0, '..')

import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from tools.utils import plot_radial_profiles_new, LWE_basis, cropper, draw_PSF_stack
from PSF_models.TipToy_SPHERE_multisrc import TipTorch
from data_processing.SPHERE_preproc_utils import SPHERE_preprocess, SamplesByIds
from tools.config_manager import GetSPHEREonsky, ConfigManager
from project_globals import SPHERE_DATASET_FOLDER, device
from data_processing.normalizers import InputsTransformer, CreateTransformSequenceFromFile
from copy import copy, deepcopy

df_transforms_onsky  = CreateTransformSequenceFromFile('../data/temp/psf_df_norm_transforms.pickle')
df_transforms_fitted = CreateTransformSequenceFromFile('../data/temp/fitted_df_norm_transforms.pickle')

# device = torch.device('cpu')

#%%
# Here, all possible transormations are inputed, order and content doesn not matter here
transformer = InputsTransformer({
    'F':  df_transforms_fitted['F L'], 'bg':  df_transforms_fitted['bg L'], # bg R and bg R, as well as F are the same
    'r0': df_transforms_fitted['r0'],  'dx':  df_transforms_fitted['dx'],
    'dy': df_transforms_fitted['dy'],  'Jx':  df_transforms_fitted['Jx'],
    'Jy': df_transforms_fitted['Jy'],  'Jxy': df_transforms_fitted['Jxy'],
    'dn': df_transforms_fitted['dn'],  'basis_coefs': df_transforms_fitted['LWE coefs']
})

# Here, only the variables that need to be predicted must be added, order and content matter
inp_dict = {
    'r0':  torch.ones([1]),
    'F':   torch.ones([1, 2]),
    'dx':  torch.zeros([1]),
    'dy':  torch.zeros([1]),
    'bg':  torch.zeros([1, 2]),
    'dn':  torch.zeros([1]),
    'Jx':  torch.ones([1])*0.5,
    'Jy':  torch.ones([1])*0.5,
    'Jxy': torch.ones([1])*0.1,
    'basis_coefs': torch.zeros([1, 11])
}

fixed_entries  = ['dx', 'dy', 'bg L', 'bg R', 'Jxy'] #, 'LWE coefs']

# Remove 'L' and 'R' endings
fixed_entries_ = copy(fixed_entries)
for i, entry in enumerate(fixed_entries_):
    if entry.endswith(' L'): fixed_entries_[i] = entry.replace(' L', '')
    if entry.endswith(' R'): fixed_entries_[i] = entry.replace(' R', '')
    if entry == 'LWE coefs': fixed_entries_[i] = 'basis_coefs'

fixed_entries_ = list(set(fixed_entries_))

for entry in fixed_entries_:
    _ = inp_dict.pop(entry)

x0 = transformer.stack(inp_dict, no_transform=True) # to create index mapping and initial values

#%%
PSF_data, _, init_config = SPHERE_preprocess(
    sample_ids    = [100],
    norm_regime   = 'sum',
    split_cube    = False,
    PSF_loader    = lambda x: SamplesByIds(x, synth=False),
    config_loader = GetSPHEREonsky,
    framework     = 'pytorch',
    device        = device)

toy = TipTorch(init_config, None, device)
_ = toy()

basis = LWE_basis(toy)

config_manager = ConfigManager(GetSPHEREonsky())

batch_directory = SPHERE_DATASET_FOLDER + 'batch_test.pkl'
with open(batch_directory, 'rb') as handle:
    batch_data = pickle.load(handle)

#%%
# PSF_0
# PSF_var
# bg
# norms
# configs
# IDs
# fitted data
# onsky data
# NN input

class Gnosis(nn.Module):
    def __init__(self, in_size, out_size, hidden_size=100, dropout_p=0.5):
        super(Gnosis, self).__init__()
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
net = Gnosis(batch_data['NN input'].shape[1], transformer.get_packed_size(), 200, 0.25)
net.to(device)
net.float()

net.load_state_dict(torch.load('../data/weights/gnosis_new_weights_v1.dict'))


x0 = x0.float().repeat(len(batch_data['IDs']), 1).to(device)
# x0.requires_grad = True

# I = batch_data['NN input'].to(device).float()

# A = torch.linalg.pinv(I) @ x0
# A += torch.randn([batch_data['NN input'].shape[1], transformer.get_packed_size()], device=device) * 1e-2
# A.requires_grad = True

#%%
def get_fixed_inputs(batch, entries=[]):
    if len(entries) == 0:
        return {}
    
    fixed_inputs = { entry: batch['fitted data'][entry] for entry in entries }

    for items, keys in fixed_inputs.items():
        fixed_inputs[items] = torch.tensor(keys).float().to(device)

    if 'bg L' in fixed_inputs or 'bg R' in fixed_inputs:
        fixed_inputs['bg'] = torch.stack([fixed_inputs['bg L'], fixed_inputs['bg R']], axis=1)

        _ = fixed_inputs.pop('bg L')
        _ = fixed_inputs.pop('bg R')
        
    return fixed_inputs


def run_model(model, batch, predicted_inputs, fixed_inputs={}):
    current_configs = batch['configs']
    config_manager.Convert(current_configs, framework='pytorch', device=device)

    model.config = current_configs
    model.Update(reinit_grids=False, reinit_pupils=False)

    x_unpacked = predicted_inputs | fixed_inputs # It overides the values from the destacked dictionary
    if 'basis_coefs' in x_unpacked:
        return model(x_unpacked, None, lambda: basis(x_unpacked['basis_coefs'].float()))
    elif 'LWE coefs' in x_unpacked:
        return model(x_unpacked, None, lambda: basis(x_unpacked['LWE coefs'].float()))
    else:
        return model(x_unpacked)


def get_data(batch):
    x_0    = batch['NN input'].float().to(device)
    PSF_0  = batch['PSF_0'].float().to(device)
    config = batch['configs']
    fixed_inputs = get_fixed_inputs(batch, fixed_entries)
    config_manager.Convert(config, framework='pytorch', device=device)
    return x_0, fixed_inputs, PSF_0, config


def func(x_, fixed_inputs):
    pred_inputs = transformer.destack(net(x_))
    return run_model(toy, batch_data, pred_inputs, fixed_inputs)

#%%
x0, fixed_inputs, PSF_0, init_config = get_data(batch_data)

toy.config = init_config
toy.Update(reinit_grids=True, reinit_pupils=False)

crop_all    = cropper(PSF_0, 101)
crop_center = cropper(PSF_0, 7)

#%%
'''
Grouping data by wavelengths...
         Count  λ left (nm)  λ right (nm)
λ group                                  
6          842         1625          1625
8          781         1667          1593
10         588         2251          2110
9          488         2182          2182
3          266         1245          1245
4           79         1273          1190
0           30         1043          1043
5           14         1573          1573
1           12         1076          1022
2           11         1213          1213
'''

groups = [6, 8, 10, 9, 3]

batches_train_all, batches_val_all = {}, {}
train_ids_all, val_ids_all = {}, {}
wvls_L_all, wvls_R_all = [], []

for group in groups:
    print(f'\n>>>>> Loading λ-group #{group}')
    train_files = [ SPHERE_DATASET_FOLDER+'train/'+file      for file in os.listdir(SPHERE_DATASET_FOLDER+'train')      if '_grp_'+str(group)+'.pkl' in file ]
    val_files   = [ SPHERE_DATASET_FOLDER+'validation/'+file for file in os.listdir(SPHERE_DATASET_FOLDER+'validation') if '_grp_'+str(group)+'.pkl' in file ]

    batches_train, batches_val = [], []

    print('Loading train batches...')
    for file in tqdm(train_files):
        with open(file, 'rb') as handle:
            batches_train.append( pickle.load(handle) )
            
    print('Loading validation batches...')
    for file in tqdm(val_files):
        with open(file, 'rb') as handle:
            batches_val.append( pickle.load(handle) )

    batch_size = len(batch_data['IDs'])

    train_ids = np.arange(len(batches_train)).tolist()
    val_ids   = np.arange(len(batches_val)).tolist()

    wvl_L = np.array(batches_train[0]['onsky data']['λ left (nm)'])[0]
    wvl_R = np.array(batches_train[0]['onsky data']['λ right (nm)'])[0]

    batches_train_all[wvl_L] = copy(batches_train)
    batches_val_all[wvl_L]   = copy(batches_val)
    train_ids_all[wvl_L]     = copy(train_ids)
    val_ids_all[wvl_L]       = copy(val_ids)
    wvls_L_all.append(wvl_L)
    wvls_R_all.append(wvl_R)

wvls_L_all = np.array(wvls_L_all)
wvls_R_all = np.array(wvls_R_all)

#%%
optimizer = optim.Adam(net.parameters(), lr=0.0001)  # Learning rate is set to 0.1

def loss_fn(A, B):
    return nn.L1Loss(reduction='sum')(A[crop_all], B[crop_all]) + nn.MSELoss(reduction='sum')(A[crop_center], B[crop_center])

epochs = 100

loss_train, loss_val = [], []

net.train()

for epoch in range(epochs):
    print(f'>>>>>>>>> Epoch: {epoch+1}/{epochs}')
    
    epoch_train_loss, epoch_val_loss = [], []
    
    np.random.shuffle(wvls_L_all)
    
    for l, wvl_L in enumerate(wvls_L_all):
        print(f'Wavelengths: {wvl_L}, {wvl_R}')
        train_ids = train_ids_all[wvl_L]
        
        # np.random.shuffle(train_ids)

        for i, id in enumerate(train_ids):
            optimizer.zero_grad()
            
            batch_data = batches_train_all[wvl_L][id]

            x0, fixed_inputs, PSF_0, current_config = get_data(batch_data)
            # toy = TipTorch(current_config, None, device)
            toy.config = current_config
            toy.Update(reinit_grids=True, reinit_pupils=True)
            basis.model = toy

            batch_size = len(batch_data['IDs'])

            loss = loss_fn(func(x0, fixed_inputs), PSF_0)
            loss.backward()#retain_graph=True)
            optimizer.step()

            loss_train.append(loss.item()/batch_size)
            epoch_train_loss.append(loss.item()/batch_size)
            
            print(f'Running loss ({i+1}/{len(train_ids)}): {loss.item()/batch_size:.4f}', end="\r", flush=True)
        
    print(f'Epoch: {epoch+1}/{epochs}, train loss: {np.array(epoch_train_loss).mean().item()}')
    
    
    for l, wvl_L in enumerate(wvls_L_all[4:]):
        print(f'Wavelengths: {wvl_L}, {wvl_R}')
        val_ids = val_ids_all[wvl_L]
        np.random.shuffle(val_ids)
        
        for i, id in enumerate(val_ids):
            with torch.no_grad():
                batch_data = batches_val_all[wvl_L][id]
                
                x0, fixed_inputs, PSF_0, current_config = get_data(batch_data)
                # toy = TipTorch(current_config, None, device)
                toy.config = current_config
                toy.Update(reinit_grids=True, reinit_pupils=True)
                basis.model = toy

                batch_size = len(batch_data['IDs'])

                loss = loss_fn(func(x0, fixed_inputs), PSF_0) 
                loss_val.append(loss.item()/batch_size)
                epoch_val_loss.append(loss.item()/batch_size)
                
                print(f'Running loss ({i+1}/{len(val_ids)}): {loss.item()/batch_size:.4f}', end="\r", flush=True)

    print(f'Epoch: {epoch+1}/{epochs}, validation loss: {np.array(epoch_val_loss).mean().item()}\n')

# Save weights
# torch.save(net.state_dict(), '../data/weights/gnosis_new_weights_v2.dict')

#%%
PSFs_0_val_all = {}
PSFs_1_val_all = {}
PSFs_2_val_all = {}

net.eval()
with torch.no_grad():
    for wvl in wvls_L_all:
        PSFs_0_val, PSFs_1_val, PSFs_2_val = [], [], []
        val_ids = val_ids_all[wvl]

        for i in tqdm(val_ids):
            # ------------------------- Validate predicted -------------------------
            batch_data = batches_val_all[wvl][i]
            
            x0, fixed_inputs, PSF_0, init_config = get_data(batch_data)
            toy.config = init_config
            toy.Update(reinit_grids=True, reinit_pupils=True)

            batch_size = len(batch_data['IDs'])
            
            fixed_inputs['Jxy'] *= 0
            
            PSFs_0_val.append(PSF_0.cpu())
            PSFs_1_val.append(func(x0, fixed_inputs).cpu())

            # ------------------------- Validate direct -------------------------
            inputs = {
                'F':   torch.ones([1, 2]),
                'Jx':  torch.ones([1])*33.0,
                'Jy':  torch.ones([1])*33.0,
                'Jxy': torch.zeros([1]),
                'dn':  torch.zeros([1]),
                'basis_coefs': torch.zeros([1, 11])
            }
            
            current_batch_size = len(batch_data['IDs'])

            for key, value in inputs.items():
                inputs[key] = value.float().to(device).repeat(current_batch_size, 1).squeeze()
            
            PSFs_2_val.append(run_model(toy, batch_data, inputs).cpu())

        PSFs_0_val = torch.cat(PSFs_0_val, dim=0)[:,0,...].numpy()
        PSFs_1_val = torch.cat(PSFs_1_val, dim=0)[:,0,...].numpy()
        PSFs_2_val = torch.cat(PSFs_2_val, dim=0)[:,0,...].numpy()

        PSFs_0_val_all[wvl] = PSFs_0_val.copy()
        PSFs_1_val_all[wvl] = PSFs_1_val.copy()
        PSFs_2_val_all[wvl] = PSFs_2_val.copy()

        fig, ax = plt.subplots(1, 2, figsize=(10, 3))
        plot_radial_profiles_new(PSFs_0_val, PSFs_2_val, 'Data', 'TipTorch', title='Direct prediction',     ax=ax[0])
        plot_radial_profiles_new(PSFs_0_val, PSFs_1_val, 'Data', 'TipTorch', title='Calibrated prediction', ax=ax[1])
        fig.suptitle(f'λ = {wvl} [nm]')
        # plt.savefig(f'C:/Users/akuznets/Desktop/presa_buf/PSF_validation_{wvl}.png', dpi=200)


#%%
PSF_cube_data   = torch.tensor(PSFs_0_val_all[wvl]).unsqueeze(1)
PSF_cube_calib  = torch.tensor(PSFs_1_val_all[wvl]).unsqueeze(1)
PSF_cube_direct = torch.tensor(PSFs_2_val_all[wvl]).unsqueeze(1)

draw_PSF_stack(PSF_cube_data, PSF_cube_direct, average=True, crop=40)
plt.show()
draw_PSF_stack(PSF_cube_data, PSF_cube_calib,  average=True, crop=40)
plt.show()


#%% ==========================================================================================
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
