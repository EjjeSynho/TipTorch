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
from project_globals import SPHERE_DATASET_FOLDER, SPHERE_DATA_FOLDER, device
from data_processing.normalizers import InputsTransformer, CreateTransformSequenceFromFile
from copy import copy, deepcopy

df_transforms_onsky  = CreateTransformSequenceFromFile('../data/temp/psf_df_norm_transforms.pickle')
df_transforms_fitted = CreateTransformSequenceFromFile('../data/temp/fitted_df_norm_transforms.pickle')

with open('../data/temp/fitted_df_norm.pickle', 'rb') as handle:
    fitted_df_norm = pickle.load(handle)

with open('../data/temp/psf_df_norm.pickle', 'rb') as handle:
    psf_df_norm = pickle.load(handle)
    

#%%
# Here, all possible transormations are inputed, order and content doesn not matter here
transformer = InputsTransformer({
    'F':   df_transforms_fitted['F L'],
    'bg':  df_transforms_fitted['bg L'], # bg R and bg L, as well as dx,dy, and F and are the same
    'r0':  df_transforms_fitted['r0'],
    'dx':  df_transforms_fitted['dx L'],
    'dy':  df_transforms_fitted['dy L'], 
    'Jx':  df_transforms_fitted['Jx'],
    'Jy':  df_transforms_fitted['Jy'],
    'Jxy': df_transforms_fitted['Jxy'],
    'dn':  df_transforms_fitted['dn'],
    'basis_coefs': df_transforms_fitted['LWE coefs'],
    'wind_dir':    df_transforms_fitted['Wind dir'],
    'wind_speed':  df_transforms_fitted['Wind speed']
})

# Here, only the variables that need to be predicted must be added, order and content matter
inp_dict = {
    'r0':  torch.ones([1]),
    'F':   torch.ones([1, 2]),
    'dx':  torch.zeros([1, 2]),
    'dy':  torch.zeros([1, 2]),
    'bg':  torch.zeros([1, 2]),
    'dn':  torch.zeros([1]),
    'Jx':  torch.ones([1])*0.5,
    'Jy':  torch.ones([1])*0.5,
    'Jxy': torch.ones([1])*0.1,
    'basis_coefs': torch.zeros([1, 12])
}

fixed_entries  = ['dx L', 'dx R', 'dy L', 'dy R', 'bg L', 'bg R', 'Jxy', 'LWE coefs']

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
    def __init__(self, in_size, out_size, hidden_size=100, dropout_p=0.25):
        super(Gnosis, self).__init__()
        self.fc1 = nn.Linear(in_size, hidden_size)
        # self.dropout1 = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # self.dropout2 = nn.Dropout(dropout_p)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        # self.fc3 = nn.Linear(hidden_size, hidden_size*2)
        self.dropout3 = nn.Dropout(dropout_p)
        # self.fc4 = nn.Linear(hidden_size*2, out_size)
        self.fc4 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        # x = self.dropout1(x)
        x = torch.tanh(self.fc2(x))
        # x = self.dropout2(x)
        x = torch.tanh(self.fc3(x))
        x = self.dropout3(x)
        x = torch.tanh(self.fc4(x))
        return x
    
# Initialize the network, loss function and optimizer
net = Gnosis(batch_data['NN input'].shape[1], transformer.get_packed_size(), 200, 0.1)#0.25)
net.to(device)
net.float()

# net.load_state_dict(torch.load('../data/weights/gnosis_new_weights_v3.dict'))
net.load_state_dict(torch.load('../data/weights/gnosis_new_weights_noLWE_v1.dict'))

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

    entries_to_remove = []
    dict_to_add = {}

    for entry, value in fixed_inputs.items():
        if entry.endswith(' L'):
            entry_ = entry.replace(' L', '')
            
            dict_to_add[entry_] = torch.stack([
                torch.tensor(np.array(fixed_inputs[entry_+' L'])).float().to(device),
                torch.tensor(np.array(fixed_inputs[entry_+' R'])).float().to(device)
            ], axis=1)
            
            entries_to_remove.append(entry_+' L')
            entries_to_remove.append(entry_+' R')

        elif entry == 'LWE coefs':
            dict_to_add['basis_coefs'] = torch.tensor(np.array(fixed_inputs[entry])).float().to(device)
            entries_to_remove.append('LWE coefs')
            
        elif entry == 'Wind dir':
            dict_to_add['wind_dir'] = torch.tensor(np.array(fixed_inputs[entry])).float().to(device)
            entries_to_remove.append('Wind dir')   
                    
        else:
            fixed_inputs[entry] = torch.tensor(np.array(value)).float().to(device)
            
    for entry in entries_to_remove: _ = fixed_inputs.pop(entry)
    
    return fixed_inputs | dict_to_add


def run_model(model, batch, predicted_inputs, fixed_inputs={}):
    current_configs = batch['configs']
    config_manager.Convert(current_configs, framework='pytorch', device=device)

    model.config = current_configs
    model.Update(reinit_grids=False, reinit_pupils=False)

    x_unpacked = predicted_inputs | fixed_inputs # It overides the values from the destacked dictionary
    if 'basis_coefs' in x_unpacked:
        return model(x_unpacked, None, lambda: basis(x_unpacked['basis_coefs'].float()))
    else:
        return model(x_unpacked)


def get_data(batch, fixed_entries):
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
x0, fixed_inputs, PSF_0, init_config = get_data(batch_data, fixed_entries)

toy.config = init_config
toy.Update(reinit_grids=True, reinit_pupils=False)

crop_all    = cropper(PSF_0, 91)
crop_center = cropper(PSF_0, 7)

#%%
with open(SPHERE_DATA_FOLDER + 'fitted_and_data_PSF_profiles.pickle', 'rb') as handle:
    profiles_L, profiles_R = pickle.load(handle)

with open(SPHERE_DATA_FOLDER + 'images_df.pickle', 'rb') as handle:
    images = pickle.load(handle)

profs_L = np.stack([profiles_L[id_test][0] for id_test in batch_data['IDs']], axis=0).squeeze()
profs_R = np.stack([profiles_R[id_test][0] for id_test in batch_data['IDs']], axis=0).squeeze()

PSF_0 = batch_data['PSF_0'].cpu().numpy()

fitted_entries = ['F L', 'F R', 'bg L', 'bg R', 'dx L', 'dx R', 'dy L', 'dy R', 'Wind dir', 'r0', 'Jx', 'Jy', 'Jxy', 'dn', 'LWE coefs']

fitted_dict = get_fixed_inputs(batch_data, fitted_entries)

PSF_1 = run_model(toy, batch_data, {}, fixed_inputs=fitted_dict).cpu().numpy()

p_0, p_1, p_err = plot_radial_profiles_new(
    PSF_0[:,0,...], PSF_1[:,0,...], return_profiles=True, suppress_plot=True
)    

#%%
from tools.utils import render_profile

fig = plt.figure(figsize=(6, 4), dpi=300)
ax  = fig.gca()

render_profile(profs_L, 'tab:blue', 'El datto', linestyle='-',  linewidth=1, ax=ax)
render_profile(p_0, 'tab:orange', 'El fitto', linestyle='-',  linewidth=1, ax=ax)
render_profile(np.abs(profs_L-p_0), 'tab:green',  'El error', linestyle='--', linewidth=1, ax=ax)

y_lim = max([profs_L.max(), p_0.max()])

ax.set_yscale('symlog', linthresh=5e-1)
ax.set_ylim(1e-2, y_lim)
ax.set_xlim(0, 20)

ax.set_xlabel('Pixels from on-axis, [pix]')
ax.set_ylabel('Normalized intensity, [%]')
ax.grid()
plt.show()


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

#%% =============================== Monochromatic training ==================================
A = 50

pattern_pos = torch.tensor([[0,0,0,0,  0,-1,1,0,  1,0,0,-1]]).to(device).float() * A
pattern_neg = torch.tensor([[0,0,0,0,  0,1,-1,0, -1,0,0, 1]]).to(device).float() * A
pattern_1   = torch.tensor([[0,0,0,0,  0,-1,1,0, -1,0,0, 1]]).to(device).float() * A
pattern_2   = torch.tensor([[0,0,0,0,  0,1,-1,0,  1,0,0,-1]]).to(device).float() * A
pattern_3   = torch.tensor([[0,0,0,0,  1,0,0,-1,  0,1,-1,0]]).to(device).float() * A
pattern_4   = torch.tensor([[0,0,0,0,  -1,0,0,1,  0,-1,1,0]]).to(device).float() * A

gauss_penalty = lambda A, x, x_0, sigma: A * torch.exp(-torch.sum((x - x_0) ** 2) / (2 * sigma ** 2))
img_punish    = lambda A, B: (A-B)[crop_all].flatten().abs().sum()
center_punish = lambda A, B: (A-B)[crop_center].flatten().abs().sum()
Gauss_err     = lambda pattern, coefs: (pattern * gauss_penalty(5, coefs, pattern, A/2)).flatten().abs().sum()
coefs_compare = lambda A,B: (A-B).pow(2).sum().sqrt() / A.numel()

LWE_regularizer = lambda c: \
    Gauss_err(pattern_pos, c) + Gauss_err(pattern_neg, c) + \
    Gauss_err(pattern_1, c)   + Gauss_err(pattern_2, c)   + \
    Gauss_err(pattern_3, c)   + Gauss_err(pattern_4, c)

def loss_fn(x_, PSF_data):
    pred_inputs = transformer.destack(net(x_))
    PSF_pred = run_model(toy, batch_data, pred_inputs, fixed_inputs)
    # coefs_ = transformer.destack(x_)['basis_coefs']
    # coefs_gt = torch.tensor(np.array(batch_data['fitted data']['LWE coefs']), device=device).float()
    # loss = img_punish(PSF_pred, PSF_data) + LWE_regularizer(coefs_) + (coefs_**2).mean()*1e-4 + center_punish(PSD_pred, PSF_data)
    # loss = img_punish(PSF_pred, PSF_data) + LWE_regularizer(coefs_) + (coefs_**2).mean()*1e-4
    # loss = img_punish(PSF_pred, PSF_data) + LWE_regularizer(coefs_) + (coefs_**2).mean()*1e-4 + coefs_compare(coefs_, coefs_gt)*0.25
    # loss = img_punish(PSF_pred, PSF_data) + coefs_compare(coefs_, coefs_gt)*0.75
    loss = img_punish(PSF_pred, PSF_data)
    return loss

# def loss_fn(A, B):
#     return nn.L1Loss(reduction='sum')(A[crop_all], B[crop_all]) + nn.MSELoss(reduction='sum')(A[crop_center], B[crop_center])

#%%
optimizer = optim.Adam(net.parameters(), lr=0.0001)

loss_stats_train, loss_stats_val = [], []

epochs = 50

loss_train, loss_val = [], []

basis.model = toy
        
net.train()

wvl_L, wvl_R = 1625, 1625
train_ids = train_ids_all[wvl_L]
val_ids   = val_ids_all[wvl_L]

for epoch in range(epochs):
    print(f'>>>>>>>>> Epoch: {epoch+1}/{epochs}')
    
    epoch_train_loss, epoch_val_loss = [], []
    
    np.random.shuffle(train_ids)

    for i, id in enumerate(train_ids):
        optimizer.zero_grad()
        
        batch_data = batches_train_all[wvl_L][id]

        x0, fixed_inputs, PSF_0, current_config = get_data(batch_data, fixed_entries)
        toy.config = current_config
        toy.Update(reinit_grids=True, reinit_pupils=True)

        batch_size = len(batch_data['IDs'])

        loss = loss_fn(x0, PSF_0)
        loss.backward()
        optimizer.step()

        loss_train.append(loss.item()/batch_size)
        epoch_train_loss.append(loss.item()/batch_size)
        
        print(f'Running loss ({i+1}/{len(train_ids)}): {loss.item()/batch_size:.4f}', end="\r", flush=True)
    
    loss_stats_train.append(np.array(epoch_train_loss).mean().item())
    print(f'Epoch: {epoch+1}/{epochs}, train loss: {np.array(epoch_train_loss).mean().item()}')
    
    
    np.random.shuffle(val_ids)
    
    for i, id in enumerate(val_ids):
        with torch.no_grad():
            batch_data = batches_val_all[wvl_L][id]
            
            x0, fixed_inputs, PSF_0, current_config = get_data(batch_data, fixed_entries)
            toy.config = current_config
            toy.Update(reinit_grids=True, reinit_pupils=True)

            batch_size = len(batch_data['IDs'])

            loss = loss_fn(x0, PSF_0)
            loss_val.append(loss.item()/batch_size)
            epoch_val_loss.append(loss.item()/batch_size)
            
            print(f'Running loss ({i+1}/{len(val_ids)}): {loss.item()/batch_size:.4f}', end="\r", flush=True)


    loss_stats_val.append(np.array(epoch_val_loss).mean().item())
    print(f'Epoch: {epoch+1}/{epochs}, validation loss: {np.array(epoch_val_loss).mean().item()}\n')
    torch.save(net.state_dict(), f'../data/weights/gnosis_mono_ep_{epoch+1}.dict')


loss_stats_val   = np.array(loss_stats_val)
loss_stats_train = np.array(loss_stats_train)

plt.plot(loss_stats_val)
plt.plot(loss_stats_train)
plt.show()

#%%

np.save('../data/temp/loss_stats_val.npy', loss_stats_val)
np.save('../data/temp/loss_stats_train.npy', loss_stats_train)


#%% =============================== Polychromatic training ==================================
optimizer = optim.Adam(net.parameters(), lr=0.0001)

loss_stats_train, loss_stats_val = [], []

def loss_fn(x_, PSF_0):
    A = func(x_, fixed_inputs)
    B = PSF_0
    
    loss_1 = nn.L1Loss (reduction='sum')(A[crop_all],    B[crop_all])
    loss_2 = nn.MSELoss(reduction='sum')(A[crop_center], B[crop_center])
    
    return loss_1 + loss_2


epochs = 50

loss_train, loss_val = [], []

net.train()

for epoch in range(epochs):
    print(f'>>>>>>>>> Epoch: {epoch+1}/{epochs}')
    
    epoch_train_loss, epoch_val_loss = [], []
    
    np.random.shuffle(wvls_L_all)
    
    for l, wvl_L in enumerate(wvls_L_all):
        print(f'Wavelengths: {wvl_L}, {wvl_R}')
        train_ids = train_ids_all[wvl_L]
        np.random.shuffle(train_ids)

        for i, id in enumerate(train_ids):
            optimizer.zero_grad()
            
            batch_data = batches_train_all[wvl_L][id]

            x0, fixed_inputs, PSF_0, current_config = get_data(batch_data, fixed_entries)
            # toy = TipTorch(current_config, None, device)
            toy.config = current_config
            toy.Update(reinit_grids=True, reinit_pupils=True)
            
            batch_size = len(batch_data['IDs'])

            # loss = loss_fn(func(x0, fixed_inputs), PSF_0)
            loss = loss_fn(x0, PSF_0)
            
            loss.backward()
            optimizer.step()

            loss_train.append(loss.item()/batch_size)
            epoch_train_loss.append(loss.item()/batch_size)
            
            print(f'Running loss ({i+1}/{len(train_ids)}): {loss.item()/batch_size:.4f}', end="\r", flush=True)
            
    print()
    loss_stats_train.append(np.array(epoch_train_loss).mean().item())
    print(f'Epoch: {epoch+1}/{epochs}, train loss: {np.array(epoch_train_loss).mean().item()}')
    
    for l, wvl_L in enumerate(wvls_L_all[4:]):
        print(f'Wavelengths: {wvl_L}, {wvl_R}')
        val_ids = val_ids_all[wvl_L]
        np.random.shuffle(val_ids)
        
        for i, id in enumerate(val_ids):
            with torch.no_grad():
                batch_data = batches_val_all[wvl_L][id]
                
                x0, fixed_inputs, PSF_0, current_config = get_data(batch_data, fixed_entries)
                # toy = TipTorch(current_config, None, device)
                toy.config = current_config
                toy.Update(reinit_grids=True, reinit_pupils=True)

                batch_size = len(batch_data['IDs'])

                # loss = loss_fn(func(x0, fixed_inputs), PSF_0)
                
                loss = loss_fn(x0, PSF_0)
                
                loss_val.append(loss.item()/batch_size)
                epoch_val_loss.append(loss.item()/batch_size)
                
                print(f'Running loss ({i+1}/{len(val_ids)}): {loss.item()/batch_size:.4f}', end="\r", flush=True)

    print()
    loss_stats_val.append(np.array(epoch_val_loss).mean().item())
    print(f'Epoch: {epoch+1}/{epochs}, validation loss: {np.array(epoch_val_loss).mean().item()}\n')
    torch.save(net.state_dict(), f'../data/weights/gnosis_poly_ep_{epoch+1}.dict')


loss_stats_val   = np.array(loss_stats_val)
loss_stats_train = np.array(loss_stats_train)

# np.save('../data/loss_stats_val.npy', loss_stats_val)
# np.save('../data/loss_stats_train.npy', loss_stats_train)

# Save weights
# torch.save(net.state_dict(), '../data/weights/gnosis_new_weights_v3.dict')


#%% ============================================ Validation ============================================
from joblib import load

df_norm = pd.concat([psf_df_norm, fitted_df_norm], axis=1)#.fillna(0)

with open('../data/LWE.predictor', 'rb') as handle:
    LWE_predictor_dict = load(handle)

entries = LWE_predictor_dict['inputs']
mor = LWE_predictor_dict['LWE WFE predictor']
pca = LWE_predictor_dict['PCA']
gbr_LWE = LWE_predictor_dict['LWE coefs predictor']

def predict_LWE(IDs):
    X_inp = df_norm.loc[IDs][entries].to_numpy()
    LWE_WFE_pred = df_transforms_fitted['LWE coefs'].backward(gbr_LWE.predict(X_inp))
    LWE_coefs_pred_pca = mor.predict(X_inp)
    LWE_coefs_pred = df_transforms_fitted['LWE coefs'].backward(pca.inverse_transform(LWE_coefs_pred_pca))
    
    LWE_coefs_pred /= np.linalg.norm(LWE_coefs_pred, ord=2, axis=1)[:, np.newaxis]
    LWE_coefs_pred *= LWE_WFE_pred[:, np.newaxis]
    return torch.tensor(LWE_coefs_pred, device=device).float()

# coefs = predict_LWE(batch_data['IDs'])

#%%
PSFs_0_val_all = {}
PSFs_1_val_all = {}
PSFs_2_val_all = {}
PSFs_3_val_all = {}

fitted_entries = [
    'F L', 'F R', 'bg L', 'bg R', 'dx L', 'dx R', 'dy L', 'dy R',
    'Wind dir', 'r0', 'Jx', 'Jy', 'Jxy', 'dn', 'LWE coefs'
]

net.eval()
with torch.no_grad():
    for wvl in [1625]: #wvls_L_all:
        # wvl = 1625
        
        PSFs_0_val, PSFs_1_val, PSFs_2_val, PSFs_3_val = [], [], [], []
        val_ids = val_ids_all[wvl]

        for i in tqdm(val_ids):
            # ------------------------- Validate calibrated -------------------------
            batch_data = batches_val_all[wvl][i]
            
            x0, fixed_inputs, PSF_0, init_config = get_data(batch_data, fixed_entries)
            toy.config = init_config
            toy.Update(reinit_grids=True, reinit_pupils=True)

            batch_size = len(batch_data['IDs'])
            
            fixed_inputs['Jxy'] *= 0
            
            fixed_inputs['basis_coefs'] = predict_LWE(batch_data['IDs'])
            
            PSFs_0_val.append(PSF_0.cpu())
            PSFs_1_val.append(func(x0, fixed_inputs).cpu())

            # ------------------------- Validate direct -------------------------
            inputs = {
                'F':   torch.ones([1, 2]),
                'Jx':  torch.ones([1])*33.0,
                'Jy':  torch.ones([1])*33.0,
                'Jxy': torch.zeros([1]),
                'dn':  torch.zeros([1]),
                'basis_coefs': torch.zeros([1, 12])
            }
            
            current_batch_size = len(batch_data['IDs'])

            for key, value in inputs.items():
                inputs[key] = value.float().to(device).repeat(current_batch_size, 1).squeeze()
            
            PSFs_2_val.append(run_model(toy, batch_data, inputs).cpu())

            # ------------------------- Validate fitted -------------------------
            fitted_dict = get_fixed_inputs(batch_data, fitted_entries)

            PSFs_3_val.append(run_model(toy, batch_data, {}, fixed_inputs=fitted_dict).cpu())


        PSFs_0_val = torch.cat(PSFs_0_val, dim=0)[:,0,...].numpy()
        PSFs_1_val = torch.cat(PSFs_1_val, dim=0)[:,0,...].numpy()
        PSFs_2_val = torch.cat(PSFs_2_val, dim=0)[:,0,...].numpy()
        PSFs_3_val = torch.cat(PSFs_3_val, dim=0)[:,0,...].numpy()

        PSFs_0_val_all[wvl] = PSFs_0_val.copy()
        PSFs_1_val_all[wvl] = PSFs_1_val.copy()
        PSFs_2_val_all[wvl] = PSFs_2_val.copy()
        PSFs_3_val_all[wvl] = PSFs_3_val.copy()

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        plot_radial_profiles_new(PSFs_0_val, PSFs_2_val, 'Data', 'TipTorch', title='Direct prediction',     ax=ax[0])
        plot_radial_profiles_new(PSFs_0_val, PSFs_1_val, 'Data', 'TipTorch', title='Calibrated prediction', ax=ax[1])
        # plot_radial_profiles_new(PSFs_0_val, PSFs_3_val, 'Data', 'TipTorch', title='Fitted', ax=ax[2])
        fig.suptitle(f'λ = {wvl} [nm]')
        plt.tight_layout()
        # plt.show()
        plt.savefig(f'C:/Users/akuznets/Desktop/presa_buf/PSF_validation_{wvl}.png', dpi=200)
        
#%%
# wvl = np.random.choice(wvls_L_all)
wvl = 1625
i = np.random.choice(val_ids_all[wvl])

batch_data = batches_val_all[wvl][i]

j = np.random.choice(batch_data['IDs'])
index_ = batch_data['IDs'].tolist().index(j)

x0, fixed_inputs, _, _ = get_data(batch_data, fitted_entries)

pred_coefs = transformer.destack(net(x0))['basis_coefs'][index_, ...]
test_coefs = fixed_inputs['basis_coefs'][index_, ...]
        
with torch.no_grad():
    phase_pred = (pred_coefs.unsqueeze(-1).unsqueeze(-1) * basis.modal_basis).sum(dim=0)
    phase_test = (test_coefs.unsqueeze(-1).unsqueeze(-1) * basis.modal_basis).sum(dim=0)

    MSE_1 = ( (phase_pred-phase_test)**2 ).mean()
    MSE_2 = ( (phase_pred+phase_test)**2 ).mean()

    if MSE_1 > MSE_2:
        phase_pred *= -1

    img = torch.hstack([phase_pred, phase_test, phase_test-phase_pred]).cpu()
    plt.imshow(img)

#%% -=-===-=-=-=-=-=-=-=-=--=-=-==-=-=-=-=-=-=-=-=-=-===-=-=-=-=-=-=-=-=-=-=--=-=-==-=-=-=-=-=-=-=-=-=-===-=-=-=-=-=-=-=-=--=-=-==-=-=-=-=-=-=-=-=

PSFs_data, PSFs_fitted  = [], []

net.eval()

# for wvl in wvls_L_all:
wvl = 1625

val_ids = val_ids_all[wvl]

fitted_entries = ['F L', 'F R', 'bg L', 'bg R', 'dx L', 'dx R', 'dy L', 'dy R', 'r0', 'Jx', 'Jy', 'Jxy', 'dn', 'Wind dir', 'LWE coefs']

batch_directory = SPHERE_DATASET_FOLDER + 'batch_test.pkl'


with open(batch_directory, 'rb') as handle:
    batch_data = pickle.load(handle)


with torch.no_grad():
    # ------------------------- Validate calibrated -------------------------
    x0, fixed_inputs, PSF_0, init_config = get_data(batch_data, fitted_entries)
    toy.config = init_config
    toy.Update(reinit_grids=True, reinit_pupils=True)
    
    PSFs_data.append(PSF_0.cpu())
    fitted_dict = get_fixed_inputs(batch_data, fitted_entries)
    PSFs_fitted.append(run_model(toy, batch_data, {}, fixed_inputs=fitted_dict).cpu())

PSFs_data   = torch.cat(PSFs_data,   dim=0)[:,0,...].numpy()
PSFs_fitted = torch.cat(PSFs_fitted, dim=0)[:,0,...].numpy()

# plot_radial_profiles_new(PSFs_data, PSFs_fitted, 'Data', 'TipTorch', title='Fitted')

index_ = batch_data['IDs'].index(2818)

plot_radial_profiles_new(PSFs_data[index_,...], PSFs_fitted[index_,...], 'Data', 'TipTorch', title='Fitted')
plt.show()

# draw_PSF_stack(PSFs_data[index_,...], PSFs_fitted[index_,...], average=True, crop=40, min_val=1e-6, max_val=1e-1)
# plt.show()

# dPSF = np.abs(PSFs_data[index_,...] - PSFs_fitted[index_,...]) / PSFs_data[index_,...].max()

# print( 100*dPSF.max().item() )

#%%

new_dict = {}

for entry in fitted_dict:
    new_dict[entry] = fitted_dict[entry][index_].unsqueeze(0)
    # item_ = fitted_dict[entry][index_]
    
    # if len(item_.shape) == 0:
    #     print(f'{entry}: {item_.item()}')
    # else:
    #     print(entry, end=': ')
    #     for x in item_.cpu().numpy().tolist():
    #         print(f'{x}',  end=' ')
    #     print('')

# _, _, merged_config = SPHERE_preprocess(
#     sample_ids    = [2818],
#     norm_regime   = 'sum',
#     split_cube    = False,
#     PSF_loader    = lambda x: SamplesByIds(x, synth=False),
#     config_loader = GetSPHEREonsky,
#     framework     = 'pytorch',
#     device        = device)

# toy = TipTorch(merged_config, None, device)
toy = TipTorch(init_config, None, device)

new_dict = fitted_dict

_ = toy()

PSF_test = toy(new_dict, None, lambda: basis(new_dict['basis_coefs'].float())).cpu().numpy()[index_,0,...]

#%%
plot_radial_profiles_new(PSFs_data[index_,...], PSF_test, 'Data', 'TipTorch', title='Fitted')
plt.show()


#%%

# A = (PSF_test - PSF_test.min()) / PSF_test.max()
# B = (PSFs_fitted[index_,...] - PSFs_fitted[index_,...].min()) / PSFs_fitted[index_,...].max()
A = PSF_test
B = PSFs_fitted[index_,...]

# draw_PSF_stack(PSF_test, PSFs_fitted[index_,...], average=True, crop=40, min_val=1e-6, max_val=1e-1)
# plt.show()

# draw_PSF_stack(A, B, average=True, crop=40, min_val=1e-6, max_val=1e-1)
plot_radial_profiles_new(A, B, 'Data', 'TipTorch', title='Fitted')
plt.show()

#%%
wvl = 1625

PSF_cube_data   = torch.tensor(PSFs_0_val_all[wvl]).unsqueeze(1)
PSF_cube_calib  = torch.tensor(PSFs_1_val_all[wvl]).unsqueeze(1)
PSF_cube_direct = torch.tensor(PSFs_2_val_all[wvl]).unsqueeze(1)
PSF_cube_fitted = torch.tensor(PSFs_3_val_all[wvl]).unsqueeze(1)

draw_PSF_stack(PSF_cube_data, PSF_cube_direct, average=True, crop=40, min_val=1e-5, max_val=1e-1)
plt.show()
draw_PSF_stack(PSF_cube_data, PSF_cube_calib,  average=True, crop=40, min_val=1e-5, max_val=1e-1)
plt.show()
draw_PSF_stack(PSF_cube_data, PSF_cube_fitted,  average=True, crop=40, min_val=1e-5, max_val=1e-1)
plt.show()

#%%
rand_id = np.random.randint(0, PSFs_0_val_all[wvl].shape[0])
#%
PSF_0_val = PSFs_0_val_all[wvl][rand_id,...].copy()
PSF_1_val = PSFs_1_val_all[wvl][rand_id,...].copy()
PSF_2_val = PSFs_2_val_all[wvl][rand_id,...].copy()
PSF_3_val = PSFs_3_val_all[wvl][rand_id,...].copy()

#%
from matplotlib.colors import LogNorm

# PSF_0 = PSF_0_val
# PSF_1 = PSF_1_val

# if PSF_0.ndim == 2: PSF_0 = PSF_0[None, None, ...]
# if PSF_1.ndim == 2: PSF_1 = PSF_1[None, None, ...]

# if PSF_0.ndim == 3: PSF_0 = PSF_0[None, ...]
# if PSF_1.ndim == 3: PSF_1 = PSF_1[None, ...]

# def draw_PSF_stack_new(PSF_0, PSF_1, crop_size, norm, ax, titles):

size = 2.5

titles = [
    ['Data', 'Direct pred.',     'Difference'],
    ['Data', 'Calibrated pred.', 'Difference']
]

crop_size = 60

PSF_0 = PSF_0_val.copy()
PSF_1 = PSF_1_val.copy()
PSF_2 = PSF_2_val.copy()

vmin = np.min( [PSF_0_val.min(), PSF_1_val.min(), PSF_2_val.min()] )
vmax = np.max( [PSF_0_val.max(), PSF_1_val.max(), PSF_2_val.max()] )

PSF_0 += 2e-5
PSF_1 += 2e-5
PSF_2 += 2e-5

norm = LogNorm(vmin=vmin, vmax=vmax)

_, ax = plt.subplots(2, 3, figsize=(size*2.5, 2*size))

PSF_0_ = PSF_0.cpu().numpy().copy() if isinstance(PSF_0, torch.Tensor) else PSF_0.copy()
PSF_1_ = PSF_1.cpu().numpy().copy() if isinstance(PSF_1, torch.Tensor) else PSF_1.copy()
PSF_2_ = PSF_2.cpu().numpy().copy() if isinstance(PSF_2, torch.Tensor) else PSF_2.copy()

crop = cropper(PSF_0_, crop_size)

diff1 = PSF_0_-PSF_1_
diff2 = PSF_0_-PSF_2_

ax[0,0].imshow(np.abs(PSF_0_)[crop], norm=norm)
ax[0,1].imshow(np.abs(PSF_2_)[crop], norm=norm)
ax[0,2].imshow(np.abs(diff2) [crop], norm=norm)

ax[1,0].imshow(np.abs(PSF_0_)[crop], norm=norm)
ax[1,1].imshow(np.abs(PSF_1_)[crop], norm=norm)
ax[1,2].imshow(np.abs(diff1) [crop], norm=norm)

for i in range(0,3):
    ax[0,i].set_axis_off()
    ax[0,i].set_title(titles[0][i])
    
    ax[1,i].set_axis_off()
    ax[1,i].set_title(titles[1][i])

plt.tight_layout()
plt.suptitle(f'{rand_id}')

# _, ax1 = plt.subplots(1,3, figsize=(size*2.5, size))
# draw_PSF_stack_new(PSF_0_val, PSF_1_val, 60, norm, ax1, ['Data', 'Calibrated pred.', 'Difference'])
# plt.show()

# _, ax2 = plt.subplots(1,3, figsize=(size*2.5, size))
# draw_PSF_stack_new(PSF_0_val, PSF_0_val, 60, norm, ax2, ['Data', 'Direct pred.', 'Difference'])
# plt.tight_layout()
# plt.show()

#%%
ids_example = [120, 97, 173, 144, 152, 103, 18, 29, 133, 66, 42, 124]

def save_PSF_img(PSF_, filename, norm, size=2.5):
    fig, ax = plt.subplots(1,1, figsize=(size, size))
    ax.imshow(np.abs(PSF_.copy()), cmap='viridis', norm=norm)
    ax.axis('off')  # Remove axes

    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(f'C:/Users/akuznets/Desktop/didgereedo/PSF_examples/{filename}.png', bbox_inches=extent, pad_inches=0)

    plt.tight_layout()
    plt.close(fig)


for id in ids_example:
    crop = cropper(PSFs_0_val_all[1625][0,...], 60)
    
    temp_0 = PSFs_0_val_all[1625][id,...][crop].copy()
    temp_1 = PSFs_1_val_all[1625][id,...][crop].copy() + 1.25e-5
    temp_2 = PSFs_2_val_all[1625][id,...][crop].copy()
        
    vmin = np.min( [temp_0.min(), temp_1.min(), temp_2.min()] )
    vmax = np.max( [temp_0.max(), temp_1.max(), temp_2.max()] )
    norm = LogNorm(vmin=vmin, vmax=vmax)
        
    save_PSF_img(temp_0, f'{id}_data',   norm, size=2.5)
    save_PSF_img(temp_1, f'{id}_calib',  norm, size=2.5)
    save_PSF_img(temp_2, f'{id}_direct', norm, size=2.5)
    
    save_PSF_img(np.abs(temp_0-temp_2), f'{id}_diff_direct',   norm, size=2.5)
    save_PSF_img(np.abs(temp_0-temp_1), f'{id}_diff_calib',   norm, size=2.5)


#%%
from tools.utils import save_GIF_RGB
from matplotlib import cm
from PIL import Image
from PIL.Image import Resampling
    
base_dir = 'C:/Users/akuznets/Desktop/didgereedo/PSF_examples/'

def save_stack_GIF(name):
    gif_anim = []
    path_save = base_dir + f'PSF_{name}.gif'

    for filename in os.listdir(base_dir+f'{name}/'):
        if filename.endswith('.png'):
            img = Image.open(base_dir + f'{name}/' + filename)

            img = np.array(img)
            img = img[:,1:,:3]
            img = Image.fromarray(img)

            gif_anim.append(img)
            gif_anim[0].save(path_save, save_all=True, append_images=gif_anim[1:], optimize=True, duration=2e3, loop=0)

save_stack_GIF('data')
save_stack_GIF('calib')
save_stack_GIF('direct')
save_stack_GIF('diff_calib')
save_stack_GIF('diff_direct')

#%%
# fig, ax = plt.subplots(1, 2, figsize=(10, 3))
# plot_radial_profiles_new(PSF_0_val, PSF_2_val, 'Data', 'TipTorch', title='Direct prediction',     ax=ax[0])
# plot_radial_profil(PSF_0_val, PSF_1_val, 'Data', 'TipTorch', title='Calibrated prediction', ax=ax[1])
# fig.suptitle(f'λ = {wvl} [nm]')
# plt.show()

fig, ax = plt.subplots(2, 1, figsize=(5, 4))

draw_PSF_stack(PSF_0_val, PSF_2_val, average=True, crop=60, ax=ax[0])
ax[0].set_title('Direct prediction')

draw_PSF_stack(PSF_0_val, PSF_1_val,  average=True, crop=60, ax=ax[1])
ax[1].set_title('Calibrated prediction')

plt.tight_layout()
# plt.savefig(f'C:/Users/akuznets/Desktop/didgereedo/{rand_id}_pred.png', dpi=300)
# plt.show()


#%%
norm_ = PSF_cube_data.median(dim=0)[0].max()

dPSF_1 = (torch.abs(PSF_cube_calib  - PSF_cube_data).median(dim=0)[0].max() / norm_).item() * 100
dPSF_2 = (torch.abs(PSF_cube_direct - PSF_cube_data).median(dim=0)[0].max() / norm_).item() * 100
dPSF_3 = (torch.abs(PSF_cube_fitted - PSF_cube_data).median(dim=0)[0].max() / norm_).item() * 100

#%
print(f"Calib.: {dPSF_1:.2f} \nDirect: {dPSF_2:.2f} \nFitted: {dPSF_3:.2f} \n")

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