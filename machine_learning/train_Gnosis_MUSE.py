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
from tools.utils import plot_radial_profiles_new, cropper, draw_PSF_stack
from data_processing.SPHERE_preproc_utils import SPHERE_preprocess, SamplesByIds
from tools.config_manager import GetSPHEREonsky, ConfigManager
from project_globals import MUSE_DATASET_FOLDER, MUSE_DATA_FOLDER, device
from data_processing.normalizers import InputsTransformer, CreateTransformSequenceFromFile
from copy import copy, deepcopy

from PSF_models.TipToy_MUSE_multisrc import TipTorch
from tools.utils import SausageFeature

df_transforms_onsky  = CreateTransformSequenceFromFile('../data/temp/muse_df_norm_transforms.pickle')
df_transforms_fitted = CreateTransformSequenceFromFile('../data/temp/muse_df_fitted_transforms.pickle')

# with open('../data/temp/fitted_df_norm.pickle', 'rb') as handle:
#     fitted_df_norm = pickle.load(handle)

with open(MUSE_DATA_FOLDER+'muse_df_norm_imputed.pickle', 'rb') as handle:
    muse_df_norm = pickle.load(handle)

config_manager = ConfigManager()

#%%
with open(MUSE_DATASET_FOLDER + 'batch_test.pkl', 'rb') as handle:
    batch_init = pickle.load(handle)

init_config = batch_init['configs']

config_manager.Convert(init_config, framework='pytorch', device=device)
wavelength = batch_init['Wvls']

wavelength_selected = wavelength # For now, the same wvl span is used
ids_wavelength_selected = [i for i, wvl in enumerate(wavelength) if wvl in wavelength_selected]
N_wvl = len(ids_wavelength_selected)

init_config['sources_science']['Wavelength'] = init_config['sources_science']['Wavelength'][:, ids_wavelength_selected]

PSF_0_init = batch_init['PSF_0'][:, ids_wavelength_selected, ...]


#%%
toy = TipTorch(init_config, 'sum', device, TipTop=True, PSFAO=True, oversampling=1)
sausage_absorber = SausageFeature(toy)
sausage_absorber.OPD_map = sausage_absorber.OPD_map.flip(dims=(-1,-2))

toy.PSD_include['fitting'] = True
toy.PSD_include['WFS noise'] = True
toy.PSD_include['spatio-temporal'] = False
toy.PSD_include['aliasing'] = False
toy.PSD_include['chromatism'] = False
toy.PSD_include['Moffat'] = True


toy.to_float()
# toy.to_double()
_ = toy()
toy.s_pow = torch.zeros([toy.N_src,1], device=toy.device).float()

#%%
'''
x0 = [
    0.0, #r0
    *([1.0,]*N_wvl), # F
    # *([0.0,]*N_wvl), # dx
    # *([0.0,]*N_wvl), # dy
    # 0.0,
    # 0.0,
    # *([0.0,]*N_wvl), # bg
    -1, # dn
    # -0.9,
    # -0.9,
    *([-0.9,]*N_wvl), # Jx
    *([-0.9,]*N_wvl), # Jy
    # 0.0, # Jxy

    -1, # s_pow
    
    # PSFAO realm
    -1,
    -1,
    -1
    # *([ 1.0,]*N_wvl),
    # *([ 0.0,]*N_wvl),
    # *([ 0.3,]*N_wvl)
]

x0 = torch.tensor(x0).float().to(device).repeat(toy.N_src, 1)
'''

transforms = {
    'r0':    df_transforms_fitted['r0'],
    'F':     df_transforms_fitted['F'],
    'bg':    df_transforms_fitted['bg'],
    'dx':    df_transforms_fitted['dx'],
    'dy':    df_transforms_fitted['dy'],
    'Jx':    df_transforms_fitted['Jx'],
    'Jy':    df_transforms_fitted['Jy'],
    'Jxy':   df_transforms_fitted['Jxy'],
    'dn':    df_transforms_fitted['dn'],
    's_pow': df_transforms_fitted['s_pow'],
    'amp':   df_transforms_fitted['amp'],
    'b':     df_transforms_fitted['b'],
    'alpha': df_transforms_fitted['alpha'],
    'beta':  df_transforms_fitted['beta'],
    'ratio': df_transforms_fitted['ratio'],
    'theta': df_transforms_fitted['theta']
}

fixed_entries     = ['dx', 'dy', 'Jxy', 'bg']
# predicted_entries = ['r0', 'F', 'dn', 'Jx', 'Jy', 's_pow', 'amp', 'b', 'alpha']
predicted_entries = ['r0', 'F', 'dn', 'Jx', 'Jy', 's_pow']

transformer = InputsTransformer({ entry: transforms[entry] for entry in predicted_entries })

# Here, only the variables that need to be predicted must be added, order and content matter
inp_dict = {
    'r0':    torch.ones ( toy.N_src, device=toy.device)*0.1,
    'F':     torch.ones ([toy.N_src, N_wvl], device=toy.device),
    'dn':    torch.ones ( toy.N_src, device=toy.device)*1.5,
    'Jx':    torch.ones ([toy.N_src, N_wvl], device=toy.device)*10,
    'Jy':    torch.ones ([toy.N_src, N_wvl], device=toy.device)*10,
    's_pow': torch.zeros(toy.N_src, device=toy.device),
    'amp':   torch.zeros(toy.N_src, device=toy.device),
    'b':     torch.zeros(toy.N_src, device=toy.device),
    'alpha': torch.ones (toy.N_src, device=toy.device)*0.1,
}

inp_dict_ = {}

for entry in inp_dict.keys():
    if entry in predicted_entries:
        inp_dict_[entry] = inp_dict[entry]

inp_dict = inp_dict_

# fitted_buf = {
#     'Jxy': torch.zeros( toy.N_src, device=toy.device),
#     'bg':  torch.zeros([toy.N_src, N_wvl], device=toy.device),
#     'dx':  torch.zeros([toy.N_src, N_wvl], device=toy.device),
#     'dy':  torch.zeros([toy.N_src, N_wvl], device=toy.device)
# }

_ = transformer.stack(inp_dict, no_transform=True) # to create index mapping and initial values
# testo = toy(transformer.destack(x0) | fitted_buf, None, lambda: sausage_absorber(toy.s_pow.flatten()))

#%%
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
net = Gnosis(batch_init['NN input'].shape[1], transformer.get_packed_size(), 200, 0.1)#0.25)
net.to(device)
net.float()

# net.load_state_dict(torch.load('../data/weights/gnosis_MUSE_v1.dict'))

#%%
crop_all    = cropper(PSF_0_init, 91)
crop_center = cropper(PSF_0_init, 7)
wvl_weights = torch.linspace(1.0, 0.5, N_wvl).to(device).view(1, N_wvl, 1, 1) * 2

def loss_MSE(A, B):
    diff = (A-B) * wvl_weights
    return diff.pow(2).sum() * 200 / PSF_0.shape[0] / PSF_0.shape[1]


def loss_MAE(A, B):
    diff = (A-B) * wvl_weights
    return diff.abs().sum() / PSF_0.shape[0] / PSF_0.shape[1]


def img_punish(A, B): 
    return loss_MSE(A, B) + loss_MAE(A, B) * 0.4


#%%
def get_fixed_inputs(batch, entries=['dx', 'dy', 'Jxy', 'bg']):
    return { entry: batch['fitted data'][entry].to(device) for entry in entries }


def get_NN_pretrain_data(batch, predicted_entries):
    buf_dict = { entry: batch['fitted data'][entry].to(device) for entry in predicted_entries }
    return transformer.stack(buf_dict)


def get_data(batch, fixed_entries):
    x_0    = batch['NN input'].float().to(device)
    PSF_0  = batch['PSF_0'][:, ids_wavelength_selected, ...].float().to(device)
    config = batch['configs']
    fixed_inputs = get_fixed_inputs(batch, fixed_entries)
    config_manager.Convert(config, framework='pytorch', device=device)
    return x_0, fixed_inputs, PSF_0, config


def run_model(model, batch, predicted_inputs, fixed_inputs={}):
    current_configs = batch['configs']
    config_manager.Convert(current_configs, framework='pytorch', device=device)

    model.config = current_configs
    model.Update(reinit_grids=False, reinit_pupils=True)

    x_unpacked = predicted_inputs | fixed_inputs # It overrides the values from the destacked dictionary
    if 's_pow' in x_unpacked:
        return model(x_unpacked, None, lambda: sausage_absorber(toy.s_pow.flatten()))
    else:
        return model(x_unpacked)


def func(x_, batch, fixed_inputs):
    # y_pred = torch.clamp(net(x_), min=-5.0, max=5.0)
    y_pred = net(x_)
    pred_inputs = transformer.destack(y_pred)
    return run_model(toy, batch, pred_inputs, fixed_inputs)


def loss_fn(x_, PSF_data, batch):
    loss = img_punish(func(x_, batch, fixed_inputs), PSF_data)
    return loss

#%%
x0, fixed_inputs, PSF_0, current_config = get_data(batch_init, fixed_entries)
toy.config = current_config
toy.Update(reinit_grids=True, reinit_pupils=True)

batch_size = len(batch_init['IDs'])

#%%
batches_train, batches_val = [], []
train_ids, val_ids = [], []

train_files = [ MUSE_DATASET_FOLDER+'train/'+file      for file in os.listdir(MUSE_DATASET_FOLDER+'train')      if '.pkl' in file ]
val_files   = [ MUSE_DATASET_FOLDER+'validation/'+file for file in os.listdir(MUSE_DATASET_FOLDER+'validation') if '.pkl' in file ]

batches_train, batches_val = [], []

print('Loading train batches...')
for file in tqdm(train_files):
    with open(file, 'rb') as handle:
        batches_train.append( pickle.load(handle) )
        
print('Loading validation batches...')
for file in tqdm(val_files):
    with open(file, 'rb') as handle:
        batches_val.append( pickle.load(handle) )

train_ids = np.arange(len(batches_train)).tolist()
val_ids   = np.arange(len(batches_val)).tolist()

#%%

batch = batches_train[id]

x, fixed_inputs, PSF_0, current_config = get_data(batch, fixed_entries)
batch_size = len(batch['IDs'])

y_pred = net(x)
pred_inputs = transformer.destack(y_pred)
PSF_pred = run_model(toy, batch, pred_inputs, fixed_inputs)


#%%
optimizer = optim.Adam(net.parameters(), lr=0.00001)
loss_train, loss_val = [], []
loss_stats_train, loss_stats_val = [], []

loss_MSE = nn.MSELoss(reduction='mean')

epochs = 50
net.train()

for epoch in range(epochs):
    print(f'>>>>>>>>> Epoch: {epoch+1}/{epochs}')
    
    epoch_train_loss, epoch_val_loss = [], []
    
    np.random.shuffle(train_ids)

    for i, id in enumerate(train_ids):
        optimizer.zero_grad()
        
        batch = batches_train[id]
        X = batch['NN input'].float().to(device)
        y = get_NN_pretrain_data(batch, predicted_entries)

        batch_size = len(batch['IDs'])

        loss = loss_MSE(net(X), y)
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
            batch = batches_val[id]
            X = batch['NN input'].float().to(device)
            y = get_NN_pretrain_data(batch, predicted_entries)

            batch_size = len(batch['IDs'])

            loss = loss_MSE(net(X), y)
            loss_val.append(loss.item()/batch_size)
            epoch_val_loss.append(loss.item()/batch_size)
            
            print(f'Running loss ({i+1}/{len(val_ids)}): {loss.item()/batch_size:.4f}', end="\r", flush=True)


    loss_stats_val.append(np.array(epoch_val_loss).mean().item())
    print(f'Epoch: {epoch+1}/{epochs}, validation loss: {np.array(epoch_val_loss).mean().item()}\n')
    torch.save(net.state_dict(), f'../data/weights/gnosis_MUSE_ep_{epoch+1}.dict')


loss_stats_val   = np.array(loss_stats_val)
loss_stats_train = np.array(loss_stats_train)

plt.plot(loss_stats_val)
plt.plot(loss_stats_train)
plt.show()


#%%
optimizer = optim.Adam(net.parameters(), lr=0.00001)
loss_train, loss_val = [], []
loss_stats_train, loss_stats_val = [], []

epochs = 50
net.train()

for epoch in range(epochs):
    print(f'>>>>>>>>> Epoch: {epoch+1}/{epochs}')
    
    epoch_train_loss, epoch_val_loss = [], []
    
    np.random.shuffle(train_ids)

    for i, id in enumerate(train_ids):
        optimizer.zero_grad()
        
        batch = batches_train[id]

        x, fixed_inputs, PSF_0, current_config = get_data(batch, fixed_entries)
        batch_size = len(batch['IDs'])

        # print(x.norm())
        
        y_pred = net(x)
        pred_inputs = transformer.destack(y_pred)
        PSF_pred = run_model(toy, batch, pred_inputs, fixed_inputs)

        loss = img_punish(PSF_pred, PSF_0)#+ y_pred.sum()**2

        if loss.isnan():
            raise ValueError(f'Loss is NaN, batch {id}' )

        loss.backward()
        
        # torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        
        optimizer.step()

        loss_train.append(loss.item()/batch_size)
        epoch_train_loss.append(loss.item()/batch_size)
        
        print(f'Running loss ({i+1}/{len(train_ids)}): {loss.item()/batch_size:.4f}', end="\r", flush=True)
    
    loss_stats_train.append(np.array(epoch_train_loss).mean().item())
    print(f'Epoch: {epoch+1}/{epochs}, train loss: {np.array(epoch_train_loss).mean().item()}')
    
    
    np.random.shuffle(val_ids)
    
    for i, id in enumerate(val_ids):
        with torch.no_grad():
            batch = batches_val[id]
            
            x0, fixed_inputs, PSF_0, current_config = get_data(batch, fixed_entries)
            # toy.config = current_config
            # toy.Update(reinit_grids=True, reinit_pupils=True)

            batch_size = len(batch['IDs'])

            loss = loss_fn(x0, PSF_0, batch)
            loss_val.append(loss.item()/batch_size)
            epoch_val_loss.append(loss.item()/batch_size)
            
            print(f'Running loss ({i+1}/{len(val_ids)}): {loss.item()/batch_size:.4f}', end="\r", flush=True)


    loss_stats_val.append(np.array(epoch_val_loss).mean().item())
    print(f'Epoch: {epoch+1}/{epochs}, validation loss: {np.array(epoch_val_loss).mean().item()}\n')
    torch.save(net.state_dict(), f'../data/weights/gnosis_MUSE_ep_{epoch+1}.dict')


loss_stats_val   = np.array(loss_stats_val)
loss_stats_train = np.array(loss_stats_train)

plt.plot(loss_stats_val)
plt.plot(loss_stats_train)
plt.show()

#%%
batch = batches_train[id]

x, fixed_inputs, PSF_0, current_config = get_data(batch, fixed_entries)
batch_size = len(batch['IDs'])

# print(x.norm())

y_pred = net(x)
pred_inputs = transformer.destack(y_pred)
PSF_pred = run_model(toy, batch, pred_inputs, fixed_inputs)



#%%

np.save('../data/temp/loss_stats_val.npy', loss_stats_val)
np.save('../data/temp/loss_stats_train.npy', loss_stats_train)


#%%
# fitted_entries = [
#     'F L', 'F R', 'bg L', 'bg R', 'dx L', 'dx R', 'dy L', 'dy R',
#     'Wind dir', 'r0', 'Jx', 'Jy', 'Jxy', 'dn', 'LWE coefs'
# ]

PSFs_0_val, PSFs_1_val, PSFs_2_val, PSFs_3_val = [], [], [], []
net.eval()

with torch.no_grad():
    for i in tqdm(val_ids):
        # ------------------------- Validate calibrated -------------------------
        batch = batches_val[i]
        
        x0, fixed_inputs, PSF_0, config = get_data(batch, fixed_entries)
        toy.config = config
        toy.Update(reinit_grids=True, reinit_pupils=True)

        batch_size = len(batch['IDs'])
        
        PSFs_0_val.append(PSF_0.cpu())
        PSFs_1_val.append(func(x0, batch, fixed_inputs).cpu())

        # # ------------------------- Validate direct -------------------------
        inputs = {
            'F':   torch.ones([1, N_wvl]),
            'Jx':  torch.ones([1, N_wvl])*33.0,
            'Jy':  torch.ones([1, N_wvl])*33.0,
            'Jxy': torch.zeros([1]),
            'dn':  torch.zeros([1]),
            'amp': torch.zeros([1])
            # 'basis_coefs': torch.zeros([1, 12])
        }
        
        current_batch_size = len(batch['IDs'])

        for key, value in inputs.items():
            inputs[key] = value.float().to(device).repeat(current_batch_size, 1).squeeze()
        
        PSFs_2_val.append(run_model(toy, batch, inputs).cpu())

        # # ------------------------- Validate fitted -------------------------
        # fitted_dict = get_fixed_inputs(batch, fitted_entries)

        # PSFs_3_val.append(run_model(toy, batch, {}, fixed_inputs=fitted_dict).cpu())


    # PSFs_0_val = torch.cat(PSFs_0_val, dim=0)[:,0,...].numpy()
    # PSFs_1_val = torch.cat(PSFs_1_val, dim=0)[:,0,...].numpy()
    
    PSFs_0_val = torch.cat(PSFs_0_val, dim=0).mean(axis=1).numpy()
    PSFs_1_val = torch.cat(PSFs_1_val, dim=0).mean(axis=1).numpy()
    PSFs_2_val = torch.cat(PSFs_2_val, dim=0).mean(axis=1).numpy()
    
    # PSFs_2_val = torch.cat(PSFs_2_val, dim=0)[:,0,...].numpy()
    # PSFs_3_val = torch.cat(PSFs_3_val, dim=0)[:,0,...].numpy()

    # fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    fig = plt.figure(figsize=(10, 4))
    plot_radial_profiles_new(PSFs_0_val, PSFs_2_val, 'Data', 'TipTorch', title='Direct prediction')
    # plot_radial_profiles_new(PSFs_0_val, PSFs_1_val, 'Data', 'TipTorch', title='Calibrated prediction')
    # plot_radial_profiles_new(PSFs_0_val, PSFs_3_val, 'Data', 'TipTorch', title='Fitted', ax=ax[2])
    plt.tight_layout()
    # plt.show()
    # plt.savefig(f'C:/Users/akuznets/Desktop/presa_buf/PSF_validation_{wvl}.png', dpi=200)
    
# %%
