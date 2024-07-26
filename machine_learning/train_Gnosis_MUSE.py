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

predict_Moffat = True
# predict_Moffat = False

#%%
with open(MUSE_DATASET_FOLDER + 'batch_test.pkl', 'rb') as handle:
    batch_init = pickle.load(handle)

init_config = batch_init['configs']
wavelength  = batch_init['Wvls']

wavelength_selected = wavelength # For now, the same wvl span is used
# ids_wavelength_selected = [i for i, wvl in enumerate(wavelength) if wvl in wavelength_selected]
ids_wavelength_selected = np.arange(0, len(wavelength), 2)
# ids_wavelength_selected = [11,12]

N_wvl = len(ids_wavelength_selected)

PSF_0_init = batch_init['PSF_0'][:, ids_wavelength_selected, ...]
config_manager.Convert(init_config, framework='pytorch', device=device)
AAA = init_config['sources_science']['Wavelength'].clone().detach().cpu().numpy()
AAA = torch.tensor(AAA[:, ids_wavelength_selected]).to(device)
init_config['sources_science']['Wavelength'] = AAA


#%%
toy = TipTorch(init_config, 'sum', device, TipTop=True, PSFAO=True, oversampling=1)
sausage_absorber = SausageFeature(toy)
sausage_absorber.OPD_map = sausage_absorber.OPD_map.flip(dims=(-1,-2))

toy.PSD_include['fitting']         = True
toy.PSD_include['WFS noise']       = True
toy.PSD_include['spatio-temporal'] = True
toy.PSD_include['aliasing']        = False
toy.PSD_include['chromatism']      = True
toy.PSD_include['Moffat']          = predict_Moffat

toy.to_float()
_ = toy()
toy.s_pow = torch.zeros([toy.N_src,1], device=toy.device).float()

#%%
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

fixed_entries      = ['dx', 'dy', 'Jxy', 'bg']
# additional_params  = ['amp', 'b', 'alpha']
# predicted_entries  = ['r0', 'F', 'dn', 'Jx', 'Jy', 'amp', 'b', 'alpha', 's_pow']
predicted_entries  = ['r0', 'F', 'dn', 'Jx', 'Jy', 's_pow']

if predict_Moffat:
    predicted_entries += ['amp', 'b', 'alpha']


transformer = InputsTransformer({ entry: transforms[entry] for entry in predicted_entries })

# Here, only the variables that need to be predicted must be added, order and content matter
inp_dict = {
    'r0':    torch.ones ( toy.N_src, device=toy.device)*0.1,
    'F':     torch.ones ([toy.N_src, N_wvl], device=toy.device),
    'Jx':    torch.ones ([toy.N_src, N_wvl], device=toy.device)*10,
    'Jy':    torch.ones ([toy.N_src, N_wvl], device=toy.device)*10,
    'dn':    torch.ones (toy.N_src, device=toy.device)*1.5,
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
# net.load_state_dict(torch.load('../data/weights/gnosis_MUSE_some_real_good_weights.dict'))
net.load_state_dict(torch.load('../data/weights/gnosis_MUSE_v3_7wvl_yes_Mof_no_ssg.dict'))

#%%
crop_all    = cropper(PSF_0_init, 91)
crop_center = cropper(PSF_0_init, 25)
wvl_weights = torch.linspace(1.0, 0.5, 13)[ids_wavelength_selected].to(device).view(1, N_wvl, 1, 1) * 2

def loss_MSE(A, B):
    diff = (A-B) * wvl_weights
    return diff.pow(2).sum() * 200 / PSF_0.shape[0] / PSF_0.shape[1]


def loss_MAE(A, B):
    diff = (A-B) * wvl_weights
    return diff.abs().sum() / PSF_0.shape[0] / PSF_0.shape[1]


def img_punish(A, B): 
    return loss_MSE(A, B) + loss_MAE(A, B)
    # return loss_MAE(A, B)# * 0.4


#%%
def get_fixed_inputs(batch, entries):
    return {
        entry: \
            batch['fitted data'][entry][:,ids_wavelength_selected].to(device) \
            if batch['fitted data'][entry].ndim == 2 else batch['fitted data'][entry].to(device) \
        for entry in entries
    }


def get_NN_pretrain_data(batch, predicted_entries):
    buf_dict = { entry: batch['fitted data'][entry].to(device) for entry in predicted_entries }
    return transformer.stack(buf_dict)


def get_data(batch, fixed_entries):
    x_0    = batch['NN input'].float().to(device)
    PSF_0  = batch['PSF_0'][:, ids_wavelength_selected, ...].float().to(device)
    config = batch['configs']
    
    if config['sources_science']['Wavelength'].shape[-1] != N_wvl:
        buf = config['sources_science']['Wavelength'].clone().detach().cpu().numpy()
        buf = torch.tensor(buf[:, ids_wavelength_selected])
        config['sources_science']['Wavelength'] = buf
    
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
    y_pred = net(x_)
    pred_inputs = transformer.destack(y_pred)
    return run_model(toy, batch, pred_inputs, fixed_inputs)


def loss_fn(x_, PSF_data, batch):
    loss = img_punish(func(x_, batch, fixed_inputs), PSF_data)
    return loss

#%%
x0, fixed_inputs, PSF_0, current_config = get_data(batch_init, fixed_entries)
toy.config = init_config
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

#%
# batch = batches_train[id]

# x, fixed_inputs, PSF_0, current_config = get_data(batch, fixed_entries)
# batch_size = len(batch['IDs'])

# y_pred = net(x)
# pred_inputs = transformer.destack(y_pred)
# PSF_pred = run_model(toy, batch, pred_inputs, fixed_inputs)

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
# optimizer = optim.Adam(net.parameters(), lr=5e-6)
optimizer = optim.Adam(net.parameters(), lr=1e-3)
loss_train, loss_val = [], []
loss_stats_train, loss_stats_val = [], []

# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

epochs = 100
net.train()

for epoch in range(epochs):
    print(f'>>>>>>>>> Epoch: {epoch+1}/{epochs}')
    
    epoch_train_loss, epoch_val_loss = [], []
    
    np.random.shuffle(train_ids)

    for i, id in enumerate(train_ids):
        optimizer.zero_grad()
        
        batch = batches_train[id]

        x, fixed_inputs, PSF_0, current_config = get_data(batch, fixed_entries)
        batch_size = len(batch['IDs']) / N_wvl
        
        y_pred = net(x)
        pred_inputs = transformer.destack(y_pred)
        PSF_pred = run_model(toy, batch, pred_inputs, fixed_inputs)

        loss = img_punish(PSF_pred, PSF_0)

        if loss.isnan():
            raise ValueError(f'Loss is NaN, batch {id}' )

        loss.backward()
        
        optimizer.step()
        # scheduler.step()

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

            batch_size = len(batch['IDs']) / N_wvl

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
# batch = batches_train[id]

# x, fixed_inputs, PSF_0, current_config = get_data(batch, fixed_entries)
# batch_size = len(batch['IDs'])

# y_pred = net(x)
# pred_inputs = transformer.destack(y_pred)
# PSF_pred = run_model(toy, batch, pred_inputs, fixed_inputs)


#%
# np.save('../data/temp/loss_stats_val.npy', loss_stats_val)
# np.save('../data/temp/loss_stats_train.npy', loss_stats_train)

torch.save(net.state_dict(), f'../data/weights/gnosis_MUSE_current.dict')

#%
all_entries = [\
    'r0', 'F', 'dn', 'dx', 'dy', 'bg', 'Jx', 'Jy', 'Jxy', 's_pow',
    'amp', 'b', 'alpha', 'beta', 'ratio', 'theta'
]

base_J = [3.2, 3.1, 3.2, 3.7, 4.4, 5.2, 4.9, 6, 7.7, 8.9, 10, 11.8, 13]
jitter = [base_J[i] for i in ids_wavelength_selected]

inputs_direct = {
    'F':   torch.ones([1, N_wvl]) * 0.8,
    'Jx':  torch.tensor(jitter)*5,
    'Jy':  torch.tensor(jitter)*5,
    'Jxy': torch.zeros([1]),
    'dn':  torch.ones([1]) * 2.8 * 2,
    'amp': torch.zeros([1])
}


PSFs_0_val, PSFs_1_val, PSFs_2_val, PSFs_3_val = [], [], [], []
net.eval()

ids_0 = []

with torch.no_grad():
    for i in tqdm(val_ids):
    # for i in tqdm(train_ids):
        # ------------------------- Validate calibrated -------------------------
        toy.PSD_include['fitting']         = True
        toy.PSD_include['WFS noise']       = True
        toy.PSD_include['spatio-temporal'] = True
        toy.PSD_include['aliasing']        = False
        toy.PSD_include['chromatism']      = True
        toy.PSD_include['Moffat']          = predict_Moffat
        
        batch = batches_val[i]
        # batch = batches_train[i]
        x0, fixed_inputs, PSF_0, config = get_data(batch, fixed_entries)
        PSFs_0_val.append(PSF_0.cpu())
        current_batch_size = len(batch['IDs'])
        ids_0.append(batch['IDs'])
        
        toy.config = config
        toy.Update(reinit_grids=True, reinit_pupils=True)
        PSFs_1_val.append(func(x0, batch, fixed_inputs).cpu())
           
        # # ------------------------- Validate direct -------------------------
        _, _, _, config = get_data(batch, fixed_entries)
        toy.config = config
        
        toy.PSD_include['fitting']         = True
        toy.PSD_include['WFS noise']       = True
        toy.PSD_include['spatio-temporal'] = True
        toy.PSD_include['aliasing']        = False
        toy.PSD_include['chromatism']      = True
        toy.PSD_include['Moffat']          = False

        toy.Update(reinit_grids=True, reinit_pupils=True)
        
        inputs_direct_ = { 
            key: value.float().to(device).repeat(current_batch_size, 1).squeeze() for key, value in inputs_direct.items()
        }
        
        PSFs_2_val.append(run_model(toy, batch, inputs_direct_).cpu())
        
        # ------------------------- Validate fitted -------------------------
        _, all_inputs, _, config = get_data(batch, all_entries)
        toy.config = config
        
        toy.PSD_include['fitting']         = True
        toy.PSD_include['WFS noise']       = True
        toy.PSD_include['spatio-temporal'] = True
        toy.PSD_include['aliasing']        = False
        toy.PSD_include['chromatism']      = True
        toy.PSD_include['Moffat']          = True
        
        toy.Update(reinit_grids=True, reinit_pupils=True)
        PSFs_3_val.append(run_model(toy, batch, {}, all_inputs).cpu())


PSFs_0_val = torch.cat(PSFs_0_val, dim=0).mean(axis=1).numpy()
PSFs_1_val = torch.cat(PSFs_1_val, dim=0).mean(axis=1).numpy()
PSFs_2_val = torch.cat(PSFs_2_val, dim=0).mean(axis=1).numpy()
PSFs_3_val = torch.cat(PSFs_3_val, dim=0).mean(axis=1).numpy()

#%%
# fig, ax = plt.subplots(1, 3, figsize=(12, 4))

fig = plt.figure(figsize=(9, 4))

cutoff = 40

plot_radial_profiles_new(PSFs_0_val, PSFs_1_val, 'Data', 'TipTorch', title='Calibrated prediction', cutoff=cutoff)#, ax=ax[0])
# plot_radial_profiles_new(PSFs_0_val, PSFs_2_val, 'Data', 'TipTorch', title='Direct prediction', cutoff=cutoff, ax=ax[1])
# plot_radial_profiles_new(PSFs_0_val, PSFs_3_val, 'Data', 'TipTorch', title='Fitted', cutoff=cutoff, ax=ax[2])
plt.tight_layout()
# plt.show()
# plt.savefig(f'../data/temp/PSFs_current.pdf', dpi=200)

#%%
ids_all = []
for id_turp in ids_0:
    ids_all.extend(id_turp)

# ii = np.random.randint(0, PSFs_0_val.shape[0])

for ii in range(PSFs_0_val.shape[0]):
    d_PSF = PSFs_1_val - PSFs_0_val
    A = np.abs(PSFs_0_val[ii,...])
    B = np.abs(PSFs_1_val[ii,...])
    C = np.abs(d_PSF[ii,...])

    plt.title(ids_all[ii])
    plt.imshow(np.log10( np.maximum(np.hstack([A, B, C]), 5e-7) ))
    plt.tight_layout()
    plt.axis('off')
    plt.show()
    # plt.savefig(f'C:/Users/akuznets/Desktop/didgereedo/MUSE/PSF_val_{ii}.png', dpi=400)


#%% ====================================================================================================
from data_processing.MUSE_preproc_utils import GetConfig, LoadImages, LoadMUSEsampleByID, rotate_PSF

_, all_inputs, _, config = get_data(batch_init, all_entries)

id_local = 1

toy.config = config

toy.PSD_include['fitting'] = True
toy.PSD_include['WFS noise'] = True
toy.PSD_include['spatio-temporal'] = True
toy.PSD_include['aliasing'] = False
toy.PSD_include['chromatism'] = True
toy.PSD_include['Moffat'] = True

toy.Update(reinit_grids=True, reinit_pupils=True)
PSFs_1 = run_model(toy, batch_init, {}, all_inputs)

#%
PSF_0, _, _, _ = LoadImages(sample := LoadMUSEsampleByID(batch_init['IDs'][id_local]))
config_file, PSF_0 = GetConfig(sample, PSF_0)
# PSF_0 = rotate_PSF(PSF_0, -sample['All data']['Pupil angle'].item())
config_file['telescope']['PupilAngle'] = 0

model = TipTorch(config_file, 'sum', device, TipTop=True, PSFAO=True, oversampling=1)

model.PSD_include['fitting'] = True
model.PSD_include['WFS noise'] = True
model.PSD_include['spatio-temporal'] = True
model.PSD_include['aliasing'] = False
model.PSD_include['chromatism'] = True
model.PSD_include['Moffat'] = True

model.to_float()
setattr(model, 's_pow', 0.0)

phytos_dfos = batch_init['fitted data']

inputs_tiptorch = {
    'F':     phytos_dfos['F' ][id_local,...].unsqueeze(0).to(model.device),
    'dx':    phytos_dfos['dx'][id_local,...].unsqueeze(0).to(model.device),
    'dy':    phytos_dfos['dy'][id_local,...].unsqueeze(0).to(model.device),
    'bg':    phytos_dfos['bg'][id_local,...].unsqueeze(0).to(model.device),
    'Jx':    phytos_dfos['Jx'][id_local,...].unsqueeze(0).to(model.device),
    'Jy':    phytos_dfos['Jy'][id_local,...].unsqueeze(0).to(model.device),
    
    'r0':    phytos_dfos['r0'   ][id_local].to(model.device),
    'dn':    phytos_dfos['dn'   ][id_local].to(model.device),
    'Jxy':   phytos_dfos['Jxy'  ][id_local].to(model.device),
    'amp':   phytos_dfos['amp'  ][id_local].to(model.device),
    'b':     phytos_dfos['b'    ][id_local].to(model.device),
    'alpha': phytos_dfos['alpha'][id_local].to(model.device),
    'beta':  phytos_dfos['beta' ][id_local].to(model.device),
    'ratio': phytos_dfos['ratio'][id_local].to(model.device),
    'theta': phytos_dfos['theta'][id_local].to(model.device),
    's_pow': phytos_dfos['s_pow'][id_local].to(model.device)
}

PSF_1 = model(inputs_tiptorch, None, lambda: sausage_absorber(model.s_pow.flatten()))
#%
for entry in ['fitting', 'WFS noise', 'spatio-temporal', 'aliasing', 'chromatism', 'Moffat']:
    try:
        A = toy.PSDs[entry][id_local, ...]
        B = model.PSDs[entry][0, ...]
        print( entry, (A-B).abs().sum().item() )
    except:
        continue

print((toy.dx[id_local, ...] - model.dx[0, ...]).sum().item())
print((toy.dy[id_local, ...] - model.dy[0, ...]).sum().item())
print((toy.F [id_local, ...] - model.F [0, ...]).sum().item())
print((toy.bg[id_local, ...] - model.bg[0, ...]).sum().item())
print((toy.Jx[id_local, ...] - model.Jx[0, ...]).sum().item())
print((toy.Jy[id_local, ...] - model.Jy[0, ...]).sum().item())

print((toy.dn[id_local, ...] - model.dn).item())
print((toy.s_pow[id_local]   - model.s_pow).item())
print((toy.r0[id_local]      - model.r0).item())
print((toy.Jxy[id_local]     - model.Jxy).item())

print((toy.amp[id_local]     - model.amp).item())
print((toy.b[id_local]       - model.b).item())
print((toy.alpha[id_local]   - model.alpha).item())
print((toy.beta[id_local]    - model.beta).item())
print((toy.ratio[id_local]   - model.ratio).item())
print((toy.theta[id_local]   - model.theta).item())


#%%
A = toy.OTF[id_local,0,...]
B = model.OTF[0,0,...]
print(f'PSF error: {(A-B).abs().sum().item() / A.sum().item() * 100:.2f}%' )

plt.imshow(A.abs().log10().cpu().numpy())
plt.show()
plt.imshow(B.abs().log10().cpu().numpy())
plt.show()
plt.imshow((A-B).abs().log10().cpu().numpy())
plt.show()

#%%
A = toy.OTF_static_standart[0,0,...]
B = model.OTF_static_standart[0,0,...]
print(f'PSF error: {(A-B).abs().sum().item() / A.sum().item() * 100:.2f}%' )

plt.imshow(A.abs().log10().cpu().numpy())
plt.show()
plt.imshow(B.abs().log10().cpu().numpy())
plt.show()
plt.imshow((A-B).abs().log10().cpu().numpy())
plt.show()

#%%
id_wvl = 5

A = sausage_absorber(model.s_pow.flatten())#[0,0,...]
B = sausage_absorber(toy.s_pow.flatten())#[id_local,0,...]

AA = model.Phase2OTF(A, 2)[0,id_wvl,...]
BB = toy.Phase2OTF(B, 2)[id_local,id_wvl,...]
#%

# C1 = fftAutoCorr(AA)[0, id_wvl,...]
# C2 = fftAutoCorr(BB)[id_local, id_wvl,...]
# print( (C1-C2).abs().sum().item() )

print( (AA-BB).abs().sum().item() )

# print(AA.sum())

#%%

# plt.imshow(AA.abs().log10().cpu().numpy())
# plt.show()
# plt.imshow(BB.abs().log10().cpu().numpy())
# plt.show()
# plt.imshow((AA-BB).abs().log10().cpu().numpy())
# plt.show()


# plt.imshow(A.imag.cpu().numpy())
# plt.show()
# plt.imshow(B.imag.cpu().numpy())
# plt.show()
# plt.imshow((A-B).imag.cpu().numpy())


#%%
A = PSFs_1[id_local,0,...]
B = PSF_1[0,0,...]
print(f'PSF error: {(A-B).abs().sum().item() / A.sum().item() * 100:.2f}%' )

plt.imshow(A.abs().log10().cpu().numpy())
plt.show()
plt.imshow(B.abs().log10().cpu().numpy())
plt.show()
plt.imshow((A-B).abs().log10().cpu().numpy())
plt.show()
#%%
center = np.array([PSF_0.shape[-2]//2, PSF_0.shape[-1]//2])

plot_radial_profiles_new(
    PSFs_1[id_local,0,...].cpu().numpy(),
    PSF_1[0,0,...].cpu().numpy(),
    'Batch', 'Single', title='Fitted', cutoff=cutoff, centers=center)#, ax=ax[2])

