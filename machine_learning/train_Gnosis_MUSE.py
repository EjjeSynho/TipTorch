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
from tools.config_manager import ConfigManager
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
# testo = toy(transformer.unstack(x0) | fitted_buf, None, lambda: sausage_absorber(toy.s_pow.flatten()))

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
net = Gnosis(batch_init['NN input'].shape[1], transformer.get_stacked_size(), 200, 0.1)#0.25)
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
    pred_inputs = transformer.unstack(y_pred)
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
# pred_inputs = transformer.unstack(y_pred)
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
        pred_inputs = transformer.unstack(y_pred)
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

#%%
def NoiseVarianceIRLOS(r0, WFS_nPix, WFS_Nph, WFS_psInMas, WFS_RON):
    from tools.utils import rad2arc, r0_new
    WFS_wvl = (1.215e-6 + 1.625e-6) / 2
    WFS_excessive_factor = 1
    WFS_d_sub = 4 # D / 2 subaps.
    r0_WFS =  r0_new(r0, WFS_wvl, 0.5e-6)
    WFS_pixelScale = WFS_psInMas / 1e3 # [arcsec]
    
    # Read-out noise calculation
    nD = np.maximum(rad2arc*WFS_wvl/WFS_d_sub/WFS_pixelScale, 1.0)
    # Photon-noise calculation
    nT = np.maximum(rad2arc*WFS_wvl/r0_WFS / WFS_pixelScale, 1.0)

    varRON  = np.pi**2/3 * (WFS_RON**2/WFS_Nph**2) * (WFS_nPix**2/nD)**2
    varShot = np.pi**2 / (2*WFS_Nph) * (nT/nD)**2
    # Noise variance calculation
    varNoise = WFS_excessive_factor * (varRON+varShot) * (500e-9/WFS_wvl)**2

    return varNoise


def get_TT_jitter(r0, WFS_nPix, WFS_Nph, WFS_psInMas, WFS_RON):
    from tools.utils import rad2mas
    TT_var = NoiseVarianceIRLOS(r0, WFS_nPix, WFS_Nph, WFS_psInMas, WFS_RON)

    TT_OPD = np.sqrt(TT_var) * 0.5e-6 / 2 / np.pi # [nm]

    TT_max = 1.9665198 # Value at the edge of tip/tilt mode
    jitter_STD = lambda TT_WFE: np.arctan(2*TT_max/toy.D * TT_WFE) * rad2mas # [mas], TT_WFE in [m]

    return jitter_STD(TT_OPD) # [mas]


def get_jitter_from_ids(ids):
    from tools.utils import r0 as r0_from_seeing
    get_entry = lambda entry, ids: np.array( df_transforms_onsky[entry].backward(muse_df_norm[entry].loc[ids]).tolist() )

    r0      = r0_from_seeing(get_entry('Seeing (header)', ids), 500e-9)
    IR_ph   = get_entry('IRLOS photons, [photons/s/m^2]', ids)
    IR_win  = get_entry('window', ids)
    IR_freq = get_entry('frequency', ids)
    IR_pix  = get_entry('plate scale, [mas/pix]', ids)
    IR_RON  = get_entry('RON, [e-]', ids)

    M1_area = (8-1.12)**2 * np.pi / 4

    IR_ph = IR_ph / IR_freq * M1_area # [photons/frame/aperture]

    return get_TT_jitter(r0, IR_win, IR_ph, IR_pix, IR_RON)


#%%
# batch = batches_train[id]

# x, fixed_inputs, PSF_0, current_config = get_data(batch, fixed_entries)
# batch_size = len(batch['IDs'])

# y_pred = net(x)
# pred_inputs = transformer.unstack(y_pred)
# PSF_pred = run_model(toy, batch, pred_inputs, fixed_inputs)


#%
# np.save('../data/temp/loss_stats_val.npy', loss_stats_val)
# np.save('../data/temp/loss_stats_train.npy', loss_stats_train)

# torch.save(net.state_dict(), f'../data/weights/gnosis_MUSE_current.dict')

#%
all_entries = [\
    'r0', 'F', 'dn', 'dx', 'dy', 'bg', 'Jx', 'Jy', 'Jxy', 's_pow',
    'amp', 'b', 'alpha', 'beta', 'ratio', 'theta'
]

base_J = [3.2, 3.1, 3.2, 3.7, 4.4, 5.2, 4.9, 6, 7.7, 8.9, 10, 11.8, 13]
jitter = [base_J[i] for i in ids_wavelength_selected]

ids, norms, bgs = [], [], []

direct_tuned = True

if direct_tuned:
    inputs_direct = {
        'F':   torch.ones([1, N_wvl]) * 0.8,
        'Jx':  torch.tensor(jitter)*5,
        'Jy':  torch.tensor(jitter)*5,
        'Jxy': torch.zeros([1]),
        'dn':  torch.ones([1]) * 2.8 * 2,
        'amp': torch.zeros([1])
    }
else:
    inputs_direct = {
        'F':   torch.ones([1, N_wvl]),
        'Jx':  torch.zeros([1]),
        'Jy':  torch.zeros([1]),
        'Jxy': torch.zeros([1]),
        'dn':  torch.zeros([1]),
        'amp': torch.zeros([1])
    }


PSFs_0_val_poly, PSFs_1_val_poly, PSFs_2_val_poly, PSFs_3_val_poly = [], [], [], []
net.eval()

pred_inputs_stats = []
configs = []

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
        
        ids.append(batch['IDs'])
        bgs.append(batch['bgs'].squeeze()[:, ids_wavelength_selected].numpy())
        norms.append(batch['norms'].squeeze()[:, ids_wavelength_selected].numpy())
        
        # batch = batches_train[i]
        x0, fixed_inputs, PSF_0, config = get_data(batch, fixed_entries)
        PSFs_0_val_poly.append(PSF_0.cpu())
        current_batch_size = len(batch['IDs'])
        
        pred_inputs_stats.append(transformer.unstack(net(x0)))
        configs.append(deepcopy(config))
        
        toy.config = config
        toy.Update(reinit_grids=True, reinit_pupils=True)
        PSFs_1_val_poly.append(func(x0, batch, fixed_inputs).cpu())
           
        # ------------------------- Validate direct -------------------------
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
        
        if not direct_tuned:
            TT_analytical = get_jitter_from_ids(batch['IDs'])
            inputs_direct_['Jx'] = torch.tensor(TT_analytical).float().to(device).squeeze()
            inputs_direct_['Jy'] = torch.tensor(TT_analytical).float().to(device).squeeze()
        
        PSFs_2_val_poly.append(run_model(toy, batch, inputs_direct_).cpu())
        
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
        PSFs_3_val_poly.append(run_model(toy, batch, {}, all_inputs).cpu())


PSFs_0_val_poly = torch.cat(PSFs_0_val_poly, dim=0).numpy()
PSFs_1_val_poly = torch.cat(PSFs_1_val_poly, dim=0).numpy()
PSFs_2_val_poly = torch.cat(PSFs_2_val_poly, dim=0).numpy()
PSFs_3_val_poly = torch.cat(PSFs_3_val_poly, dim=0).numpy()

ids = np.hstack(ids)
bgs = np.vstack(bgs)[..., np.newaxis, np.newaxis]
norms = np.vstack(norms)[..., np.newaxis, np.newaxis]

PSFs_0_val_white = np.mean(PSFs_0_val_poly, axis=1)
PSFs_1_val_white = np.mean(PSFs_1_val_poly, axis=1)
PSFs_2_val_white = np.mean(PSFs_2_val_poly, axis=1)
PSFs_3_val_white = np.mean(PSFs_3_val_poly, axis=1)

# Store the results
with open('../data/temp/PSFs_val_data_polychrome.pickle', 'wb') as handle:
    pickle.dump(PSFs_0_val_poly, handle)
    
with open('../data/temp/PSFs_val_calibrated_polychrome.pickle', 'wb') as handle:
    pickle.dump(PSFs_1_val_poly, handle)

if direct_tuned:
    with open('../data/temp/PSFs_val_tuned_polychrome.pickle', 'wb') as handle:
        pickle.dump(PSFs_2_val_poly, handle)
else:
    with open('../data/temp/PSFs_val_direct_polychrome_raw.pickle', 'wb') as handle:
        pickle.dump(PSFs_2_val_poly, handle)
    
with open('../data/temp/PSFs_val_fitted_polychrome.pickle', 'wb') as handle:
    pickle.dump(PSFs_3_val_poly, handle)

save_dir = '/home/akuznets/Projects/TipTorch/data/temp/plots/'

#%%
# fig, ax = plt.subplots(1, 3, figsize=(12, 4))
# plt.tight_layout()

fig = plt.figure(figsize=(9, 4))

draw_calibrated = True
draw_direct     = True
draw_fitted     = False
save_profiles   = False
draw_white      = True

cutoff = 40

if draw_white:
    # White profiles
    if draw_calibrated:
        plot_radial_profiles_new(PSFs_0_val_white, PSFs_1_val_white, 'Data', 'TipTorch', title='Calibrated prediction', cutoff=cutoff)#, ax=ax[0])
        if save_profiles:
            plt.savefig(save_dir+'PSFs_calibrated.pdf', dpi=300)

    if draw_direct:
        plot_radial_profiles_new(PSFs_0_val_white, PSFs_2_val_white, 'Data', 'TipTorch', title='Direct prediction', cutoff=cutoff)#, ax=ax[1])
        if save_profiles:
            postfix = '_tuned' if direct_tuned else '_raw'
            plt.savefig(save_dir+f'PSFs_direct_{postfix}.pdf', dpi=300)

    if draw_fitted:
        plot_radial_profiles_new(PSFs_0_val_white, PSFs_3_val_white, 'Data', 'TipTorch', title='Fitted', cutoff=cutoff)#, ax=ax[2])
        if save_profiles:
            plt.savefig(save_dir+'PSFs_fitted.pdf', dpi=300)

else:
    # Polychromatic profiles
    for i, wvl in tqdm(enumerate(wavelength_selected[ids_wavelength_selected].tolist())):
        if draw_calibrated:
            plot_radial_profiles_new(PSFs_0_val_poly[:,i,...], PSFs_1_val_poly[:,i,...], 'Data', 'TipTorch', title='Calibrated prediction', cutoff=cutoff)#, ax=ax[0])
            if save_profiles:
                plt.savefig(save_dir+f'PSFs_calibrated_{wvl}.pdf', dpi=300)
                
        if draw_direct:
            plot_radial_profiles_new(PSFs_0_val_poly[:,i,...], PSFs_2_val_poly[:,i,...], 'Data', 'TipTorch', title='Direct prediction', cutoff=cutoff)#, ax=ax[1])
            if save_profiles:
                postfix = '_tuned' if direct_tuned else '_raw'
                plt.savefig(save_dir+f'PSFs_direct_{postfix}_{wvl}.pdf', dpi=300)
        
        if draw_fitted:
            plot_radial_profiles_new(PSFs_0_val_poly[:,i,...], PSFs_3_val_poly[:,i,...], 'Data', 'TipTorch', title='Fitted', cutoff=cutoff)#, ax=ax[2])
            if save_profiles:
                plt.savefig(save_dir+f'PSFs_fitted_{wvl}.pdf', dpi=300)


#%%

# ii = np.random.randint(0, PSFs_0_val.shape[0])

for ii in range(PSFs_0_val_poly.shape[0]):
    d_PSF = PSFs_1_val_white - PSFs_0_val_white
    A = np.abs(PSFs_0_val_white[ii,...])
    
    data_max = A.max()
    A = A / data_max * 100
    B = np.abs(PSFs_1_val_white[ii,...]) / data_max * 100
    C = np.abs(d_PSF[ii,...]) / data_max * 100

    plt.title(str(ids[ii]) + f', max. err {C.max():.1f}%')
    plt.imshow(np.log10( np.maximum(np.hstack([A, B, C]), 5e-7) ), cmap='gray')
    plt.tight_layout()
    plt.axis('off')
    plt.show()
    # plt.savefig(f'C:/Users/akuznets/Desktop/didgereedo/MUSE/PSF_val_{ii}.png', dpi=400)

#%%
from matplotlib.colors import LogNorm
from matplotlib import cm
from matplotlib.gridspec import GridSpec
import matplotlib as mpl

with open(MUSE_DATA_FOLDER+'muse_df.pickle', 'rb') as handle:
    muse_df = pickle.load(handle)


def plot_MUSE_PSF(id, save_dir=None):
    
    ii = np.where(ids == id)[0][0]
    
    cmap = mpl.colormaps.get_cmap('inferno')  # viridis is the default colormap for imshow
    cmap.set_bad(color='black')

    temp_0 = PSFs_0_val_white[ii,...].copy() # Data
    temp_1 = PSFs_1_val_white[ii,...].copy() # Calibrted
    temp_2 = PSFs_2_val_white[ii,...].copy() # Direct

    v_min_thresh = 2e-6

    temp_0  = np.abs(np.maximum(temp_0, v_min_thresh))
    temp_1  = np.abs(np.maximum(temp_1, v_min_thresh))
    temp_d1 = np.maximum(np.abs(temp_1-temp_0), v_min_thresh)
    temp_d2 = np.maximum(np.abs(temp_2-temp_0), v_min_thresh)

    vmax = np.max( [temp_0.max(), temp_1.max(), temp_2.max()] )

    temp_0  = temp_0  / vmax * 100
    temp_1  = temp_1  / vmax * 100
    temp_2  = temp_2  / vmax * 100
    temp_d1 = temp_d1 / vmax * 100
    temp_d2 = temp_d2 / vmax * 100

    temp_d1_max = temp_d1.max()
    temp_d2_max = temp_d2.max()

    norm = LogNorm(vmin=7e-4, vmax=100)

    fig = plt.figure(figsize=(9, 5.5))
    gs = GridSpec(2, 4, width_ratios=[1.25, 1.25, 1.25, 0.15])

    ax0  = fig.add_subplot(gs[0,0])
    ax1  = fig.add_subplot(gs[0,1])
    ax2  = fig.add_subplot(gs[0,2])
    cax1 = fig.add_subplot(gs[0,3])

    ax3  = fig.add_subplot(gs[1,0])
    ax4  = fig.add_subplot(gs[1,1])
    ax5  = fig.add_subplot(gs[1,2])
    cax2 = fig.add_subplot(gs[1,3])

    # Calibrated
    ax0.imshow(temp_0, norm=norm, cmap=cmap)
    ax0.set_title('On-sky PSF')
    ax0.axis('off')

    ax1.imshow(temp_1, norm=norm, cmap=cmap)
    ax1.set_title('Calibrated')
    ax1.axis('off')

    img1 = ax2.imshow(temp_d1, norm=norm, cmap=cmap)
    ax2.set_title('Abs. difference')
    ax2.axis('off')

    ax2.text(100, 180, f'Max. {temp_d1_max:.1f}%', color='white', fontsize=12, ha='center', va='center')

    cbar1 = fig.colorbar(img1, cax=cax1, orientation='vertical')
    cbar1.set_label('Relative intensity [%]')

    # Direct
    ax3.imshow(temp_0, norm=norm, cmap=cmap)
    ax3.set_title('On-sky PSF')
    ax3.axis('off')

    ax4.imshow(temp_2, norm=norm, cmap=cmap)
    ax4.set_title('Direct (tuned)')
    ax4.axis('off')

    img2 = ax5.imshow(temp_d2, norm=norm, cmap=cmap)
    ax5.set_title('Abs. difference')
    ax5.axis('off')

    ax5.text(100, 180, f'Max. {temp_d2_max:.1f}%', color='white', fontsize=12, ha='center', va='center')

    cbar2 = fig.colorbar(img2, cax=cax2, orientation='vertical')
    cbar2.set_label('Relative intensity [%]')


    plt.suptitle(muse_df.loc[ids[ii], 'name'])
    # plt.suptitle(f'ID: {}')
    # plt.tight_layout()
    if save_dir is not None:
        plt.savefig(save_dir+f'PSF_{ids[ii]}.pdf', dpi=300)
    else:
        plt.show()


good_ids = [123, 298, 94, 180, 110, 86, 292, 125, 100]

for id in tqdm(good_ids):
    plot_MUSE_PSF(id, save_dir='/home/akuznets/Projects/TipTorch/data/temp/plots/MUSE_PSFs/')


#%%

# Read the results
with open('../data/temp/PSFs_val_data_polychrome.pickle', 'rb') as handle:
    PSFs_0_val_poly_data = pickle.load(handle)
    
    
with open('../data/temp/PSFs_val_calibrated_polychrome.pickle', 'rb') as handle:
    PSFs_1_val_poly_calib = pickle.load(handle)


with open('../data/temp/PSFs_val_tuned_polychrome.pickle', 'rb') as handle:
    PSFs_2_val_poly_tuned = pickle.load(handle)


with open('../data/temp/PSFs_val_direct_polychrome_raw.pickle', 'rb') as handle:
    PSFs_2_val_poly_direct = pickle.load(handle)
    
# with open('../data/temp/PSFs_val_fitted_polychrome.pickle', 'wb') as handle:
#     PSFs_3_val_poly = pickle.load(handle)


#%%
from tools.utils import FWHM_fitter, FitMoffat2D_astropy, FitGauss2D_astropy

FWHM_data   = FWHM_fitter(PSFs_0_val_poly_data,   verbose=True)
FWHM_calib  = FWHM_fitter(PSFs_1_val_poly_calib,  verbose=True)
FWHM_tuned  = FWHM_fitter(PSFs_2_val_poly_tuned,  verbose=True)
FWHM_direct = FWHM_fitter(PSFs_2_val_poly_direct, verbose=True)

#%%
ROI_ = np.s_[np.random.randint(0,77), 0, 199//2-25:199//2+25, 199//2-25:199//2+25]

A = PSFs_0_val_poly[ROI_]
B = PSFs_1_val_poly[ROI_]

F_x_A, F_y_A, A_fit, A_ = FitMoffat2D_astropy(A)
F_x_B, F_y_B, B_fit, B_ = FitMoffat2D_astropy(B)

max_point = np.unravel_index(np.argmax(A_), A_.shape)

F_A = np.sqrt(F_x_A**2 + F_y_A**2)
F_B = np.sqrt(F_x_B**2 + F_y_B**2)

#%
plt.title(f'FWHM: {F_A:.2f} -> {F_B:.2f}, rel. diff. {np.abs(F_A-F_B)/F_A*100:.1f}%')
plt.legend(['Data', 'Calibrated'])
plt.plot(A_[max_point[0],...], color='blue')
plt.plot(B_[max_point[0],...], color='red')
plt.plot(A_fit[max_point[0],...], color='blue', linestyle='--')
plt.plot(B_fit[max_point[0],...], color='red', linestyle='--')
plt.show()


#%%
FWHMy = lambda FWHM, l: np.sqrt(FWHM[:,l,0]**2 + FWHM[:,l,1]**2)
FWHMy_white = lambda FWHM: np.mean(np.array([FWHMy(FWHM, l) for l in range(FWHM.shape[1])]), axis=0)

x = FWHMy_white(FWHM_data)
y = FWHMy_white(FWHM_calib)
z = FWHMy_white(FWHM_tuned)
w = FWHMy_white(FWHM_direct)

# plt.plot(np.linspace(0, x.max(), 100), np.linspace(0, x.max(), 100), 'k--')
# plt.scatter(x, y)
# plt.ylabel('Direct')
# plt.xlabel('Data')
# plt.xlim([0, 10])
# plt.ylim([0, 10])
# plt.axis('equal')
# plt.show()

# Compute relative FWHM error
relative_err_calib = np.abs(x-y) / x * 100
absolute_err_calib = np.abs(x-y) * 25

relative_err_tuned = np.abs(x-z) / x * 100
absolute_err_tuned = np.abs(x-z) * 25

relative_err_direct = np.abs(x-w) / x * 100
absolute_err_direct = np.abs(x-w) * 25

print(f'Median relative FWHM error (calib): {np.median(relative_err_calib):.1f}%')
print(f'Median absolute FWHM error (calib): {np.median(absolute_err_calib):.1f} [mas]')

print(f'Median relative FWHM error (tuned): {np.median(relative_err_tuned):.1f}%')
print(f'Median absolute FWHM error (tuned): {np.median(absolute_err_tuned):.1f} [mas]')

print(f'Median relative FWHM error (direct): {np.median(relative_err_direct):.1f}%')
print(f'Median absolute FWHM error (direct): {np.median(absolute_err_direct):.1f} [mas]')

#%%
from tools.utils import hist_thresholded

hist_thresholded(
    datasets=[absolute_err_calib, absolute_err_tuned],
    threshold=np.round( np.percentile(absolute_err_calib, 90)),
    bins=10,
    title="Absolute FWHM error",
    xlabel=r"$\Delta\:$FWHM, [mas]",
    ylabel="Percentage, [%]",
    labels=['Caibrated', 'Tuned'],
    colors=None,
    alpha=0.6
)
# plt.savefig(save_dir+'FWHM_absolute_MUSE.pdf', dpi=300)


hist_thresholded(
    datasets=[relative_err_calib, relative_err_tuned],
    threshold=np.round( np.percentile(relative_err_calib, 90)),
    bins=10,
    title="Relative FWHM error",
    xlabel=r"$\Delta\:$FWHM $\, / \,$ FWHM$_{\: data}$, [%]",
    ylabel="Percentage, [%]",
    labels=['Caibrated', 'Tuned'],
    colors=None,
    alpha=0.6
)
# plt.savefig(save_dir+'FWHM_relative_MUSE.pdf', dpi=300)

#%%
# PSFs_0_val_poly_data, 
# PSFs_1_val_poly_calib,
# PSFs_2_val_poly_tuned,
# PSFs_2_val_poly_direct

var_data  = np.var(PSFs_0_val_poly_data.mean(axis=1), axis=(-2,-1))
var_delta = np.var((PSFs_1_val_poly_calib - PSFs_0_val_poly_data).mean(axis=1), axis=(-2,-1))

FVU = var_delta / var_data

print(np.median(FVU))

# plt.hist(FVU, bins=100)

#%%
var_data  = np.var(PSFs_0_val_poly_data.mean(axis=1), axis=(-2,-1))
var_delta = np.var((PSFs_2_val_poly_tuned - PSFs_0_val_poly_data).mean(axis=1), axis=(-2,-1))

FVU = var_delta / var_data

print(np.median(FVU))

#%%

var_data  = np.var(PSFs_0_val_poly_data.mean(axis=1), axis=(-2,-1))
var_delta = np.var((PSFs_2_val_poly_direct - PSFs_0_val_poly_data).mean(axis=1), axis=(-2,-1))

FVU = var_delta / var_data

print(np.median(FVU))


#%%
import seaborn as sns
from tools.utils import r0
from matplotlib.ticker import MaxNLocator

seeings = np.array( [seeing for config in configs for seeing in config['atmosphere']['Seeing'].tolist()] )
r0_init_stats = r0(seeings, 500e-9)

r0_stats    = np.array( [x.item()   for pred_inputs in pred_inputs_stats for x in pred_inputs['r0']] )
dn_stats    = np.array( [x.item()   for pred_inputs in pred_inputs_stats for x in pred_inputs['dn']] )
s_pow_stats = np.array( [x.item()   for pred_inputs in pred_inputs_stats for x in pred_inputs['s_pow']] )
amp_stats   = np.array( [x.item()   for pred_inputs in pred_inputs_stats for x in pred_inputs['amp']] )
b_stats     = np.array( [x.item()   for pred_inputs in pred_inputs_stats for x in pred_inputs['b']] )
alpha_stats = np.array( [x.item()   for pred_inputs in pred_inputs_stats for x in pred_inputs['alpha']] )
F_stats     = np.array( [x.tolist() for pred_inputs in pred_inputs_stats for x in pred_inputs['F']] )
Jx_stats    = np.array( [x.tolist() for pred_inputs in pred_inputs_stats for x in pred_inputs['Jx']] )
Jy_stats    = np.array( [x.tolist() for pred_inputs in pred_inputs_stats for x in pred_inputs['Jy']] )
J_stats     = np.sqrt(Jx_stats**2 + Jy_stats**2)

# Create grid plot for 8 plots
fig, ax = plt.subplots(2, 4, figsize=(12, 6.5))

# Plot r0
sns.histplot(r0_stats, fill=True, label=r'$r_0$ (pred.)', ax=ax[0,0], zorder=-1, element='step', alpha=0.35)
sns.histplot(r0_init_stats, fill=True, label=r'$r_0$ (data.)', ax=ax[0,0], zorder=-2, element='step', alpha=0.35)
ax[0,0].legend(loc='upper right')

sns.histplot(dn_stats, fill=True, label=r'$r_0$', ax=ax[0,1], zorder=-1, element='step', alpha=0.4)
sns.histplot(alpha_stats, fill=True, label=r'$\alpha_M$', ax=ax[0,2], zorder=-1, element='step', alpha=0.4)
sns.histplot(s_pow_stats, fill=True, label=r'$a_b$', ax=ax[0,3], zorder=-1, element='step', alpha=0.4)
sns.histplot(amp_stats, fill=True, label='A', ax=ax[1,0], zorder=-1, element='step', alpha=0.4)
sns.histplot(b_stats, fill=True, label='b', ax=ax[1,1], zorder=-1, element='step', alpha=0.4)

for i in range(2):
    for j in range(4):
        ax[i,j].grid(zorder=-10, alpha=0.5)
        ax[i,j].yaxis.set_major_locator(MaxNLocator(integer=True))

colors  = sns.color_palette("rainbow", J_stats.shape[1])

for i in range(F_stats.shape[1]):
    sns.histplot(F_stats[:, i], fill=True, color=colors[i], ax=ax[1,2], label=f'{wavelength_selected[ids_wavelength_selected][i]:.0f} nm', zorder=-i+1, element='step', alpha=0.05)
    ax[1,2].legend(title='Wavelength', loc='upper left')

for i in range(J_stats.shape[1]):
    sns.histplot(J_stats[:, i], fill=True, color=colors[i], ax=ax[1,3], label=f'{wavelength_selected[ids_wavelength_selected][i]:.0f} nm', zorder=-i+1, element='step', alpha=0.05)
    # ax[1,3].legend(title='Wavelength', loc='upper left')

ax[0,0].set_xlabel(r'$r_0$, [m]')
ax[0,1].set_xlabel(r'$\Delta n$, [rad$^2$]')
ax[0,2].set_xlabel(r'$\alpha_M$, [a.u.]')
ax[0,3].set_xlabel(r'$a_b$, [a.u.]')
ax[1,0].set_xlabel('A, [a.u.]')
ax[1,1].set_xlabel('b, [a.u.]')
ax[1,2].set_xlabel('F, [a.u.]')
ax[1,3].set_xlabel('J, [mas]')

ax[0,1].set_ylabel('')
ax[0,2].set_ylabel('')
ax[0,3].set_ylabel('')
ax[1,1].set_ylabel('')
ax[1,2].set_ylabel('')
ax[1,3].set_ylabel('')

plt.suptitle('Predicted parameters distribution')
plt.tight_layout()
plt.subplots_adjust(hspace=0.3)  # Increase the space between rows
plt.savefig(save_dir+'predicted_params_MUSE.pdf', dpi=300)

#%%


