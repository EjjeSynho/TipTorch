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
toy.to_float()
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
predicted_entries = ['r0', 'F', 'dn', 'Jx', 'Jy', 's_pow', 'amp', 'b', 'alpha']

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


def func(x_, fixed_inputs):
    pred_inputs = transformer.destack(net(x_))
    return run_model(toy, batch_init, pred_inputs, fixed_inputs)


def loss_fn(x_, PSF_data):
    loss = img_punish(func(x_, fixed_inputs), PSF_data)
    return loss

#%%
x0, fixed_inputs, PSF_0, current_config = get_data(batch_init, fixed_entries)
toy.config = current_config
toy.Update(reinit_grids=True, reinit_pupils=True)

batch_size = len(batch_init['IDs'])

#%%

groups = [6, 8, 10, 9, 3]

batches_train, batches_val = {}, {}
train_ids, val_ids = {}, {}

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
optimizer = optim.Adam(net.parameters(), lr=0.0001)

loss_stats_train, loss_stats_val = [], []

epochs = 50

loss_train, loss_val = [], []
       
net.train()

wvl_L, wvl_R = 1625, 1625
train_ids = train_ids[wvl_L]
val_ids   = val_ids[wvl_L]

for epoch in range(epochs):
    print(f'>>>>>>>>> Epoch: {epoch+1}/{epochs}')
    
    epoch_train_loss, epoch_val_loss = [], []
    
    np.random.shuffle(train_ids)

    for i, id in enumerate(train_ids):
        optimizer.zero_grad()
        
        batch_init = batches_train_all[wvl_L][id]

        x0, fixed_inputs, PSF_0, current_config = get_data(batch_init, fixed_entries)
        batch_size = len(batch_init['IDs'])

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
            batch_init = batches_val[wvl_L][id]
            
            x0, fixed_inputs, PSF_0, current_config = get_data(batch_init, fixed_entries)
            toy.config = current_config
            toy.Update(reinit_grids=True, reinit_pupils=True)

            batch_size = len(batch_init['IDs'])

            loss = loss_fn(x0, PSF_0)
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
        train_ids = train_ids[wvl_L]
        np.random.shuffle(train_ids)

        for i, id in enumerate(train_ids):
            optimizer.zero_grad()
            
            batch_init = batches_train_all[wvl_L][id]

            x0, fixed_inputs, PSF_0, current_config = get_data(batch_init, fixed_entries)
            # toy = TipTorch(current_config, None, device)
            toy.config = current_config
            toy.Update(reinit_grids=True, reinit_pupils=True)
            
            batch_size = len(batch_init['IDs'])

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
        val_ids = val_ids[wvl_L]
        np.random.shuffle(val_ids)
        
        for i, id in enumerate(val_ids):
            with torch.no_grad():
                batch_init = batches_val[wvl_L][id]
                
                x0, fixed_inputs, PSF_0, current_config = get_data(batch_init, fixed_entries)
                # toy = TipTorch(current_config, None, device)
                toy.config = current_config
                toy.Update(reinit_grids=True, reinit_pupils=True)

                batch_size = len(batch_init['IDs'])

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
        val_ids = val_ids[wvl]

        for i in tqdm(val_ids):
            # ------------------------- Validate calibrated -------------------------
            batch_init = batches_val[wvl][i]
            
            x0, fixed_inputs, PSF_0, init_config = get_data(batch_init, fixed_entries)
            toy.config = init_config
            toy.Update(reinit_grids=True, reinit_pupils=True)

            batch_size = len(batch_init['IDs'])
            
            fixed_inputs['Jxy'] *= 0
            
            fixed_inputs['basis_coefs'] = predict_LWE(batch_init['IDs'])
            
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
            
            current_batch_size = len(batch_init['IDs'])

            for key, value in inputs.items():
                inputs[key] = value.float().to(device).repeat(current_batch_size, 1).squeeze()
            
            PSFs_2_val.append(run_model(toy, batch_init, inputs).cpu())

            # ------------------------- Validate fitted -------------------------
            fitted_dict = get_fixed_inputs(batch_init, fitted_entries)

            PSFs_3_val.append(run_model(toy, batch_init, {}, fixed_inputs=fitted_dict).cpu())


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
i = np.random.choice(val_ids[wvl])

batch_init = batches_val[wvl][i]

j = np.random.choice(batch_init['IDs'])
index_ = batch_init['IDs'].tolist().index(j)

x0, fixed_inputs, _, _ = get_data(batch_init, fitted_entries)

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

val_ids = val_ids[wvl]

fitted_entries = ['F L', 'F R', 'bg L', 'bg R', 'dx L', 'dx R', 'dy L', 'dy R', 'r0', 'Jx', 'Jy', 'Jxy', 'dn', 'Wind dir', 'LWE coefs']

batch_directory = SPHERE_DATASET_FOLDER + 'batch_test.pkl'


with open(batch_directory, 'rb') as handle:
    batch_init = pickle.load(handle)


with torch.no_grad():
    # ------------------------- Validate calibrated -------------------------
    x0, fixed_inputs, PSF_0, init_config = get_data(batch_init, fitted_entries)
    toy.config = init_config
    toy.Update(reinit_grids=True, reinit_pupils=True)
    
    PSFs_data.append(PSF_0.cpu())
    fitted_dict = get_fixed_inputs(batch_init, fitted_entries)
    PSFs_fitted.append(run_model(toy, batch_init, {}, fixed_inputs=fitted_dict).cpu())

PSFs_data   = torch.cat(PSFs_data,   dim=0)[:,0,...].numpy()
PSFs_fitted = torch.cat(PSFs_fitted, dim=0)[:,0,...].numpy()

# plot_radial_profiles_new(PSFs_data, PSFs_fitted, 'Data', 'TipTorch', title='Fitted')

index_ = batch_init['IDs'].index(2818)

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
