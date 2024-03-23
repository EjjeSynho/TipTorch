#%%
%reload_ext autoreload
%autoreload 2

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
from tools.utils import plot_radial_profiles, EarlyStopping, LWE_basis, cropper, SR, draw_PSF_stack, rad2mas, pdims, mask_circle
from PSF_models.TipToy_SPHERE_multisrc import TipTorch
from data_processing.SPHERE_preproc_utils import SPHERE_preprocess, SamplesByIds
from tools.config_manager import GetSPHEREonsky, ConfigManager
from project_globals import SPHERE_DATA_FOLDER, SPHERE_DATASET_FOLDER, device
from data_processing.normalizers import InputsTransformer, CreateTransformSequenceFromFile

df_transforms_onsky  = CreateTransformSequenceFromFile('../data/temp/psf_df_norm_transforms.pickle')
df_transforms_fitted = CreateTransformSequenceFromFile('../data/temp/fitted_df_norm_transforms.pickle')

# device = torch.device('cpu')

#%%
# Here, all possible transormations are inputted
transformer = InputsTransformer({
    'F':  df_transforms_fitted['F L'], 'bg':  df_transforms_fitted['bg L'], # bg R and bg R, as well as F are the same
    'r0': df_transforms_fitted['r0'],  'dx':  df_transforms_fitted['dx'],
    'dy': df_transforms_fitted['dy'],  'Jx':  df_transforms_fitted['Jx'],
    'Jy': df_transforms_fitted['Jy'],  'Jxy': df_transforms_fitted['Jxy'],
    'basis_coefs': df_transforms_fitted['LWE coefs']
})

# Here, only the variables that need to be predicted must be added
inp_dict = {
    'r0':  torch.zeros([1]),
    'F':   torch.zeros([1, 2]),
    'dx':  torch.zeros([1]),
    'dy':  torch.zeros([1]),
    'bg':  torch.zeros([1, 2]),
    'Jx':  torch.zeros([1]),
    'Jy':  torch.zeros([1]),
    'Jxy': torch.zeros([1]),
    # 'basis_coefs': torch.zeros([1, 11])
}
_ = transformer.stack(inp_dict) # to create index mapping

# test = torch.zeros([11, transformer.get_packed_size()])
# test2 = transformer.destack(test)

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

    x_unpacked = predicted_inputs | fixed_inputs # It ovverides the values from the destacked dictionary
    if 'basis_coefs' in x_unpacked:
        return model(x_unpacked, None, lambda: basis(x_unpacked['basis_coefs'].float()))
    else:
        return model(x_unpacked)


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
# toy.optimizables = []

basis = LWE_basis(toy)

config_manager = ConfigManager(GetSPHEREonsky())

batch_directory = SPHERE_DATASET_FOLDER + 'train/batch_0_train_grp_6.pkl'
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

#%%
'''
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
'''

A = torch.randn(
    [ batch_data['NN input'].shape[1], transformer.get_packed_size() ],
    device = device,
    requires_grad = True
)

#%%
x0 = [1.0,
      1.0, 1.0,
      0.0,
      0.0,
      0.0, 0.0,
      0.5,
      0.5,
      0.1] #+ [0,]*3 + [0,]*8

x0 = torch.tensor(x0).float().repeat(len(batch_data['IDs']), 1).to(device)
x0.requires_grad = True

#%%
optimizer = optim.LBFGS([x0], lr=10, history_size=20, max_iter=4, line_search_fn="strong_wolfe")
# optimizer = optim.LBFGS([A], lr=10, history_size=20, max_iter=4, line_search_fn="strong_wolfe")
# optimizer = optim.Adam([A], lr=0.1)  # Learning rate is set to 0.1

# x0 = batch_data['NN input'].float().to(device)
PSF_0 = batch_data['PSF_0'].float().to(device)
additional_inputs = get_fixed_inputs(batch_data, ['dx', 'dy', 'bg L', 'bg R'])

init_config = batch_data['configs']
config_manager.Convert(init_config, framework='pytorch', device=device)

toy.config = init_config
toy.Update(reinit_grids=True, reinit_pupils=False)

def func(x_):
    # X_ = x_ @ A
    # pred_inputs = transformer.destack(X_)
    pred_inputs = transformer.destack(x_)
    return run_model(toy, batch_data, pred_inputs)#, additional_inputs)

early_stopping = EarlyStopping(patience=2, tolerance=1e-4, relative=False)

crop_all = cropper(PSF_0, 100)

def loss_fn(A, B):
    return nn.L1Loss(reduction='sum')(A[crop_all], B[crop_all])

#%%
num_iter = 100

for i in range(num_iter):
    optimizer.zero_grad()
    batch_size = len(batch_data['IDs'])
    loss = loss_fn(func(x0), PSF_0)

    if np.isnan(loss.item()): break
    
    early_stopping(loss)

    loss.backward(retain_graph=True)
    optimizer.step( lambda: loss_fn(func(x0), PSF_0) )

    print(f'Loss ({i+1}/{num_iter}): {loss.item()/batch_size}', end="\r", flush=True)

    if early_stopping.stop:
        print(f'Stopped at it. {i} with loss value: {loss.item()/batch_size}', flush=True)
        break

#%%

np.save('../data/temp/A_matrix_LWE.npy', A.detach().cpu().numpy())

#%%

with torch.no_grad():
    PSF_1 = func(x0)
    destack = lambda PSF_stack: [ x for x in np.split(PSF_stack[:,0,...], PSF_stack.shape[0], axis=0) ]
    plot_radial_profiles(destack(PSF_0),  destack(PSF_1),  'Data', 'Predicted', title='Matrix fitting', dpi=200, cutoff=32, scale='log')
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
