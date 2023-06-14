#%%
%reload_ext autoreload
%autoreload 2

import sys
sys.path.append('..')

from project_globals import SPHERE_DATA_FOLDER, WEIGHTS_FOLDER, device
from data_processing.SPHERE_preproc_utils import SPHERE_preprocess
from data_processing.normalizers import Uniform, TransformSequence
from PSF_models.TipToy_SPHERE_multisrc import TipTorch
from tools.utils import seeing, plot_radial_profiles
from torch import nn, optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pickle
import torch
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
# psf_df = psf_df[psf_df['λ left (nm)'] == 1625]
psf_df = psf_df[psf_df['λ left (nm)'] < 1626]
psf_df = psf_df[psf_df['λ left (nm)'] > 1624]

psf_df['Wind direction (200 mbar)'].fillna(psf_df['Wind direction (header)'], inplace=True)
psf_df['Wind speed (200 mbar)'].fillna(psf_df['Wind speed (header)'], inplace=True)
psf_df.rename(columns={'Filter WFS': 'λ WFS (nm)'}, inplace=True)
psf_df.rename(columns={'Nph WFS': 'Nph WFS (data)'}, inplace=True)
psf_df.rename(columns={'r0 (SPARTA)': 'r0 (data)'}, inplace=True)

#%%
# ids_class_C = set(psf_df.index[psf_df['Class C'] == True])
# ids_wvls = set(psf_df.index[psf_df['λ left (nm)'] > 1600]).intersection(set(psf_df.index[psf_df['λ left (nm)'] < 1700]))
# ids_to_exclude_later = ids_class_C.union(ids_wvls)

good_ids = psf_df.index.values.tolist()
print(len(good_ids), 'samples are in the dataset')

#% Select the entries to be used in training
selected_entries = ['Airmass',
                    'r0 (data)',
                    'FWHM',
                    'Strehl',
                    'Wind direction (header)',
                    'Wind speed (header)',
                    'Wind direction (200 mbar)',
                    'Wind speed (200 mbar)',
                    'Tau0 (header)',
                    'Flux WFS',
                    'Nph WFS (data)',
                    'λ WFS (nm)',
                    'Rate',
                    'Jitter X',
                    'Jitter Y',
                    'λ left (nm)',
                    'λ right (nm)']

psf_df = psf_df[selected_entries]
psf_df.sort_index(inplace=True)

#%% Create fitted parameters dataset
#check if file exists
if not os.path.isfile('E:/ESO/Data/SPHERE/fitted_df.pickle'):
    fitted_dict_raw = {key: [] for key in ['F', 'dx', 'dy', 'r0', 'n', 'dn', 'bg', 'Jx', 'Jy', 'Jxy', 'Nph WFS', 'SR data', 'SR fit']}
    ids = []

    images_data = []
    images_fitted = []

    fitted_folder = 'E:/ESO/Data/SPHERE/IRDIS_fitted_1P21I/'
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
    
    fitted_dict['SR data (left)']  = fitted_dict['SR data'][:,0]
    fitted_dict['SR data (right)'] = fitted_dict['SR data'][:,1]
    fitted_dict['SR fit (left)']   = fitted_dict['SR fit'][:,0]
    fitted_dict['SR fit (right)']  = fitted_dict['SR fit'][:,1]
    
    fitted_dict.pop('F')
    fitted_dict.pop('bg')

    for key in fitted_dict.keys():
        fitted_dict[key] = fitted_dict[key].tolist()

    fitted_df = pd.DataFrame(fitted_dict)
    fitted_df.set_index('ID', inplace=True)

    # Save dataframe
    with open('E:/ESO/Data/SPHERE/fitted_df.pickle', 'wb') as handle:
        pickle.dump(fitted_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
else:
    with open('E:/ESO/Data/SPHERE/fitted_df.pickle', 'rb') as handle:
        print('Loading dataframe "fitted_df.pickle"...')
        fitted_df = pickle.load(handle)

fitted_ids = list( set( fitted_df.index.values.tolist() ).intersection( set(psf_df.index.values.tolist()) ) )
fitted_df = fitted_df[fitted_df.index.isin(fitted_ids)]

for entry in ['r0','Jx','Jy','Jxy','F (left)','F (right)']:
    fitted_df[entry] = fitted_df[entry].abs()

fitted_df['dn'] = fitted_df['dn'].abs()
fitted_df['n']  = fitted_df['n'].abs()
fitted_df['Rec. noise'] = np.sqrt((fitted_df['n']+fitted_df['dn']).abs())*psf_df['λ WFS (nm)']/2/np.pi * 1e9
fitted_df.rename(columns={'r0': 'r0 (fit)'}, inplace=True)
fitted_df.rename(columns={'Nph WFS': 'Nph WFS (fit)'}, inplace=True)
fitted_df['SR fit']  = 0.5 * (fitted_df['SR fit (left)'] + fitted_df['SR fit (right)'])
fitted_df['SR data'] = 0.5 * (fitted_df['SR data (left)'] + fitted_df['SR data (right)'])
fitted_df['J']       = np.sqrt(fitted_df['Jx'].pow(2) + fitted_df['Jy'].pow(2))
fitted_df['F']       = 0.5 * (fitted_df['F (left)'] + fitted_df['F (right)'])

ids = fitted_df.index.intersection(psf_df.index)
fitted_df = fitted_df.loc[ids]
synth_df = psf_df.loc[ids]

df = pd.concat([synth_df, fitted_df], axis=1).fillna(0)

#%% Compute data transformations

transforms_input = {}
transforms_input['Tau0 (header)']             = TransformSequence( transforms = [ Uniform(a=0.0, b=0.025) ])
transforms_input['λ left (nm)']               = TransformSequence( transforms = [ Uniform(a=psf_df['λ left (nm)'].min(), b=psf_df['λ left (nm)'].max()) ])
transforms_input['λ right (nm)']              = TransformSequence( transforms = [ Uniform(a=psf_df['λ right (nm)'].min(), b=psf_df['λ right (nm)'].max()) ])
transforms_input['FWHM']                      = TransformSequence( transforms = [ Uniform(a=0.5, b=3.0) ])
transforms_input['Jitter X']                  = TransformSequence( transforms = [ Uniform(a=0.0, b=60.0) ])
transforms_input['Jitter Y']                  = TransformSequence( transforms = [ Uniform(a=0.0, b=60.0) ])
transforms_input['Airmass']                   = TransformSequence( transforms = [ Uniform(a=1.0, b=1.5) ])
transforms_input['Strehl']                    = TransformSequence( transforms = [ Uniform(a=0,   b=1)] )
transforms_input['r0 (data)']                 = TransformSequence( transforms = [ Uniform(a=0.1, b=0.5)] )
transforms_input['Rate']                      = TransformSequence( transforms = [ Uniform(a=0,   b=1380)] )
transforms_input['Nph WFS (data)']            = TransformSequence( transforms = [ Uniform(a=0,   b=200)] )
transforms_input['Flux WFS']                  = TransformSequence( transforms = [ Uniform(a=0,   b=2000)] )
transforms_input['Wind speed (header)']       = TransformSequence( transforms = [ Uniform(a=0,   b=20)] )
transforms_input['Wind direction (header)']   = TransformSequence( transforms = [ Uniform(a=0,   b=360)] )
transforms_input['Wind direction (200 mbar)'] = TransformSequence( transforms = [ Uniform(a=0,   b=360)] ) 
transforms_input['Wind speed (200 mbar)'   ]  = TransformSequence( transforms = [ Uniform(a=0,   b=70)] )

transforms_output = {}
transforms_output['dx']              = TransformSequence( transforms = [ Uniform(a=-0.5, b=0.5) ] )
transforms_output['dy']              = TransformSequence( transforms = [ Uniform(a=-0.5, b=0.5) ] )
transforms_output['r0 (fit)']        = TransformSequence( transforms = [ Uniform(a=0.0,  b=1.0) ] )
transforms_output['n']               = TransformSequence( transforms = [ Uniform(a=0.0,  b=17.0 ) ] )
transforms_output['dn']              = TransformSequence( transforms = [ Uniform(a=0, b=50) ] )
transforms_output['Rec. noise']      = TransformSequence( transforms = [ Uniform(a=0, b=1250) ] )
transforms_output['J']               = TransformSequence( transforms = [ Uniform(a=0, b=40) ] )
transforms_output['Jx']              = TransformSequence( transforms = [ Uniform(a=0, b=40) ] )
transforms_output['Jy']              = TransformSequence( transforms = [ Uniform(a=0, b=40) ] )
transforms_output['Jxy']             = TransformSequence( transforms = [ Uniform(a=0, b=300) ] )
transforms_output['Nph WFS (fit)']   = TransformSequence( transforms = [ Uniform(a=0, b=300) ] )
transforms_output['F']               = TransformSequence( transforms = [ Uniform(a=0, b=1.5) ] )
transforms_output['F (left)']        = TransformSequence( transforms = [ Uniform(a=0, b=1.5) ] )
transforms_output['F (right)']       = TransformSequence( transforms = [ Uniform(a=0, b=1.5) ] )
transforms_output['bg (left)']       = TransformSequence( transforms = [ Uniform(a=-1e-5, b=1e-5) ] )
transforms_output['bg (right)']      = TransformSequence( transforms = [ Uniform(a=-1e-5, b=1e-5) ] )
transforms_output['SR data']         = TransformSequence( transforms = [ Uniform(a=0.0, b=1.0) ] )
transforms_output['SR fit']          = TransformSequence( transforms = [ Uniform(a=0.0, b=1.0) ] )
transforms_output['SR data (left)']  = TransformSequence( transforms = [ Uniform(a=0.0, b=1.0) ] )
transforms_output['SR data (right)'] = TransformSequence( transforms = [ Uniform(a=0.0, b=1.0) ] )
transforms_output['SR fit (left)']   = TransformSequence( transforms = [ Uniform(a=0.0, b=1.0) ] )
transforms_output['SR fit (right)']  = TransformSequence( transforms = [ Uniform(a=0.0, b=1.0) ] )

transforms = {**transforms_input, **transforms_output}

trans_df = df.copy()
for entry in transforms:
    trans_df[entry] = transforms[entry].forward(trans_df[entry])

#%%
selected_X = ['r0 (data)',
              'Rate',
              'Nph WFS (data)',
              'Wind speed (header)',
              'Wind direction (header)',
            #   'Flux WFS',
              'Wind speed (200 mbar)',
              'Wind direction (200 mbar)']

selected_Y = [
    'Rec. noise',
    'Jx',
    'Jy',
    'F (left)',
    'F (right)']

NN2in  = lambda X: { selected: transforms[selected].backward(X[:,i]) for i,selected in enumerate(selected_X) }
NN2fit = lambda Y: { selected: transforms[selected].backward(Y[:,i]) for i,selected in enumerate(selected_Y) }
in2NN  = lambda inp: torch.from_numpy(( np.stack([transforms[a].forward(inp[a].values) for a in selected_X]))).T
fit2NN = lambda out: torch.from_numpy(( np.stack([transforms[a].forward(out[a].values) for a in selected_Y]))).T

for entry in selected_X:
    trans_df = trans_df[trans_df[entry].abs() < 3]
    # print(trans_df[entry].abs() < 3)

for entry in selected_Y:
    trans_df = trans_df[trans_df[entry].abs() < 3]

#%%
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
      
    
def prepare_batch_configs(batches):
    batches_dict = []
    for i in tqdm(range(len(batches))):
        sample_ids = batches[i].index.tolist()
        PSF_0, _, _, _, config_files = SPHERE_preprocess(sample_ids, 'different', 'sum', device)

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

batch_test = prepare_batch_configs([df.iloc[:64]])

#%%
# Load the model
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

net = torch.load(WEIGHTS_FOLDER/f'param_predictor_{len(selected_X)}x{len(selected_Y)}.pth')




#%%
toy = TipTorch(batch_test[0]['configs'], 'sum', device, TipTop=True, PSFAO=False)
toy.optimizables = []

# y_pred = torch.from_numpy(mlp.predict(batch_test[0]['X'])).float()
y_pred = net(batch_test[0]['X'].float().to(device))
# y_pred = batch_test[0]['Y']

PSF_0 = batch_test[0]['PSF (data)'].cpu().numpy()
PSF_2 = toy_run( toy, batch_test[0]['df'], NN2fit(y_pred.to(device)) )

destack = lambda PSF_stack: [ x for x in np.split(PSF_stack[:,0,...], PSF_stack.shape[0], axis=0) ]
plot_radial_profiles(destack(PSF_0),  destack(PSF_2),  'Data', 'Predicted', title='Synth prediction accuracy worked', dpi=200, cutoff=32, scale='log')



#%%
from project_globals import WEIGHTS_FOLDER

# Assume we have some input data in a pandas DataFrame
# inputs and targets should be your actual data
loss_trains = []
loss_vals   = []

# Training loop
num_iters = 200

for epoch in range(num_iters):  # number of epochs
    optimizer.zero_grad()   # zero the gradient buffers
    y_pred = net(X_train.to(device))   # forward pass
    loss = loss_fn(y_pred, Y_train_1.to(device))  # compute loss
    loss_trains.append(loss.detach().cpu().item())
    loss.backward()  # backpropagation
    optimizer.step()  # update weights
    print('Epoch [%d/%d], Loss: %.4f' % (epoch+1, num_iters, loss.item()))

    with torch.no_grad():
        y_pred_val = net(X_valid.to(device))
        loss_val = loss_fn(y_pred_val, Y_valid_1.to(device))
        loss_vals.append(loss_val.detach().cpu().item())
        # print('Epoch [%d/%d], Val Loss: %.4f' % (epoch+1, 100, loss_val.item()))

loss_trains = np.array(loss_trains)
loss_vals   = np.array(loss_vals)

# %
plt.plot(loss_trains, label='Train')
plt.plot(loss_vals, label='Val')
plt.legend()
plt.grid()

torch.save(net.state_dict(), WEIGHTS_FOLDER/'gnosis2_TT_weights.dict')

torch.cuda.empty_cache()

#%% One shot validation

with torch.no_grad():
    Y_pred = gnosis2PAO_1(net(X_valid.to(device)))
    Y_test = gnosis2PAO_1(Y_valid_1.to(device))

    for key in Y_pred.keys():
        
        A = Y_pred[key].detach().cpu().numpy()
        B = Y_test[key].detach().cpu().numpy()
        
        m = np.mean(np.abs(A-B)/A * 100)
        s = np.std( np.abs(A-B)/A * 100)
        
        print(key, end=': ')
        print('{:.2f} +- {:.2f}'.format(m, s))
    
#%
# with torch.no_grad():
#     Y_pred = net(X_valid.to(device))
    
# i = 3
# plt.title(fitted_subset[i])
# datafrafa = pd.DataFrame({'Y_valid':Y_valid_1[:,i].detach().cpu().numpy(), 'Y_pred':Y_pred[:,i].detach().cpu().numpy()})
# # plt.scatter(Y_valid[:,i].detach().cpu().numpy(), Y_pred[:,i].detach().cpu().numpy(), s=1)
# plt.plot([-1,-1], [1,1])
# sns.kdeplot(data=datafrafa, x='Y_valid', y='Y_pred', fill=True, thresh=0.02, levels=5)
# plt.xlim(-1,1)
# plt.ylim(-1,1)

#%%
from tools.utils import plot_radial_profiles #, draw_PSF_stack
from data_processing.SPHERE_preproc_utils import SPHERE_preprocess
from PSF_models.TipToy_SPHERE_multisrc import TipTorch

norm_regime = 'sum'

def toy_run(model, fit_dict, data_dict):
        model.F  = torch.vstack([fit_dict['F (left)'],   fit_dict['F (right)' ]]).T
        model.bg = torch.vstack([data_dict['bg (left)'], data_dict['bg (right)']]).T
        
        for attr in ['Jy','Jxy','Jx']: setattr(model, attr, fit_dict[attr])
        for attr in ['dx','dy', 'r0', 'dn']: setattr(model, attr, data_dict[attr])
        
        model.WFS_Nph = data_dict['Nph WFS']
        
        return model.forward()


def prepare_batch_configs(batches_in, batches_out):
    batches_dict = []
    for i in tqdm(range(len(batches_in))):
        sample_ids = batches_in[i].index.tolist()
        PSF_0, _, _, _, config_files = SPHERE_preprocess(sample_ids, 'different', norm_regime, device)
        PSF_0 = PSF_0[..., 1:, 1:].cpu()
        config_files['sensor_science']['FieldOfView'] = 255
        
        delay = lambda r: (0.0017+81e-6)*r
        config_files['RTC']['LoopDelaySteps_HO'] = delay(config_files['RTC']['SensorFrameRate_HO'])
        config_files['sensor_HO']['NumberPhotons'] *= config_files['RTC']['SensorFrameRate_HO']
        
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
ids_same_train = []
# print('Train batches:')
for i, batch in enumerate(batches_dict_train):
    wvl = (batches_dict_train[i]['configs']['sources_science']['Wavelength'][0,...].cpu().numpy() * 1e9).tolist()
    if wvl == [1625.0, 1625.0]: ids_same_train.append(i)
    # print( f'Batch {i}: {wvl} [nm]' )

ids_same_valid = []
# print('Valid batches:')
for i, batch in enumerate(batches_dict_valid):
    wvl = (batches_dict_valid[i]['configs']['sources_science']['Wavelength'][0,...].cpu().numpy() * 1e9).tolist()
    if wvl == [1625.0, 1625.0]: ids_same_valid.append(i)
    # print( f'Batch {i}: {wvl} [nm]' )

#%%
# Load weights
net.load_state_dict(torch.load(WEIGHTS_FOLDER/'gnosis2_TT_weights.dict'))

#%%
class Agnosis(nn.Module):
    def __init__(self, size_in, size_mid, size_out):
        super(Agnosis, self).__init__()
        self.fc1 = nn.Linear(size_in,  size_mid)
        self.fc2 = nn.Linear(size_mid, size_mid)
        self.fc3 = nn.Linear(size_mid, size_out)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

torch.cuda.empty_cache()

net2 = Agnosis(X_train.shape[1], X_train.shape[1], Y_train_1.shape[1])
net2.to(device)
net2.double()
# optimizer = optim.Adam(net2.parameters(), lr=0.0001)
optimizer = optim.LBFGS(net2.parameters(), lr=1, history_size=20, max_iter=4, line_search_fn="strong_wolfe")

loss_PSF = nn.L1Loss(reduction='sum')
# loss_PSF = nn.MSELoss(reduction='sum')

def loss_fn(output, target):
    return loss_PSF(output, target) #+ torch.amax(torch.abs(output-target), dim=(-2,-1)).sum() * 1e1
    # return torch.amax(torch.abs(output-target), dim=(-2,-1)).mean() * 1e2
#%%
epochs = 26
torch.cuda.empty_cache()

toy = TipTorch(batches_dict_train[0]['configs'], None, device, TipTop=False, PSFAO=True)
toy.optimizables = []

# optimizer = optim.Adam(net.parameters(), lr=0.000005)
# loss_fn = nn.L1Loss(reduction='sum')

for epoch in range(epochs):
    loss_train_average = []
    # for batch in batches_dict_train:
    np.random.shuffle(ids_same_train)
    for i in ids_same_train:
        batch = batches_dict_train[i]
        
        toy.config = batch['configs']
        toy.Update(reinit_grids=True, reinit_pupils=False)
        
        X = batch['X'].to(device)
        Y = batch['Y'].to(device)
        PSF_0 = batch['PSF (data)'].to(device)
        
        for _ in range(5):
            optimizer.zero_grad()
            loss = loss_fn(toy_run(toy, gnosis2PAO_1(net2(X)), gnosis2PAO(Y)), PSF_0)
            loss.backward()
            # optimizer.step()
            optimizer.step( lambda: loss_fn( toy_run(toy, gnosis2PAO_1(net2(X)), gnosis2PAO(Y)), PSF_0 ))
            
            loss_train_average.append(loss.item()/PSF_0.shape[0])
            print('Current loss:', loss.item()/PSF_0.shape[0], end='\r')
 
    loss_valid_average = []
    with torch.no_grad():
        # for batch in batches_dict_valid:
        for i in ids_same_valid:
            batch = batches_dict_valid[i]
            
            toy.config = batch['configs']
            toy.Update(reinit_grids=True, reinit_pupils=False)

            X = batch['X'].to(device)
            Y = batch['Y'].to(device)
            PSF_0 = batch['PSF (data)'].to(device)
            Y_pred_1 = net2(X)
            PSF_pred = toy_run(toy, gnosis2PAO_1(Y_pred_1), gnosis2PAO(Y))
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
PSFs_1 = []
PSFs_0 = []
ids_all = []

with torch.no_grad():
    for i in ids_same_train:
        batch = batches_dict_train[i]
        
        toy.config = batch['configs']
        toy.Update(reinit_grids=True, reinit_pupils=False)
        
        ids = batch['ids']
        ids_all += ids
        X = batch['X'].to(device)
        Y = batch['Y'].to(device)
        PSF_0 = batch['PSF (data)'].to(device)
        PSFs_0.append(PSF_0.cpu())
        toy_run(toy, gnosis2PAO_1(Y), gnosis2PAO(Y))
        Y_pred_1 = net2(X)
        PSFs_1.append( toy_run(toy, gnosis2PAO_1(Y_pred_1), gnosis2PAO(Y)).cpu() )
    
    
    for i in ids_same_valid:
        batch = batches_dict_valid[i]
        
        toy.config = batch['configs']
        toy.Update(reinit_grids=True, reinit_pupils=False)
        
        ids = batch['ids']
        ids_all += ids
        X = batch['X'].to(device)
        Y = batch['Y'].to(device)
        PSF_0 = batch['PSF (data)'].to(device)
        PSFs_0.append(PSF_0.cpu()) 
        toy_run(toy, gnosis2PAO_1(Y), gnosis2PAO(Y))
        Y_pred_1 = net2(X)
        PSFs_1.append( toy_run(toy, gnosis2PAO_1(Y_pred_1), gnosis2PAO(Y)).cpu() )
        
        
PSFs_1 = torch.vstack(PSFs_1)
PSFs_0 = torch.vstack(PSFs_0)

wvl_title_1 = str( np.round( batch['configs']['sources_science']['Wavelength'][0].cpu().numpy().tolist()[0]*1e9 ).astype('uint') )
wvl_title_2 = str( np.round( batch['configs']['sources_science']['Wavelength'][1].cpu().numpy().tolist()[0]*1e9 ).astype('uint') )
titla = 'λ = '+str(wvl_title_1)+', '+str(wvl_title_2)+' nm'

destack = lambda PSF_stack: [ x for x in np.split(PSF_stack[:,0,...], PSF_stack.shape[0], axis=0) ]
plot_radial_profiles(destack(PSFs_0),  destack(PSFs_1),  'Data', 'Fitted', title=titla, dpi=200, cutoff=32, scale='log')
plt.show()



#%%
dirca = 'E:/ESO/Data/SPHERE/predicts_TipTorch/'


for i in range(len(ids_all)):
    
    PSF_0_ = PSFs_0[i,0,...].numpy()
    PSF_1_ = PSFs_1[i,0,...].numpy()
    
    plot_radial_profiles(PSF_0_,  PSF_1_, 'Data', 'Predicted', title=titla, dpi=200, cutoff=32, scale='log')
    filename = dirca + str(ids_all[i]) + '.png'
    plt.savefig(filename, dpi=200)



#%%
ids_to_select = [
    263,  382,  398,  440,  464,   485, 575,  634,  934,  1042, 1043, 1092,
    1101, 1115, 1133, 1426, 1445, 1446, 1485, 1490, 1499, 1504, 1515]

diffs = []
for i in range(PSFs_0.shape[0]):
    id = ids_all[i]
    if id not in ids_to_select: 
        max_0 = PSFs_0[i,0,...].max()
        max_1 = PSFs_1[i,0,...].max()  
        diffs.append( (max_0-max_1)/max_0 * 100 )
    
diffs = torch.stack(diffs)

diffs_mean = diffs.abs().median()
diffs_std = diffs.abs().std()

print('Mean difference: %.4f' % diffs_mean)
print('Std. difference: %.4f' % diffs_std)




#%% ===================== Batch overfit =====================

class Agnosis(nn.Module):
    def __init__(self, size_in, size_mid, size_out):
        super(Agnosis, self).__init__()
        self.fc1 = nn.Linear(size_in,  size_mid)
        self.fc2 = nn.Linear(size_mid, size_mid)
        self.fc3 = nn.Linear(size_mid, size_out)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

torch.cuda.empty_cache()

batch = batches_dict_valid[5]
Y = batch['Y'].double().to(device)

# Create a limited subset of data for Y
PAO2gnosis_1_torch = lambda out: ( torch.stack([transforms_output[a].forward(out[a]) for a in fitted_subset])).T
Y_1 = PAO2gnosis_1_torch( gnosis2PAO(Y) )

net2 = Agnosis(X.shape[1], X.shape[1], Y_1.shape[1])
net2.to(device)
net2.double()

toy = TipTorch(batch['configs'], None, device)
toy.optimizables = []

PSF_0_ = batch['PSF (data)'].to(device)

vec = torch.rand(Y_1.shape[1]).to(device)
vec.requires_grad = True

optimizer = optim.LBFGS(net2.parameters(), lr=10, history_size=20, max_iter=4, line_search_fn="strong_wolfe")
loss_fn = nn.L1Loss(reduction='sum')


for _ in range(27*2):
    optimizer.zero_grad()
    PSF_1 = toy_run(toy, gnosis2PAO_1(net2(X)), gnosis2PAO(Y))
    loss = loss_fn(PSF_1, PSF_0_)
    loss.backward()
    optimizer.step( lambda: loss_fn( PSF_0_, toy_run(toy, gnosis2PAO_1(net2(X)), gnosis2PAO(Y)) ))
    print('Current loss:', loss.item()/PSF_0_.shape[0], end='\r')


#%%
PSF_1 = toy_run(toy, gnosis2PAO_1(net2(X)), gnosis2PAO(Y))

wvl_title_1 = str( np.round( batch['configs']['sources_science']['Wavelength'][0].cpu().numpy().tolist()[0]*1e9 ).astype('uint') )
wvl_title_2 = str( np.round( batch['configs']['sources_science']['Wavelength'][1].cpu().numpy().tolist()[0]*1e9 ).astype('uint') )
titla = 'λ = '+str(wvl_title_1)+', '+str(wvl_title_2)+' nm'

destack = lambda PSF_stack: [ x for x in np.split(PSF_stack[:,0,...], PSF_stack.shape[0], axis=0) ]
plot_radial_profiles(destack(PSF_0),  destack(PSF_1),  'Data', 'Fitted', title=titla, dpi=200, cutoff=32, scale='log')
plt.show()

#%%
a = net2(X)
b = Y_1
c = torch.abs(a/b)
c.mean()

#%%
with torch.no_grad():
    Y_pred = gnosis2PAO_1(net2(X.to(device)))
    Y_test = gnosis2PAO_1(Y_1.to(device))

    for key in Y_pred.keys():
        
        A = Y_pred[key].detach().cpu().numpy()
        B = Y_test[key].detach().cpu().numpy()
        
        m = np.median(np.abs(A-B)/A * 100)
        s = np.std( np.abs(A-B)/A * 100)
        
        print(key, end=': ')
        print('{:.2f} +- {:.2f}'.format(m, s))


#%%
epochs = 26
torch.cuda.empty_cache()

toy = TipTorch(batches_dict_train[0]['configs'], norm_regime, device, TipTop=False, PSFAO=True)
toy.optimizables = []

optimizer = optim.Adam(net.parameters(), lr=0.0005)
# loss_fn = nn.L1Loss(reduction='sum')

batch = batches_dict_train[1]['ids']

for i in range(100):
    optimizer.zero_grad()
    
    toy.config = batch['configs']
    toy.Update(reinit_grids=True, reinit_pupils=False)
    
    X = batch['X'].to(device)
    Y = batch['Y'].to(device)
    PSF_0 = batch['PSF (data)'].to(device)
    Y_pred_1 = net(X)
    PSF_pred = toy_run(toy, gnosis2PAO_1(Y_pred_1), gnosis2PAO(Y))
    loss = loss_fn(PSF_pred, PSF_0.to(device))
    loss.backward()
    optimizer.step()
    
    loss_train_average.append(loss.item()/PSF_0.shape[0])
    print('Current loss:', loss.item()/PSF_0.shape[0], end='\r')



#%%

dirca = 'C:/Users/akuznets/Projects/TipToy/data/temp/sound of brand new world/'


for sample in tqdm(psf_df.index.tolist()):
    filename = dirca + str(sample) + '.png'
    
    # check if file exists
    if not os.path.isfile(filename):
        try:
        
            PSF_0, _, norms, _, init_config = SPHERE_preprocess([sample], '1P21I', norm_regime, device)
            PSF_0 = PSF_0[...,1:,1:].to(device)
            init_config['sensor_science']['FieldOfView'] = 255
            
            delay = lambda r: (0.0017+81e-6)*r
            init_config['RTC']['LoopDelaySteps_HO'] = delay(init_config['RTC']['SensorFrameRate_HO'])
            init_config['sensor_HO']['NumberPhotons'] *= init_config['RTC']['SensorFrameRate_HO']

            norms = norms[:, None, None].cpu().numpy()

            file = 'E:/ESO/Data/SPHERE/IRDIS_fitted_1P21I/' + str(sample) + '.pickle'

            with open(file, 'rb') as handle:
                data = pickle.load(handle)

            # toy = TipTorch(init_config, norm_regime, device, TipTop=True, PSFAO=False)
            toy = TipTorch(init_config, None, device, TipTop=True, PSFAO=False)
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

            destack = lambda PSF_stack: [ x for x in np.split(PSF_stack[:,0,...], PSF_stack.shape[0], axis=0) ]
            plot_radial_profiles(destack(PSF_0),  destack(PSF_1),  'Data', 'Fit', title='Fitted TipTop', dpi=200, cutoff=32, scale='log')
            plt.savefig(dirca + str(sample) + '.png', dpi=200)
        except:
            print('Error with sample', sample)
            pass

        

#%%
def UnitTest_PredictedAccuracy():
    # toy = TipTorch(batches_dict_valid[0]['configs'], norm_regime, device, TipTop=False, PSFAO=True)
    toy1 = TipTorch(batches_dict_valid[0]['configs'], None, device, TipTop=True, PSFAO=False)
    toy1.optimizables = []
    
    for i, batch in tqdm(enumerate(batches_dict_valid)):
        torch.cuda.empty_cache()
        
        # Test in batched fashion
        batch = batches_dict_valid[i]

        # Read data
        sample_ids = batch['ids']
        X, Y  = batch['X'].to(device), batch['Y'].to(device)
        PSF_0 = batch['PSF (data)']
        config_file = batch['configs']

        Y_1 = net(X)

        toy1.config = config_file
        toy1.Update(reinit_grids=True, reinit_pupils=False)

        PSF_1 = toy_run(toy1, gnosis2PAO_1(Y_1), gnosis2PAO(Y))

        wvl_title_1 = str( np.round( batches_dict_valid[i]['configs']['sources_science']['Wavelength'][0].cpu().numpy().tolist()[0]*1e9 ).astype('uint') )
        wvl_title_2 = str( np.round( batches_dict_valid[i]['configs']['sources_science']['Wavelength'][1].cpu().numpy().tolist()[0]*1e9 ).astype('uint') )
        titla = 'λ = '+str(wvl_title_1)+', '+str(wvl_title_2)+' nm'

        destack = lambda PSF_stack: [ x for x in np.split(PSF_stack[:,0,...], PSF_stack.shape[0], axis=0) ]
        plot_radial_profiles(destack(PSF_0),  destack(PSF_1),  'Data', 'Predicted', title=titla, dpi=200, cutoff=32, scale='log')
        plt.show()    


def UnitTest_FittedAccuracy():
    for i, batch in tqdm(enumerate(batches_dict_valid)):
        # toy = TipTorch(batches_dict_valid[0]['configs'], norm_regime, device, TipTop=False, PSFAO=True)
        toy1 = TipTorch(batches_dict_valid[0]['configs'], None, device, TipTop=True, PSFAO=False)
        toy1.optimizables = []

        torch.cuda.empty_cache()
        # Test in batched fashion
        batch = batches_dict_valid[i]
        
        # Read data
        sample_ids = batch['ids']
        X, Y  = batch['X'].to(device), batch['Y'].to(device)
        PSF_0 = batch['PSF (data)']
        config_file = batch['configs']
                
        toy1.config = config_file
        toy1.Update(reinit_grids=True, reinit_pupils=False)

        PSF_1 = toy_run(toy1, gnosis2PAO(Y), gnosis2PAO(Y))

        wvl_title_1 = str( np.round( batches_dict_valid[i]['configs']['sources_science']['Wavelength'][0].cpu().numpy().tolist()[0]*1e9 ).astype('uint') )
        wvl_title_2 = str( np.round( batches_dict_valid[i]['configs']['sources_science']['Wavelength'][1].cpu().numpy().tolist()[0]*1e9 ).astype('uint') )
        titla = 'λ = '+str(wvl_title_1)+', '+str(wvl_title_2)+' nm'
        
        destack = lambda PSF_stack: [ x for x in np.split(PSF_stack[:,0,...], PSF_stack.shape[0], axis=0) ]
        plot_radial_profiles(destack(PSF_0),  destack(PSF_1),  'Data', 'Fitted', title=titla, dpi=200, cutoff=32, scale='log')
        plt.show()    


def UnitTest_DirectPredictions():
    for i, batch in tqdm(enumerate(batches_dict_valid)):

        sample_ids = batch['ids']

        # Now test individual samples
        PSFs_test = []
        for sample in tqdm(sample_ids):
            PSF_0, bg, norms, _, init_config = SPHERE_preprocess([sample], '1P21I', norm_regime, device)
            PSF_0 = PSF_0[...,1:,1:].to(device)
            init_config['sensor_science']['FieldOfView'] = 255
            
            delay = lambda r: (0.0017+81e-6)*r
            init_config['RTC']['LoopDelaySteps_HO'] = delay(init_config['RTC']['SensorFrameRate_HO'])
            init_config['sensor_HO']['NumberPhotons'] *= init_config['RTC']['SensorFrameRate_HO']

            Jx = init_config['sensor_HO']['Jitter X'].abs()
            Jy = init_config['sensor_HO']['Jitter Y'].abs()
            init_config['sensor_HO']['NumberPhotons'] *= init_config['RTC']['SensorFrameRate_HO']
            
            file = 'E:/ESO/Data/SPHERE/IRDIS_fitted_1P21I/' + str(sample) + '.pickle'

            with open(file, 'rb') as handle:
                data = pickle.load(handle)

            # toy = TipTorch(init_config, norm_regime, device, TipTop=True, PSFAO=False)
            toy = TipTorch(init_config, None, device, TipTop=True, PSFAO=False)
            toy.optimizables = []

            tensy = lambda x: torch.tensor(x).to(device)
            toy.F     = tensy( [[1.0, 1.0]] )
            toy.bg    = bg
            toy.Jy    = Jy
            toy.Jx    = Jx
            toy.Jxy   = tensy( [0.0] )
            toy.dn    = tensy( [0.0] ) 
            toy.dx    = tensy( data['dx']  ).flatten()
            toy.dy    = tensy( data['dy']  ).flatten()
            toy.r0    = tensy( data['r0'] ) #TODO: grab from psf_df
            PSFs_test.append( toy().detach().clone() )

        PSFs_test = torch.stack(PSFs_test).squeeze()
        if PSFs_test.ndim == 3: PSFs_test = PSFs_test.unsqueeze(0)

        wvl_title_1 = str( np.round( batches_dict_valid[i]['configs']['sources_science']['Wavelength'][0].cpu().numpy().tolist()[0]*1e9 ).astype('uint') )
        wvl_title_2 = str( np.round( batches_dict_valid[i]['configs']['sources_science']['Wavelength'][1].cpu().numpy().tolist()[0]*1e9 ).astype('uint') )
        titla = 'λ = '+str(wvl_title_1)+', '+str(wvl_title_2)+' nm'

        destack = lambda PSF_stack: [ x for x in np.split(PSF_stack[:,0,...], PSF_stack.shape[0], axis=0) ]
        plot_radial_profiles(destack(PSF_0),  destack(PSFs_test),  'Data', 'Predicted (direct)', title=titla, dpi=200, cutoff=32, scale='log')
        plt.show()


def UnitTest_Batch_vs_Individuals():
    for i, batch in tqdm(enumerate(batches_dict_valid)):
        # toy = TipTorch(batches_dict_valid[0]['configs'], norm_regime, device, TipTop=False, PSFAO=True)
        toy1 = TipTorch(batches_dict_valid[0]['configs'], None, device, TipTop=True, PSFAO=False)
        toy1.optimizables = []

        torch.cuda.empty_cache()
        # Test in batched fashion
        batch = batches_dict_valid[i]
        
        # Read data
        sample_ids = batch['ids']
        X, Y  = batch['X'].to(device), batch['Y'].to(device)
        PSF_0 = batch['PSF (data)']
        config_file = batch['configs']
                
        toy1.config = config_file
        toy1.Update(reinit_grids=True, reinit_pupils=False)

        PSF_1 = toy_run(toy1, gnosis2PAO(Y), gnosis2PAO(Y))

        # Now test individual samples
        PSFs_test = []
        for sample in tqdm(sample_ids):
            PSF_0, _, norms, _, init_config = SPHERE_preprocess([sample], '1P21I', norm_regime, device)
            PSF_0 = PSF_0[...,1:,1:].to(device)
            init_config['sensor_science']['FieldOfView'] = 255
            
            delay = lambda r: (0.0017+81e-6)*r
            init_config['RTC']['LoopDelaySteps_HO'] = delay(init_config['RTC']['SensorFrameRate_HO'])
            init_config['sensor_HO']['NumberPhotons'] *= init_config['RTC']['SensorFrameRate_HO']

            norms = norms[:, None, None].cpu().numpy()

            file = 'E:/ESO/Data/SPHERE/IRDIS_fitted_1P21I/' + str(sample) + '.pickle'

            with open(file, 'rb') as handle:
                data = pickle.load(handle)

            # toy = TipTorch(init_config, norm_regime, device, TipTop=True, PSFAO=False)
            toy = TipTorch(init_config, None, device, TipTop=True, PSFAO=False)
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
        'dn',
        'dx',
        'dy',
        'r0',
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

    batch = batches_dict_valid[5]
    Y = batch['Y'].double().to(device)
    
    # Create a limited subset of data for Y
    PAO2gnosis_1_torch = lambda out: ( torch.stack([transforms_output[a].forward(out[a]) for a in fitted_subset])).T
    Y_1 = PAO2gnosis_1_torch( gnosis2PAO_1(Y) )

    # toy = TipTorch(batch['configs'], norm_regime, device)
    toy = TipTorch(batch['configs'], None, device)
    toy.optimizables = []

    PSF_0_ = toy_run(toy, gnosis2PAO_1(Y_1), gnosis2PAO(Y))

    vec = torch.rand(Y_1.shape[1]).to(device)
    vec.requires_grad = True

    optimizer = optim.LBFGS([vec], lr=10, history_size=20, max_iter=4, line_search_fn="strong_wolfe")
    loss_fn = nn.L1Loss(reduction='sum')

    for _ in range(27):
        optimizer.zero_grad()
        PSF_1 = toy_run(toy, gnosis2PAO_1(vec * Y_1), gnosis2PAO(Y))
        loss = loss_fn(PSF_1, PSF_0_)
        loss.backward()
        optimizer.step( lambda: loss_fn( PSF_0_, toy_run(toy, gnosis2PAO_1(vec * Y_1), gnosis2PAO(Y)) ))
        print('Current loss:', loss.item(), end='\r')


def UnitTest_TipTorch_and_NN():
    class Agnosis(nn.Module):
        def __init__(self, size):
            super(Agnosis, self).__init__()
            self.fc1 = nn.Linear(size, size)
            self.fc2 = nn.Linear(size, size)

        def forward(self, x):
            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            return x

    torch.cuda.empty_cache()
    
    batch = batches_dict_valid[5]
    # sample_ids = batch['ids']
    X = batch['X'].double().to(device)
    Y = batch['Y'].double().to(device)

    # PAO2gnosis_1_torch = lambda out: ( torch.stack([transforms_output[a].forward(out[a]) for a in fitted_subset])).T
    # Y_1 = PAO2gnosis_1_torch( gnosis2PAO_1(Y) )

    net = Agnosis(Y.shape[1])
    net.to(device)
    net.double()

    # toy = TipTorch(batch['configs'], norm_regime, device)
    toy = TipTorch(batch['configs'], None, device)
    toy.optimizables = []

    PSF_0_ = toy_run(toy, gnosis2PAO(Y), gnosis2PAO(Y))

    optimizer = optim.Adam(net.parameters(), lr=0.0005)
    loss_fn = nn.L1Loss(reduction='sum')

    for _ in range(100):
        optimizer.zero_grad()
        PSF_1 = toy_run(toy, gnosis2PAO(net(Y)), gnosis2PAO(Y))
        loss = loss_fn(PSF_1, PSF_0_)
        loss.backward()
        optimizer.step( lambda: loss_fn( PSF_0_, toy_run(toy, gnosis2PAO(Y), gnosis2PAO(Y)) ))
        print('Current loss:', loss.item(), end='\r')


# UnitTest_FittedAccuracy()
# UnitTest_Batch_vs_Individuals()
# UnitTest_TransformsBackprop()
# UnitTest_TipTorch_transform_out()
# UnitTest_TipTorch_and_NN()

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
