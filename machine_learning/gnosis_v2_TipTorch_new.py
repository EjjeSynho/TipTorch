#%%
%reload_ext autoreload
%autoreload 2

import sys
sys.path.append('..')

from project_globals import SPHERE_DATA_FOLDER, WEIGHTS_FOLDER, device
from data_processing.SPHERE_preproc_utils import SPHERE_preprocess
from data_processing.normalizers import Uniform, TransformSequence
from PSF_models.TipToy_SPHERE_multisrc import TipTorch
from tools.utils import seeing, plot_radial_profiles, plot_radial_profiles_new
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
#%%
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
if not os.path.isfile('F:/ESO/Data/SPHERE/fitted_df.pickle'):
    fitted_dict_raw = {key: [] for key in ['F', 'dx', 'dy', 'r0', 'n', 'dn', 'bg', 'Jx', 'Jy', 'Jxy', 'Nph WFS', 'SR data', 'SR fit']}
    ids = []

    images_data = []
    images_fitted = []

    fitted_folder = 'F:/ESO/Data/SPHERE/IRDIS_fitted_1P21I/'
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
    with open('F:/ESO/Data/SPHERE/fitted_df.pickle', 'wb') as handle:
        pickle.dump(fitted_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
else:
    with open('F:/ESO/Data/SPHERE/fitted_df.pickle', 'rb') as handle:
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
      
    # model.Jx *= 0.85
    # model.Jy *= 0.85
     
    model.WFS_Nph = conv(data['Nph WFS (data)'])
    inv_a2 = conv( 1 / (data['λ WFS (nm)']*1e9/2/np.pi)**2 )
    model.dn = inv_a2 * pred['Rec. noise']**2 - conv(data['n'])
    
    return model.forward()


def toy_run_direct(model, data):
    conv = lambda x: torch.from_numpy(np.array(x)).to(device).float()
    model.F  = torch.tensor([1.0, 1.0]).to(device).float()
    model.bg = torch.tensor([0., 0.]).to(device).float()

    model.Jx = torch.tensor([28.0]).to(device).float()
    model.Jy = torch.tensor([28.0]).to(device).float()

    for attr in ['dx', 'dy', 'r0 (data)', 'Jxy']:
        setattr(model, attr, conv(data[attr]))
        
    model.WFS_Nph = conv(data['Nph WFS (data)'])
    model.dn = torch.zeros(1).to(device)
    
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


def split_dataframe(df, chunk_size):
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks


batch_train = prepare_batch_configs(split_dataframe(df.iloc[:-64], 64))
batch_test  = prepare_batch_configs([df.iloc[-64:]])

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

# net = torch.load(WEIGHTS_FOLDER/f'param_predictor_{len(selected_X)}x{len(selected_Y)}.pth')
# net = torch.load(WEIGHTS_FOLDER/f'param_predictor_{len(selected_X)}x{len(selected_Y)}_tuned.pth')
net = torch.load(WEIGHTS_FOLDER/f'param_predictor_{len(selected_X)}x{len(selected_Y)}_real.pth')
# net = ParamPredictor(len(selected_X), len(selected_Y), 100, 0.05)
# net = net.to(device)

#%%
# batch_test[0] = batch_train[2]

toy = TipTorch(batch_test[0]['configs'], 'sum', device, TipTop=True, PSFAO=False)
toy.optimizables = []

with torch.no_grad():
    # y_pred = torch.from_numpy(mlp.predict(batch_test[0]['X'])).float()
    y_pred = net(batch_test[0]['X'].float().to(device))
    # y_pred = batch_test[0]['Y']

PSF_0 = batch_test[0]['PSF (data)'].cpu().numpy()
PSF_2 = toy_run( toy, batch_test[0]['df'], NN2fit(y_pred.to(device)) ).cpu().numpy()

#%%
PSF_2[36,...] = PSF_0[36,...]

destack = lambda PSF_stack: [ x for x in np.split(PSF_stack[:,0,...], PSF_stack.shape[0], axis=0) ]

plot_radial_profiles_new(
    PSF_0[:,0,...],
    PSF_2[:,0,...], 
    'Data',
    'Predicted',
    title  = 'Prediction accuracy',
    scale  = 'log',
    center = (255/2, 255/2)
)
# plot_radial_profiles(destack(PSF_0), destack(PSF_2),  'Data', 'Predicted', title='Prediction accuracy', scale='log')

plt.savefig('C:/Users/akuznets/Desktop/AO4ELT/trained_on_fitted_log.pdf')

#%%
# batch_test[0] = batch_train[2]

toy = TipTorch(batch_test[0]['configs'], 'sum', device, TipTop=True, PSFAO=False)
toy.optimizables = []

with torch.no_grad():
    # y_pred = torch.from_numpy(mlp.predict(batch_test[0]['X'])).float()
    y_pred = net(batch_test[0]['X'].float().to(device))
    # y_pred = batch_test[0]['Y']

PSF_0 = batch_test[0]['PSF (data)'].cpu().numpy()
PSF_1 = toy_run_direct( toy, batch_test[0]['df'] ).cpu().numpy()

destack = lambda PSF_stack: [ x for x in np.split(PSF_stack[:,0,...], PSF_stack.shape[0], axis=0) ]

plot_radial_profiles_new(PSF_0[:,0,...], PSF_1[:,0,...],  'Data', 'Predicted', title='Prediction accuracy', scale='lin')
# plt.savefig('C:/Users/akuznets/Desktop/AO4ELT/direct_lin.pdf')


#%%
epochs = 26
torch.cuda.empty_cache()

toy = TipTorch(batch_train[0]['configs'], None, device, TipTop=False, PSFAO=True)
toy.optimizables = []

optimizer = optim.Adam(net.parameters(), lr=0.00001)
loss_fn = nn.L1Loss(reduction='sum')

for epoch in range(epochs):
    loss_train_average = []
    # for batch in batches_dict_train:
    for batch in batch_train:       
        toy.config = batch['configs']
        toy.Update(reinit_grids=True, reinit_pupils=False)
        
        X = torch.nan_to_num(batch['X'].float().to(device))
        Y = torch.nan_to_num(batch['Y'].float().to(device))
        PSF_0 = batch['PSF (data)'].to(device)
        
        for _ in range(5):
            optimizer.zero_grad()
            
            y_pred = net(X)
            PSF_1 = toy_run( toy, batch['df'], NN2fit(y_pred.to(device)) )
            loss = loss_fn(PSF_0, PSF_1)
            loss.backward()
            optimizer.step()
            
            loss_train_average.append(loss.item() / PSF_0.shape[0])
            print('Current loss:', loss.item() / PSF_0.shape[0], end='\r')
 
    loss_valid_average = []
    with torch.no_grad():
        for batch in batch_test:
            toy.config = batch['configs']
            toy.Update(reinit_grids=True, reinit_pupils=False)

            X = torch.nan_to_num(batch['X'].float().to(device))
            Y = torch.nan_to_num(batch['Y'].float().to(device))
            PSF_0 = batch['PSF (data)'].to(device)
            
            y_pred = net(X)
            PSF_1 = toy_run( toy, batch['df'], NN2fit(y_pred.to(device)) )
            loss = loss_fn(PSF_1, PSF_0)
            loss_valid_average.append(loss.item() / PSF_0.shape[0])
            print('Current loss:', loss.item() / PSF_0.shape[0], end='\r')
            
    print('Epoch %d/%d: ' % (epoch+1, epochs))
    print('  Train loss:  %.4f' % (np.array(loss_train_average).mean()))
    print('  Valid. loss: %.4f' % (np.array(loss_valid_average).mean()))
    print('')
    
    torch.cuda.empty_cache()
    
# torch.save(net.state_dict(), WEIGHTS_FOLDER/'gnosis2_weights_tuned.dict')
# torch.save(net, WEIGHTS_FOLDER/f'param_predictor_{len(selected_X)}x{len(selected_Y)}_tuned.pth')

#%% ===========================================================================
#==============================================================================
#==============================================================================
#==============================================================================


