#%%
import os
import pickle
from tools.config_manager import ConfigManager, GetSPHEREonsky
from tqdm import tqdm
from globals import SPHERE_DATA_FOLDER
# from globals import SPHERE_FITTING_FOLDER

#%%
folda = SPHERE_DATA_FOLDER + 'IRDIS_fitted_NP2NI/'
foldb = folda[:-1] + '_2/'
# foldb = folda[:-3] + '_2/'
   

files = os.listdir(folda)
for i in range(len(files)):
   with open(folda + files[i], 'rb') as f:
      data = pickle.load(f)

   config = data['config']
   config_manager = ConfigManager(GetSPHEREonsky())
   config_manager.Convert(config, framework='numpy')
   config_manager.process_dictionary(config)
   data['config'] = config
   data['SR data'] = data['SR data'].detach().cpu().numpy()
   data['SR fit'] = data['SR fit'].detach().cpu().numpy()

   # save pickle file
   with open(foldb + files[i], 'wb') as f:
      pickle.dump(data, f)
   print('File ' + files[i] + ' converted')


# %%

pathss = 'F:/ESO/Data/MUSE/files.txt'
with open(pathss, 'r') as f:
   files = f.readlines()
files.pop(-1)
   
for i in tqdm(range(len(files))):
   files[i] = int(files[i].strip().split('.')[0])

files = set(files)

#%%
with open('F:/ESO/Data/MUSE/muse_df.pickle', 'rb') as f:
   psf_df = pickle.load(f)

psf_df = psf_df[psf_df['Corrupted']   == False]
psf_df = psf_df[psf_df['Bad quality'] == False]

ids = set(psf_df.index.values)

ids_diff = ids - files

#%%

ids_diff = [18, 49, 62, 68, 71, 75, 79, 85, 90, 94, 95, 99, 122, 126, 127,
            133, 144, 149, 151, 157, 202, 207, 225, 233, 260, 283, 288, 289,
            292, 313, 314, 315, 324, 338, 373, 388, 389, 393, 394, 397]