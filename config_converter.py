#%%
import os
import pickle
from tools.config_manager import ConfigManager, GetSPHEREonsky
from tqdm import tqdm
from globals import SPHERE_DATA_FOLDER
# from globals import SPHERE_FITTING_FOLDER

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

