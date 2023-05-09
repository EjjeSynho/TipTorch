#%%
import os
import pickle


folda = 'E:/ESO/Data/SPHERE/IRDIS_fitted_1P21I/'

files = os.listdir(folda)

# Open pickle file
with open(folda + files[0], 'rb') as f:
   data = pickle.load(f)    


# %%
from tools.config_manager import ConfigManager, GetSPHEREonsky


config_manager = ConfigManager(GetSPHEREonsky())
merged_config  = config_manager.Merge([config_manager.Modify(config_file, sample) for sample in data_samples])
config_manager.Convert(merged_config, framework='numpy')