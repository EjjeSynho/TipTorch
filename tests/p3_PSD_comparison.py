#%%
%reload_ext autoreload
%autoreload 2

from logging import config
import sys
import os
import tempfile

sys.path.append('..')

from project_settings import DATA_FOLDER, PROJECT_PATH, DATA_FOLDER

# import TIPTOP dependencies
for module in ['MASTSEL', 'P3', 'SEEING', 'SYMAO', 'TIPTOP']:
    sys.path.append(str(TIPTOP_PATH:=(PROJECT_PATH / '../astro-tiptop') / f'{module}'))
    sys.path.append(str(TIPTOP_PATH:=(PROJECT_PATH / '../astro-tiptop') / f'{module}/{module.lower()}'))

sys.path.append(str(TIPTOP_PATH:=(PROJECT_PATH / '../astro-tiptop')))
# from P3.p3.aoSystem.fourierModel import fourierModel
from aoSystem.fourierModel import fourierModel

# Verify paths
# for i, path in enumerate(sys.path):
    # print(f"{i}: {path}")

#%%
path_ini = str(DATA_FOLDER / "parameter_files/muse_ltao.ini")

# Create a temporary modified version of the ini file
with open(path_ini, 'r') as f:
    ini_content = f.read()

# Replace the calibration path to the one understandable by P3
modified_content = ini_content.replace('$PROJECT_PATH$/data/calibrations/', '/aoSystem/data/')
temp_dir = os.path.dirname(path_ini)
temp_fd, temp_path_ini = tempfile.mkstemp(suffix='.ini', dir=temp_dir)

with os.fdopen(temp_fd, 'w') as temp_file:
    temp_file.write(modified_content)

# Run P3 Fourier model
tiptop_HO_model = fourierModel(
    temp_path_ini,
    path_root=None,
    calcPSF=False,
    verbose=False,
    display=False,
    getErrorBreakDown=False,
    getFWHM=False,
    getEncircledEnergy=False,
    getEnsquaredEnergy=False,
    displayContour=False
)

# Remove temp config
os.remove(temp_path_ini)

#%%
# Compute different PSD contributors
p3_fitting  = tiptop_HO_model.fittingPSD().squeeze()
p3_aliasing = tiptop_HO_model.aliasingPSD().squeeze()
p3_diff_ref = tiptop_HO_model.differentialRefractionPSD().squeeze()
p3_chromatism = tiptop_HO_model.chromatismPSD().squeeze()
p3_spatioTemporal = tiptop_HO_model.spatioTemporalPSD().squeeze()
p3_noise = tiptop_HO_model.noisePSD().squeeze()

#%%
%reload_ext autoreload
%autoreload 2

import torch
from PSF_models.TipTorch import TipTorch
from tools.utils import PupilVLT
from managers.config_manager import ConfigManager, SingleTargetInSingleObservation
from managers.parameter_parser import ParameterParser
from project_settings import device, DATA_FOLDER
import numpy as np

#%%
path_ini = str(DATA_FOLDER / "parameter_files/muse_ltao.ini")

config_manager = ConfigManager()
config_dict = ParameterParser(path_ini).params

config_dict['NumberSources'] = 1  # Single on-axis source for PSD comparison

config_dict['sources_HO']['Height'] = np.array([config_dict['sources_HO']['Height']])  # Single source height
config_dict['telescope']['ZenithAngle'] = np.array([config_dict['telescope']['ZenithAngle']])  # Ensure correct type

config_dict['RTC']['SensorFrameRate_HO'] =  np.array([config_dict['RTC']['SensorFrameRate_HO']])
config_dict['RTC']['LoopDelaySteps_HO'] =  np.array([config_dict['RTC']['LoopDelaySteps_HO']])
config_dict['RTC']['LoopGain_HO'] =  np.array([config_dict['RTC']['LoopGain_HO']])

config_dict['sensor_HO']['SizeLenslets'] = np.array([config_dict['sensor_HO']['SizeLenslets'][0]])
config_dict['sensor_HO']['NumberLenslets'] = np.array([config_dict['sensor_HO']['NumberLenslets'][0]])

config_dict['sensor_HO']['ClockRate'] = np.array([config_dict['sensor_HO']['ClockRate'][0]])
config_dict['atmosphere']['L0'] = np.array([config_dict['atmosphere']['L0']])

config_dict = SingleTargetInSingleObservation(config_dict, device=device)

#%%

# TODO:
# - fix default pupil loading errors
# - compute noise variance for every WFS separately and then average it
# - make function to preserve only necessary entries in configs

#%%

# pupil = torch.tensor(PupilVLT(samples=320, rotation_angle=0)).to(device)

# Define which PSDs to include
PSD_include = {
    'fitting':         True,
    'WFS noise':       True,
    'spatio-temporal': True,
    'aliasing':        True,
    'chromatism':      True,
    'diff. refract':   True,
    'Moffat':          False
}

# Initialize TipTorch model
tiptorch_model = TipTorch(
    AO_config=config_dict,
    AO_type='LTAO',
    PSD_include=PSD_include,
    norm_regime='sum',
    device=device,
    oversampling=1
)

#%%
PSF_1 = tiptorch_model()

#%%
tiptorch_model.PSDs['WFS noise'].shape
