import sys
sys.path.insert(0, '..')

import torch
import numpy as np
import cupy as cp
from copy import deepcopy
from functools import reduce
import operator
from pprint import pprint


class ConfigManager():
    def __init__(self, match_table=[], uniqualized=[]):
        self.match_table = match_table

        if len(uniqualized) == 0:
            self.uniqualized = [
                ['atmosphere', 'Wavelength'],
                ['DM', 'AoArea'],
                ['DM', 'DmHeights'],
                ['DM', 'DmPitchs'],
                ['DM', 'InfCoupling'],
                ['DM', 'InfModel'],
                ['DM', 'NumberActuators'],
                ['DM', 'OptimizationAzimuth'],
                ['DM', 'OptimizationConditioning'],
                ['DM', 'OptimizationWeight'],
                ['sensor_HO', 'Algorithm'],
                ['sensor_HO', 'Binning'],
                ['sensor_HO', 'Dispersion'],
                ['sensor_HO', 'ExcessNoiseFactor'],
                ['sensor_HO', 'FieldOfView'],
                ['sensor_HO', 'Modulation'],
                ['sensor_HO', 'NewValueThrPix'],
                ['sensor_HO', 'NoiseVariance'],
                ['sensor_HO', 'NumberLenslets'],
                ['sensor_HO', 'PixelScale'],
                ['sensor_HO', 'SigmaRON'],
                ['sensor_HO', 'SizeLenslets'],
                ['sensor_HO', 'ThresholdWCoG'],
                ['sensor_HO', 'WfsType'],
                ['sensor_HO', 'WindowRadiusWCoG'],
                ['sensor_science', 'Binning'],
                ['sensor_science', 'ExcessNoiseFactor'],
                ['sensor_science', 'FieldOfView'],
                ['sensor_science', 'Name'],
                ['sensor_science', 'PixelScale'],
                ['sensor_science', 'Saturation'],
                ['sensor_science', 'SigmaRON'],
                ['telescope', 'ObscurationRatio'],
                ['telescope', 'PathApodizer'],
                ['telescope', 'PathPupil'],
                ['telescope', 'PathStatModes'],
                ['telescope', 'PathStaticOff'],
                ['telescope', 'PathStaticOn'],
                ['telescope', 'PathStaticPos'],
                ['telescope', 'PupilAngle'],
                ['telescope', 'Resolution'],
                ['telescope', 'TelescopeDiameter']
            ]


    def set_value(self, root, items, value):
        """Set a value in a nested object in root by item sequence."""
        self.get_value(root, items[:-1])[items[-1]] = value


    def get_value(self, root, items):
        """Access a nested object in root by item sequence."""
        return reduce(operator.getitem, items, root)


    def is_valid(self, value):
        """Checks if value contains Nones"""
        not_None = lambda x: False if x is None else True
        if hasattr(value, '__len__'):
            return np.all( np.array([self.is_valid(x) for x in value]) )
        else: return not_None(value)

    # Collapse all singleton dimensions to ensure that later multiplification of dimensions will go fine
    def squeeze_values(self, config):
        zero_d = lambda x: x.item() if x.ndim == 0 else x.tolist()
        for entry in config:
            value = config[entry]
            if isinstance(value, dict):
                self.squeeze_values(value)
            elif isinstance(value, str):
                pass
            else:
                value = zero_d( np.squeeze(np.array(value)) )
            config[entry] = value


    def Modify(self, config, modifier):
        buf_config = deepcopy(config)
        buf_modifier = deepcopy(modifier)

        self.squeeze_values(buf_config)
        self.squeeze_values(buf_modifier)

        for config_entry, modifier_entry, modifier_func in self.match_table:
            if modifier_func is None: modifier_func = lambda x: x
            self.set_value(buf_config, config_entry, modifier_func(self.get_value(buf_modifier, modifier_entry)))
        return buf_config


    def Merge(self, configs):
        """Merges config files for multiple targets"""
        if not isinstance(configs, list):
            return configs
        new_config = deepcopy(configs[0]) # creating a buffer configuration

        def get_all_entries(d):
            entries = []
            def dict_path(path, my_dict):
                for k,v in my_dict.items():
                    if isinstance(v, dict):
                        dict_path(path+"|"+k, v)
                    else:
                        entries.append(path+"|"+str(k))
            dict_path('', d)
            for i,v in enumerate(entries): entries[i] = v.split('|')[1:]
            return entries

        all_entries = get_all_entries(new_config)
        for entry in all_entries:
            if entry in self.uniqualized:
                value_to_set = self.get_value(configs[0], entry)
            else:
                value_to_set = [self.get_value(config, entry) for config in configs]
            self.set_value(new_config, entry, value_to_set)

        new_config['NumberSources'] = len(configs)
        return new_config


    def Convert(self, config, framework='pytorch', device=None):
        if framework == 'pytorch':
            if device is None: device = torch.device('cpu')
            convert_value = lambda x: torch.tensor(x, device=device).float()
        elif framework == 'numpy':
            convert_value = lambda x: np.array(x).astype(np.float32)
        elif framework == 'cupy':
            convert_value = lambda x: cp.array(x, dtype=cp.float32)
        else:
            raise NotImplementedError('Unknown framework name \"'+framework+'\"!')

        zero_d = lambda x: x if type(x) == float else convert_value(x)
        for entry in config:
            value = config[entry]
            if isinstance(value, dict):
                self.Convert(value, framework, device)
            elif isinstance(value, str) or not self.is_valid(value):
                pass
            else:
                value = zero_d(value)
            config[entry] = value


def GetSPHEREonsky():
    func = lambda x: 90.0 - x
    match_table = [
        (['atmosphere','Seeing'],          ['seeing','SPARTA'],         None),
        (['atmosphere','WindSpeed'],       ['Wind speed','header'],     None),
        (['atmosphere','WindDirection'],   ['Wind direction','header'], None),
        (['sensor_science','Zenith'],      ['telescope','altitude'],    func),
        (['telescope','ZenithAngle'],      ['telescope','altitude'],    func),
        (['sensor_science','Azimuth'],     ['telescope','azimuth'],     None),
        (['sensor_science','SigmaRON'],    ['Detector','ron'],          None),
        (['sensor_science','Gain'],        ['Detector','gain'],         None),
        (['sensor_HO','NumberPhotons'],    ['WFS','Nph vis'],           None),
        (['RTC','SensorFrameRate_HO'],     ['WFS','rate'],              None),
        (['sensor_science','Gain'],        ['Detector', 'gain'],        None),
        (['sensor_science','PixelScale'],  ['Detector', 'psInMas'],     None),
        (['sensor_science','SigmaRON'],    ['Detector', 'ron'],         None),
        (['sources_science','Wavelength'], ['spectrum','lambda'],       None)
    ]
    return match_table