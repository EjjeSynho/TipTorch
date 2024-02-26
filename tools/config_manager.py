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
    def __init__(self, uniqualized_values=None):
        #'uniqualized' paremeters are those are defined only one in a config file for all targets
        if uniqualized_values is None:
            self.uniqualized_values = [
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


    def process_value(self, x):
        if x is None or isinstance(x, str) or isinstance(x, int):
            return x
        
        elif isinstance(x, (list, tuple)):
            if hasattr(x, '__len__') and len(x) == 1:
                return self.process_value(x[0])
            else:
                return x
            
        elif isinstance(x, (torch.Tensor, np.ndarray)):
            if x.ndim == 0:
                return float(x.item())
            else:
                x = x.squeeze()
                if x.ndim == 0:
                    return float(x.item())
                else:
                    return x
        else:
            return float(x)


    def process_dictionary(self, d):
        """Collapse all singleton dimensions of stored values to ensure that later addition of dimensions by TipTorch will go fine"""
        for key, value in d.items():
            if isinstance(value, dict):
                d[key] = self.process_dictionary(value)
            else:
                d[key] = self.process_value(value)
        return d


    def is_valid(self, value):
        """Checks if value contains Nones"""
        not_None = lambda x: False if x is None else True
        value = self.process_value(value)
        if hasattr(value, '__len__'):
            return np.all( np.array([self.is_valid(x) for x in value]) )
        else:
            return not_None(value)


    def Modify(self, config, modifier, conversion_table, processor_func=None):
        """
        Modifies a config file with an extrenal dictionary of values.
        Requires the mathing table to math values from extrenal source to config.
        Optionally, requires a processor function to tranform the values inside the config.
        """	
        self.process_dictionary(buf_config   := deepcopy(config))
        self.process_dictionary(buf_modifier := deepcopy(modifier))

        for config_entry, modifier_entry in conversion_table:
            self.set_value(buf_config, config_entry, self.get_value(buf_modifier, modifier_entry))
        
        if processor_func is not None:
            buf_config = processor_func(buf_config, buf_modifier)
        
        return buf_config


    def Merge(self, configs):
        """Merges config files for multiple targets into one config dictionary"""
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
            if entry in self.uniqualized_values:
                value_to_set = self.get_value(configs[0], entry)
            else:
                value_to_set = [self.get_value(config, entry) for config in configs]
            self.set_value(new_config, entry, value_to_set)

        new_config['NumberSources'] = int(len(configs))
        return new_config


    def Convert(self, config, framework='pytorch', device=None):
        """Converts all values in a config file to a specified framework"""
        if framework.lower() == 'pytorch':
            if device is None: device = torch.device('cpu')
            convert_value = lambda x: torch.tensor(x, device=device).float()
        elif framework.lower() == 'numpy':
            convert_value = lambda x: np.array(x.cpu()) if isinstance(x, torch.Tensor) else np.array(x)
        elif framework.lower() == 'cupy':
            convert_value = lambda x: cp.array(x.cpu()) if isinstance(x, torch.Tensor) else cp.array(x)
        else:
            raise NotImplementedError('Unsupported framework \"'+framework+'\"!')

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
    conversion_table = [
        (['atmosphere','Seeing'],          ['seeing','SPARTA']        ),
        (['atmosphere','WindSpeed'],       ['Wind speed','header']    ),
        (['atmosphere','WindDirection'],   ['Wind direction','header']),
        (['sensor_science','Zenith'],      ['telescope','altitude']   ),
        (['telescope','ZenithAngle'],      ['telescope','altitude']   ), #TODO: difference between zenith and zenithAngle?
        (['sensor_science','Azimuth'],     ['telescope','azimuth']    ),
        (['sensor_science','SigmaRON'],    ['Detector','ron']         ),
        (['sensor_science','Gain'],        ['Detector','gain']        ),
        (['sources_HO', 'Wavelength'],     ['WFS', 'wavelength']      ),
        (['sensor_HO','NumberPhotons'],    ['WFS','Nph vis']          ),
        (['sensor_HO','Jitter X'],         ['WFS','TT jitter X']      ),
        (['sensor_HO','Jitter Y'],         ['WFS','TT jitter Y']      ),
        (['RTC','SensorFrameRate_HO'],     ['WFS','rate']             ),
        (['sensor_science','PixelScale'],  ['Detector', 'psInMas']    ),
        (['sensor_science','SigmaRON'],    ['Detector', 'ron']        ),
        (['sources_science','Wavelength'], ['spectra']                )
    ]
    
    def processor_func(config, modifier):
        config['sources_science']['Zenith'] = 90.0 - modifier['telescope']['altitude']
        config['telescope']['ZenithAngle']  = 90.0 - modifier['telescope']['altitude']
        
        def frame_delay(loop_freq):
            if not isinstance(loop_freq, torch.Tensor):
                loop_freq_ = torch.tensor(loop_freq)
            return torch.clamp(loop_freq_/1e3 * 2.3, min=1.0)
        
        config['RTC']['LoopDelaySteps_HO'] = frame_delay(config['RTC']['SensorFrameRate_HO'])

        return config
    
    return conversion_table, processor_func


def GetSPHEREsynth():
    conversion_table = [
        (['atmosphere','Cn2Weights'],      ['Cn2','profile']     ),
        (['atmosphere','Cn2Heights'],      ['Cn2','heights']      ),
        (['atmosphere','Seeing'],          ['seeing']             ),
        (['atmosphere','WindSpeed'],       ['Wind speed']         ),
        (['atmosphere','WindDirection'],   ['Wind direction']     ),
        (['telescope','Zenith'],           ['telescope','zenith'] ),
        (['telescope','ZenithAngle'],      ['telescope','zenith'] ),
        (['sensor_science','SigmaRON'],    ['Detector','ron']     ),
        (['sensor_science','Gain'],        ['Detector','gain']    ),
        (['sensor_HO','NumberPhotons'],    ['WFS','Nph vis']      ),
        (['sources_HO','Wavelength'],      ['WFS','wavelength']   ),
        (['RTC','SensorFrameRate_HO'],     ['RTC','loop rate']    ),
        (['RTC','LoopGain_HO'],            ['RTC','loop gain']    ),
        (['RTC','LoopDelaySteps_HO'],      ['RTC','frames delay'] ),
        (['sensor_science','PixelScale'],  ['Detector', 'psInMas']),
        (['sensor_science','SigmaRON'],    ['Detector','ron']     ),
        (['sources_science','Wavelength'], ['spectra']            ),
    ]
    return conversion_table, None